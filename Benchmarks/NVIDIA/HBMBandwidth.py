import json
import os
import statistics
import subprocess
import time
import csv
from Infra import tools
from prettytable import PrettyTable

class HBMBandwidth:
    def __init__(self, path: str, machine: str):
        self.name = "HBMBandwidth"
        self.machine_name = machine
        config = self.get_config(path)
        self.num_runs, self.interval = self.config_conversion(config)

        self.buffer = []

    def get_config(self, path: str):
        file = open(path)
        data = json.load(file)
        file.close()
        try:
            return data[self.name]
        except KeyError:
            raise KeyError("no value found")

    def parse_json(self, config):
        return config["inputs"]["num_runs"], config["inputs"]["interval"]

    def config_conversion(self, config) -> tuple[list, list, list]:
        return self.parse_json(config)

    def build(self):
        current = os.getcwd()

        path = "BabelStream"
        isdir = os.path.isdir(path)
        if not isdir:
            results = subprocess.run(
                ["git", "clone", "https://github.com/gitaumark/BabelStream",  path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            tools.write_log(tools.check_error(results))

        build_path = os.path.join(current, "BabelStream")
        os.chdir(build_path)

        babelstream_build_path = os.path.join(build_path, "build")

        arch ="sm_90"
        if "A100" in self.machine_name:
            arch = "sm_80"
            
        if not os.path.isdir(babelstream_build_path):
            os.mkdir(babelstream_build_path)
            os.chdir(babelstream_build_path)
            results = subprocess.run(
                [
                    "cmake",
                    "../",
                    "-DMODEL=cuda",
                    "-DCUDA_ARCH=" + arch,
                    "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            tools.write_log(tools.check_error(results))
           
            results = subprocess.run(
                ["make"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            tools.write_log(tools.check_error(results))
        else:
            os.chdir(babelstream_build_path)


    def run(self):
        current = os.getcwd()
        print("Running HBM Bandwidth...")

        runs_executed = 0
        buffer = []
        while runs_executed < self.num_runs:
            results = subprocess.run(
                ["./cuda-stream"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            tools.write_log(tools.check_error(results))
            log = results.stdout.decode("utf-8").strip().split("\n")[14:19]
            for i in range(len(log)):
                temp = log[i].split()
                log[i] = [temp[0], temp[1]]
            buffer.append(log)
            runs_executed += 1
            time.sleep(int(self.interval))

    
        self.buffer = buffer
        os.chdir(current)
        self.save_results()

    def process_stats(self, results):
        mean = statistics.mean(results)/1000000
        maximum = max(results)/1000000
        minimum = min(results)/1000000
        return [round(minimum, 2), round(maximum, 2), round(mean, 2)]
    

    def save_results(self):
        copy = ["Copy"]
        mul = ["Mul"]
        add = ["Add"]
        triad = ["Triad"]
        dot = ["Dot"]
        for log in self.buffer:
            copy.append(float(log[0][1]))
            mul.append(float(log[1][1]))
            add.append(float(log[2][1]))
            triad.append(float(log[3][1]))
            dot.append(float(log[4][1]))

        copy[1:] = self.process_stats(copy[1:])
        mul[1:] = self.process_stats(mul[1:])
        add[1:] = self.process_stats(add[1:])
        triad[1:] = self.process_stats(triad[1:])
        dot[1:] = self.process_stats(dot[1:])
        
        table1 = PrettyTable()
        table1.field_names = ["Operation","Min (TB/s)", "Max (TB/s)", "Mean (TB/s)"]
        table1.add_row(copy)
        table1.add_row(mul)
        table1.add_row(add)
        table1.add_row(triad)
        table1.add_row(dot)
        print(table1)

        with open('../../Outputs/HBMBandwidth_Performance_results_' + self.machine_name +'.csv', 'w') as csvFile:
            writer = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["Operation","Min (TB/s)", "Max (TB/s)", "Mean (TB/s)"])
            writer.writerow(copy)
            writer.writerow(mul)
            writer.writerow(add)
            writer.writerow(triad)
            writer.writerow(dot)
