import json
import os
import statistics
import subprocess
import time
import csv
from Infra import tools
from prettytable import PrettyTable

class Multichase:
    def __init__(self):
        self.name = "Multichase"
    

    def build(self):
        current = os.getcwd()

        path = "multichase"
        isdir = os.path.isdir(path)
        if not isdir:
            results = subprocess.run(
                ["git", "clone", "https://github.com/google/multichase",  path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            tools.write_log(tools.check_error(results))

            build_path = os.path.join(current, "multichase")
            os.chdir(build_path)
            
            results = subprocess.run("make", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            tools.write_log(tools.check_error(results))
            os.chdir(current)



    def run(self):
        current = os.getcwd()
        print("Running Multichase...", current)

        results = subprocess.run("cd Benchmarks/NVIDIA && chmod 777 run_multichase.sh && ./run_multichase.sh",shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        tools.write_log(tools.check_error(results))
        print(results.stdout.decode("utf-8"))
        print(results.stderr.decode("utf-8"))
           

    
