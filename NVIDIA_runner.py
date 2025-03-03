import os
import sys
import subprocess
from Benchmarks.NVIDIA import GEMMCublasLt as gemm
from Benchmarks.NVIDIA import HBMBandwidth as HBM
from Benchmarks.NVIDIA import NVBandwidth as NV
from Benchmarks.NVIDIA import NCCLBandwidth as NCCL
from Benchmarks.NVIDIA import FlashAttention as FA
from Benchmarks.NVIDIA import Multichase as Multichase
from Infra import tools
from Benchmarks.NVIDIA import CPUStream as CPU


machine_name = ""
current = os.getcwd()
tools.create_dir("Outputs")

def run_CublasLt():
    test = gemm.GEMMCublastLt("config.json",machine_name) 
    test.build()
    test.run_model_sizes()
    
def run_HBMBandwidth():
    results = subprocess.run("./Benchmarks/NVIDIA/stream_vectorized_double_test -n1073741824 -r1", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    print(results.stdout.decode('utf-8'))
    print(results.stderr.decode('utf-8'))
    
def run_NVBandwidth():
    test = NV.NVBandwidth("config.json", machine_name)
    test.build()
    test.run()

def run_NCCLBandwidth():
    test = NCCL.NCCLBandwidth("config.json", machine_name)
    test.build()
    test.run()

def run_FlashAttention():
    test = FA.FlashAttention("config.json", machine_name)
    test.run()
    
def run_Multichase():
    test = Multichase.Multichase()
    test.build()
    test.run()

def run_CPUStream():
    test = CPU.CPUStream()
    test.build()
    test.run()
    

machine_name = "GB200"
arguments = []
match = False
for arg in sys.argv:
    arguments.append(arg.lower())

if ("gemm" in arguments):
    match = True
    run_CublasLt()
    os.chdir(current)
    
if ("nccl" in arguments):
    match = True
    run_NCCLBandwidth()
    os.chdir(current)
    
if ("hbm" in arguments):
    match = True
    run_HBMBandwidth()
    os.chdir(current)
    
if ("nv" in arguments):
    match = True
    run_NVBandwidth()
    os.chdir(current)
    
if ("fa"  in arguments):
    match = True
    run_FlashAttention()
    os.chdir(current)
        
if ("multichase" in arguments):
    match = True
    run_Multichase()
    os.chdir(current)

if ("cpustream" in arguments):
    match = True
    run_CPUStream()
    os.chdir(current)

if ("all" in arguments):
    match = True
    run_CublasLt()
    os.chdir(current)
    run_NCCLBandwidth()
    run_Multichase()
    run_CPUStream()
    os.chdir(current)
    run_HBMBandwidth()
    os.chdir(current)
    run_NVBandwidth()
    os.chdir(current)
    run_FlashAttention()
    os.chdir(current)

if not match: 
    print("Usage: python3 NVIDIA_runner.py [arg]\n   or: python3 NVIDIA_runner.py [arg1] [arg2] ... to run more than one test e.g python3 NVIDIA_runner.py hbm nccl\nArguments are as follows, and are case insensitive:\nAll tests:  all\nCuBLASLt GEMM:  gemm\nNCCL Bandwidth: nccl\nHBMBandwidth:   hbm\nNV Bandwidth:   nv\nFlash Attention: fa\nCPU Stream: cpustream\nMultichase:  multichase")
    
