# Azure AI Benchmarking Guide

## BEFORE CLONING:

The tests need to run in this docker container. Start it with the following command

```
sudo docker run  --rm -it --ipc=host --network=host --privileged --security-opt seccomp=unconfined --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE --device=/dev/kfd --device=/dev/dri --device=/dev/mem --gpus all nvcr.io/nvidia/pytorch:25.01-py3
```

# HOW TO RUN THE BENCHMARKS

All the requirements for the benchmarks can be installed with a simple command: `pip3 install -r requirements.txt`.


Usage: `python3 NVIDIA_runner.py [arg]`\
   or: `python3 NVIDIA_runner.py [arg1] [arg2]` ... to run more than one test e.g `python3 NVIDIA_runner.py hbm nccl`\
Arguments are as follows, and are case insensitive:\
All tests:   `all`\
CuBLASLt GEMM:   `gemm`\
NCCL Bandwidth:  `nccl`\
HBMBandwidth:    `hbm`\
NV Bandwidth:   `nv`\
CPU STREAM: `cpustream`\
Multichase: `multichase`\
Flash Attention: `fa`\



### Extras

- Test results will be stored in the `Outputs` directory. 

You can find example of results for the ND A100 v4, ND H100 v5 and ND H200 v5 virtual machines stored under [`Azure_Results`](https://github.com/Azure/AI-benchmarking-guide/tree/main/Azure_Results).
