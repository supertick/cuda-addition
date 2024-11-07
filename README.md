# cuda-addition
Minimal example of a Java-CPP CUDA kernel

Simple example of a CUDA kernel that adds two arrays together.

## Requirements
* JDK 19
* Maven
* NVIDIA GPU
* CUDA 12.3
* CUDA Toolkit 12.3

## To Compile CUDA Kernel

```bash
/usr/local/cuda/bin/nvcc -ptx add_kernel.cu -o add_kernel.ptx
```

## To Run
```bash
mvn clean compile exec:java -Dexec.mainClass="CudaArrayAddition"
```

## Tools
```bash
# Check GPU status
nvidia-smi

# Check GPU memory usage
nvidia-smi --query-gpu=memory.used --format=csv

# Graph GPU Utilization
nvtop

# Profile GPU
nvprof

```