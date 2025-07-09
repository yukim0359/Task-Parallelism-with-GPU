# Task-Parallelism-with-GPU

GPU-based task parallelism implementations for Fibonacci calculation using different parallelization strategies.

## Overview

This project demonstrates three different approaches to parallelize recursive Fibonacci computation with GPU acceleration:

1. **GPU Fork-Join** (`fib_forkjoin_gpu.cu`): Complete GPU-based task parallelism implementation
2. **OpenMP + GPU** (`fib_omp_cuda.cu`): Task parallelism using OpenMP with GPU operations
3. **Serial + GPU** (`fib_cuda.cu`): Serial recursive execution with GPU heavy operations
