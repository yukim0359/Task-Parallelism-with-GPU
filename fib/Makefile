NVCC=nvcc
CC=gcc
# CFLAGS=-O2
OMPFLAGS=-fopenmp

all: fib_omp_cuda fib_cuda fib_forkjoin_gpu

fib_omp_cuda: fib_omp_cuda.cu
	$(NVCC) -Xcompiler $(OMPFLAGS) -o $@ $<

fib_cuda: fib_cuda.cu
	$(NVCC) -o $@ $<

fib_forkjoin_gpu: fib_forkjoin_gpu.cu
	$(NVCC) -o $@ $<

clean:
	rm -f fib_omp_cuda fib_cuda fib_forkjoin_gpu
