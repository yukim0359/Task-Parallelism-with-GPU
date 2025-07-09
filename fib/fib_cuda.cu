// タスク並列はせずに，シリアルに実行する
// ただしheavy operationはGPUで実行する
#include <stdio.h>
#include <cuda_runtime.h>

#define DATA_LENGTH 1000

// CUDA kernel: heavy operation on GPU
__global__ void heavy_operation_kernel() {
    __shared__ int sdata[DATA_LENGTH];
    for (int i = threadIdx.x; i < DATA_LENGTH; i += blockDim.x) {
        sdata[i] = i;
    }
}

// Host function to launch the CUDA kernel
void heavy_operation_gpu() {
    heavy_operation_kernel<<<1, 128>>>();
}

// Serial recursive Fibonacci with GPU heavy operation at each step
int fib_cuda(int n) {
    int x, y;
    if (n < 2) {
        heavy_operation_gpu(); // Base case: GPU heavy op
        return n;
    } else {
        x = fib_cuda(n - 1);
        y = fib_cuda(n - 2);
        heavy_operation_gpu(); // Combine: GPU heavy op
        return x + y;
    }
}

int main() {
    int n = 18;
    int result;
    cudaSetDevice(0);
    float milliseconds;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    result = fib_cuda(n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Fibonacci(%d) = %d\n", n, result);
    printf("Kernel execution time: %.3f ms (%.6f s)\n", milliseconds, milliseconds/1000.0);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaDeviceReset();
    return 0;
} 
