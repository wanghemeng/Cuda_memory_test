#include <stdio.h>
#include <cuda_runtime.h>

#include <sys/time.h>
#include <unistd.h>
// gettimeofday()

#define MAX_N 102400000

#define VALUE_TYPE int

#define WARP_SIZE 32

#define THREADS_PER_BLOCK 128

#define BENCH_REPEAT 100

__global__ void Kernel(VALUE_TYPE *A)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MAX_N)
    {
        for (size_t i = 0; i < 3500; i++)
        {
            A[idx] = A[idx] * idx;
        }
        // A[idx] = 1;
    }
}

int main()
{
    struct timeval tv_start, tv_end;

    VALUE_TYPE *A;
    VALUE_TYPE *d_A;

    size_t n = MAX_N;

    dim3 gridDim;
    dim3 blockDim;
    blockDim.x = 128;
    gridDim.x = (n + blockDim.x - 1) / blockDim.x;
    printf("CUDA kernel launch with %d blocks of %d threads\n", gridDim.x, blockDim.x);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (!prop.canMapHostMemory)
    {
        printf("Not support zero copy\n");
        return 0;
    }
    
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaHostAlloc(&A, n * sizeof(VALUE_TYPE), cudaHostAllocMapped);
    cudaMemset(A, 0, n * sizeof(VALUE_TYPE));
    cudaHostGetDevicePointer((void **)&d_A, A, 0);

    gettimeofday(&tv_start, NULL);
    for (size_t i = 0; i < BENCH_REPEAT; i++)
    {
        Kernel<<<gridDim, blockDim>>>(d_A);
    }
    cudaDeviceSynchronize();
    gettimeofday(&tv_end, NULL);
    printf("Kernel Time: %fms\n", ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0 + (tv_end.tv_usec - tv_start.tv_usec) / 1000.0) / BENCH_REPEAT);

    // for (size_t i = 0; i < MAX_N; i++)
    // {
    //     if (A[i] != 1)
    //     {
    //         printf("error\n");
    //         break;
    //     }
    // }

    cudaFreeHost(A);

    return 0;
}