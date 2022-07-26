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

#define DEVICE_N 25600000 // device memory

#define NUM_STREAM 2

#define BUF_SIZE (DEVICE_N / NUM_STREAM) // multi buffer

#define REP (MAX_N / BUF_SIZE / NUM_STREAM)

__global__ void Kernel(VALUE_TYPE *A)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MAX_N)
    {
        for (size_t i = 0; i < 2000; i++)
        {
            A[idx] = A[idx] * idx;
        }
        // A[idx] = 1;
    }
}

int main()
{
    struct timeval tv_start, tv_end;

    cudaStream_t stream[NUM_STREAM];
    VALUE_TYPE *A;
    VALUE_TYPE *d_A[NUM_STREAM];

    size_t n = MAX_N;
    size_t buf_size = BUF_SIZE;

    dim3 gridDim;
    dim3 blockDim;
    blockDim.x = 128;
    gridDim.x = (buf_size + blockDim.x - 1) / blockDim.x;
    printf("CUDA kernel launch with %d blocks of %d threads\n", gridDim.x, blockDim.x);

    for (size_t i = 0; i < NUM_STREAM; i++)
    {
        cudaStreamCreate(&stream[i]);
        cudaMalloc((void **)&(d_A[i]), buf_size * sizeof(VALUE_TYPE));
    }

    cudaHostAlloc(&A, n * sizeof(VALUE_TYPE), cudaHostAllocDefault);
    cudaMemset(A, 0, MAX_N * sizeof(VALUE_TYPE));

    gettimeofday(&tv_start, NULL);

    for (size_t bench = 0; bench < BENCH_REPEAT; bench++)
    {
        for (size_t batch = 0; batch < REP; batch++)
        {
            size_t A_offset = batch * buf_size * NUM_STREAM;
            for (size_t i = 0; i < NUM_STREAM; i++)
            {
                cudaMemcpyAsync(d_A[i], A + A_offset, buf_size * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice, stream[i]);
                Kernel<<<gridDim, blockDim>>>(d_A[i]);
                cudaMemcpyAsync(A + A_offset, d_A[i], buf_size * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost, stream[i]);
                A_offset += buf_size;
            }
        }
    }

    cudaDeviceSynchronize();
    gettimeofday(&tv_end, NULL);
    printf("MemOp & Kernel Time: %fms\n", ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0 + (tv_end.tv_usec - tv_start.tv_usec) / 1000.0) / BENCH_REPEAT);

    for (size_t i = 0; i < NUM_STREAM; i++)
    {
        cudaStreamDestroy(stream[i]);
        cudaFree(d_A[i]);
    }

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