#include <cuda_runtime.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>

#include "sort.h"


#define CUDA_CHECK(err)                                                      \
    {                                                                        \
        cudaError_t err_ = (err);                                            \
        if (err_ != cudaSuccess)                                             \
        {                                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err_));                               \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    }


void int_scan_step(unsigned int *d_in, unsigned int size);


const int N = 1 << 28;
const int THREADS_PER_BLOCK = 256;
const int NUM_BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

const int RADIX_BITS = 8;
const int RADIX_SIZE = 1 << RADIX_BITS;


unsigned int next_power_of_two(unsigned int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}


__global__ void bitonic_sort_step_kernel(int *g_data, int stride, int k)
{
    unsigned int gtid = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = gtid ^ stride;

    if (ixj > gtid)
    {
        if ((gtid & k) == 0)
        {
            // зростаючий
            if (g_data[gtid] > g_data[ixj])
            {
                int temp = g_data[gtid];
                g_data[gtid] = g_data[ixj];
                g_data[ixj] = temp;
            }
        }
        else
        {
            // спадаючий
            if (g_data[gtid] < g_data[ixj])
            {
                int temp = g_data[gtid];
                g_data[gtid] = g_data[ixj];
                g_data[ixj] = temp;
            }
        }
    }
}


__global__ void pad_kernel(int *data, int original_size, int padded_size)
{
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gtid >= original_size && gtid < padded_size)
    {
        data[gtid] = INT_MAX;
    }
}


void bitonic_sort(int *h_data, unsigned int size)
{
    int *d_data;
    unsigned int size_of_padded_array = next_power_of_two(size);

    unsigned array_size_bytes = size_of_padded_array * sizeof(int);

    unsigned int threads_per_block = 512;
    unsigned int num_blocks = (size_of_padded_array + threads_per_block - 1) / threads_per_block;

    cudaMalloc(&d_data, array_size_bytes);
    cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);
    pad_kernel<<<num_blocks, threads_per_block>>>(d_data, size, size_of_padded_array);

    for (int k = 2; k <= size_of_padded_array; k <<= 1)
    {
        for (int stride = k >> 1; stride > 0; stride >>= 1)
        {
            bitonic_sort_step_kernel<<<num_blocks, threads_per_block>>>(d_data, stride, k);
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}


__global__ void radix_histogram_kernel(const unsigned int *d_in, unsigned int *d_histograms,
                                       int pass)
{
    __shared__ unsigned int s_hist[RADIX_SIZE];

    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = gridDim.x * blockDim.x;

    if (tid < RADIX_SIZE)
    {
        s_hist[tid] = 0;
    }
    __syncthreads();

    for (int i = gtid; i < N; i += grid_size)
    {
        unsigned int value = d_in[i];
        unsigned int digit = (value >> (pass * RADIX_BITS)) & (RADIX_SIZE - 1);
        atomicAdd_block(&s_hist[digit], 1);
    }
    __syncthreads();

    if (tid < RADIX_SIZE)
    {
        d_histograms[blockIdx.x * RADIX_SIZE + tid] = s_hist[tid];
    }
}


__global__ void radix_reorder_kernel(const unsigned int *d_in, unsigned int *d_out,
                                     const unsigned int *d_scanned_offsets,
                                     const unsigned int *d_block_histograms, int pass)
{
    __shared__ unsigned int s_offsets[RADIX_SIZE];
    __shared__ unsigned int s_block_offsets[RADIX_SIZE];
    __shared__ int digit_max[RADIX_SIZE];
    __shared__ int digit_min[RADIX_SIZE];

    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = gridDim.x * blockDim.x;

    if (tid < RADIX_SIZE)
    {
        s_offsets[tid] = d_scanned_offsets[tid * NUM_BLOCKS + blockIdx.x];
        s_block_offsets[tid] = d_block_histograms[blockIdx.x * RADIX_SIZE + tid];
        digit_max[tid] = -1;
        digit_min[tid] = 1024;
    }
    __syncthreads();

    for (int i = gtid; i < N; i += grid_size)
    {
        unsigned int value = d_in[i];
        unsigned int digit = (value >> (pass * RADIX_BITS)) & (RADIX_SIZE - 1);
        int local_pos;
        while (true)
        {
            atomicMax_block(&digit_max[digit], tid);
            __syncthreads();
            if (digit_max[digit] == tid)
            {
                local_pos = --s_block_offsets[digit];
                digit_max[digit] = -1;
                break;
            }
            __syncthreads();
        }
        int global_pos = s_offsets[digit] + local_pos;

        d_out[global_pos] = value;
    }
}


#define TILE_DIM 16


__global__ void tiled_transpose(unsigned int *d_out, const unsigned int *d_in, int M, int N)
{
    __shared__ unsigned int tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (x < M && y < N)
    {
        tile[threadIdx.x][threadIdx.y] = d_in[x * N + y];
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (x < N && y < M)
    {
        d_out[x * M + y] = tile[threadIdx.y][threadIdx.x];
    }
}


void radix_sort(unsigned int *h_in, unsigned int *h_out)
{
    unsigned int *d_histograms, *d_transposed_hist, *d_in, *d_out;

    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(unsigned int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_histograms, NUM_BLOCKS * RADIX_SIZE * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_transposed_hist, NUM_BLOCKS * RADIX_SIZE * sizeof(unsigned int)));

    const unsigned int hist_size = NUM_BLOCKS * RADIX_SIZE;

    int num_passes = sizeof(unsigned int) * 8 / RADIX_BITS;
    unsigned int *d_current_in = d_in;
    unsigned int *d_current_out = d_out;

    for (int pass = 0; pass < num_passes; ++pass)
    {
        radix_histogram_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_current_in, d_histograms, pass);
        CUDA_CHECK(cudaDeviceSynchronize());

        dim3 blockDim(TILE_DIM, TILE_DIM);
        dim3 gridDim((NUM_BLOCKS + blockDim.x - 1) / blockDim.x,
                     (RADIX_SIZE + blockDim.y - 1) / blockDim.y);

        tiled_transpose<<<gridDim, blockDim>>>(d_transposed_hist, d_histograms, NUM_BLOCKS,
                                               RADIX_SIZE);

        int_scan_step(d_transposed_hist, hist_size);

        radix_reorder_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
            d_current_in, d_current_out, d_transposed_hist, d_histograms, pass);
        CUDA_CHECK(cudaDeviceSynchronize());

        unsigned int *temp = d_current_in;
        d_current_in = d_current_out;
        d_current_out = temp;
    }

    if (num_passes % 2 != 0)
    {
        CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }
    else
    {
        CUDA_CHECK(cudaMemcpy(h_out, d_in, N * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }


    CUDA_CHECK(cudaFree(d_histograms));
    CUDA_CHECK(cudaFree(d_transposed_hist));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}


extern "C" int cuda_sort_test()
{
    unsigned int size = (1 << 28) - 11;
    int *h_data1, *h_data2;

    h_data1 = (int *)malloc(size * sizeof(int));
    h_data2 = (int *)malloc(size * sizeof(int));


    printf("Generating %d random integers...\n", N);
    srand(time(NULL));
    for (unsigned int i = 0; i < size; i++)
    {
        h_data1[i] = (int)rand();
    }


    memcpy(h_data2, h_data1, size * sizeof(int));

    struct timespec start1, end1;
    double time_spent1;

    clock_gettime(CLOCK_MONOTONIC, &start1);

    bitonic_sort(h_data1, size);

    clock_gettime(CLOCK_MONOTONIC, &end1);

    time_spent1 = (end1.tv_sec - start1.tv_sec) + (end1.tv_nsec - start1.tv_nsec) / 1e9;

    printf("CUDA bitonic_sort() run time: %f milliseconds\n", time_spent1 * 1000);

    for (unsigned int i = 0; i < 16; i++)
    {
        printf("%d ", h_data1[i]);
    }

    printf("...... ");

    for (unsigned int i = size - 16; i < size; i++)
    {
        printf("%d ", h_data1[i]);
    }

    printf("\n\n");

    struct timespec start2, end2;
    double time_spent2;

    clock_gettime(CLOCK_MONOTONIC, &start2);

    std::sort(h_data2, h_data2 + size);

    clock_gettime(CLOCK_MONOTONIC, &end2);

    time_spent2 = (end2.tv_sec - start2.tv_sec) + (end2.tv_nsec - start2.tv_nsec) / 1e9;

    for (unsigned int i = 0; i < 16; i++)
    {
        printf("%d ", h_data2[i]);
    }

    printf("...... ");

    for (unsigned int i = size - 16; i < size; i++)
    {
        printf("%d ", h_data2[i]);
    }

    printf("\n");
    printf("CPU subprogram run time: %f milliseconds\n", time_spent2 * 1000);
    printf("\n\n");

    free(h_data1);
    free(h_data2);


    unsigned int *h_in = (unsigned int *)malloc(N * sizeof(unsigned int));
    unsigned int *h_data_cpu = (unsigned int *)malloc(N * sizeof(unsigned int));
    unsigned int *h_out = (unsigned int *)malloc(N * sizeof(unsigned int));

    srand((unsigned int)time(NULL));

    printf("Generating %d random unsigned integers...\n", N);
    for (int i = 0; i < N; ++i)
    {
        h_in[i] = (unsigned int)rand();
    }

    memcpy(h_data_cpu, h_in, N * sizeof(unsigned int));


    struct timespec start3, end3;
    double time_spent3;

    clock_gettime(CLOCK_MONOTONIC, &start3);

    radix_sort(h_in, h_out);

    clock_gettime(CLOCK_MONOTONIC, &end3);

    time_spent3 = (end3.tv_sec - start3.tv_sec) + (end3.tv_nsec - start3.tv_nsec) / 1e9;

    printf("CUDA radix_sort() run time: %f milliseconds\n", time_spent3 * 1000);


    for (unsigned int i = 0; i < 16; i++)
    {
        printf("%u ", h_out[i]);
    }

    printf("...... ");

    for (unsigned int i = N - 16; i < N; i++)
    {
        printf("%u ", h_out[i]);
    }

    printf("\n\n");

    std::sort(h_data_cpu, h_data_cpu + N);


    for (unsigned int i = 0; i < 16; i++)
    {
        printf("%u ", h_data_cpu[i]);
    }

    printf("...... ");

    for (unsigned int i = N - 16; i < N; i++)
    {
        printf("%u ", h_data_cpu[i]);
    }


    free(h_in);
    free(h_data_cpu);
    free(h_out);


    return 0;
}
