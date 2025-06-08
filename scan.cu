#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "scan.h"


#define NUM_BANKS 32


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


__global__ void block_scan_sum_kernel1(float *g_data, float *sums, unsigned int size)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;

    unsigned int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    int bufout = 0, bufin = 1;
    sdata[bufout * block_size + tid] = (gtid < size) ? g_data[gtid] : 0.0f;
    __syncthreads();

    for (unsigned int offset = 1; offset < block_size; offset <<= 1)
    {
        bufout = 1 - bufout;
        bufin = 1 - bufout;
        if (tid >= offset)
            sdata[bufout * block_size + tid] =
                sdata[bufin * block_size + tid] + sdata[bufin * block_size + tid - offset];
        else
            sdata[bufout * block_size + tid] = sdata[bufin * block_size + tid];
        __syncthreads();
    }

    if (gtid < size) g_data[gtid] = sdata[bufout * block_size + tid];
    if (tid == block_size - 1)
    {
        sums[blockIdx.x] = sdata[bufout * block_size + tid];
    }
}


__global__ void block_scan_sum_kernel2(float *g_data, float *sums, unsigned int size)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;
    unsigned int buffer_size = block_size << 1;

    unsigned int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[2 * tid] = (2 * gtid < size) ? g_data[2 * gtid] : 0.0f;
    sdata[2 * tid + 1] = (2 * gtid + 1 < size) ? g_data[2 * gtid + 1] : 0.0f;

    unsigned int offset = 1;
    for (unsigned int d = block_size; d > 0; d >>= 1)
    {
        __syncthreads();
        if (tid < d)
        {
            unsigned int ai = offset * (2 * tid + 1) - 1;
            unsigned int bi = offset * (2 * tid + 2) - 1;
            sdata[bi] += sdata[ai];
        }
        offset <<= 1;
    }

    if (tid == 0)
    {
        sdata[buffer_size - 1] = 0;
    }

    for (unsigned int d = 1; d < buffer_size; d <<= 1)
    {
        offset >>= 1;
        __syncthreads();
        if (tid < d)
        {
            unsigned int ai = offset * (2 * tid + 1) - 1;
            unsigned int bi = offset * (2 * tid + 2) - 1;
            float t = sdata[ai];
            sdata[ai] = sdata[bi];
            sdata[bi] += t;
        }
    }
    __syncthreads();

    if (tid == block_size - 1)
    {
        if (2 * gtid + 1 < size)
            sums[blockIdx.x] = sdata[2 * tid + 1] + g_data[2 * gtid + 1];
        else
            sums[blockIdx.x] = sdata[2 * tid + 1];
    }

    if (2 * gtid < size) g_data[2 * gtid] = sdata[2 * tid];
    if (2 * gtid + 1 < size) g_data[2 * gtid + 1] = sdata[2 * tid + 1];
}


__inline__ __device__ unsigned int conflict_avoid_offset(unsigned int idx)
{
    return idx / NUM_BANKS;
}


__global__ void block_scan_sum_kernel3(float *g_data, float *sums, unsigned int size)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;
    unsigned int buffer_size = block_size << 1;

    unsigned int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[2 * tid + conflict_avoid_offset(2 * tid)] = (2 * gtid < size) ? g_data[2 * gtid] : 0.0f;
    sdata[2 * tid + 1 + conflict_avoid_offset(2 * tid + 1)] =
        (2 * gtid + 1 < size) ? g_data[2 * gtid + 1] : 0.0f;

    unsigned int offset = 1;
    for (unsigned int d = block_size; d > 0; d >>= 1)
    {
        __syncthreads();
        if (tid < d)
        {
            unsigned int ai = offset * (2 * tid + 1) - 1;
            unsigned int bi = offset * (2 * tid + 2) - 1;
            ai += conflict_avoid_offset(ai);
            bi += conflict_avoid_offset(bi);
            sdata[bi] += sdata[ai];
        }
        offset <<= 1;
    }

    if (tid == 0)
    {
        sdata[buffer_size - 1 + conflict_avoid_offset(buffer_size - 1)] = 0;
    }

    for (unsigned int d = 1; d < buffer_size; d <<= 1)
    {
        offset >>= 1;
        __syncthreads();
        if (tid < d)
        {
            unsigned int ai = offset * (2 * tid + 1) - 1;
            unsigned int bi = offset * (2 * tid + 2) - 1;
            ai += conflict_avoid_offset(ai);
            bi += conflict_avoid_offset(bi);
            float t = sdata[ai];
            sdata[ai] = sdata[bi];
            sdata[bi] += t;
        }
    }
    __syncthreads();

    if (tid == block_size - 1)
    {
        if (2 * gtid + 1 < size)
            sums[blockIdx.x] =
                sdata[2 * tid + 1 + conflict_avoid_offset(2 * tid + 1)] + g_data[2 * gtid + 1];
        else
            sums[blockIdx.x] = sdata[2 * tid + 1 + conflict_avoid_offset(2 * tid + 1)];
    }

    if (2 * gtid < size) g_data[2 * gtid] = sdata[2 * tid + conflict_avoid_offset(2 * tid)];
    if (2 * gtid + 1 < size)
        g_data[2 * gtid + 1] = sdata[2 * tid + 1 + conflict_avoid_offset(2 * tid + 1)];
}


__global__ void add_offsets_kernel1(float *g_data, float *sums, unsigned int size)
{
    unsigned int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gtid < size && blockIdx.x > 0) g_data[gtid] += sums[blockIdx.x - 1];
}


__global__ void add_offsets_kernel2(float *g_data, float *sums, unsigned int size)
{
    unsigned int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    if (2 * gtid < size) g_data[2 * gtid] += sums[blockIdx.x];
    if (2 * gtid + 1 < size) g_data[2 * gtid + 1] += sums[blockIdx.x];
}


void cuda_scan1(float *h_in, float *h_out, unsigned int size)
{
    unsigned int array_size_bytes = size * sizeof(float);
    float *d_in, *sums;

    CUDA_CHECK(cudaMalloc(&d_in, array_size_bytes));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, array_size_bytes, cudaMemcpyHostToDevice));

    unsigned int threads_per_block = 512;
    unsigned int num = size;

    unsigned int num_blocks = (num + threads_per_block - 1) / threads_per_block;

    size_t shared_mem_size = 2 * threads_per_block * sizeof(float);  // подвійний буфер

    float **buffers_for_processing = (float **)malloc(sizeof(float *) * 100);
    unsigned int *buffer_lengths_for_processing =
        (unsigned int *)malloc(sizeof(unsigned int) * 100);

    int depth = 0;

#ifdef BENCHMARK
    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    CUDA_CHECK(cudaEventRecord(start_event, 0));
#endif

    while (num > 1)
    {
        CUDA_CHECK(cudaMalloc(&sums, num_blocks * sizeof(float)));

        block_scan_sum_kernel1<<<num_blocks, threads_per_block, shared_mem_size>>>(d_in, sums, num);

        CUDA_CHECK(cudaDeviceSynchronize());

        buffers_for_processing[depth] = d_in;
        buffer_lengths_for_processing[depth] = num;

        num = num_blocks;
        num_blocks = (num + threads_per_block - 1) / threads_per_block;

        d_in = sums;

        depth++;
    }

    depth--;

    for (; depth >= 0; depth--)
    {
        d_in = buffers_for_processing[depth];
        num = buffer_lengths_for_processing[depth];

        add_offsets_kernel1<<<num_blocks, threads_per_block>>>(d_in, sums, num);

        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(sums));

        sums = d_in;
        num_blocks = num;
    }

#ifdef BENCHMARK
    CUDA_CHECK(cudaEventRecord(stop_event, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));

    printf("kernel1 run time: %f milliseconds\n", milliseconds);
#endif

    CUDA_CHECK(cudaMemcpy(h_out, d_in, array_size_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));

    free(buffer_lengths_for_processing);
    free(buffers_for_processing);
}


void cuda_scan2(float *h_in, float *h_out, unsigned int size)
{
    unsigned int array_size_bytes = size * sizeof(float);
    float *d_in, *sums;

    CUDA_CHECK(cudaMalloc(&d_in, array_size_bytes));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, array_size_bytes, cudaMemcpyHostToDevice));

    unsigned int threads_per_block = 512;
    unsigned int elements_per_block = threads_per_block * 2;
    unsigned int num = size;

    unsigned int num_blocks = (num + elements_per_block - 1) / elements_per_block;

    size_t shared_mem_size = (elements_per_block + elements_per_block / 32 + 1) * sizeof(float);

    float **buffers_for_processing = (float **)malloc(sizeof(float *) * 100);
    unsigned int *buffer_lengths_for_processing =
        (unsigned int *)malloc(sizeof(unsigned int) * 100);

    int depth = 0;

#ifdef BENCHMARK
    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    CUDA_CHECK(cudaEventRecord(start_event, 0));
#endif

    while (num > 1)
    {
        CUDA_CHECK(cudaMalloc(&sums, num_blocks * sizeof(float)));

        block_scan_sum_kernel2<<<num_blocks, threads_per_block, shared_mem_size>>>(d_in, sums, num);

        CUDA_CHECK(cudaDeviceSynchronize());

        buffers_for_processing[depth] = d_in;
        buffer_lengths_for_processing[depth] = num;

        num = num_blocks;
        num_blocks = (num + elements_per_block - 1) / elements_per_block;

        d_in = sums;

        depth++;
    }

    depth--;

    int toggle = 0;
    for (; depth >= 0; depth--)
    {
        d_in = buffers_for_processing[depth];
        num = buffer_lengths_for_processing[depth];
        if (toggle)
        {
            add_offsets_kernel2<<<num_blocks, threads_per_block>>>(d_in, sums, num);
        }
        else
            toggle = 1;
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(sums));

        sums = d_in;
        num_blocks = num;
    }

#ifdef BENCHMARK
    CUDA_CHECK(cudaEventRecord(stop_event, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));

    printf("kernel2 run time: %f milliseconds\n", milliseconds);
#endif

    CUDA_CHECK(cudaMemcpy(h_out, d_in, array_size_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));

    free(buffer_lengths_for_processing);
    free(buffers_for_processing);
}


void cuda_scan3(float *h_in, float *h_out, unsigned int size)
{
    unsigned int array_size_bytes = size * sizeof(float);
    float *d_in, *sums;

    CUDA_CHECK(cudaMalloc(&d_in, array_size_bytes));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, array_size_bytes, cudaMemcpyHostToDevice));

    unsigned int threads_per_block = 512;
    unsigned int elements_per_block = threads_per_block * 2;
    unsigned int num = size;

    unsigned int num_blocks = (num + elements_per_block - 1) / elements_per_block;

    size_t shared_mem_size = (elements_per_block + elements_per_block / 32 + 1) * sizeof(float);

    float **buffers_for_processing = (float **)malloc(sizeof(float *) * 100);
    unsigned int *buffer_lengths_for_processing =
        (unsigned int *)malloc(sizeof(unsigned int) * 100);

    int depth = 0;

#ifdef BENCHMARK
    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    CUDA_CHECK(cudaEventRecord(start_event, 0));
#endif

    while (num > 1)
    {
        CUDA_CHECK(cudaMalloc(&sums, num_blocks * sizeof(float)));

        block_scan_sum_kernel3<<<num_blocks, threads_per_block, shared_mem_size>>>(d_in, sums, num);

        CUDA_CHECK(cudaDeviceSynchronize());

        buffers_for_processing[depth] = d_in;
        buffer_lengths_for_processing[depth] = num;

        num = num_blocks;
        num_blocks = (num + elements_per_block - 1) / elements_per_block;

        d_in = sums;

        depth++;
    }

    depth--;

    int toggle = 0;
    for (; depth >= 0; depth--)
    {
        d_in = buffers_for_processing[depth];
        num = buffer_lengths_for_processing[depth];
        if (toggle)
        {
            add_offsets_kernel2<<<num_blocks, threads_per_block>>>(d_in, sums, num);
        }
        else
            toggle = 1;
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(sums));

        sums = d_in;
        num_blocks = num;
    }

#ifdef BENCHMARK
    CUDA_CHECK(cudaEventRecord(stop_event, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));

    printf("kernel3 run time: %f milliseconds\n", milliseconds);
#endif

    CUDA_CHECK(cudaMemcpy(h_out, d_in, array_size_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));

    free(buffer_lengths_for_processing);
    free(buffers_for_processing);
}


extern "C" int cuda_scan_test()
{
    unsigned int size = (1 << 29) - 37;
    unsigned int array_size_bytes = size * sizeof(float);

    float *h_in, *h_out;
    double *h_out_cpu;

    h_in = (float *)malloc(array_size_bytes);
    h_out = (float *)malloc(array_size_bytes);
    h_out_cpu = (double *)malloc(size * sizeof(double));

    for (unsigned int i = 0; i < size; i++)
    {
        h_in[i] = (float)(i % 100) + 0.1f;
    }

    struct timespec start1, end1;
    double time_spent1;

    clock_gettime(CLOCK_MONOTONIC, &start1);

    cuda_scan3(h_in, h_out, size);

    clock_gettime(CLOCK_MONOTONIC, &end1);

    time_spent1 = (end1.tv_sec - start1.tv_sec) + (end1.tv_nsec - start1.tv_nsec) / 1e9;

    printf("CUDA scan run time: %f milliseconds\n", time_spent1 * 1000);

    for (unsigned int i = size - 10; i < size; i++)
    {
        printf("%f ", h_out[i]);
    }

    printf("\n\n");

    struct timespec start2, end2;
    double time_spent2;

    clock_gettime(CLOCK_MONOTONIC, &start2);

    h_out_cpu[0] = h_in[0];

    for (unsigned int i = 1; i < size; i++)
    {
        h_out_cpu[i] = h_out_cpu[i - 1] + h_in[i];
        if (i >= size - 10) printf("%lf ", h_out_cpu[i]);
    }
    printf("\n");

    clock_gettime(CLOCK_MONOTONIC, &end2);

    time_spent2 = (end2.tv_sec - start2.tv_sec) + (end2.tv_nsec - start2.tv_nsec) / 1e9;

    printf("CPU subprogram run time: %f milliseconds\n", time_spent2 * 1000);

    free(h_in);
    free(h_out);
    free(h_out_cpu);

    return 0;
}
