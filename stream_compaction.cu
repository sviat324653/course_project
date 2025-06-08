#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "stream_compaction.h"


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


__inline__ __device__ int conflict_avoid_offset(int idx)
{
    return idx / NUM_BANKS;
}


__global__ void block_scan_sum_kernel_int(unsigned int *g_data, unsigned int *sums,
                                          unsigned int size)
{
    extern __shared__ unsigned int sdata[];

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
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
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
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            ai += conflict_avoid_offset(ai);
            bi += conflict_avoid_offset(bi);
            unsigned int t = sdata[ai];
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


__global__ void add_offsets_kernel_int(unsigned int *g_data, unsigned int *sums, unsigned int size)
{
    unsigned int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    if (2 * gtid < size) g_data[2 * gtid] += sums[blockIdx.x];
    if (2 * gtid + 1 < size) g_data[2 * gtid + 1] += sums[blockIdx.x];
}


__global__ void mark_elements_kernel(const float *g_in, unsigned int *g_flags, unsigned int size,
                                     int (*predicate)(float))
{
    unsigned int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gtid < size)
    {
        g_flags[gtid] = (predicate(g_in[gtid])) ? 1 : 0;
    }
}


__global__ void scatter_elements_kernel(const float *g_in, const unsigned int *g_flags,
                                        const unsigned int *g_destionation_indices, float *g_out,
                                        unsigned int size)
{
    unsigned int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gtid < size)
    {
        if (g_flags[gtid] == 1)
        {
            unsigned int destination_idx = g_destionation_indices[gtid];
            g_out[destination_idx] = g_in[gtid];
        }
    }
}


void int_scan_step(unsigned int *d_in, unsigned int size)
{
    unsigned int array_size_bytes = size * sizeof(unsigned int);
    unsigned int *sums;

    unsigned int threads_per_block = 512;
    unsigned int elements_per_block = threads_per_block * 2;
    unsigned int num = size;

    unsigned int num_blocks = (num + elements_per_block - 1) / elements_per_block;

    size_t shared_mem_size =
        (elements_per_block + elements_per_block / 32 + 1) * sizeof(unsigned int);

    unsigned int **buffers_for_processing = (unsigned int **)malloc(sizeof(unsigned int *) * 100);
    unsigned int *buffer_lengths_for_processing =
        (unsigned int *)malloc(sizeof(unsigned int) * 100);

    int depth = 0;

    while (num > 1)
    {
        CUDA_CHECK(cudaMalloc(&sums, num_blocks * sizeof(unsigned int)));

        block_scan_sum_kernel_int<<<num_blocks, threads_per_block, shared_mem_size>>>(d_in, sums,
                                                                                      num);

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
            add_offsets_kernel_int<<<num_blocks, threads_per_block>>>(d_in, sums, num);
        }
        else
            toggle = 1;
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(sums));

        sums = d_in;
        num_blocks = num;
    }

    free(buffer_lengths_for_processing);
    free(buffers_for_processing);
}


__device__ __host__ int predicate1(float value)
{
    return value > 0;
}


typedef int (*predicate_t)(float);

__device__ predicate_t d_predicate = predicate1;
predicate_t h_predicate;


void cuda_stream_compaction(float *h_in, float **h_out, unsigned int size, unsigned int *new_size)
{
    unsigned int array_size_bytes = size * sizeof(float);
    float *d_in, *d_out;
    unsigned int *d_flags, *d_destination_indeces;
    CUDA_CHECK(cudaMalloc(&d_in, array_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_flags, array_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_destination_indeces, array_size_bytes));


#ifdef BENCHMARK
    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    CUDA_CHECK(cudaEventRecord(start_event, 0));
#endif

    CUDA_CHECK(cudaMemcpy(d_in, h_in, array_size_bytes, cudaMemcpyHostToDevice));

    unsigned int threads_per_block = 512;
    unsigned int num_blocks = (size + threads_per_block - 1) / threads_per_block;

    CUDA_CHECK(cudaMemcpyFromSymbol(&h_predicate, d_predicate, sizeof(predicate_t)));

    mark_elements_kernel<<<num_blocks, threads_per_block>>>(d_in, d_flags, size, h_predicate);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(
        cudaMemcpy(d_destination_indeces, d_flags, array_size_bytes, cudaMemcpyDeviceToDevice));

    int_scan_step(d_destination_indeces, size);
    unsigned int last_flag;
    CUDA_CHECK(cudaMemcpy(new_size, d_destination_indeces + size - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_flag, d_flags + size - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    if (last_flag == 1)
    {
        (*new_size)++;
    }
    unsigned int output_array_size = *new_size;
    CUDA_CHECK(cudaMalloc(&d_out, output_array_size * sizeof(float)));

    scatter_elements_kernel<<<num_blocks, threads_per_block>>>(d_in, d_flags, d_destination_indeces,
                                                               d_out, size);
    CUDA_CHECK(cudaDeviceSynchronize());

#ifdef BENCHMARK
    CUDA_CHECK(cudaEventRecord(stop_event, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));

    printf("kernels run time: %f milliseconds\n", milliseconds);
#endif

    *h_out = (float *)malloc(output_array_size * sizeof(float));

    CUDA_CHECK(
        cudaMemcpy(*h_out, d_out, output_array_size * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_in);
    cudaFree(d_flags);
    cudaFree(d_destination_indeces);
    cudaFree(d_out);
}


void cpu_stream_compaction(float *h_in, float **h_out, unsigned int size, unsigned int *new_size)
{
    *new_size = 0;
    for (unsigned int i = 0; i < size; i++)
    {
        if (predicate1(h_in[i])) (*new_size)++;
    }

    *h_out = (float *)malloc(*new_size * sizeof(float));

    unsigned int j = 0;
    for (unsigned int i = 0; i < size; i++)
    {
        if (predicate1(h_in[i]))
        {
            (*h_out)[j] = h_in[i];
            j++;
        }
    }
}


extern "C" int cuda_stream_compaction_test()
{
    unsigned int size = (1 << 28) - 37;
    unsigned int new_size, new_size_cpu;

    float *h_in, *h_out, *h_out_cpu;
    h_in = (float *)malloc(size * sizeof(float));

    int sign = -1;

    srand(time(NULL));
    for (unsigned int i = 0; i < size; i++)
    {
        h_in[i] = sign * (float)(rand() % 100) + 0.1f;
        sign = (rand() % 2) ? 1 : -1;
    }

    struct timespec start1, end1;
    double time_spent1;

    clock_gettime(CLOCK_MONOTONIC, &start1);

    cuda_stream_compaction(h_in, &h_out, size, &new_size);

    clock_gettime(CLOCK_MONOTONIC, &end1);

    time_spent1 = (end1.tv_sec - start1.tv_sec) + (end1.tv_nsec - start1.tv_nsec) / 1e9;

    printf("full cuda_stream_compaction() run time: %f milliseconds\n", time_spent1 * 1000);

    printf("gpu results:\n");
    for (unsigned int i = 0; i < 10; i++)
    {
        printf("%f ", h_out[i]);
    }
    printf("..... ");
    for (unsigned int i = new_size - 10; i < new_size; i++)
    {
        printf("%f ", h_out[i]);
    }

    printf("\n\n");

    struct timespec start2, end2;
    double time_spent2;

    clock_gettime(CLOCK_MONOTONIC, &start2);

    cpu_stream_compaction(h_in, &h_out_cpu, size, &new_size_cpu);

    clock_gettime(CLOCK_MONOTONIC, &end2);

    time_spent2 = (end2.tv_sec - start2.tv_sec) + (end2.tv_nsec - start2.tv_nsec) / 1e9;


    printf("cpu results:\n");
    for (unsigned int i = 0; i < 10; i++)
    {
        printf("%f ", h_out_cpu[i]);
    }
    printf("..... ");
    for (unsigned int i = new_size_cpu - 10; i < new_size_cpu; i++)
    {
        printf("%f ", h_out_cpu[i]);
    }

    printf("\n\n");
    printf("CPU subprogram run time: %f milliseconds\n", time_spent2 * 1000);


    free(h_in);
    free(h_out);
    free(h_out_cpu);

    return 0;
}
