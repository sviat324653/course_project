#include <stdio.h>
#include <cuda_runtime.h>
#include <float.h>
#include <time.h>
#include <stdlib.h>

#include "reduction.h"


#define CUDA_CHECK(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
        exit(EXIT_FAILURE); \
    } \
}


__global__ void block_reduce_sum_kernel1(float *g_in, float *g_out, unsigned int size) {
    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;

    unsigned int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    
    __syncthreads();

    for (unsigned int s = block_size >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            g_in[gtid] += g_in[gtid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_out[blockIdx.x] = g_in[gtid];
    }
}


__global__ void block_reduce_sum_kernel2(const float *g_in, float *g_out, unsigned int size) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;

    unsigned int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float value = (gtid < size) ? g_in[gtid] : 0.0f;

    sdata[tid] = value;

    __syncthreads();

    for (unsigned int s = block_size >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_out[blockIdx.x] = sdata[0];
    }
}


__inline__ __device__ float warp_reduce_sum(float val, unsigned int start_offset) {
    #pragma unroll
    for (unsigned int offset = start_offset; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}


__global__ void block_reduce_sum_kernel3(const float *g_in, float *g_out, unsigned int size) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;

    unsigned int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    float value = (gtid < size) ? g_in[gtid] : 0.0f;

    value = warp_reduce_sum(value, 16);

    if (tid % 32 == 0) {
        sdata[tid / 32] = value;
    }

    __syncthreads();

    if (tid < 8) {
        float final = (tid < block_size / 32) ? sdata[tid] : 0.0f;
        final = warp_reduce_sum(final, 4);

        if (tid == 0) {
            g_out[blockIdx.x] = final;
        }
    }
}


void cuda_reduction2(float *h_in, float *h_out, unsigned int size)
{

    unsigned int array_size_bytes = size * sizeof(float);
    float *d_in, *d_intermediate, *d_final_gpu_result;

    CUDA_CHECK(cudaMalloc(&d_in, array_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_final_gpu_result, sizeof(float))); 

    CUDA_CHECK(cudaMemcpy(d_in, h_in, array_size_bytes, cudaMemcpyHostToDevice));

    unsigned int threads_per_block = 1024;
    unsigned int num_elements_to_reduce = size;
    float *current_d_in = d_in;
    float *current_d_out;

#ifdef BENCHMARK
    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    CUDA_CHECK(cudaEventRecord(start_event, 0));
#endif

    while (num_elements_to_reduce > 1) {
        unsigned int num_blocks = (num_elements_to_reduce + threads_per_block - 1) / threads_per_block;
        
        if (num_blocks == 1) {
            current_d_out = d_final_gpu_result; 
        } else {
            CUDA_CHECK(cudaMalloc(&d_intermediate, num_blocks * sizeof(float)));
            current_d_out = d_intermediate;
        }

        size_t shared_mem_size = threads_per_block * sizeof(float);

        block_reduce_sum_kernel2<<<num_blocks, threads_per_block, shared_mem_size>>>(
            current_d_in, current_d_out, num_elements_to_reduce
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        if (current_d_in != d_in) {
             CUDA_CHECK(cudaFree(current_d_in));
        }
        
        current_d_in = current_d_out;
        num_elements_to_reduce = num_blocks;
    }

#ifdef BENCHMARK
    CUDA_CHECK(cudaEventRecord(stop_event, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));

    printf("kernel2 run time: %f milliseconds\n", milliseconds);
#endif

    CUDA_CHECK(cudaMemcpy(h_out, d_final_gpu_result, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_final_gpu_result));
}


void cuda_reduction3(float *h_in, float *h_out, unsigned int size)
{

    unsigned int array_size_bytes = size * sizeof(float);
    float *d_in, *d_intermediate, *d_final_gpu_result;

    CUDA_CHECK(cudaMalloc(&d_in, array_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_final_gpu_result, sizeof(float))); 

    CUDA_CHECK(cudaMemcpy(d_in, h_in, array_size_bytes, cudaMemcpyHostToDevice));

    unsigned int threadsPerBlock = 256;
    unsigned int numElementsToReduce = size;
    float *current_d_in = d_in;
    float *current_d_out;

    int pass = 0;

#ifdef BENCHMARK
    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    CUDA_CHECK(cudaEventRecord(start_event, 0));
#endif

    while (numElementsToReduce > 1) {
        pass++;
        unsigned int numBlocks = (numElementsToReduce + threadsPerBlock - 1) / threadsPerBlock;
        
        if (numBlocks == 1) {
            current_d_out = d_final_gpu_result; 
        } else {
            CUDA_CHECK(cudaMalloc(&d_intermediate, numBlocks * sizeof(float)));
            current_d_out = d_intermediate;
        }

        size_t sharedMemSize = threadsPerBlock * sizeof(float) / 32;

        block_reduce_sum_kernel3<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
            current_d_in, current_d_out, numElementsToReduce
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        if (current_d_in != d_in) {
             CUDA_CHECK(cudaFree(current_d_in));
        }
        
        current_d_in = current_d_out;
        numElementsToReduce = numBlocks;
    }


#ifdef BENCHMARK
    CUDA_CHECK(cudaEventRecord(stop_event, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));

    printf("kernel3 run time: %f milliseconds\n", milliseconds);
#endif

    CUDA_CHECK(cudaMemcpy(h_out, d_final_gpu_result, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_final_gpu_result));
}


extern "C" int cuda_reduction_test() {
    unsigned int size = (1 << 29);
    unsigned int array_size_bytes = size * sizeof(float);

    float *h_in, *h_out_gpu;

    cudaFree(0); // простро рандомна CUDA API функція, щоб ініціалізувати контекст


    struct timespec start1, end1;
    double time_spent1;
    clock_gettime(CLOCK_MONOTONIC, &start1);
    // cudaHostAlloc(&h_in, array_size_bytes, cudaHostAllocDefault);
    h_in = (float*)malloc(array_size_bytes);
    h_out_gpu = (float*)malloc(sizeof(float));

    clock_gettime(CLOCK_MONOTONIC, &end1);

    time_spent1 = (end1.tv_sec - start1.tv_sec) +
                 (end1.tv_nsec - start1.tv_nsec) / 1e9;

    printf("memory allocation run time: %f milliseconds\n", time_spent1 * 1000);

    double cpu_sum = 0.0;
    for (unsigned int i = 0; i < size; i++) {
        h_in[i] = (float)(i % 100) + 0.1f;
    }

    struct timespec start3, end3;
    double time_spent3;

    clock_gettime(CLOCK_MONOTONIC, &start3);
    for (unsigned int i = 0; i < size; i++) {
        cpu_sum += h_in[i];
    }

    clock_gettime(CLOCK_MONOTONIC, &end3);

    time_spent3 = (end3.tv_sec - start3.tv_sec) +
                 (end3.tv_nsec - start3.tv_nsec) / 1e9;

    printf("CPU subprogram run time: %f milliseconds\n", time_spent3 * 1000);
    
    
    struct timespec start2, end2;
    double time_spent2;

    clock_gettime(CLOCK_MONOTONIC, &start2);

    cuda_reduction2(h_in, h_out_gpu, size);

    clock_gettime(CLOCK_MONOTONIC, &end2);

    time_spent2 = (end2.tv_sec - start2.tv_sec) +
                 (end2.tv_nsec - start2.tv_nsec) / 1e9;

    printf("CUDA reduction run time: %f milliseconds\n", time_spent2 * 1000);

    printf("Sum computed by kernel2: %f\n", *h_out_gpu);
    cuda_reduction3(h_in, h_out_gpu, size);
    printf("Sum computed by kernel3: %f\n", *h_out_gpu);

    float diff = fabsf(*h_out_gpu - (float)cpu_sum);
    float rel_err = diff / fabsf((float)cpu_sum);
    printf("Difference: %e\n", diff);
    printf("Relative Error: %e\n", rel_err);
    if (rel_err < 1e-5) {
        printf("SUCCESS: GPU and CPU results match within tolerance.\n");
    } else {
        printf("FAILURE: GPU and CPU results differ significantly.\n");
    }


    free(h_in);
    free(h_out_gpu);

    return 0;
}
