#include <stdio.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <pthread.h>
#include <immintrin.h>
#include <stdint.h>
#include <time.h>


#define NUM_THREADS 128 
#define AVX_FLOAT_COUNT 8


typedef struct {
    const float *data;
    size_t start_index;
    size_t end_index;
    float partial_sum;
} thread_data_t;


static inline float horizontal_sum_avx(__m256 acc) {
    __m128 sum_halves = _mm_add_ps(_mm256_extractf128_ps(acc, 0), _mm256_extractf128_ps(acc, 1));
    sum_halves = _mm_hadd_ps(sum_halves, sum_halves);
    sum_halves = _mm_hadd_ps(sum_halves, sum_halves);
    return _mm_cvtss_f32(sum_halves);
}


void *sum_reduction_thread(void *arg) {
    thread_data_t *t_data = (thread_data_t *)arg;
    t_data->partial_sum = 0.0f;

    __m256 avx_sum_vec = _mm256_setzero_ps();

    size_t i = t_data->start_index;

    for (; i < t_data->end_index; i += AVX_FLOAT_COUNT) {
        __m256 data_vec = _mm256_loadu_ps(&t_data->data[i]);
        avx_sum_vec = _mm256_add_ps(avx_sum_vec, data_vec);
    }

    t_data->partial_sum = horizontal_sum_avx(avx_sum_vec);

    for (; i < t_data->end_index; ++i) {
        t_data->partial_sum += t_data->data[i];
    }

    pthread_exit(NULL);
}


int avx_reduction() {
    unsigned int N = (1 << 29) - 37;
    printf("Performing sum reduction on %u floats with %d threads and AVX2.\n", N, NUM_THREADS);

    float *data = (float *)malloc(N * sizeof(float));

    for (unsigned int i = 0; i < N; ++i) {
        data[i] = (float)(i % 100) + 0.1f;
    }

    pthread_t threads[NUM_THREADS];
    thread_data_t thread_args[NUM_THREADS];
    float total_sum = 0.0f;

    size_t chunk_size = N / NUM_THREADS;
    size_t remainder = N % NUM_THREADS;
    size_t current_start = 0;

    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    for (int i = 0; i < NUM_THREADS; ++i) {
        thread_args[i].data = data;
        thread_args[i].start_index = current_start;
        thread_args[i].end_index = current_start + chunk_size + (i < remainder ? 1 : 0);
        current_start = thread_args[i].end_index;

        if (pthread_create(&threads[i], NULL, sum_reduction_thread, &thread_args[i])) {
            perror("Error creating thread");
            free(data);
            return 1;
        }
    }

    for (int i = 0; i < NUM_THREADS; ++i) {
        pthread_join(threads[i], NULL);
        total_sum += thread_args[i].partial_sum;
    }

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double time_taken = (end_time.tv_sec - start_time.tv_sec) * 1e9;
    time_taken = (time_taken + (end_time.tv_nsec - start_time.tv_nsec)) * 1e-9;

    printf("AVX2 + Threaded Sum: %f\n", total_sum);
    printf("Time taken: %f milliseconds\n", time_taken * 1000);

    free(data);
    return 0;
}
