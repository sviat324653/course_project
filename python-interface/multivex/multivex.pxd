cdef extern from "scan.h":
    void cuda_scan3(float *h_in, float *h_out, unsigned int n)

cdef extern from "reduction.h":
    void cuda_reduction3(float *h_in, float *h_out, unsigned int size)

cdef extern from "stream_compaction.h":
    void cuda_stream_compaction(float *h_in, float **h_out, unsigned int size, unsigned int *new_size)

cdef extern from "sort.h":
    void bitonic_sort(int *h_data, unsigned int size)
    void radix_sort(unsigned int *h_in, unsigned int *h_out)
