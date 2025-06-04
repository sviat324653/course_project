cdef extern from "scan.h":
    void call_scan_kernel3(float *h_in, float *h_out, unsigned int n)
