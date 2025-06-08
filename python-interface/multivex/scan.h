
#ifndef SCAN_H
#define SCAN_H

void cuda_scan1(float *h_in, float *h_out, unsigned int n);

void cuda_scan2(float *h_in, float *h_out, unsigned int n);

void cuda_scan3(float *h_in, float *h_out, unsigned int n);

#endif // SCAN_H
