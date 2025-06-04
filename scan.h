
#ifndef SCAN_H
#define SCAN_H

extern "C" void call_scan_kernel1(float *h_in, float *h_out, unsigned int n);

extern "C" void call_scan_kernel2(float *h_in, float *h_out, unsigned int n);

extern "C" void call_scan_kernel3(float *h_in, float *h_out, unsigned int n);

#endif // SCAN_H
