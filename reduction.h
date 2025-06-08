#ifndef REDUCTION_H
#define REDUCTION_H

extern "C" void cuda_reduction2(float *h_in, float *h_out, unsigned int size);

extern "C" void cuda_reduction3(float *h_in, float *h_out, unsigned int size);

#endif // REDUCTION_H
