#ifndef SORT_H
#define SORT_H

extern "C" void bitonic_sort(int *h_data, unsigned int size);

extern "C" void radix_sort(unsigned int *h_in, unsigned int *h_out);

#endif // SORT_H
