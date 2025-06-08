#include <stdio.h>


int cuda_reduction_test();
int avx_reduction();


int reduction_test() {
    cuda_reduction_test();
    printf("\n");
    avx_reduction();
    return 0;
}
