#include <stdio.h>


int cuda_reduction();
int avx_reduction();


int reduction_test() {
    cuda_reduction();
    printf("\n");
    avx_reduction();
    return 0;
}
