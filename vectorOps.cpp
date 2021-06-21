#include <iostream>
#include "vectorOps.h"



namespace blas1{
    __global__ void sdot_256(float* vectorA, float* vectorB, size_t elements){
#define NUM_THREADS 256
        __shared__ float cache[NUM_THREADS];

    }
}