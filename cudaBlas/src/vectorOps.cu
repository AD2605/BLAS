#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>

#include "../include/defines.cuh"


namespace blas1{

    namespace cudaBlas {


        __global__ void vectorAdd(const float* v1, const float * v2, float * out, size_t N){

            unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

            if (tid < N)
                out[tid] = v1[tid] + v2[tid];

        }

        __global__ void vectorAdd_(float* v1, const float * v2, size_t N){

            unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

            if(tid < N)
                v1[tid] += v2[tid];

        }

        __global__ void scalarMulitplication(const float* vector, float* out, float scalar, size_t N){

            unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

            if (tid < N)
                out[tid] = vector[tid] * scalar;

        }


        __global__ void scalarMulitplication_(float* vector, float* out, float scalar, size_t N){

            unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

            if (tid < N)
                vector[tid] = vector[tid] * scalar;

        }


        __global__ void full_dot(const float * v1, const float * v2, float * out, size_t N) {

            __shared__ float cache[BLOCK_SIZE];
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            cache[threadIdx.x] = 0.f;

            while (i < N) {
                cache[threadIdx.x] += v1[i] * v2[i];
                i += gridDim.x * blockDim.x;
            }

            __syncthreads();
            i = BLOCK_SIZE / 2;

            while (i > 0) {
                if (threadIdx.x < i) cache[threadIdx.x] += cache[threadIdx.x + i];
                __syncthreads();
                i /= 2;  // not sure bitwise operations are actually faster
            }

            if (threadIdx.x == 0) {
                atomicAdd(out, cache[0]);
            }

        }


    }

}