#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>

#include "../include/defines.cuh"


__global__ void dotTemplate(float* a, float* b, float* c, size_t size){
    ;
}

__global__ void dot1024(float* vectorA, float* vectorB, float* result, size_t numElements){

    const unsigned int threadPerBlock = 1024;
    __shared__ float cached[threadPerBlock];

    unsigned int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int cacheIndex = threadIdx.x;

    float sum = 0;

    while (Idx < numElements){
        sum += vectorA[Idx] * vectorB[Idx];
        Idx += blockDim.x * gridDim.x;
    }

    cached[cacheIndex] = sum;
    __syncthreads();

    unsigned int i = blockDim.x/2;

    while (i != 0){
        if(cacheIndex < i){
            cached[cacheIndex] += cached[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }
    if(cacheIndex == 0)
        result[blockIdx.x] = cached[0];
}
__global__ void dot512(float* vectorA, float* vectorB, float* result, size_t numElements){

    const unsigned int threadPerBlock = 512;
    __shared__ float cached[threadPerBlock];

    unsigned int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int cacheIndex = threadIdx.x;

    float sum = 0;

    while (Idx < numElements){
        sum += vectorA[Idx] * vectorB[Idx];
        Idx += blockDim.x * gridDim.x;
    }

    cached[cacheIndex] = sum;
    __syncthreads();

    unsigned int i = blockDim.x/2;

    while (i != 0){
        if(cacheIndex < i){
            cached[cacheIndex] += cached[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }
    if(cacheIndex == 0)
        result[blockIdx.x] = cached[0];
}

__global__ void dot256(float* vectorA, float* vectorB, float* result, size_t numElements){

    const unsigned int threadPerBlock = 1024;
    __shared__ float cached[threadPerBlock];

    unsigned int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int cacheIndex = threadIdx.x;

    float sum = 0;

    while (Idx < numElements){
        sum += vectorA[Idx] * vectorB[Idx];
        Idx += blockDim.x * gridDim.x;
    }

    cached[cacheIndex] = sum;
    __syncthreads();

    unsigned int i = blockDim.x/2;

    while (i != 0){
        if(cacheIndex < i){
            cached[cacheIndex] += cached[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }
    if(cacheIndex == 0)
        result[blockIdx.x] = cached[0];
}


namespace blas1{

    namespace cudaBlas {

        float sdot(float *vectorA, float *vectorB, size_t numElements) {
            /*This is a hybrid CPU GPU algorithm.
             * *Probably not that efficient due to memory allocation
             */

            float *h_vectorA, *h_vectorB;
            size_t elementsToCopy;
            int numthreads;

            auto funcPtr = dotTemplate;

            if (numElements > 1024) {
                elementsToCopy = (numElements / 1024) * 1024;
                numthreads = 1024;
                funcPtr = dot1024;
            } else if (512 < numElements and numElements < 1024) {
                elementsToCopy = (numElements / 512) * 512;
                numthreads = 512;
                funcPtr = dot512;
            } else {
                elementsToCopy = (numElements / 256) * 256;
                numthreads = 256;
                funcPtr = dot256;
            }

            auto remaining_elements = numElements - elementsToCopy;

            h_vectorA = (float *) malloc(remaining_elements * sizeof(float));
            h_vectorB = (float *) malloc(remaining_elements * sizeof(float));


            cudaMemcpy(h_vectorA, vectorA + elementsToCopy, remaining_elements * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_vectorB, vectorB + elementsToCopy, remaining_elements * sizeof(float), cudaMemcpyDeviceToHost);

            dim3 threadsPerBlock(numthreads);
            dim3 blocksPerGrid(elementsToCopy / threadsPerBlock.x);

            float *d_result, *h_result;
            cudaMalloc(&d_result, blocksPerGrid.x * sizeof(float));
            h_result = (float *) malloc(blocksPerGrid.x * sizeof(float));

            float GPU_result = 0;
            float cpu_result = 0;

            funcPtr<<<blocksPerGrid, numthreads>>>(vectorA, vectorB, d_result, elementsToCopy);

            //Asynchronous kernel execution, Handle remaining values with CPU

            for (size_t i = 0; i < remaining_elements; i++) {
                cpu_result += h_vectorA[i] * h_vectorB[i];
            }

            cudaDeviceSynchronize(); // Wait for kernel execution to end in Case it didnt

            cudaMemcpy(h_result, d_result, elementsToCopy * sizeof(float), cudaMemcpyDeviceToHost);

            for (size_t i = 0; i < blocksPerGrid.x; i++) {
                GPU_result += h_result[i];
            }

            return GPU_result + cpu_result;
        }

    }

}