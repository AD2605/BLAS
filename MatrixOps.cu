#include <cmath>
#include <cuda_runtime_api.h>

#include "MatrixOps.cuh"

namespace blas1{

    __global__ void naiveTranspose(float* input, float* output, size_t width, size_t height){
        unsigned int row_id = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int column_id = blockIdx.x * blockDim.x + threadIdx.x;

        if((row_id < height) && (column_id < width)){
            output[row_id + height  * column_id] = input[row_id * width + column_id];
        }
    }


    __global__ void transpose_32(float* input, float* output, size_t width, size_t height){
        const unsigned int BLOCK_SIZE = 32;

        __shared__ float sharedMem[BLOCK_SIZE][BLOCK_SIZE + 1]; // Bank conflict

        unsigned int column_index = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int row_index = blockIdx.y * blockDim.y + threadIdx.y;

        if((column_index < width) && (row_index < height)){

            unsigned element_index = row_index * width + column_index;
            sharedMem[threadIdx.y][threadIdx.x] = input[element_index];
        }

        __syncthreads();

        unsigned int local_row = blockIdx.y * BLOCK_SIZE + threadIdx.x;  // coalesced access
        unsigned int local_column = blockIdx.x * BLOCK_SIZE + threadIdx.y; // coalesced access

        if((local_row < height) && (local_column < width)){
            output[local_column * height + local_row] = sharedMem[threadIdx.x][threadIdx.y];
        }
    }

    __global__ void transpose_16(float* input, float* output, size_t width, size_t height){
        const unsigned int BLOCK_SIZE = 16;

        __shared__ float sharedMem[BLOCK_SIZE][BLOCK_SIZE + 1]; // Bank conflict

        unsigned int column_index = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int row_index = blockIdx.y * blockDim.y + threadIdx.y;

        if((column_index < width) && (row_index < height)){

            unsigned element_index = row_index * width + column_index;
            sharedMem[threadIdx.y][threadIdx.x] = input[element_index];
        }

        __syncthreads();

        unsigned int local_row = blockIdx.y * BLOCK_SIZE + threadIdx.x;  // coalesced access
        unsigned int local_column = blockIdx.x * BLOCK_SIZE + threadIdx.y; // coalesced access

        if((local_row < height) && (local_column < width)){
            output[local_column * height + local_row] = sharedMem[threadIdx.x][threadIdx.y];
        }
    }
}