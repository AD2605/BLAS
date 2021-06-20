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

    __global__ void transposeRowBlock_32(float* input, float* output, size_t width, size_t heights){
        const unsigned int TILE_SIZE = 32;
        const unsigned int BLOCK_ROWS = 8;

        __shared__ float tile[TILE_SIZE][TILE_SIZE+1];

        unsigned int x = blockIdx.x * TILE_SIZE + threadIdx.x;
        unsigned int y = blockIdx.y * TILE_SIZE + threadIdx.y;
        auto w = gridDim.x * TILE_SIZE;

        for(int j=0; j<TILE_SIZE; j += BLOCK_ROWS){
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * w + x];
        }
        __syncthreads();

        auto local_row = blockIdx.y * TILE_SIZE + threadIdx.x;
        auto local_column = blockIdx.x * TILE_SIZE + threadIdx.y;

#pragma unroll
        for(int j = 0; j < TILE_SIZE; j += BLOCK_ROWS){
            output[ (local_column + j) * w + local_row] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}