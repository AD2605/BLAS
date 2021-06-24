#include <iostream>
#include "matrixMultiplication.cuh"

namespace blas3{

    __global__ void naiveMatrixMultiplication(float* MatA, float* MatB, float* result, size_t m, size_t n, size_t k){
        unsigned int column_id = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int row_id = blockIdx.y * blockDim.y + threadIdx.y;

        float sum = 0.0f;

        if(row_id < m && column_id < k) {

            for (size_t i = 0; i < k; i++) {
                sum += MatA[n * row_id + i] * MatB[i * k + column_id];
            }
            result[row_id * n + column_id] = sum;
        }
    }


    __global__ void sharedMatrixMultiplication(float* MatA, float* MatB, float* result, size_t M, size_t K, size_t N) {
        /*
         * Matrix Multiplication with
         * Tiling
         * Coalesced Access
         * No Bank Conflict
         * */
#define BLOCK_SIZE 32

        unsigned int block_x_id = blockIdx.x;
        unsigned int block_y_id = blockIdx.y;

        unsigned int thread_x_id = threadIdx.x;
        unsigned int thread_y_id = threadIdx.y;

            __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE];
            __shared__ float B_shared[BLOCK_SIZE][BLOCK_SIZE];

            auto a_start_index = K * BLOCK_SIZE * block_y_id;
            auto a_end = a_start_index + K - 1;
            int a_step = BLOCK_SIZE;

            auto b_start_index = BLOCK_SIZE * block_x_id;
            auto b_step = BLOCK_SIZE * N;

            float sum = 0.0f;

            for (int a = a_start_index, b = b_start_index; a <= a_end; a += a_step, b += b_step) {
                A_shared[thread_y_id][thread_x_id] = MatA[a + K * thread_y_id + thread_x_id];
                B_shared[thread_y_id][thread_x_id] = MatB[b + N * thread_y_id + thread_x_id];

                __syncthreads();


#pragma unroll
                for (int k = 0; k < BLOCK_SIZE; k++) {
                    sum += A_shared[thread_y_id][k] * B_shared[k][thread_x_id];
                }

                __syncthreads();
            }

            auto result_index = N * BLOCK_SIZE * block_y_id + BLOCK_SIZE * block_x_id;
            result[result_index + N * thread_y_id + thread_x_id] = sum;


        }


    __global__ void MatrixMultiplication_reducedOps32(float* MatA, float* MatB, float* result, size_t M, size_t N, size_t K){
        const int TILE_SIZE = 32;
        const int VECTOR_SIZE = 4;

        unsigned int block_x = blockIdx.x;  unsigned int thread_x = threadIdx.x;
        unsigned int block_y = blockIdx.y;  unsigned int thread_y = threadIdx.y;

        __shared__ float A_shared[TILE_SIZE * TILE_SIZE];
        float result_vector[TILE_SIZE] = {0};

        auto a_start_index = K * TILE_SIZE * block_y;
        auto a_end = a_start_index + K - 1;
        auto a_step = TILE_SIZE;

        auto b_start_index = TILE_SIZE * VECTOR_SIZE * block_x;
        auto b_step = TILE_SIZE * N;

        for(int a = a_start_index, b = b_start_index; a <= a_end; a += a_step, b += b_step){

#pragma unroll
            for(int i=0; i < TILE_SIZE/VECTOR_SIZE; ++i){

                A_shared[ (i * VECTOR_SIZE + thread_y) + TILE_SIZE * thread_x ] =
                        MatA[a + K*(i * VECTOR_SIZE + thread_y) + thread_x];
            }
            __syncthreads();

            float* a_shared_base = A_shared;
            float* b_base = MatB + b + TILE_SIZE * thread_y + thread_x;

#pragma unroll
            for(int i=0; i<TILE_SIZE; i++){
                float b_value = *b_base;

                result_vector[0] += a_shared_base[0] * b_value;
                result_vector[1] += a_shared_base[1] * b_value;
                result_vector[2] += a_shared_base[2] * b_value;
                result_vector[3] += a_shared_base[3] * b_value;
                result_vector[4] += a_shared_base[4] * b_value;
                result_vector[5] += a_shared_base[5] * b_value;
                result_vector[6] += a_shared_base[6] * b_value;
                result_vector[7] += a_shared_base[7] * b_value;
                result_vector[8] += a_shared_base[8] * b_value;
                result_vector[9] += a_shared_base[9] * b_value;

                result_vector[10] += a_shared_base[10] * b_value;
                result_vector[11] += a_shared_base[11] * b_value;
                result_vector[12] += a_shared_base[12] * b_value;
                result_vector[13] += a_shared_base[13] * b_value;
                result_vector[14] += a_shared_base[14] * b_value;
                result_vector[15] += a_shared_base[15] * b_value;
                result_vector[16] += a_shared_base[16] * b_value;
                result_vector[17] += a_shared_base[17] * b_value;
                result_vector[18] += a_shared_base[18] * b_value;
                result_vector[19] += a_shared_base[19] * b_value;

                result_vector[20] += a_shared_base[20] * b_value;
                result_vector[21] += a_shared_base[21] * b_value;
                result_vector[22] += a_shared_base[22] * b_value;
                result_vector[23] += a_shared_base[23] * b_value;
                result_vector[24] += a_shared_base[24] * b_value;
                result_vector[25] += a_shared_base[25] * b_value;
                result_vector[26] += a_shared_base[26] * b_value;
                result_vector[27] += a_shared_base[27] * b_value;
                result_vector[28] += a_shared_base[28] * b_value;
                result_vector[29] += a_shared_base[29] * b_value;

                result_vector[30] += a_shared_base[30] * b_value;
                result_vector[31] += a_shared_base[31] * b_value;

                a_shared_base += TILE_SIZE;
                b_base += N;
            }
            __syncthreads();
        }

        auto c_ptr = N * TILE_SIZE * block_y + TILE_SIZE * VECTOR_SIZE * block_x;
        c_ptr += TILE_SIZE * thread_y + thread_x;

#pragma unroll
        for(int i=0; i<TILE_SIZE; ++i){
            result[c_ptr] = result_vector[i];
            c_ptr += N;
        }
    }


    __global__ void MatrixMultiplication_reducedOps16(float* MatA, float* MatB, float* result, size_t M, size_t N, size_t K){
        const int TILE_SIZE_16 = 16;
        const int VECTOR_SIZE = 4;

        unsigned int block_x = blockIdx.x;  unsigned int thread_x = threadIdx.x;
        unsigned int block_y = blockIdx.y;  unsigned int thread_y = threadIdx.y;

        __shared__ float A_shared[TILE_SIZE_16 * TILE_SIZE_16];
        float result_vector[TILE_SIZE_16] = {0};

        auto a_start_index = K * TILE_SIZE_16 * block_y;
        auto a_end = a_start_index + K - 1;
        auto a_step = TILE_SIZE_16;

        auto b_start_index = TILE_SIZE_16 * VECTOR_SIZE * block_x;
        auto b_step = TILE_SIZE_16 * N;

        for(int a = a_start_index, b = b_start_index; a <= a_end; a += a_step, b += b_step){
#pragma unroll
            for(int i=0; i < TILE_SIZE_16/VECTOR_SIZE; ++i){
                A_shared[ (i * VECTOR_SIZE + thread_y) + TILE_SIZE_16 * thread_x ] =
                        MatA[a + K*(i * VECTOR_SIZE + thread_y) + thread_x];
            }
            __syncthreads();

            float* a_shared_base = A_shared;
            float* b_base = MatB + b + TILE_SIZE_16 * thread_y + thread_x;

#pragma unroll
            for(int i=0; i<TILE_SIZE_16; i++){
                float b_value = *b_base;
                result_vector[0] += a_shared_base[0] * b_value;
                result_vector[1] += a_shared_base[1] * b_value;
                result_vector[2] += a_shared_base[2] * b_value;
                result_vector[3] += a_shared_base[3] * b_value;
                result_vector[4] += a_shared_base[4] * b_value;
                result_vector[5] += a_shared_base[5] * b_value;
                result_vector[6] += a_shared_base[6] * b_value;
                result_vector[7] += a_shared_base[7] * b_value;
                result_vector[8] += a_shared_base[8] * b_value;
                result_vector[9] += a_shared_base[9] * b_value;

                result_vector[10] += a_shared_base[10] * b_value;
                result_vector[11] += a_shared_base[11] * b_value;
                result_vector[12] += a_shared_base[12] * b_value;
                result_vector[13] += a_shared_base[13] * b_value;
                result_vector[14] += a_shared_base[14] * b_value;
                result_vector[15] += a_shared_base[15] * b_value;

                a_shared_base += TILE_SIZE_16;
                b_base += N;
            }
            __syncthreads();
        }

        auto c_ptr = N * TILE_SIZE_16 * block_y + TILE_SIZE_16 * VECTOR_SIZE * block_x;
        c_ptr += TILE_SIZE_16 * thread_y + thread_x;

#pragma unroll
        for(int i=0; i<TILE_SIZE_16; ++i){
            result[c_ptr] = result_vector[i];
            c_ptr += N;
        }
    }

    __global__ void sharedGEMM(float* MatA, float* MatB, float* Result, size_t M, size_t N, size_t K){
        const unsigned int TILE_SIZE =32;

        unsigned int row_id = blockIdx.y * TILE_SIZE + threadIdx.y;
        unsigned int column_id = blockIdx.x * TILE_SIZE + threadIdx.x;

        __shared__ float A_shared[TILE_SIZE][TILE_SIZE];
        __shared__ float B_shared[TILE_SIZE][TILE_SIZE];

        float sum = 0.0f;

        for(size_t m = 0; m < (TILE_SIZE + N - 1)/TILE_SIZE; m++){
            if(m * TILE_SIZE + threadIdx.x < N and row_id < M)
                A_shared[threadIdx.y][threadIdx.x] = MatA[row_id * N + m * TILE_SIZE + threadIdx.x];
            else
                A_shared[threadIdx.y][threadIdx.x] = 0.0f;

            if (m * TILE_SIZE + threadIdx.y < N && column_id < K) {
                B_shared[threadIdx.y][threadIdx.x] = MatB[(m * TILE_SIZE + threadIdx.y) * K + column_id]; //Coalesced ??
            }
            else
                B_shared[threadIdx.y][threadIdx.x] = 0.0f;

            __syncthreads();


            for(int n=0; n<TILE_SIZE; n++){
                sum += A_shared[threadIdx.y][n] * B_shared[n][threadIdx.x];
            }
            __syncthreads();
        }
        if(row_id < M and column_id < K){
            Result[((blockIdx.y * blockDim.y + threadIdx.y) * K) +
            blockIdx.x * blockDim.x + threadIdx.x ] = sum;
        }
    }

}

/*
int main(){
    size_t m = 102;
    size_t n = 604;
    size_t k = 366;

    float* h_a, *h_b, *h_c;
    float* d_a, *d_b, *d_c;

    h_a = (float*) malloc(m * n * sizeof(float ));
    h_b = (float*) malloc(n * k * sizeof(float ));
    h_c = (float*) malloc(m * k * sizeof(float ));

    cudaMalloc(&d_a, m * n * sizeof(float ));
    cudaMalloc(&d_b, n * k * sizeof(float ));
    cudaMalloc(&d_c, m * k * sizeof(float ));


    for(int i = 0; i < m * n;i++){
        h_a[i] = 1.0f;
    }

    for(int i=0; i< n*k; i++){
        h_b[i] = 1.0f;
    }

    cudaMemcpy(d_a, h_a, m * n * sizeof(float ), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * k * sizeof(float ), cudaMemcpyHostToDevice);

    dim3 threads(32, 32);
    dim3 blocksPerGrid((n - 1) / threads.x + 1, (n - 1) / threads.y + 1);

    blas3::sharedGEMM<<<blocksPerGrid, threads>>>(d_a, d_b, d_c, m ,n, k);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c, d_c, m * k * sizeof(float ), cudaMemcpyDeviceToHost);
    for(int i=0; i<500; i++){
        std::cout<<h_c[i]<<"  ";
    }
    std::cout<<std::endl;
}

 */