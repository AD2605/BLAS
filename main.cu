#include <iostream>
#include <cassert>
#include "matrixMultiplication.cuh"

int main() {
    size_t m = 1024;
    size_t n = 1024;
    size_t k = 1024;


    auto MatA = (float *) malloc(m * n * sizeof(float));    
    auto MatB = (float *) malloc(n * k * sizeof(float));
    auto result = (float *) malloc(m * k * sizeof(float));

    float *device_MatA, *device_MatB, *device_result;

    cudaMalloc(&device_MatA, m * n * sizeof(float));
    cudaMalloc(&device_MatB, n * k * sizeof(float));
    cudaMalloc(&device_result, m * k * sizeof(float));



    //std::cout<<"=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-"<<std::endl;


    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            *(MatA + i * n + j) = 1.0f;
        }
    }


    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < k; j++) {
            *(MatB + i * k + j) = 2.0f;
        }
    }

    cudaMemcpy(device_MatA, MatA, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_MatB, MatB, n * k * sizeof(float), cudaMemcpyHostToDevice);


    dim3 num_threads(32, 4);
    dim3 num_blocks(m / (num_threads.x * num_threads.y), k / num_threads.x);

    //MatrixMultiplication<<<num_blocks, num_threads>>>(device_MatA, device_MatB, device_result, m, k);
    blas3::MatrixMultiplication_reducedOps32<<<num_blocks, num_threads>>>(device_MatA, device_MatB, device_result, m, n, k);
    //matmul_CompOpt<<<num_blocks, num_threads>>>(device_MatA, device_MatB, device_result, m, n, k);
    cudaDeviceSynchronize();
    cudaMemcpy(result, device_result, m * k * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << *(result + 1) << std::endl;

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < k; j++) {
            //std::cout<<*(result + i * m + j)<<"  ";
            assert(*(result + i * k + j) == 2048);
        }
        //std::cout<<std::endl<<"================================================================"<<std::endl;
    }
     
}
