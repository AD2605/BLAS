#ifndef MATRIXMULTIPLICATION_MATRIXMULTIPLICATION_SYCL_H
#define MATRIXMULTIPLICATION_MATRIXMULTIPLICATION_SYCL_H

#include <CL/sycl.hpp>
#include <iostream>

#include "defines.h"

using namespace  cl::sycl;

class naive_MatMul_kernel;
class sharedMatrixMultiplication_kernel;
class matmulreducedOps;


namespace blas3{

    namespace sycl_blas{

        void naiveMatrixMultiplication(sycl_buffer MatA, sycl_buffer MatB, sycl_buffer result, size_t M, size_t N, size_t K,
                                       queue deviceQueue, int numThreads);

        void sharedMatrixMultiplication(sycl_buffer MatrixA, sycl_buffer MatrixB, sycl_buffer Result, size_t M, size_t N, size_t K,
                                        queue deviceQueue, size_t TILE_SIZE);

        void sharedGEMM(sycl_buffer MatrixA, sycl_buffer MatrixB, sycl_buffer Result, size_t M, size_t N, size_t K,
                        queue deviceQueue, size_t TILE_SIZE);

        void matmul_reducedOps_sycl(sycl_buffer MatA, sycl_buffer MatB, sycl_buffer result, size_t M, size_t N, size_t K,
                                    int TILE_SIZE, int VECTOR_SIZE, queue deviceQueue);

    }
}


#endif //MATRIXMULTIPLICATION_MATRIXMULTIPLICATION_SYCL_H
