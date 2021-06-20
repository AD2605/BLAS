#ifndef MATRIXMULTIPLICATION_MATRIXMULTIPLICATION_CUH
#define MATRIXMULTIPLICATION_MATRIXMULTIPLICATION_CUH

namespace blas3 {

    __global__ void sharedMatrixMultiplication(float *MatA, float *MatB, float *result, size_t M, size_t K, size_t N);

    __global__ void naiveMatrixMultiplication(float *MatA, float *MatB, float *result, size_t m, size_t n, size_t k);

    __global__ void MatrixMultiplication_reducedOps32(float* MatA, float* MatB, float* result, size_t M, size_t N, size_t K);

    __global__ void MatrixMultiplication_reducedOps16(float* MatA, float* MatB, float* result, size_t M, size_t N, size_t K);

    __global__ void sharedGEMM(float* MatA, float* MatB, float* Result, size_t M, size_t N, size_t K);

}
#endif //MATRIXMULTIPLICATION_MATRIXMULTIPLICATION_CUH
