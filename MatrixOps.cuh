#ifndef MATRIXMULTIPLICATION_MATRIXOPS_CUH
#define MATRIXMULTIPLICATION_MATRIXOPS_CUH

namespace blas1{
    __global__ void naiveTranspose(float* input, float* output, size_t width, size_t height);

    __global__ void naiveCopy(float* input, float* output, size_t width, size_t height);

    __global__ void transpose_32(float* input, float* output, size_t width, size_t height);

    __global__ void transpose_16(float* input, float* output, size_t width, size_t height);

    __global__ void transposeRowBlock_32(float* input, float* output, size_t width, size_t heights);

    __global__ void MatrixCopy_32(float* input, float* output, size_t width, size_t height);

    __global__ void MatrixCopy_16(float* input, float* output, size_t width, size_t height);
}


#endif //MATRIXMULTIPLICATION_MATRIXOPS_CUH