#ifndef MATRIXMULTIPLICATION_VECTOROPS_CUH
#define MATRIXMULTIPLICATION_VECTOROPS_CUH


namespace blas1{

    namespace cudaBlas {

        __global__ void vectorAdd(const float* v1, const float * v2, float * out, size_t N);

        __global__ void vectorAdd_(float* v1, const float * v2, size_t N);

        __global__ void scalarMulitplication(const float* vector, float* out, float scalar, size_t N);

        __global__ void scalarMulitplication_(float* vector, float* out, float scalar, size_t N);

        __global__ void full_dot(const float * v1, const float * v2, float * out, size_t N);

    }
}

#endif //MATRIXMULTIPLICATION_VECTOROPS_CUH
