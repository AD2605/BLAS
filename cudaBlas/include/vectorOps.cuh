#ifndef MATRIXMULTIPLICATION_VECTOROPS_CUH
#define MATRIXMULTIPLICATION_VECTOROPS_CUH


namespace blas1{

    namespace cudaBlas {

        float sdot(float *vectorA, float *vectorB, size_t numElements);

    }
}

#endif //MATRIXMULTIPLICATION_VECTOROPS_CUH
