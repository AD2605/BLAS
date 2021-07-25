#ifndef MATRIXMULTIPLICATION_MATRIXOPS_SYCL_H
#define MATRIXMULTIPLICATION_MATRIXOPS_SYCL_H

#include <CL/sycl.hpp>

#include "defines.h"

class naiveCopyKernel;
class naiveTransposeKernel;
class sharedTransposeKernel;

using namespace cl::sycl;

namespace blas1{

    namespace sycl_blas{

        void naiveTranspose(sycl_buffer MatrixA, sycl_buffer MatrixB, size_t M, size_t N, queue deviceQueue,
                            int num_threads);

        void naiveCopy(sycl_buffer src, sycl_buffer dst, size_t width, size_t height, int num_threads, queue deviceQueue);

        void sharedTranspose(sycl_buffer src, sycl_buffer dst, size_t width, size_t height, queue deviceQueue, int num_threads);

    }

}

#endif //MATRIXMULTIPLICATION_MATRIXOPS_SYCL_H
