#ifndef MATRIXMULTIPLICATION_MATRIXOPS_SYCL_H
#define MATRIXMULTIPLICATION_MATRIXOPS_SYCL_H

#include <CL/sycl.hpp>

#include "defines.h"

using namespace cl::sycl;

namespace blas1{

    namespace sycl_blas{

        void naiveTranspose(sycl_buffer MatrixA, sycl_buffer MatrixB, size_t M, size_t N, queue deviceQueue,
                            int num_threads);

    }

}

#endif //MATRIXMULTIPLICATION_MATRIXOPS_SYCL_H
