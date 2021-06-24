#ifndef MATRIXMULTIPLICATION_MATRIXMULTIPLICATION_SYCL_H
#define MATRIXMULTIPLICATION_MATRIXMULTIPLICATION_SYCL_H

#include <CL/sycl.hpp>
#include <iostream>

typedef std::unique_ptr<cl::sycl::buffer<float, 1>> sycl_buffer;

using namespace  cl::sycl;

namespace blas3{

    namespace sycl_blas{

        void naiveMatrixMultiplication(sycl_buffer MatA, sycl_buffer MatB, sycl_buffer result, size_t M, size_t N, size_t K,
                                       queue deviceQueue, int numThreads);

    }
}


#endif //MATRIXMULTIPLICATION_MATRIXMULTIPLICATION_SYCL_H
