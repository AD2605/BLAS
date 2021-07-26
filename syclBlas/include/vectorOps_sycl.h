

#ifndef MATRIXMULTIPLICATION_VECTOROPS_SYCL_H
#define MATRIXMULTIPLICATION_VECTOROPS_SYCL_H



#include "../include/defines.h"

#include <CL/sycl.hpp>

using namespace cl::sycl;

class vectorAddKernel;
class vectorAddKernel_;
class scalarMulitplicationKernel;
class scalarMultiplicationKernel_;


namespace blas1{

    namespace sycl_blas{

        void vectorAdd(sycl_buffer v1, sycl_buffer v2, sycl_buffer out, size_t N, int num_threads, queue deviceQueue);

        void vectorAdd_(sycl_buffer v1, sycl_buffer v2, size_t N, int num_threads, queue deviceQueue);

        void scalarMultiplication(sycl_buffer v1, sycl_buffer out, float scalar, size_t N, int num_threads, queue deviceQueue);

        void scalarMultiplication_(sycl_buffer v1, float scalar, size_t N, int  num_threads, queue deviceQueue);


    }

}

#endif //MATRIXMULTIPLICATION_VECTOROPS_SYCL_H