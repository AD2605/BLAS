
#include "matrixMultiplication_sycl.h"
#include <CL/sycl.hpp>
#include <iostream>

typedef std::unique_ptr<cl::sycl::buffer<float, 1>> sycl_buffer;

using namespace cl::sycl;

class naive_MatMul;

namespace blas3{

    namespace sycl_blas{

        void naiveMatrixMultiplication(sycl_buffer MatA, sycl_buffer MatB, sycl_buffer result, size_t M, size_t N, size_t K,
                                        queue deviceQueue, int numThreads){

            nd_range<2> launchParams = nd_range<2>(cl::sycl::range<2>(M / numThreads + 1, K / numThreads + 1),
                    cl::sycl::range<2>(numThreads, numThreads));

            deviceQueue.submit([&MatA, &MatB, &result, M, N, K, launchParams](handler& cgh){

                auto MatA_accessor = MatA->get_access<access::mode::read>(cgh);
                auto MatB_accessor = MatB->get_access<access::mode::read>(cgh);
                auto result_accessor = result->get_access<access::mode::read_write>(cgh);

                cgh.parallel_for<naive_MatMul>(launchParams, [MatA_accessor, MatB_accessor, result_accessor, M, N, K]
                                                                (nd_item<2> ndItem){

                    auto column_index = ndItem.get_group(0) * ndItem.get_local_range(0) + ndItem.get_local_id(0);
                    auto row_index = ndItem.get_group(1) * ndItem.get_local_range(1) + ndItem.get_local_id(1);

                    if(row_index < M && column_index < K){
                        float sum = 0.0f;
                        for (int i = 0; i < N; i++) {
                            sum += MatA_accessor[N * row_index + i] * MatB_accessor[i * N + column_index];
                        }
                        result_accessor[N * row_index + column_index] = sum;
                    }
                });
            });
            deviceQueue.wait();
        }
    }
}