#include "../include/MatrixOps_sycl.h"
#include "../include/defines.h"

#include <CL/sycl.hpp>
#include <iostream>


class naiveTranspose;


using namespace cl::sycl;

namespace blas1{

    namespace sycl_blas{

        void naiveTranspose(sycl_buffer MatrixA, sycl_buffer MatrixB, size_t M, size_t N, queue deviceQueue,
                            int numThreads){

            auto local_range = range<2>(numThreads, numThreads);
            auto global_range = range<2>(M / numThreads + 1, N / numThreads + 1) * local_range;

            auto launchParams = nd_range<2>(global_range, local_range);



            deviceQueue.submit([&MatrixA, &MatrixB, M, N, launchParams](handler& cgh){


                auto MatrixA_accessor = MatrixA->get_access<access::mode::read>(cgh);
                auto MatrixB_accessor = MatrixB->get_access<access::mode::write>(cgh);

                cgh.parallel_for<class transpose>(launchParams, [MatrixA_accessor, MatrixB_accessor, M, N](nd_item<2> ndItem){
                    unsigned int row_id = ndItem.get_group(1) * ndItem.get_local_range(1) + ndItem.get_local_id(1);
                    unsigned int column_id = ndItem.get_group(0) * ndItem.get_local_range(0) + ndItem.get_local_id(0);

                    if (column_id < N && row_id < M)
                        MatrixB_accessor[row_id + M * column_id] = MatrixA_accessor[row_id * N + column_id];
                });


            }).wait();

        }

    }

}