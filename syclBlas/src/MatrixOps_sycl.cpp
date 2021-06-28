#include "../include/MatrixOps_sycl.h"
#include "../include/defines.h"

#include <CL/sycl.hpp>
#include <iostream>


class naiveTranspose;


using namespace cl::sycl;

namespace blas1{

    namespace sycl_blas{

        void naiveTranspose(sycl_buffer MatrixA, sycl_buffer MatrixB, size_t M, size_t N, queue deviceQueue,
                            int num_threads){

            nd_range<2> launchParams = nd_range<2>(range<2>(M / num_threads + 1, N / num_threads + 1),
                    range<2>(num_threads, num_threads));


            deviceQueue.submit([&MatrixA, &MatrixB, M, N, launchParams](handler& cgh){


                auto MatrixA_accessor = MatrixA->get_access<access::mode::read>(cgh);
                auto MatrixB_accessor = MatrixB->get_access<access::mode::write>(cgh);

                cgh.parallel_for<naiveTranspose>(launchParams, [MatrixA_accessor, MatrixB_accessor, M, N](nd_item<2> ndItem){

                    unsigned int row_id = ndItem.get_group(1) * ndItem.get_local_range(1) + ndItem.get_local_id(1);
                    unsigned int column_id = ndItem.get_group(0) * ndItem.get_local_range(0) + ndItem.get_local_id(0);

                    if (column_id < N && row_id < M)
                        MatrixB_accessor[row_id + M * column_id] = MatrixA_accessor[row_id * N + column_id];

                });
            });
            deviceQueue.wait();
        }




    }

}