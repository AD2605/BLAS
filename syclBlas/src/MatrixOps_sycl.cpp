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

                cgh.parallel_for<naiveTransposeKernel>(launchParams, [MatrixA_accessor, MatrixB_accessor, M, N](nd_item<2> ndItem){
                    unsigned int row_id = ndItem.get_group(1) * ndItem.get_local_range(1) + ndItem.get_local_id(1);
                    unsigned int column_id = ndItem.get_group(0) * ndItem.get_local_range(0) + ndItem.get_local_id(0);

                    if (column_id < N && row_id < M)
                        MatrixB_accessor[row_id + M * column_id] = MatrixA_accessor[row_id * N + column_id];
                });


            }).wait();

        }

        void naiveCopy(sycl_buffer src, sycl_buffer dst, size_t width, size_t height, int num_threads, queue deviceQueue){

            auto local_range = range<2>(num_threads, num_threads);
            auto global_range = range<2>(width / num_threads + 1, height / num_threads + 1) * local_range;
            auto launch_params = nd_range<2>(global_range, local_range);

            deviceQueue.submit([&src, &dst, width, height, launch_params](handler& cgh){

                auto src_accessor = src->get_access<access::mode::read>(cgh);
                auto dst_accessor = dst->get_access<access::mode::write>(cgh);

                cgh.parallel_for<naiveCopyKernel>(launch_params, [src_accessor, dst_accessor, width, height](nd_item<2> ndItem){

                    unsigned int row_index = ndItem.get_global_id(0);
                    unsigned int column_index = ndItem.get_global_id(1);

                    if (row_index < height and column_index < width){

                        dst_accessor[row_index * width + column_index] = src_accessor[row_index * width + column_index];

                    }

                });

            }).wait();

        }

        void sharedTranspose(sycl_buffer src, sycl_buffer dst, size_t width, size_t height, queue deviceQueue, int num_threads){

            auto local_range = range<2>(num_threads, num_threads);
            auto global_range = range<2>(height / num_threads , width / num_threads) * local_range;
            auto launch_params = nd_range<2>(global_range, local_range);

            deviceQueue.submit([&src, &dst, width, height, launch_params, num_threads](handler& cgh){

                auto src_accessor = src->get_access<access::mode::read>(cgh);
                auto dst_accessor = src->get_access<access::mode::read_write>(cgh);

                cl::sycl::accessor<float, 2, access::mode::read_write, access::target::local> sharedMem(range<2>(num_threads,
                                                                                                                 num_threads + 1), cgh);

                cgh.parallel_for<sharedTransposeKernel>(launch_params, [src_accessor, dst_accessor, sharedMem, width, height, num_threads](nd_item<2> ndItem){

                    unsigned int row_index = ndItem.get_group(0) * ndItem.get_local_range(0) + ndItem.get_local_id(0);
                    unsigned int column_index = ndItem.get_group(1) * ndItem.get_local_range(1) + ndItem.get_local_id(1);

                    unsigned int row_thread = ndItem.get_local_id(0);
                    unsigned int column_thread = ndItem.get_local_id(1);

                    unsigned int element_index = row_index * width + column_index;
                    sharedMem[row_thread][column_thread] = src_accessor[element_index];

                    ndItem.barrier();

                    unsigned int local_row = ndItem.get_group(0) * num_threads + column_thread;
                    unsigned int local_column = ndItem.get_group(1) * num_threads + row_thread;

                    dst_accessor[local_column * height + local_row] = sharedMem[local_row][local_column];

                });

            }).wait();

        }

    }

}