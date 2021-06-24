
#include "matrixMultiplication_sycl.h"
#include <CL/sycl.hpp>
#include <iostream>

typedef std::unique_ptr<cl::sycl::buffer<float, 1>> sycl_buffer;

using namespace cl::sycl;

class naive_MatMul_kernel;
class sharedMatrixMultiplication_kernel;

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

                cgh.parallel_for<naive_MatMul_kernel>(launchParams, [MatA_accessor, MatB_accessor, result_accessor, M, N, K]
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

        void sharedMatrixMultiplication(sycl_buffer MatrixA, sycl_buffer MatrixB, sycl_buffer Result, size_t M, size_t N, size_t K,
                                        queue deviceQueue, size_t TILE_SIZE){

            nd_range<2> launchParams = nd_range<2>(cl::sycl::range<2>(M / TILE_SIZE, K / TILE_SIZE),
                                                   cl::sycl::range<2>(TILE_SIZE, TILE_SIZE));

            deviceQueue.submit([&MatrixA, &MatrixB, &Result, M, N, K, launchParams, TILE_SIZE](handler& cgh){

                auto MatA_accessor = MatrixA->get_access<access::mode::read>(cgh);
                auto MatB_accessor = MatrixB->get_access<access::mode::read>(cgh);
                auto result_accessor = Result->get_access<access::mode::write>(cgh);

                accessor<float, 2, access::mode::read_write, access::target::local> shared_A(range<2>(TILE_SIZE,
                                                                                                      TILE_SIZE), cgh);
                accessor<float, 2, access::mode::read_write, access::target::local> shared_B(range<2>(TILE_SIZE,
                                                                                                      TILE_SIZE), cgh);

                cgh.parallel_for<sharedMatrixMultiplication_kernel>(launchParams, [MatA_accessor, MatB_accessor, result_accessor,
                                                                                   shared_A, shared_B, M, N, K, TILE_SIZE](nd_item<2> ndItem){

                    unsigned int block_x_id = ndItem.get_group(0);
                    unsigned int block_y_id = ndItem.get_group(1);

                    unsigned int thread_x_id = ndItem.get_local_id(0);
                    unsigned int thread_y_id = ndItem.get_local_id(1);


                    auto a_start_index = K * TILE_SIZE * block_y_id;
                    auto a_end = a_start_index + K - 1;
                    auto a_step = TILE_SIZE;

                    auto b_start_index = TILE_SIZE * block_x_id;
                    auto b_step = TILE_SIZE * N;

                    float sum = 0.0f;

                    for(int a = a_start_index, b = b_start_index; a <= a_end; a += a_step, b += b_step){
                        shared_A[thread_y_id][thread_x_id] = MatA_accessor[a + K * thread_y_id + thread_x_id];
                        shared_B[thread_y_id][thread_x_id] = MatB_accessor[b + N * thread_y_id + thread_x_id];

                        ndItem.barrier();

                        for(int i = 0; i < TILE_SIZE; i++){
                            sum += shared_A[thread_y_id][i] * shared_B[i][thread_x_id];
                        }

                        ndItem.barrier();

                    }

                    auto result_index = N * TILE_SIZE * block_y_id + TILE_SIZE * block_x_id;
                    result_accessor[result_index + N * thread_y_id + thread_x_id] = sum;
;
                });

            });

            deviceQueue.wait();

        }


    }
}

