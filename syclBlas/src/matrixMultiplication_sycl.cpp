
#include "../include/matrixMultiplication_sycl.h"
#include "../include/defines.h"

#include <CL/sycl.hpp>
#include <iostream>



using namespace cl::sycl;


namespace blas3{

    namespace sycl_blas{

        void naiveMatrixMultiplication(sycl_buffer MatA, sycl_buffer MatB, sycl_buffer result, size_t M, size_t N, size_t K,
                                        queue deviceQueue, int numThreads){

            auto local_range = range<2>(numThreads, numThreads);
            auto global_range = range<2>(M / numThreads + 1, K / numThreads + 1) * local_range;

            auto launchParams = nd_range<2>(global_range, local_range);

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
            }).wait();

        }

        void sharedMatrixMultiplication(sycl_buffer MatrixA, sycl_buffer MatrixB, sycl_buffer Result, size_t M, size_t N, size_t K,
                                        queue deviceQueue, size_t TILE_SIZE){

            auto local_range = range<2>(TILE_SIZE, TILE_SIZE);
            auto global_range = range<2>(M / TILE_SIZE, K / TILE_SIZE) * local_range;

            auto launchParams = nd_range<2>(global_range, local_range);


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

        void sharedGEMM(sycl_buffer MatrixA, sycl_buffer MatrixB, sycl_buffer Result, size_t M, size_t N, size_t K,
                        queue deviceQueue, size_t TILE_SIZE){

            auto device = deviceQueue.get_device();

            auto local_range = range<2>(TILE_SIZE, TILE_SIZE);
            auto global_range = range<2>(M / TILE_SIZE + 1, K / TILE_SIZE + 1) * local_range;

            auto launchParams = nd_range<2>(global_range, local_range);


            deviceQueue.submit([&MatrixA, &MatrixB, &Result, M, N, K, TILE_SIZE, launchParams](handler& cgh){

                auto MatA_accessor = MatrixA->get_access<access::mode::read>(cgh);
                auto MatB_accessor = MatrixB->get_access<access::mode::read>(cgh);
                auto result_accessor = Result->get_access<access::mode::read_write>(cgh);

                accessor<float, 2, access::mode::read_write, access::target::local> A_shared(range<2>(TILE_SIZE,
                                                                                                      TILE_SIZE), cgh);
                accessor<float, 2, access::mode::read_write, access::target::local> B_shared(range<2>(TILE_SIZE,
                                                                                                      TILE_SIZE), cgh);

                cgh.parallel_for<class sharedGemm>(launchParams, [MatA_accessor, MatB_accessor, result_accessor, A_shared, B_shared,
                                                M, N, K, TILE_SIZE](nd_item<2> item){
                    /*
                    unsigned int group_id_x = item.get_group(0);
                    unsigned int group_id_y = item.get_group(1);

                    unsigned int thread_x = item.get_local_id(0);
                    unsigned int thread_y = item.get_local_id(1);

                    auto a_start_index = K * TILE_SIZE * group_id_y;
                    auto a_end = a_start_index + K - 1;
                    auto a_step = TILE_SIZE;*/

                    unsigned int row_id = item.get_group(1) * TILE_SIZE + item.get_local_id(1);
                    unsigned int column_id = item.get_group(0) * TILE_SIZE + item.get_local_id(0);

                    float sum = 0.0f;

                    for(size_t i = 0; i < (TILE_SIZE + N - 1) / TILE_SIZE; i++){
                        if(i * TILE_SIZE + item.get_local_id(0) < N && row_id < M){
                            A_shared[item.get_local_id(1)][item.get_local_id(0)] =
                                    MatA_accessor[row_id * N + i * TILE_SIZE + item.get_local_id(0)];
                        }
                        else
                            A_shared[item.get_local_id(1)][item.get_local_id(0)] = 0.0f;

                        if(i * TILE_SIZE + item.get_local_id(1) < N && column_id < K)
                            B_shared[item.get_local_id(1)][item.get_local_id(0)] =
                                    MatB_accessor[(i* TILE_SIZE + item.get_local_id(1) * K + column_id)];

                        else
                            B_shared[item.get_local_id(1)][item.get_local_id(0)] = 0.0f;
                    }

                    item.barrier();

                    for(int j = 0; j < TILE_SIZE; j++)
                        sum += A_shared[item.get_local_id(1)][j] * B_shared[j][item.get_local_id(0)];

                    item.barrier();

                    if(row_id < M and column_id < K){
                        result_accessor[((item.get_group(1) * item.get_group_range(1) + item.get_local_id(1)) * K) +
                        item.get_group(0) * item.get_local_range(0) + item.get_local_id(0)] = sum;
                    }

                });

            });

            deviceQueue.wait();
        }

        template<int TILE_SIZE, INT VECTOR_SIZE>
        void matmul_reducedOps_sycl(sycl_buffer MatA, sycl_buffer MatB, sycl_buffer result, size_t M, size_t N, size_t K,
                                   queue deviceQueue){

            auto local_range = range<2>(TILE_SIZE, VECTOR_SIZE);
            auto global_range = range<2>(K / (TILE_SIZE * VECTOR_SIZE), M / (TILE_SIZE)) * local_range;

            nd_range<2> launchParams = nd_range<2>(global_range, local_range);


            deviceQueue.submit([&MatA, &MatB, &result, M, N, K, TILE_SIZE, VECTOR_SIZE, launchParams](handler& cgh){

                auto MatA_accessor = MatA->get_access<access::mode::read>(cgh);
                auto MatB_accessor = MatB->get_access<access::mode::read>(cgh);
                auto result_accessor = result->get_access<access::mode::read_write>(cgh);

                accessor<float, 1, access::mode::read_write, access::target::local> A_shared{range<1>(TILE_SIZE * TILE_SIZE),
                        cgh};

                cgh.parallel_for<matmulreducedOps<TILE_SIZE, VECTOR_SIZE>>(launchParams, [MatA_accessor, MatB_accessor, result_accessor, A_shared, M, N, K,
                                                VECTOR_SIZE, TILE_SIZE](nd_item<2> item){

                    float result_vector[TILE_SIZE];

#pragma unroll
                    for (int i = 0; i < TILE_SIZE; ++i) {
                        result_vector[i] = 0.0f;
                    }

                    unsigned int block_x = item.get_group(0); unsigned int thread_x = item.get_local_id(0);
                    unsigned int block_y = item.get_group(1); unsigned int thread_y = item.get_local_id(1);

                    auto a_start_index = K * TILE_SIZE * block_y;
                    auto a_end = a_start_index + K - 1;

                    auto a_step = TILE_SIZE;

                    auto b_start_index = TILE_SIZE * VECTOR_SIZE * block_x;
                    auto b_step = TILE_SIZE * N;

                    for (int a = a_start_index, b = b_start_index; a <= a_end;
                                                a += a_step, b += b_step) {
#pragma unroll
                        for(int i=0; i < TILE_SIZE / VECTOR_SIZE; i++){
                            A_shared[( i * VECTOR_SIZE + thread_y) + TILE_SIZE * thread_x] =
                                    MatA_accessor[a + K * (i * VECTOR_SIZE + thread_y) + thread_x];
                        }

                        item.barrier();

                        auto b_base = b + TILE_SIZE * thread_y + thread_x;

                        for(int i=0; i < TILE_SIZE; i++){

                            float b_value = MatB_accessor[b_base + i*N];
#pragma unroll
                            for(int j=0 ; j<TILE_SIZE; j++){
                                result_vector[j] += A_shared[i * TILE_SIZE + j] * b_value;
                            }

                        }

                        item.barrier();

                    }

                    auto c_ptr = N * TILE_SIZE * block_y + TILE_SIZE * VECTOR_SIZE * block_x;
                    c_ptr += TILE_SIZE * thread_y + thread_x;

#pragma unroll
                    for(int i=0; i < TILE_SIZE; i++){
                        result_accessor[c_ptr] = result_vector[i];
                        c_ptr += N;
                    }

                });
            });
            deviceQueue.wait();
        }

    }
}

