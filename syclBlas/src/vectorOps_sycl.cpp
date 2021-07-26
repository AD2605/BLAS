#include "../include/vectorOps_sycl.h"
#include "../include/defines.h"

#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

namespace blas1{

    namespace sycl_blas{

        void vectorAdd(sycl_buffer v1, sycl_buffer v2, sycl_buffer out, size_t N, int num_threads, queue deviceQueue){

            auto local_range = range<1>(num_threads);
            auto global_range = range<1>(N / num_threads + 1) * local_range;
            auto launch_params = nd_range<1>(global_range, local_range);

            deviceQueue.submit([&v1, &v2, &out, N, launch_params](handler& cgh){

                auto v1_accessor = v1->get_access<access::mode::read>(cgh);
                auto v2_accessor = v2->get_access<access::mode::read>(cgh);
                auto result_accessor = out->get_access<access::mode::write>(cgh);

                cgh.parallel_for<vectorAddKernel>(launch_params, [v1_accessor, v2_accessor, result_accessor, N](nd_item<1> ndItem){

                    unsigned int idx = ndItem.get_global_id(0);

                    if(idx < N) {
                        result_accessor[idx] = v1_accessor[idx] + v2_accessor[idx];
                    }

                });

            }).wait();

        }

        void vectorAdd_(sycl_buffer v1, sycl_buffer v2, size_t N, int num_threads, queue deviceQueue){

            auto local_range = range<1>(num_threads);
            auto global_range = range<1>(N / num_threads + 1) * local_range;
            auto launch_params = nd_range<1>(global_range, local_range);

            deviceQueue.submit([&v1, &v2, N, launch_params](handler& cgh){

                auto v1_accessor = v1->get_access<access::mode::read_write>(cgh);
                auto v2_accessor = v2->get_access<access::mode::read>(cgh);

                cgh.parallel_for<vectorAddKernel_>(launch_params, [v1_accessor, v2_accessor, N](nd_item<1> ndItem){

                    unsigned int idx = ndItem.get_global_id(0);

                    if (idx < N){
                        v1_accessor[idx] +=  v2_accessor[idx];
                    }

                });

            }).wait();

        }

        void scalarMultiplication(sycl_buffer v1, sycl_buffer out, float scalar, size_t N, int num_threads, queue deviceQueue){

            auto local_range = range<1>(num_threads);
            auto global_range = range<1>(N / num_threads + 1) * local_range;
            auto launch_params = nd_range<1>(global_range, local_range);

            deviceQueue.submit([&v1, &out, scalar, N, launch_params](handler& cgh){

               auto v1_accessor = v1->get_access<access::mode::read>(cgh);
               auto out_accessor = out->get_access<access::mode::write>(cgh);

               cgh.parallel_for<scalarMulitplicationKernel>(launch_params, [v1_accessor, out_accessor, scalar, N](nd_item<1> ndItem){

                   auto idx = ndItem.get_global_id(0);

                   if(idx < N){

                       out_accessor[idx] = scalar * v1_accessor[idx];
                   }

               });

            }).wait();

        }

        void scalarMultiplication_(sycl_buffer v1, float scalar, size_t N, int  num_threads, queue deviceQueue){

            auto local_range = range<1>(num_threads);
            auto global_range = range<1>(N / num_threads + 1) * local_range;
            auto launch_params = nd_range<1>(global_range, local_range);

            deviceQueue.submit([&v1, scalar, N, launch_params](handler& cgh){

                auto v1_accessor = v1->get_access<access::mode::read_write>(cgh);

                cgh.parallel_for<scalarMultiplicationKernel_>(launch_params, [v1_accessor, N, scalar](nd_item<1> ndItem){

                    unsigned int idx = ndItem.get_global_id(0);

                    if (idx < N){

                        v1_accessor[idx] *= scalar;

                    }

                });

            }).wait();

        }

    }

}