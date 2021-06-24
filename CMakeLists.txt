cmake_minimum_required(VERSION 3.15)
project(MatrixMultiplication CUDA)

function(add_sample)
        set(options NO_TEST)
        set(one_value_args TARGET)
        set(multi_value_args SOURCES)
        cmake_parse_arguments(SDK_ADD_SAMPLE
                "${options}"
                "${one_value_args}"
                "${multi_value_args}"
                ${ARGN}
                )

        add_executable(${SDK_ADD_SAMPLE_TARGET} ${SDK_ADD_SAMPLE_SOURCES})
        add_sycl_to_target(TARGET ${SDK_ADD_SAMPLE_TARGET}
                SOURCES ${SDK_ADD_SAMPLE_SOURCES})
        if(NOT SDK_ADD_SAMPLE_NO_TEST)
                add_test(NAME ${SDK_ADD_SAMPLE_TARGET} COMMAND ${SDK_ADD_SAMPLE_TARGET})
        endif()
        install(TARGETS ${SDK_ADD_SAMPLE_TARGET} RUNTIME DESTINATION bin)
endfunction(add_sample)

set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_MODULE_PATH /home/atharva/computecpp-sdk/cmake/Modules/)
find_package(ComputeCpp)

include_directories($(COMPUTECPP_INCLUDE_DIRECTORY))

add_sample(TARGET matmul_CPP SOURCES syclBlas/matrixMultiplication_sycl.cpp)

add_executable(MatrixMultiplication main.cu matrixMultiplication.cu matrixMultiplication.cuh MatrixOps.cu MatrixOps.cuh
        vectorOps.cu vectorOps.cuh defines.cuh)
set_property(TARGET MatrixMultiplication PROPERTY CUDA_ARCHITECTURES 75)


set_target_properties(
        MatrixMultiplication
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        )