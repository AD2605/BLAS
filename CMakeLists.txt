cmake_minimum_required(VERSION 3.15)
project(MatrixMultiplication CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(MatrixMultiplication main.cu matrixMultiplication.cu matrixMultiplication.cuh)
set_property(TARGET MatrixMultiplication PROPERTY CUDA_ARCHITECTURES 75)


set_target_properties(
        MatrixMultiplication
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        )