cmake_minimum_required(VERSION 3.17)
project(MatrixMultiplication CUDA CXX)

function(create_sycl_lib)

        set(options NO_TEST)
        set(one_value_args TARGET)
        set(multi_value_args SOURCES)
        cmake_parse_arguments(SDK_ADD_SAMPLE
                "${options}"
                "${one_value_args}"
                "${multi_value_args}"
                ${ARGN}
                )

        add_library(${SDK_ADD_SAMPLE_TARGET} SHARED ${SDK_ADD_SAMPLE_SOURCES})

        add_sycl_to_target(TARGET ${SDK_ADD_SAMPLE_TARGET}
                SOURCES ${SDK_ADD_SAMPLE_SOURCES})
        #if(NOT SDK_ADD_SAMPLE_NO_TEST)
        #    add_test(NAME ${SDK_ADD_SAMPLE_TARGET} COMMAND ${SDK_ADD_SAMPLE_TARGET})
        #endif()
        #install(TARGETS ${SDK_ADD_SAMPLE_TARGET} RUNTIME DESTINATION bin)
endfunction(create_sycl_lib)


function(add_sycl_to_target)
        set(options)
        set(one_value_args
                TARGET
                )
        set(multi_value_args
                SOURCES
                )
        cmake_parse_arguments(SDK_ADD_SYCL
                "${options}"
                "${one_value_args}"
                "${multi_value_args}"
                ${ARGN}
                )
        if ("${SDK_ADD_SYCL_SOURCES}" STREQUAL "")
                message(WARNING "No source files provided to add_sycl_to_target. "
                        "SYCL integration headers may not be generated.")
        endif()
        set_target_properties(${SDK_ADD_SAMPLE_TARGET} PROPERTIES SOVERSION 1)
        set_target_properties(${SDK_ADD_SYCL_TARGET} PROPERTIES LINKER_LANGUAGE CXX)

        # If the CXX compiler is set to compute++ enable the driver.
        get_filename_component(cmakeCxxCompilerFileName "${CMAKE_CXX_COMPILER}" NAME)
        if("${cmakeCxxCompilerFileName}" STREQUAL "compute++")
                if(MSVC)
                        message(FATAL_ERROR "The compiler driver is not supported by this system,
                           revert the CXX compiler to your default host compiler.")
                endif()

                get_target_property(includeAfter ${SDK_ADD_SYCL_TARGET} COMPUTECPP_INCLUDE_AFTER)
                if(includeAfter)
                        list(APPEND COMPUTECPP_USER_FLAGS -fsycl-ih-last)
                endif()
                list(INSERT COMPUTECPP_DEVICE_COMPILER_FLAGS 0 -sycl-driver)
                # Prepend COMPUTECPP_DEVICE_COMPILER_FLAGS and append COMPUTECPP_USER_FLAGS
                foreach(prop COMPILE_OPTIONS INTERFACE_COMPILE_OPTIONS)
                        get_target_property(target_compile_options ${SDK_ADD_SYCL_TARGET} ${prop})
                        if(NOT target_compile_options)
                                set(target_compile_options "")
                        endif()
                        set_property(
                                TARGET ${SDK_ADD_SYCL_TARGET}
                                PROPERTY ${prop}
                                ${COMPUTECPP_DEVICE_COMPILER_FLAGS}
                                ${target_compile_options}
                                ${COMPUTECPP_USER_FLAGS}
                        )
                endforeach()
        else()
                set(fileCounter 0)
                list(INSERT COMPUTECPP_DEVICE_COMPILER_FLAGS 0 -sycl)
                # Add custom target to run compute++ and generate the integration header
                foreach(sourceFile ${SDK_ADD_SYCL_SOURCES})
                        if(NOT IS_ABSOLUTE ${sourceFile})
                                set(sourceFile "${CMAKE_CURRENT_SOURCE_DIR}/${sourceFile}")
                        endif()
                        __build_ir(
                                TARGET     ${SDK_ADD_SYCL_TARGET}
                                SOURCE     ${sourceFile}
                                COUNTER    ${fileCounter}
                        )
                        MATH(EXPR fileCounter "${fileCounter} + 1")
                endforeach()
        endif()

        set_property(TARGET ${SDK_ADD_SYCL_TARGET}
                APPEND PROPERTY LINK_LIBRARIES ComputeCpp::ComputeCpp)
        set_property(TARGET ${SDK_ADD_SYCL_TARGET}
                APPEND PROPERTY INTERFACE_LINK_LIBRARIES ComputeCpp::ComputeCpp)
        add_library(${SDK_ADD_SYCL_TARGET} SHARED ${SDK_ADD_SYCL_SOURCES})
        #target_compile_definitions(${SDK_ADD_SYCL_TARGET} INTERFACE
        #        SYCL_LANGUAGE_VERSION=${SYCL_LANGUAGE_VERSION})
endfunction(add_sycl_to_target)

set(CMAKE_CUDA_STANDARD 14)

find_package(CUDA REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES   75)

set(ComputeCpp_DIR /home/atharva/ComputeCPP/computeCPP/)
set(CMAKE_MODULE_PATH /home/atharva/computecpp-sdk/cmake/Modules/)

find_package(ComputeCpp)

include_directories($(COMPUTECPP_INCLUDE_DIRECTORY))

file(GLOB_RECURSE cuda_src CONFIGURE_DEPENDS cudaBlas/src/*.cu)
file(GLOB_RECURSE cuda_include CONFIGURE_DEPENDS cudaBlas/include/*.cuh)

file(GLOB_RECURSE sycl_src CONFIGURE_DEPENDS syclBlas/src/*.cpp)
file(GLOB_RECURSE sycl_include CONFIGURE_DEPENDS syclBlas/include/*.h)

set_source_files_properties(${cuda_src} PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${cuda_include} PROPERTIES LANGUAGE CUDA)

add_library(cudaBlasLib STATIC ${cuda_src} ${cuda_include})
set_target_properties(cudaBlasLib PROPERTIES SOVERSION 1)


create_sycl_lib(TARGET syclBlasLib SOURCES ${sycl_src} ${sycl_include})