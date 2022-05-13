cmake_minimum_required(VERSION 3.20)
project(sig_kernels)


set(CMAKE_CXX_STANDARD 14)
enable_language(CUDA)

if (EXISTS "${CMAKE_SOURCE_DIR}/venv")
    set(Python_ROOT_DIR  "${CMAKE_SOURCE_DIR}/venv/bin")
endif ()

set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Thrust CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

message(STATUS "${Python_EXECUTABLE}")

add_library(sig_kernels_cuda STATIC
        src/kernel_compute_cuda.h
        src/kernel_compute_cuda.cu
        )
target_link_libraries(sig_kernels_cuda PUBLIC Thrust::Thrust)


add_library(sig_kernels STATIC
        src/kernel_backends.h
        src/batch_varpar.cpp
        src/batch_varpar_naive.cpp
        src/kernel_compute.h
        src/random_paths.cpp
        src/random_paths.h
        src/dot_product.h
        src/dot_product.cpp
        src/antidiagonal_matrix.cpp
        src/antidiagonal_matrix.h
        src/implementation_types.h
        )
target_link_libraries(sig_kernels PUBLIC pybind11::pybind11 Thrust::Thrust OpenMP::OpenMP_CXX)
set_target_properties(sig_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(sig_kernels PUBLIC -O3 -fno-omit-frame-pointer)

pybind11_add_module(sigker_backends MODULE
        src/py_kernels.cpp
        )
target_link_libraries(sigker_backends PRIVATE sig_kernels)


set_target_properties(sigker_backends PROPERTIES
        LIBRARY_OUTPUT_NAME sigker_backends
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        )



add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE sig_kernels pybind11::embed)


find_package(benchmark REQUIRED)


add_executable(BM_kernels
        src/BM_kernel_compute.cpp
        src/BM_varpar_kernel.cpp
        src/BM_cuda_kernel.cpp
        src/bench_args.cpp
        src/bench_args.h)
target_link_libraries(BM_kernels PRIVATE
        pybind11::embed
        benchmark::benchmark
        benchmark::benchmark_main
        PUBLIC
        sig_kernels)

target_compile_options(BM_kernels PUBLIC -O3 -fno-omit-frame-pointer)
