cmake_minimum_required(VERSION 3.18)
project(gpu_wavelets LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)

include_directories(src third_party)

add_executable(wavelet src/main.cu)

add_executable(bench bench/bench.cu)

enable_testing()
foreach(t test_1d test_2d test_access test_3d test_fp16 test_io)
  add_executable(${t} tests/${t}.cu)
  add_test(NAME ${t} COMMAND ${t})
endforeach()
