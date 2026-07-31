cmake_minimum_required(VERSION 3.18)
project(gpu_wavelets LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)  # T4 on Colab; override for A100 (80), etc.
endif()

# -v prints per-kernel register/smem usage for the performance study (item 11).
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)

include_directories(src third_party)

# CLI driver (item 13); also exercises image I/O (item 4) and the fp16 path (item 10).
add_executable(wavelet src/main.cu)

# Performance study harness (item 11) — prints markdown tables for the report.
add_executable(bench bench/bench.cu)

# Tests (item 14) — small deterministic binaries returning nonzero on failure.
enable_testing()
foreach(t test_1d test_2d test_access test_3d test_fp16 test_io)
  add_executable(${t} tests/${t}.cu)
  add_test(NAME ${t} COMMAND ${t})
endforeach()
