cmake_minimum_required(VERSION 3.24)
project(kineticsim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(KS_MODULES kineticsim_cuda kineticsim_naive)
pybind11_add_module(kineticsim_cuda  src/kineticsim_kernels.cu)
pybind11_add_module(kineticsim_naive src/naive_kernels.cu)

foreach(t ${KS_MODULES})
  target_include_directories(${t} PRIVATE ${CMAKE_SOURCE_DIR}/src)
  set_target_properties(${t} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python)
  target_compile_options(${t} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;-lineinfo;--expt-relaxed-constexpr>)
endforeach()

message(STATUS "KineticSim: building for sm_${CMAKE_CUDA_ARCHITECTURES}")
