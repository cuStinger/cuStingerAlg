cmake_minimum_required(VERSION 3.2)
project( custinger )

set(CUDA_HOST_COMPILER /usr/bin/gcc-4.8)  
FIND_PACKAGE(CUDA REQUIRED)

INCLUDE(FindCUDA)

INCLUDE_DIRECTORIES(/usr/local/cuda/include)
INCLUDE_DIRECTORIES(include)
# INCLUDE_DIRECTORIES(include/static_triangle_counting)

# Adding external libraries
INCLUDE_DIRECTORIES(externals/moderngpu/src/moderngpu)
INCLUDE_DIRECTORIES(externals/cub/cub/)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# set(CMAKE_BUILD_TYPE Debug)
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -lineinfo -G -DTHRUST_DEBUG")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

LIST(APPEND CUDA_NCCC_FLAGS --compiler-options)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
LIST(APPEND CUDA_NVCC_FLAGS "-O3;-DVERBOSE;")
LIST(APPEND CUDA_NVCC_FLAGS "-Xcompiler -Wall -D_FORCE_INLINES; --expt-extended-lambda;-use_fast_math")

# set(cuSTINGER_INCLUDE_DIRS
#   ${CMAKE_SOURCE_DIR}/cuStinger/include
#   CACHE PATH
#   "Directory to cuSTINGER include files")

set(cuSTINGER_LIBRARY_DIR
  ${CMAKE_SOURCE_DIR}/externals/cuStinger/build
  CACHE PATH
  "Directory to the cuSTINGER library")

link_directories(${cuSTINGER_LIBRARY_DIR})

set(cuSTINGER_INCLUDE_DIR
  ${CMAKE_SOURCE_DIR}/externals/cuStinger/include
  CACHE PATH
  "Directory to the cuSTINGER include files")

INCLUDE_DIRECTORIES(${cuSTINGER_INCLUDE_DIR})

#Useful for building a library of algorithms
cuda_add_library(custingerAlgs
          src/streaming_triangle_counting/cct_upd.cu
          src/static_triangle_counting/cct.cu
          src/static_breadth_first_search/bfs_top_down.cu
          src/static_connected_components/cc_baseline.cu
          src/static_connected_components/cc_concurrent.cu
          src/static_connected_components/cc_load_balanced.cu
          # src/streaming_triangle_counting/cct_upd.cu
          src/static_page_rank/pr.cu              
          src/load_balance.cu
          src/static_betweenness_centrality/bc.cu
          src/static_betweenness_centrality/bc_tree.cu
        )


cuda_add_executable(cuAlgCCT 
          tests/cct.cpp
          tests/cct_csr.cu
           )

cuda_add_executable(cuAlgStCCT 
          tests/cct_upd.cpp
           )

cuda_add_executable(static_bc_tester
          tests/static_bc.cu
          src/static_betweenness_centrality/bc.cu
          src/static_betweenness_centrality/bc_tree.cu
          src/load_balance.cu
          )

cuda_add_executable(cus-alg-tester 
          tests/alg-tester.cu
          )


TARGET_LINK_LIBRARIES(static_bc_tester custingerAlgs custinger  /usr/local/cuda/lib64/libcudart.so)

TARGET_LINK_LIBRARIES(cuAlgCCT custingerAlgs custinger  /usr/local/cuda/lib64/libcudart.so)
TARGET_LINK_LIBRARIES(cuAlgStCCT custingerAlgs custinger  /usr/local/cuda/lib64/libcudart.so)

TARGET_LINK_LIBRARIES(cus-alg-tester custingerAlgs custinger /usr/local/cuda/lib64/libcudart.so)
