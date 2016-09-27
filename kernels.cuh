#ifndef BC_KERNELS
#define BC_KERNELS

#include <cstdio>

//Kernels
__global__ void bc_gpu_naive(float *bc, int *R, int *C, int n, int m); //Simple baseline for testing, etc. 
__global__ void bc_gpu_opt(float *bc, const int *__restrict__ F, const int *__restrict__ C, const int n, const int m, int *__restrict__ d, unsigned long long *__restrict__ sigma, float *__restrict__ delta, size_t pitch, size_t pitch_sigma, const int start, const int end);
__global__ void bc_gpu_opt_approx(float *bc, const int *__restrict__ F, const int *__restrict__ C, const int n, const int m, int *__restrict__ d, unsigned long long *__restrict__ sigma, float *__restrict__ delta, size_t pitch, size_t pitch_sigma, const int *__restrict__ sources, const int K, const int start, const int end);
template<bool approx>
extern __global__ void bc_gpu_update_edge(float *bc, const int *__restrict__ F, const int *__restrict__ C, const int n, const int m, int *__restrict__ d, unsigned long long *__restrict__ sigma, float *__restrict__ delta, size_t pitch, size_t pitch_sigma, const int *__restrict__ sources, const int start, const int end, const int src, const int dst);

template<bool approx>
__global__ void bc_gpu_update_edge(float *bc, const int *__restrict__ F, const int *__restrict__ C, const int n, const int m, int *__restrict__ d, unsigned long long *__restrict__ sigma, float *__restrict__ delta, size_t pitch, size_t pitch_sigma, const int *__restrict__ sources, const int start, const int end, const int src, const int dst);

#include "streaming_kernels.cu"
#endif
