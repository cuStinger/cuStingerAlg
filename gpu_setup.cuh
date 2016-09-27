#ifndef GPU_SETUP
#define GPU_SETUP

#include <iostream>
#include <vector>
#include <cstdio>
#include "util.cuh"
#include "metis_to_csr.h"
#include "kernels.cuh"

float single_gpu_full_computation(csr_graph g, program_options op, int number_of_SMs, int max_threads_per_block, float *bc_gpu, int *source_nodes, bool recomp);
float multi_gpu_full_computation(csr_graph g, program_options op, float *bc_gpu, int *source_nodes);
void single_gpu_streaming_computation(csr_graph &g, program_options op, float *bc_gpu, int *source_nodes, int max_threads_per_block, int number_of_SMs, std::set< std::pair<int,int> > removals_gpu, float &time_gpu_update, float &time_min_update_gpu, float &time_max_update_gpu, std::pair<int,int> &min_update_edge_gpu, std::pair<int,int> &max_update_edge_gpu, std::vector< std::vector<int> > &d_gpu_v, std::vector< std::vector<unsigned long long> > &sigma_gpu_v, std::vector< std::vector<float> > &delta_gpu_v, bool opt, int node);
void single_gpu_streaming_computation_SOA(csr_graph &g, program_options op, float *bc_gpu, int *source_nodes, int max_threads_per_block, int number_of_SMs, std::set< std::pair<int,int> > removals_gpu, float &time_gpu_update, float &time_min_update_gpu, float &time_max_update_gpu);
void single_gpu_streaming_computation_AOS(csr_graph &g, program_options op, float *bc_gpu, int *source_nodes, int max_threads_per_block, int number_of_SMs, std::set< std::pair<int,int> > removals_gpu, float &time_gpu_update, float &time_min_update_gpu, float &time_max_update_gpu);
void heterogeneous_streaming_computation(csr_graph &g, program_options op, float *bc_gpu, int *source_nodes, int max_threads_per_block, int number_of_SMs, std::set< std::pair<int,int> > removals_gpu, float &time_heterogeneous_update, float &time_min_update_hetero, float &time_max_update_hetero, std::pair<int,int> &min_update_edge_hetero, std::pair<int,int> &max_update_edge_hetero, float &time_for_accumulation, float &time_CPU);

#endif
