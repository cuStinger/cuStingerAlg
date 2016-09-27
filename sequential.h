#ifndef BC_SEQ
#define BC_SEQ

#include <stack>
#include <queue>
#include <list>
#include <iostream>
#include <fstream>

#include "metis_to_csr.h"

struct thread_data
{
	int start;
	int end;
	csr_graph g;
	bool approx;
	int k;
	int src;
	int dst;
};

extern std::vector< std::vector<int> > d_gpu_v;
extern std::vector< std::vector<unsigned long long> > sigma_gpu_v;
extern std::vector< std::vector<float> > delta_gpu_v;
extern int *source_nodes;

float* bc(csr_graph g, bool print_results, int *d_check, int *sigma_check, float *delta_check);
float* bc_no_parents(csr_graph g, bool print_results, bool streaming, std::vector< std::vector<int> > &d_old, std::vector< std::vector<unsigned long long> > &sigma_old, std::vector< std::vector<float> > &delta_old, int start, int end);
float* bc_no_parents_approx(csr_graph g, int k, int *source_nodes, bool print_results, bool streaming, std::vector< std::vector<int> > &d_old, std::vector< std::vector<unsigned long long> > &sigma_old, std::vector< std::vector<float> > &delta_old, int start, int end);
void bc_update_edge(csr_graph g, bool approx, int k, int *source_nodes, int src, int dst, std::vector< std::vector<int> > &d_old, std::vector< std::vector<unsigned long long> > &sigma_old, std::vector< std::vector<float> > &delta_old, float *bc, std::vector<unsigned long long> &case_stats, std::vector< std::vector<unsigned int> > &nodes_touched, bool debug);
void adjacent_level_insertion(csr_graph g, int u_low, int u_high, std::vector<int> &d_old, std::vector<unsigned long long> &sigma_old, std::vector<float> &delta_old, float *bc, int i, std::vector<unsigned int> &nodes_touched, bool hetero);
void non_adjacent_level_insertion(csr_graph g, bool approx, int k, int u_low, int u_high, std::vector<int> &d_old, std::vector<unsigned long long> &sigma_old, std::vector<float> &delta_old, float *bc, int i, int distance, std::vector<unsigned int> &nodes_touched, bool hetero);
void recompute_root(csr_graph g, std::vector<int> &d_old, std::vector<unsigned long long> &sigma_old, std::vector<float> &delta_old, float *bc, int i);
//void heterogeneous_update(csr_graph g, bool approx, int k, int src, int dst, std::vector<int> &d_old, std::vector<unsigned long long> &sigma_old, std::vector<float> &delta_old, int root);
void *heterogeneous_update(void *arg);
#endif
