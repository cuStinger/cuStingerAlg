#ifndef BC_UTIL
#define BC_UTIL

#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cstdio>
#include <nvml.h>
#include <pthread.h>

#include "metis_to_csr.h"

#define nreps 1

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#ifndef checkCudaErrors
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(cudaError_t err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        std::cerr << "CUDA Error = " << err << ": " << cudaGetErrorString(err) << " from file " << file  << ", line " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}
#endif

#ifndef checkNVMLErrors
#define checkNVMLErrors(err) __checkNVMLErrors (err, __FILE__, __LINE__)

inline void __checkNVMLErrors(nvmlReturn_t result, const char *file, const int line)
{
    if(result!=NVML_SUCCESS)
    {
	std::cerr << "NVML Error = " << result << ": " << nvmlErrorString(result) << " from file " << file << ", line " << line << std::endl;
	exit(EXIT_FAILURE);
    }
}
#endif
//Interface for handling program options
class program_options
{
public:
	program_options() : printBCscores(false), verify_results(false), verify_all(false), infile(NULL), approximate(false), k(-1), debug(false), device(-1), streaming(false), insertions(-1), multi(false), experiment(false), result_file(NULL), nvml(false), power_file(NULL), node(-1), no_cpu(false), export_edge_graph(false)
	{
		
	}

	bool printBCscores;
	bool verify_results; //Verify results on the CPU?
	bool verify_all; //Verify all data structures
	char *infile;
	bool approximate; //Use all source nodes by default
	int k;
	bool debug;
	int device; //Which GPU to select?
	bool streaming; //Static or dynamic BC?
	int insertions; //Number of streaming insertions
	bool multi;
	bool experiment; //Running an experiment?
	char *result_file;
	bool nvml; //Use the NVML library to measure power?
	char *power_file;
	int node;
	bool no_cpu; //Don't bother streaming via the CPU? Default is to stream
	bool export_edge_graph;
};

void parse_arguments(int argc, char **argv, program_options &op);

//Timing routines
void start_clock(cudaEvent_t &start, cudaEvent_t &end);
float end_clock(cudaEvent_t &start, cudaEvent_t &end);

//Verification/debug routines
void verify(float *expected, float *actual, csr_graph g);
void verify_all(float *bc_expected, float *bc_actual, int *d_expected, int *d_actual, int *sigma_expected, int *sigma_actual, float *delta_expected, float *delta_actual, csr_graph g);

template <typename T>
void stream_debug_print(csr_graph g, T full, T stream, char *desc)
{
	for(int i=0; i<g.n; i++)
	{
		if(abs(full[i]-stream[i]) > 0)
		{
			std::cout << "Full computation: " << std::endl;
			std::cout << desc << "[" << i << "] = " << full[i] << std::endl;
			std::cout << "Streaming: " << std::endl;
			std::cout << desc << "[" << i << "] = " << stream[i] << std::endl; 
		}
	}
}

template <typename T>
void stream_debug_print_all(csr_graph g, T full, T stream, char *desc)
{
	for(int i=0; i<g.n; i++)
	{
		std::cout << "Full computation: " << std::endl;
		std::cout << desc << "[" << i << "] = " << full[i] << std::endl;
		std::cout << "Streaming: " << std::endl;
		std::cout << desc << "[" << i << "] = " << stream[i] << std::endl; 
	}
}

template<template <typename, typename> class Container, class V, class A>
std::ostream& operator<<(std::ostream& out, Container<V, A> const &v)
{
	out << '[';
	if(!v.empty())
	{
		for(typename Container<V, A>::const_iterator i = v.begin(); ;)
		{
			out << *i;
			if(++i == v.end())
			{
				break;
			}
			out << ",";
		}
	}
	out << ']';
	return out;
}

//GPU information routines
void choose_device(int &max_threads_per_block, int &number_of_SMs, int &choice);
void print_devices();
void print_intermediates(int *d, int *d_gpu, int *sigma, int *sigma_gpu, float *delta, float *delta_gpu, float *bc, float *bc_gpu, csr_graph g);

//Streaming routines

//This function modifies g so it needs to be passed by reference
void remove_edges(csr_graph &g, std::set< std::pair<int,int> > &removals, program_options op); 

//Pthread function for power sampling
void *power_sample(void *period);
extern bool *psample;
//Note: period is the sampling period in milliseconds
void start_power_sample(program_options op, pthread_t &thread, long period);
float end_power_sample(program_options op, pthread_t &thread);
#endif
