#ifndef _CU_STATIC_BC_STREAMING_INCLUDE_HPP_
#define _CU_STATIC_BC_STREAMING_INCLUDE_HPP_


#include <stdio.h>
#include <inttypes.h>

#include "utils.hpp"
#include "update.hpp"
#include "cuStinger.hpp"

// typedef int32_t triangle_t;


// void callDeviceAllTriangles(cuStinger& custing,
//     triangle_t * const __restrict__ outPutTriangles, const int threads_per_block,
//     const int number_blocks, const int shifter, const int thread_blocks, const int blockdim);


typedef struct
{
	bool streaming;
	bool approx;
	// number of vertices used. If approx, set here via CLI.
	// otherwise defaults to all vertices
	int numRoots;
	bool verbose;  // print debug info
	int edgesToAdd;  // edges to add
	char *infile;
} program_options;

// void bc_static(cuStinger& custing, void* func_meta_data);
// void bc_static(cuStinger& custing, float *bc, int numRoots,
// 	int max_threads_per_block, int number_of_SMs);


#endif
