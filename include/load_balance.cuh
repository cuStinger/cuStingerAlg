#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>

#include <typeinfo>
#include "utils.hpp"
#include "update.hpp"
#include "cuStinger.hpp"
#include "algs.cuh"


#include "kernel_scan.hxx"
// #include "search.hxx"
// #include "cta_search.hxx"

// ModernGPU
#include <kernel_sortedsearch.hxx>
#include <transform.hxx>
//CUB headers
#include <cub.cuh>

#include <block/block_radix_sort.cuh>



using namespace mgpu;
using namespace cub;


namespace cuStingerAlgs{

class cusLoadBalance{
public:
	cusLoadBalance(length_t maxArrayLen_):
	context(false)
	{
		internalAlloc=false;
		maxArrayLen      = maxArrayLen_;
		devPartitionsPoints = (length_t*)allocDeviceArray(initNumPartition+1,sizeof(length_t));		
		devlbArray = (length_t*)allocDeviceArray(maxArrayLen+1,sizeof(length_t));
		devPrefixArray = (length_t*)allocDeviceArray(maxArrayLen+1,sizeof(length_t));
		// Creating needles
		devNeedles = (length_t*)allocDeviceArray(initNumPartition,sizeof(length_t));
	}

	cusLoadBalance(cuStinger &custing, bool queueZeros=true, bool queueRandomly=false):
	context(false)
	{
		internalAlloc=true;
		maxArrayLen      = custing.nv;
		currArray = (vertexId_t*)allocDeviceArray(maxArrayLen+1,sizeof(vertexId_t));
		currArrayLen = maxArrayLen;

		devPartitionsPoints = (length_t*)allocDeviceArray(initNumPartition+1,sizeof(length_t));		
		devlbArray = (length_t*)allocDeviceArray(maxArrayLen+1,sizeof(length_t));
		devPrefixArray = (length_t*)allocDeviceArray(maxArrayLen+1,sizeof(length_t));
		// Creating needles
		devNeedles = (length_t*)allocDeviceArray(initNumPartition,sizeof(length_t));

		queueVertices(custing,queueZeros,queueRandomly);
		LoadBalanceArray(custing);
	}

	~cusLoadBalance(){
		if(internalAlloc)
			freeDeviceArray(currArray);
		
		freeDeviceArray(devPartitionsPoints);
		freeDeviceArray(devPrefixArray);
		freeDeviceArray(devlbArray);
		freeDeviceArray(devNeedles);
	}

	void LoadBalanceArray(cuStinger &custing){

		estimateWorkPerVertex(currArray,currArrayLen,custing,devlbArray);

		scan(devlbArray, currArrayLen+1, devPrefixArray, context);

		length_t expectedWork;
		copyArrayDeviceToHost(devPrefixArray+currArrayLen,&expectedWork,1 , sizeof(length_t));
		float avgWork = float(expectedWork)/float(currArrayLen);

		if(avgWork < 4)
			foundNumPartition = 200;
		else if(avgWork < 7)
			foundNumPartition = 500;
		else
			foundNumPartition = 1000;

		if(foundNumPartition>currArrayLen){
			foundNumPartition=currArrayLen;
		}

		createNeedles(devNeedles,foundNumPartition,devPrefixArray, currArrayLen+1);

		sorted_search<bounds_lower>(devNeedles, foundNumPartition, 
			devPrefixArray, currArrayLen-1, devPartitionsPoints, less_t<int>(), context);
	}

private:
	void createNeedles(length_t* dNeedles,length_t numNeedles, length_t* dPrefixArray, length_t prefixSize);
	void estimateWorkPerVertex(vertexId_t* verArray, length_t len,cuStinger& custing,void* metadata);
	void queueVertices(cuStinger &custing, bool queueZeros,bool queueRandomly);

private:
	length_t* devlbArray;
	length_t* devPrefixArray;
	length_t  maxArrayLen;
	// length_t  initNumPartition;
	length_t* nedevedles;
	length_t* devNeedles;

	standard_context_t context;
	bool internalAlloc;
public:
	vertexId_t* currArray;	
	length_t    currArrayLen; 
	length_t*   devPartitionsPoints;
	length_t    foundNumPartition; // (foundPartitionPoints <= initPartitionPoints)    ALWAYS TRUE!!

	// bool vertexBalancing;
	// cusSubKernelVertex* dLoadBalanceVertex;
	// cusSubKernelEdge*   dLoadBalanceEdge; // Currently not supported
	const int initNumPartition = 1000;
};


} // cuStingerAlgs namespace