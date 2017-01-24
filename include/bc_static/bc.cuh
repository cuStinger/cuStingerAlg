#pragma once

#include "algs.cuh"
#include <stdio.h>

// Betweenness Centrality

namespace cuStingerAlgs {

typedef Queue<int> sigmas;


class bcStaticData {
public:
	vertexQueue queue;
	vertexId_t* level;
	vertexId_t currLevel;
	vertexId_t root;
	
	Vector<int> *d;  // depth
	Vector<long> *sigma;
	Vector<float> *delta;

	length_t nv;
};


class StaticBreadthFirstSearch:public StaticAlgorithm {
public:	

	void Init(cuStinger& custing);
	void Reset();
	void Run(cuStinger& custing);
	void Release();

	void Run2(cuStinger& custing);

	void SyncHostWithDevice()
	{
		copyArrayDeviceToHost(devicebcStaticData,&hostbcStaticData,1, sizeof(bcStaticData));
	}
	void SyncDeviceWithHost()
	{
		copyArrayHostToDevice(&hostbcStaticData,devicebcStaticData,1, sizeof(bcStaticData));
	}
	
	length_t getLevels(){return hostbcStaticData.currLevel;}
	length_t getElementsFound(){return hostbcStaticData.queue.getQueueEnd();}

	void setInputParameters(vertexId_t root);

	// User is responsible for de-allocating memory.
	vertexId_t* getLevelArrayHost()
	{
		vertexId_t* hostArr = (vertexId_t*)allocHostArray(hostbcStaticData.nv, sizeof(vertexId_t));
		copyArrayDeviceToHost(hostbcStaticData.level, hostArr, hostbcStaticData.nv, sizeof(vertexId_t) );
		return hostArr;
	}

	// User sends pre-allocated array.
	void getLevelArrayForHost(vertexId_t* hostArr)
	{
		copyArrayDeviceToHost(hostbcStaticData.level, hostArr, hostbcStaticData.nv, sizeof(vertexId_t) );
	}

	bcStaticData hostbcStaticData, *devicebcStaticData;
};


class bcOperator:public StaticAlgorithm {
public:

	static __device__ __forceinline__ void bcExpandFrontier(cuStinger* custing,
		vertexId_t src, vertexId_t dst, void* metadata)
	{
		bcStaticData* bcd = (bcStaticData*) metadata;
		vertexId_t nextLevel = bcd->currLevel + 1;

		vertexId_t prev = atomicCAS(bcd->level+dst, INT32_MAX, nextLevel);
		if (prev == INT32_MAX) {
			bcd->queue.enqueue(dst);
		}
		if (bcd->level[dst] == nextLevel) {
			bcd->sigma[dst] += bcd->sigma[src];
		}

	}

	static __device__ __forceinline__ void setLevelInfinity(cuStinger* custing,
		vertexId_t src, void* metadata)
	{
		bcStaticData* bcd = (bcStaticData*) metadata;
		bcd->level[src] = INT32_MAX;
	}

}; // bcOperator


} //Namespace