#pragma once

#include "algs.cuh"

// Betweenness Centrality

namespace cuStingerAlgs {

typedef Queue<int> sigmas;


class bcStaticData {
public:
	vertexQueue queue;
	vertexId_t* level;
	vertexId_t currLevel;
	vertexId_t root;
	
	int *d;  // depth
	long *sigma;
	float *delta;

	length_t nv;
};


class StaticBC:public StaticAlgorithm {
public:	

	void Init(cuStinger& custing);
	void Reset();
	void Run(cuStinger& custing);
	void Release();

	void DependencyAccumulation(cuStinger& custing, float *bc);

	void SyncHostWithDevice()
	{
		copyArrayDeviceToHost(deviceBcStaticData, &hostBcStaticData, 1, sizeof(bcStaticData));
	}
	void SyncDeviceWithHost()
	{
		copyArrayHostToDevice(&hostBcStaticData, deviceBcStaticData, 1, sizeof(bcStaticData));
	}
	
	length_t getLevels(){return hostBcStaticData.currLevel;}
	length_t getElementsFound(){return hostBcStaticData.queue.getQueueEnd();}

	void setInputParameters(vertexId_t root);

	// User is responsible for de-allocating memory.
	vertexId_t* getLevelArrayHost()
	{
		vertexId_t* hostArr = (vertexId_t*)allocHostArray(hostBcStaticData.nv, sizeof(vertexId_t));
		copyArrayDeviceToHost(hostBcStaticData.level, hostArr, hostBcStaticData.nv, sizeof(vertexId_t) );
		return hostArr;
	}

	// User sends pre-allocated array.
	void getLevelArrayForHost(vertexId_t* hostArr)
	{
		copyArrayDeviceToHost(hostBcStaticData.level, hostArr, hostBcStaticData.nv, sizeof(vertexId_t) );
	}

	bcStaticData hostBcStaticData, *deviceBcStaticData;
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


__global__ void hostDependencyAccumulation(cuStinger* custing, bcStaticData *deviceBcStaticData, vertexId_t *queue, float *bc);


__device__ void deviceDependencyAccumulation(cuStinger* custing, bcStaticData *deviceBcStaticData, vertexId_t *queue, float *bc);

} //Namespace