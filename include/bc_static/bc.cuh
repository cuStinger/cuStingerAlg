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
	unsigned long long *sigma;
	float *delta;

	int *offsets;  // length of each frontier. May have up to (custing.nv) frontiers

	length_t nv;
};


class StaticBC:public StaticAlgorithm {
public:	

	void Init(cuStinger& custing);
	void Reset();
	void Run(cuStinger& custing);
	void Release();

	void SyncHostWithDevice()
	{
		copyArrayDeviceToHost(deviceBcStaticData, &hostBcStaticData, 1, sizeof(bcStaticData));
	}
	void SyncDeviceWithHost()
	{
		copyArrayHostToDevice(&hostBcStaticData, deviceBcStaticData, 1, sizeof(bcStaticData));
	}

	void RunBfsTraversal(cuStinger& custing);
	void DependencyAccumulation(cuStinger& custing);
	
	length_t getLevel() { return hostBcStaticData.currLevel; }

	// Must pass in a root node vertex id, and a pointer to bc values (of length custing.nv)
	void setInputParameters(vertexId_t root, float *bc_array);

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

private:
	float *bc;  // the actual bc values array on the host
	// a float array which will contain a copy of the device delta array during dependency accumulation
	float *host_deltas;
};


class bcOperator:public StaticAlgorithm {
public:

	static __device__ __forceinline__ void bcExpandFrontier(cuStinger* custing,
		vertexId_t src, vertexId_t dst, void* metadata)
	{
		bcStaticData* bcd = (bcStaticData*) metadata;
		vertexId_t nextLevel = bcd->currLevel + 1;

		vertexId_t v = src;
		vertexId_t w = dst;

		vertexId_t prev = atomicCAS(bcd->level + w, INT32_MAX, nextLevel);
		if (prev == INT32_MAX) {
			bcd->queue.enqueue(w);
		}
		if (bcd->level[w] == nextLevel) {
			atomicAdd(bcd->sigma + w, bcd->sigma[v]);
		}

	}

	// Use macro to clear values in arrays to 0
	static __device__ __forceinline__ void clearArrays(cuStinger* custing,
		vertexId_t src, void* metadata)
	{
		bcStaticData* bcd = (bcStaticData*) metadata;
		bcd->d[src] = INT32_MAX;
		bcd->sigma[src] = 0;
		bcd->delta[src] = 0.0;
	}

	static __device__ __forceinline__ void setLevelInfinity(cuStinger* custing,
		vertexId_t src, void* metadata)
	{
		bcStaticData* bcd = (bcStaticData*) metadata;
		bcd->level[src] = INT32_MAX;
	}

	// Dependency accumulation for one frontier
	static __device__ __forceinline__ void dependencyAccumulation(cuStinger* custing,
		vertexId_t src, vertexId_t dst, void* metadata)
	{
		bcStaticData* bcd = (bcStaticData*) metadata;

		vertexId_t *d = bcd->level;  // depth
		unsigned long long *sigma = bcd->sigma;
		float *delta = bcd->delta;

		vertexId_t v = src;
		vertexId_t w = dst;

		if (d[w] == d[v] + 1)
		{
			atomicAdd(delta + v, ((float) sigma[v] / (float) sigma[w]) * (1 + delta[w]));
		}
	}

}; // bcOperator

} //Namespace