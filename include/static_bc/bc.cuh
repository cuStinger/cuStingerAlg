#pragma once

#include "algs.cuh"
#include "load_balance.cuh"
#include "static_bc/bc_tree.cuh"

// Betweenness Centrality

namespace cuStingerAlgs {

class StaticBC:public StaticAlgorithm {
public:	

	void Init(cuStinger& custing);
	void Reset();
	void Run(cuStinger& custing);
	void Release();

	void SyncHostWithDevice()
	{
		copyArrayDeviceToHost(deviceBcTree, hostBcTree, 1, sizeof(bcTree));
	}
	void SyncDeviceWithHost()
	{
		copyArrayHostToDevice(hostBcTree, deviceBcTree, 1, sizeof(bcTree));
	}

	void RunBfsTraversal(cuStinger& custing);
	void DependencyAccumulation(cuStinger& custing);
	
	length_t getLevel() { return hostBcTree->currLevel; }

	// Must pass in a root node vertex id, and a pointer to bc values (of length custing.nv)
	void setInputParameters(vertexId_t root, float *bc_array);

	// User is responsible for de-allocating memory.
	vertexId_t* getLevelArrayHost()
	{
		vertexId_t* hostArr = (vertexId_t*)allocHostArray(hostBcTree->nv, sizeof(vertexId_t));
		copyArrayDeviceToHost(hostBcTree->d, hostArr, hostBcTree->nv, sizeof(vertexId_t) );
		return hostArr;
	}

	// User sends pre-allocated array.
	void getLevelArrayForHost(vertexId_t* hostArr)
	{
		copyArrayDeviceToHost(hostBcTree->d, hostArr, hostBcTree->nv, sizeof(vertexId_t) );
	}

	bcTree *hostBcTree, *deviceBcTree;

private:
	float *bc;  // the actual bc values array on the host
	// a float array which will contain a copy of the device delta array during dependency accumulation
	float *host_deltas;
	cusLoadBalance* cusLB;
};


class bcOperator:public StaticAlgorithm {
public:

	static __device__ __forceinline__ void bcExpandFrontier(cuStinger* custing,
		vertexId_t src, vertexId_t dst, void* metadata)
	{
		bcTree* bcd = (bcTree*) metadata;
		vertexId_t nextLevel = bcd->currLevel + 1;

		vertexId_t v = src;
		vertexId_t w = dst;

		vertexId_t prev = atomicCAS(bcd->d + w, INT32_MAX, nextLevel);
		if (prev == INT32_MAX) {
			bcd->queue.enqueue(w);
		}
		if (bcd->d[w] == nextLevel) {
			atomicAdd(bcd->sigma + w, bcd->sigma[v]);
		}

	}

	// Use macro to clear values in arrays to 0
	static __device__ __forceinline__ void clearArrays(cuStinger* custing,
		vertexId_t src, void* metadata)
	{
		bcTree* bcd = (bcTree*) metadata;
		bcd->d[src] = INT32_MAX;
		bcd->sigma[src] = 0;
		bcd->delta[src] = 0.0;
	}

	static __device__ __forceinline__ void setLevelInfinity(cuStinger* custing,
		vertexId_t src, void* metadata)
	{
		bcTree* bcd = (bcTree*) metadata;
		bcd->d[src] = INT32_MAX;
	}

	// Dependency accumulation for one frontier
	static __device__ __forceinline__ void dependencyAccumulation(cuStinger* custing,
		vertexId_t src, vertexId_t dst, void* metadata)
	{
		bcTree* bcd = (bcTree*) metadata;

		vertexId_t *d = bcd->d;  // depth
		vertexId_t *sigma = bcd->sigma;
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