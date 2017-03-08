#pragma once

#include "algs.cuh"

// Breadth First Search

namespace cuStingerAlgs {


class bfsDataHybrid {
public:
	vertexQueue activeQueue;
	vertexQueue unMarkedQueue;
	vertexId_t* level;
	vertexId_t currLevel;
	vertexId_t root;
	vertexId_t rootSize;	
	length_t nv;
	length_t ne;	
	length_t verticesFound;
	length_t edgesTraversed; // Used by the TD algorithm for counting edges in the frontier.

	int alpha;
	int beta;
};


class bfsHybrid:public StaticAlgorithm{
public:	

	void Init(cuStinger& custing);
	void Reset();
	void Run(cuStinger& custing);
	void Release();

	void Run2(cuStinger& custing);

	void SyncHostWithDevice(){
		copyArrayDeviceToHost(deviceBfsData,&hostBfsData,1, sizeof(bfsDataHybrid));
	}
	void SyncDeviceWithHost(){
		copyArrayHostToDevice(&hostBfsData,deviceBfsData,1, sizeof(bfsDataHybrid));
	}
	
	length_t getLevels(){return hostBfsData.currLevel;}
	length_t getElementsFound(cuStinger& custing);

	void setInputParameters(vertexId_t root,int alpha=15,int beta=18);

	// User is responsible for de-allocating memory.
	vertexId_t* getLevelArrayHost(){
		vertexId_t* hostArr = (vertexId_t*)allocHostArray(hostBfsData.nv, sizeof(vertexId_t));
		copyArrayDeviceToHost(hostBfsData.level, hostArr, hostBfsData.nv, sizeof(vertexId_t) );
		return hostArr;
	}

	// User sends pre-allocated array.
	void getLevelArrayForHost(vertexId_t* hostArr){
		copyArrayDeviceToHost(hostBfsData.level, hostArr, hostBfsData.nv, sizeof(vertexId_t) );
	}

	

// protected: 
	bfsDataHybrid hostBfsData, *deviceBfsData;
};


class bfsHybridOperator:public StaticAlgorithm{
public:
static __device__ __forceinline__ void bfsExpandFrontierBU(cuStinger* custing,vertexId_t src, vertexId_t dst, void* metadata){
	bfsDataHybrid* bd = (bfsDataHybrid*)metadata;
	vertexId_t src__=dst, dst__=src;
	if(bd->level[src__]!=INT32_MAX)
		return;
	if(bd->level[dst__]==INT32_MAX)
		return;
	if(bd->level[dst__]==bd->currLevel){
		// atomicCAS(bd->level+src__,INT32_MAX,bd->currLevel+1);
		// atomicAdd(&bd->verticesFound,1);
		vertexId_t prevVal = atomicCAS(bd->level+src__,INT32_MAX,bd->currLevel+1);
		if (prevVal==INT32_MAX)
			atomicAdd(&bd->verticesFound,1);

	}

}

static __device__ __forceinline__ void setLevelInfinity(cuStinger* custing,vertexId_t src, void* metadata){
	bfsDataHybrid* bd = (bfsDataHybrid*)metadata;
	bd->level[src]=INT32_MAX;
}

static __device__ __forceinline__ void countFound(cuStinger* custing,vertexId_t src, void* metadata){
	bfsDataHybrid* bd = (bfsDataHybrid*)metadata;
	if(bd->level[src]!=INT32_MAX)
		atomicAdd(&bd->verticesFound,1);
}

static __device__ __forceinline__ void countEdges(cuStinger* custing,vertexId_t src, void* metadata){
	bfsDataHybrid* bd = (bfsDataHybrid*)metadata;
	atomicAdd(&bd->ne,custing->dVD->getUsed()[src]);
}

static __device__ __forceinline__ void getRootSize(cuStinger* custing,vertexId_t src, void* metadata){
	bfsDataHybrid* bd = (bfsDataHybrid*)metadata;
	if (src==bd->root){
		printf("The root is %d\n",src);
		bd->rootSize=custing->dVD->getUsed()[src];
	}
}

static __device__ __forceinline__ void createActiveFrontierQueue(cuStinger* custing,vertexId_t src, void* metadata){
	bfsDataHybrid* bd = (bfsDataHybrid*)metadata;
	if(bd->level[src]==bd->currLevel)
		bd->activeQueue.enqueue(src);
}

static __device__ __forceinline__ void bfsExpandFrontierTD(cuStinger* custing,vertexId_t src, vertexId_t dst, void* metadata){
	bfsDataHybrid* bd = (bfsDataHybrid*)metadata;
	vertexId_t nextLevel=bd->currLevel+1;
	vertexId_t prev = atomicCAS(bd->level+dst,INT32_MAX,nextLevel);
	if(prev==INT32_MAX){
		bd->activeQueue.enqueue(dst);
		atomicAdd(&bd->edgesTraversed,custing->dVD->getUsed()[dst]);
	}
}

}; // bfsOperator


} //Namespace