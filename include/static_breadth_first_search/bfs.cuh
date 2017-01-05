#pragma once

#include "algs.cuh"

// Breadth First Search

namespace cuStingerAlgs {


class bfsData {
public:
// struct bfsData{
	vertexQueue queue;
	vertexId_t* level;
	vertexId_t currLevel;
	vertexId_t root;
	length_t nv;
};


class StaticBreadthFirstSearch:public StaticAlgorithm{
public:	

	void Init(cuStinger& custing);
	void Reset();
	void Run(cuStinger& custing);
	void Release();

	void Run2(cuStinger& custing);

	void SyncHostWithDevice(){
		copyArrayDeviceToHost(deviceBfsData,&hostBfsData,1, sizeof(bfsData));
	}
	void SyncDeviceWithHost(){
		copyArrayHostToDevice(&hostBfsData,deviceBfsData,1, sizeof(bfsData));
	}
	
	length_t getLevels(){return hostBfsData.currLevel;}
	length_t getElementsFound(){return hostBfsData.queue.getQueueEnd();}



	void setInputParameters(vertexId_t root);

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
	bfsData hostBfsData, *deviceBfsData;
};


class bfsOperator:public StaticAlgorithm{
public:
static __device__ __forceinline__ void bfsExpandFrontier(cuStinger* custing,vertexId_t src, vertexId_t dst, void* metadata){
	bfsData* bd = (bfsData*)metadata;
	vertexId_t nextLevel=bd->currLevel+1;

	vertexId_t prev = atomicCAS(bd->level+dst,INT32_MAX,nextLevel);
	if(prev==INT32_MAX)
		bd->queue.enqueue(dst);
}

static __device__ __forceinline__ void setLevelInfinity(cuStinger* custing,vertexId_t src, void* metadata){
	bfsData* bd = (bfsData*)metadata;
	bd->level[src]=INT32_MAX;
}


}; // bfsOperator


} //Namespace