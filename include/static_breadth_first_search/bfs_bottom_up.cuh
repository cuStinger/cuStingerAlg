#pragma once

#include "algs.cuh"

// Breadth First Search

namespace cuStingerAlgs {


class bfsDataBU {
public:
	vertexQueue queue;
	vertexId_t* level;
	vertexId_t currLevel;
	vertexId_t root;
	length_t nv;
	length_t verticesFound;
};


class bfsBU:public StaticAlgorithm{
public:	

	void Init(cuStinger& custing);
	void Reset();
	void Run(cuStinger& custing);
	void Release();

	void Run2(cuStinger& custing);

	void SyncHostWithDevice(){
		copyArrayDeviceToHost(deviceBfsData,&hostBfsData,1, sizeof(bfsDataBU));
	}
	void SyncDeviceWithHost(){
		copyArrayHostToDevice(&hostBfsData,deviceBfsData,1, sizeof(bfsDataBU));
	}
	
	length_t getLevels(){return hostBfsData.currLevel;}
	length_t getElementsFound(cuStinger& custing);

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
	bfsDataBU hostBfsData, *deviceBfsData;
};


class bfsBottomUpOperator:public StaticAlgorithm{
public:
static __device__ __forceinline__ void bfsExpandFrontier(cuStinger* custing,vertexId_t src, vertexId_t dst, void* metadata){
	bfsDataBU* bd = (bfsDataBU*)metadata;
	vertexId_t src__=dst, dst__=src;
	if(bd->level[src__]!=INT32_MAX)
		return;
	if(bd->level[dst__]==INT32_MAX)
		return;
	if(bd->level[dst__]==bd->currLevel){
		vertexId_t prevVal = atomicCAS(bd->level+src__,INT32_MAX,bd->currLevel+1);
		if (prevVal==INT32_MAX)
			atomicAdd(&bd->verticesFound,1);
	}
}

static __device__ __forceinline__ void setLevelInfinity(cuStinger* custing,vertexId_t src, void* metadata){
	bfsDataBU* bd = (bfsDataBU*)metadata;
	bd->level[src]=INT32_MAX;
}

static __device__ __forceinline__ void countFound(cuStinger* custing,vertexId_t src, void* metadata){
	bfsDataBU* bd = (bfsDataBU*)metadata;
	if(bd->level[src]!=INT32_MAX)
		atomicAdd(&bd->verticesFound,1);

}


}; // bfsOperator


} //Namespace