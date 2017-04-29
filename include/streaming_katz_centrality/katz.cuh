#pragma once

#include "algs.cuh"
#include "operators.cuh"

#include "static_katz_centrality/katz.cuh"


typedef unsigned long long int ulong_t;

namespace cuStingerAlgs {

class katzDataStreaming: public katzData{
public:
	ulong_t*    newPathsCurr;
	ulong_t*    newPathsPrev;
	vertexQueue activeQueue; // Stores all the active vertices
	int*		active;
	length_t iterationStatic;
};

class katzCentralityStreaming{
public:
	void setInitParametersUndirected(length_t maxIteration_, length_t K_,length_t maxDegree_);
	void setInitParametersDirected(length_t maxIteration_, length_t K_,length_t maxDegree_,cuStinger* invertedGraph);

	void Init(cuStinger& custing);
	void runStatic(cuStinger& custing);

	void insertedBatchUpdate(cuStinger& custing,BatchUpdate &bu);
	// void deletedBatchUpdate(cuStinger& custing);
	void Release();


	void SyncHostWithDevice(){
		copyArrayDeviceToHost(deviceKatzData,&hostKatzData,1, sizeof(katzDataStreaming));
	}
	void SyncDeviceWithHost(){
		copyArrayHostToDevice(&hostKatzData,deviceKatzData,1, sizeof(katzDataStreaming));
	}

	length_t getIterationCount();

	virtual void copyKCToHost(double* hostArray){
		kcStatic.copyKCToHost(hostArray);
	}
	virtual void copynPathsToHost(ulong_t* hostArray){
		kcStatic.copynPathsToHost(hostArray);
	}

protected:
	katzDataStreaming hostKatzData, *deviceKatzData;
private:
	cusLoadBalance* cusLB;
	katzCentrality kcStatic;

	cuStinger* invertedGraph;
	bool isDirected;
};


// #define PRINT_SRC(string, src, val) if(src <= 976047) printf ("PREV[%d] = %d   : %s\n", src, val,string);
#define PRINT_SRC(src, val,string) if(src == 590578) printf ("PREV[%d] = %lld : %s\n", src, val,string);

class katzCentralityStreamingOperator{
public:

// Used only once when the streaming katz data structure is initialized
static __device__ void initStreaming(cuStinger* custing,vertexId_t src, void* metadata){
	katzDataStreaming* kd = (katzDataStreaming*)metadata;
	kd->newPathsCurr[src]=0;
	kd->newPathsPrev[src]= kd->nPaths[1][src];
	kd->active[src]=0;
}

static __device__ void setupInsertions(cuStinger* custing,vertexId_t src, vertexId_t dst, void* metadata){
	katzDataStreaming* kd = (katzDataStreaming*)metadata;
	atomicAdd(kd->KC+src, kd->alpha);
	atomicAdd(kd->newPathsPrev+src, 1);
	vertexId_t prev = atomicCAS(kd->active+src,0,kd->iteration);
	if(prev==0){
		kd->activeQueue.enqueue(src);
	}
}

static __device__ void initActiveNewPaths(cuStinger* custing,vertexId_t src, void* metadata){
	katzDataStreaming* kd = (katzDataStreaming*)metadata;
	kd->newPathsCurr[src]= kd->nPaths[kd->iteration][src];
}

static __device__ void findNextActive(cuStinger* custing,vertexId_t src, vertexId_t dst, void* metadata){
	katzDataStreaming* kd = (katzDataStreaming*)metadata;

	vertexId_t prev = atomicCAS(kd->active+dst,0,kd->iteration);
	if(prev==0){
		kd->activeQueue.enqueue(dst);
		kd->newPathsCurr[dst]= kd->nPaths[kd->iteration][dst];
	}
}

static __device__ void updateActiveNewPaths(cuStinger* custing,vertexId_t src, vertexId_t dst, void* metadata){
	katzDataStreaming* kd = (katzDataStreaming*)metadata;

	if(kd->active[src] < kd->iteration){
		ulong_t valToAdd = kd->newPathsPrev[src] - kd->nPaths[kd->iteration-1][src];
		atomicAdd(kd->newPathsCurr+dst, valToAdd);
	}
}

static __device__ void updateNewPathsBatch(cuStinger* custing,vertexId_t src, vertexId_t dst, void* metadata){
	katzDataStreaming* kd = (katzDataStreaming*)metadata;
	ulong_t valToAdd = kd->nPaths[kd->iteration-1][dst];
	atomicAdd(kd->newPathsCurr+src, valToAdd);
}


static __device__ void updatePrevWithCurr(cuStinger* custing,vertexId_t src, void* metadata){
	katzDataStreaming* kd = (katzDataStreaming*)metadata;
	
	kd->KC[src] += kd->alphaI*(kd->newPathsCurr[src] - kd->nPaths[kd->iteration][src]);
	if(kd->active[src] < kd->iteration){
		kd->nPaths[kd->iteration-1][src] = kd->newPathsPrev[src];
	}
	kd->newPathsPrev[src] = kd->newPathsCurr[src];
}

static __device__ void updateLastIteration(cuStinger* custing,vertexId_t src, void* metadata){
	katzDataStreaming* kd = (katzDataStreaming*)metadata;

	if(kd->active[src] < (kd->iteration)){
		kd->nPaths[kd->iteration-1][src] = kd->newPathsPrev[src];
	}
}

static __device__ void printPointers(cuStinger* custing,vertexId_t src, void* metadata){
	katzDataStreaming* kd = (katzDataStreaming*)metadata;
	if(threadIdx.x==0 && blockIdx.x==0 && src==0)
		printf("\n# %p %p %p %p %p %p %p %p #\n",kd->nPathsData,kd->nPaths, kd->nPathsPrev, kd->nPathsCurr, kd->KC,kd->lowerBound,kd->lowerBoundSort,kd->upperBound);

}



};



} // cuStingerAlgs namespace
