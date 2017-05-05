#pragma once

#include "algs.cuh"
#include "operators.cuh"

#include "static_katz_centrality/katz.cuh"

// typedef unsigned long long int ulong_t;

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

	void batchUpdateInserted(cuStinger& custing,BatchUpdate &bu);
	void batchUpdateDeleted(cuStinger& custing,BatchUpdate &bu);
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
	void processUpdate(cuStinger& custing,BatchUpdate &bu, bool isInsert);

	cusLoadBalance* cusLB;
	katzCentrality kcStatic;

	cuStinger* invertedGraph;
	bool isDirected;
};

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

static __device__ void setupDeletions(cuStinger* custing,vertexId_t src, vertexId_t dst, void* metadata){
	katzDataStreaming* kd = (katzDataStreaming*)metadata;
	double minusAlpha = -kd->alpha;
	atomicAdd(kd->KC+src, minusAlpha);
	atomicAdd(kd->newPathsPrev+src, -1);
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

static __device__ void updateNewPathsBatchInsert(cuStinger* custing,vertexId_t src, vertexId_t dst, void* metadata){
	katzDataStreaming* kd = (katzDataStreaming*)metadata;
	ulong_t valToAdd = kd->nPaths[kd->iteration-1][dst];
	atomicAdd(kd->newPathsCurr+src, valToAdd);
}

static __device__ void updateNewPathsBatchDelete(cuStinger* custing,vertexId_t src, vertexId_t dst, void* metadata){
	katzDataStreaming* kd = (katzDataStreaming*)metadata;
	ulong_t valToRemove = -kd->nPaths[kd->iteration-1][dst];
	atomicAdd(kd->newPathsCurr+src, valToRemove);
}


static __device__ void updatePrevWithCurr(cuStinger* custing,vertexId_t src, void* metadata){
	katzDataStreaming* kd = (katzDataStreaming*)metadata;

	// Note the conversion to signed long long int!! Especially important for edge deletions where this diff can be negative
	long long int pathsDiff = kd->newPathsCurr[src] - kd->nPaths[kd->iteration][src];
	
	kd->KC[src] += kd->alphaI*(pathsDiff);
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
