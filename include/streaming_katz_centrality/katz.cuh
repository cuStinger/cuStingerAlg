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
	vertexQueue nextIterQueue; // Stores all the active vertices

	int*		active;

};

class katzCentralityStreaming{
public:
	void setInitParameters(length_t maxIteration_, length_t K_,length_t maxDegree_);

	void Init(cuStinger& custing);
	// virtual void Reset();

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


protected:
	katzDataStreaming hostKatzData, *deviceKatzData;
private:
	cusLoadBalance* cusLB;
	katzCentrality kcStatic;
};




class katzCentralityStreamingOperator{
public:


static __device__ void printPointers(cuStinger* custing,vertexId_t src, void* metadata){
	katzDataStreaming* kd = (katzDataStreaming*)metadata;
	// if(threadIdx.x==0 && blockIdx.x==0 && src==0)
	// 	printf("\n# %p %p %p %p %p %p #\n",kd->nPathsPrev, kd->nPathsCurr, kd->KC,kd->lowerBound,kd->lowerBoundSort,kd->upperBound);
	if(threadIdx.x==0 && blockIdx.x==0 && src==0)
		printf("\n# %d #\n",kd->iteration);

}


// Used only once when the streaming katz data structure is initialized
static __device__ void initStreaming(cuStinger* custing,vertexId_t src, void* metadata){
	katzDataStreaming* kd = (katzDataStreaming*)metadata;
	kd->newPathsCurr[src]=0;
	kd->newPathsPrev[src]= kd->nPaths[0][src];
	kd->active[src]=0;
}

static __device__ void setupInsertions(cuStinger* custing,vertexId_t src, vertexId_t dst, void* metadata){
	katzDataStreaming* kd = (katzDataStreaming*)metadata;
	atomicAdd(kd->KC+src, kd->alpha);
	atomicAdd(kd->newPathsCurr+src, 1);

	// vertexId_t prev = atomicCAS(kd->active+src,0,1);
	vertexId_t prev = atomicCAS(kd->active+src,0,kd->iteration);
	if(prev==0){
		kd->activeQueue.enqueue(src);
	}


}

static __device__ void initUpdateNewPaths(cuStinger* custing,vertexId_t src, void* metadata){
	katzDataStreaming* kd = (katzDataStreaming*)metadata;
	kd->newPathsCurr[src]= kd->nPaths[kd->iteration-1][src];
}

static __device__ void findNextActive(cuStinger* custing,vertexId_t src, vertexId_t dst, void* metadata){
	katzDataStreaming* kd = (katzDataStreaming*)metadata;

	// vertexId_t prev = atomicCAS(kd->active+dst,0,1);
	vertexId_t prev = atomicCAS(kd->active+dst,0,kd->iteration);
	if(prev==0){
		kd->nextIterQueue.enqueue(dst);
		kd->newPathsCurr[dst]= kd->nPaths[kd->iteration-1][dst];
	}
}

static __device__ void updateActiveNewPaths(cuStinger* custing,vertexId_t src, vertexId_t dst, void* metadata){
	katzDataStreaming* kd = (katzDataStreaming*)metadata;
	ulong_t valToAdd = kd->newPathsPrev[src] - kd->nPaths[kd->iteration-2][src];
	atomicAdd(kd->newPathsCurr+dst, valToAdd);

}


static __device__ void enqueueNextToActive(cuStinger* custing,vertexId_t src, void* metadata){
	katzDataStreaming* kd = (katzDataStreaming*)metadata;
	kd->activeQueue.enqueue(src);


	// kd->newPathsCurr[src]= kd->nPaths[kd->iteration-1][src];
}


static __device__ void updateNewPathsBatch(cuStinger* custing,vertexId_t src, vertexId_t dst, void* metadata){
	katzDataStreaming* kd = (katzDataStreaming*)metadata;
	atomicAdd(kd->newPathsCurr+src, kd->nPaths[kd->iteration-2][dst]);

}


// // Used at the very beginning
// static __device__ void init(cuStinger* custing,vertexId_t src, void* metadata){
// 	katzDataStreaming* kd = (katzDataStreaming*)metadata;
// 	kd->nPathsPrev[src]=1;
// 	kd->nPathsCurr[src]=0;
// 	kd->KC[src]=0.0;
// }

// // Used every iteration
// static __device__ void initNumPathsPerIteration(cuStinger* custing,vertexId_t src, void* metadata){
// 	katzData* kd = (katzData*)metadata;
// 	kd->nPathsCurr[src]=0;
// }


// static __device__ void updatePathCount(cuStinger* custing,vertexId_t src, vertexId_t dst, void* metadata){
// 	katzData* kd = (katzData*)metadata;
// 	atomicAdd(kd->nPathsCurr+src, kd->nPathsPrev[dst]);
// }


// static __device__ void updateKatzAndBounds(cuStinger* custing,vertexId_t src, void* metadata){
// 	katzData* kd = (katzData*)metadata;
// 	kd->KC[src]=kd->KC[src] + kd->alphaI * (double)kd->nPathsCurr[src];
// 	kd->lowerBound[src]=kd->KC[src] + kd->lowerBoundConst * (double)kd->nPathsCurr[src];
// 	kd->upperBound[src]=kd->KC[src] + kd->upperBoundConst * (double)kd->nPathsCurr[src];   
// 	kd->lowerBoundSort[src]=kd->lowerBound[src];
// 	kd->vertexArray[src]=src;
// }

// static __device__ void printKID(cuStinger* custing,vertexId_t src, void* metadata){

// 	katzData* kd = (katzData*)metadata;
// 	if(kd->vertexArray[src]==kd->K)
// 		printf("%d\n",src);
  

// }

// static __device__ void countActive(cuStinger* custing,vertexId_t src, void* metadata){
// 	katzData* kd = (katzData*)metadata;
// 		if (kd->upperBound[src] > kd->lowerBound[kd->vertexArray[kd->K-1]]) {
// 		atomicAdd(&(kd -> nActive),1);
// 		// kd -> nActive ++; // TODO how can i do this as an atomic instruction?
// 	}
// }


};



} // cuStingerAlgs namespace
