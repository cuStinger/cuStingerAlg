#pragma once

#include "algs.cuh"
#include "operators.cuh"

#include "static_katz_centrality/katz.cuh"


typedef unsigned long long int ulong_t;

namespace cuStingerAlgs {

class katzDataStreaming: public katzData{
public:
	ulong_t*   newPathsCurr;
	ulong_t*   newPathsPrev;
};

class katzCentralityStreaming:protected katzCentrality{
public:
	void setInitParameters(length_t K_,length_t maxDegree_, length_t maxIteration_);

	virtual void Init(cuStinger& custing);
	// virtual void Reset();

	void runStatic(cuStinger& custing);

	// virtual void insertedBatchUpdate(cuStinger& custing);
	// virtual void deletedBatchUpdate(cuStinger& custing);
	virtual void Release();


	void SyncHostWithDevice(){
		printf("this should never happen\n"); fflush(stdout);
		copyArrayDeviceToHost(deviceKatzData,&hostKatzData,1, sizeof(katzDataStreaming));
	}
	void SyncDeviceWithHost(){
		printf("this should never happen\n"); fflush(stdout);
		copyArrayHostToDevice(&hostKatzData,deviceKatzData,1, sizeof(katzDataStreaming));
	}

	length_t getIterationCount();
// protected:
// 	katzDataStreaming hostKatzDataStr, *deviceKatzDataStr;
private:
	cusLoadBalance* cusLB;
	katzCentrality kcStatic;
};




class katzCentralityStreamingOperator{
public:

// Used at the very beginning
static __device__ void init(cuStinger* custing,vertexId_t src, void* metadata){
	katzDataStreaming* kd = (katzDataStreaming*)metadata;
	kd->newPathsCurr[src]=0;
	kd->newPathsPrev[src]=0; // kd->nPaths[0][src]
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
