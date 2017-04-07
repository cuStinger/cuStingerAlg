#pragma once

#include "algs.cuh"
#include "operators.cuh"

typedef unsigned long long int ulong_t;

namespace cuStingerAlgs {

class katzDataStreaming{
public:
	ulong_t*   newPathsCurr;
	ulong_t*   newPathsPrev;

	ulong_t*   nPathsData;
	ulong_t**  nPaths;
	// unsigned long long int*   nPathsPrev;

	double*     KC;
	// double*     lowerBound;
	// double*     lowerBoundSort;
	// double*     upperBound;

	vertexId_t*     vertexArray; // Sorting

	vertexQueue queue; // Stores all the active vertices
	double alpha;
	double* alphaI; // Alpha to the power of I  (being the iteration)

	// double lowerBoundConst;
	// double upperBoundConst;

	length_t K;

	length_t maxDegree;
	length_t iteration;
	length_t maxIteration;
	// number of active vertices at each iteration
	// length_t nActive;
};

class katzCentralityStreaming{
public:
	void setInitParameters(length_t K_,length_t maxDegree_, length_t maxIteration_);

	virtual void Init(cuStinger& custing);
	virtual void Reset();

	// virtual void insertedBatchUpdate(cuStinger& custing);
	// virtual void deletedBatchUpdate(cuStinger& custing);
	virtual void Release();


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
