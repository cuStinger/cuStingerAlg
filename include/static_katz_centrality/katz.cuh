#pragma once

#include "algs.cuh"
#include "operators.cuh"


typedef unsigned long long int ulong_t;


namespace cuStingerAlgs {

class katzData{
public:

	ulong_t*   nPathsData;
	ulong_t**  nPaths;     // Will be used for dynamic graph algorithm which requires storing paths of all iterations.

	ulong_t*   nPathsCurr;
	ulong_t*   nPathsPrev;

	double*     KC;
	double*     lowerBound;
	double*     lowerBoundSort;
	double*     upperBound;

	vertexId_t*     vertexArray; // Sorting

	// vertexQueue queue; // Stores all the active vertices
	double alpha;
	double alphaI; // Alpha to the power of I  (being the iteration)

	double lowerBoundConst;
	double upperBoundConst;

	length_t K;

	length_t maxDegree;
	length_t iteration;
	length_t maxIteration;
	// number of active vertices at each iteration
	length_t nActive;
	length_t nv;
};

// Label propogation is based on the values from the previous iteration.
class katzCentrality:public StaticAlgorithm{
public:
	void setInitParameters(length_t maxIteration_,length_t K_,length_t maxDegree_, bool isStatic_=true);
	virtual void Init(cuStinger& custing);
	virtual void Reset();
	virtual void Run(cuStinger& custing);
	virtual void Release();

	virtual void SyncHostWithDevice(){
		copyArrayDeviceToHost(deviceKatzData,hostKatzData,1, sizeof(katzData));
	}
	virtual void SyncDeviceWithHost(){
		copyArrayHostToDevice(hostKatzData,deviceKatzData,1, sizeof(katzData));
	}

	length_t getIterationCount();

	const katzData* getHostKatzData(){return hostKatzData;}
	const katzData* getDeviceKatzData(){return deviceKatzData;}

	virtual void copyKCToHost(double* hostArray){
		copyArrayDeviceToHost(hostKatzData->KC,hostArray, hostKatzData->nv, sizeof(double));
	}

	virtual void copynPathsToHost(ulong_t* hostArray){
		copyArrayDeviceToHost(hostKatzData->nPathsData,hostArray, (hostKatzData->nv)*hostKatzData->maxIteration, sizeof(ulong_t));
	}


protected:
	// katzData hostKatzData, *deviceKatzData;
	katzData *hostKatzData, *deviceKatzData;

private:
	cusLoadBalance* cusLB;
	bool isStatic;
	ulong_t** hPathsPtr;  // Will be used to store pointers to all iterations of the Katz centrality results

};


class katzCentralityOperator{
public:

// Used at the very beginning
static __device__ void init(cuStinger* custing,vertexId_t src, void* metadata){
	katzData* kd = (katzData*)metadata;
	kd->nPathsPrev[src]=1;
	kd->nPathsCurr[src]=0;
	kd->KC[src]=0.0;
}

// Used every iteration
static __device__ void initNumPathsPerIteration(cuStinger* custing,vertexId_t src, void* metadata){
	katzData* kd = (katzData*)metadata;
	kd->nPathsCurr[src]=0;
}

static __device__ void updatePathCount(cuStinger* custing,vertexId_t src, vertexId_t dst, void* metadata){
	katzData* kd = (katzData*)metadata;
	atomicAdd(kd->nPathsCurr+src, kd->nPathsPrev[dst]);
}

static __device__ void updateKatzAndBounds(cuStinger* custing,vertexId_t src, void* metadata){
	katzData* kd = (katzData*)metadata;
	kd->KC[src]=kd->KC[src] + kd->alphaI * (double)kd->nPathsCurr[src];
	kd->lowerBound[src]=kd->KC[src] + kd->lowerBoundConst * (double)kd->nPathsCurr[src];
	kd->upperBound[src]=kd->KC[src] + kd->upperBoundConst * (double)kd->nPathsCurr[src];   
	kd->lowerBoundSort[src]=kd->lowerBound[src];
	kd->vertexArray[src]=src;
}

static __device__ void printKID(cuStinger* custing,vertexId_t src, void* metadata){
	katzData* kd = (katzData*)metadata;
	if(kd->nPathsPrev[src]!=1)
		printf("%d %ld\n ", src,kd->nPathsPrev[src]);
	if(kd->nPathsCurr[src]!=0)
		printf("%d %ld\n ", src,kd->nPathsCurr[src]);
}

static __device__ void countActive(cuStinger* custing,vertexId_t src, void* metadata){
	katzData* kd = (katzData*)metadata;
		if (kd->upperBound[src] > kd->lowerBound[kd->vertexArray[kd->K-1]]) {
		atomicAdd(&(kd -> nActive),1);
	}
}

static __device__ void printPointers(cuStinger* custing,vertexId_t src, void* metadata){
	katzData* kd = (katzData*)metadata;
	if(threadIdx.x==0 && blockIdx.x==0 && src==0)
		printf("\n@ %p %p %p %p %p %p %p %p @\n",kd->nPathsData,kd->nPaths, kd->nPathsPrev, kd->nPathsCurr, kd->KC,kd->lowerBound,kd->lowerBoundSort,kd->upperBound);
}



};



} // cuStingerAlgs namespace
