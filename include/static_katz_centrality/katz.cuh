#pragma once

#include "algs.cuh"
#include "operators.cuh"

namespace cuStingerAlgs {

class katzData{
public:
	length_t*   nPathsCurr;
	length_t*   nPathsPrev;

	double*     KC;
	double*     lowerBound;
	double*     upperBound;

	vertexId_t*     vertexArray; // Sorting

	vertexQueue queue; // Stores all the active vertices	
	double alpha;
	double alphaI; // Alpha to the power of I  (being the iteration)

	double lowerBoundConst;
	double upperBoundConst;

	length_t K;

	length_t maxDegree;
	length_t iteration;
};

// Label propogation is based on the values from the previous iteration.
class katzCentrality:public StaticAlgorithm{
public:
	virtual void Init(cuStinger& custing);
	virtual void Reset();
	virtual void Run(cuStinger& custing);
	virtual void Release();

	void setInputParameters(length_t K_,length_t maxDegree_){hostKatzData.K=K_;hostKatzData.maxDegree=maxDegree_;}

	void SyncHostWithDevice(){
		copyArrayDeviceToHost(deviceKatzData,&hostKatzData,1, sizeof(katzData));
	}
	void SyncDeviceWithHost(){
		copyArrayHostToDevice(&hostKatzData,deviceKatzData,1, sizeof(katzData));
	}

	length_t GetIterationCount();
protected: 
	katzData hostKatzData, *deviceKatzData;
private: 
	cusLoadBalance* cusLB;		
};




class katzCentralityOperator{
public:

// Used at the very beginning
static __device__ void initNumPaths(cuStinger* custing,vertexId_t src, void* metadata){
	katzData* kd = (katzData*)metadata;
	kd->nPathsPrev[src]=1;
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
	kd->KC[src]=kd->KC[src] + kd->alphaI * kd->nPathsCurr[src];
	kd->lowerBound[src]=kd->KC[src] + kd->lowerBoundConst * kd->nPathsCurr[src];
	// kd->upperBound[src]=kd->KC[src] + kd->upperBoundConst * kd->nPathsCurr[src];
	// kd->lowerBound[src]=0;
	kd->upperBound[src]=0;

	kd->vertexArray[src]=src;	

}

static __device__ void printKID(cuStinger* custing,vertexId_t src, void* metadata){

	katzData* kd = (katzData*)metadata;
	if(kd->vertexArray[src]==kd->K)
		printf("%d\n",src);
}


};



} // cuStingerAlgs namespace