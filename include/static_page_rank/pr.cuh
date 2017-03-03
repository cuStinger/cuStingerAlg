#pragma once

#include "algs.cuh"

namespace cuStingerAlgs {

typedef float prType;
class pageRankData{
public:
	prType* prevPR;
	prType* currPR;
	prType* absDiff;
	// void* reduction;
	prType* reductionOut;
	prType* contri;

	length_t iteration;
	length_t iterationMax;
	length_t nv;
	prType threshhold;
	prType damp;
	prType normalizedDamp;
};

// Label propogation is based on the values from the previous iteration.
class StaticPageRank:public StaticAlgorithm{
public:
	virtual void Init(cuStinger& custing);
	virtual void Reset();
	virtual void Run(cuStinger& custing);
	virtual void Release();

	void SyncHostWithDevice(){
		copyArrayDeviceToHost(devicePRData,&hostPRData,1, sizeof(pageRankData));
	}
	void SyncDeviceWithHost(){
		copyArrayHostToDevice(&hostPRData,devicePRData,1, sizeof(pageRankData));
	}
	void setInputParameters(length_t iterationMax = 20, prType threshhold = 0.001 ,prType damp=0.85);

	length_t getIterationCount();

	// User is responsible for de-allocating memory.
	prType* getPageRankScoresHost(){
		prType* hostArr = (prType*)allocHostArray(hostPRData.nv, sizeof(prType));
		copyArrayDeviceToHost(hostPRData.currPR, hostArr, hostPRData.nv, sizeof(prType) );
		return hostArr;
	}

	// User sends pre-allocated array.
	void getPageRankScoresHost(vertexId_t* hostArr){
		copyArrayDeviceToHost(hostPRData.currPR, hostArr, hostPRData.nv, sizeof(prType) );
	}

	void printRankings(cuStinger& custing);

protected: 
	pageRankData hostPRData, *devicePRData;
	length_t reductionBytes;
private: 
	cusLoadBalance* cusLB;	
};



class StaticPageRankOperator{
public:
static __device__ void init(cuStinger* custing,vertexId_t src, void* metadata){
	pageRankData* pr = (pageRankData*)metadata;
	pr->absDiff[src]=pr->currPR[src]=0.0;
	pr->prevPR[src]=1/float(pr->nv);
	// printf("%f, ", pr->prevPR[src]);
	*(pr->reductionOut)=0;
}

static __device__ void resetCurr(cuStinger* custing,vertexId_t src, void* metadata){
	pageRankData* pr = (pageRankData*)metadata;
	pr->currPR[src]=0.0;
	*(pr->reductionOut)=0;
}

static __device__ void computeContribuitionPerVertex(cuStinger* custing,vertexId_t src, void* metadata){
	pageRankData* pr = (pageRankData*)metadata;
	length_t sizeSrc = custing->dVD->getUsed()[src];
	if(sizeSrc==0)
		pr->contri[src]=0.0;
	else
		pr->contri[src]=pr->prevPR[src]/sizeSrc;
}


static __device__ void addContribuitions(cuStinger* custing,vertexId_t src, vertexId_t dst, void* metadata){
	pageRankData* pr = (pageRankData*)metadata;
	atomicAdd(pr->currPR+dst,pr->contri[src]);
}

static __device__ void addContribuitionsUndirected(cuStinger* custing,vertexId_t src, vertexId_t dst, void* metadata){
	pageRankData* pr = (pageRankData*)metadata;
	atomicAdd(pr->currPR+src,pr->contri[dst]);

}

static __device__ void dampAndDiffAndCopy(cuStinger* custing,vertexId_t src, void* metadata){
	pageRankData* pr = (pageRankData*)metadata;
	// pr->currPR[src]=(1-pr->damp)/float(pr->nv)+pr->damp*pr->currPR[src];
	pr->currPR[src]=pr->normalizedDamp+pr->damp*pr->currPR[src];

	pr->absDiff[src]= fabsf(pr->currPR[src]-pr->prevPR[src]);
	pr->prevPR[src]=pr->currPR[src];
}

static __device__ void sum(cuStinger* custing,vertexId_t src, void* metadata){
	pageRankData* pr = (pageRankData*)metadata;
	atomicAdd(pr->reductionOut,pr->absDiff[src] );
}

static __device__ void sumPr(cuStinger* custing,vertexId_t src, void* metadata){
	pageRankData* pr = (pageRankData*)metadata;
	atomicAdd(pr->reductionOut,pr->prevPR[src] );
}



static __device__ void setIds(cuStinger* custing,vertexId_t src, void* metadata){
	vertexId_t* ids = (vertexId_t*)metadata;
	ids[src]=src;
}

static __device__ void print(cuStinger* custing,vertexId_t src, void* metadata){
	int* ids = (int*)metadata;
	if(threadIdx.x==0 & blockIdx.x==0){
		// printf("The wheels on the bus go round and round and round and round %d\n",*ids);
	}

}



// static __device__ void addDampening(cuStinger* custing,vertexId_t src, void* metadata){
// 	pageRankData* pr = (pageRankData*)metadata;
// 	pr->currPR[src]=(1-pr->damp)/float(pr->nv)+pr->damp*pr->currPR[src];
// }

// static __device__ void absDiff(cuStinger* custing,vertexId_t src, void* metadata){
// 	pageRankData* pr = (pageRankData*)metadata;
// 	pr->absDiff[src]= abs(pr->currPR[src]-pr->prevPR[src]);
// }

// static __device__ void prevEqualCurr(cuStinger* custing,vertexId_t src, void* metadata){
// 	pageRankData* pr = (pageRankData*)metadata;
// 	pr->prevPR[src]=pr->currPR[src];
// }



};



} // cuStingerAlgs namespace