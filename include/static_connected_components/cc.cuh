#pragma once

#include "algs.cuh"

namespace cuStingerAlgs {

class ccDataBaseline{
public:
	vertexId_t* currState;
	vertexId_t* prevState;
	vertexId_t changeCurrIter;
	vertexId_t connectedComponentCount;
	length_t iteration;
};

class ccDataOptimized:public ccDataBaseline{
public:
	vertexQueue currQueue;
	vertexQueue nextQueue;
	length_t* updated;
	length_t nv;
	vertexId_t countVertexChange;
};



// Label propogation is based on the values from the previous iteration.
class ccBaseline:public StaticAlgorithm{
public:
	virtual void Init(cuStinger& custing);
	virtual void Reset();
	virtual void Run(cuStinger& custing);
	virtual void Release();

	void SyncHostWithDevice(){
		copyArrayDeviceToHost(deviceCCData,&hostCCData,1, sizeof(ccDataBaseline));
	}
	void SyncDeviceWithHost(){
		copyArrayHostToDevice(&hostCCData,deviceCCData,1, sizeof(ccDataBaseline));
	}

	// IGNORES connected components of size 1.
	length_t CountConnectComponents(cuStinger& custing);
	// INCLUDES connected components of size 1.
	length_t CountConnectComponentsAll(cuStinger& custing);

	length_t GetIterationCount();
protected: 
	ccDataBaseline hostCCData, *deviceCCData;
};


// Label propogation is done using a single array.
class ccConcurrent:public StaticAlgorithm{
public:
	virtual void Init(cuStinger& custing);
	virtual void Reset();
	virtual void Run(cuStinger& custing);
	virtual void Release();

	void SyncHostWithDevice(){
		copyArrayDeviceToHost(deviceCCData,&hostCCData,1, sizeof(ccDataBaseline));
	}
	void SyncDeviceWithHost(){
		copyArrayHostToDevice(&hostCCData,deviceCCData,1, sizeof(ccDataBaseline));
	}

	// IGNORES connected components of size 1.
	length_t CountConnectComponents(cuStinger& custing);
	// INCLUDES connected components of size 1.
	length_t CountConnectComponentsAll(cuStinger& custing);

	length_t GetIterationCount();

protected: 
	ccDataBaseline hostCCData, *deviceCCData;
};


// Label propogation is done using a single array.
class ccConcurrentLB:public StaticAlgorithm{
public:
	virtual void Init(cuStinger& custing);
	virtual void Reset();
	virtual void Run(cuStinger& custing);
	virtual void Release();

	void SyncHostWithDevice(){
		copyArrayDeviceToHost(deviceCCData,&hostCCData,1, sizeof(ccDataBaseline));
	}
	void SyncDeviceWithHost(){
		copyArrayHostToDevice(&hostCCData,deviceCCData,1, sizeof(ccDataBaseline));
	}

	// IGNORES connected components of size 1.
	length_t CountConnectComponents(cuStinger& custing);
	// INCLUDES connected components of size 1.
	length_t CountConnectComponentsAll(cuStinger& custing);

	length_t GetIterationCount();

protected: 
	ccDataBaseline hostCCData, *deviceCCData;
};


// Label propogation is done using a single array.
class ccConcurrentOptimized:public StaticAlgorithm{
public:
	virtual void Init(cuStinger& custing);
	virtual void Reset();
	virtual void Run(cuStinger& custing);
	virtual void Release();

	void SyncHostWithDevice(){
		copyArrayDeviceToHost(deviceCCData,&hostCCData,1, sizeof(ccDataOptimized));
	}
	void SyncDeviceWithHost(){
		copyArrayHostToDevice(&hostCCData,deviceCCData,1, sizeof(ccDataOptimized));
	}

	// IGNORES connected components of size 1.
	length_t CountConnectComponents(cuStinger& custing);
	// INCLUDES connected components of size 1.
	length_t CountConnectComponentsAll(cuStinger& custing);

	length_t GetIterationCount();

protected: 
	ccDataOptimized hostCCData, *deviceCCData;
};




class StaticConnectedComponentsOperator{
public:
static __device__ void init(cuStinger* custing,vertexId_t src, void* metadata){
	ccDataBaseline* ccd = (ccDataBaseline*)metadata;
	ccd->currState[src]=src;
}


static __device__ void swapWithPrev(cuStinger* custing,vertexId_t src, vertexId_t dst, void* metadata){
	ccDataBaseline* ccd = (ccDataBaseline*)metadata;
	vertexId_t prev=ccd->currState[src];
	if(prev<ccd->prevState[dst]){
		ccd->currState[src]=ccd->prevState[dst];
		atomicAdd(&(ccd->changeCurrIter),1);
	}
	prev=ccd->currState[dst];
	if(prev<ccd->prevState[src]){
		ccd->currState[dst]=ccd->prevState[src];
		atomicAdd(&(ccd->changeCurrIter),1);
	}
}

static __device__ void shortcut(cuStinger* custing,vertexId_t src, void* metadata){
	ccDataBaseline* ccd = (ccDataBaseline*)metadata;
	ccd->currState[src] = ccd->currState[ccd->currState[src]]; 
}

static __device__ void count(cuStinger* custing,vertexId_t src, void* metadata){
	ccDataBaseline* ccd = (ccDataBaseline*)metadata;
	if(ccd->currState[src] == src && custing->dVD->used[src] > 0){
		atomicAdd(&(ccd->connectedComponentCount),1);	
	}
}

static __device__ void countAll(cuStinger* custing,vertexId_t src, void* metadata){
	ccDataBaseline* ccd = (ccDataBaseline*)metadata;
	if(ccd->currState[src] == src){
		atomicAdd(&(ccd->connectedComponentCount),1);	
	}
}

static __device__ void swapLocal(cuStinger* custing,vertexId_t src, vertexId_t dst, void* metadata){
	ccDataBaseline* ccd = (ccDataBaseline*)metadata;

	vertexId_t prevSrc=ccd->currState[src];
	vertexId_t prevDst=ccd->currState[dst];
	// vertexId_t curr=prev;
	if(prevSrc<prevDst){
		ccd->currState[src]=prevDst;
		atomicAdd(&(ccd->changeCurrIter),1);
	}
	else if (prevSrc>prevDst){
		ccd->currState[dst]=prevSrc;
		atomicAdd(&(ccd->changeCurrIter),1);
	}


	// vertexId_t prev=ccd->currState[src];
	// // vertexId_t curr=prev;
	// if(prev<ccd->currState[dst]){
	// 	ccd->currState[src]=ccd->currState[dst];
	// 	atomicAdd(&(ccd->changeCurrIter),1);
	// }
	// prev=ccd->currState[dst];
	// if(prev<ccd->currState[src]){
	// 	ccd->currState[dst]=ccd->currState[src];
	// 	atomicAdd(&(ccd->changeCurrIter),1);
	// }

}
};

class StaticConnectedComponentsOptimizedOperator{
public:

static __device__ void initOptimized(cuStinger* custing,vertexId_t src, void* metadata){
	ccDataOptimized* ccd = (ccDataOptimized*)metadata;
	ccd->currState[src]=src;
	ccd->updated[src] = 0;
}

static __device__ void queueAll(cuStinger* custing,vertexId_t src, void* metadata){
	ccDataOptimized* ccd = (ccDataOptimized*)metadata;
	ccd->currQueue.enqueue(src);
	ccd->updated[src]=0;
}

static __device__ void resetAll(cuStinger* custing,vertexId_t src, void* metadata){
	ccDataOptimized* ccd = (ccDataOptimized*)metadata;
	ccd->updated[src]=0;
}


static __device__ void resetQueueElements(cuStinger* custing,vertexId_t src, void* metadata){
	ccDataOptimized* ccd = (ccDataOptimized*)metadata;
	ccd->updated[src]=0;
}

static __device__ void swapLocalAndCount(cuStinger* custing,vertexId_t src, vertexId_t dst, void* metadata){
	// return;
	ccDataOptimized* ccd = (ccDataOptimized*)metadata;

	vertexId_t prev=ccd->currState[src];
	if(prev<ccd->currState[dst]){
		ccd->currState[src]=ccd->currState[dst];
		atomicAdd(&(ccd->changeCurrIter),1);
		vertexId_t wasUpdated = atomicCAS(ccd->updated+src,0,1);
		if(wasUpdated==0){
			atomicAdd(&(ccd->countVertexChange),1);
		}
	}

	prev=ccd->currState[dst];
	if(prev<ccd->currState[src]){
		ccd->currState[dst]=ccd->currState[src];
		atomicAdd(&(ccd->changeCurrIter),1);
		vertexId_t wasUpdated = atomicCAS(ccd->updated+dst,0,1);
		if(wasUpdated==0){
			atomicAdd(&(ccd->countVertexChange),1);
		}
	}
}

static __device__ void swapLocalAndQueue(cuStinger* custing,vertexId_t src, vertexId_t dst, void* metadata){
	ccDataOptimized* ccd = (ccDataOptimized*)metadata;

	vertexId_t prev=ccd->currState[src];
	if(prev<ccd->currState[dst]){
		ccd->currState[src]=ccd->currState[dst];
		vertexId_t wasUpdated = atomicCAS(ccd->updated+src,0,1);
		if(wasUpdated==0){
			atomicAdd(&(ccd->changeCurrIter),1);
			ccd->nextQueue.enqueue(src);
		}

	}

	prev=ccd->currState[dst];
	if(prev<ccd->currState[src]){
		ccd->currState[dst]=ccd->currState[src];
		vertexId_t wasUpdated = atomicCAS(ccd->updated+dst,0,1);
		if(wasUpdated==0){
			atomicAdd(&(ccd->changeCurrIter),1);			
			ccd->nextQueue.enqueue(dst);
		}
	}


}

static __device__ void queueNeighbors(cuStinger* custing,vertexId_t src, vertexId_t dst, void* metadata){
	ccDataOptimized* ccd = (ccDataOptimized*)metadata;
	vertexId_t wasUpdated = atomicCAS(ccd->updated+dst,0,1);
	if(wasUpdated==0){
		// ccd->nextQueue.enqueue(dst);
	}
}



};


} // cuStingerAlgs namespace