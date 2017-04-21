


#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>

#include <math.h>

#include "update.hpp"
#include "cuStinger.hpp"

#include "operators.cuh"

#include "static_katz_centrality/katz.cuh"
#include "streaming_katz_centrality/katz.cuh"

#include "kernel_mergesort.hxx"
using namespace mgpu;


namespace cuStingerAlgs {

void katzCentralityStreaming::setInitParameters(length_t maxIteration_, length_t K_,length_t maxDegree_){
	kcStatic.setInitParameters(maxIteration_,K_,maxDegree_,false);
}


void katzCentralityStreaming::Init(cuStinger& custing){
	kcStatic.Init(custing);

	deviceKatzData = (katzDataStreaming*)allocDeviceArray(1, sizeof(katzDataStreaming));
	
	copyArrayHostToHost((void*)kcStatic.getHostKatzData(),&hostKatzData, 1, sizeof(katzData));
	copyArrayDeviceToDevice((void*)kcStatic.getDeviceKatzData(),deviceKatzData, 1, sizeof(katzData));

	// allVinG_TraverseVertices<katzCentralityStreamingOperator::printPointers>(custing,deviceKatzData);

	hostKatzData.newPathsCurr = (ulong_t*) allocDeviceArray((hostKatzData.nv), sizeof(ulong_t));
	hostKatzData.newPathsPrev = (ulong_t*) allocDeviceArray((hostKatzData.nv), sizeof(ulong_t));
	hostKatzData.active 	  = (int*)    allocDeviceArray((hostKatzData.nv), sizeof(int));

	cusLB = new cusLoadBalance(custing.nv+1);

	hostKatzData.activeQueue.Init(custing.nv+1);
	// hostKatzData.nextIterQueue.Init(custing.nv+1);
	hostKatzData.activeQueue.resetQueue();
	// hostKatzData.nextIterQueue.resetQueue();

	SyncDeviceWithHost();

	allVinG_TraverseVertices<katzCentralityStreamingOperator::initStreaming>(custing,deviceKatzData);
}


void katzCentralityStreaming::runStatic(cuStinger& custing){
	// Run(custing);
	kcStatic.Reset();
	kcStatic.Run(custing);
	copyArrayHostToHost((void*)kcStatic.getHostKatzData(),&hostKatzData, 1, sizeof(katzData));
	copyArrayDeviceToDevice((void*)kcStatic.getDeviceKatzData(),deviceKatzData, 1, sizeof(katzData));
	hostKatzData.iterationStatic = hostKatzData.iteration;
	SyncDeviceWithHost();

}

void katzCentralityStreaming::Release(){
	delete cusLB;

	hostKatzData.activeQueue.freeQueue();
	// hostKatzData.nextIterQueue.freeQueue();
	freeDeviceArray(hostKatzData.newPathsCurr);
	freeDeviceArray(hostKatzData.newPathsPrev);
	freeDeviceArray(hostKatzData.active);
	freeDeviceArray(deviceKatzData);
	kcStatic.Release();

}


length_t katzCentralityStreaming::getIterationCount(){
	SyncHostWithDevice();
	return hostKatzData.iteration;
}

void katzCentralityStreaming::insertedBatchUpdate(cuStinger& custing,BatchUpdate &bu){

	hostKatzData.activeQueue.resetQueue();
	SyncDeviceWithHost();

	allEinA_TraverseEdges<katzCentralityStreamingOperator::setupInsertions>(custing, deviceKatzData,bu);
	SyncHostWithDevice();


	hostKatzData.iteration = 2;
	
	cout << "The number of queued elements is: " << hostKatzData.activeQueue.getQueueEnd() << endl;
	hostKatzData.nActive = hostKatzData.activeQueue.getQueueEnd();

	while(hostKatzData.iteration < hostKatzData.maxIteration && hostKatzData.iteration < hostKatzData.iterationStatic){

		hostKatzData.alphaI = pow(hostKatzData.alpha,hostKatzData.iteration);
		allVinA_TraverseVertices<katzCentralityStreamingOperator::initActiveNewPaths>(custing, deviceKatzData, hostKatzData.activeQueue.getQueue(), hostKatzData.nActive);

		SyncDeviceWithHost();

		allVinA_TraverseEdges_LB<katzCentralityStreamingOperator::findNextActive>(custing,deviceKatzData, *cusLB,hostKatzData.activeQueue);	
		allVinA_TraverseEdges_LB<katzCentralityStreamingOperator::updateActiveNewPaths>(custing,deviceKatzData, *cusLB,hostKatzData.activeQueue);

		allEinA_TraverseEdges<katzCentralityStreamingOperator::updateNewPathsBatch>(custing, deviceKatzData,bu);

		SyncHostWithDevice();
		hostKatzData.nActive = hostKatzData.activeQueue.getQueueEnd();

		allVinA_TraverseVertices<katzCentralityStreamingOperator::updatePrevWithCurr>(custing, deviceKatzData, hostKatzData.activeQueue.getQueue(), hostKatzData.nActive);

		SyncHostWithDevice();
		cout << "AFTER  - The number of queued elements is: " << hostKatzData.iteration  << "   " << hostKatzData.activeQueue.getQueueEnd() << endl;

		SyncHostWithDevice();
		hostKatzData.iteration++;

		SyncDeviceWithHost();
	}
	allVinA_TraverseVertices<katzCentralityStreamingOperator::updateLastIteration>(custing, deviceKatzData, hostKatzData.activeQueue.getQueue(), hostKatzData.nActive);


}


}// cuStingerAlgs namespace

