


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
	hostKatzData.nextIterQueue.Init(custing.nv+1);
	hostKatzData.activeQueue.resetQueue();
	hostKatzData.nextIterQueue.resetQueue();

	SyncDeviceWithHost();

	allVinG_TraverseVertices<katzCentralityStreamingOperator::initStreaming>(custing,deviceKatzData);
}


void katzCentralityStreaming::runStatic(cuStinger& custing){
	// Run(custing);
	kcStatic.Reset();
	kcStatic.Run(custing);
	copyArrayHostToHost((void*)kcStatic.getHostKatzData(),&hostKatzData, 1, sizeof(katzData));
	copyArrayDeviceToDevice((void*)kcStatic.getDeviceKatzData(),deviceKatzData, 1, sizeof(katzData));

}

void katzCentralityStreaming::Release(){
	delete cusLB;

	hostKatzData.activeQueue.freeQueue();
	hostKatzData.nextIterQueue.freeQueue();
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
// return;
	hostKatzData.nActive = 0;
	length_t prevActive=0;
	while(hostKatzData.iteration < hostKatzData.maxIteration){

		hostKatzData.alphaI = pow(hostKatzData.alpha,hostKatzData.iteration);
		allVinA_TraverseVertices<katzCentralityStreamingOperator::initUpdateNewPaths>(custing, deviceKatzData, hostKatzData.activeQueue.getQueue(), hostKatzData.activeQueue.getQueueEnd());

		hostKatzData.nextIterQueue.resetQueue();
		SyncDeviceWithHost();

		// cout << "BEFORE - The number of queued elements is: " << hostKatzData.iteration  << "   " << hostKatzData.activeQueue.getQueueEnd() << "   " << hostKatzData.nextIterQueue.getQueueEnd() << endl;


		allVinA_TraverseEdges_LB<katzCentralityStreamingOperator::findNextActive>(custing,deviceKatzData, *cusLB,hostKatzData.activeQueue);	
		allVinA_TraverseEdges_LB<katzCentralityStreamingOperator::updateActiveNewPaths>(custing,deviceKatzData, *cusLB,hostKatzData.activeQueue);


		allEinA_TraverseEdges<katzCentralityStreamingOperator::updateNewPathsBatch>(custing, deviceKatzData,bu);
		SyncHostWithDevice();

		allVinA_TraverseVertices<katzCentralityStreamingOperator::enqueueNextToActive>(custing, deviceKatzData, hostKatzData.nextIterQueue.getQueue(), hostKatzData.nextIterQueue.getQueueEnd());


		SyncHostWithDevice();
		cout << "AFTER  - The number of queued elements is: " << hostKatzData.iteration  << "   " << hostKatzData.activeQueue.getQueueEnd() << "   " << hostKatzData.nextIterQueue.getQueueEnd() << endl;

		// allVinA_TraverseVertices<katzCentralityStreamingOperator::updateKC>(custing, deviceKatzData, hostKatzData.activeQueue.getQueue(), hostKatzData.activeQueue.getQueueEnd());


	// cout << "The number of queued elements is: " << hostKatzData.activeQueue.getQueueEnd() << endl;

			// allVinA_TraverseEdges_LB<katzCentralityStreamingOperator::updateActiveNewPaths>(custing,deviceKatzData, *cusLB,hostKatzData.activeQueue);	

		// allVinA_TraverseEdges_LB<katzCentralityOperator::updatePathCount>(custing,deviceKatzData,*cusLB);


		SyncHostWithDevice();
		hostKatzData.iteration++;
		ulong_t* temp = hostKatzData.newPathsCurr; hostKatzData.newPathsCurr=hostKatzData.newPathsPrev; hostKatzData.newPathsPrev=temp;	

		SyncDeviceWithHost();

	}

}


}// cuStingerAlgs namespace

