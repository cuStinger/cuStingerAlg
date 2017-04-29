


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

void katzCentralityStreaming::setInitParametersUndirected(length_t maxIteration_, length_t K_,length_t maxDegree_){
	kcStatic.setInitParameters(maxIteration_,K_,maxDegree_,false);
	isDirected=false;
}
void katzCentralityStreaming::setInitParametersDirected(length_t maxIteration_, length_t K_,length_t maxDegree_, cuStinger* invertedGraph__){
	kcStatic.setInitParameters(maxIteration_,K_,maxDegree_,false);
	invertedGraph=invertedGraph__;
	isDirected=true;

}


void katzCentralityStreaming::Init(cuStinger& custing){


	kcStatic.Init(custing);

	deviceKatzData = (katzDataStreaming*)allocDeviceArray(1, sizeof(katzDataStreaming));
	
	copyArrayHostToHost((void*)kcStatic.getHostKatzData(),&hostKatzData, 1, sizeof(katzData));
	copyArrayDeviceToDevice((void*)kcStatic.getDeviceKatzData(),deviceKatzData, 1, sizeof(katzData));

	hostKatzData.newPathsCurr = (ulong_t*) allocDeviceArray((hostKatzData.nv), sizeof(ulong_t));
	hostKatzData.newPathsPrev = (ulong_t*) allocDeviceArray((hostKatzData.nv), sizeof(ulong_t));
	hostKatzData.active 	  = (int*)    allocDeviceArray((hostKatzData.nv), sizeof(int));

	cusLB = new cusLoadBalance(custing.nv+1);

	hostKatzData.activeQueue.Init(custing.nv+1);
	hostKatzData.activeQueue.resetQueue();

	SyncDeviceWithHost();
}


void katzCentralityStreaming::runStatic(cuStinger& custing){

	kcStatic.Reset();
	kcStatic.Run(custing);

	copyArrayHostToHost((void*)kcStatic.getHostKatzData(),&hostKatzData, 1, sizeof(katzData));
	copyArrayDeviceToDevice((void*)kcStatic.getDeviceKatzData(),deviceKatzData, 1, sizeof(katzData));
	hostKatzData.iterationStatic = hostKatzData.iteration;
	SyncDeviceWithHost();

	allVinG_TraverseVertices<katzCentralityStreamingOperator::initStreaming>(custing,deviceKatzData);
}

void katzCentralityStreaming::Release(){
	delete cusLB;

	hostKatzData.activeQueue.freeQueue();
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
	hostKatzData.iteration = 1;

	SyncDeviceWithHost();

	allEinA_TraverseEdges<katzCentralityStreamingOperator::setupInsertions>(custing, deviceKatzData,bu);	
	SyncHostWithDevice();

	hostKatzData.iteration = 2;
	hostKatzData.nActive = hostKatzData.activeQueue.getQueueEnd();

	while(hostKatzData.iteration < hostKatzData.maxIteration && hostKatzData.iteration < hostKatzData.iterationStatic){
		hostKatzData.alphaI = pow(hostKatzData.alpha,hostKatzData.iteration);
		SyncDeviceWithHost();

		allVinA_TraverseVertices<katzCentralityStreamingOperator::initActiveNewPaths>(custing, deviceKatzData, hostKatzData.activeQueue.getQueue(), hostKatzData.nActive);

		if(!isDirected){
			allVinA_TraverseEdges_LB<katzCentralityStreamingOperator::findNextActive>(custing,deviceKatzData, *cusLB,hostKatzData.activeQueue);	
			SyncHostWithDevice();
			allVinA_TraverseEdges_LB<katzCentralityStreamingOperator::updateActiveNewPaths>(custing,deviceKatzData, *cusLB,hostKatzData.activeQueue);
			SyncHostWithDevice();
		}
		else{
			allVinA_TraverseEdges_LB<katzCentralityStreamingOperator::findNextActive>(*invertedGraph,deviceKatzData, *cusLB,hostKatzData.activeQueue);	
			SyncHostWithDevice();
			allVinA_TraverseEdges_LB<katzCentralityStreamingOperator::updateActiveNewPaths>(*invertedGraph,deviceKatzData, *cusLB,hostKatzData.activeQueue);
			SyncHostWithDevice();
		}

		allEinA_TraverseEdges<katzCentralityStreamingOperator::updateNewPathsBatch>(custing, deviceKatzData,bu);

		SyncHostWithDevice();
		hostKatzData.nActive = hostKatzData.activeQueue.getQueueEnd();
		SyncDeviceWithHost();

		allVinA_TraverseVertices<katzCentralityStreamingOperator::updatePrevWithCurr>(custing, deviceKatzData, hostKatzData.activeQueue.getQueue(), hostKatzData.nActive);

		SyncHostWithDevice();

		hostKatzData.iteration++;

		SyncDeviceWithHost();
	}
	if(hostKatzData.iteration>2)
		allVinA_TraverseVertices<katzCentralityStreamingOperator::updateLastIteration>(custing, deviceKatzData, hostKatzData.activeQueue.getQueue(), hostKatzData.nActive);
}


}// cuStingerAlgs namespace

