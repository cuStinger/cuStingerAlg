
	
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>


#include "update.hpp"
#include "cuStinger.hpp"

#include "macros.cuh"

#include "static_connected_components/cc.cuh"


namespace cuStingerAlgs {

void ccConcurrent::Init(cuStinger& custing){
	hostCCData.currState = (vertexId_t*) allocDeviceArray(custing.nv+1, sizeof(vertexId_t));
	// hostCCData.prevState = (vertexId_t*) allocDeviceArray(custing.nv+1, sizeof(vertexId_t));

	deviceCCData = (ccDataBaseline*)allocDeviceArray(1, sizeof(ccDataBaseline));
	SyncDeviceWithHost();
	Reset();
}

void ccConcurrent::Reset(){
	hostCCData.changeCurrIter=1;
	hostCCData.connectedComponentCount=0;
	hostCCData.iteration = 0;

	SyncDeviceWithHost();
}


void ccConcurrent::Release(){
	freeDeviceArray(deviceCCData);
	freeDeviceArray(hostCCData.currState);
	// freeDeviceArray(hostCCData.prevState);
}


void ccConcurrent::Run(cuStinger& custing){
	allVinG_TraverseVertices<StaticConnectedComponentsOperator::init>(custing,deviceCCData);
	
	hostCCData.iteration = 0;
	while(hostCCData.changeCurrIter){
		hostCCData.changeCurrIter=0;

		SyncDeviceWithHost();
		allEinG_TraverseEdges<StaticConnectedComponentsOperator::swapLocal>(custing,deviceCCData);
		allVinG_TraverseVertices<StaticConnectedComponentsOperator::shortcut>(custing,deviceCCData);
		allVinG_TraverseVertices<StaticConnectedComponentsOperator::shortcut>(custing,deviceCCData);
		SyncHostWithDevice();
		hostCCData.iteration++;
	}
}

length_t ccConcurrent::CountConnectComponents(cuStinger& custing){
	allVinG_TraverseVertices<StaticConnectedComponentsOperator::count>(custing,deviceCCData);
	SyncHostWithDevice();

	return hostCCData.connectedComponentCount;
}

// Includes connected components of size 1.
length_t ccConcurrent::CountConnectComponentsAll(cuStinger& custing){
	allVinG_TraverseVertices<StaticConnectedComponentsOperator::countAll>(custing,deviceCCData);
	SyncHostWithDevice();


	return hostCCData.connectedComponentCount;
}

length_t ccConcurrent::GetIterationCount(){
	return hostCCData.iteration;
}


}// cuStingerAlgs namespace
