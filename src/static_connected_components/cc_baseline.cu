	
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>


#include "update.hpp"
#include "cuStinger.hpp"

#include "macros.cuh"

#include "static_connected_components/cc.cuh"


namespace cuStingerAlgs {

void ccBaseline::Init(cuStinger& custing){
	hostCCData.currState = (vertexId_t*) allocDeviceArray(custing.nv+1, sizeof(vertexId_t));
	hostCCData.prevState = (vertexId_t*) allocDeviceArray(custing.nv+1, sizeof(vertexId_t));

	deviceCCData = (ccDataBaseline*)allocDeviceArray(1, sizeof(ccDataBaseline));
	SyncDeviceWithHost();
	Reset();
}

void ccBaseline::Reset(){
	hostCCData.changeCurrIter=1;
	hostCCData.connectedComponentCount=0;
	hostCCData.iteration = 0;

	SyncDeviceWithHost();
	copyArrayHostToDevice(&hostCCData,deviceCCData,1, sizeof(ccDataBaseline));
}


void ccBaseline::Release(){
	freeDeviceArray(deviceCCData);
	freeDeviceArray(hostCCData.currState);
	freeDeviceArray(hostCCData.prevState);
}


void ccBaseline::Run(cuStinger& custing){
	allVinG_TraverseVertices<StaticConnectedComponentsOperator::init>(custing,deviceCCData);
	copyArrayDeviceToDevice(hostCCData.currState,hostCCData.prevState,custing.nv,sizeof(vertexId_t));
	
	hostCCData.iteration = 0;
	while(hostCCData.changeCurrIter){
		hostCCData.changeCurrIter=0;
		SyncDeviceWithHost();
		allEinG_TraverseEdges<StaticConnectedComponentsOperator::swapWithPrev>(custing,deviceCCData);
		allVinG_TraverseVertices<StaticConnectedComponentsOperator::shortcut>(custing,deviceCCData);
		allVinG_TraverseVertices<StaticConnectedComponentsOperator::shortcut>(custing,deviceCCData);
		SyncHostWithDevice();

		hostCCData.iteration++;
		// Swapping pointers
		if(hostCCData.changeCurrIter){
			vertexId_t* temp     = hostCCData.prevState;
			hostCCData.prevState = hostCCData.currState;
			hostCCData.currState = temp;
			SyncDeviceWithHost();
		}
	}
}

length_t ccBaseline::CountConnectComponents(cuStinger& custing){
	allVinG_TraverseVertices<StaticConnectedComponentsOperator::count>(custing,deviceCCData);
	SyncHostWithDevice();

	return hostCCData.connectedComponentCount;
}

// Includes connected components of size 1.
length_t ccBaseline::CountConnectComponentsAll(cuStinger& custing){
	allVinG_TraverseVertices<StaticConnectedComponentsOperator::countAll>(custing,deviceCCData);
	SyncHostWithDevice();


	return hostCCData.connectedComponentCount;
}

length_t ccBaseline::GetIterationCount(){
	return hostCCData.iteration;
}

}// cuStingerAlgs namespace
