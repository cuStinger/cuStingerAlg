

	
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>


#include "update.hpp"
#include "cuStinger.hpp"

#include "operators.cuh"

#include "static_connected_components/cc.cuh"


namespace cuStingerAlgs {

void ccConcurrentLB::Init(cuStinger& custing){
	hostCCData.currState = (vertexId_t*) allocDeviceArray(custing.nv+1, sizeof(vertexId_t));

	deviceCCData = (ccDataBaseline*)allocDeviceArray(1, sizeof(ccDataBaseline));

	cusLB = new cusLoadBalance(custing);

	SyncDeviceWithHost();
	Reset();
}

void ccConcurrentLB::Reset(){
	hostCCData.changeCurrIter=1;
	hostCCData.connectedComponentCount=0;
	hostCCData.iteration = 0;

	SyncDeviceWithHost();
	copyArrayHostToDevice(&hostCCData,deviceCCData,1, sizeof(ccDataBaseline));
}


void ccConcurrentLB::Release(){
	free(cusLB);
	freeDeviceArray(deviceCCData);
	freeDeviceArray(hostCCData.currState);
}

void ccConcurrentLB::Run(cuStinger& custing){
	cusLoadBalance cusLB(custing);


//	// cusLoadBalance cusLB(custing,true,true);

	allVinG_TraverseVertices<StaticConnectedComponentsOperator::init>(custing,deviceCCData);
	hostCCData.iteration = 0;
	while(hostCCData.changeCurrIter){
		hostCCData.changeCurrIter=0;

		SyncDeviceWithHost();
		// allVinA_TraverseEdges_LB<StaticConnectedComponentsOperator::swapLocal>(custing,deviceCCData,*cusLB);
		allVinA_TraverseEdges_LB<StaticConnectedComponentsOperator::swapLocal>(custing,deviceCCData,cusLB);
		allVinG_TraverseVertices<StaticConnectedComponentsOperator::shortcut>(custing,deviceCCData);
		allVinG_TraverseVertices<StaticConnectedComponentsOperator::shortcut>(custing,deviceCCData);
		SyncHostWithDevice();

		hostCCData.iteration++;
	}
}


length_t ccConcurrentLB::CountConnectComponents(cuStinger& custing){
	allVinG_TraverseVertices<StaticConnectedComponentsOperator::count>(custing,deviceCCData);
	SyncHostWithDevice();

	return hostCCData.connectedComponentCount;
}

// Includes connected components of size 1.
length_t ccConcurrentLB::CountConnectComponentsAll(cuStinger& custing){
	allVinG_TraverseVertices<StaticConnectedComponentsOperator::countAll>(custing,deviceCCData);
	SyncHostWithDevice();


	return hostCCData.connectedComponentCount;
}

length_t ccConcurrentLB::GetIterationCount(){
	return hostCCData.iteration;
}


}// cuStingerAlgs namespace

