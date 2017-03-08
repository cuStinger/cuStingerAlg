#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>


#include "utils.hpp"
#include "update.hpp"
#include "cuStinger.hpp"

#include "operators.cuh"

#include "static_breadth_first_search/bfs_bottom_up.cuh"

using namespace std;

namespace cuStingerAlgs {

void bfsBU::Init(cuStinger& custing){
	hostBfsData.nv = custing.nv;
	hostBfsData.queue.Init(custing.nv);

	hostBfsData.level = (vertexId_t*) allocDeviceArray(hostBfsData.nv, sizeof(vertexId_t));

	deviceBfsData = (bfsDataBU*)allocDeviceArray(1, sizeof(bfsDataBU));
	copyArrayHostToDevice(&hostBfsData,deviceBfsData,1, sizeof(bfsDataBU));

	Reset();
}

void bfsBU::Reset(){
	hostBfsData.queue.resetQueue();
	hostBfsData.currLevel=0;
	hostBfsData.verticesFound=0;

	copyArrayHostToDevice(&hostBfsData,deviceBfsData,1, sizeof(bfsDataBU));
}

void bfsBU::setInputParameters(vertexId_t root){
	hostBfsData.root = root;
}

void bfsBU::Release(){
	freeDeviceArray(deviceBfsData);
	freeDeviceArray(hostBfsData.level);
}

void bfsBU::Run(cuStinger& custing){

	cusLoadBalance cusLB(custing);

	allVinG_TraverseVertices<bfsBottomUpOperator::setLevelInfinity>(custing,deviceBfsData);
	hostBfsData.queue.enqueueFromHost(hostBfsData.root);

	SyncDeviceWithHost();
	copyArrayHostToDevice(&hostBfsData.currLevel,hostBfsData.level+hostBfsData.root,1,sizeof(length_t));

	length_t prevEnd=1; length_t foundThisIteration=1;

	while(foundThisIteration>0){

		allVinA_TraverseEdges_LB<bfsBottomUpOperator::bfsExpandFrontier>(custing,deviceBfsData,cusLB);

		SyncHostWithDevice();
		hostBfsData.queue.setQueueCurr(prevEnd);
		prevEnd = hostBfsData.queue.getQueueEnd();

		foundThisIteration = hostBfsData.verticesFound;
		hostBfsData.verticesFound=0;
		hostBfsData.currLevel++;
		SyncDeviceWithHost();
	}
}

length_t bfsBU::getElementsFound(cuStinger& custing){
	hostBfsData.verticesFound=0;
	SyncDeviceWithHost();
	allVinG_TraverseVertices<bfsBottomUpOperator::countFound>(custing,deviceBfsData	);
	SyncHostWithDevice();
	return hostBfsData.verticesFound;
}


} // cuStingerAlgs namespace 