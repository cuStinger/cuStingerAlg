#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>


#include "utils.hpp"
#include "update.hpp"
#include "cuStinger.hpp"

#include "macros.cuh"

#include "static_breadth_first_search/bfs.cuh"

// #include "load_balance.cuh"


using namespace std;

namespace cuStingerAlgs {


void StaticBreadthFirstSearch::Init(cuStinger& custing){
	hostBfsData.nv = custing.nv;
	hostBfsData.queue.Init(custing.nv);

	hostBfsData.level = (vertexId_t*) allocDeviceArray(hostBfsData.nv, sizeof(vertexId_t));

	deviceBfsData = (bfsData*)allocDeviceArray(1, sizeof(bfsData));
	copyArrayHostToDevice(&hostBfsData,deviceBfsData,1, sizeof(bfsData));

	Reset();
}

void StaticBreadthFirstSearch::Reset(){
	hostBfsData.queue.resetQueue();
	hostBfsData.currLevel=0;

	copyArrayHostToDevice(&hostBfsData,deviceBfsData,1, sizeof(bfsData));
}

void StaticBreadthFirstSearch::setInputParameters(vertexId_t root){
	hostBfsData.root = root;
}



void StaticBreadthFirstSearch::Release(){
	freeDeviceArray(deviceBfsData);
	freeDeviceArray(hostBfsData.level);
}


void StaticBreadthFirstSearch::Run(cuStinger& custing){

	cusLoadBalance cusLB(hostBfsData.nv);

	allVinG_TraverseVertices<bfsOperator::setLevelInfinity>(custing,deviceBfsData);
	hostBfsData.queue.enqueueFromHost(hostBfsData.root);

	SyncDeviceWithHost();
	copyArrayHostToDevice(&hostBfsData.currLevel,hostBfsData.level+hostBfsData.root,1,sizeof(length_t));

	length_t prevEnd=1;
	while((hostBfsData.queue.getActiveQueueSize())>0){

		allVinA_TraverseEdges_LB<bfsOperator::bfsExpandFrontier>(custing,deviceBfsData,cusLB,hostBfsData.queue);

		SyncHostWithDevice();
		hostBfsData.queue.setQueueCurr(prevEnd);
		prevEnd = hostBfsData.queue.getQueueEnd();

		hostBfsData.currLevel++;
		SyncDeviceWithHost();
	}
}



} // cuStingerAlgs namespace 