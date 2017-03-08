#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>


#include "utils.hpp"
#include "update.hpp"
#include "cuStinger.hpp"

#include "operators.cuh"

#include "static_breadth_first_search/bfs_hybrid.cuh"

using namespace std;

namespace cuStingerAlgs {


void bfsHybrid::Init(cuStinger& custing){
	hostBfsData.nv = custing.nv;
	hostBfsData.activeQueue.Init(custing.nv);
	hostBfsData.unMarkedQueue.Init(custing.nv);

	hostBfsData.level = (vertexId_t*) allocDeviceArray(hostBfsData.nv, sizeof(vertexId_t));

	deviceBfsData = (bfsDataHybrid*)allocDeviceArray(1, sizeof(bfsDataHybrid));
	SyncDeviceWithHost();
	Reset();
}

void bfsHybrid::Reset(){
	hostBfsData.activeQueue.resetQueue();
	hostBfsData.unMarkedQueue.resetQueue();
	hostBfsData.currLevel=0;
	hostBfsData.verticesFound=0;
	hostBfsData.edgesTraversed=0;
	SyncDeviceWithHost();
}

void bfsHybrid::setInputParameters(vertexId_t root,int alpha,int beta){
	hostBfsData.root  = root;
	hostBfsData.alpha = alpha;
	hostBfsData.beta  = beta;
	SyncDeviceWithHost();
}

void bfsHybrid::Release(){
	freeDeviceArray(deviceBfsData);
	freeDeviceArray(hostBfsData.level);
}


void bfsHybrid::Run(cuStinger& custing){
	cusLoadBalance cusLB(hostBfsData.nv);
	cusLoadBalance cusLBAllVertices2(custing);

	allVinG_TraverseVertices<bfsHybridOperator::getRootSize>(custing,deviceBfsData);
	allVinG_TraverseVertices<bfsHybridOperator::countEdges>(custing,deviceBfsData);
	SyncHostWithDevice();

	allVinG_TraverseVertices<bfsHybridOperator::setLevelInfinity>(custing,deviceBfsData);
	hostBfsData.activeQueue.enqueueFromHost(hostBfsData.root);
	SyncDeviceWithHost();


	copyArrayHostToDevice(&hostBfsData.currLevel,hostBfsData.level+hostBfsData.root,1,sizeof(length_t));

	length_t foundThisIteration=1;
	length_t sum=0;
	length_t edgesToCheck = hostBfsData.ne;
	length_t scoutCount = hostBfsData.rootSize;

	// cout << "The initial size of edgesToCheck  " << edgesToCheck << endl;

	length_t prevEnd=1;
	while((hostBfsData.activeQueue.getActiveQueueSize())>0 && (scoutCount <	 edgesToCheck / hostBfsData.alpha)){

		allVinA_TraverseEdges_LB<bfsHybridOperator::bfsExpandFrontierTD>(custing,deviceBfsData,cusLB,hostBfsData.activeQueue);

		SyncHostWithDevice();
		scoutCount=hostBfsData.edgesTraversed;
		edgesToCheck -= scoutCount;
		hostBfsData.edgesTraversed=0;

		// cout << hostBfsData.currLevel << " Edges to check " <<  edgesToCheck << "  scoutCount " << scoutCount << endl;
		hostBfsData.activeQueue.setQueueCurr(prevEnd);
		prevEnd = hostBfsData.activeQueue.getQueueEnd();

		hostBfsData.currLevel++;
		SyncDeviceWithHost();
	}
	if(hostBfsData.activeQueue.getActiveQueueSize()==0)
		return;
	
	int64_t awakeCount = prevEnd, oldAwakeCount;

	cusLoadBalance cusLBAllVertices(custing);
	do {
		oldAwakeCount = awakeCount;
		allVinA_TraverseEdges_LB<bfsHybridOperator::bfsExpandFrontierBU>(custing,deviceBfsData,cusLBAllVertices);

		SyncHostWithDevice();
		awakeCount=hostBfsData.verticesFound;
		hostBfsData.verticesFound=0;
		hostBfsData.currLevel++;
		SyncDeviceWithHost();
	} while ((awakeCount >= oldAwakeCount) || (awakeCount > hostBfsData.nv / hostBfsData.beta));

	if(awakeCount==0)
		return;

	hostBfsData.activeQueue.resetQueue();
	SyncDeviceWithHost();
	allVinG_TraverseVertices<bfsHybridOperator::createActiveFrontierQueue>(custing,deviceBfsData);
	SyncHostWithDevice();

	// cout << "Queue size just before the 3rd phase "<<hostBfsData.activeQueue.getActiveQueueSize() << endl;

	prevEnd = hostBfsData.activeQueue.getActiveQueueSize();

	while((hostBfsData.activeQueue.getActiveQueueSize())>0){

		allVinA_TraverseEdges_LB<bfsHybridOperator::bfsExpandFrontierTD>(custing,deviceBfsData,cusLB,hostBfsData.activeQueue);

		SyncHostWithDevice();
		scoutCount=hostBfsData.edgesTraversed;
		edgesToCheck -= scoutCount;
		hostBfsData.edgesTraversed=0;

		// cout << hostBfsData.currLevel << " Edges to check " <<  edgesToCheck << "  scoutCount " << scoutCount << endl;
		hostBfsData.activeQueue.setQueueCurr(prevEnd);
		prevEnd = hostBfsData.activeQueue.getQueueEnd();

		hostBfsData.currLevel++;
		SyncDeviceWithHost();
	}
	// if(hostBfsData.activeQueue.getActiveQueueSize()==0)
	// 	return;



}


length_t bfsHybrid::getElementsFound(cuStinger& custing){
	hostBfsData.verticesFound=0;
	SyncDeviceWithHost();
	allVinG_TraverseVertices<bfsHybridOperator::countFound>(custing,deviceBfsData	);
	SyncHostWithDevice();
	return hostBfsData.verticesFound;
}


} // cuStingerAlgs namespace 