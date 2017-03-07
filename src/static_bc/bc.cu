#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>


#include "utils.hpp"
#include "update.hpp"
#include "cuStinger.hpp"

#include "operators.cuh"
#include "algs.cuh"
#include "static_bc/bc.cuh"
#include "static_bc/bc_tree.cuh"


using namespace std;

namespace cuStingerAlgs {


void StaticBC::Init(cuStinger& custing)
{
	hostBcTree = createHostBcTree(custing.nv);
	hostBcTree->nv = custing.nv;
	hostBcTree->queue.Init(custing.nv);

	// this sets hostBcTree's pointers for d, sigma, delta and copies its contents
	// to deviceBcTree on the device
	deviceBcTree = createDeviceBcTree(custing.nv, hostBcTree);
	
	host_deltas = new float[custing.nv];
	cusLB = new cusLoadBalance(custing.nv);
	Reset();
}


void StaticBC::Reset()
{
	hostBcTree->queue.resetQueue();
	hostBcTree->currLevel = 0;

	// initialize all offsets to zero
	for (int i = 0; i < hostBcTree->nv; i++)
	{
		hostBcTree->offsets[i] = 0;
	}

	SyncDeviceWithHost();
}

// Must pass in a root node vertex id, and a pointer to bc values (of length custing.nv)
void StaticBC::setInputParameters(vertexId_t root, float *bc_array)
{
	hostBcTree->root = root;
	bc = bc_array;
}

void StaticBC::Release()
{
	delete cusLB;
	destroyDeviceBcTree(deviceBcTree);
	destroyHostBcTree(hostBcTree);

	delete[] host_deltas;
}


void StaticBC::Run(cuStinger& custing)
{
	RunBfsTraversal(custing);
	DependencyAccumulation(custing);
}

void StaticBC::RunBfsTraversal(cuStinger& custing)
{
	// Clear out array values first
	allVinG_TraverseVertices<bcOperator::clearArrays>(custing,deviceBcTree);

	allVinG_TraverseVertices<bcOperator::setLevelInfinity>(custing,deviceBcTree);
	hostBcTree->queue.enqueueFromHost(hostBcTree->root);

	SyncDeviceWithHost();

	// set d[root] <- 0
	int zero = 0;
	copyArrayHostToDevice(&zero, hostBcTree->d + hostBcTree->root,
		1, sizeof(length_t));

	// set sigma[root] <- 1
	int one = 1;
	copyArrayHostToDevice(&one, hostBcTree->sigma + hostBcTree->root,
		1, sizeof(length_t));

	length_t prevEnd = 1;
	hostBcTree->offsets[0] = 1;
	
	while( hostBcTree->queue.getActiveQueueSize() > 0)
	{

		allVinA_TraverseEdges_LB<bcOperator::bcExpandFrontier>(custing, 
			deviceBcTree,*cusLB,hostBcTree->queue);
		SyncHostWithDevice();

		// Update cumulative offsets from start of queue
		hostBcTree->queue.setQueueCurr(prevEnd);
		
		vertexId_t level = getLevel();
		hostBcTree->offsets[level + 1] = hostBcTree->queue.getActiveQueueSize() + hostBcTree->offsets[level];
		
		prevEnd = hostBcTree->queue.getQueueEnd();

		hostBcTree->currLevel++;
		SyncDeviceWithHost();
	}
}


void StaticBC::DependencyAccumulation(cuStinger& custing)
{
	// Iterate backwards through depths, starting from 2nd deepest frontier
	// Begin with the 2nd deepest frontier as the active queue
	hostBcTree->currLevel -= 2;
	SyncDeviceWithHost();

	while (getLevel() >= 0)
	{
		length_t start = hostBcTree->offsets[getLevel()];
		length_t end = hostBcTree->offsets[getLevel() + 1];
		
		// // set queue start and end so the queue holds all nodes in one frontier
		hostBcTree->queue.setQueueCurr(start);
		hostBcTree->queue.setQueueEnd(end);
		hostBcTree->queue.SyncDeviceWithHost();
		SyncDeviceWithHost();

		// Now, run the macro for all outbound edges over this queue
		allVinA_TraverseEdges_LB<bcOperator::dependencyAccumulation>(custing, deviceBcTree, *cusLB, hostBcTree->queue);
		
		SyncHostWithDevice();

		hostBcTree->currLevel -= 1;
		SyncDeviceWithHost();
	}

	// Now, copy over delta values to host
	copyArrayDeviceToHost(hostBcTree->delta, host_deltas, hostBcTree->nv, sizeof(float));

	// // Finally, update the bc values
	for (vertexId_t w = 0; w < hostBcTree->nv; w++)
	{
		if (w != hostBcTree->root)
		{
			bc[w] += host_deltas[w];
		}
	}
}


} // cuStingerAlgs namespace 