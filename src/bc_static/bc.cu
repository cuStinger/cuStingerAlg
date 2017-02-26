#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>


#include "utils.hpp"
#include "update.hpp"
#include "cuStinger.hpp"

#include "macros.cuh"

#include "bc_static/bc.cuh"


using namespace std;

namespace cuStingerAlgs {


void StaticBC::Init(cuStinger& custing)
{
	hostBcStaticData.nv = custing.nv;
	hostBcStaticData.queue.Init(custing.nv);

	host_deltas = new float[custing.nv];

	// TODO: Maybe replace with depth array *d?
	hostBcStaticData.level = (vertexId_t*) allocDeviceArray(hostBcStaticData.nv, sizeof(vertexId_t));

	hostBcStaticData.d = (int*) allocDeviceArray(hostBcStaticData.nv, sizeof(int));
	hostBcStaticData.sigma = (unsigned long long*) allocDeviceArray(hostBcStaticData.nv, sizeof(long));
	hostBcStaticData.delta = (float*) allocDeviceArray(hostBcStaticData.nv, sizeof(float));

	// on the host only
	hostBcStaticData.offsets = new int[hostBcStaticData.nv];

	deviceBcStaticData = (bcStaticData*) allocDeviceArray(1, sizeof(bcStaticData));
	copyArrayHostToDevice(&hostBcStaticData, deviceBcStaticData, 1, sizeof(bcStaticData));

	Reset();
}


void StaticBC::Reset()
{
	hostBcStaticData.queue.resetQueue();
	hostBcStaticData.currLevel = 0;

	// initialize all offsets to zero
	for (int i = 0; i < hostBcStaticData.nv; i++)
	{
		hostBcStaticData.offsets[i] = 0;
	}

	SyncDeviceWithHost();
}

// Must pass in a root node vertex id, and a pointer to bc values (of length custing.nv)
void StaticBC::setInputParameters(vertexId_t root, float *bc_array)
{
	hostBcStaticData.root = root;
	bc = bc_array;
}

void StaticBC::Release()
{
	freeDeviceArray(deviceBcStaticData);
	freeDeviceArray(hostBcStaticData.level);

	freeDeviceArray(hostBcStaticData.d);
	freeDeviceArray(hostBcStaticData.sigma);
	freeDeviceArray(hostBcStaticData.delta);

	delete[] hostBcStaticData.offsets;
	delete[] host_deltas;
}


void StaticBC::Run(cuStinger& custing)
{
	RunBfsTraversal(custing);
	DependencyAccumulation(custing);
}

void StaticBC::RunBfsTraversal(cuStinger& custing)
{

	cusLoadBalance cusLB(hostBcStaticData.nv);

	// Clear out array values first
	allVinG_TraverseVertices<bcOperator::clearArrays>(custing,deviceBcStaticData);

	allVinG_TraverseVertices<bcOperator::setLevelInfinity>(custing,deviceBcStaticData);
	hostBcStaticData.queue.enqueueFromHost(hostBcStaticData.root);

	SyncDeviceWithHost();
	// set level[root] <- 0
	copyArrayHostToDevice(&hostBcStaticData.currLevel,
		hostBcStaticData.level+hostBcStaticData.root, 1, sizeof(length_t));

	// set sigma[root] <- 1
	int one = 1;
	copyArrayHostToDevice(&one,
		hostBcStaticData.sigma+hostBcStaticData.root, 1, sizeof(length_t));

	length_t prevEnd = 1;
	hostBcStaticData.offsets[0] = 1;
	while( hostBcStaticData.queue.getActiveQueueSize() > 0)
	{

		allVinA_TraverseEdges_LB<bcOperator::bcExpandFrontier>(custing, 
			deviceBcStaticData,cusLB,hostBcStaticData.queue);

		SyncHostWithDevice();  // update host

		// Update cumulative offsets from start of queue

		hostBcStaticData.queue.setQueueCurr(prevEnd);
		
		vertexId_t level = getLevel();
		hostBcStaticData.offsets[level + 1] = hostBcStaticData.queue.getActiveQueueSize() + hostBcStaticData.offsets[level];
		
		prevEnd = hostBcStaticData.queue.getQueueEnd();

		hostBcStaticData.currLevel++;
		SyncDeviceWithHost();  // update device
	}
}


void StaticBC::DependencyAccumulation(cuStinger& custing)
{
	// for load balancing
	cusLoadBalance cusLB(hostBcStaticData.nv);

	// Iterate backwards through depths, starting from 2nd deepest frontier
	
	// Begin with the 2nd deepest frontier as the active queue
	hostBcStaticData.currLevel -= 2;
	SyncDeviceWithHost();

	while (getLevel() >= 0)
	{
		length_t start = hostBcStaticData.offsets[getLevel()];
		length_t end = hostBcStaticData.offsets[getLevel() + 1];
		
		// // set queue start and end so the queue holds all nodes in one frontier
		hostBcStaticData.queue.setQueueCurr(start);
		hostBcStaticData.queue.setQueueEnd(end);
		hostBcStaticData.queue.SyncDeviceWithHost();
		SyncDeviceWithHost();

		// Now, run the macro for all outbound edges over this queue
		allVinA_TraverseEdges_LB<bcOperator::dependencyAccumulation>(custing, deviceBcStaticData, cusLB, hostBcStaticData.queue);
		
		SyncHostWithDevice();

		hostBcStaticData.currLevel -= 1;
		SyncDeviceWithHost();
	}

	// Now, copy over delta values to host
	copyArrayDeviceToHost(hostBcStaticData.delta, host_deltas, hostBcStaticData.nv, sizeof(float));

	// // Finally, update the bc values
	for (vertexId_t w = 0; w < hostBcStaticData.nv; w++)
	{
		if (w != hostBcStaticData.root)
		{
			bc[w] += host_deltas[w];
		}
	}
}


} // cuStingerAlgs namespace 