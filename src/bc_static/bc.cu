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

	// TODO: Maybe replace with depth array *d?
	hostBcStaticData.level = (vertexId_t*) allocDeviceArray(hostBcStaticData.nv, sizeof(vertexId_t));

	hostBcStaticData.d = (int*) allocDeviceArray(hostBcStaticData.nv, sizeof(int));
	hostBcStaticData.sigma = (unsigned long long*) allocDeviceArray(hostBcStaticData.nv, sizeof(long));
	hostBcStaticData.delta = (float*) allocDeviceArray(hostBcStaticData.nv, sizeof(float));

	// on the host only
	hostBcStaticData.offsets = new int[hostBcStaticData.nv];
	// initialize to all zeros
	for (int i = 0; i < custing.nv; i++)
	{
		hostBcStaticData.offsets[i] = 0;
	}

	printf("custing.nv size: --> %d\n", custing.nv);

	deviceBcStaticData = (bcStaticData*) allocDeviceArray(1, sizeof(bcStaticData));
	copyArrayHostToDevice(&hostBcStaticData, deviceBcStaticData, 1, sizeof(bcStaticData));

	Reset();
}


void StaticBC::Reset()
{
	hostBcStaticData.queue.resetQueue();
	hostBcStaticData.currLevel = 0;

	// re-initialize to all zeros
	for (int i = 0; i < hostBcStaticData.nv; i++)
	{
		hostBcStaticData.offsets[i] = 0;
	}

	SyncDeviceWithHost();
}


void StaticBC::setInputParameters(vertexId_t root)
{
	hostBcStaticData.root = root;
}


void StaticBC::Release()
{
	freeDeviceArray(deviceBcStaticData);
	freeDeviceArray(hostBcStaticData.level);

	freeDeviceArray(hostBcStaticData.d);
	freeDeviceArray(hostBcStaticData.sigma);
	freeDeviceArray(hostBcStaticData.delta);

	delete[] hostBcStaticData.offsets;
}


void StaticBC::Run(cuStinger& custing)
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
	while( hostBcStaticData.queue.getActiveQueueSize() > 0)
	{

		allVinA_TraverseEdges_LB<bcOperator::bcExpandFrontier>(custing, 
			deviceBcStaticData,cusLB,hostBcStaticData.queue);


		if (hostBcStaticData.currLevel == 0)
		{
			// Frontier 0 is always size 1 because it's just the root
			hostBcStaticData.offsets[hostBcStaticData.currLevel] = 1;
		} else {
			// Update cumulative offsets from start of queue
			vertexId_t level = hostBcStaticData.currLevel;
			hostBcStaticData.offsets[level] = hostBcStaticData.queue.getActiveQueueSize() + hostBcStaticData.offsets[level - 1];
		}

		SyncHostWithDevice();  // update host
		
		hostBcStaticData.queue.setQueueCurr(prevEnd);
		prevEnd = hostBcStaticData.queue.getQueueEnd();

		hostBcStaticData.currLevel++;
		SyncDeviceWithHost();  // update device
	}
}


void StaticBC::DependencyAccumulation(cuStinger& custing, float *delta_copy, float *bc)
{
	// for load balancing
	cusLoadBalance cusLB(hostBcStaticData.nv);

	// printf("Original currLevel: %d\n", hostBcStaticData.currLevel);

	// for (int i = 0; i < hostBcStaticData.nv; i++)
	// {
	// 	if (hostBcStaticData.offsets[i] > 0 || i < 2)
	// 	{
	// 		printf("LEVEL: %d-->%d\n", i, hostBcStaticData.offsets[i]);
	// 	}
	// }

	// // copy sigmas over
	// unsigned long long *h_sigma = new unsigned long long[custing.nv];
	// copyArrayDeviceToHost(hostBcStaticData.sigma, h_sigma, hostBcStaticData.nv, sizeof(unsigned long long));
	// for (vertexId_t i = 0; i < custing.nv; i++)
	// {
	// 	printf("[%d]: %llu\n", i, h_sigma[i]);
	// }

	// delete[] h_sigma;
	// return;

	// Iterate backwards through depths
	// Begin with the 2nd deepest frontier as the active queue
	hostBcStaticData.currLevel -= 2;
	// hostBcStaticData.currLevel = -1;
	// SyncHostWithDevice();
	// printf("New currLevel: %d\n", hostBcStaticData.currLevel);

	while (hostBcStaticData.currLevel >= -1)
	{
		length_t start;
		if (hostBcStaticData.currLevel >= 0)
		{
			start = hostBcStaticData.offsets[hostBcStaticData.currLevel];
		} else
		{
			// If looking at the 1st frontier (which only contains root),
			// starting position is always index 0 in the queue
			start = 0;
		}
		length_t end = hostBcStaticData.offsets[hostBcStaticData.currLevel + 1];

		// printf("S: %d\tE: %d\n", start, end);
		
		// set queue start and end so the queue holds all nodes in one frontier
		hostBcStaticData.queue.setQueueCurr(start);
		hostBcStaticData.queue.setQueueEnd(end);
		SyncDeviceWithHost();

		// Now, run the macro for all outbound edges over this queue
		allVinA_TraverseEdges_LB<bcOperator::dependencyAccumulation>(custing, deviceBcStaticData, cusLB, hostBcStaticData.queue);
		
		hostBcStaticData.currLevel -= 1;
		SyncDeviceWithHost();
	}

	// Now, copy over delta values to host
	copyArrayDeviceToHost(hostBcStaticData.delta, delta_copy, hostBcStaticData.nv, sizeof(float));

	// Finally, update the bc values
	for (vertexId_t w = 0; w < hostBcStaticData.nv; w++)
	{
		if (w != hostBcStaticData.root)
		{
			bc[w] += delta_copy[w];
		}
		if (delta_copy[w] > 1)
		{
			// printf("GREATER THAN 1 =====> idx: %d\tval:%f\n", w, delta_copy[w]);
		}
	}

	// printf("Done with bc vals\n");

}


} // cuStingerAlgs namespace 