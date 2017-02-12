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
	hostBcStaticData.sigma = (long*) allocDeviceArray(hostBcStaticData.nv, sizeof(long));
	hostBcStaticData.delta = (float*) allocDeviceArray(hostBcStaticData.nv, sizeof(float));

	deviceBcStaticData = (bcStaticData*) allocDeviceArray(1, sizeof(bcStaticData));
	copyArrayHostToDevice(&hostBcStaticData, deviceBcStaticData, 1, sizeof(bcStaticData));

	Reset();
}


void StaticBC::Reset()
{
	hostBcStaticData.queue.resetQueue();
	hostBcStaticData.currLevel = 0;

	copyArrayHostToDevice(&hostBcStaticData, deviceBcStaticData, 1, sizeof(bcStaticData));
}


void StaticBC::setInputParameters(vertexId_t root)
{
	hostBcStaticData.root = root;
}


void StaticBC::Release()
{
	freeDeviceArray(deviceBcStaticData);
	freeDeviceArray(hostBcStaticData.level);
}


void StaticBC::Run(cuStinger& custing)
{

	cusLoadBalance cusLB(hostBcStaticData.nv);

	allVinG_TraverseVertices<bcOperator::setLevelInfinity>(custing,deviceBcStaticData);
	hostBcStaticData.queue.enqueueFromHost(hostBcStaticData.root);

	SyncDeviceWithHost();
	copyArrayHostToDevice(&hostBcStaticData.currLevel,
		hostBcStaticData.level+hostBcStaticData.root, 1, sizeof(length_t));

	length_t prevEnd = 1;
	while( hostBcStaticData.queue.getActiveQueueSize() > 0)
	{

		allVinA_TraverseEdges_LB<bcOperator::bcExpandFrontier>(custing,
			deviceBcStaticData,cusLB,hostBcStaticData.queue);

		SyncHostWithDevice();
		hostBcStaticData.queue.setQueueCurr(prevEnd);
		prevEnd = hostBcStaticData.queue.getQueueEnd();

		hostBcStaticData.currLevel++;
		SyncDeviceWithHost();
	}
}

// __global__ void StaticBC::DependencyAccumulation(cuStinger& custing, float *bc)
// {
// 	// We want to traverse backwards from Queue
// 	// Walk back from the queue in reverse
// 	// vertexQueue vxQueue = hostBcStaticData.queue;
	
// 	// will be length nv
// 	vertexId_t *vxQueue = new vertexId_t[custing.nv];
// 	// copy all data over from device
// 	copyArrayDeviceToHost(hostBcStaticData.queue.getQueue(), vxQueue, custing.nv, sizeof(vertexId_t));

// 	// vxQueue.setQueueCurr(0);  // 0 is the position of the cu
// 	// vertexId_t *start = vxQueue.getQueue();
// 	// vertexId_t *end = vxQueue.getQueue() + vxQueue.getQueueEnd() - 1;

// 	vertexId_t nv = custing.nv;

// 	vertexId_t* level = new vertexId_t[nv];
// 	long *sigma = new long[nv];
// 	float *delta = new float[nv];

// 	copyArrayDeviceToHost(hostBcStaticData.level, level, nv, sizeof(vertexId_t));
// 	copyArrayDeviceToHost(hostBcStaticData.sigma, sigma, nv, sizeof(long));
// 	copyArrayDeviceToHost(hostBcStaticData.delta, delta, nv, sizeof(float));

// 	printf("Begin Dep accumulation\n");

// 	int idx = custing.nv - 1;

// 	// Keep iterating backwards in the queue
// 	while (idx >= 0)
// 	{
// 		// Look at adjacencies for this vertex at end
// 		vertexId_t w = vxQueue[idx];
// 		printf("Looking at all neighbors of vertex %d\n", w);
// 		length_t numNeighbors = (custing.getHostVertexData()->used)[w];
// 		printf("Num neighbors: %d\n", numNeighbors);
// 		if (numNeighbors > 0)
// 		{
// 			// Get adjacency list
// 			printf("Get adjacency list\n");
// 			cuStinger::cusEdgeData *adj = (custing.getHostVertexData()->adj)[w];
// 			for(int k = 0; k < numNeighbors; k++)
// 			{
// 				// neighbord v of w from the adjacency list
// 				vertexId_t v = adj->dst[k];
// 				// if depth is less than depth of w
// 				if (level[v] == level[w] + 1)
// 				{
// 					printf("{%d} is a neighbor of {%d} at depth +1\n", v, w);
// 					delta[v] += (delta[v] / delta[w]) * (1 + delta[w]);
// 				}
// 			}
// 		}

// 		// Now, put values into bc[]
// 		if (w != hostBcStaticData.root)
// 		{
// 			bc[w] += delta[w];
// 		}

// 		idx--;
// 	}

// 	delete[] level;
// 	delete[] sigma;
// 	delete[] delta;
// }

void StaticBC::DependencyAccumulation(cuStinger& custing, float *bc)
{
	// We want to traverse backwards from Queue
	// Walk back from the queue in reverse
	// vertexQueue vxQueue = hostBcStaticData.queue;
	
	// will be length nv
	// vertexId_t *vxQueue = new vertexId_t[custing.nv];
	// // copy all data over from device
	// copyArrayDeviceToHost(hostBcStaticData.queue.getQueue(), vxQueue, custing.nv, sizeof(vertexId_t));

	// // vxQueue.setQueueCurr(0);  // 0 is the position of the cu
	// // vertexId_t *start = vxQueue.getQueue();
	// // vertexId_t *end = vxQueue.getQueue() + vxQueue.getQueueEnd() - 1;

	// vertexId_t nv = custing.nv;

	// vertexId_t* level = new vertexId_t[nv];
	// long *sigma = new long[nv];
	// float *delta = new float[nv];

	// copyArrayDeviceToHost(hostBcStaticData.level, level, nv, sizeof(vertexId_t));
	// copyArrayDeviceToHost(hostBcStaticData.sigma, sigma, nv, sizeof(long));
	// copyArrayDeviceToHost(hostBcStaticData.delta, delta, nv, sizeof(float));

	// printf("Begin Dep accumulation\n");

	// int idx = custing.nv - 1;

	// // Keep iterating backwards in the queue
	// while (idx >= 0)
	// {
	// 	// Look at adjacencies for this vertex at end
	// 	vertexId_t w = vxQueue[idx];
	// 	printf("Looking at all neighbors of vertex %d\n", w);
	// 	length_t numNeighbors = (custing.getHostVertexData()->used)[w];
	// 	printf("Num neighbors: %d\n", numNeighbors);
	// 	if (numNeighbors > 0)
	// 	{
	// 		// Get adjacency list
	// 		printf("Get adjacency list\n");
	// 		cuStinger::cusEdgeData *adj = (custing.getHostVertexData()->adj)[w];
	// 		for(int k = 0; k < numNeighbors; k++)
	// 		{
	// 			// neighbord v of w from the adjacency list
	// 			vertexId_t v = adj->dst[k];
	// 			// if depth is less than depth of w
	// 			if (level[v] == level[w] + 1)
	// 			{
	// 				printf("{%d} is a neighbor of {%d} at depth +1\n", v, w);
	// 				delta[v] += (delta[v] / delta[w]) * (1 + delta[w]);
	// 			}
	// 		}
	// 	}

	// 	// Now, put values into bc[]
	// 	if (w != hostBcStaticData.root)
	// 	{
	// 		bc[w] += delta[w];
	// 	}

	// 	idx--;
	// }

	// delete[] level;
	// delete[] sigma;
	// delete[] delta;

	float *dev_bc;
	vertexId_t *queue = hostBcStaticData.queue.getQueue();
	checkCudaErrors(cudaMalloc(&dev_bc, sizeof(float) * custing.nv));
	hostDependencyAccumulation<<<1, 1>>>(custing.devicePtr(), deviceBcStaticData, queue, dev_bc);

	float *delta = new float[custing.nv];
	copyArrayDeviceToHost(dev_bc, delta, custing.nv, sizeof(float));


	for (int i = 0; i < custing.nv; i++)
	{
		bc[i] += delta[i];
	}

	// Free host mem
	delete[] delta;
	// free cuda mem
	checkCudaErrors(cudaFree(dev_bc));
}


__global__ void hostDependencyAccumulation(cuStinger *custing, bcStaticData *deviceBcStaticData, vertexId_t *queue, float *dev_bc)
{
	printf("Global dep accum\n");
	deviceDependencyAccumulation(custing, deviceBcStaticData, queue, dev_bc);
}

__device__ void deviceDependencyAccumulation(cuStinger* custing, bcStaticData *deviceBcStaticData, vertexId_t *queue, float *bc)
{
	printf("Device dep accum\n");
	// Iterate backwards over queue
	vertexId_t nv = custing->nv;
	int idx = nv - 1;


	vertexId_t* level = deviceBcStaticData->level;
	long *sigma = deviceBcStaticData->sigma;
	float *delta = deviceBcStaticData->delta;


	while (idx >= 0)
	{
		vertexId_t w = queue[idx];

		printf("Looking at vertex w: {%d}\n", w);
		// Look at all neighbors

		length_t numNeighbors = (custing->dVD->used)[w];
		printf("Num neighbors: %d\n", numNeighbors);
		if (numNeighbors > 0)
		{
			// Get adjacency list
			printf("Get adjacency list\n");
			cuStinger::cusEdgeData *adj = (custing->dVD->adj)[w];
			for(int k = 0; k < numNeighbors; k++)
			{
				// neighbord v of w from the adjacency list
				vertexId_t v = adj->dst[k];
				// if depth is less than depth of w
				if (level[v] == level[w] + 1)
				{
					printf("{%d} is a neighbor of {%d} at depth +1\n", v, w);
					delta[v] += (delta[v] / delta[w]) * (1 + delta[w]);
				}
			}
		}

		// // Now, put values into bc
		if (w != deviceBcStaticData->root)
		{
			bc[w] = delta[w];
		}

		idx--;
	}
}

} // cuStingerAlgs namespace 