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

	deviceBcStaticData = (bfsData*) allocDeviceArray(1, sizeof(bfsData));
	copyArrayHostToDevice(&hostBcStaticData, deviceBcStaticData, 1, sizeof(bfsData));

	Reset();
}


void StaticBC::Reset()
{
	hostBcStaticData.queue.resetQueue();
	hostBcStaticData.currLevel = 0;

	copyArrayHostToDevice(&hostBcStaticData, deviceBcStaticData, 1, sizeof(bfsData));
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


} // cuStingerAlgs namespace 