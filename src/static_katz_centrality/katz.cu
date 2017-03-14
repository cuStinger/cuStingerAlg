


#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>

#include <math.h>

#include "update.hpp"
#include "cuStinger.hpp"

#include "operators.cuh"

#include "static_katz_centrality/katz.cuh"

#include "kernel_mergesort.hxx"
using namespace mgpu;


namespace cuStingerAlgs {

void katzCentrality::Init(cuStinger& custing){
	hostKatzData.nPathsCurr = (length_t*) allocDeviceArray(custing.nv+1, sizeof(length_t));
	hostKatzData.nPathsPrev = (length_t*) allocDeviceArray(custing.nv+1, sizeof(length_t));
	hostKatzData.vertexArray = (vertexId_t*) allocDeviceArray(custing.nv+1, sizeof(vertexId_t));
	hostKatzData.KC         = (double*) allocDeviceArray(custing.nv+1, sizeof(double));
	hostKatzData.lowerBound = (double*) allocDeviceArray(custing.nv+1, sizeof(double));
	hostKatzData.upperBound = (double*) allocDeviceArray(custing.nv+1, sizeof(double));

	deviceKatzData = (katzData*)allocDeviceArray(1, sizeof(katzData));

	cusLB = new cusLoadBalance(custing);

	SyncDeviceWithHost();
	Reset();
}

void katzCentrality::Reset(){
	hostKatzData.iteration = 1;

	SyncDeviceWithHost();
	copyArrayHostToDevice(&hostKatzData,deviceKatzData,1, sizeof(katzData));
}


void katzCentrality::Release(){
	// free(cusLB);
	delete cusLB;
	freeDeviceArray(deviceKatzData);
	freeDeviceArray(hostKatzData.nPathsCurr);
	freeDeviceArray(hostKatzData.nPathsPrev);
	freeDeviceArray(hostKatzData.vertexArray);
	freeDeviceArray(hostKatzData.KC);
	freeDeviceArray(hostKatzData.lowerBound);
	freeDeviceArray(hostKatzData.upperBound);
}

void katzCentrality::Run(cuStinger& custing){

	allVinG_TraverseVertices<katzCentralityOperator::initNumPaths>(custing,deviceKatzData);

	// GET MAX DEGREE
	standard_context_t context(false);

	hostKatzData.iteration = 1;
	
	hostKatzData.nActive = custing.nv;
	while(hostKatzData.nActive  > hostKatzData.K ){

		hostKatzData.alphaI = pow(hostKatzData.alpha,hostKatzData.iteration);
		hostKatzData.upperBoundConst = pow(hostKatzData.alpha,hostKatzData.iteration+1)/((1.0-hostKatzData.alpha*(double)hostKatzData.maxDegree));
		hostKatzData.lowerBoundConst = pow(hostKatzData.alpha,hostKatzData.iteration+1)/((1.0-hostKatzData.alpha));
		SyncDeviceWithHost();

		allVinG_TraverseVertices<katzCentralityOperator::initNumPathsPerIteration>(custing,deviceKatzData);
		allVinA_TraverseEdges_LB<katzCentralityOperator::updatePathCount>(custing,deviceKatzData,*cusLB);
		allVinG_TraverseVertices<katzCentralityOperator::updateKatzAndBounds>(custing,deviceKatzData);

		SyncHostWithDevice();
		hostKatzData.iteration++;

		length_t* temp = hostKatzData.nPathsCurr; hostKatzData.nPathsCurr=hostKatzData.nPathsPrev; hostKatzData.nPathsPrev=temp;
		SyncDeviceWithHost();

		mergesort(hostKatzData.lowerBound,hostKatzData.vertexArray,custing.nv, less_t<double>(),context);

		// TODO I don't know when I need to sync the device with the host
		hostKatzData.nActive = 0;
		allVinG_TraverseVertices<katzCentralityOperator::countActive>(custing,deviceKatzData);

	}
}




}// cuStingerAlgs namespace
