


#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>

#include <math.h>

#include "update.hpp"
#include "cuStinger.hpp"

#include "operators.cuh"

#include "static_katz_centrality/katz.cuh"
#include "streaming_katz_centrality/katz.cuh"

#include "kernel_mergesort.hxx"
using namespace mgpu;


namespace cuStingerAlgs {

void katzCentralityStreaming::setInitParameters(length_t K_,length_t maxDegree_, length_t maxIteration_){
	hostKatzData.K=K_;
	hostKatzData.maxDegree=maxDegree_;
	hostKatzData.maxIteration=maxIteration_;
	hostKatzData.alpha = 1.0/((double)hostKatzData.maxDegree+1.0);

	if(maxIteration_==0){
		cout << "Number of max iterations should be greater than zero" << endl;
		return;
	}
}


void katzCentralityStreaming::Init(cuStinger& custing){

	hostKatzData.newPathsCurr = (ulong_t*) allocDeviceArray(custing.nv+1, sizeof(ulong_t));
	hostKatzData.newPathsPrev = (ulong_t*) allocDeviceArray(custing.nv+1, sizeof(ulong_t));

	hostKatzData.KC         = (double*) allocDeviceArray(custing.nv+1, sizeof(double));
	hostKatzData.nPaths = (ulong_t**) allocDeviceArray(hostKatzData.maxIteration, sizeof(ulong_t*));
	hostKatzData.nPathsData = (ulong_t*) allocDeviceArray((custing.nv+1)*hostKatzData.maxIteration, sizeof(ulong_t));
	hostKatzData.alphaI         = (double*) allocDeviceArray(custing.nv+1, sizeof(double));

	double* hAlphiI = (double*)allocHostArray(hostKatzData.maxIteration, sizeof(double));
	ulong_t** hPathsPtr = (ulong_t**)allocHostArray(hostKatzData.maxIteration, sizeof(ulong_t*));
	hAlphiI[0]=hostKatzData.alpha;
	hPathsPtr[0] = hostKatzData.nPathsData;

	for(int i=1; i< hostKatzData.maxIteration; i++){
		hAlphiI[i]=hAlphiI[i-1]*hostKatzData.alpha;
		hPathsPtr[i] = (hostKatzData.nPathsData+(custing.nv+1)*i);
	}
	copyArrayHostToDevice(hAlphiI,hostKatzData.alphaI,hostKatzData.maxIteration,sizeof(double));
	copyArrayHostToDevice(hPathsPtr,hostKatzData.nPaths,hostKatzData.maxIteration,sizeof(double));

	freeHostArray(hAlphiI);
	freeHostArray(hPathsPtr);

	deviceKatzData = (katzDataStreaming*)allocDeviceArray(1, sizeof(katzDataStreaming));

	cusLB = new cusLoadBalance(custing);

	SyncDeviceWithHost();
	Reset();
}

void katzCentralityStreaming::Reset(){
	hostKatzData.iteration = 1;

	SyncDeviceWithHost();
	copyArrayHostToDevice(&hostKatzData,deviceKatzData,1, sizeof(katzDataStreaming));
}



void katzCentralityStreaming::Release(){
	// free(cusLB);
	delete cusLB;
	freeDeviceArray(deviceKatzData);

	freeDeviceArray(hostKatzData.newPathsCurr);
	freeDeviceArray(hostKatzData.newPathsPrev);

	freeDeviceArray(hostKatzData.nPaths);
	freeDeviceArray(hostKatzData.nPathsData);
	// freeDeviceArray(hostKatzData.vertexArray);
	freeDeviceArray(hostKatzData.KC);
	freeDeviceArray(hostKatzData.alphaI);



}


length_t katzCentralityStreaming::getIterationCount(){
	SyncHostWithDevice();
	return hostKatzData.iteration;
}


}// cuStingerAlgs namespace
