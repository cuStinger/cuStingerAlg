


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

void katzCentralityStreaming::setInitParameters(length_t maxIteration_, length_t K_,length_t maxDegree_){
	// setInitParameters(maxIteration_,maxDegree_,K_,false);
	katzCentrality::setInitParameters(maxIteration_,maxDegree_,K_,false);
}


void katzCentralityStreaming::Init(cuStinger& custing){
	// kcStatic.Init(custing);
	katzCentrality::Init(custing);
	// copyArrayHostToHost(&katzCentrality::hostKatzData,&hostKatzData, 1, sizeof(katzData));
	// copyArrayDeviceToDevice(katzCentrality::deviceKatzData,, deviceKatzData, sizeof(katzData));
	katzDataStreaming* replaceStaticHost = new katzDataStreaming();
	
	copyArrayHostToHost(katzCentrality::hostKatzData,replaceStaticHost, 1, sizeof(katzData));
	delete katzCentrality::hostKatzData;

	katzCentrality::hostKatzData = replaceStaticHost;
	katzDataStreaming* replaceStaticDevice = (katzDataStreaming*)allocDeviceArray(1, sizeof(katzDataStreaming));
	copyArrayDeviceToDevice(katzCentrality::deviceKatzData,replaceStaticDevice, 1, sizeof(katzData));
	freeDeviceArray(katzCentrality::deviceKatzData);	
	katzCentrality::deviceKatzData = replaceStaticDevice;


	// deviceKatzData = (katzDataStreaming*)allocDeviceArray(1, sizeof(katzDataStreaming));
	// hostKatzData.newPathsCurr = (ulong_t*) allocDeviceArray((hostKatzData.nv), sizeof(ulong_t));
	// hostKatzData.newPathsPrev = (ulong_t*) allocDeviceArray((hostKatzData.nv), sizeof(ulong_t));


	cusLB = new cusLoadBalance(custing);

	// SyncDeviceWithHost();


}


void katzCentralityStreaming::runStatic(cuStinger& custing){
	Run(custing);
	katzCentrality::Run(custing);
}

void katzCentralityStreaming::Release(){
	delete cusLB;
	// freeDeviceArray(deviceKatzData);

	// freeDeviceArray(hostKatzData.newPathsCurr);
	// freeDeviceArray(hostKatzData.newPathsPrev);
	// kcStatic.Release();
	katzCentrality::Release();
}


length_t katzCentralityStreaming::getIterationCount(){
	// SyncHostWithDevice();
	// return hostKatzData.iteration;
}


}// cuStingerAlgs namespace
