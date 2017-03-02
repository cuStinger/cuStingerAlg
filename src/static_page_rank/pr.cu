

	
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>

#include <iomanip> 

#include <cub.cuh>
#include <util_allocator.cuh>

#include <device/device_reduce.cuh>
#include <kernel_mergesort.hxx>


#include "update.hpp"
#include "cuStinger.hpp"

#include "operators.cuh"

#include "static_page_rank/pr.cuh"


using namespace cub;
using namespace mgpu;

CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

namespace cuStingerAlgs {

void StaticPageRank::Init(cuStinger& custing){
	hostPRData.nv = custing.nv;
	hostPRData.prevPR  = (prType*) allocDeviceArray(hostPRData.nv+1, sizeof(prType));
	hostPRData.currPR  = (prType*) allocDeviceArray(hostPRData.nv+1, sizeof(prType));
	hostPRData.absDiff = (prType*) allocDeviceArray(hostPRData.nv+1, sizeof(prType));
	hostPRData.contri = (prType*) allocDeviceArray(hostPRData.nv+1, sizeof(prType));

	hostPRData.reductionOut = (prType*) allocDeviceArray(1, sizeof(prType));
	// hostPRData.reduction=NULL;

	devicePRData = (pageRankData*)allocDeviceArray(1, sizeof(pageRankData));
	SyncDeviceWithHost();

	Reset();
}

void StaticPageRank::Reset(){
	hostPRData.iteration = 0;

	SyncDeviceWithHost();
}


void StaticPageRank::Release(){
	freeDeviceArray(devicePRData);
	freeDeviceArray(hostPRData.currPR);
	freeDeviceArray(hostPRData.prevPR);
	freeDeviceArray(hostPRData.absDiff);
	// freeDeviceArray(hostPRData.reduction);
	freeDeviceArray(hostPRData.reductionOut);
	freeDeviceArray(hostPRData.contri);
}

void StaticPageRank::Run(cuStinger& custing){
	// cusLoadBalance cusLB(custing);
	// cusLoadBalance cusLB(custing,true,false);
	cusLoadBalance cusLB(custing,false,true);
	cout << "The number of non zeros is : " << cusLB.currArrayLen << endl;

	allVinG_TraverseVertices<StaticPageRankOperator::init>(custing,devicePRData);
	hostPRData.iteration = 0;

	prType h_out = hostPRData.threshhold+1;

	while(hostPRData.iteration < hostPRData.iterationMax && h_out>hostPRData.threshhold){
		SyncDeviceWithHost();

		allVinA_TraverseVertices<StaticPageRankOperator::resetCurr>(custing,devicePRData,cusLB);
		allVinA_TraverseVertices<StaticPageRankOperator::computeContribuitionPerVertex>(custing,devicePRData,cusLB);
		allVinA_TraverseEdges_LB<StaticPageRankOperator::addContribuitionsUndirected>(custing,devicePRData,cusLB);
		// allVinA_TraverseEdges_LB<StaticPageRankOperator::addContribuitions>(custing,devicePRData,cusLB);
		allVinA_TraverseVertices<StaticPageRankOperator::dampAndDiffAndCopy>(custing,devicePRData,cusLB);

		// allVinG_TraverseVertices<StaticPageRankOperator::resetCurr>(custing,devicePRData);
		// allVinG_TraverseVertices<StaticPageRankOperator::computeContribuitionPerVertex>(custing,devicePRData);
		// allVinA_TraverseEdges_LB<StaticPageRankOperator::addContribuitionsUndirected>(custing,devicePRData,cusLB);
		// allVinG_TraverseVertices<StaticPageRankOperator::dampAndDiffAndCopy>(custing,devicePRData);

		// copyArrayDeviceToDevice(hostPRData.currPR,hostPRData.prevPR, hostPRData.nv,sizeof(prType));

		// allVinG_TraverseVertices<StaticPageRankOperator::print>(custing,d_out);

		allVinG_TraverseVertices<StaticPageRankOperator::sum>(custing,devicePRData);
		SyncHostWithDevice();

		copyArrayDeviceToHost(hostPRData.reductionOut,&h_out, 1, sizeof(prType));
		// h_out=hostPRData.threshhold+1;
		// cout << "The number of elements : " << hostPRData.nv << endl;

		hostPRData.iteration++;
	}
}

void StaticPageRank::setInputParameters(length_t prmIterationMax, prType prmThreshhold,prType prmDamp){
	hostPRData.iterationMax=prmIterationMax;
	hostPRData.threshhold=prmThreshhold;
	hostPRData.damp=prmDamp;
	hostPRData.normalizedDamp=(1-hostPRData.damp)/float(hostPRData.nv);
	SyncDeviceWithHost();
}

length_t StaticPageRank::getIterationCount(){
	return hostPRData.iteration;
}

void StaticPageRank::printRankings(cuStinger& custing){

	prType* d_scores = (prType*)allocDeviceArray(hostPRData.nv, sizeof(prType));
	vertexId_t* d_ids = (vertexId_t*)allocDeviceArray(hostPRData.nv, sizeof(vertexId_t));

	copyArrayDeviceToDevice(hostPRData.currPR, d_scores,hostPRData.nv, sizeof(prType));


	allVinG_TraverseVertices<StaticPageRankOperator::setIds>(custing,d_ids);

	standard_context_t context(false);
	mergesort(d_scores,d_ids,hostPRData.nv,greater_t<float>(),context);

	prType* h_scores = (prType*)allocHostArray(hostPRData.nv, sizeof(prType));
	vertexId_t* h_ids    = (vertexId_t*)allocHostArray(hostPRData.nv, sizeof(vertexId_t));

	copyArrayDeviceToHost(d_scores,h_scores,hostPRData.nv, sizeof(prType));
	copyArrayDeviceToHost(d_ids,h_ids,hostPRData.nv, sizeof(vertexId_t));

	for(int v=0; v<10; v++){
		printf("Pr[%d]:= %f\n",h_ids[v],h_scores[v]);
	}

	allVinG_TraverseVertices<StaticPageRankOperator::resetCurr>(custing,devicePRData);
	allVinG_TraverseVertices<StaticPageRankOperator::sumPr>(custing,devicePRData);

		// SyncHostWithDevice();
	prType h_out;

		copyArrayDeviceToHost(hostPRData.reductionOut,&h_out, 1, sizeof(prType));
		cout << "                     " << setprecision(9) << h_out << endl;


	freeDeviceArray(d_scores);
	freeDeviceArray(d_ids);
	freeHostArray(h_scores);
	freeHostArray(h_ids);
}

}// cuStingerAlgs namespace





// void StaticPageRank::Run(cuStinger& custing){
// 	// cusLoadBalance cusLB(custing);
// 	// cusLoadBalance cusLB(custing,true,false);
// 	cusLoadBalance cusLB(custing,false,true);
// 	cout << "The number of non zeros is : " << cusLB.currArrayLen << endl;

// 	allVinG_TraverseVertices<StaticPageRankOperator::init>(custing,devicePRData);
// 	hostPRData.iteration = 0;

// 	prType h_out = hostPRData.threshhold+1;

// 	while(hostPRData.iteration < hostPRData.iterationMax && h_out>hostPRData.threshhold){
// 		SyncDeviceWithHost();

// 		// allVinA_TraverseVertices<StaticPageRankOperator::resetCurr>(custing,devicePRData,cusLB);
// 		// allVinA_TraverseVertices<StaticPageRankOperator::computeContribuitionPerVertex>(custing,devicePRData,cusLB);
// 		// allVinA_TraverseEdges_LB<StaticPageRankOperator::addContribuitionsUndirected>(custing,devicePRData,cusLB);
// 		// allVinA_TraverseVertices<StaticPageRankOperator::dampAndDiffAndCopy>(custing,devicePRData,cusLB);

// 		// allVinG_TraverseVertices<StaticPageRankOperator::resetCurr>(custing,devicePRData);
// 		// allVinG_TraverseVertices<StaticPageRankOperator::computeContribuitionPerVertex>(custing,devicePRData);
// 		// allVinA_TraverseEdges_LB<StaticPageRankOperator::addContribuitionsUndirected>(custing,devicePRData,cusLB);
// 		// allVinG_TraverseVertices<StaticPageRankOperator::dampAndDiffAndCopy>(custing,devicePRData);

// 		// copyArrayDeviceToDevice(hostPRData.currPR,hostPRData.prevPR, hostPRData.nv,sizeof(prType));

//     // void            *d_temp_storage = NULL;
//     // size_t          temp_storage_bytes = 0;
//     // DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, (prType*)hostPRData.absDiff, (prType*)hostPRData.reductionOut, hostPRData.nv);
//     // d_temp_storage = (void*)allocDeviceArray(temp_storage_bytes,1);
//     // DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, (prType*)hostPRData.absDiff, (prType*)hostPRData.reductionOut, hostPRData.nv);


//     // freeDeviceArray(d_temp_storage);
//     // printf("\n%d\n", temp_storage_bytes);
//     // Run

//     // float* h_in = new float[10000];
//     // float  h_reference;
//     // for(int i=0;i<10000; i++)
//     // 	h_in[i]=(float)i/float(10000);
//     // float* d_referernce, *d_in;
//     // d_in=(float*)allocDeviceArray(10000,sizeof(float));
//     // d_referernce=(float*)allocDeviceArray(1,sizeof(float));
//     // copyArrayHostToDevice(h_in,d_in,10000,sizeof(float));

//     // void            *d_temp_storage = NULL;
//     // size_t          temp_storage_bytes = 0;
//     // DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_referernce, hostPRData.nv);
//     // d_temp_storage = (void*)allocDeviceArray(temp_storage_bytes,1);
//     // DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_referernce, hostPRData.nv);


//     // copyArrayDeviceToHost(d_referernce,&h_reference,1, sizeof(float));
//     // cout << "Hello MF "<<  h_reference << endl;

//     int* h_in = new int[10000];
//     int  h_reference;
//     for(int i=0;i<10000; i++)
//     	h_in[i]=(int)i;
//     int *d_out=NULL, *d_in=NULL;
//     // d_in=(int*)allocDeviceArray(10000,sizeof(int));
//     // d_out=(int*)allocDeviceArray(1,sizeof(int));
//     g_allocator.DeviceAllocate((void**)&d_in, sizeof(float) * 10000);
//     g_allocator.DeviceAllocate((void**)&d_out, sizeof(float) * 1);

//     copyArrayHostToDevice(h_in,d_in,10000,sizeof(int));

//     void            *d_temp_storage = NULL;
//     size_t          temp_storage_bytes = 0;
//     DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, 10000);
//     // d_temp_storage = (void*)allocDeviceArray(temp_storage_bytes,1);
//     g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);
//     DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, 10000);
//     DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, 10000);
//     DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, 10000);
//     DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, 10000);
//     DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, 10000);

    
// 	allVinG_TraverseVertices<StaticPageRankOperator::print>(custing,d_out);


//     copyArrayDeviceToHost(d_out,&h_reference,1, sizeof(int));
//     cout << "Hello MF "<<  h_reference << endl;
// 	// allVinG_TraverseVertices<StaticPageRankOperator::print>(custing,d_out);

//     // freeDeviceArray(d_temp_storage);
//     // freeDeviceArray(d_in);
//     // freeDeviceArray(d_out);
//     if (d_temp_storage) g_allocator.DeviceFree(d_temp_storage);
//     if (d_in) g_allocator.DeviceFree(d_in);
//     if (d_out) g_allocator.DeviceFree(d_out);


// 		// allVinG_TraverseVertices<StaticPageRankOperator::sum>(custing,devicePRData);

// 		SyncHostWithDevice();

// 		// copyArrayDeviceToHost(hostPRData.reductionOut,&h_out, 1, sizeof(prType));
// 		// h_out=hostPRData.threshhold+1;
// 		// cout << "The number of elements : " << hostPRData.nv << endl;

// 		hostPRData.iteration++;
// 	}
// }
