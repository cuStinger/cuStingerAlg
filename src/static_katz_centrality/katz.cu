


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

void katzCentrality::setInitParameters(length_t maxIteration_, bool isStatic_){
	hostKatzData.maxIteration=maxIteration_;
	isStatic = isStatic_;

	if(maxIteration_==0){
		cout << "Number of max iterations should be greater than zero" << endl;
		return;
	}
}


void katzCentrality::Init(cuStinger& custing){

	hostKatzData.nv = custing.nv;

	if(isStatic==true){
		hostKatzData.nPathsData = (ulong_t*) allocDeviceArray(2*(hostKatzData.nv), sizeof(ulong_t));
		hostKatzData.nPathsPrev = hostKatzData.nPathsData;
		hostKatzData.nPathsCurr = hostKatzData.nPathsData+(hostKatzData.nv);
	}
	else{
		hostKatzData.nPathsData = (ulong_t*) allocDeviceArray((hostKatzData.nv)*hostKatzData.maxIteration, sizeof(ulong_t));
		hostKatzData.nPaths = (ulong_t**) allocDeviceArray(hostKatzData.maxIteration, sizeof(ulong_t*));

		// Allocating 
		hPathsPtr = (ulong_t**)allocHostArray(hostKatzData.maxIteration, sizeof(ulong_t*));
		for(int i=0; i< hostKatzData.maxIteration; i++){
			hPathsPtr[i] = (hostKatzData.nPathsData+(hostKatzData.nv)*i);
		}
		hostKatzData.nPathsPrev = hPathsPtr[0];
		hostKatzData.nPathsCurr = hPathsPtr[1];

		copyArrayHostToDevice(hPathsPtr,hostKatzData.nPaths,hostKatzData.maxIteration,sizeof(double));
	}

	hostKatzData.vertexArray = (vertexId_t*) allocDeviceArray(hostKatzData.nv, sizeof(vertexId_t));
	hostKatzData.KC         = (double*) allocDeviceArray(hostKatzData.nv, sizeof(double));
	hostKatzData.lowerBound = (double*) allocDeviceArray(hostKatzData.nv, sizeof(double));
	hostKatzData.lowerBoundSort = (double*) allocDeviceArray(hostKatzData.nv, sizeof(double));
	hostKatzData.upperBound = (double*) allocDeviceArray(hostKatzData.nv, sizeof(double));

	deviceKatzData = (katzData*)allocDeviceArray(1, sizeof(katzData));
	cusLB = new cusLoadBalance(custing);

	SyncDeviceWithHost();
	Reset();
}

void katzCentrality::Reset(){
	hostKatzData.iteration = 1;

	if(isStatic==true){
		hostKatzData.nPathsPrev = hostKatzData.nPathsData;
		hostKatzData.nPathsCurr = hostKatzData.nPathsData+(hostKatzData.nv);
	}
	else{
		hostKatzData.nPathsPrev = hPathsPtr[0];
		hostKatzData.nPathsCurr = hPathsPtr[1];
	}

	SyncDeviceWithHost();
	copyArrayHostToDevice(&hostKatzData,deviceKatzData,1, sizeof(katzData));
}

void katzCentrality::setInputParameters(length_t K_,length_t maxDegree_){
	hostKatzData.K=K_;
	hostKatzData.maxDegree=maxDegree_;
	hostKatzData.alpha = 1.0/((double)hostKatzData.maxDegree+1.0);
}


void katzCentrality::Release(){
	delete cusLB;
	freeDeviceArray(deviceKatzData);
	freeDeviceArray(hostKatzData.nPathsData);

	if (!isStatic){
		freeDeviceArray(hostKatzData.nPaths);
		freeHostArray(hPathsPtr);
	}

	freeDeviceArray(hostKatzData.vertexArray);
	freeDeviceArray(hostKatzData.KC);
	freeDeviceArray(hostKatzData.lowerBound);
	freeDeviceArray(hostKatzData.lowerBoundSort);
	freeDeviceArray(hostKatzData.upperBound);
}

void katzCentrality::Run(cuStinger& custing){

	allVinG_TraverseVertices<katzCentralityOperator::init>(custing,deviceKatzData);
	// allVinG_TraverseVertices<katzCentralityOperator::printKID>(custing,deviceKatzData);
	// printf("\n");
	standard_context_t context(false);

	hostKatzData.iteration = 1;
	
	hostKatzData.nActive = hostKatzData.nv;
	while(hostKatzData.nActive  > hostKatzData.K && hostKatzData.iteration < hostKatzData.maxIteration){

		hostKatzData.alphaI          = pow(hostKatzData.alpha,hostKatzData.iteration);
		hostKatzData.lowerBoundConst = pow(hostKatzData.alpha,hostKatzData.iteration+1)/((1.0-hostKatzData.alpha));
		hostKatzData.upperBoundConst = pow(hostKatzData.alpha,hostKatzData.iteration+1)/((1.0-hostKatzData.alpha*(double)hostKatzData.maxDegree));

		//cout << hostKatzData.iteration << " " << hostKatzData.alphaI << " " << hostKatzData.lowerBoundConst << " " << hostKatzData.upperBoundConst << endl;
		SyncDeviceWithHost();

		allVinG_TraverseVertices<katzCentralityOperator::initNumPathsPerIteration>(custing,deviceKatzData);
		allVinA_TraverseEdges_LB<katzCentralityOperator::updatePathCount>(custing,deviceKatzData,*cusLB);
		allVinG_TraverseVertices<katzCentralityOperator::updateKatzAndBounds>(custing,deviceKatzData);

		SyncHostWithDevice();
		hostKatzData.iteration++;

		if(isStatic){
			// printf("^\n");
			// Swapping pointers.
			ulong_t* temp = hostKatzData.nPathsCurr; hostKatzData.nPathsCurr=hostKatzData.nPathsPrev; hostKatzData.nPathsPrev=temp;	
		// printf("prev  - %p\n ", hostKatzData.nPathsPrev);
		// printf("curr  - %p\n ", hostKatzData.nPathsCurr);
		// return;
			// copyArrayDeviceToDevice(hostKatzData.nPathsCurr,hostKatzData.nPathsPrev,hostKatzData.nv, sizeof(ulong_t));
		}else{
			// printf("@\n");
			hostKatzData.nPathsPrev = hPathsPtr[hostKatzData.iteration - 1];
			hostKatzData.nPathsCurr = hPathsPtr[hostKatzData.iteration - 0];
		}
		// printf("prev  - %p\n ", hostKatzData.nPathsPrev);
		// printf("curr  - %p\n ", hostKatzData.nPathsCurr);


		hostKatzData.nActive = 0;
		SyncDeviceWithHost();

		mergesort(hostKatzData.lowerBoundSort,hostKatzData.vertexArray,custing.nv, greater_t<double>(),context);

		allVinG_TraverseVertices<katzCentralityOperator::countActive>(custing,deviceKatzData);
		//allVinA_TraverseVertices<katzCentralityOperator::printKID>(custing,deviceKatzData,hostKatzData.vertexArray, custing.nv);


/* 	ulong_t* nPathsCurr = (ulong_t*) allocHostArray(hostKatzData.nv, sizeof(ulong_t));
	ulong_t* nPathsPrev = (ulong_t*) allocHostArray(hostKatzData.nv, sizeof(ulong_t));
	vertexId_t* vertexArray = (vertexId_t*) allocHostArray(hostKatzData.nv, sizeof(vertexId_t));
	double* KC         = (double*) allocHostArray(hostKatzData.nv, sizeof(double));
	double* lowerBound = (double*) allocHostArray(hostKatzData.nv, sizeof(double));
	double* upperBound = (double*) allocHostArray(hostKatzData.nv, sizeof(double));
    
	copyArrayDeviceToHost(hostKatzData.lowerBound,lowerBound,custing.nv, sizeof(double)) ;
	copyArrayDeviceToHost(hostKatzData.upperBound,upperBound,custing.nv, sizeof(double)) ;
	copyArrayDeviceToHost(hostKatzData.KC,KC,custing.nv, sizeof(double)) ;
	copyArrayDeviceToHost(hostKatzData.vertexArray,vertexArray,custing.nv, sizeof(vertexId_t)) ;

//	for (int i=0; i<10; i++){
//	  printf("%d : katz = %g    lower = %g    upper=%g\n",vertexArray[i], KC[vertexArray[i]],lowerBound[vertexArray[i]],upperBound[vertexArray[i]]);
//	}

  	freeHostArray(nPathsCurr);
	freeHostArray(nPathsPrev);
    freeHostArray(vertexArray);
	freeHostArray(KC);
    freeHostArray(lowerBound);
	freeHostArray(upperBound);
*/		
		SyncHostWithDevice();
		cout << hostKatzData.nActive << endl;
	}
}

length_t katzCentrality::getIterationCount(){
	SyncHostWithDevice();
	return hostKatzData.iteration;
}


}// cuStingerAlgs namespace
