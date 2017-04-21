#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>

#include <math.h>


#include "utils.hpp"
#include "update.hpp"
#include "cuStinger.hpp"

#include "algs.cuh"

#include "static_katz_centrality/katz.cuh"
#include "streaming_katz_centrality/katz.cuh"

using namespace cuStingerAlgs;


#define CUDA(call, ...) do {                        \
        cudaError_t _e = (call);                    \
        if (_e == cudaSuccess) break;               \
        fprintf(stdout,                             \
                "CUDA runtime error: %s (%d)\n",    \
                cudaGetErrorString(_e), _e);        \
        return -1;                                  \
    } while (0)


void generateEdgeUpdates(length_t nv, length_t numEdges, vertexId_t* edgeSrc, vertexId_t* edgeDst){
	for(int32_t e=0; e<numEdges; e++){
		edgeSrc[e] = rand()%nv;
		edgeDst[e] = rand()%nv;
	}
}



int main(const int argc, char *argv[]){
	int device=0;
    cudaSetDevice(device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
 
    length_t nv, ne,*off;
    vertexId_t *adj;

	bool isDimacs,isSNAP,isRmat=false,isMarket;
	string filename(argv[1]);
	isDimacs = filename.find(".graph")==std::string::npos?false:true;
	isSNAP   = filename.find(".txt")==std::string::npos?false:true;
	isRmat 	 = filename.find("kron")==std::string::npos?false:true;
	isMarket = filename.find(".mtx")==std::string::npos?false:true;

	if(isDimacs){
	    readGraphDIMACS(argv[1],&off,&adj,&nv,&ne,isRmat);
	}
	else if(isSNAP){
	    readGraphSNAP(argv[1],&off,&adj,&nv,&ne,isRmat);
	}
	else if(isMarket){
		readGraphMatrixMarket(argv[1],&off,&adj,&nv,&ne,(isRmat)?false:true);
	}
	else{ 
		cout << "Unknown graph type" << endl;
	}

	cout << "Vertices: " << nv << "    Edges: " << ne << endl;

	cudaEvent_t ce_start,ce_stop;
	cuStinger custing(defaultInitAllocater,defaultUpdateAllocater);

	cuStingerInitConfig cuInit;
	cuInit.initState =eInitStateCSR;
	cuInit.maxNV = nv+1;
	cuInit.useVWeight = false;
	cuInit.isSemantic = false;  // Use edge types and vertex types
	cuInit.useEWeight = false;
	// CSR data
	cuInit.csrNV 			= nv;
	cuInit.csrNE	   		= ne;
	cuInit.csrOff 			= off;
	cuInit.csrAdj 			= adj;
	cuInit.csrVW 			= NULL;
	cuInit.csrEW			= NULL;

	custing.initializeCuStinger(cuInit);

	
	float totalTime;

	// Finding largest vertex
	vertexId_t maxV=0;
	length_t   maxLen=0;
	for(int v=1; v<nv;v++){
		if((off[v+1]-off[v])>maxLen){
			maxV=v;
			maxLen=off[v+1]-off[v];
		}
	}
	katzCentrality kc;
	kc.setInitParameters(20,100,maxLen,true);
	kc.Init(custing);
	kc.Reset();
	start_clock(ce_start, ce_stop);
	kc.Run(custing);
	totalTime = end_clock(ce_start, ce_stop);
	cout << "The number of iterations      : " << kc.getIterationCount() << endl;
	cout << "Total time for KC             : " << totalTime << endl; 
	cout << "Average time per iteartion    : " << totalTime/(float)kc.getIterationCount() << endl; 
	kc.Release();

	katzCentralityStreaming kcs;

	kcs.setInitParameters(20,100,maxLen);
	kcs.Init(custing);
	start_clock(ce_start, ce_stop);
	kcs.runStatic(custing);
	totalTime = end_clock(ce_start, ce_stop);
	cout << "The number of iterations      : " << kcs.getIterationCount() << endl;
	cout << "Total time for KC             : " << totalTime << endl; 
	cout << "Average time per iteartion    : " << totalTime/(float)kcs.getIterationCount() << endl; 

	int numBatchEdges=1;

	BatchUpdateData bud(numBatchEdges,true);

	generateEdgeUpdates(nv, numBatchEdges, bud.getSrc(),bud.getDst());

	// BatchUpdate bu(bud);
	BatchUpdate* bu = new BatchUpdate(bud);

	start_clock(ce_start, ce_stop);
	kcs.insertedBatchUpdate(custing,*bu);
	totalTime = end_clock(ce_start, ce_stop);


	// cout << "The number of iterations      : " << kcs.getIterationCount() << endl;
	cout << "Total time for KC streaming   : " << totalTime << endl; 
	// cout << "Average time per iteartion    : " << totalTime/(float)kcs.getIterationCount() << endl; 


	katzCentrality kcPostUpdate;
	kcPostUpdate.setInitParameters(20,100,maxLen,true);
	kcPostUpdate.Init(custing);
	kcPostUpdate.Reset();
	start_clock(ce_start, ce_stop);
	kcPostUpdate.Run(custing);
	totalTime = end_clock(ce_start, ce_stop);
	cout << "The number of iterations      : " << kcPostUpdate.getIterationCount() << endl;
	cout << "Total time for KC             : " << totalTime << endl; 
	cout << "Average time per iteartion    : " << totalTime/(float)kcPostUpdate.getIterationCount() << endl; 


	double* kcScoresStreaming  = (double*) allocHostArray(custing.nv, sizeof(double));
	double* kcScoresPostUpdate = (double*) allocHostArray(custing.nv, sizeof(double));

	kcs.copyKCToHost(kcScoresStreaming);
	kcPostUpdate.copyKCToHost(kcScoresPostUpdate);

	for(int i=0; i < 100; i++){
		// printf("%1.11lf, ", kcScoresStreaming[i]-kcScoresPostUpdate[i]);
	}
	printf("\n");

	double sum=0.0;
	for(int i=0; i < custing.nv; i++){
		sum += fabs(kcScoresStreaming[i]-kcScoresPostUpdate[i]);
	}
	printf("Sum of difference %4.11lf \n", sum);


	kcPostUpdate.Release();
	kcs.Release();

	custing.freecuStinger();

	free(off);
	free(adj);

	cudaDeviceReset();
    return 0;	
}

