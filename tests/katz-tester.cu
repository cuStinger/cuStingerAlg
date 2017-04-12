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

	for (int r=0; r<1; r++){
		katzCentrality kc;
		kc.setInitParameters(20,true);
		kc.Init(custing);
		kc.Reset();
		kc.setInputParameters(100,maxLen);
		start_clock(ce_start, ce_stop);
		kc.Run(custing);
		totalTime = end_clock(ce_start, ce_stop);
		cout << "The number of iterations      : " << kc.getIterationCount() << endl;
		cout << "Total time for KC             : " << totalTime << endl; 
		cout << "Average time per iteartion    : " << totalTime/(float)kc.getIterationCount() << endl; 
		kc.Release();
	}

	for (int r=0; r<1; r++){

		katzCentrality kc2;
		kc2.setInitParameters(20,false);
		kc2.Init(custing);
		kc2.Reset();
		kc2.setInputParameters(100,maxLen);
		start_clock(ce_start, ce_stop);
		kc2.Run(custing);
		totalTime = end_clock(ce_start, ce_stop);
		cout << "The number of iterations      : " << kc2.getIterationCount() << endl;
		cout << "Total time for KC             : " << totalTime << endl; 
		cout << "Average time per iteartion    : " << totalTime/(float)kc2.getIterationCount() << endl; 

		kc2.Release();
	}
	// katzCentralityStreaming kcs;

	// kcs.setInitParameters(100,maxLen,20);
	// kcs.Init(custing);
	// kcs.Reset();
	// start_clock(ce_start, ce_stop);
	// // kcs.Run(custing);
	// totalTime = end_clock(ce_start, ce_stop);
	// // cout << "The number of iterations      : " << kc.getIterationCount() << endl;
	// // cout << "Total time for KC             : " << totalTime << endl; 
	// // cout << "Average time per iteartion    : " << totalTime/(float)kc.getIterationCount() << endl; 

	// kcs.Release();




	custing.freecuStinger();

	free(off);
	free(adj);
    return 0;	
}

