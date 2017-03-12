#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>

#include <math.h>


#include "utils.hpp"
#include "update.hpp"
#include "cuStinger.hpp"

#include "algs.cuh"

#include "static_breadth_first_search/bfs_top_down.cuh"
// #include "static_breadth_first_search/bfs_bottom_up.cuh"
// #include "static_breadth_first_search/bfs_hybrid.cuh"
#include "static_connected_components/cc.cuh"
#include "static_page_rank/pr.cuh"
#include "static_betweenness_centrality/bc.cuh"


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

	ccBaseline scc;
	scc.Init(custing);
	scc.Reset();
	start_clock(ce_start, ce_stop);
//	scc.Run(custing);
	totalTime = end_clock(ce_start, ce_stop);
	// cout << "The number of iterations           : " << scc.GetIterationCount() << endl;
	// cout << "The number of connected-compoents  : " << scc.CountConnectComponents(custing) << endl;
	// cout << "Total time for connected-compoents : " << totalTime << endl; 
	scc.Release();

	ccConcurrent scc2;
	scc2.Init(custing);
	scc2.Reset();
	start_clock(ce_start, ce_stop);
    // scc2.Run(custing);
	totalTime = end_clock(ce_start, ce_stop);
	// cout << "The number of iterations           : " << scc2.GetIterationCount() << endl;
	// cout << "The number of connected-compoents  : " << scc2.CountConnectComponents(custing) << endl;
	// cout << "Total time for connected-compoents : " << totalTime << endl; 
	scc2.Release();


	ccConcurrentLB scc3;
	scc3.Init(custing);
	scc3.Reset();
	start_clock(ce_start, ce_stop);
	scc3.Run(custing);
	totalTime = end_clock(ce_start, ce_stop);
	cout << "The number of iterations           : " << scc3.GetIterationCount() << endl;
	cout << "The number of connected-compoents  : " << scc3.CountConnectComponents(custing) << endl;
	cout << "Total time for connected-compoents : " << totalTime << endl; 
	scc3.Release();


	// ccConcurrentOptimized scc4;
	// scc4.Init(custing);
	// scc4.Reset();
	// start_clock(ce_start, ce_stop);
	// scc4.Run(custing);
	// totalTime = end_clock(ce_start, ce_stop);
	// cout << "The number of iterations           : " << scc4.GetIterationCount() << endl;
	// cout << "The number of connected-compoents  : " << scc4.CountConnectComponents(custing) << endl;
	// cout << "Total time for connected-compoents : " << totalTime << endl; 
	// scc4.Release();

	// Finding largest vertex

	vertexId_t maxV=0;
	length_t   maxLen=0;
	for(int v=1; v<nv;v++){
		if((off[v+1]-off[v])>maxLen){
			maxV=v;
			maxLen=off[v+1]-off[v];
		}
	}
	// cout << "Largest vertex is: " << maxV << "   With the length of :" << maxLen << endl;

	bfsTD bfs;
	bfs.Init(custing);
	bfs.Reset();
	bfs.setInputParameters(maxV);
	start_clock(ce_start, ce_stop);
	bfs.Run(custing);
	totalTime = end_clock(ce_start, ce_stop);

	cout << "The number of levels          : " << bfs.getLevels() << endl;
	cout << "The number of elements found  : " << bfs.getElementsFound() << endl;
	cout << "Total time for BFS - Top-Down : " << totalTime << endl; 

	bfs.Release();

	// bfsBU bfsbu;
	// bfsbu.Init(custing);
	// bfsbu.Reset();
	// bfsbu.setInputParameters(maxV);
	// start_clock(ce_start, ce_stop);
	// bfsbu.Run(custing);
	// totalTime = end_clock(ce_start, ce_stop);

	// cout << "The number of levels          : " << bfsbu.getLevels() << endl;
	// cout << "The number of elements found  : " << bfsbu.getElementsFound(custing) << endl;
	// cout << "Total time for BFS - Bottom-up: " << totalTime << endl; 

	// bfsbu.Release();

	// bfsHybrid bfsHy;
	// bfsHy.Init(custing);
	// bfsHy.Reset();
	// bfsHy.setInputParameters(maxV);
	// start_clock(ce_start, ce_stop);
	// bfsHy.Run(custing);
	// totalTime = end_clock(ce_start, ce_stop);

	// cout << "The number of levels          : " << bfsHy.getLevels() << endl;
	// cout << "The number of elements found  : " << bfsHy.getElementsFound(custing) << endl;
	// cout << "Total time for BFS - Hybrid   : " << totalTime << endl; 

	// bfsHy.Release();



	StaticPageRank pr;

	pr.Init(custing);
	pr.Reset();
	pr.setInputParameters(5,0.001);
	start_clock(ce_start, ce_stop);
	pr.Run(custing);
	totalTime = end_clock(ce_start, ce_stop);
	cout << "The number of iterations      : " << pr.getIterationCount() << endl;
	cout << "Total time for pagerank       : " << totalTime << endl; 
	cout << "Average time per iteartion    : " << totalTime/(float)pr.getIterationCount() << endl; 
	// pr.printRankings(custing);

	pr.Release();


	StaticPageRank pr2;// =new StaticPageRank();

	pr2.Init(custing);
	pr2.Reset();
	pr2.setInputParameters(5,0.001);
	start_clock(ce_start, ce_stop);
	pr2.Run(custing);
	totalTime = end_clock(ce_start, ce_stop);
	// cout << "The number of iterations      : " << pr2.getIterationCount() << endl;
	// cout << "Total time for pagerank       : " << totalTime << endl; 
	// cout << "Average time per iteartion    : " << totalTime/(float)pr2.getIterationCount() << endl; 
	// pr2.printRankings(custing);

	pr2.Release();


	float *bc = new float[nv];
	for (int k = 0; k < nv; k++)
	{
		bc[k] = 0;
	}
	StaticBC sbc(bc);
	sbc.Init(custing);
	sbc.Reset();

	start_clock(ce_start, ce_stop);
	sbc.Run(custing);

	totalTime = end_clock(ce_start, ce_stop);
	cout << "Total time for Static Betweenness Centrality: " << totalTime << endl;

	sbc.Reset();
	sbc.Release();
	delete[] bc;




	custing.freecuStinger();

	free(off);
	free(adj);
    return 0;	
}

