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
		// printf("Batch update: (#%d) (%d %d)\n", e,edgeSrc[e],edgeDst[e]);
	}
}
void generateEdgeUpdatesInverted(length_t nv, length_t numEdges, vertexId_t* edgeSrc, vertexId_t* edgeDst){
	for(int32_t e=0; e<numEdges; e++){
		edgeDst[e] = rand()%nv;
		edgeSrc[e] = rand()%nv;
		// printf("Batch update: (#%d) (%d %d)\n", e,edgeSrc[e],edgeDst[e]);
	}
}

void generateUndirectedEdgeUpdates(length_t nv, length_t numEdges, vertexId_t* edgeSrc, vertexId_t* edgeDst){
	for(int32_t e=0; e<numEdges; e++){
		edgeSrc[2*e] = rand()%nv;
		edgeDst[2*e] = rand()%nv;
		edgeSrc[2*e+1] = edgeDst[2*e];
		edgeDst[2*e+1] = edgeSrc[2*e];
		// printf("Batch update: (#%d) (%d %d)\n", e,edgeSrc[e],edgeDst[e]);
	}
}

void invertGraph(length_t nv, length_t numEdges, length_t* off, vertexId_t* adj,length_t** offInvert, vertexId_t** adjInvert){
	*offInvert = (length_t*)allocHostArray(nv+1, sizeof(length_t));
	*adjInvert = (vertexId_t*)allocHostArray(numEdges, sizeof(vertexId_t));

	length_t*  poffI=*offInvert;
	vertexId_t* pAdjI=*adjInvert;

	for(int v=0; v<nv;v++)
		poffI[v]=0;	

	for(int v=0; v<nv;v++){
		length_t srclen=off[v+1]-off[v];
		for (length_t s=0; s<srclen; s++){
			vertexId_t dest=adj[off[v]+s];
			poffI[dest]++;
		}
	}

	length_t* tempPrefix = (length_t*)allocHostArray(nv+1, sizeof(length_t));
	length_t* poffI2 = (length_t*)allocHostArray(nv+1, sizeof(length_t));
	tempPrefix[0]=0;
	for(int v=0; v<nv;v++){
		tempPrefix[v+1]=tempPrefix[v]+poffI[v];	
	}

	for(int v=0; v<=nv;v++)
		poffI2[v]=0;	

	for(int v=0; v<nv;v++){
		length_t srclen=off[v+1]-off[v];
		for (length_t s=0; s<srclen; s++){
			vertexId_t dest=adj[off[v]+s];
			pAdjI[tempPrefix[dest]+poffI2[dest]]=v;
			poffI2[dest]++;
		}
	}

	for(int v=0; v<nv;v++)
		if(poffI[v]!=poffI2[v])
			cout << "Sanity checking has failed" << endl;	

	for(int v=0; v<=nv;v++){
		poffI[v]=tempPrefix[v];	
	}

	freeHostArray(tempPrefix);
	freeHostArray(poffI2);
}

void releaseInvert(length_t** offInvert, vertexId_t** adjInvert){
	freeHostArray(*offInvert);
	freeHostArray(*adjInvert);
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
	    readGraphDIMACS(argv[1],&off,&adj,&nv,&ne,false);
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

    length_t *offInvert;
    vertexId_t *adjInvert;

	// Finding largest vertex
	vertexId_t maxV=0;
	length_t   maxLen=0;
	for(int v=1; v<nv;v++){
		if((off[v+1]-off[v])>maxLen){
			maxV=v;
			maxLen=off[v+1]-off[v];
		}
	}

	invertGraph(nv, ne, off, adj,&offInvert, &adjInvert);

	for(int b=0; b<2; b++){
		bool isDirected;
		if (b==1)
			isDirected=true;
		else
			isDirected=false;

		cudaEvent_t ce_start,ce_stop;
		cuStinger custing(defaultInitAllocater,defaultUpdateAllocater);

		cuStingerInitConfig cuInit;
		cuInit.initState =eInitStateCSR;
		cuInit.maxNV = nv+1;
		cuInit.useVWeight = false;	cuInit.isSemantic = false;  cuInit.useEWeight = false;
		cuInit.csrNV 			= nv;		cuInit.csrNE	   		= ne;
		cuInit.csrOff 			= off;		cuInit.csrAdj 			= adj;
		cuInit.csrVW 			= NULL;		cuInit.csrEW			= NULL;
		custing.initializeCuStinger(cuInit);
		
		cuStingerInitConfig cuInitInv;
		cuInitInv.initState =eInitStateCSR;
		cuInitInv.maxNV = nv+1;
		cuInitInv.useVWeight = false;  cuInitInv.isSemantic = false;  cuInitInv.useEWeight = false;
		cuInitInv.csrNV 		= nv;			cuInitInv.csrNE	   		= ne;
		cuInitInv.csrOff 		= offInvert; 	cuInitInv.csrAdj 		= adjInvert;
		cuInitInv.csrVW 		= NULL; 		cuInitInv.csrEW			= NULL;
		cuStinger custingInv(defaultInitAllocater,defaultUpdateAllocater);
		custingInv.initializeCuStinger(cuInitInv);

		float totalTime;

		katzCentralityStreaming kcs;
		int maxIterations=20;
		int topK=100;
		int numBatchEdges=100;

		if(isDirected)
			kcs.setInitParametersDirected(maxIterations,topK,maxLen,&custingInv);
		else
			kcs.setInitParametersUndirected(maxIterations,topK,maxLen);

		kcs.Init(custing);
		start_clock(ce_start, ce_stop);
		kcs.runStatic(custing);
		totalTime = end_clock(ce_start, ce_stop);
		cout << "The number of iterations      : " << kcs.getIterationCount() << endl;
		cout << "Total time for KC             : " << totalTime << endl; 
		cout << "Average time per iteartion    : " << totalTime/(float)kcs.getIterationCount() << endl; 

		BatchUpdateData *bud, *budInverted;
		srand (1);
		if (isDirected){
			bud = new BatchUpdateData(numBatchEdges,true);
			generateEdgeUpdates(nv, numBatchEdges, bud->getSrc(),bud->getDst());			
			srand (1);
			budInverted = new BatchUpdateData(numBatchEdges,true);
			generateEdgeUpdatesInverted(nv, numBatchEdges, budInverted->getSrc(),budInverted->getDst());			
		}
		else{		
			bud = new BatchUpdateData(numBatchEdges*2,true);
			generateUndirectedEdgeUpdates(nv, numBatchEdges, bud->getSrc(),bud->getDst());
			numBatchEdges*=2;
		}

		BatchUpdate* bu = new BatchUpdate(*bud);

		length_t allocs;
		custing.edgeInsertions(*bu,allocs);
		if(isDirected){
			BatchUpdate* buInv = new BatchUpdate(*budInverted);
			custingInv.edgeInsertions(*buInv,allocs);
			delete buInv;
		}

		katzCentrality kcPostUpdate;	
		kcPostUpdate.setInitParameters(maxIterations,topK,maxLen,false);
		kcPostUpdate.Init(custing);
		kcPostUpdate.Reset();
		start_clock(ce_start, ce_stop);
		kcPostUpdate.Run(custing);
		totalTime = end_clock(ce_start, ce_stop);
		cout << "The number of iterations      : " << kcPostUpdate.getIterationCount() << endl;
		// cout << "Total time for KC             : " << totalTime << endl; 
		// cout << "Average time per iteartion    : " << totalTime/(float)kcPostUpdate.getIterationCount() << endl; 

		start_clock(ce_start, ce_stop);
		kcs.insertedBatchUpdate(custing,*bu);
		totalTime = end_clock(ce_start, ce_stop);

		double* kcScoresStreaming  = (double*) allocHostArray(custing.nv, sizeof(double));
		double* kcScoresPostUpdate = (double*) allocHostArray(custing.nv, sizeof(double));
		ulong_t* nPathsStreaming  = (ulong_t*) allocHostArray(custing.nv*maxIterations, sizeof(ulong_t));
		ulong_t* nPathsPostUpdate = (ulong_t*) allocHostArray(custing.nv*maxIterations, sizeof(ulong_t));
		kcs.copyKCToHost(kcScoresStreaming);
		kcPostUpdate.copyKCToHost(kcScoresPostUpdate);
		kcs.copynPathsToHost(nPathsStreaming);
		kcPostUpdate.copynPathsToHost(nPathsPostUpdate);

		double sum=0.0;
		for(int v=0; v < custing.nv; v++){
			sum += fabs(kcScoresStreaming[v]-kcScoresPostUpdate[v]);
			double diff = kcScoresStreaming[v]-kcScoresPostUpdate[v];
			if(fabs(diff)>1e-10){
				printf("%d, %1.11lf, %1.11lf, %1.11lf \n",v,diff,kcScoresStreaming[v],kcScoresPostUpdate[v]);
			}
		}
		printf("\nSum of difference %4.11lf \n", sum);

		for (int iter=0; iter<kcs.getIterationCount(); iter++)
		{
			for(int v=0; v < custing.nv; v++){
				ulong_t *nPathsStream = nPathsStreaming + custing.nv*iter,*nPathsStatic = nPathsPostUpdate + custing.nv*iter;
				ulong_t *nPathsStreamNext = nPathsStreaming + custing.nv*(iter+1),*nPathsStaticNext = nPathsPostUpdate + custing.nv*(iter+1);
				if (nPathsStream[v]!=nPathsStatic[v])
					printf("^^^^^^ %d, %d, %lld, %lld   \n",iter, v,nPathsStream[v],nPathsStatic[v]);

			}


		}

		if(isDirected)
			delete budInverted;
		delete bu;
		delete bud;


		kcPostUpdate.Release();
		kcs.Release();

		custing.freecuStinger();
		custingInv.freecuStinger();
	}

	releaseInvert(&offInvert, &adjInvert);

	free(off);
	free(adj);

	cudaDeviceReset();
    return 0;	
}

