#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>


// #include "moderngpu.cuh"
// #include "moderngpu/src/kernel_scan.hxx"
#include "kernel_scan.hxx"
#include "search.hxx"
#include "cta_search.hxx"

// ModernGPU
#include <kernel_sortedsearch.hxx>
#include <transform.hxx>
using namespace mgpu;

//CUB headers
#include <cub.cuh>
using namespace cub;


#include <block/block_radix_sort.cuh>

#include <iostream>

#include "bc_streaming/bc.hpp"


__global__ void bc_static_brandes(cuStinger *custing, unsigned long long *sigma,
	float *delta, int *d, float *bc_d, int numRoots) {

    printf("Running bc_static w/ ptrs:\ncusting: %p\tsigma: %p\tdelta: %p\td: %p\tbc: %p\n", 
		custing, sigma, delta, d, bc_d);

	printf("Doing Nothing\n");
}

void bcMain(cuStinger& custing, void* func_meta_data) {
    printf("YOOOOOOO oldMain");
}

/*
	########    ######          ######## ##     ## ##    ##  ######  ########  #######  ########   ######
	##     ##  ##               ##       ##     ## ###   ## ##    ##    ##    ##     ## ##     ## ##    ##
	##     ##  ##               ##       ##     ## ####  ## ##          ##    ##     ## ##     ## ##
	########   ##       ####### ######   ##     ## ## ## ## ##          ##    ##     ## ########   ######
	##     ##  ##               ##       ##     ## ##  #### ##          ##    ##     ## ##   ##         ##
	##     ##  ##               ##       ##     ## ##   ### ##    ##    ##    ##     ## ##    ##  ##    ##
	########    ######          ##        #######  ##    ##  ######     ##     #######  ##     ##  ######
*/

// typedef struct {
// 	vertexId_t* queue;
// 	length_t queueCurr;
// 	length_t queueEnd;
// 	vertexId_t* level;
// 	vertexId_t currLevel;
// }bfsData;

// __device__ void bcExpandFrontier(cuStinger* custing,vertexId_t src, void* metadata){
// 	bfsData* bd = (bfsData*)metadata;
// 	length_t srcLen=custing->dVD->used[src];
// 	vertexId_t* adj_src=custing->dVD->adj[src]->dst;

// 	vertexId_t nextLevel=bd->currLevel+1;
// 	for(vertexId_t adj=threadIdx.x; adj<srcLen; adj+=blockDim.x){
// 		vertexId_t dest = adj_src[adj];
// 		vertexId_t prev = atomicCAS(bd->level+dest,INT32_MAX,nextLevel);
// 		if(prev==INT32_MAX){
// 			length_t prevPos = atomicAdd(&(bd->queueEnd),1);
// 			bd->queue[prevPos] = dest;
// 		}
// 	}
// }

// typedef struct {
// 	vertexId_t* queue;
// 	length_t queueCurr;
// 	length_t queueEnd;
// 	vertexId_t* level;
// 	vertexId_t currLevel;
// } bcTreeData;

// __device__ void bcExpandFrontier(cuStinger* custing,vertexId_t src, vertexId_t dst, void* metadata) {
// 	bcTreeData* bctd = (bcTreeData*)metadata;
// 	vertexId_t nextLevel=bctd->currLevel+1;

// 	vertexId_t prev = atomicCAS(bctd->level+dst,INT32_MAX,nextLevel);
// 	if(prev == INT32_MAX) {
// 		length_t prevPos = atomicAdd(&(bctd->queueEnd),1);
// 		bctd->queue[prevPos] = dst;
// 	}

// 	if (bctd->level[dst] == nextLevel) {
// 		bctd->sigma[dst] = atomicAdd(&(bctd->sigma[dst]), bctd->sigma[src]);
// 	}
// }

// __device__ cusSubKernel ptrbcExpandFrontier = bcExpandFrontier;

// __device__ void bcExpandFrontierMacro(cuStinger* custing, vertexId_t src, void* metadata) {
// 	bcTreeData* bd = (bcTreeData*)metadata;
// 	vertexId_t nextLevel=bd->currLevel+1;

// 	CUSTINGER_FOR_ALL_EDGES_OF_VERTEX_PAR_THREAD_BLOCK_BEGIN(custing, src) 
// 		vertexId_t prev = atomicCAS(bd->level+CUSTINGER_EDGE_DEST,INT32_MAX,nextLevel);
// 		if(prev==INT32_MAX) {
// 			length_t prevPos = atomicAdd(&(bd->queueEnd),1);
// 			bd->queue[prevPos] = CUSTINGER_EDGE_DEST;
// 		}
// 	CUSTINGER_FOR_ALL_EDGES_OF_VERTEX_PAR_THREAD_BLOCK_END();
// }

// __device__ cusSubKernel ptrbcExpandFrontierMacro = bcExpandFrontierMacro;


// __device__ void setLevelInfinity(cuStinger* custing,vertexId_t src, void* metadata){
// 	bcTreeData* bd = (bcTreeData*)metadata;
// 	bd->level[src]=INT32_MAX;
// }
// __device__ cusSubKernel ptrSetLevelInfinity = setLevelInfinity;


// void bcMain(cuStinger& custing, void* func_meta_data)
// {
// 	cudaEvent_t ce_start,ce_stop;	
// 	start_clock(ce_start, ce_stop);

// 	bcTreeData hostbcTreeData;
// 	hostbcTreeData.queue = (vertexId_t*) allocDeviceArray(custing.nv, sizeof(vertexId_t));
// 	hostbcTreeData.level = (vertexId_t*) allocDeviceArray(custing.nv, sizeof(vertexId_t));
// 	hostbcTreeData.queueCurr=0;
//     hostbcTreeData.queueEnd=1;
// 	hostbcTreeData.currLevel=0;

// 	bfsData* deviceBfsData = (bfsData*)allocDeviceArray(1, sizeof(bfsData));
// 	copyArrayHostToDevice(&hostBfsData,deviceBfsData,1, sizeof(bfsData));

// 	dim3 numBlocks(1, 1);
// 	int32_t threads=32;
// 	dim3 threadsPerBlock(threads, 1);
// 	int32_t verticesPerThreadBlock;

// 	numBlocks.x = ceil((float)custing.nv/(float)threads);
// 	if (numBlocks.x>64000){
// 		numBlocks.x=64000;
// 	}	
// 	verticesPerThreadBlock = ceil(float(custing.nv)/float(numBlocks.x-1));

// 	cusSubKernel* dSetInfinity2 = (cusSubKernel*)allocDeviceArray(1,sizeof(cusSubKernel));
// 	cudaMemcpyFromSymbol( dSetInfinity2, ptrSetLevelInfinity, sizeof(cusSubKernel),0,cudaMemcpyDeviceToDevice);
// 	allVerticesInGraphParallelVertexPerThreadBlock<<<numBlocks, threadsPerBlock>>>(custing.devicePtr(),deviceBfsData,dSetInfinity2,verticesPerThreadBlock);
// 	freeDeviceArray(dSetInfinity2);

// 	vertexId_t root=2; length_t level=0;
// 	copyArrayHostToDevice(&root,hostBfsData.queue,1,sizeof(vertexId_t));
// 	copyArrayHostToDevice(&level,hostBfsData.level+root,1,sizeof(length_t));

// 	cusSubKernel* dTraverseEdges = (cusSubKernel*)allocDeviceArray(1,sizeof(cusSubKernel));
// 	// cudaMemcpyFromSymbol( dTraverseEdges, ptrBFSExpandFrontierMacro, sizeof(cusSubKernel),0,cudaMemcpyDeviceToDevice);
// 	cudaMemcpyFromSymbol( dTraverseEdges, ptrBFSExpandFrontier, sizeof(cusSubKernel),0,cudaMemcpyDeviceToDevice);

// 	length_t prevEnd=1;
// 	while(hostBfsData.queueEnd-hostBfsData.queueCurr>0){
// 		allVerticesInArrayOneVertexPerTB<<<numBlocks, threadsPerBlock>>>(hostBfsData.queue+hostBfsData.queueCurr,
// 										hostBfsData.queueEnd-hostBfsData.queueCurr,custing.devicePtr(),
// 										deviceBfsData,dTraverseEdges,verticesPerThreadBlock);
// 		copyArrayDeviceToHost(deviceBfsData,&hostBfsData,1, sizeof(bfsData));

// 		hostBfsData.queueCurr=prevEnd;
// 		prevEnd = hostBfsData.queueEnd;
// 		hostBfsData.currLevel++;
// 		copyArrayHostToDevice(&hostBfsData,deviceBfsData,1, sizeof(bfsData));
// 	}

// 	freeDeviceArray(dTraverseEdges);
// 	cout << "The queue end  :" << hostBfsData.queueEnd << endl;

// 	float totalBFSTime = end_clock(ce_start, ce_stop);
// 	cout << "Total time for the BFS : " << totalBFSTime << endl; 

// 	freeDeviceArray(deviceBfsData);
// 	freeDeviceArray(hostBfsData.queue);
// 	freeDeviceArray(hostBfsData.level);
// }


// Adam's Optimized BC GPU Kernel
/*
1) const/restrict keywords
2) edge-based parallelism
3) use of pitch for d, sigma, and delta
*/
// __global__ void bc_gpu_opt(const cuStinger *custing, int n, int m, float *bc,
// 	unsigned long long *__restrict__ sigma, float *__restrict__ delta,
// 	int *__restrict__ d, size_t pitch_sigma, size_t pitch_delta,
// 	size_t pitch_d, const int start, const int end)
// {
// 	printf("Beginning bc_gpu_opt");

	

// 	for(int i=start+blockIdx.x; i<end; i+=gridDim.x) //Overall i = [0, 1, ..., n-1] (inclusive)
// 	{
// 		int j = threadIdx.x;

// 		for(int k=threadIdx.x; k<n; k+=blockDim.x)
// 		{
// 			int *d_row = (int*)((char*)d + blockIdx.x*pitch_d);
// 			unsigned long long *sigma_row = (unsigned long long*)((char*)sigma + blockIdx.x*pitch_sigma);
// 			if(k == i) //If its the source node
// 			{
// 				sigma_row[k] = 1;
// 				d_row[k] = 0;
// 			}
// 			else
// 			{
// 				sigma_row[k] = 0;
// 				d_row[k] = INT_MAX;
// 			}

// 			float *delta_row = (float*)((char*)delta + blockIdx.x*pitch_delta);
// 			delta_row[k] = 0;
// 		}
		
// 		int current_depth = 0;
// 		__shared__ bool done;
// 		if(j == 0)
// 		{
// 			done = false;
// 		}
// 		__syncthreads();

// 		while(!done)
// 		{
// 			__syncthreads();
// 			done = true;
// 			__syncthreads();

// 			for(int k = threadIdx.x; k < 2*m; k += blockDim.x)
// 			{
// 				// We must find the k-th edge of the graph, both src and dest
// 				int v = 0; // source vertex of k-th edge
// 				int offset = 0;  // offset from adj list of v to get dest
				
// 				// Tells us the length of each adj list per src vertex
// 				length_t *srcLens = custing->dVD->used;

// 				// sum of edges for all edges for each source vNum we'vector
// 				// already looked at (see vNum in the loop, below)
// 				int counted = 0;

// 				// Find the k-th edge
// 				for(int vNum = 0; vNum < n; vNum++) {
// 					if (k > counted && k <= counted + srcLens[vNum]) {
// 						// We know that k-th edge falls in vNum's adj list
// 						offset = k - counted;
// 						v = vNum;
// 						break;
// 					} else {
// 						counted += srcLens[vNum];
// 					}
// 				}
// 				// now, v is the true src of the k-th edge

// 				int *d_row = (int *)((char*)d + blockIdx.x*pitch_d);
// 				if(d_row[v] == current_depth)
// 				{
// 					// the dest vertex of the k-th edge
// 					// get the adj list for v, offset by offset
// 					int w = (int) ((custing->dVD->getAdj())[v]->dst)[offset];
// 					if(d_row[w] == INT_MAX)
// 					{
// 						d_row[w] = current_depth + 1; 
// 						done = false;
// 					}
// 					if(d_row[w] == (current_depth + 1)) 
// 					{
// 						unsigned long long *sigma_row = (unsigned long long*)((char*)sigma + blockIdx.x*pitch_sigma);
// 						atomicAdd(&sigma_row[w],sigma_row[v]);
// 					}
// 				}
// 			}

// 			__syncthreads();
// 			current_depth++;
// 		}

// 		__syncthreads();
// 		current_depth--;
// 		while(current_depth > 0)
// 		{
// 			for(int k=threadIdx.x; k<2*m; k+=blockDim.x)
// 			{
// 				// We must find the k-th edge of the graph, both src and dest
// 				int w = 0; // source vertex of k-th edge
// 				// int w = F[k];
// 				int offset = 0;  // offset from adj list of v to get dest

// 				// Tells us the length of each adj list per src vertex
// 				length_t *srcLens = custing->dVD->used;

// 				// sum of edges for all edges for each source vNum we'vector
// 				// already looked at (see vNum in the loop, below)
// 				int counted = 0;

// 				// Find the k-th edge
// 				for(int vNum = 0; vNum < n; vNum++) {
// 					if (k > counted && k <= counted + srcLens[vNum]) {
// 						// We know that k-th edge falls in vNum's adj list
// 						offset = k - counted;
// 						w = vNum;
// 						break;
// 					} else {
// 						counted += srcLens[vNum];
// 					}
// 				}
// 				// now, w is the true src of the k-th edge

// 				int *d_row = (int *)((char*)d + blockIdx.x*pitch_d);
// 				if(d_row[w] == current_depth)
// 				{
// 					// int v = C[k];
// 					// the dest vertex of the k-th edge
// 					// get the adj list for v, offset by offset
// 					int v = (int) ((custing->dVD->getAdj())[w]->dst)[offset];
// 					if(d_row[w] == (d_row[v]+1))
// 					{
// 						float *delta_row = (float*)((char*)delta + blockIdx.x*pitch_d);
// 						unsigned long long *sigma_row = (unsigned long long*)((char*)sigma + blockIdx.x*pitch_sigma);
// 						float change = (sigma_row[v] / (float)sigma_row[w]) * (1.0f + delta_row[w]);
// 						atomicAdd(&delta_row[v], change);
// 					}
// 				}
// 			}
// 			__syncthreads();
// 			current_depth--;
// 		}

// 		for(int k=threadIdx.x; k<n; k+=blockDim.x)
// 		{
// 			if(k != i) //Don't count the source node
// 			{
// 				float *delta_row = (float*)((char*)delta + blockIdx.x*pitch_delta);
// 				atomicAdd(&bc[k], delta_row[k]); //Does this need to be atomic?
// 			}
// 		}
// 		__syncthreads();
// 	}
// }

// __global__ void bc_gpu_naive(float *bc, int *R, int *C, int n, int m) 
// {
// 	int i = blockIdx.x;
// 	int j = threadIdx.x;

// 	__shared__ int sigma[4000]; //Arbitrarily large enough size for testing purposes, for now
// 	__shared__ int d[4000];
// 	__shared__ float delta[4000];

// 	if(i == j)
// 	{
// 		sigma[j] = 1;
// 		d[j] = 0;
// 	}
// 	else
// 	{
// 		sigma[j] = 0;
// 		d[j] = INT_MAX;
// 	}

//     int current_depth;
// 	__shared__ bool done;
// 	if(j == 0)
// 	{
// 		done = false;
// 		current_depth = 0;
// 	}
// 	delta[j] = 0;
// 	__syncthreads();

// 	while(!done)
// 	{
// 		__syncthreads();
// 		done = true;
// 		__syncthreads();

// 		if(d[j] == current_depth)
// 		{
// 			for(int k=R[j]; k<R[j+1]; k++)
// 			{
// 				int w = C[k];
// 				if(d[w] == INT_MAX)
// 				{
// 					d[w] = d[j] + 1;
// 					done = false;
// 				}
// 				if(d[w] == (d[j] + 1))
// 				{
// 					atomicAdd(&sigma[w],sigma[j]);
// 				}
// 			}
// 		}

// 		__syncthreads();
// 		current_depth++;
// 	}

// 	__syncthreads();
// 	//current_depth-- before loop? Should save an iteration...
// 	while(current_depth > 0)
// 	{
// 		if(d[j] == current_depth)
// 		{
// 			for(int k=R[j]; k<R[j+1]; k++)
// 			{
// 				int v = C[k];
// 				if(d[j] == (d[v]+1))
// 				{
// 					float change = (sigma[v]/(float)sigma[j])*(1.0f+delta[j]);
// 					atomicAdd(&delta[v],change);
// 				}
// 			}

// 			if(i != j)
// 			{
// 				atomicAdd(&bc[j],delta[j]); //j from two different blocks could simultaneously update this value
// 			}
// 		}

// 		__syncthreads();
// 		current_depth--;
// 	}

// 	/*if(i != j)
// 	{
// 		atomicAdd(&bc[j],delta[j]);
// 	}*/
// }

// void bc_static(cuStinger& custing, float *bc, int numRoots,
// 	int max_threads_per_block, int number_of_SMs) {

//     int thread_blocks = 1;
//     int blockdim = 1;
// 	vertexId_t numVertices = custing.nv;

//     std::cout << "Allocating mem for sigma, delta, d" << std::endl;

//     unsigned long long *sigma;
// 	float *delta;
// 	int *d;

// 	// Optimization used for 2D arrays
// 	size_t pitch_sigma, pitch_delta, pitch_d;

// 	// sigma, delta, and d are all 2D arrays of size (numRoots * numVertices)
// 	// checkCudaErrors( cudaMalloc(&sigma, sizeof(unsigned long long) * numRoots * numVertices) );
// 	// checkCudaErrors( cudaMalloc(&delta, sizeof(float) * numRoots * numVertices) );
// 	// checkCudaErrors( cudaMalloc(&d, sizeof(int) * numRoots * numVertices) );

// 	checkCudaErrors( cudaMallocPitch(&sigma, &pitch_sigma, sizeof(unsigned long long) * numVertices, numRoots) );
// 	checkCudaErrors( cudaMallocPitch(&delta, &pitch_delta, sizeof(float) * numVertices, numRoots) );
// 	checkCudaErrors( cudaMallocPitch(&d, &pitch_d, sizeof(int) * numVertices, numRoots) );

// 	float *bc_d;  // The actual bc values we'll keep after computation, on the device
// 	checkCudaErrors( cudaMalloc(&bc_d, sizeof(float) * numVertices) );

// 	//Set kernel dimensions
// 	dim3 dimBlock, dimGrid;
// 	dimBlock.x = max_threads_per_block;
// 	dimBlock.y = 1;
// 	dimBlock.z = 1;
// 	dimGrid.x = number_of_SMs;
// 	dimGrid.y = 1;
// 	dimGrid.z = 1;

// 	// n = # vertices in graph
// 	int n = custing.nv;
// 	// m = # edges in graph
// 	int m = custing.getNumberEdgesUsed();

// 	// Does nothing
// 	std::cout << "About to call empty brandes function" << std::endl;
//     bc_static_brandes<<<thread_blocks, blockdim>>>(custing.devicePtr(), sigma, delta, d, bc_d, numRoots);

// 	// TODO: Why did Adam choose 30?
// 	for(int i=1; i<=30; i++) //Might want to show for a larger number of blocks as well
// 	{
// 		dimGrid.x = i;
// 		// start_clock(start,end);
// 		std::cout << "About actual bc_gpu_opt function" << std::endl;
// 		bc_gpu_opt<<<dimGrid, dimBlock>>>(custing.devicePtr(), n, m, bc_d,
// 			sigma, delta, d, pitch_sigma, pitch_delta, pitch_d, 0, numRoots);

// 		checkCudaErrors(cudaPeekAtLastError()); //Check for kernel launch errors
// 		// time_gpu_opt = end_clock(start,end);
// 		// std::cout << "," << time_gpu_opt/(float)1000; //Time in seconds
// 	}


//     // std::cout << "PTR sigma: " << sigma << std::endl;
//     // std::cout << "PTR delta: " << delta << std::endl;
//     // std::cout << "PTR d: " << d << std::endl;

// 	// std::cout << std::endl;
// 	// std::cout << "PTR bc: " << bc << std::endl;

// 	// Put results from GPU bc array into CPU bc array
// 	checkCudaErrors(cudaMemcpy(bc, bc_d, sizeof(float) * custing.nv, cudaMemcpyDeviceToHost));

// 	// Free memory
//     checkCudaErrors( cudaFree(sigma) );
// 	checkCudaErrors( cudaFree(delta) );
// 	checkCudaErrors( cudaFree(d) );
// 	checkCudaErrors( cudaFree(bc_d) );

// }
