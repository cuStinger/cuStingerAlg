
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>


#include <typeinfo>
#include "utils.hpp"
#include "update.hpp"
#include "cuStinger.hpp"
#include "algs.cuh"

#include "load_balance.cuh"


#include "kernel_scan.hxx"
// #include "search.hxx"
// #include "cta_search.hxx"

// ModernGPU
#include <kernel_sortedsearch.hxx>
#include <transform.hxx>
//CUB headers
#include <cub.cuh>

#include <block/block_radix_sort.cuh>

using namespace mgpu;
using namespace cub;

namespace cuStingerAlgs {

template <cusSubKernelVertex cusSK>
static __global__ void device_allVinG_TraverseVertices(cuStinger* custing,void* metadata,int32_t verticesPerThreadBlock){
	vertexId_t v_init=blockIdx.x*verticesPerThreadBlock+threadIdx.x;
	length_t nv = custing->getMaxNV();

	for (vertexId_t v_hat=0; v_hat<verticesPerThreadBlock; v_hat+=blockDim.x){
		vertexId_t v=v_init+v_hat;
		if(v>=nv){
			break;
		}
		(cusSK)(custing,v,metadata);
	}
}

template <cusSubKernelVertex cusSK>
static void allVinG_TraverseVertices(cuStinger& custing,void* metadata){

	dim3 numBlocks(1, 1); int32_t threads=32;
	dim3 threadsPerBlock(threads, 1);
	// int32_t verticesPerThreadBlock=128;
	int32_t verticesPerThreadBlock=512;

	numBlocks.x = ceil((float)custing.nv/(float)verticesPerThreadBlock);
	device_allVinG_TraverseVertices<cusSK><<<numBlocks, threadsPerBlock>>>(custing.devicePtr(),metadata,verticesPerThreadBlock);
}

template <cusSubKernelEdge cusSK>
static __global__ void device_allEinG_TraverseEdges(cuStinger* custing,void* metadata, int32_t verticesPerThreadBlock){
	vertexId_t v_init=blockIdx.x*verticesPerThreadBlock;
	length_t nv = custing->getMaxNV();

	for (vertexId_t v_hat=0; v_hat<verticesPerThreadBlock; v_hat++){
		vertexId_t src=v_init+v_hat;
		if(src>=nv){
			break;
		}

		length_t srcLen=custing->dVD->used[src];
		vertexId_t* adj_src=custing->dVD->adj[src]->dst;
		for(vertexId_t adj=threadIdx.x; adj<srcLen; adj+=blockDim.x){
			vertexId_t dst = adj_src[adj];
			(cusSK)(custing,src,dst,metadata);
		}
	}
}

template <cusSubKernelEdge cusSK>
static void allEinG_TraverseEdges(cuStinger& custing,void* metadata){
	dim3 numBlocks(1, 1); int32_t threads=32;
	dim3 threadsPerBlock(threads, 1);
	int32_t verticesPerThreadBlock=128;

	numBlocks.x = ceil((float)custing.nv/(float)verticesPerThreadBlock);

	device_allEinG_TraverseEdges<cusSK><<<numBlocks, threadsPerBlock>>>(custing.devicePtr(),metadata,verticesPerThreadBlock);
}

template <cusSubKernelVertex cusSK>
static __global__ void device_allVinA_TraverseVertices(cuStinger* custing,void* metadata, int32_t verticesPerThreadBlock,vertexId_t* verArray, length_t len){
	vertexId_t v_init=blockIdx.x*verticesPerThreadBlock+threadIdx.x;

	for (vertexId_t v_hat=0; v_hat<verticesPerThreadBlock; v_hat+=blockDim.x){
		vertexId_t v=v_init+v_hat;
		if(v>=len){
			break;
		}
		vertexId_t src =verArray[v];		
		(cusSK)(custing,src,metadata);
	}
}

template <cusSubKernelVertex cusSK>
static void allVinA_TraverseVertices(cuStinger& custing,void* metadata,vertexId_t* verArray, length_t len ){
	dim3 numBlocks(1, 1); int32_t threads=32;
	dim3 threadsPerBlock(threads, 1);
	int32_t verticesPerThreadBlock=512;

	numBlocks.x = ceil((float)len/(float)verticesPerThreadBlock);
	device_allVinA_TraverseVertices<cusSK><<<numBlocks, threadsPerBlock>>>(custing.devicePtr(),metadata,verticesPerThreadBlock,verArray,len);
}

template <cusSubKernelVertex cusSK>
static void allVinA_TraverseVertices(cuStinger& custing,void* metadata,cusLoadBalance& cusLB){
	dim3 numBlocks(1, 1); int32_t threads=32;
	dim3 threadsPerBlock(threads, 1);
	int32_t verticesPerThreadBlock=512;

	numBlocks.x = ceil((float)cusLB.currArrayLen/(float)verticesPerThreadBlock);
	device_allVinA_TraverseVertices<cusSK><<<numBlocks, threadsPerBlock>>>(custing.devicePtr(),metadata,verticesPerThreadBlock,cusLB.currArray,cusLB.currArrayLen);
}





template <cusSubKernelEdge cusSK>
static __global__ void device_allVinA_TraverseEdges(cuStinger* custing,void* metadata, int32_t verticesPerThreadBlock,vertexId_t* verArray, length_t len){
	vertexId_t v_init=blockIdx.x*verticesPerThreadBlock;
	length_t nv = custing->getMaxNV();

	for (vertexId_t v_hat=0; v_hat<verticesPerThreadBlock; v_hat++){
		vertexId_t v=v_init+v_hat;
		if(v>=len){
			break;
		}
		vertexId_t src =verArray[v];

		length_t srcLen=custing->dVD->used[src];
		vertexId_t* adj_src=custing->dVD->adj[src]->dst;
		for(vertexId_t adj=threadIdx.x; adj<srcLen; adj+=blockDim.x){
			vertexId_t dst = adj_src[adj];
			(cusSK)(custing,src,dst,metadata);
		}
	}
}

template <cusSubKernelEdge cusSK>
static void allVinA_TraverseEdges(cuStinger& custing,void* metadata,vertexId_t* verArray, length_t len){
	dim3 numBlocks(1, 1); int32_t threads=32;
	dim3 threadsPerBlock(threads, 1);
	int32_t verticesPerThreadBlock=128;

	numBlocks.x = ceil((float)len/(float)verticesPerThreadBlock);
	device_allVinA_TraverseEdges<cusSK><<<numBlocks, threadsPerBlock>>>(custing.devicePtr(),metadata,verticesPerThreadBlock,verArray,len);
}



template <cusSubKernelEdge cusSK>
static __global__ void device_allVerticesInArray_LoadBalanaced(cuStinger* custing,void* metadata,length_t* needles, length_t numNeedles,vertexId_t* verArray, length_t len)
{
	vertexId_t v_init=needles[blockIdx.x];
	vertexId_t v_max=needles[blockIdx.x+1]-needles[blockIdx.x];
	if(blockIdx.x==(numNeedles-1))
		v_max=len-needles[blockIdx.x];

	const int elePerThread = 8;
	const int bdx=32; //blockDim.x
    // Specialize BlockRadixSort type for our thread block
    typedef BlockRadixSort<length_t, bdx, elePerThread,vertexId_t> BlockRadixSortT;

    __shared__ typename BlockRadixSortT::TempStorage temp_storage;
    // Obtain a segment of consecutive items that are blocked across threads
    length_t   thread_keys[elePerThread];
    vertexId_t thread_values[elePerThread];

    __shared__ length_t   sorted_keys[bdx*elePerThread];
    __shared__ vertexId_t sorted_values[bdx*elePerThread];

    vertexId_t v_hat=0;

    while(v_hat<v_max){
    	vertexId_t v_hat2=v_hat+threadIdx.x;
    	for(int i=0; i<elePerThread; i++){
			vertexId_t vpos=v_init+v_hat2;
    		if(v_hat2>=v_max){
    			thread_keys[i]   = -1;
    			thread_values[i] = INT32_MAX;
    		}
    		else{
    			vertexId_t temp  = verArray[vpos];
    			thread_keys[i]   = custing->dVD->used[temp];
    			thread_values[i] = temp;
    		}
    		v_hat2+=bdx;
    	}
    	__syncthreads();
	    BlockRadixSortT(temp_storage).SortDescending(thread_keys,thread_values);
    	__syncthreads();

	    int pos=threadIdx.x*elePerThread;
	    for(int i=0; i< elePerThread; i++){
		    sorted_keys[pos+i]     = thread_keys[i];
		    sorted_values[pos+i]   = thread_values[i];
	    }
	    __syncthreads();

	    vertexId_t break_out_vertex=-1;
		for (vertexId_t u_hat=0; u_hat<bdx*elePerThread; u_hat++){
			if(sorted_keys[u_hat]==-1)
				break;
			if(sorted_keys[u_hat]<64){
				break_out_vertex=u_hat;
				break;
			}
			vertexId_t src=sorted_values[u_hat];

			length_t srcLen=custing->dVD->used[src];
			vertexId_t* adj_src=custing->dVD->adj[src]->dst;
			for(vertexId_t adj=threadIdx.x; adj<srcLen; adj+=bdx){
				vertexId_t dst = adj_src[adj];
				(cusSK)(custing,src,dst,metadata);
			}
		}
		if(break_out_vertex==-1){
			v_hat+=bdx*elePerThread;
			continue;
		}

		vertexId_t cont_vertex = break_out_vertex;
		break_out_vertex=-1;
		__shared__ vertexId_t next_vertex;
		next_vertex=bdx*elePerThread;
	    __syncthreads();


		for (vertexId_t u_hat=cont_vertex+(threadIdx.x>>4); u_hat<bdx*elePerThread; u_hat+=2){
			if(sorted_keys[u_hat]!=-1 && sorted_keys[u_hat] >= 16)
			{
				vertexId_t src=sorted_values[u_hat];
				length_t srcLen=custing->dVD->used[src];
				vertexId_t* adj_src=custing->dVD->adj[src]->dst;

				for(vertexId_t adj=(threadIdx.x%16); adj<srcLen; adj+=16){	
					vertexId_t dst = adj_src[adj];
					(cusSK)(custing,src,dst,metadata);
				}
			}
			else{
				if (sorted_keys[u_hat] <16){
					atomicMin(&next_vertex,u_hat);
				}
				break;				
			}
		}
		__syncthreads();

		if(next_vertex==bdx*elePerThread){
			v_hat+=bdx*elePerThread;
			continue;
		}
		cont_vertex = next_vertex;

		for (vertexId_t u_hat=cont_vertex+threadIdx.x; u_hat<bdx*elePerThread; u_hat+=bdx){
			if(sorted_keys[u_hat]!=-1)
			{
				vertexId_t src=sorted_values[u_hat];
				length_t srcLen=custing->dVD->used[src];
				vertexId_t* adj_src=custing->dVD->adj[src]->dst;

				for(vertexId_t adj=0; adj<srcLen; adj++){	
					vertexId_t dst = adj_src[adj];
					(cusSK)(custing,src,dst,metadata);
				}
			}
			else
				break;
			// __syncthreads();
		}

	    v_hat+=bdx*elePerThread;
    }
}


template <cusSubKernelEdge cusSK>
static void allVinA_TraverseEdges_LB(cuStinger& custing,void* metadata, cusLoadBalance& cusLB,vertexQueue& vQueue, bool needLoadBalance=true){
		cusLB.currArray=vQueue.getQueueAtCurr();
		cusLB.currArrayLen=vQueue.getActiveQueueSize();

		if(cusLB.currArrayLen==0)
			return;
		if(needLoadBalance){
			cusLB.LoadBalanceArray(custing);
		}
		device_allVerticesInArray_LoadBalanaced<cusSK><<<cusLB.foundNumPartition,32>>>(custing.devicePtr(),metadata,cusLB.devPartitionsPoints,cusLB.foundNumPartition,cusLB.currArray,cusLB.currArrayLen);
}

template <cusSubKernelEdge cusSK>
static void allVinA_TraverseEdges_LB(cuStinger& custing,void* metadata, cusLoadBalance& cusLB){
		if(cusLB.currArrayLen==0)
			return;

		device_allVerticesInArray_LoadBalanaced<cusSK><<<cusLB.foundNumPartition,32>>>(custing.devicePtr(),metadata,cusLB.devPartitionsPoints,cusLB.foundNumPartition,cusLB.currArray,cusLB.currArrayLen);
}




template <cusSubKernelEdge cusSK>
static __global__ void device_allEinA_TraverseEdges(cuStinger* custing,void* metadata, BatchUpdateData* bud, int32_t edgesPerThreadBlock){
	length_t e_init=blockIdx.x*edgesPerThreadBlock+threadIdx.x;
	length_t batchSize = *(bud->getBatchSize());
	vertexId_t* d_updatesSrc    = bud->getSrc();
	vertexId_t* d_updatesDst    = bud->getDst();

	for (length_t e_hat=0; e_hat<edgesPerThreadBlock; e_hat+=blockDim.x){
		length_t e=e_init+e_hat;
		if(e>=batchSize){
			break;
		}
		(cusSK)(custing,d_updatesSrc[e],d_updatesDst[e],metadata);
	}
}


template <cusSubKernelEdge cusSK>
static void allEinA_TraverseEdges(cuStinger& custing, void* metadata, BatchUpdate &bu){
	length_t batchSize = *(bu.getHostBUD()->getBatchSize());
	if(batchSize==0)
			return;

	dim3 numBlocks(1, 1); int32_t threads=32;
	dim3 threadsPerBlock(threads, 1);
	int32_t edgesPerThreadBlock=512;

	numBlocks.x = ceil((float)batchSize/(float)edgesPerThreadBlock);
	device_allEinA_TraverseEdges<cusSK><<<numBlocks, threadsPerBlock>>>(custing.devicePtr(), metadata, bu.getDeviceBUD()->devicePtr(), edgesPerThreadBlock);


}



}


