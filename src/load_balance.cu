
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>

#include "load_balance.cuh"
#include "algs.cuh"
#include "macros.cuh"

namespace cuStingerAlgs{

	class loadBalanceOperators{
	public:
		static __device__ void inPlaceQueue(cuStinger* custing,vertexId_t src, void* metadata){
			vertexId_t* arr = (vertexId_t*)metadata;
			arr[src]=src;
		}

		static __device__ void randomQueue(cuStinger* custing,vertexId_t src, void* metadata){
			vertexQueue* queue = (vertexQueue*)metadata;
			queue->enqueue(src);
		}
		static __device__ void randomQueueNoZeros(cuStinger* custing,vertexId_t src, void* metadata){
			vertexQueue* queue = (vertexQueue*)metadata;
			if (custing->dVD->getUsed()[src] > 0)
				queue->enqueue(src);
		}		
	};


	__global__ void deviceEstimateWorkPerVertex(vertexId_t* verArray, length_t len,cuStinger* custing,void* metadata, int32_t verticesPerThreadBlock){
		vertexId_t ind_init=blockIdx.x*verticesPerThreadBlock+threadIdx.x;
		length_t* workload = (length_t*)metadata;

		for (vertexId_t init_hat=0; init_hat<verticesPerThreadBlock; init_hat+=blockDim.x){
			vertexId_t ind=ind_init+init_hat;
			if(ind>=len){
				break;
			}
			vertexId_t src=verArray[ind];
			workload[ind]=custing->dVD->used[src];
		}
	}

	void cusLoadBalance::estimateWorkPerVertex(vertexId_t* verArray, length_t len,cuStinger& custing,void* metadata){
		dim3 numBlocks(1, 1); int32_t threads=32;
		dim3 threadsPerBlock(threads, 1);
		// int32_t verticesPerThreadBlock=128;
		int32_t verticesPerThreadBlock=512;

		numBlocks.x = ceil((float)len/(float)verticesPerThreadBlock);
		deviceEstimateWorkPerVertex<<<numBlocks, threadsPerBlock>>>(verArray,len,custing.devicePtr(),metadata,verticesPerThreadBlock);
	}


	__global__ void deviceCreateNeedles(length_t* dNeedles,length_t numNeedles, length_t* dPrefixArray, length_t prefixSize){
		if(threadIdx.x==0){
			dNeedles[blockIdx.x]=dPrefixArray[prefixSize-1]*((double)(blockIdx.x)/(double)numNeedles);
		}
	}


	void cusLoadBalance::createNeedles(length_t* dNeedles,length_t numNeedles, length_t* dPrefixArray, length_t prefixSize){
		deviceCreateNeedles<<<foundNumPartition,32>>>(devNeedles,foundNumPartition,devPrefixArray, currArrayLen+1);

	}

	void cusLoadBalance::queueVertices(cuStinger &custing,bool queueZeros,bool queueRandomly){

		if(queueZeros){
			if(queueRandomly){
				vertexQueue vqueue;
				vqueue.Init(currArrayLen);
				allVinG_TraverseVertices<loadBalanceOperators::randomQueue>(custing,vqueue.devPtr());
				vqueue.SyncHostWithDevice();
				// cout << "!!!!!!  Queuing all elements randomly" << endl;
				copyArrayDeviceToDevice(vqueue.getQueue(), currArray,vqueue.getActiveQueueSize(),sizeof(vertexId_t));
			}
			else{
				// cout << "!!!!!!  Queuing all elements inorder" << endl;				
				allVinG_TraverseVertices<loadBalanceOperators::inPlaceQueue>(custing,currArray);
			}
		}
		else{
				vertexQueue vqueue;
				vqueue.Init(currArrayLen);
				allVinG_TraverseVertices<loadBalanceOperators::randomQueueNoZeros>(custing,vqueue.devPtr());
				vqueue.SyncHostWithDevice();
				// cout << "!!!!!!  Queuing all non zeros randomly" << endl;				
				copyArrayDeviceToDevice(vqueue.getQueue(), currArray,vqueue.getActiveQueueSize(),sizeof(vertexId_t));
				currArrayLen = vqueue.getActiveQueueSize();
		}


	}

} // cuStingerAlgs namespace
