#include <stdio.h>

//SoA structures
struct graph_data
{
	int *C;
	int *F;
	int *R;
};

struct vertex_data
{
	int *touched;
	unsigned long long *sigma_hat;
	float *delta_hat;
	int *moved;
	int *movement;
	int *Q;
	int *Q2;
	int *QQ;
};

//AoS structures
struct graph_data_aos
{
	int F;
	int C;
};

struct vertex_data_aos
{
	int touched;
	unsigned long long sigma_hat;
	float delta_hat;
	int moved;
	int movement;
	int Q;
	int Q2;
	int QQ;
};
	
template<bool approx>
__global__ void bc_gpu_update_edge(float *bc, const int *__restrict__ R, const int *__restrict__ F, const int *__restrict__ C, const int n, const int m, int *__restrict__ d, unsigned long long *__restrict__ sigma, float *__restrict__ delta, size_t pitch, size_t pitch_sigma, const int *__restrict__ sources, const int start, const int end, const int src, const int dst)
{
	for(int l=start+blockIdx.x; l<end; l+=gridDim.x) //Overall i = [0, 1, ..., n-1] (inclusive)
	{
		int i;
		if(approx)
		{
			i = sources[l]; //i is the absolute value of the source node, l is the relative value
		}
		else
		{
			i = l;
		}
		int j = threadIdx.x;

		__shared__ int recompute;
		__shared__ int u_low;
		__shared__ int u_high;

		if(j == 0)
		{		
			//Find out which case we're dealing with
			int *d_row = (int *)((char*)d + l*pitch);
			int src_level = d_row[src];
			int dst_level = d_row[dst];

			if(abs(src_level-dst_level)==0) 
			{
				//Case 1 or 5
				recompute = 0;
			}
			else if((src_level == INT_MAX) || (dst_level == INT_MAX))
			{
				//Case 4 - Either one or both nodes is in a different connected component, but not both
				recompute = 3;
			}
			else if(abs(src_level-dst_level) == 1)
			{
				//Case 2
				recompute = 1;
				if(src_level > dst_level)
				{
					u_low = src;
					u_high = dst;
				}
				else
				{
					u_high = src;
					u_low = dst;
				}
			}
			else
			{
				//Case 3 
				recompute = 2;
				if(src_level > dst_level)
				{
					u_low = src;
					u_high = dst;
				}
				else
				{
					u_high = src;
					u_low = dst;
				}
			}
		}
		__syncthreads();

		if(recompute==1)
		{
			__shared__ int *touched;
			__shared__ int *dP;
			__shared__ unsigned long long *sigma_hat;
			__shared__ float *delta_hat;
			__shared__ bool *QQ;

			if(j==0)
			{
				touched = (int *) malloc(n*sizeof(int));
				dP = (int *) malloc(n*sizeof(int));
				sigma_hat = (unsigned long long *) malloc(n*sizeof(unsigned long long));
				delta_hat = (float *) malloc(n*sizeof(float));
				QQ = (bool *) malloc (n*sizeof(bool));
			}

			__syncthreads();

			if((touched == NULL) || (dP == NULL) || (sigma_hat == NULL) || (delta_hat == NULL) || (QQ == NULL))
			{
				asm("trap;");
			}

			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				if(k==u_low)
				{
					touched[k] = -1;
					unsigned long long *sigma_row = (unsigned long long*)((char*)sigma + l*pitch_sigma);
					dP[k] = sigma_row[u_high];
					sigma_hat[k] = sigma_row[k] + dP[k];
					QQ[k] = true;
				}
				else
				{
					touched[k] = 0;
					dP[k] = 0;	
					unsigned long long *sigma_row = (unsigned long long*)((char*)sigma + l*pitch_sigma);
					sigma_hat[k] = sigma_row[k];
					QQ[k] = false;
				}

				delta_hat[k] = 0;
			}

			int *d_row = (int *)((char*)d + l*pitch);
			int current_depth = d_row[u_low];
			__shared__ bool done;
			if(j == 0)
			{
				done = false;
			}
			__syncthreads();

			while(!done)
			{
				__syncthreads();
				done = true;
				__syncthreads();

				for(int k=threadIdx.x; k<2*m; k+=blockDim.x)
				{
					int v = F[k];
					int *d_row = (int *)((char*)d + l*pitch);
					if(d_row[v] == current_depth)
					{
						int w = C[k];
						if(d_row[w] == (current_depth + 1)) 
						{
							//if(touched[w] == 0) //Thread safe? This might need a test and set of some sort
							if(atomicExch(&touched[w],-1) == 0)
							{
								//touched[w] = -1;
								done = false;
								dP[w] = dP[v];
								QQ[w] = true;
							}
							else
							{
								atomicAdd(&dP[w],dP[v]);
								//dP[w] += dP[v];
							}
							atomicAdd(&sigma_hat[w],dP[v]);
						}
					}
				}

				__syncthreads();
				current_depth++;
			}

			__syncthreads();
			current_depth--;
			while(current_depth > 0)
			{
				for(int k=threadIdx.x; k<2*m; k+=blockDim.x)
				{
					int w = F[k];
					int *d_row = (int *)((char*)d + l*pitch);
					if(d_row[w] == current_depth)
					{
						int v = C[k];
						if((d_row[w] == (d_row[v]+1)) && (QQ[w]))
						{
							//if(touched[v] == 0)
							if((touched[v] != -1) && (atomicExch(&touched[v],1) == 0)) //Can't use !atomicExch() here because touched could be -1 to begin with
							{
								touched[v] = 1;
								float *delta_row = (float*)((char*)delta + l*pitch); //This is constant for each root
								atomicAdd(&delta_hat[v],delta_row[v]);
								QQ[v] = true;
							}
							float new_change = (sigma_hat[v]/(float)sigma_hat[w])*(1+delta_hat[w]);
							atomicAdd(&delta_hat[v],new_change); 
							if((touched[v] == 1) && ((v != u_high) || (w != u_low)))
							{
								float *delta_row = (float*)((char*)delta + l*pitch);
								unsigned long long *sigma_row = (unsigned long long*)((char*)sigma + l*pitch_sigma);
								float old_change = (sigma_row[v]/(float)sigma_row[w])*(1+delta_row[w]);
								//Again faking an atomicSub here
								atomicAdd(&delta_hat[v],-1*old_change);  
							}
						}
					}
				}
				__syncthreads();
				current_depth--;
			}

			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				if((k != i) && (touched[k]!=0)) //Don't count the source node
				{
					float *delta_row = (float*)((char*)delta + l*pitch);
					float delta_change = delta_hat[k] - delta_row[k];
					atomicAdd(&bc[k],delta_change); //Does this need to be atomic?
				}
			}
			//__syncthreads();

			//Copy back for the next update (and for debugging purposes)
			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				//d doesn't change for this case
				unsigned long long *sigma_row = (unsigned long long*)((char*)sigma + l*pitch_sigma);
				float *delta_row = (float*)((char*)delta + l*pitch);
				sigma_row[k] = sigma_hat[k];
				if(touched[k] != 0)
				{
					delta_row[k] = delta_hat[k];
				}
			}
			
			__syncthreads();

			if(j == 0)
			{
				free(touched);
				free(dP);
				free(sigma_hat);
				free(delta_hat);
				free(QQ);
			}
		}
		else if(recompute==2)
		{
			__shared__ int *touched_row;
			__shared__ unsigned long long *sigma_hat_row;	
			__shared__ float *delta_hat_row;
			__shared__ int *moved_row;
			__shared__ int *movement_row;
			__shared__ unsigned long long *dP;
			__shared__ int *Q;
			__shared__ int *Q2;
			__shared__ int *QQ;
			__shared__ int *d_row;
			__shared__ unsigned long long *sigma_row;
			__shared__ float *delta_row;

			if(j==0)
			{
				touched_row = (int*) malloc(sizeof(int)*n); //Keeping old names for convenience
				sigma_hat_row = (unsigned long long*) malloc(sizeof(unsigned long long)*n);
				delta_hat_row = (float*) malloc(sizeof(float)*n);
				moved_row = (int*) malloc(sizeof(int)*n);
				movement_row = (int*) malloc(sizeof(int)*n);
				dP = (unsigned long long *) malloc(sizeof(unsigned long long)*n);
				Q = (int *) malloc(sizeof(int)*n); 
				Q2 = (int *) malloc(sizeof(int)*n);
				QQ = (int *) malloc(sizeof(int)*n);
				d_row = (int*)((char*)d + l*pitch);
				sigma_row = (unsigned long long*)((char*)sigma + l*pitch_sigma);
				delta_row = (float*)((char*)delta + l*pitch);
			}

			__syncthreads();
			
			if((touched_row == NULL) || (dP == NULL) || (sigma_hat_row == NULL) || (delta_hat_row == NULL) || (moved_row == NULL) || (movement_row == NULL) || (Q == NULL) || (Q2 == NULL) || (QQ == NULL))
			{
				asm("trap;");
			}

			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				if(k==u_low)
				{
					touched_row[k] = -1;
					sigma_hat_row[k] = sigma_row[u_high];
					moved_row[k] = 1;
					movement_row[k] = d_row[u_low]-d_row[u_high] - 1;
					dP[k] = sigma_row[u_high];
					Q[k] = 1;
				}
				else
				{
					touched_row[k] = 0;
					sigma_hat_row[k] = 0;
					moved_row[k] = 0;
					movement_row[k] = 0;
					dP[k] = 0;
					Q[k] = 0;
				}

				delta_hat_row[k] = 0;
				QQ[k] = 0;
				Q2[k] = 0;
			}

			__shared__ int current_depth;
			__shared__ bool done;
			if(j == 0)
			{
				current_depth = d_row[u_low];
				done = false;
			}
			__syncthreads();

			while(!done)
			{
				__syncthreads();
				done = true;
				__syncthreads();

				for(int v=threadIdx.x; v<n; v+=blockDim.x)
				{
					if(Q[v]) 
					{
						for(int k=R[v]; k<R[v+1]; k++) //For all neighbors of v...
						{
							int w = C[k];
							int computed_distance = (movement_row[v] - movement_row[w]) - (d_row[v] - d_row[w] + 1);
							if(computed_distance > 0)
							{
								atomicAdd(&(dP[w]),dP[v]);
								atomicAdd(&(sigma_hat_row[w]),dP[v]);
								moved_row[w] = 1;
								atomicMax(&(movement_row[w]),computed_distance);
								if(touched_row[w] == 0)
								{
									touched_row[w] = -1;
									done = false;
									Q2[w] = 1;
								}
							}
							else if((computed_distance==0) && (atomicExch(&(touched_row[w]),-1)==0))
							{
								atomicAdd(&(dP[w]),dP[v]);
								atomicAdd(&(sigma_hat_row[w]),dP[v]);
								done = false;
								Q2[w] = 1;
							}
							else if(touched_row[w] == -1) 
							{
								if(computed_distance >= 0)
								{
									atomicAdd(&(sigma_hat_row[w]),dP[v]);
									atomicAdd(&(dP[w]),dP[v]);
								}
							}
						}

						d_row[v] -= movement_row[v];
						Q[v] = 0;
						QQ[v] = 1;
					}
				}

				__syncthreads();
				
				for(int k=threadIdx.x; k<n; k+=blockDim.x)
				{
					if(Q2[k] == 1)
					{
						Q[k] = 1;
						Q2[k] = 0;
					}
				}

				__syncthreads();

				if(j==0)
				{
					current_depth++;
				}
			}

			__syncthreads();
			
			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				if(((k!=u_low) && (moved_row[k]==0)) || (touched_row[k]==0))
				{
					sigma_hat_row[k] += sigma_row[k];
				}
				Q2[k] = 0;
			}

			__syncthreads();
			
			current_depth--;
			__shared__ bool repeat;
			if(j==0)
			{
				repeat = false;
			}
			__syncthreads();
			while(current_depth > 0)
			{
				for(int k=threadIdx.x; k<2*m; k+=blockDim.x)
				{
					int w = F[k];
					float dsv = 0;
					if((QQ[w]) && (d_row[w] == current_depth)) 
					{
						int v = C[k];
						if(d_row[v] == (current_depth-1))
						{
							if((touched_row[v] != -1) && (atomicExch(&(touched_row[v]),1) == 0)) //atomicCAS here?
							{
								dsv = delta_row[v];
								touched_row[v] = 1;
								QQ[v] = 1; //Checking for depth should handle this
							}
							float new_change = (sigma_hat_row[v]/(float)sigma_hat_row[w])*(1+delta_hat_row[w]);
							dsv += new_change;
							if((touched_row[v] > 0) && ((v!=u_high) || (w!=u_low)))
							{
								float old_change = -1*(sigma_row[v]/(float)sigma_row[w])*(1+delta_row[w]);
								dsv += old_change;
							}
							atomicAdd(&(delta_hat_row[v]),dsv);
						}
						else if((d_row[v] == current_depth) && (moved_row[w]) && (!moved_row[v])) //Sometimes we get in this block when we shouldn't
						{
							if((touched_row[v] != -1) && (atomicExch(&(touched_row[v]),1) == 0))
							{
								dsv = delta_row[v];
								touched_row[v] = 1;
								//Need to repeat this level
								QQ[v] = 2;
								repeat = true;
								//current_depth++;
							}
							float old_change = -1*(sigma_row[v]/(float)sigma_row[w])*(1+delta_row[w]);
							dsv += old_change;
							atomicAdd(&(delta_hat_row[v]),dsv);
						}
					}
				}
				__syncthreads();	
			
				if(repeat)
				{
					if(j==0)
					{
						for(int k=0; k<n; k++)
						{
							if((QQ[k]>0)&&(d_row[k] == current_depth))
							{
								QQ[k]--;
							}
						}
						current_depth++;
						repeat = false;
					}
				}
				__syncthreads();
				current_depth--;
			}

			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				if(delta_hat_row[k] < 0)
				{
					delta_hat_row[k] = 0;
				}
			}

			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				if((k != i) && (touched_row[k]!=0)) //Don't count the source node
				{
					float delta_change = delta_hat_row[k] - delta_row[k];
					atomicAdd(&bc[k],delta_change); //Does this need to be atomic?
				}
			}

			//Copy back for the next update (and for debugging purposes)
			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				//d was updated in place
				sigma_row[k] = sigma_hat_row[k];
				if(touched_row[k] != 0)
				{
					delta_row[k] = delta_hat_row[k];
				}
			}

			__syncthreads();

			if(j==0)
			{				
				free(touched_row);
				free(sigma_hat_row);
				free(delta_hat_row);
				free(moved_row);
				free(movement_row);
				free(dP);
				free(Q);
				free(Q2);
				free(QQ);
			}
		}
		else if(recompute==3)
		{
			//Subtract off current value of delta. atomicSub doesn't have a float overload.
			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				if(k != i) //Don't count the source node
				{
					float *delta_row = (float*)((char*)delta + l*pitch);
					atomicAdd(&bc[k],-1*delta_row[k]); //Does this need to be atomic?
				}
			}

			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				int *d_row = (int*)((char*)d + l*pitch);
				unsigned long long *sigma_row = (unsigned long long*)((char*)sigma + l*pitch_sigma);
				if(k == i) //If its the source node
				{
					sigma_row[k] = 1;
					d_row[k] = 0;
				}
				else
				{
					sigma_row[k] = 0;
					d_row[k] = INT_MAX;
				}

				float *delta_row = (float*)((char*)delta + l*pitch);
				delta_row[k] = 0;
			}
	
			int current_depth = 0;
			__shared__ bool done;
			if(j == 0)
			{
				done = false;
			}
			__syncthreads();

			while(!done)
			{
				__syncthreads();
				done = true;
				__syncthreads();

				for(int k=threadIdx.x; k<2*m; k+=blockDim.x)
				{
					int v = F[k];
					int *d_row = (int *)((char*)d + l*pitch);
					if(d_row[v] == current_depth)
					{
						int w = C[k];
						if(d_row[w] == INT_MAX)
						{
							d_row[w] = current_depth + 1; 
							done = false;
						}
						if(d_row[w] == (current_depth + 1)) 
						{
							unsigned long long *sigma_row = (unsigned long long*)((char*)sigma + l*pitch_sigma);
							atomicAdd(&sigma_row[w],sigma_row[v]);
						}
					}
				}

				__syncthreads();
				current_depth++;
			}

			__syncthreads();
			current_depth--;
			while(current_depth > 0)
			{
				for(int k=threadIdx.x; k<2*m; k+=blockDim.x)
				{
					int w = F[k];
					int *d_row = (int *)((char*)d + l*pitch);
					if(d_row[w] == current_depth)
					{
						int v = C[k];
						if(d_row[w] == (d_row[v]+1))
						{
							float *delta_row = (float*)((char*)delta + l*pitch);
							unsigned long long *sigma_row = (unsigned long long*)((char*)sigma + l*pitch_sigma);
							float change = (sigma_row[v]/(float)sigma_row[w])*(1.0f+delta_row[w]);
							atomicAdd(&delta_row[v],change);
						}
					}
				}
				__syncthreads();
				current_depth--;
			}

			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				if(k != i) //Don't count the source node
				{
					float *delta_row = (float*)((char*)delta + l*pitch);
					atomicAdd(&bc[k],delta_row[k]); //Does this need to be atomic?
				}
			}
			__syncthreads();
		}
	}
}

template __global__ void bc_gpu_update_edge<false>(float *bc, const int *__restrict__ R, const int *__restrict__ F, const int *__restrict__ C, const int n, const int m, int *__restrict__ d, unsigned long long *__restrict__ sigma, float *__restrict__ delta, size_t pitch, size_t pitch_sigma, const int *__restrict__ sources, const int start, const int end, const int src, const int dst);
template __global__ void bc_gpu_update_edge<true>(float *bc, const int *__restrict__ R, const int *__restrict__ F, const int *__restrict__ C, const int n, const int m, int *__restrict__ d, unsigned long long *__restrict__ sigma, float *__restrict__ delta, size_t pitch, size_t pitch_sigma, const int *__restrict__ sources, const int start, const int end, const int src, const int dst);

//Optimized kernel
/*List of optimizations:
 1) d_row should be a shared int pointer (in fact, d_row, sigma_row, and delta_row are all constant with the block and should probably have a wider scope)
 2) removal of dP from case 2
 3) removal of repeated computations of d_row, sigma_row, and delta_row
 4) removal of QQ from case 2
 5) removal of malloc from case 2. Preallocate the worst case before the kernel.
 6) workload stealing for coarse-grain parallelism
*/

//Helper functions for linear BFS - including really sad template hacks. Cuda 5.0+ shouldn't require that though.
template<bool bogus>
__device__ int next_power_of_two(int N)
{
	int M = N-1;
	M |= M >> 1;
	M |= M >> 2;
	M |= M >> 4;
	M |= M >> 8;
	M |= M >> 16;
	M++;

	return M;
}

template __device__ int next_power_of_two<true>(int N);

template<bool bogus>
__device__ void bitonic_sort(int *values, int M)
{
	unsigned int idx = threadIdx.x;

	for (int k = 2; k <= M; k <<= 1) 
	{
		for (int j = k >> 1; j > 0; j = j >> 1) 
		{
		 	while(idx < M) {
		 		int ixj = idx^j;
		  		if (ixj > idx) {
		 			if ((idx&k) == 0 && values[idx] > values[ixj]) {
						//exchange(idx, ixj);
						int tmp = values[idx];
						values[idx] = values[ixj];
						values[ixj] = tmp;
					}
					if ((idx&k) != 0 && values[idx] < values[ixj]) {
						//exchange(idx, ixj);
						int tmp = values[idx];
						values[idx] = values[ixj];
						values[ixj] = tmp;
					}
		 		}
				idx += blockDim.x;	
		 	}
			__syncthreads();
			idx = threadIdx.x;
		}
	}
}

template __device__ void bitonic_sort<true>(int *values, int M);

template <bool bogus>
__device__ void find_duplicates(int *values, int *pred, int M)
{
 	unsigned int idx = threadIdx.x;
	
	//Now the array is sorted, so find duplicates
	while(idx < M)
	{
		if(idx == 0)
		{
			pred[idx] = 1;
		}
		else
		{
			if(values[idx] == values[idx-1])
			{
				pred[idx] = 0;
			}
			else
			{
				pred[idx] = 1;
			}
		}
		idx += blockDim.x;
	}
}

template __device__ void find_duplicates<true>(int *values, int *pred, int M);

template <bool bogus>
__device__ void prefix_sum(int *pred, int *temp, int M)
{
 	unsigned int idx = threadIdx.x;

	//Now do an inclusive prefix sum (note: this isn't work optimal)
	int pout = 0, pin = 1;
	while(idx < M)
	{
		temp[idx] = pred[idx];
		idx += blockDim.x;
	}
	idx = threadIdx.x;
	__syncthreads();
	for(int offset=1; offset < M; offset <<= 1)
	{
		pout = 1-pout;
		pin = 1-pin;
		__syncthreads();
		while(idx < M)
		{
			temp[pout*M+idx] = temp[pin*M+idx];

			if(idx >= offset)
			{
				temp[pout*M+idx] += temp[pin*M+idx - offset];
			}
			idx += blockDim.x;
		}
		idx = threadIdx.x;
	}
	while(idx < M)
	{
		pred[idx] = temp[pout*M+idx];
		idx += blockDim.x;
	}
}
template __device__ void prefix_sum<true>(int *pred, int *temp, int M);

//Template parameters:
// approx - do we want an exact or approximate BC calculation?
// node - do we want node-based parallelism or edge-based parallelism? 

#define SHARED_Q_SIZE 3050
template<bool approx, int node>
__global__ void bc_gpu_update_edge_opt(float *bc, const int *__restrict__ R, const int *__restrict__ F, const int *__restrict__ C, const int n, const int m, int *__restrict__ d, unsigned long long *__restrict__ sigma, float *__restrict__ delta, int *touched, unsigned long long *sigma_hat, float *delta_hat, int *moved, int *movement, int *Q, int *Q2, int *QQ, int *temp, int *taken, size_t pitch, size_t pitch_sigma, size_t pitch_Q, size_t pitch_temp, const int *__restrict__ sources, const int start, const int end, const int src, const int dst)
{
	__shared__ int l;
	int j = threadIdx.x;
	if(j == 0)
	{
		l = blockIdx.x+start;
	}
	__syncthreads();
	//for(l=blockIdx.x+start; l<end; l++)
	while(l < end)
	{
		__shared__ int i;
		if(j==0)
		{
			if(approx)
			{
				i = sources[l]; //i is the absolute value of the source node, l is the relative value
			}
			else
			{
				i = l;
			}
		}
		__syncthreads();

		__shared__ bool compute;
		if(j == 0)
		{
			if(atomicExch(&taken[l],1) == 0)
			{
				compute = true;
			}
			else
			{
				compute = false;
			}
		}
		__syncthreads();

		if(!compute)
		{
			if(j==0)
			{
				l++;
			}
			__syncthreads();

			continue;
		}
		__shared__ int recompute;
		__shared__ int u_low;
		__shared__ int u_high;

		__shared__ int *d_row;
		__shared__ unsigned long long *sigma_row;
		__shared__ float *delta_row;

		__shared__ int src_level, dst_level;

		if(j == 0)
		{
			d_row = (int*)((char*)d + l*pitch);
			sigma_row = (unsigned long long*)((char*)sigma + l*pitch_sigma);
			delta_row = (float*)((char*)delta + l*pitch); 
				
			//Find out which case we're dealing with
			src_level = d_row[src];
			dst_level = d_row[dst];

			if(abs(src_level-dst_level)==0) 
			{
				//Case 1 or 5
				recompute = 0;
			}
			else if((src_level == INT_MAX) || (dst_level == INT_MAX))
			{
				//Case 4 - Either one or both nodes is in a different connected component, but not both
				recompute = 2;
				if(src_level > dst_level)
				{
					u_low = src;
					u_high = dst;
				}
				else
				{
					u_high = src;
					u_low = dst;
				}
			}
			else if(abs(src_level-dst_level) == 1)
			{
				//Case 2
				recompute = 1;
				if(src_level > dst_level)
				{
					u_low = src;
					u_high = dst;
				}
				else
				{
					u_high = src;
					u_low = dst;
				}
			}
			else
			{
				//Case 3 
				recompute = 2;
				if(src_level > dst_level)
				{
					u_low = src;
					u_high = dst;
				}
				else
				{
					u_high = src;
					u_low = dst;
				}
			}
		}
		__syncthreads();

		__shared__ int Q_len;
		__shared__ int Q2_len;
		__shared__ int QQ_len;
		__shared__ int Q_shared[SHARED_Q_SIZE]; //If the frontier is small enough, use shared memory
		__shared__ int Q2_shared[SHARED_Q_SIZE];

		if(recompute==1)
		{
			__shared__ int *touched_row;
			__shared__ unsigned long long *sigma_hat_row;	
			__shared__ float *delta_hat_row;
			//Test linear BFS
			__shared__ int *Q_row; //Old frontier
			__shared__ int *Q2_row; //New frontier
			__shared__ int *temp_row; //temp variable for prefix sum
			//Test dependency linear work
			__shared__ int *QQ_row; //Stack

			if(j==0)
			{
				touched_row = (int*)((char*)touched + blockIdx.x*pitch);
				sigma_hat_row = (unsigned long long*)((char*)sigma_hat + blockIdx.x*pitch_sigma);
				delta_hat_row = (float*)((char*)delta_hat + blockIdx.x*pitch); 
				Q_row = (int*)((char*)Q + blockIdx.x*pitch_Q);
				Q2_row = (int*)((char*)Q2 + blockIdx.x*pitch_Q);
				temp_row = (int *)((char*)temp + blockIdx.x*pitch_temp);
				QQ_row = (int *)((char*)QQ + blockIdx.x*pitch);
			}

			__syncthreads();

			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				if(k==u_low)
				{
					touched_row[k] = -1;
					sigma_hat_row[k] = sigma_row[k] + sigma_row[u_high];
				}
				else
				{
					touched_row[k] = 0;
					sigma_hat_row[k] = sigma_row[k];
				}

				delta_hat_row[k] = 0;
				Q_row[k] = -1;
				Q2_row[k] = -1;
				QQ_row[k] = -1;
			}

			__shared__ int current_depth;
			__shared__ bool done;
		        __shared__ int temp[2*SHARED_Q_SIZE];	
			if(j == 0)
			{
				current_depth = d_row[u_low];
				done = false;
				Q_shared[0] = u_low; //Frontier starts at size 1, so we know we'll be using the shared array
				Q_len = 1;
				Q2_len = 0;
				QQ_row[0] = u_low;
				QQ_len = 1;
			}
			__syncthreads();

			if(node >= 1)
			{
				int k = threadIdx.x;
				while(1)
				{
					if(k < Q_len)
					{
						int v = (k < SHARED_Q_SIZE) ? Q_shared[k] : Q_row[k]; //Wasting space for now
						for(int r=R[v]; r<R[v+1]; r++)
						{
							int w = C[r];
							if(d_row[w] == (d_row[v]+1))
							{
								if(touched_row[w] == 0) //Now we can have duplicate entries in the queue. Queues might be too small also.
								{
									int t = atomicAdd(&Q2_len,1);
									if(t < SHARED_Q_SIZE)
									{
										Q2_shared[t] = w;
									}
									else
									{
										Q2_row[t] = w;	
									}
									//Push into the queue
									//touched_row[w] = -1;
								}
								atomicAdd(&(sigma_hat_row[w]),(sigma_hat_row[v]-sigma_row[v]));
							}
						}
					}
					__syncthreads();
					if(j==0)
					{
						Q_len -= blockDim.x;
					}
					__syncthreads();

					if(Q_len <= 0)
					{
						if(Q2_len == 0)
						{
							break;
						}
						else
						{
							for(int m=threadIdx.x; m<Q2_len; m+=blockDim.x)
							{
								if(m < SHARED_Q_SIZE)
								{
									touched_row[Q2_shared[m]] = -1;
								}
								else
								{
									touched_row[Q2_row[m]] = -1;
								}
							}
							//Remove duplicates from Q2
							int M;
							if((Q2_len&&(Q2_len-1)) == 0)
							{
								M = Q2_len;
							}
							else
							{
								M = next_power_of_two<true>(Q2_len);
								//Alleviate noise in sorting
								for(int m=threadIdx.x+Q2_len; m<M; m+=blockDim.x)
								{
									if(m < SHARED_Q_SIZE)
									{
										Q2_shared[m] = INT_MAX;	
									}
									else
									{
										Q2_row[m] = INT_MAX;
									}
								}
							}
							__syncthreads();

							if(M >= SHARED_Q_SIZE)
							{
								//Copy the rest of the queue into global memory
					//NOTE: For graphs with lots of edges/duplicates, this will break because the intermediate arrays are not large enough
								for(int m=threadIdx.x; m<SHARED_Q_SIZE; m+=blockDim.x)
								{
									Q2_row[m] = Q2_shared[m];
								}
								//Sort Q2
								bitonic_sort<true>(Q2_row,M);
								find_duplicates<true>(Q2_row,Q_row,M);
								__syncthreads();
								prefix_sum<true>(Q_row,temp_row,M);
								__syncthreads();

								if(j==0)
								{
									Q_len = Q_row[Q2_len-1];
									Q2_len = 0;
								}
								__syncthreads();
								for(int m=threadIdx.x; m<M; m+=blockDim.x)
								{
									if(m==0)
									{
										if(Q_row[m] == 1)
										{
											temp_row[Q_row[m]-1] = Q2_row[m];
										}
									}
									else
									{
										if((Q_row[m] - Q_row[m-1]) == 1)
										{
											temp_row[Q_row[m]-1] = Q2_row[m];
										}
									}
								}
								__syncthreads();
								for(int m=threadIdx.x; m<Q_len; m+=blockDim.x)
								{
									Q_row[m] = temp_row[m];
									//Push into the stack
									int t = atomicAdd(&QQ_len,1);
									QQ_row[t] = temp_row[m];
								}

								k = threadIdx.x;
							}
							else
							{
								//1) Sort Q2
								bitonic_sort<true>(Q2_shared,M);
								//2) Mark which indices are unique, store them in Q
								find_duplicates<true>(Q2_shared,Q_shared,M);
								__syncthreads();
								//3) Prefix sum to see where the unique elements are to be placed
								prefix_sum<true>(Q_shared,temp,M);
								__syncthreads();

								if(j==0)
								{
									Q_len = Q_shared[Q2_len-1];
									Q2_len = 0;
								}
								__syncthreads();
								//The limit on this loop should be Q2_len BEFORE it is reset. Using M is still safe.
								for(int m=threadIdx.x; m<M; m+=blockDim.x)
								{
									if(m==0)
									{
										if(Q_shared[m] == 1)
										{
											temp[Q_shared[m]-1] = Q2_shared[m];
										}
									}
									else
									{
										if((Q_shared[m] - Q_shared[m-1]) == 1)
										{
											temp[Q_shared[m]-1] = Q2_shared[m];
										}
									}
								}
								__syncthreads();
								for(int m=threadIdx.x; m<Q_len; m+=blockDim.x)
								{
									Q_shared[m] = temp[m];
									//Push into the stack
									int t = atomicAdd(&QQ_len,1);
									QQ_row[t] = temp[m];
								}
								k = threadIdx.x;
							}
						}
					}
					else
					{
						k += blockDim.x;
					}
					__syncthreads();
				}
				__syncthreads();
				
				for(int m=threadIdx.x; m<n; m+=blockDim.x)
				{
					atomicMax(&current_depth,d_row[m]+1);
				}
			}
			else
			{
				while(!done)
				{
					__syncthreads();
					done = true;
					__syncthreads();
					
					for(int k=threadIdx.x; k<2*m; k+=blockDim.x)
					{
						int v = F[k];
						if(d_row[v] == current_depth)
						{
							int w = C[k];
							if(d_row[w] == (current_depth + 1)) 
							{
								if(touched_row[w] == 0)
								{
									touched_row[w] = -1;
									done = false;
								}
								atomicAdd(&(sigma_hat_row[w]),(sigma_hat_row[v]-sigma_row[v]));
							}
						}
					}

					__syncthreads();
					if(current_depth != INT_MAX)
					{
						current_depth++;
					}
					
				}
			}

			__syncthreads();
			if(j==0)
			{
				current_depth--;
			}
			__syncthreads();
			if(node == 0)
			{
				while(current_depth > 1)
				{
					for(int k=threadIdx.x; k<2*m; k+=blockDim.x)
					{
						int w = F[k];
						if(d_row[w] == current_depth)
						{
							int v = C[k];
							if(d_row[v] == (current_depth-1))
							{
								float dsv = 0;
								if((touched_row[v] != -1) && (atomicExch(&(touched_row[v]),1) == 0)) //Can't use !atomicExch() here because touched could be -1 to begin with
								{
									dsv += delta_row[v];
								}
								float new_change = (sigma_hat_row[v]/(float)sigma_hat_row[w])*(1+delta_hat_row[w]);
								dsv += new_change;
								if((touched_row[v] == 1) && ((v != u_high) || (w != u_low)))
								{
									float old_change = -1*(sigma_row[v]/(float)sigma_row[w])*(1+delta_row[w]);
									dsv += old_change;
								}
								atomicAdd(&(delta_hat_row[v]),dsv);
							}
						}
					}
					__syncthreads();
					if(j==0)
					{
						current_depth--;
					}
					__syncthreads();
				}
			}
			else
			{
				if(j==0)
				{
					Q2_len = 0;
				}
				__syncthreads();
				while(current_depth > 1)
				{
					//Node level parallelism w/o atomics
					//For better work complexity, try QQ_len
					for(int m=threadIdx.x; m<QQ_len; m+=blockDim.x)
					{
						int w = QQ_row[m];
						if((w != -1) && (d_row[w] == current_depth))
						{
							for(int k=R[w]; k<R[w+1]; k++) //For all neighbors of w...
							{
								int v = C[k];
								float dsv = 0;
								if(d_row[v] == (current_depth-1)) 
								{
									//if(touched_row[v] == 0)
									if((touched_row[v] != -1) && (atomicExch(&(touched_row[v]),1) == 0)) 
									{
										dsv += delta_row[v];
										//Push onto stack - Don't need a second queue because anything added is at a lower depth
										int t = atomicAdd(&Q2_len,1);
										QQ_row[t+QQ_len] = v;
									}
									dsv += (sigma_hat_row[v]/(float)sigma_hat_row[w])*(1+delta_hat_row[w]);
									if((touched_row[v] == 1) && ((v != u_high) || (w != u_low)))
									{
										dsv -= (sigma_row[v]/(float)sigma_row[w])*(1+delta_row[w]);
									}
								}
								atomicAdd(&(delta_hat_row[v]),dsv);
							}
						}
					}
					__syncthreads();
					if(j==0)
					{
						QQ_len += Q2_len;
						Q2_len = 0;
						current_depth--;
					}
					__syncthreads();
				}
			}

			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				if((k != i) && (touched_row[k]!=0)) //Don't count the source node
				{
					float delta_change = delta_hat_row[k] - delta_row[k];
					atomicAdd(&bc[k],delta_change); //Does this need to be atomic?
				}
				sigma_row[k] = sigma_hat_row[k];
				if(touched_row[k] != 0)
				{
					delta_row[k] = delta_hat_row[k];
				}
			}

			__syncthreads();
		}
		else if(recompute==2)
		{
			__shared__ int *touched_row;
			__shared__ unsigned long long *sigma_hat_row;	
			__shared__ float *delta_hat_row;
			__shared__ int *moved_row;
			__shared__ int *movement_row;
			__shared__ int *Q_row;
			__shared__ int *Q2_row;
			__shared__ int *QQ_row;

			if(j==0)
			{
				touched_row = (int*)((char*)touched + blockIdx.x*pitch);
				sigma_hat_row = (unsigned long long*)((char*)sigma_hat + blockIdx.x*pitch_sigma);
				delta_hat_row = (float*)((char*)delta_hat + blockIdx.x*pitch); 
				moved_row = (int*)((char*)moved + blockIdx.x*pitch);
				movement_row = (int*)((char*)movement + blockIdx.x*pitch);
				Q_row = (int*)((char*)Q + blockIdx.x*pitch_Q);
				Q2_row = (int*)((char*)Q2 + blockIdx.x*pitch_Q);
				QQ_row = (int*)((char*)QQ + blockIdx.x*pitch);
			}

			__syncthreads();

			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				if(k==u_low)
				{
					touched_row[k] = -1;
					sigma_hat_row[k] = sigma_row[u_high];
					moved_row[k] = 1;
					movement_row[k] = d_row[u_low]-d_row[u_high] - 1;
					Q_row[k] = 1;
				}
				else
				{
					touched_row[k] = 0;
					sigma_hat_row[k] = 0;
					moved_row[k] = 0;
					movement_row[k] = 0;
					Q_row[k] = 0;
				}

				delta_hat_row[k] = 0;
				QQ_row[k] = 0;
				Q2_row[k] = 0;
			}

			__shared__ int current_depth;
			__shared__ bool done;
			if(j == 0)
			{
				current_depth = d_row[u_low]; 
				done = false;
			}
			__syncthreads();

			if(node!=1)
			{
				while(!done)
				{
					__syncthreads();
					done = true;
					__syncthreads();

					for(int k=threadIdx.x; k<2*m; k+=blockDim.x)
					{
						int v = F[k];
						if(Q_row[v])
						{
							int w = C[k];
							int computed_distance = (movement_row[v] - movement_row[w]) - (d_row[v] - d_row[w] + 1);
							if(computed_distance > 0)
							{
								atomicAdd(&(sigma_hat_row[w]),sigma_hat_row[v]);
								moved_row[w] = 1;
								atomicMax(&(movement_row[w]),computed_distance);
								if(touched_row[w] == 0)
								{
									touched_row[w] = -1;
									done = false;
									Q2_row[w] = 1;
								}
							}
							else if((computed_distance==0) && (atomicExch(&(touched_row[w]),-1)==0))
							{
								atomicAdd(&(sigma_hat_row[w]),sigma_hat_row[v]);
								done = false;
								Q2_row[w] = 1;
							}
							else if(touched_row[w] == -1) 
							{
								if(computed_distance >= 0)
								{
									atomicAdd(&(sigma_hat_row[w]),sigma_hat_row[v]);
								}
							}
						}
					}

					__syncthreads();

					for(int k=threadIdx.x; k<n; k+=blockDim.x)
					{
						if(Q_row[k])
						{	
							Q_row[k] = 0;
							d_row[k] -= movement_row[k];
							QQ_row[k] = 1;
						}
					}

					__syncthreads();
					
					for(int k=threadIdx.x; k<n; k+=blockDim.x)
					{
						if(Q2_row[k] == 1)
						{
							Q_row[k] = 1;
							Q2_row[k] = 0;
						}
					}

					__syncthreads();

					if((j==0) && (current_depth != INT_MAX)) //Prevent overflow
					{
						current_depth++;
					}
				}
			}
			else
			{
				while(!done)
				{
					__syncthreads();
					done = true;
					__syncthreads();

					for(int v=threadIdx.x; v<n; v+=blockDim.x)
					{
						if(Q_row[v]) 
						{
							for(int k=R[v]; k<R[v+1]; k++) //For all neighbors of v...
							{
								int w = C[k];
								int computed_distance = (movement_row[v] - movement_row[w]) - (d_row[v] - d_row[w] + 1);
								if(computed_distance > 0)
								{
									atomicAdd(&(sigma_hat_row[w]),sigma_hat_row[v]);
									moved_row[w] = 1;
									atomicMax(&(movement_row[w]),computed_distance);
									if(touched_row[w] == 0)
									{
										touched_row[w] = -1;
										done = false;
										Q2_row[w] = 1;
									}
								}
								else if((computed_distance==0) && (atomicExch(&(touched_row[w]),-1)==0))
								{
									atomicAdd(&(sigma_hat_row[w]),sigma_hat_row[v]);
									done = false;
									Q2_row[w] = 1;
								}
								else if(touched_row[w] == -1) 
								{
									if(computed_distance >= 0)
									{
										atomicAdd(&(sigma_hat_row[w]),sigma_hat_row[v]);
									}
								}
							}

							d_row[v] -= movement_row[v];
							Q_row[v] = 0;
							QQ_row[v] = 1;
						}
					}

					__syncthreads();
					
					for(int k=threadIdx.x; k<n; k+=blockDim.x)
					{
						if(Q2_row[k] == 1)
						{
							Q_row[k] = 1;
							Q2_row[k] = 0;
						}
					}

					__syncthreads();

					if((j==0) && (current_depth != INT_MAX)) //Prevent overflow
					{
						current_depth++;
					}
				}
			}

			__syncthreads();

			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				if(((k!=u_low) && (moved_row[k]==0)) || (touched_row[k]==0))
				{
					sigma_hat_row[k] += sigma_row[k];
				}
				Q2_row[k] = 0;
			}

			__syncthreads();
			
			//current_depth--;
			__shared__ bool repeat;
			if(j==0)
			{
				repeat = false;
			}
			__syncthreads();
			while(current_depth > 0)
			{
				if(node!=1)
				{
					for(int k=threadIdx.x; k<2*m; k+=blockDim.x)
					{
						int w = F[k];
						float dsv = 0;
						if((QQ_row[w]) && (d_row[w] == current_depth)) 
						{
							int v = C[k];
							if(d_row[v] == (current_depth-1))
							{
								if((touched_row[v] != -1) && (atomicExch(&(touched_row[v]),1) == 0)) //atomicCAS here?
								{
									dsv = delta_row[v];
									touched_row[v] = 1;
									QQ_row[v] = 1; 
								}
								float new_change = (sigma_hat_row[v]/(float)sigma_hat_row[w])*(1+delta_hat_row[w]);
								dsv += new_change;
								if((touched_row[v] > 0) && ((v!=u_high) || (w!=u_low)))
								{
									float old_change = -1*(sigma_row[v]/(float)sigma_row[w])*(1+delta_row[w]);
									dsv += old_change;
								}
								atomicAdd(&(delta_hat_row[v]),dsv);
							}
							else if((d_row[v] == current_depth) && (moved_row[w]) && (!moved_row[v])) //Sometimes we get in this block when we shouldn't
							{
								if((touched_row[v] != -1) && (atomicExch(&(touched_row[v]),1) == 0))
								{
									dsv = delta_row[v];
									touched_row[v] = 1;
									//Need to repeat this level
									QQ_row[v] = 2;
									repeat = true;
								}
								float old_change = -1*(sigma_row[v]/(float)sigma_row[w])*(1+delta_row[w]);
								dsv += old_change;
								atomicAdd(&(delta_hat_row[v]),dsv);
							}
						}
					}
				}
				else
				{
					for(int w=threadIdx.x; w<n; w+=blockDim.x)
					{
						if((QQ_row[w]) && (d_row[w] == current_depth)) 
						{
							for(int k=R[w]; k<R[w+1]; k++) //For all neighbors of w...
							{
								int v = C[k];
								float dsv = 0;
								if(d_row[v] == (current_depth-1))
								{
									if((touched_row[v] != -1) && (atomicExch(&(touched_row[v]),1) == 0)) //atomicCAS here?
									{
										dsv = delta_row[v];
										touched_row[v] = 1;
										QQ_row[v] = 1;
									}
									float new_change = (sigma_hat_row[v]/(float)sigma_hat_row[w])*(1+delta_hat_row[w]);
									dsv += new_change;
									if((touched_row[v] > 0) && ((v!=u_high) || (w!=u_low)))
									{
										float old_change = -1*(sigma_row[v]/(float)sigma_row[w])*(1+delta_row[w]);
										dsv += old_change;
									}
									atomicAdd(&(delta_hat_row[v]),dsv);
								}
								else if((d_row[v] == current_depth) && (moved_row[w]) && (!moved_row[v])) //Sometimes we get in this block when we shouldn't
								{
									if((touched_row[v] != -1) && (atomicExch(&(touched_row[v]),1) == 0))
									{
										dsv = delta_row[v];
										touched_row[v] = 1;
										//Need to repeat this level
										QQ_row[v] = 2;
										repeat = true;
									}
									float old_change = -1*(sigma_row[v]/(float)sigma_row[w])*(1+delta_row[w]);
									dsv += old_change;
									atomicAdd(&(delta_hat_row[v]),dsv);
								}
							}
						}
					}
				}
				__syncthreads();	
			
				if(repeat)
				{
					for(int k=threadIdx.x; k<n; k+=blockDim.x)
					{
						if((QQ_row[k]>0)&&(d_row[k] == current_depth))
						{
							QQ_row[k]--;
						}
					}
					if((j==0) && (current_depth != INT_MAX))
					{
						current_depth++;
						repeat = false;
					}
				}
				else
				{
					for(int k=threadIdx.x; k<n; k+=blockDim.x)
					{
						if((QQ_row[k]>0)&&(d_row[k] == current_depth))
						{
							QQ_row[k]--;
						}
					}
				}
				__shared__ bool QQfound; //Try to prevent useless while loop iterations. Could test to see if current_depth is 'large enough' such that this will be useful.
				__shared__ int maxQQ;
				if((current_depth == INT_MAX) && (!repeat))
				{
					if(j==0)
					{
						QQfound = false;
						maxQQ = 0;
					}
					__syncthreads();
					for(int k=threadIdx.x; k<n; k+=blockDim.x)
					{
						if(QQ_row[k] > 0)
						{
							QQfound = true;
							atomicMax(&maxQQ,d_row[k]); 
						}
					}
				}
				else
				{
					QQfound = true;
					maxQQ = current_depth;
				}
				__syncthreads();
				if(j==0)
				{
					if(repeat)
					{
						repeat = false;
					}
					else
					{
						if(!QQfound)
						{
							current_depth = 0; 
						}
						else
						{
							if(current_depth == INT_MAX)
							{
								current_depth = maxQQ;
							}
							else
							{
								current_depth--;
							}
						}
					}
				}
				__syncthreads();
			}

			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				if(delta_hat_row[k] < 0)
				{
					delta_hat_row[k] = 0;
				}

				if((k != i) && (touched_row[k]!=0)) //Don't count the source node
				{
					float delta_change = delta_hat_row[k] - delta_row[k];
					atomicAdd(&bc[k],delta_change); //Does this need to be atomic?
				}
				
				sigma_row[k] = sigma_hat_row[k];
				if(touched_row[k] != 0)
				{
					delta_row[k] = delta_hat_row[k];
				}
			}

			__syncthreads();
		}
		else if(recompute==3)
		{
			//Subtract off current value of delta. atomicSub doesn't have a float overload.
			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				if(k != i) //Don't count the source node
				{
					atomicAdd(&bc[k],-1*delta_row[k]); //Does this need to be atomic?
				}
			}

			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				if(k == i) //If its the source node
				{
					sigma_row[k] = 1;
					d_row[k] = 0;
				}
				else
				{
					sigma_row[k] = 0;
					d_row[k] = INT_MAX;
				}

				delta_row[k] = 0;
			}
	
			__shared__ int current_depth;
			__shared__ bool done;
			if(j == 0)
			{
				current_depth = 0;
				done = false;
			}
			__syncthreads();

			while(!done)
			{
				__syncthreads();
				done = true;
				__syncthreads();

				for(int k=threadIdx.x; k<2*m; k+=blockDim.x)
				{
					int v = F[k];
					if(d_row[v] == current_depth)
					{
						int w = C[k];
						if(d_row[w] == INT_MAX)
						{
							d_row[w] = current_depth + 1; 
							done = false;
						}
						if(d_row[w] == (current_depth + 1)) 
						{
							atomicAdd(&sigma_row[w],sigma_row[v]);
						}
					}
				}

				__syncthreads();
				current_depth++;
			}

			__syncthreads();
			current_depth--;
			while(current_depth > 0)
			{
				for(int k=threadIdx.x; k<2*m; k+=blockDim.x)
				{
					int w = F[k];
					if(d_row[w] == current_depth)
					{
						int v = C[k];
						if(d_row[w] == (d_row[v]+1))
						{
							float change = (sigma_row[v]/(float)sigma_row[w])*(1.0f+delta_row[w]);
							atomicAdd(&delta_row[v],change);
						}
					}
				}
				__syncthreads();
				current_depth--;
			}

			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				if(k != i) //Don't count the source node
				{
					atomicAdd(&bc[k],delta_row[k]); //Does this need to be atomic?
				}
			}
			__syncthreads();
		}
		
		//If its the first loop iteration, don't bother to check 0...gridDim.x-1 because they'll all be taken, one per SM
		__syncthreads();
		if(j == 0)
		{	
			if(l == blockIdx.x)
			{
				l = gridDim.x-1;
			}
			l++;
		}
		__syncthreads();
	}
}

template __global__ void bc_gpu_update_edge_opt<false,0>(float *bc, const int *__restrict__ R, const int *__restrict__ F, const int *__restrict__ C, const int n, const int m, int *__restrict__ d, unsigned long long *__restrict__ sigma, float *__restrict__ delta, int *touched, unsigned long long *sigma_hat, float *delta_hat, int *moved, int *movement, int *Q, int *Q2, int *QQ, int *temp, int *taken, size_t pitch, size_t pitch_sigma, size_t pitch_Q, size_t pitch_temp, const int *__restrict__ sources, const int start, const int end, const int src, const int dst);
template __global__ void bc_gpu_update_edge_opt<true,0>(float *bc, const int *__restrict__ R, const int *__restrict__ F, const int *__restrict__ C, const int n, const int m, int *__restrict__ d, unsigned long long *__restrict__ sigma, float *__restrict__ delta, int *touched, unsigned long long *sigma_hat, float *delta_hat, int *moved, int *movement, int *Q, int *Q2, int *QQ, int *temp, int *taken, size_t pitch, size_t pitch_sigma, size_t pitch_Q, size_t pitch_temp, const int *__restrict__ sources, const int start, const int end, const int src, const int dst);
template __global__ void bc_gpu_update_edge_opt<false,1>(float *bc, const int *__restrict__ r, const int *__restrict__ f, const int *__restrict__ c, const int n, const int m, int *__restrict__ d, unsigned long long *__restrict__ sigma, float *__restrict__ delta, int *touched, unsigned long long *sigma_hat, float *delta_hat, int *moved, int *movement, int *q, int *q2, int *qq, int *temp, int *taken, size_t pitch, size_t pitch_sigma, size_t pitch_q, size_t pitch_temp, const int *__restrict__ sources, const int start, const int end, const int src, const int dst);
template __global__ void bc_gpu_update_edge_opt<true,1>(float *bc, const int *__restrict__ r, const int *__restrict__ f, const int *__restrict__ c, const int n, const int m, int *__restrict__ d, unsigned long long *__restrict__ sigma, float *__restrict__ delta, int *touched, unsigned long long *sigma_hat, float *delta_hat, int *moved, int *movement, int *q, int *q2, int *qq, int *temp, int *taken, size_t pitch, size_t pitch_sigma, size_t pitch_q, size_t pitch_temp, const int *__restrict__ sources, const int start, const int end, const int src, const int dst);
template __global__ void bc_gpu_update_edge_opt<false,2>(float *bc, const int *__restrict__ r, const int *__restrict__ f, const int *__restrict__ c, const int n, const int m, int *__restrict__ d, unsigned long long *__restrict__ sigma, float *__restrict__ delta, int *touched, unsigned long long *sigma_hat, float *delta_hat, int *moved, int *movement, int *q, int *q2, int *qq, int *temp, int *taken, size_t pitch, size_t pitch_sigma, size_t pitch_q, size_t pitch_temp, const int *__restrict__ sources, const int start, const int end, const int src, const int dst);
template __global__ void bc_gpu_update_edge_opt<true,2>(float *bc, const int *__restrict__ r, const int *__restrict__ f, const int *__restrict__ c, const int n, const int m, int *__restrict__ d, unsigned long long *__restrict__ sigma, float *__restrict__ delta, int *touched, unsigned long long *sigma_hat, float *delta_hat, int *moved, int *movement, int *q, int *q2, int *qq, int *temp, int *taken, size_t pitch, size_t pitch_sigma, size_t pitch_q, size_t pitch_temp, const int *__restrict__ sources, const int start, const int end, const int src, const int dst);

#define two_d(k) (k + (blockIdx.x*n))
template<bool approx>
__global__ void bc_gpu_update_edge_SOA(float *bc, graph_data *g, const int n, const int m, int *__restrict__ d, unsigned long long *__restrict__ sigma, float *__restrict__ delta, vertex_data *vv, int *taken, size_t pitch, size_t pitch_sigma, const int *__restrict__ sources, const int start, const int end, const int src, const int dst)
{
	for(int l=blockIdx.x; l<end; l++)
	{
		int i;
		if(approx)
		{
			i = sources[l]; //i is the absolute value of the source node, l is the relative value
		}
		else
		{
			i = l;
		}
		int j = threadIdx.x;

		__shared__ bool compute;
		if(j == 0)
		{
			if(atomicExch(&taken[l],1) == 0)
			{
				compute = true;
			}
			else
			{
				compute = false;
			}
		}
		__syncthreads();

		if(!compute)
		{
			continue;
		}
		__shared__ int recompute;
		__shared__ int u_low;
		__shared__ int u_high;

		__shared__ int *d_row;
		__shared__ unsigned long long *sigma_row;
		__shared__ float *delta_row;

		if(j == 0)
		{
			d_row = (int*)((char*)d + l*pitch);
			sigma_row = (unsigned long long*)((char*)sigma + l*pitch_sigma);
			delta_row = (float*)((char*)delta + l*pitch); 
				
			//Find out which case we're dealing with
			int src_level = d_row[src];
			int dst_level = d_row[dst];

			if(abs(src_level-dst_level)==0) 
			{
				//Case 1 or 5
				recompute = 0;
			}
			else if((src_level == INT_MAX) || (dst_level == INT_MAX))
			{
				//Case 4 - Either one or both nodes is in a different connected component, but not both
				recompute = 3;
			}
			else if(abs(src_level-dst_level) == 1)
			{
				//Case 2
				recompute = 1;
				if(src_level > dst_level)
				{
					u_low = src;
					u_high = dst;
				}
				else
				{
					u_high = src;
					u_low = dst;
				}
			}
			else
			{
				//Case 3 
				recompute = 2;
				if(src_level > dst_level)
				{
					u_low = src;
					u_high = dst;
				}
				else
				{
					u_high = src;
					u_low = dst;
				}
			}
		}
		__syncthreads();

		if(recompute==1)
		{
			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				if(k==u_low)
				{
					vv->touched[two_d(k)] = -1;
					vv->sigma_hat[two_d(k)] = sigma_row[k] + sigma_row[u_high];
				}
				else
				{
					vv->touched[two_d(k)] = 0;
					vv->sigma_hat[two_d(k)] = sigma_row[k];
				}

				vv->delta_hat[two_d(k)] = 0; 
			}

			__shared__ int current_depth;
			__shared__ bool done;
			if(j == 0)
			{
				current_depth = d_row[u_low];
				done = false;
			}
			__syncthreads();

			while(!done)
			{
				__syncthreads();
				done = true;
				__syncthreads();

				for(int k=threadIdx.x; k<2*m; k+=blockDim.x)
				{
					int v = g->F[k];
					if(d_row[v] == current_depth)
					{
						int w = g->C[k];
						if(d_row[w] == (current_depth + 1)) 
						{
							if(vv->touched[two_d(w)] == 0)
							{
								vv->touched[two_d(w)] = -1;
								done = false;
							}
							atomicAdd(&(vv->sigma_hat[two_d(w)]),(vv->sigma_hat[two_d(v)]-sigma_row[v]));
						}
					}
				}

				__syncthreads();
				current_depth++;
				
			}

			__syncthreads();
			current_depth--;
			while(current_depth > 1)
			{
				for(int k=threadIdx.x; k<2*m; k+=blockDim.x)
				{
					int w = g->F[k];
					if(d_row[w] == current_depth)
					{
						int v = g->C[k];
						if(d_row[v] == (current_depth-1))
						{
							float dsv = 0;
							if((vv->touched[two_d(v)] != -1) && (atomicExch(&(vv->touched[two_d(v)]),1) == 0))
							{
								dsv += delta_row[v];
							}
							float new_change = (vv->sigma_hat[two_d(v)]/(float)vv->sigma_hat[two_d(w)])*(1+vv->delta_hat[two_d(w)]);
							dsv += new_change;
							if((vv->touched[two_d(v)]==1) && ((v!=u_high) || (w!=u_low)))
							{
								float old_change = -1*(sigma_row[v]/(float)sigma_row[w])*(1+delta_row[w]);
								dsv += old_change;
							}
							atomicAdd(&(vv->delta_hat[two_d(v)]),dsv);
						}
					}
				}
				__syncthreads();
				current_depth--;
			}

			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				if((k!=i) && (vv->touched[two_d(k)]!=0))
				{
					float delta_change = vv->delta_hat[two_d(k)] - delta_row[k];
					atomicAdd(&bc[k],delta_change); //Does this need to be atomic?
				}
			}

			//Copy back for the next update (and for debugging purposes)
			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				//d doesn't change for this case
				sigma_row[k] = vv->sigma_hat[two_d(k)];
				if(vv->touched[two_d(k)] != 0)
				{
					delta_row[k] = vv->delta_hat[two_d(k)];
				}
			}

			__syncthreads();
		}
		else if(recompute==2)
		{
			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				if(k==u_low)
				{
					vv->touched[two_d(k)] = -1;
					vv->sigma_hat[two_d(k)] = sigma_row[u_high];
					vv->moved[two_d(k)] = 1;
					vv->movement[two_d(k)] = d_row[u_low]-d_row[u_high] - 1;
					vv->Q[two_d(k)] = 1;
				}
				else
				{
					vv->touched[two_d(k)] = 0;
					vv->sigma_hat[two_d(k)] = 0;
					vv->moved[two_d(k)] = 0;
					vv->movement[two_d(k)] = 0;
					vv->Q[two_d(k)] = 0;
				}

				vv->delta_hat[two_d(k)] = 0;
				vv->QQ[two_d(k)] = 0;
				vv->Q2[two_d(k)] = 0;
			}

			__shared__ int current_depth;
			__shared__ bool done;
			__shared__ bool first_iter;
			if(j == 0)
			{
				current_depth = d_row[u_low]; 
				done = false;
				first_iter = true;
			}
			__syncthreads();

			while(!done)
			{
				__syncthreads();
				done = true;
				__syncthreads();

				for(int v=threadIdx.x; v<n; v+=blockDim.x)
				{
					if(vv->Q[two_d(v)]) 
					{
						for(int k=g->R[v]; k<g->R[v+1]; k++) //For all neighbors of v...
						{
							int w = g->C[k];
							int computed_distance = (vv->movement[two_d(v)] - vv->movement[two_d(w)]) - (d_row[v] - d_row[w] + 1);
							if(computed_distance > 0)
							{
								atomicAdd(&(vv->sigma_hat[two_d(w)]),vv->sigma_hat[two_d(v)]);
								vv->moved[two_d(w)] = 1;
								atomicMax(&(vv->movement[two_d(w)]),computed_distance);
								if(vv->touched[two_d(w)] == 0)
								{
									vv->touched[two_d(w)] = -1;
									done = false;
									vv->Q2[two_d(w)] = 1;
								}
							}
							else if((computed_distance==0) && (atomicExch(&(vv->touched[two_d(w)]),-1)==0))
							{
								atomicAdd(&(vv->sigma_hat[two_d(w)]),vv->sigma_hat[two_d(v)]);
								done = false;
								vv->Q2[two_d(w)] = 1;
							}
							else if(vv->touched[two_d(w)] == -1) 
							{
								if(computed_distance >= 0)
								{
									atomicAdd(&(vv->sigma_hat[two_d(w)]),vv->sigma_hat[two_d(v)]);
								}
							}
						}

						d_row[v] -= vv->movement[two_d(v)];
						vv->Q[two_d(v)] = 0;
						vv->QQ[two_d(v)] = 1;
					}
				}

				__syncthreads();
				
				for(int k=threadIdx.x; k<n; k+=blockDim.x)
				{
					if(vv->Q2[two_d(k)] == 1)
					{
						vv->Q[two_d(k)] = 1;
						vv->Q2[two_d(k)] = 0;
					}
				}

				__syncthreads();

				if(j==0)
				{
					current_depth++;
					if(first_iter)
					{
						first_iter = false;
					}
				}
			}

			__syncthreads();
			
			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				if(((k!=u_low) && (vv->moved[two_d(k)]==0)) || (vv->touched[two_d(k)]==0))
				{
					vv->sigma_hat[two_d(k)] += sigma_row[k];
				}
				vv->Q2[two_d(k)] = 0;
			}

			__syncthreads();
			
			current_depth--;
			__shared__ bool repeat;
			if(j==0)
			{
				repeat = false;
			}
			__syncthreads();
			while(current_depth > 0)
			{
				for(int k=threadIdx.x; k<2*m; k+=blockDim.x)
				{
					int w = g->F[k];
					float dsv = 0;
					if((vv->QQ[two_d(w)]) && (d_row[w] == current_depth)) 
					{
						int v = g->C[k];
						if(d_row[v] == (current_depth-1))
						{
							if((vv->touched[two_d(v)] != -1) && (atomicExch(&(vv->touched[two_d(v)]),1) == 0)) //atomicCAS here?
							{
								dsv = delta_row[v];
								vv->touched[two_d(v)] = 1;
								vv->QQ[two_d(v)] = 1; //Checking for depth should handle this
							}
							float new_change = (vv->sigma_hat[two_d(v)]/(float)vv->sigma_hat[two_d(w)])*(1+vv->delta_hat[two_d(w)]);
							dsv += new_change;
							if((vv->touched[two_d(v)] > 0) && ((v!=u_high) || (w!=u_low)))
							{
								float old_change = -1*(sigma_row[v]/(float)sigma_row[w])*(1+delta_row[w]);
								dsv += old_change;
							}
							atomicAdd(&(vv->delta_hat[two_d(v)]),dsv);
						}
						else if((d_row[v] == current_depth) && (vv->moved[two_d(w)]) && (!vv->moved[two_d(v)])) //Sometimes we get in this block when we shouldn't
						{
							if((vv->touched[two_d(v)] != -1) && (atomicExch(&(vv->touched[two_d(v)]),1) == 0))
							{
								dsv = delta_row[v];
								vv->touched[two_d(v)] = 1;
								//Need to repeat this level
								vv->QQ[two_d(v)] = 2;
								repeat = true;
							}
							float old_change = -1*(sigma_row[v]/(float)sigma_row[w])*(1+delta_row[w]);
							dsv += old_change;
							atomicAdd(&(vv->delta_hat[two_d(v)]),dsv);
						}
					}
				}
				__syncthreads();	
			
				if(repeat)
				{
					if(j==0)
					{
						for(int k=0; k<n; k++)
						{
							if((vv->QQ[two_d(k)]>0)&&(d_row[k] == current_depth))
							{
								vv->QQ[two_d(k)]--;
							}
						}
						current_depth++;
						repeat = false;
					}
				}
				__syncthreads();
				current_depth--;
			}

			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				if(vv->delta_hat[two_d(k)] < 0)
				{
					vv->delta_hat[two_d(k)] = 0;
				}
			}

			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				if((k != i) && (vv->touched[two_d(k)]!=0)) //Don't count the source node
				{
					float delta_change = vv->delta_hat[two_d(k)] - delta_row[k];
					atomicAdd(&bc[k],delta_change); //Does this need to be atomic?
				}
			}

			//Copy back for the next update (and for debugging purposes)
			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				//d was updated in place
				sigma_row[k] = vv->sigma_hat[two_d(k)];
				if(vv->touched[two_d(k)] != 0)
				{
					delta_row[k] = vv->delta_hat[two_d(k)];
				}
			}

			__syncthreads();
		}
		else if(recompute==3)
		{
			//Subtract off current value of delta. atomicSub doesn't have a float overload.
			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				if(k != i) //Don't count the source node
				{
					atomicAdd(&bc[k],-1*delta_row[k]); //Does this need to be atomic?
				}
			}

			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				if(k == i) //If its the source node
				{
					sigma_row[k] = 1;
					d_row[k] = 0;
				}
				else
				{
					sigma_row[k] = 0;
					d_row[k] = INT_MAX;
				}

				delta_row[k] = 0;
			}
	
			__shared__ int current_depth;
			__shared__ bool done;
			if(j == 0)
			{
				current_depth = 0;
				done = false;
			}
			__syncthreads();

			while(!done)
			{
				__syncthreads();
				done = true;
				__syncthreads();

				for(int k=threadIdx.x; k<2*m; k+=blockDim.x)
				{
					int v = g->F[k];
					if(d_row[v] == current_depth)
					{
						int w = g->C[k];
						if(d_row[w] == INT_MAX)
						{
							d_row[w] = current_depth + 1; 
							done = false;
						}
						if(d_row[w] == (current_depth + 1)) 
						{
							atomicAdd(&sigma_row[w],sigma_row[v]);
						}
					}
				}

				__syncthreads();
				current_depth++;
			}

			__syncthreads();
			current_depth--;
			while(current_depth > 0)
			{
				for(int k=threadIdx.x; k<2*m; k+=blockDim.x)
				{
					int w = g->F[k];
					if(d_row[w] == current_depth)
					{
						int v = g->C[k];
						if(d_row[w] == (d_row[v]+1))
						{
							float change = (sigma_row[v]/(float)sigma_row[w])*(1.0f+delta_row[w]);
							atomicAdd(&delta_row[v],change);
						}
					}
				}
				__syncthreads();
				current_depth--;
			}

			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				if(k != i) //Don't count the source node
				{
					atomicAdd(&bc[k],delta_row[k]); //Does this need to be atomic?
				}
			}
			__syncthreads();
		}
		
		//If its the first loop iteration, don't bother to check 0...gridDim.x-1 because they'll all be taken, one per SM	
		if(l == blockIdx.x)
		{
			l = gridDim.x-1;
		}
		__syncthreads();
	}
}

template __global__ void bc_gpu_update_edge_SOA<false>(float *bc, graph_data *g, const int n, const int m, int *__restrict__ d, unsigned long long *__restrict__ sigma, float *__restrict__ delta, vertex_data *vv, int *taken, size_t pitch, size_t pitch_sigma, const int *__restrict__ sources, const int start, const int end, const int src, const int dst);
template __global__ void bc_gpu_update_edge_SOA<true>(float *bc, graph_data *g, const int n, const int m, int *__restrict__ d, unsigned long long *__restrict__ sigma, float *__restrict__ delta, vertex_data *vv, int *taken, size_t pitch, size_t pitch_sigma, const int *__restrict__ sources, const int start, const int end, const int src, const int dst);

template<bool approx>
__global__ void bc_gpu_update_edge_AOS(float *bc, const int *__restrict__ R, graph_data_aos *g, const int n, const int m, int *__restrict__ d, unsigned long long *__restrict__ sigma, float *__restrict__ delta, vertex_data_aos *vv, int *taken, size_t pitch, size_t pitch_sigma, const int *__restrict__ sources, const int start, const int end, const int src, const int dst)
{
	for(int l=blockIdx.x; l<end; l++)
	{
		int i;
		if(approx)
		{
			i = sources[l]; //i is the absolute value of the source node, l is the relative value
		}
		else
		{
			i = l;
		}
		int j = threadIdx.x;

		__shared__ bool compute;
		if(j == 0)
		{
			if(atomicExch(&taken[l],1) == 0)
			{
				compute = true;
			}
			else
			{
				compute = false;
			}
		}
		__syncthreads();

		if(!compute)
		{
			continue;
		}
		__shared__ int recompute;
		__shared__ int u_low;
		__shared__ int u_high;

		__shared__ int *d_row;
		__shared__ unsigned long long *sigma_row;
		__shared__ float *delta_row;

		if(j == 0)
		{
			d_row = (int*)((char*)d + l*pitch);
			sigma_row = (unsigned long long*)((char*)sigma + l*pitch_sigma);
			delta_row = (float*)((char*)delta + l*pitch); 
				
			//Find out which case we're dealing with
			int src_level = d_row[src];
			int dst_level = d_row[dst];

			if(abs(src_level-dst_level)==0) 
			{
				//Case 1 or 5
				recompute = 0;
			}
			else if((src_level == INT_MAX) || (dst_level == INT_MAX))
			{
				//Case 4 - Either one or both nodes is in a different connected component, but not both
				recompute = 3;
			}
			else if(abs(src_level-dst_level) == 1)
			{
				//Case 2
				recompute = 1;
				if(src_level > dst_level)
				{
					u_low = src;
					u_high = dst;
				}
				else
				{
					u_high = src;
					u_low = dst;
				}
			}
			else
			{
				//Case 3 
				recompute = 2;
				if(src_level > dst_level)
				{
					u_low = src;
					u_high = dst;
				}
				else
				{
					u_high = src;
					u_low = dst;
				}
			}
		}
		__syncthreads();

		if(recompute==1)
		{
			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				if(k==u_low)
				{
					vv[two_d(k)].touched = -1;
					vv[two_d(k)].sigma_hat = sigma_row[k] + sigma_row[u_high];
				}
				else
				{
					vv[two_d(k)].touched = 0;
					vv[two_d(k)].sigma_hat = sigma_row[k];
				}

				vv[two_d(k)].delta_hat = 0;
			}

			__shared__ int current_depth;
			__shared__ bool done;
			if(j == 0)
			{
				current_depth = d_row[u_low];
				done = false;
			}
			__syncthreads();

			while(!done)
			{
				__syncthreads();
				done = true;
				__syncthreads();

				for(int k=threadIdx.x; k<2*m; k+=blockDim.x)
				{
					int v = g[k].F;
					if(d_row[v] == current_depth)
					{
						int w = g[k].C;
						if(d_row[w] == (current_depth + 1)) 
						{
							if(vv[two_d(w)].touched == 0)
							{
								vv[two_d(w)].touched = -1;
								done = false;
							}
							atomicAdd(&(vv[two_d(w)].sigma_hat),(vv[two_d(v)].sigma_hat-sigma_row[v]));
						}
					}
				}

				__syncthreads();
				current_depth++;
				
			}

			__syncthreads();
			current_depth--;
			while(current_depth > 1)
			{
				for(int k=threadIdx.x; k<2*m; k+=blockDim.x)
				{
					int w = g[k].F;
					if(d_row[w] == current_depth)
					{
						int v = g[k].C;
						if(d_row[v] == (current_depth-1))
						{
							float dsv = 0;

							if((vv[two_d(v)].touched != -1) && (atomicExch(&(vv[two_d(v)].touched),1) == 0))
							{
								dsv += delta_row[v];
							}

							float new_change = (vv[two_d(v)].sigma_hat/(float)vv[two_d(w)].sigma_hat)*(1+vv[two_d(w)].delta_hat);
							dsv += new_change;
							if((vv[two_d(v)].touched == 1) && ((v != u_high) || (w != u_low)))
							{
								float old_change = -1*(sigma_row[v]/(float)sigma_row[w])*(1+delta_row[w]);  
								dsv += old_change;
							}
							atomicAdd(&(vv[two_d(v)].delta_hat),dsv);
						}
					}
				}
				__syncthreads();
				current_depth--;
			}

			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				if((k!=i) && (vv[two_d(k)].touched!=0))
				{
					float delta_change = vv[two_d(k)].delta_hat - delta_row[k];
					atomicAdd(&bc[k],delta_change); //Does this need to be atomic?
				}
			}

			//Copy back for the next update (and for debugging purposes)
			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				//d doesn't change for this case
				sigma_row[k] = vv[two_d(k)].sigma_hat;
				if(vv[two_d(k)].touched != 0)
				{
					delta_row[k] = vv[two_d(k)].delta_hat;
				}
			}

			__syncthreads();
		}
		else if(recompute==2)
		{
			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				if(k==u_low)
				{
					vv[two_d(k)].touched = -1;
					vv[two_d(k)].sigma_hat = sigma_row[u_high];
					vv[two_d(k)].moved = 1;
					vv[two_d(k)].movement = d_row[u_low]-d_row[u_high] - 1;
					vv[two_d(k)].Q = 1;
				}
				else
				{
					vv[two_d(k)].touched = 0;
					vv[two_d(k)].sigma_hat = 0;
					vv[two_d(k)].moved = 0;
					vv[two_d(k)].movement = 0;
					vv[two_d(k)].Q = 0;
				}

				vv[two_d(k)].delta_hat = 0;
				vv[two_d(k)].QQ = 0;
				vv[two_d(k)].Q2 = 0;
			}

			__shared__ int current_depth;
			__shared__ bool done;
			__shared__ bool first_iter;
			if(j == 0)
			{
				current_depth = d_row[u_low]; 
				done = false;
				first_iter = true;
			}
			__syncthreads();

			while(!done)
			{
				__syncthreads();
				done = true;
				__syncthreads();

				for(int v=threadIdx.x; v<n; v+=blockDim.x)
				{ 
					if(vv[two_d(v)].Q)
					{
						for(int k=R[v]; k<R[v+1]; k++) //For all neighbors of v...
						{
							int w = g[k].C;
							int computed_distance = (vv[two_d(v)].movement - vv[two_d(w)].movement) - (d_row[v] - d_row[w] + 1);
							if(computed_distance > 0)
							{
								atomicAdd(&(vv[two_d(w)].sigma_hat),vv[two_d(v)].sigma_hat);
								vv[two_d(w)].moved = 1;
								atomicMax(&(vv[two_d(w)].movement),computed_distance);
								if(vv[two_d(w)].touched == 0)
								{
									vv[two_d(w)].touched = -1;
									done = false;
									vv[two_d(w)].Q2 = 1;
								}
							}
							else if((computed_distance==0) && (atomicExch(&(vv[two_d(w)].touched),-1)==0))
							{
								atomicAdd(&(vv[two_d(w)].sigma_hat),vv[two_d(v)].sigma_hat);
								done = false;
								vv[two_d(w)].Q2 = 1;
							}
							else if(vv[two_d(w)].touched == -1)
							{
								if(computed_distance >= 0)
								{
									atomicAdd(&(vv[two_d(w)].sigma_hat),vv[two_d(v)].sigma_hat);
								}
							}
						}

						d_row[v] -= vv[two_d(v)].movement;
						vv[two_d(v)].Q = 0;
						vv[two_d(v)].QQ = 1;
					}
				}

				__syncthreads();
				
				for(int k=threadIdx.x; k<n; k+=blockDim.x)
				{
					if(vv[two_d(k)].Q2 == 1)
					{
						vv[two_d(k)].Q = 1;
						vv[two_d(k)].Q2 = 0;
					}
				}

				__syncthreads();

				if(j==0)
				{
					current_depth++;
					if(first_iter)
					{
						first_iter = false;
					}
				}
			}

			__syncthreads();
			
			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				if(((k!=u_low) && (vv[two_d(k)].moved==0)) || (vv[two_d(k)].touched == 0))
				{
					vv[two_d(k)].sigma_hat += sigma_row[k];
				}
				vv[two_d(k)].Q2 = 0;
			}

			__syncthreads();
			
			current_depth--;
			__shared__ bool repeat;
			if(j==0)
			{
				repeat = false;
			}
			__syncthreads();
			while(current_depth > 0)
			{
				for(int k=threadIdx.x; k<2*m; k+=blockDim.x)
				{
					int w = g[k].F;
					float dsv = 0;
					if((vv[two_d(w)].QQ) && (d_row[w] == current_depth))
					{
						int v = g[k].C;
						if(d_row[v] == (current_depth-1))
						{
							if((vv[two_d(v)].touched != -1) && (atomicExch(&(vv[two_d(v)].touched),1)==0))
							{
								dsv = delta_row[v];
								vv[two_d(v)].touched = 1;
								vv[two_d(v)].QQ = 1;
							}
							float new_change = (vv[two_d(v)].sigma_hat/(float)vv[two_d(w)].sigma_hat)*(1+vv[two_d(w)].delta_hat);
							dsv += new_change;
							if((vv[two_d(v)].touched > 0) && ((v!=u_high) || (w!=u_low)))
							{
								float old_change = -1*(sigma_row[v]/(float)sigma_row[w])*(1+delta_row[w]);
								dsv += old_change;
							}
							atomicAdd(&(vv[two_d(v)].delta_hat),dsv);
						}
						else if((d_row[v] == current_depth) && (vv[two_d(w)].moved) && (!vv[two_d(v)].moved))
						{
							if((vv[two_d(v)].touched != -1) && (atomicExch(&(vv[two_d(v)].touched),1) == 0))
							{
								dsv = delta_row[v];
								vv[two_d(v)].touched = 1;
								//Need to repeat this level
								vv[two_d(v)].QQ = 2;
								repeat = true;
							}
							float old_change = -1*(sigma_row[v]/(float)sigma_row[w])*(1+delta_row[w]);
							dsv += old_change;
							atomicAdd(&(vv[two_d(v)].delta_hat),dsv);
						}
					}
				}
				__syncthreads();	
			
				if(repeat)
				{
					if(j==0)
					{
						for(int k=0; k<n; k++)
						{
							if((vv[two_d(k)].QQ>0)&&(d_row[k] == current_depth))
							{
								vv[two_d(k)].QQ--;
							}
						}
						current_depth++;
						repeat = false;
					}
				}
				__syncthreads();
				current_depth--;
			}

			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				if(vv[two_d(k)].delta_hat < 0)
				{
					vv[two_d(k)].delta_hat = 0;
				}
			}

			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				if((k!=i) && (vv[two_d(k)].touched!=0))
				{
					float delta_change = vv[two_d(k)].delta_hat - delta_row[k];
					atomicAdd(&bc[k],delta_change); //Does this need to be atomic?
				}
			}

			//Copy back for the next update (and for debugging purposes)
			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				//d was updated in place
				sigma_row[k] = vv[two_d(k)].sigma_hat;
				if(vv[two_d(k)].touched != 0)
				{
					delta_row[k] = vv[two_d(k)].delta_hat;
				}
			}

			__syncthreads();
		}
		else if(recompute==3)
		{
			//Subtract off current value of delta. atomicSub doesn't have a float overload.
			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				if(k != i) //Don't count the source node
				{
					atomicAdd(&bc[k],-1*delta_row[k]); //Does this need to be atomic?
				}
			}

			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				if(k == i) //If its the source node
				{
					sigma_row[k] = 1;
					d_row[k] = 0;
				}
				else
				{
					sigma_row[k] = 0;
					d_row[k] = INT_MAX;
				}

				delta_row[k] = 0;
			}
	
			__shared__ int current_depth;
			__shared__ bool done;
			if(j == 0)
			{
				current_depth = 0;
				done = false;
			}
			__syncthreads();

			while(!done)
			{
				__syncthreads();
				done = true;
				__syncthreads();

				for(int k=threadIdx.x; k<2*m; k+=blockDim.x)
				{
					int v = g[k].F;
					if(d_row[v] == current_depth)
					{
						int w = g[k].C;
						if(d_row[w] == INT_MAX)
						{
							d_row[w] = current_depth + 1; 
							done = false;
						}
						if(d_row[w] == (current_depth + 1)) 
						{
							atomicAdd(&sigma_row[w],sigma_row[v]);
						}
					}
				}

				__syncthreads();
				current_depth++;
			}

			__syncthreads();
			current_depth--;
			while(current_depth > 0)
			{
				for(int k=threadIdx.x; k<2*m; k+=blockDim.x)
				{
					int w = g[k].F;
					if(d_row[w] == current_depth)
					{
						int v = g[k].C;
						if(d_row[w] == (d_row[v]+1))
						{
							float change = (sigma_row[v]/(float)sigma_row[w])*(1.0f+delta_row[w]);
							atomicAdd(&delta_row[v],change);
						}
					}
				}
				__syncthreads();
				current_depth--;
			}

			for(int k=threadIdx.x; k<n; k+=blockDim.x)
			{
				if(k != i) //Don't count the source node
				{
					atomicAdd(&bc[k],delta_row[k]); //Does this need to be atomic?
				}
			}
			__syncthreads();
		}
		
		//If its the first loop iteration, don't bother to check 0...gridDim.x-1 because they'll all be taken, one per SM	
		if(l == blockIdx.x)
		{
			l = gridDim.x-1;
		}
		__syncthreads();
	}
}

template __global__ void bc_gpu_update_edge_AOS<false>(float *bc, const int *__restrict__ R, graph_data_aos *g, const int n, const int m, int *__restrict__ d, unsigned long long *__restrict__ sigma, float *__restrict__ delta, vertex_data_aos *vv, int *taken, size_t pitch, size_t pitch_sigma, const int *__restrict__ sources, const int start, const int end, const int src, const int dst);
template __global__ void bc_gpu_update_edge_AOS<true>(float *bc, const int *__restrict__ R, graph_data_aos *g, const int n, const int m, int *__restrict__ d, unsigned long long *__restrict__ sigma, float *__restrict__ delta, vertex_data_aos *vv, int *taken, size_t pitch, size_t pitch_sigma, const int *__restrict__ sources, const int start, const int end, const int src, const int dst);
