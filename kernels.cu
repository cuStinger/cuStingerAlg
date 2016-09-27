__global__ void bc_gpu_naive(float *bc, int *R, int *C, int n, int m) 
{
	int i = blockIdx.x;
	int j = threadIdx.x;

	__shared__ int sigma[4000]; //Arbitrarily large enough size for testing purposes, for now
	__shared__ int d[4000];
	__shared__ float delta[4000];

	if(i == j)
	{
		sigma[j] = 1;
		d[j] = 0;
	}
	else
	{
		sigma[j] = 0;
		d[j] = INT_MAX;
	}

       	int current_depth;
	__shared__ bool done;
	if(j == 0)
	{
		done = false;
		current_depth = 0;
	}
	delta[j] = 0;
	__syncthreads();

	while(!done)
	{
		__syncthreads();
		done = true;
		__syncthreads();

		if(d[j] == current_depth)
		{
			for(int k=R[j]; k<R[j+1]; k++)
			{
				int w = C[k];
				if(d[w] == INT_MAX)
				{
					d[w] = d[j] + 1;
					done = false;
				}
				if(d[w] == (d[j] + 1))
				{
					atomicAdd(&sigma[w],sigma[j]);
				}
			}
		}

		__syncthreads();
		current_depth++;
	}

	__syncthreads();
	//current_depth-- before loop? Should save an iteration...
	while(current_depth > 0)
	{
		if(d[j] == current_depth)
		{
			for(int k=R[j]; k<R[j+1]; k++)
			{
				int v = C[k];
				if(d[j] == (d[v]+1))
				{
					float change = (sigma[v]/(float)sigma[j])*(1.0f+delta[j]);
					atomicAdd(&delta[v],change);
				}
			}

			if(i != j)
			{
				atomicAdd(&bc[j],delta[j]); //j from two different blocks could simultaneously update this value
			}
		}

		__syncthreads();
		current_depth--;
	}

	/*if(i != j)
	{
		atomicAdd(&bc[j],delta[j]);
	}*/
}

//It could be helpful to include a list of successful optimizations
/*
1) const/restrict keywords
2) edge-based parallelism
3) use of pitch for d, sigma, and delta
*/

__global__ void bc_gpu_opt(float *bc, const int *__restrict__ F, const int *__restrict__ C, const int n, const int m, int *__restrict__ d, unsigned long long *__restrict__ sigma, float *__restrict__ delta, size_t pitch, size_t pitch_sigma, const int start, const int end)
{
	for(int i=start+blockIdx.x; i<end; i+=gridDim.x) //Overall i = [0, 1, ..., n-1] (inclusive)
	{
		int j = threadIdx.x;

		for(int k=threadIdx.x; k<n; k+=blockDim.x)
		{
			int *d_row = (int*)((char*)d + blockIdx.x*pitch);
			unsigned long long *sigma_row = (unsigned long long*)((char*)sigma + blockIdx.x*pitch_sigma);
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

			float *delta_row = (float*)((char*)delta + blockIdx.x*pitch);
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
				int *d_row = (int *)((char*)d + blockIdx.x*pitch);
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
						unsigned long long *sigma_row = (unsigned long long*)((char*)sigma + blockIdx.x*pitch_sigma);
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
				int *d_row = (int *)((char*)d + blockIdx.x*pitch);
				if(d_row[w] == current_depth)
				{
					int v = C[k];
					if(d_row[w] == (d_row[v]+1))
					{
						float *delta_row = (float*)((char*)delta + blockIdx.x*pitch);
						unsigned long long *sigma_row = (unsigned long long*)((char*)sigma + blockIdx.x*pitch_sigma);
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
				float *delta_row = (float*)((char*)delta + blockIdx.x*pitch);
				atomicAdd(&bc[k],delta_row[k]); //Does this need to be atomic?
			}
		}
		__syncthreads();
	}
}

__global__ void bc_gpu_opt_approx(float *bc, const int *__restrict__ F, const int *__restrict__ C, const int n, const int m, int *__restrict__ d, unsigned long long *__restrict__ sigma, float *__restrict__ delta, size_t pitch, size_t pitch_sigma, const int *__restrict__ sources, const int K, const int start, const int end)
{
	for(int l=start+blockIdx.x; l<end; l+=gridDim.x) // i = [0, 1, ..., K-1] (inclusive)
	{
		int i = sources[l]; //i is the absolute value of the source node, blockIdx.x is the relative value
		int j = threadIdx.x;

		for(int k=threadIdx.x; k<n; k+=blockDim.x)
		{
			int *d_row = (int*)((char*)d + blockIdx.x*pitch);
			unsigned long long *sigma_row = (unsigned long long*)((char*)sigma + blockIdx.x*pitch_sigma);
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

			float *delta_row = (float*)((char*)delta + blockIdx.x*pitch);
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
				int *d_row = (int *)((char*)d + blockIdx.x*pitch);
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
						unsigned long long *sigma_row = (unsigned long long*)((char*)sigma + blockIdx.x*pitch_sigma);
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
				int *d_row = (int *)((char*)d + blockIdx.x*pitch);
				if(d_row[w] == current_depth)
				{
					int v = C[k];
					if(d_row[w] == (d_row[v]+1))
					{
						float *delta_row = (float*)((char*)delta + blockIdx.x*pitch);
						unsigned long long *sigma_row = (unsigned long long*)((char*)sigma + blockIdx.x*pitch_sigma);
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
				float *delta_row = (float*)((char*)delta + blockIdx.x*pitch);
				atomicAdd(&bc[k],delta_row[k]); //Does this need to be atomic?
			}
		}
		__syncthreads();
	}
}

