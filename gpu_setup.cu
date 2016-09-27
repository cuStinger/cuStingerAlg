#include "gpu_setup.cuh"
#include "sequential.h"
#define CPU_THREADS 2 //Number of worker threads

float single_gpu_full_computation(csr_graph g, program_options op, int number_of_SMs, int max_threads_per_block, float *bc_gpu, int *source_nodes, bool recomp)
{
	float time_gpu_opt;

	//Device Pointers
	float *bc_gpu_d, *delta_d;  
	int *C_d, *F_d, *d_d, *source_nodes_d;
	unsigned long long *sigma_d;

	//Allocate and transfer data to the GPU
	//std::cout << "Approximate memory required: " << sizeof(float)*g.n + sizeof(int)*(g.m*4) + sizeof(int)*g.n*number_of_SMs + sizeof(unsigned long long)*g.n*number_of_SMs + sizeof(float)*g.n*number_of_SMs << std::endl;
	checkCudaErrors(cudaMalloc((void**)&bc_gpu_d,sizeof(float)*g.n));
	checkCudaErrors(cudaMalloc((void**)&C_d,sizeof(int)*(g.m*2)));
	checkCudaErrors(cudaMalloc((void**)&F_d,sizeof(int)*(g.m*2)));
	if(op.approximate)
	{
		checkCudaErrors(cudaMalloc((void**)&source_nodes_d,sizeof(int)*op.k));
	}

	checkCudaErrors(cudaMemcpy(bc_gpu_d,bc_gpu,sizeof(float)*g.n,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(C_d,g.C,sizeof(int)*(g.m*2),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(F_d,g.F,sizeof(int)*(g.m*2),cudaMemcpyHostToDevice));
	if(op.approximate)
	{
		checkCudaErrors(cudaMemcpy(source_nodes_d,source_nodes,sizeof(int)*op.k,cudaMemcpyHostToDevice));
	}

	//Allocate 2D arrays so that each block has its own global data (because it won't all fit in shared)
	size_t pitch, pitch_sigma;
	//Since d and delta are all the same size they will all have the same pitch.
	//This is NOT the case in general, so we're really exploiting the pitch size here.
	if(op.experiment)
	{
		number_of_SMs = 30; //When using more than the detected number of SMs for experiments we'll go out of bounds unless we hard code this number here.
	}
	checkCudaErrors(cudaMallocPitch((void**)&d_d,&pitch,sizeof(int)*g.n,number_of_SMs));
	checkCudaErrors(cudaMallocPitch((void**)&sigma_d,&pitch_sigma,sizeof(unsigned long long)*g.n,number_of_SMs));
	checkCudaErrors(cudaMallocPitch((void**)&delta_d,&pitch,sizeof(float)*g.n,number_of_SMs));

	//Set kernel dimensions
	dim3 dimBlock, dimGrid;
	dimBlock.x = max_threads_per_block;
	dimBlock.y = 1;
	dimBlock.z = 1;
	dimGrid.x = number_of_SMs;
	dimGrid.y = 1;
	dimGrid.z = 1;

	cudaEvent_t start,end;
	if(op.approximate)
	{
		start_clock(start,end);
		pthread_t thread;
		if(op.nvml)
		{
			start_power_sample(op,thread,10);
		}
		bc_gpu_opt_approx<<<dimGrid,dimBlock>>>(bc_gpu_d,F_d,C_d,g.n,g.m,d_d,sigma_d,delta_d,pitch,pitch_sigma,source_nodes_d,op.k,0,op.k);
		checkCudaErrors(cudaPeekAtLastError()); //Check for kernel launch errors
		if(op.nvml)
		{
			float avg_power = end_power_sample(op,thread);
			if(op.power_file == NULL)
			{
				if(recomp)
				{
					std::cout << "Average power for recomputation: " << avg_power << std::endl;
				}
				else
				{
					std::cout << "Average power: " << avg_power << std::endl;
				}
			}
			else
			{
				std::ofstream ofs;
				ofs.open(op.power_file, std::ios::app);
				if(recomp)
				{
					ofs << std::endl << "Average power for recomputation: " << avg_power << std::endl; 
				}
				else
				{
					ofs << "Average power: " << avg_power << std::endl;
				}
				ofs.close();
			}
		}
		time_gpu_opt = end_clock(start,end);
	}
	else
	{
		//Experiment with block size
		if(op.experiment)
		{
			std::ofstream ofs;
			if(op.result_file != NULL)
			{
				if(!std::ifstream(op.result_file))
				{
					//File doesn't yet exist, add a header
					std::cout << "File doesn't yet exist" << std::endl;
					ofs.open(op.result_file, std::ios::out);
					ofs << "GPU";
					for(int i=1; i<=30; i++)
					{
						ofs << ",Block_" << i;
					}
					ofs << std::endl;
				}
				else
				{
					std::cout << "File exists, appending..." << std::endl;
					ofs.open(op.result_file, std::ios::app);
				}
			}
		
			//If we have a file, write results to it. Otherwise, print to stdout.	
			std::ostream &os = (op.result_file ? ofs : std::cout);
			os << std::setprecision(9);
			if(op.device == 0) //Hard-corded Denali info. Could use cudaGetDeviceProperties, but meh
			{
				if(op.nvml)
				{
					os << "TeslaC2075_Power";
				}
				else
				{
					os << "TeslaC0275";
				}
			}
			else
			{
				os << "GTX560";
			}
			if(op.nvml)
			{
				for(int i=1; i<=30; i++) //Might want to show for a larger number of blocks as well
				{
					dimGrid.x = i;
					pthread_t thread;
					float avg_power;
					start_clock(start,end);
					start_power_sample(op,thread,10);
					bc_gpu_opt<<<dimGrid,dimBlock>>>(bc_gpu_d,F_d,C_d,g.n,g.m,d_d,sigma_d,delta_d,pitch,pitch_sigma,0,g.n);
					checkCudaErrors(cudaPeekAtLastError()); //Check for kernel launch errors
					avg_power = end_power_sample(op,thread);
					time_gpu_opt = end_clock(start,end);
					os << "," << avg_power; //Power in Watts
				}
				os << std::endl;
			}
			else
			{
				for(int i=1; i<=30; i++) //Might want to show for a larger number of blocks as well
				{
					dimGrid.x = i;
					start_clock(start,end);
					bc_gpu_opt<<<dimGrid,dimBlock>>>(bc_gpu_d,F_d,C_d,g.n,g.m,d_d,sigma_d,delta_d,pitch,pitch_sigma,0,g.n);
					checkCudaErrors(cudaPeekAtLastError()); //Check for kernel launch errors
					time_gpu_opt = end_clock(start,end);
					os << "," << time_gpu_opt/(float)1000; //Time in seconds
				}
				os << std::endl;
			}
		}
		else
		{
			start_clock(start,end);
			pthread_t thread;
			if(op.nvml)
			{
				start_power_sample(op,thread,10);
			}
			bc_gpu_opt<<<dimGrid,dimBlock>>>(bc_gpu_d,F_d,C_d,g.n,g.m,d_d,sigma_d,delta_d,pitch,pitch_sigma,0,g.n);
			checkCudaErrors(cudaPeekAtLastError()); //Check for kernel launch errors
			if(op.nvml)
			{
				float avg_power = end_power_sample(op,thread);
				if(op.power_file != NULL)
				{
					std::ofstream ofs;
					ofs.open(op.power_file, std::ios::app);
					if(recomp)
					{
						ofs << std::endl << "Average power for recomputation: " << avg_power << std::endl; 
					}
					else
					{
						ofs << "Average power: " << avg_power << std::endl;
					}
					ofs.close();
				}
				else
				{
					if(recomp)
					{
						std::cout << "Average power for recomputation: " << avg_power << std::endl;
					}
					else
					{
						std::cout << "Average power: " << avg_power << std::endl;
					}
				}
			}
			time_gpu_opt = end_clock(start,end);
		}
	}
	//Receive result from the GPU
	checkCudaErrors(cudaMemcpy(bc_gpu,bc_gpu_d,sizeof(float)*g.n,cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(bc_gpu_d));
	checkCudaErrors(cudaFree(F_d));
	checkCudaErrors(cudaFree(C_d));
	checkCudaErrors(cudaFree(d_d));
	checkCudaErrors(cudaFree(sigma_d));
	checkCudaErrors(cudaFree(delta_d));
	if(op.approximate)
	{
		checkCudaErrors(cudaFree(source_nodes_d));
	}

	return time_gpu_opt;
}

float multi_gpu_full_computation(csr_graph g, program_options op, float *bc_gpu, int *source_nodes)
{
	float time_gpu_multi;
	int devcount;
	checkCudaErrors(cudaGetDeviceCount(&devcount));
	if(devcount < 2)
	{
		std::cerr << "Error: Less than 2 GPUs are present in this system." << std::endl;
		exit(-1);
	}

	//Device Pointers
	float *bc_gpu_d[devcount], *delta_d[devcount];  
	int *C_d[devcount], *F_d[devcount], *d_d[devcount], *source_nodes_d[devcount];
	unsigned long long *sigma_d[devcount];
	size_t pitch[devcount], pitch_sigma[devcount]; //Devices will probably have the same pitch values, but play it safe
	dim3 dimBlock[devcount], dimGrid[devcount];

	//Intermediate Host Result Pointers
	float *bc_gpu_tmp[devcount];

	//Allocate and transfer data to the GPUs
	for(int dev=0; dev<devcount; dev++)
	{
		checkCudaErrors(cudaSetDevice(dev));
		cudaDeviceProp prop;
		checkCudaErrors(cudaGetDeviceProperties(&prop,dev));
		//Set kernel dimensions
		dimBlock[dev].x = prop.maxThreadsPerBlock;
		dimBlock[dev].y = 1;
		dimBlock[dev].z = 1;
		if(prop.multiProcessorCount != 0)
		{
			dimGrid[dev].x = prop.multiProcessorCount;
		}
		else //Lack of updated drivers will cause this issue. Hard code for now.
		{
			dimGrid[0].x = 14; //Tesla C2075
			dimGrid[1].x = 7; //GTX 560
		}
		dimGrid[dev].y = 1;
		dimGrid[dev].z = 1;
		checkCudaErrors(cudaMalloc((void**)&bc_gpu_d[dev],sizeof(float)*g.n));
		checkCudaErrors(cudaMalloc((void**)&C_d[dev],sizeof(int)*(g.m*2)));
		checkCudaErrors(cudaMalloc((void**)&F_d[dev],sizeof(int)*(g.m*2)));
		if(op.approximate)
		{
			checkCudaErrors(cudaMalloc((void**)&source_nodes_d[dev],sizeof(int)*op.k));
		}

		bc_gpu_tmp[dev] = new float[g.n];
		for(int i=0; i<g.n; i++)
		{
			bc_gpu_tmp[dev][i] = 0;
		}

		checkCudaErrors(cudaMemcpy(bc_gpu_d[dev],bc_gpu_tmp[dev],sizeof(float)*g.n,cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(C_d[dev],g.C,sizeof(int)*(g.m*2),cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(F_d[dev],g.F,sizeof(int)*(g.m*2),cudaMemcpyHostToDevice));
		if(op.approximate)
		{
			checkCudaErrors(cudaMemcpy(source_nodes_d[dev],source_nodes,sizeof(int)*op.k,cudaMemcpyHostToDevice));
		}

		//Allocate 2D arrays so that each block has its own global data (because it won't all fit in shared)
		//Since d and delta are all the same size they will all have the same pitch.
		//This is NOT the case in general, so we're really exploiting the pitch size here.
		checkCudaErrors(cudaMallocPitch((void**)&d_d[dev],&pitch[dev],sizeof(int)*g.n,dimGrid[dev].x));
		checkCudaErrors(cudaMallocPitch((void**)&sigma_d[dev],&pitch_sigma[dev],sizeof(unsigned long long)*g.n,dimGrid[dev].x));
		checkCudaErrors(cudaMallocPitch((void**)&delta_d[dev],&pitch[dev],sizeof(float)*g.n,dimGrid[dev].x));
		std::cout << "Device " << dev << ": " << prop.name << std::endl;
		std::cout << "Size of pitch[" << dev << "]: " << pitch[dev] << std::endl;
		std::cout << "Size of pitch sigma[" << dev << "]: " << pitch_sigma[dev] << std::endl;	
		std::cout << "Threads per block: " << dimBlock[dev].x << std::endl;
		std::cout << "Blocks per grid: " << dimGrid[dev].x << std::endl << std::endl;
	}

	cudaEvent_t start,end;
	start_clock(start,end); //These events will belong to the last device
	
	for(int dev=0; dev<devcount; dev++)
	{
		//This GPU will compute sources [first_source, last_source)
		checkCudaErrors(cudaSetDevice(dev));
		if(op.approximate)
		{
			int first_source = dev*ceil(op.k/(float)devcount);
			int last_source;
			if(dev == devcount - 1)
			{
				last_source = op.k;
			}
			else
			{
				last_source = (dev+1)*ceil(op.k/(float)devcount);
			}
			bc_gpu_opt_approx<<<dimGrid[dev],dimBlock[dev]>>>(bc_gpu_d[dev],F_d[dev],C_d[dev],g.n,g.m,d_d[dev],sigma_d[dev],delta_d[dev],pitch[dev],pitch_sigma[dev],source_nodes_d[dev],op.k,first_source,last_source);
			checkCudaErrors(cudaPeekAtLastError()); //Check for kernel launch errors
		}
		else
		{
			int first_source = dev*ceil(g.n/(float)devcount);
			int last_source;
			if(dev == devcount - 1) //The last GPU will have slightly fewer elements to process due to rounding
			{
				last_source = g.n;
			}
			else
			{
				last_source = (dev+1)*ceil(g.n/(float)devcount);
			}
			bc_gpu_opt<<<dimGrid[dev],dimBlock[dev]>>>(bc_gpu_d[dev],F_d[dev],C_d[dev],g.n,g.m,d_d[dev],sigma_d[dev],delta_d[dev],pitch[dev],pitch_sigma[dev],first_source,last_source);
			checkCudaErrors(cudaPeekAtLastError()); //Check for kernel launch errors
		}
	}

	for(int dev=0; dev<devcount; dev++)
	{
		checkCudaErrors(cudaSetDevice(dev));
		//Receive result from the GPU
		checkCudaErrors(cudaMemcpy(bc_gpu_tmp[dev],bc_gpu_d[dev],sizeof(float)*g.n,cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(bc_gpu_d[dev]));
		checkCudaErrors(cudaFree(F_d[dev]));
		checkCudaErrors(cudaFree(C_d[dev]));
		checkCudaErrors(cudaFree(d_d[dev]));
		checkCudaErrors(cudaFree(sigma_d[dev]));
		checkCudaErrors(cudaFree(delta_d[dev]));
		if(op.approximate)
		{
			checkCudaErrors(cudaFree(source_nodes_d[dev]));
		}

		/*int accessible = 0;
		checkCudaErrors(cudaDeviceCanAccessPeer(&accessible,1,0)); //Can GPU 1 access memory on GPU 0?
		if(accessible)
		{
			std::cout << "GPU 1 can access the memory of GPU 0 through PCI-E." << std::endl;
		}*/

		//Sum intermediate results
		for(int i=0; i<g.n; i++)
		{
			bc_gpu[i] += bc_gpu_tmp[dev][i];
		}

		delete[] bc_gpu_tmp[dev];
	}

	time_gpu_multi = end_clock(start,end); //Note that these events still belong to the last device

	return time_gpu_multi;
}

void single_gpu_streaming_computation(csr_graph &g, program_options op, float *bc_gpu, int *source_nodes, int max_threads_per_block, int number_of_SMs, std::set< std::pair<int,int> > removals_gpu, float &time_gpu_update, float &time_min_update_gpu, float &time_max_update_gpu, std::pair<int,int> &min_update_edge_gpu, std::pair<int,int> &max_update_edge_gpu, std::vector< std::vector<int> > &d_gpu_v, std::vector< std::vector<unsigned long long> > &sigma_gpu_v, std::vector< std::vector<float> > &delta_gpu_v, bool opt, int node)
{
	//Remove edges again so that we can stream on the GPU
	g.remove_edges(removals_gpu);
	if(!opt)
	{
		std::cout << std::endl;
		std::cout << "GPU Streaming: " << std::endl;
	}
	else
	{
		if(node==1)
		{
			std::cout << "Node-based GPU Streaming: " << std::endl;
		}
		else if(node==0)
		{
			std::cout << "Edge-based GPU Streaming: " << std::endl;
		}
		else
		{
			std::cout << "Hybrid GPU Streaming: " << std::endl;
		}
	}
	bool firstiter = true;
	
	//Device Pointers
	float *bc_gpu_d, *delta_d;  
	int *C_d, *F_d, *d_d, *source_nodes_d;
	unsigned long long *sigma_d;
	//Allocate worst case scenario case 2 data structures so we don't need to malloc/free within a kernel
	int *touched_d;
	unsigned long long *sigma_hat_d;
	float *delta_hat_d;
	int *moved_d; //Could make this a bool with a possibly separate pitch to save some memory
	int *movement_d;
	int *temp_d;
	int *taken_d; //Work stealing flag
	int *taken;
	int *Q_d, *Q2_d, *QQ_d; //Case 3 queues
	int *R_d;
	//Allocate 2D arrays so that each block has its own global data (because it won't all fit in shared)
	//Since d and delta are all the same size they will all have the same pitch.
	//This is NOT the case in general, so we're really exploiting the pitch size here.
	size_t pitch, pitch_sigma, pitch_Q, pitch_temp;
	dim3 dimBlock, dimGrid;
	cudaEvent_t start,end;
	while(removals_gpu.size())
	{
		//Run full computation on the GPU so we have d, sigma, and delta
		if(firstiter)
		{
			//Reset bc scores
			for(int i=0; i<g.n; i++)
			{
				bc_gpu[i] = 0;
			}

			//Allocate and transfer data to the GPU
			checkCudaErrors(cudaMalloc((void**)&bc_gpu_d,sizeof(float)*g.n));
			checkCudaErrors(cudaMalloc((void**)&C_d,sizeof(int)*(g.m*2)));
			checkCudaErrors(cudaMalloc((void**)&F_d,sizeof(int)*(g.m*2)));
			if(op.approximate)
			{
				checkCudaErrors(cudaMalloc((void**)&source_nodes_d,sizeof(int)*op.k));
			}

			checkCudaErrors(cudaMemcpy(bc_gpu_d,bc_gpu,sizeof(float)*g.n,cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(C_d,g.C,sizeof(int)*(g.m*2),cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(F_d,g.F,sizeof(int)*(g.m*2),cudaMemcpyHostToDevice));
			if(op.approximate)
			{
				checkCudaErrors(cudaMemcpy(source_nodes_d,source_nodes,sizeof(int)*op.k,cudaMemcpyHostToDevice));
			}
			
			//Potentially dangerous quick fix:
			//Setting the number of SMs this way allows for reuse of the same kernels, which is nice because we know it works.
			//This is not good for performance, but in this case we're just generating results we can build on, so we don't care.
			//For large graphs in the exact case, this will blow up. But the n^2 storage might have blown things up anyway.	
			//This also takes advantage of the fact that k and n don't change as we stream the graph (because we only add edges).

			if(op.approximate)
			{
				dimGrid.x = op.k; //Use enough SMs to store all old data
				checkCudaErrors(cudaMallocPitch((void**)&d_d,&pitch,sizeof(int)*g.n,op.k));
				checkCudaErrors(cudaMallocPitch((void**)&sigma_d,&pitch_sigma,sizeof(unsigned long long)*g.n,op.k));
				checkCudaErrors(cudaMallocPitch((void**)&delta_d,&pitch,sizeof(float)*g.n,op.k));
			}
			else
			{
				dimGrid.x = g.n; //Use enough SMs to store all old data
				checkCudaErrors(cudaMallocPitch((void**)&d_d,&pitch,sizeof(int)*g.n,g.n));
				checkCudaErrors(cudaMallocPitch((void**)&sigma_d,&pitch_sigma,sizeof(unsigned long long)*g.n,g.n));
				checkCudaErrors(cudaMallocPitch((void**)&delta_d,&pitch,sizeof(float)*g.n,g.n));
			}

			//Set kernel dimensions
			dimBlock.x = max_threads_per_block; 
			dimBlock.y = 1;
			dimBlock.z = 1;
			//dimGrid.x = number_of_SMs;
			dimGrid.y = 1;
			dimGrid.z = 1;

			if(op.approximate)
			{
				bc_gpu_opt_approx<<<dimGrid,dimBlock>>>(bc_gpu_d,F_d,C_d,g.n,g.m,d_d,sigma_d,delta_d,pitch,pitch_sigma,source_nodes_d,op.k,0,op.k);
				checkCudaErrors(cudaPeekAtLastError()); //Check for kernel launch errors
			}
			else
			{
				bc_gpu_opt<<<dimGrid,dimBlock>>>(bc_gpu_d,F_d,C_d,g.n,g.m,d_d,sigma_d,delta_d,pitch,pitch_sigma,0,g.n);
				checkCudaErrors(cudaPeekAtLastError()); //Check for kernel launch errors
			}

			//No need to transfer back GPU results - we just want them on the device. The source nodes for the approx case won't change, so keep them on.
			checkCudaErrors(cudaFree(F_d));
			checkCudaErrors(cudaFree(C_d));
			
			firstiter = false;
			dimGrid.x = number_of_SMs; //Use optimal number of SMs

			//Allocate space for touched, sigma_hat, and delta_hat on the GPU - but only do this once
			if(opt)
			{
				//Risky - using same pitch/pitch_sigma parameters. 
				checkCudaErrors(cudaMallocPitch((void**)&touched_d,&pitch,sizeof(int)*g.n,number_of_SMs));
				checkCudaErrors(cudaMallocPitch((void**)&sigma_hat_d,&pitch_sigma,sizeof(unsigned long long)*g.n,number_of_SMs));
				checkCudaErrors(cudaMallocPitch((void**)&delta_hat_d,&pitch,sizeof(float)*g.n,number_of_SMs));
				checkCudaErrors(cudaMallocPitch((void**)&moved_d,&pitch,sizeof(int)*g.n,number_of_SMs));
				checkCudaErrors(cudaMallocPitch((void**)&movement_d,&pitch,sizeof(int)*g.n,number_of_SMs));
				checkCudaErrors(cudaMallocPitch((void**)&QQ_d,&pitch,sizeof(int)*g.n,number_of_SMs));
				if(g.n > 550000) //if the graph is larger than coPapersDBLP
				{
					checkCudaErrors(cudaMallocPitch((void**)&Q_d,&pitch_Q,sizeof(int)*g.n*3,number_of_SMs));
					checkCudaErrors(cudaMallocPitch((void**)&Q2_d,&pitch_Q,sizeof(int)*g.n*3,number_of_SMs));
					checkCudaErrors(cudaMallocPitch((void**)&temp_d,&pitch_temp,sizeof(int)*g.n*6,number_of_SMs));
				}
				else
				{
					checkCudaErrors(cudaMallocPitch((void**)&Q_d,&pitch_Q,sizeof(int)*(g.m/2),number_of_SMs));
					checkCudaErrors(cudaMallocPitch((void**)&Q2_d,&pitch_Q,sizeof(int)*(g.m/2),number_of_SMs));
					checkCudaErrors(cudaMallocPitch((void**)&temp_d,&pitch_temp,sizeof(int)*g.m,number_of_SMs));
				}
				if(op.approximate)
				{
					checkCudaErrors(cudaMalloc((void**)&taken_d,sizeof(int)*op.k));
					taken = new int[op.k];
					for(int i=0; i<op.k; i++)
					{
						taken[i] = 0;
					}
				}
				else
				{
					checkCudaErrors(cudaMalloc((void**)&taken_d,sizeof(int)*g.n));
					taken = new int[g.n];
					for(int i=0; i<g.n; i++)
					{
						taken[i] = 0;
					}
				}
			}
		}

		//Add edge
		int source = removals_gpu.begin()->first;
		int dest = removals_gpu.begin()->second;
		std::cout << "Inserting edge: (" << source << "," << dest << ")" << std::endl;
		g.reinsert_edge(source,dest);
		removals_gpu.erase(removals_gpu.begin());
		if(removals_gpu.find(std::make_pair(dest,source)) != removals_gpu.end())
		{
			removals_gpu.erase(std::make_pair(dest,source));
		}
		else
		{
			std::cerr << "Error reinserting edges: edge (" << source << "," << dest << ") found but edge (" << dest << "," << source << ") could not be found." << std::endl;
			exit(-1);
		}

		//Update on the GPU
		float prev_time = time_gpu_update;
	
		//Reallocate the updated graph
		checkCudaErrors(cudaMalloc((void**)&C_d,sizeof(int)*(g.m*2)));
		checkCudaErrors(cudaMalloc((void**)&F_d,sizeof(int)*(g.m*2)));
		checkCudaErrors(cudaMalloc((void**)&R_d,sizeof(int)*(g.n+1)));
	
		checkCudaErrors(cudaMemcpy(C_d,g.C,sizeof(int)*(g.m*2),cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(F_d,g.F,sizeof(int)*(g.m*2),cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(R_d,g.R,sizeof(int)*(g.n+1),cudaMemcpyHostToDevice));
		
		if((opt) && (node==1)) //Node-based parallelism
		{
			if(op.approximate)
			{
				checkCudaErrors(cudaMemcpy(taken_d,taken,sizeof(int)*op.k,cudaMemcpyHostToDevice));
				pthread_t thread;
				start_clock(start,end);
				if(op.nvml)
				{
					start_power_sample(op,thread,10);
				}
				bc_gpu_update_edge_opt<true,1><<<dimGrid,dimBlock>>>(bc_gpu_d,R_d,F_d,C_d,g.n,g.m,d_d,sigma_d,delta_d,touched_d,sigma_hat_d,delta_hat_d,moved_d,movement_d,Q_d,Q2_d,QQ_d,temp_d,taken_d,pitch,pitch_sigma,pitch_Q,pitch_temp,source_nodes_d,0,op.k,source,dest);
				checkCudaErrors(cudaPeekAtLastError()); //Check for kernel launch errors
				if(op.nvml)
				{
					float avg_power = end_power_sample(op,thread);
					if(op.power_file == NULL)
					{
						std::cout << "Average power: " << avg_power << std::endl;
					}
					else
					{
						std::ofstream ofs;
						ofs.open(op.power_file, std::ios::app);
						ofs << avg_power << ",";
						ofs.close();
					}
				}
				time_gpu_update += end_clock(start,end);
			}
			else
			{
				checkCudaErrors(cudaMemcpy(taken_d,taken,sizeof(int)*g.n,cudaMemcpyHostToDevice));
				pthread_t thread;
				start_clock(start,end);
				if(op.nvml)
				{
					start_power_sample(op,thread,10);
				}
				bc_gpu_update_edge_opt<false,1><<<dimGrid,dimBlock>>>(bc_gpu_d,R_d,F_d,C_d,g.n,g.m,d_d,sigma_d,delta_d,touched_d,sigma_hat_d,delta_hat_d,moved_d,movement_d,Q_d,Q2_d,QQ_d,temp_d,taken_d,pitch,pitch_sigma,pitch_Q,pitch_temp,source_nodes_d,0,g.n,source,dest);
				checkCudaErrors(cudaPeekAtLastError()); //Check for kernel launch errors
				if(op.nvml)
				{
					float avg_power = end_power_sample(op,thread);
					if(op.power_file == NULL)
					{
						std::cout << "Average power: " << avg_power << std::endl;
					}
					else
					{
						std::ofstream ofs;
						ofs.open(op.power_file, std::ios::app);
						ofs << avg_power << ",";
						ofs.close();
					}
				}
				time_gpu_update += end_clock(start,end);
			}
		}
		else if((opt) && (node==0)) //Edge-based parallelism
		{
			if(op.approximate)
			{
				checkCudaErrors(cudaMemcpy(taken_d,taken,sizeof(int)*op.k,cudaMemcpyHostToDevice));
				pthread_t thread;
				start_clock(start,end);
				if(op.nvml)
				{
					start_power_sample(op,thread,10);
				}
				bc_gpu_update_edge_opt<true,0><<<dimGrid,dimBlock>>>(bc_gpu_d,R_d,F_d,C_d,g.n,g.m,d_d,sigma_d,delta_d,touched_d,sigma_hat_d,delta_hat_d,moved_d,movement_d,Q_d,Q2_d,QQ_d,temp_d,taken_d,pitch,pitch_sigma,pitch_Q,pitch_temp,source_nodes_d,0,op.k,source,dest);
				checkCudaErrors(cudaPeekAtLastError()); //Check for kernel launch errors
				if(op.nvml)
				{
					float avg_power = end_power_sample(op,thread);
					if(op.power_file == NULL)
					{
						std::cout << "Average power: " << avg_power << std::endl;
					}
					else
					{
						std::ofstream ofs;
						ofs.open(op.power_file, std::ios::app);
						ofs << avg_power << ",";
						ofs.close();
					}
				}
				time_gpu_update += end_clock(start,end);
			}
			else
			{
				checkCudaErrors(cudaMemcpy(taken_d,taken,sizeof(int)*g.n,cudaMemcpyHostToDevice));
				pthread_t thread;
				start_clock(start,end);
				if(op.nvml)
				{
					start_power_sample(op,thread,10);
				}
				bc_gpu_update_edge_opt<false,0><<<dimGrid,dimBlock>>>(bc_gpu_d,R_d,F_d,C_d,g.n,g.m,d_d,sigma_d,delta_d,touched_d,sigma_hat_d,delta_hat_d,moved_d,movement_d,Q_d,Q2_d,QQ_d,temp_d,taken_d,pitch,pitch_sigma,pitch_Q,pitch_temp,source_nodes_d,0,g.n,source,dest);
				checkCudaErrors(cudaPeekAtLastError()); //Check for kernel launch errors
				if(op.nvml)
				{
					float avg_power = end_power_sample(op,thread);
					if(op.power_file == NULL)
					{
						std::cout << "Average power: " << avg_power << std::endl;
					}
					else
					{
						std::ofstream ofs;
						ofs.open(op.power_file, std::ios::app);
						ofs << avg_power << ",";
						ofs.close();
					}
				}
				time_gpu_update += end_clock(start,end);
			}
		}
		else if((opt) && (node==2)) //Hybrid parallelism
		{
			if(op.approximate)
			{
				checkCudaErrors(cudaMemcpy(taken_d,taken,sizeof(int)*op.k,cudaMemcpyHostToDevice));
				pthread_t thread;
				start_clock(start,end);
				if(op.nvml)
				{
					start_power_sample(op,thread,10);
				}
				bc_gpu_update_edge_opt<true,2><<<dimGrid,dimBlock>>>(bc_gpu_d,R_d,F_d,C_d,g.n,g.m,d_d,sigma_d,delta_d,touched_d,sigma_hat_d,delta_hat_d,moved_d,movement_d,Q_d,Q2_d,QQ_d,temp_d,taken_d,pitch,pitch_sigma,pitch_Q,pitch_temp,source_nodes_d,0,op.k,source,dest);
				checkCudaErrors(cudaPeekAtLastError()); //Check for kernel launch errors
				if(op.nvml)
				{
					float avg_power = end_power_sample(op,thread);
					if(op.power_file == NULL)
					{
						std::cout << "Average power: " << avg_power << std::endl;
					}
					else
					{
						std::ofstream ofs;
						ofs.open(op.power_file, std::ios::app);
						ofs << avg_power << ",";
						ofs.close();
					}
				}
				time_gpu_update += end_clock(start,end);
			}
			else
			{
				checkCudaErrors(cudaMemcpy(taken_d,taken,sizeof(int)*g.n,cudaMemcpyHostToDevice));
				pthread_t thread;
				start_clock(start,end);
				if(op.nvml)
				{
					start_power_sample(op,thread,10);
				}
				bc_gpu_update_edge_opt<false,2><<<dimGrid,dimBlock>>>(bc_gpu_d,R_d,F_d,C_d,g.n,g.m,d_d,sigma_d,delta_d,touched_d,sigma_hat_d,delta_hat_d,moved_d,movement_d,Q_d,Q2_d,QQ_d,temp_d,taken_d,pitch,pitch_sigma,pitch_Q,pitch_temp,source_nodes_d,0,g.n,source,dest);
				checkCudaErrors(cudaPeekAtLastError()); //Check for kernel launch errors
				if(op.nvml)
				{
					float avg_power = end_power_sample(op,thread);
					if(op.power_file == NULL)
					{
						std::cout << "Average power: " << avg_power << std::endl;
					}
					else
					{
						std::ofstream ofs;
						ofs.open(op.power_file, std::ios::app);
						ofs << avg_power << ",";
						ofs.close();
					}
				}
				time_gpu_update += end_clock(start,end);
			}
		}
		else
		{
			if(op.approximate)
			{
				pthread_t thread;
				start_clock(start,end);
				if(op.nvml)
				{
					start_power_sample(op,thread,10);
				}
				bc_gpu_update_edge<true><<<dimGrid,dimBlock>>>(bc_gpu_d,R_d,F_d,C_d,g.n,g.m,d_d,sigma_d,delta_d,pitch,pitch_sigma,source_nodes_d,0,op.k,source,dest);
				checkCudaErrors(cudaPeekAtLastError()); //Check for kernel launch errors
				if(op.nvml)
				{
					float avg_power = end_power_sample(op,thread);
					if(op.power_file == NULL)
					{
						std::cout << "Average power: " << avg_power << std::endl;
					}
					else
					{
						std::ofstream ofs;
						ofs.open(op.power_file, std::ios::app);
						ofs << avg_power << ",";
						ofs.close();
					}
				}
				time_gpu_update += end_clock(start,end);
			}
			else
			{
				pthread_t thread;
				start_clock(start,end);
				if(op.nvml)
				{
					start_power_sample(op,thread,10);
				}
				bc_gpu_update_edge<false><<<dimGrid,dimBlock>>>(bc_gpu_d,R_d,F_d,C_d,g.n,g.m,d_d,sigma_d,delta_d,pitch,pitch_sigma,source_nodes_d,0,g.n,source,dest);
				checkCudaErrors(cudaPeekAtLastError()); //Check for kernel launch errors
				if(op.nvml)
				{
					float avg_power = end_power_sample(op,thread);
					if(op.power_file == NULL)
					{
						std::cout << "Average power: " << avg_power << std::endl;
					}
					else
					{
						std::ofstream ofs;
						ofs.open(op.power_file, std::ios::app);
						ofs << avg_power << ",";
						ofs.close();
					}
				}
				time_gpu_update += end_clock(start,end);
			}
		}

		//No need to transfer back GPU results - we just want them on the device. The source nodes for the approx case won't change, so keep them on.
		checkCudaErrors(cudaFree(F_d));
		checkCudaErrors(cudaFree(C_d));
		checkCudaErrors(cudaFree(R_d));
		
		if(time_gpu_update-prev_time < time_min_update_gpu)
		{
			time_min_update_gpu = time_gpu_update-prev_time;
			min_update_edge_gpu.first = source;
			min_update_edge_gpu.second = dest;
		}
		if(time_gpu_update-prev_time > time_max_update_gpu)
		{
			time_max_update_gpu = time_gpu_update-prev_time;
			max_update_edge_gpu.first = source;
			max_update_edge_gpu.second = dest;
		}
	}
	std::cout << std::endl;

	//Note: If using the code below, the vectors of vectors for d, sigma, and delta must be passed to this function by reference
	if((op.debug) && (opt))
	{
		//Debugging: Copy back d, sigma, delta
		//Move to vectors so they can be easily printed
		int *d_gpu;
		unsigned long long *sigma_gpu;
		float *delta_gpu;
		if(op.approximate)
		{
			d_gpu = new int[g.n*op.k];
			sigma_gpu = new unsigned long long[g.n*op.k];
			delta_gpu = new float[g.n*op.k];
			checkCudaErrors(cudaMemcpy2D(d_gpu,sizeof(int)*g.n,d_d,pitch,sizeof(int)*g.n,op.k,cudaMemcpyDeviceToHost));	
			checkCudaErrors(cudaMemcpy2D(sigma_gpu,sizeof(unsigned long long)*g.n,sigma_d,pitch_sigma,sizeof(unsigned long long)*g.n,op.k,cudaMemcpyDeviceToHost));	
			checkCudaErrors(cudaMemcpy2D(delta_gpu,sizeof(float)*g.n,delta_d,pitch,sizeof(float)*g.n,op.k,cudaMemcpyDeviceToHost));	
			d_gpu_v.resize(op.k);
			sigma_gpu_v.resize(op.k);
			delta_gpu_v.resize(op.k);
			for(int i=0; i<op.k; i++)
			{
				d_gpu_v[i].resize(g.n);
				sigma_gpu_v[i].resize(g.n);
				delta_gpu_v[i].resize(g.n);
				for(int j=0; j<g.n; j++)
				{
					d_gpu_v[i][j] = d_gpu[i*g.n + j];
					sigma_gpu_v[i][j] = sigma_gpu[i*g.n + j];
					delta_gpu_v[i][j] = delta_gpu[i*g.n + j];	
				}
			}
		}
		else
		{
			d_gpu = new int[g.n*g.n];
			sigma_gpu = new unsigned long long[g.n*g.n];
			delta_gpu = new float[g.n*g.n];
			checkCudaErrors(cudaMemcpy2D(d_gpu,sizeof(int)*g.n,d_d,pitch,sizeof(int)*g.n,g.n,cudaMemcpyDeviceToHost));	
			checkCudaErrors(cudaMemcpy2D(sigma_gpu,sizeof(unsigned long long)*g.n,sigma_d,pitch_sigma,sizeof(unsigned long long)*g.n,g.n,cudaMemcpyDeviceToHost));	
			checkCudaErrors(cudaMemcpy2D(delta_gpu,sizeof(float)*g.n,delta_d,pitch,sizeof(float)*g.n,g.n,cudaMemcpyDeviceToHost));	
			d_gpu_v.resize(g.n);
			sigma_gpu_v.resize(g.n);
			delta_gpu_v.resize(g.n);
			for(int i=0; i<g.n; i++)
			{
				d_gpu_v[i].resize(g.n);
				sigma_gpu_v[i].resize(g.n);
				delta_gpu_v[i].resize(g.n);
				for(int j=0; j<g.n; j++)
				{
					d_gpu_v[i][j] = d_gpu[i*g.n + j];
					sigma_gpu_v[i][j] = sigma_gpu[i*g.n + j];
					delta_gpu_v[i][j] = delta_gpu[i*g.n + j];	
				}
			}
		}
		

		delete[] d_gpu;
		delete[] sigma_gpu;
		delete[] delta_gpu;
	}

	//Now that we're done, copy back bc from the updated kernel and free remaining device memory
	checkCudaErrors(cudaMemcpy(bc_gpu,bc_gpu_d,sizeof(float)*g.n,cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(bc_gpu_d));
	checkCudaErrors(cudaFree(d_d));
	checkCudaErrors(cudaFree(sigma_d));
	checkCudaErrors(cudaFree(delta_d));
	if(op.approximate)
	{
		checkCudaErrors(cudaFree(source_nodes_d));
	}
	if(opt)
	{
		checkCudaErrors(cudaFree(touched_d));
		checkCudaErrors(cudaFree(sigma_hat_d));
		checkCudaErrors(cudaFree(delta_hat_d));
		checkCudaErrors(cudaFree(moved_d));
		checkCudaErrors(cudaFree(movement_d));
		checkCudaErrors(cudaFree(taken_d));
		checkCudaErrors(cudaFree(Q_d));
		checkCudaErrors(cudaFree(Q2_d));
		checkCudaErrors(cudaFree(QQ_d));
		checkCudaErrors(cudaFree(temp_d));
	}
}

/*struct thread_data
{
	int start;
	int end;
	csr_graph g;
	bool approx;
	int k;
	int src;
	int dst;
	std::vector< std::vector<int> > &d_old;
	std::vector< std::vector<unsigned long long> > &sigma_old;
	std::vector< std::vector<float> > &delta_old;
};*/

void heterogeneous_streaming_computation(csr_graph &g, program_options op, float *bc_gpu, int *source_nodes, int max_threads_per_block, int number_of_SMs, std::set< std::pair<int,int> > removals_gpu, float &time_heterogeneous_update, float &time_min_update_hetero, float &time_max_update_hetero, std::pair<int,int> &min_update_edge_hetero, std::pair<int,int> &max_update_edge_hetero, float &time_for_accumulation, float &time_CPU)
{
	int devcount;
	checkCudaErrors(cudaGetDeviceCount(&devcount));
	if(devcount < 1)
	{
		std::cerr << "Error detecting CUDA capable GPUs. Aborting." << std::endl;
		exit(-1);
	}

	//Remove edges again so that we can stream
	g.remove_edges(removals_gpu);

	bool firstiter = true;
	int *taken;
	float *bc_gpu_tmp[devcount]; //Intermediate Host Result Pointers
	float *delta_gpu_tmp[devcount];

	//Device Pointers
	float *bc_gpu_d[devcount], *delta_d[devcount];
	int *C_d[devcount], *F_d[devcount], *d_d[devcount], *source_nodes_d[devcount];
	unsigned long long *sigma_d[devcount];
	//Case 2 and 3 Data structures
	int *touched_d[devcount];
	unsigned long long *sigma_hat_d[devcount];
	float *delta_hat_d[devcount];
	int *moved_d[devcount], *movement_d[devcount], *temp_d[devcount], *taken_d[devcount], *Q_d[devcount], *Q2_d[devcount];
	int *QQ_d[devcount], *R_d[devcount];
	
	//Other CUDA parameters
	size_t pitch[devcount], pitch_sigma[devcount], pitch_Q[devcount], pitch_temp[devcount]; //Devices will probably have the same pitch values, but play it safe
	dim3 dimBlock[devcount], dimGrid[devcount];
	cudaEvent_t start, end, start_CPU, end_CPU;

	while(removals_gpu.size())
	{
		//Run full computation on the GPU to get d, sigma, and delta
		if(firstiter)
		{
			//Allocate and transfer data to the GPUs
			for(int dev=0; dev<devcount; dev++)
			{
				checkCudaErrors(cudaSetDevice(dev));
				checkCudaErrors(cudaMalloc((void**)&bc_gpu_d[dev],sizeof(float)*g.n));
				checkCudaErrors(cudaMalloc((void**)&C_d[dev],sizeof(int)*(g.m*2)));
				checkCudaErrors(cudaMalloc((void**)&F_d[dev],sizeof(int)*(g.m*2)));
				if(op.approximate)
				{
					checkCudaErrors(cudaMalloc((void**)&source_nodes_d[dev],sizeof(int)*op.k));
				}

				checkCudaErrors(cudaMemcpy(bc_gpu_d[dev],bc_gpu,sizeof(float)*g.n,cudaMemcpyHostToDevice));
				checkCudaErrors(cudaMemcpy(C_d[dev],g.C,sizeof(int)*(g.m*2),cudaMemcpyHostToDevice));
				checkCudaErrors(cudaMemcpy(F_d[dev],g.F,sizeof(int)*(g.m*2),cudaMemcpyHostToDevice));
				if(op.approximate)
				{
					checkCudaErrors(cudaMemcpy(source_nodes_d[dev],source_nodes,sizeof(int)*op.k,cudaMemcpyHostToDevice));
				}

				//Warning: Taking advantage of the fact that n and k do not change. Setting the number of SMs this way for kernel reuse, not for performance.
				if(op.approximate)
				{
					dimGrid[dev].x = op.k; //Use enough SMs to store all old data
					checkCudaErrors(cudaMallocPitch((void**)&d_d[dev],&pitch[dev],sizeof(int)*g.n,op.k));
					checkCudaErrors(cudaMallocPitch((void**)&sigma_d[dev],&pitch_sigma[dev],sizeof(unsigned long long)*g.n,op.k));
					checkCudaErrors(cudaMallocPitch((void**)&delta_d[dev],&pitch[dev],sizeof(float)*g.n,op.k));
				}
				else
				{
					dimGrid[dev].x = g.n; //Use enough SMs to store all old data
					checkCudaErrors(cudaMallocPitch((void**)&d_d[dev],&pitch[dev],sizeof(int)*g.n,g.n));
					checkCudaErrors(cudaMallocPitch((void**)&sigma_d[dev],&pitch_sigma[dev],sizeof(unsigned long long)*g.n,g.n));
					checkCudaErrors(cudaMallocPitch((void**)&delta_d[dev],&pitch[dev],sizeof(float)*g.n,g.n));
				}

				//Cheating here: We know that the grid/block setup should be the same for both GPUs
				dimBlock[dev].x = max_threads_per_block;
				dimBlock[dev].y = 1;
				dimBlock[dev].z = 1;
				dimGrid[dev].y = 1;
				dimGrid[dev].z = 1;
			}

			//Copy memory to both devices first, then execute both kernels simultaneously to save time
			for(int dev=0; dev<devcount; dev++)
			{
				checkCudaErrors(cudaSetDevice(dev));
				if(op.approximate)
				{
					bc_gpu_opt_approx<<<dimGrid[dev],dimBlock[dev]>>>(bc_gpu_d[dev],F_d[dev],C_d[dev],g.n,g.m,d_d[dev],sigma_d[dev],delta_d[dev],pitch[dev],pitch_sigma[dev],source_nodes_d[dev],op.k,0,op.k);
					checkCudaErrors(cudaPeekAtLastError()); //Check for kernel launch errors
				}
				else
				{
					bc_gpu_opt<<<dimGrid[dev],dimBlock[dev]>>>(bc_gpu_d[dev],F_d[dev],C_d[dev],g.n,g.m,d_d[dev],sigma_d[dev],delta_d[dev],pitch[dev],pitch_sigma[dev],0,g.n);
					checkCudaErrors(cudaPeekAtLastError()); //Check for kernel launch errors
				}
			}

			//Compute initial results on the CPU
			if(op.approximate)
			{
				bc_gpu = bc_no_parents_approx(g,op.k,source_nodes,false,true,d_gpu_v,sigma_gpu_v,delta_gpu_v,0,op.k);
			}
			else
			{
				bc_gpu = bc_no_parents(g,false,true,d_gpu_v,sigma_gpu_v,delta_gpu_v,0,g.n);
			}
			
			//Reset BC scores - we only care about d, sigma, and delta initially
			for(int i=0; i<g.n; i++)
			{
				bc_gpu[i] = 0;
			}


			for(int dev=0; dev<devcount; dev++)
			{
				checkCudaErrors(cudaSetDevice(dev));
				checkCudaErrors(cudaFree(F_d[dev]));
				checkCudaErrors(cudaFree(C_d[dev]));
				dimGrid[dev].x = number_of_SMs; //Again, cheating.
				//Allocate space for touched, sigma_hat, and delta_hat on the GPU.
				checkCudaErrors(cudaMallocPitch((void**)&touched_d[dev],&pitch[dev],sizeof(int)*g.n,number_of_SMs));
				checkCudaErrors(cudaMallocPitch((void**)&sigma_hat_d[dev],&pitch_sigma[dev],sizeof(unsigned long long)*g.n,number_of_SMs));
				checkCudaErrors(cudaMallocPitch((void**)&delta_hat_d[dev],&pitch[dev],sizeof(float)*g.n,number_of_SMs));
				checkCudaErrors(cudaMallocPitch((void**)&moved_d[dev],&pitch[dev],sizeof(int)*g.n,number_of_SMs));
				checkCudaErrors(cudaMallocPitch((void**)&movement_d[dev],&pitch[dev],sizeof(int)*g.n,number_of_SMs));
				checkCudaErrors(cudaMallocPitch((void**)&QQ_d[dev],&pitch[dev],sizeof(int)*g.n,number_of_SMs));
				if(g.n > 550000) //Hack for large graphs to get things to fit into GPU memory
				{
					checkCudaErrors(cudaMallocPitch((void**)&Q_d[dev],&pitch_Q[dev],sizeof(int)*g.n*3,number_of_SMs));
					checkCudaErrors(cudaMallocPitch((void**)&Q2_d[dev],&pitch_Q[dev],sizeof(int)*g.n*3,number_of_SMs));
					checkCudaErrors(cudaMallocPitch((void**)&temp_d[dev],&pitch_temp[dev],sizeof(int)*g.n*6,number_of_SMs));
				}
				else
				{
					checkCudaErrors(cudaMallocPitch((void**)&Q_d[dev],&pitch_Q[dev],sizeof(int)*(g.m/2),number_of_SMs));
					checkCudaErrors(cudaMallocPitch((void**)&Q2_d[dev],&pitch_Q[dev],sizeof(int)*(g.m/2),number_of_SMs));
					checkCudaErrors(cudaMallocPitch((void**)&temp_d[dev],&pitch_temp[dev],sizeof(int)*g.m,number_of_SMs));
				}
				
				if(op.approximate)
				{
					checkCudaErrors(cudaMalloc((void**)&taken_d[dev],sizeof(int)*op.k));
				}
				else
				{
					checkCudaErrors(cudaMalloc((void**)&taken_d[dev],sizeof(int)*g.n));
				}

				bc_gpu_tmp[dev] = new float[g.n];
				for(int i=0; i<g.n; i++)
				{
					bc_gpu_tmp[dev][i] = 0;
				}

				if(op.approximate)
				{
					delta_gpu_tmp[dev] = new float[g.n*op.k];
				}
				else
				{
					delta_gpu_tmp[dev] = new float[g.n*g.n];
				}
			}

			if(op.approximate)
			{
				taken = new int[op.k];
				for(int i=0; i<op.k; i++)
				{
					taken[i] = 0;
				}
			}
			else
			{
				taken = new int[g.n];
				for(int i=0; i<g.n; i++)
				{
					taken[i] = 0;
				}
			}

			firstiter = false;
			std::cout << "Heterogeneous Streaming (" << devcount << " GPUs, " << CPU_THREADS << " CPU threads)" << std::endl;
		}

		//Add edge
		int source = removals_gpu.begin()->first;
		int dest = removals_gpu.begin()->second;
		std::cout << "Inserting edge: (" << source << "," << dest << ")" << std::endl;
		g.reinsert_edge(source,dest);
		removals_gpu.erase(removals_gpu.begin());
		if(removals_gpu.find(std::make_pair(dest,source)) != removals_gpu.end())
		{
			removals_gpu.erase(std::make_pair(dest,source));
		}
		else
		{
			std::cerr << "Error reinserting edges: edge (" << source << "," << dest << ") found but edge (" << dest << "," << source << ") could not be found." << std::endl;
			exit(-1);
		}

		//Update
		float prev_time = time_heterogeneous_update;

		//Reallocate the updated graph
		for(int dev=0; dev<devcount; dev++)
		{
			checkCudaErrors(cudaSetDevice(dev));
			checkCudaErrors(cudaMalloc((void**)&C_d[dev],sizeof(int)*(g.m*2)));
			checkCudaErrors(cudaMalloc((void**)&F_d[dev],sizeof(int)*(g.m*2)));
			checkCudaErrors(cudaMalloc((void**)&R_d[dev],sizeof(int)*(g.n+1)));

			checkCudaErrors(cudaMemcpy(C_d[dev],g.C,sizeof(int)*(g.m*2),cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(F_d[dev],g.F,sizeof(int)*(g.m*2),cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(R_d[dev],g.R,sizeof(int)*(g.n+1),cudaMemcpyHostToDevice));
			if(op.approximate)
			{
				checkCudaErrors(cudaMemcpy(taken_d[dev],taken,sizeof(int)*op.k,cudaMemcpyHostToDevice));
			}
			else
			{
				checkCudaErrors(cudaMemcpy(taken_d[dev],taken,sizeof(int)*g.n,cudaMemcpyHostToDevice));
			}
		}

		//Node-based parallelism only
		start_clock(start,end);
		for(int dev=0; dev<devcount; dev++) 
		{
			checkCudaErrors(cudaSetDevice(dev));
			//std::cout << "Launching GPU " << dev << " kernel" << std::endl;
			//The GPU will compute sources [first_source,last_source)
			if(op.approximate)
			{
				int first_source = dev*ceil(op.k/(float)(devcount+1));
				int last_source = (dev+1)*ceil(op.k/(float)(devcount+1));
				//std::cout << "First source: " << first_source << ". Last source: " << last_source << "." << std::endl;
				bc_gpu_update_edge_opt<true,1><<<dimGrid[dev],dimBlock[dev]>>>(bc_gpu_d[dev],R_d[dev],F_d[dev],C_d[dev],g.n,g.m,d_d[dev],sigma_d[dev],delta_d[dev],touched_d[dev],sigma_hat_d[dev],delta_hat_d[dev],moved_d[dev],movement_d[dev],Q_d[dev],Q2_d[dev],QQ_d[dev],temp_d[dev],taken_d[dev],pitch[dev],pitch_sigma[dev],pitch_Q[dev],pitch_temp[dev],source_nodes_d[dev],first_source,last_source,source,dest);
				checkCudaErrors(cudaPeekAtLastError());
			}
			else
			{
				int first_source = dev*ceil(g.n/(float)(devcount+1));
				int last_source = (dev+1)*ceil(g.n/(float)(devcount+1));
				//std::cout << "First source: " << first_source << ". Last source: " << last_source << "." << std::endl;
				bc_gpu_update_edge_opt<false,1><<<dimGrid[dev],dimBlock[dev]>>>(bc_gpu_d[dev],R_d[dev],F_d[dev],C_d[dev],g.n,g.m,d_d[dev],sigma_d[dev],delta_d[dev],touched_d[dev],sigma_hat_d[dev],delta_hat_d[dev],moved_d[dev],movement_d[dev],Q_d[dev],Q2_d[dev],QQ_d[dev],temp_d[dev],taken_d[dev],pitch[dev],pitch_sigma[dev],pitch_Q[dev],pitch_temp[dev],source_nodes_d[dev],first_source,last_source,source,dest);
				checkCudaErrors(cudaPeekAtLastError());
			}
		}
		//Calculate CPU nodes
		//std::cout << "Launching CPU" << std::endl;

		start_clock(start_CPU,end_CPU);

		int first_source;
		int last_source;
		if(op.approximate)
		{
			first_source = devcount*ceil(op.k/(float)(devcount+1));
			last_source = op.k;
		}
		else
		{
			first_source = devcount*ceil(g.n/(float)(devcount+1));
			last_source = g.n;
		}
		//std::cout << "First source: " << first_source << ". Last source: " << last_source << "." << std::endl;
		pthread_t threads[CPU_THREADS];
		thread_data thread_args[CPU_THREADS];

		for(long t=0; t<CPU_THREADS; t++)
		{
			int thread_begin = first_source + t*ceil((last_source-first_source)/(float)CPU_THREADS);
			int thread_end;
			if(t == CPU_THREADS-1)
			{
				thread_end = last_source;
			}
			else
			{
				thread_end = first_source + (t+1)*ceil((last_source-first_source)/(float)CPU_THREADS);
			}
			//std::cout << "Thread " << t << " start: " << thread_begin << ", end: " << thread_end << std::endl;
			thread_args[t].start = thread_begin;
			thread_args[t].end = thread_end;
			thread_args[t].g = g;
			thread_args[t].approx = op.approximate;
			thread_args[t].k = op.k;
			thread_args[t].src = source;
			thread_args[t].dst = dest;
			int rc = pthread_create(&threads[t],NULL,heterogeneous_update,(void*) &thread_args[t]); 
			if(rc)
			{
				std::cerr << "Error creating thread " << t << ". Return code: " << rc << std::endl;
				exit(-1);
			}
		}
		for(long t=0; t<CPU_THREADS; t++)
		{
			int rc = pthread_join(threads[t],NULL);
			if(rc)
			{
				std::cerr << "Error joining thread " << t << ". Return code: " << rc << std::endl;
			}	
		}
		/*for(int i=first_source; i<last_source; i++)
		{
			if(op.approximate)
			{
				int l = source_nodes[i];
				heterogeneous_update(g,op.approximate,op.k,source,dest,d_gpu_v[i],sigma_gpu_v[i],delta_gpu_v[i],l);
			}
			else
			{
				heterogeneous_update(g,op.approximate,op.k,source,dest,d_gpu_v[i],sigma_gpu_v[i],delta_gpu_v[i],i);
			}
		}*/
		time_CPU += end_clock(start_CPU,end_CPU);
		time_heterogeneous_update += end_clock(start,end);
		
		if(time_heterogeneous_update-prev_time < time_min_update_hetero)
		{
			time_min_update_hetero = time_heterogeneous_update-prev_time;
			min_update_edge_hetero.first = source;
			min_update_edge_hetero.second = dest;
		}
		if(time_heterogeneous_update-prev_time > time_max_update_hetero)
		{
			time_max_update_hetero = time_heterogeneous_update-prev_time;
			max_update_edge_hetero.first = source;
			max_update_edge_hetero.second = dest;
		}

		for(int dev=0; dev<devcount; dev++)
		{
			checkCudaErrors(cudaFree(F_d[dev]));
			checkCudaErrors(cudaFree(C_d[dev]));
			checkCudaErrors(cudaFree(R_d[dev]));
		}
	}
	std::cout << std::endl;

	//Now that we're done, copy back scores and free device memory
	//Copy back delta and figure out contributions from each GPU
	for(int dev=0; dev<devcount; dev++)
	{
		checkCudaErrors(cudaSetDevice(dev));
		if(op.approximate)
		{
			checkCudaErrors(cudaMemcpy2D(delta_gpu_tmp[dev],sizeof(float)*g.n,delta_d[dev],pitch[dev],sizeof(float)*g.n,op.k,cudaMemcpyDeviceToHost));	
		}
		else
		{
			checkCudaErrors(cudaMemcpy2D(delta_gpu_tmp[dev],sizeof(float)*g.n,delta_d[dev],pitch[dev],sizeof(float)*g.n,g.n,cudaMemcpyDeviceToHost));	
		}
		checkCudaErrors(cudaFree(bc_gpu_d[dev]));
		checkCudaErrors(cudaFree(d_d[dev]));
		checkCudaErrors(cudaFree(sigma_d[dev]));
		checkCudaErrors(cudaFree(delta_d[dev]));
		checkCudaErrors(cudaFree(touched_d[dev]));
		checkCudaErrors(cudaFree(sigma_hat_d[dev]));
		checkCudaErrors(cudaFree(delta_hat_d[dev]));
		checkCudaErrors(cudaFree(moved_d[dev]));
		checkCudaErrors(cudaFree(movement_d[dev]));
		checkCudaErrors(cudaFree(taken_d[dev]));
		checkCudaErrors(cudaFree(Q_d[dev]));
		checkCudaErrors(cudaFree(Q2_d[dev]));
		checkCudaErrors(cudaFree(QQ_d[dev]));
		checkCudaErrors(cudaFree(temp_d[dev]));
		if(op.approximate)
		{
			checkCudaErrors(cudaFree(source_nodes_d[dev]));
		}

		/*std::cout << "delta_GPU" << dev << "= {";
		for(int i=0; i<g.n; i++)
		{
			for(int v=0; v<g.n; v++)
			{
				if(v==g.n-1)
				{
					std::cout << delta_gpu_tmp[dev][i*g.n + v] << "}";
				}
				else if(v==0)
				{
					std::cout << "{" << delta_gpu_tmp[dev][i*g.n + v] << ",";
				}
				else
				{
					std::cout << delta_gpu_tmp[dev][i*g.n + v] << ",";
				}
			}	
		}
		std::cout << "}" << std::endl;*/

		start_clock(start,end);
		if(op.approximate)
		{
			int first_source = dev*ceil(op.k/(float)(devcount+1));
			int last_source = (dev+1)*ceil(op.k/(float)(devcount+1));
			for(int i=0; i<op.k; i++)
			{
				int l = source_nodes[i];
				if((i < last_source) && (i >= first_source))
				{
					for(int v=0; v<g.n; v++)
					{
						if(v != l)
						{
							bc_gpu[v] += delta_gpu_tmp[dev][i*g.n + v];
						}
					}
				}
			}
		}
		else
		{
			int first_source = dev*ceil(g.n/(float)(devcount+1));
			int last_source = (dev+1)*ceil(g.n/(float)(devcount+1));
			//std::cout << "First source: " << first_source << ". Last source: " << last_source << "." << std::endl;
			for(int i=0; i<g.n; i++)
			{
				if((i < last_source) && (i >= first_source))
				{
					for(int v=0; v<g.n; v++)
					{
						if(v != i)
						{
							//std::cout << "Device " << dev << " adding " << delta_gpu_tmp[dev][i*g.n +v] << " to BC[" << v << "]" << std::endl;
							bc_gpu[v] += delta_gpu_tmp[dev][i*g.n + v];	
						}
					}
				}
			}
		}
		time_for_accumulation += end_clock(start,end);
		delete[] bc_gpu_tmp[dev];
		delete[] delta_gpu_tmp[dev];
	}

	//Now add CPU side contributions
	start_clock(start,end);
	if(op.approximate)
	{
		int first_source = devcount*ceil(op.k/(float)(devcount+1));
		int last_source = op.k;
		for(int i=0; i<op.k; i++)
		{
			int l = source_nodes[i];
			if((i < last_source) && (i >= first_source))
			{
				for(int v=0; v<g.n; v++)
				{
					if(v != l)
					{
						bc_gpu[v] += delta_gpu_v[i][v];
					}
				}
			}
		}
	}
	else
	{
		int first_source = devcount*ceil(g.n/(float)(devcount+1));
		int last_source = g.n;
		//std::cout << "First source: " << first_source << ". Last source: " << last_source << "." << std::endl;
		for(int i=0; i<g.n; i++)
		{
			if((i < last_source) && (i >= first_source))
			{
				for(int v=0; v<g.n; v++)
				{
					if(v != i)
					{
						//std::cout << "Device " << devcount << " adding " << delta_gpu_v[i][v] << " to BC[" << v << "]" << std::endl;
						bc_gpu[v] += delta_gpu_v[i][v];
					}
				}
			}
		}	
	}
	time_heterogeneous_update += end_clock(start,end);

	/*std::cout << "BC = {";
	for(int i=0; i<g.n; i++)
	{
		if(i != g.n-1)
		{
			std::cout << bc_gpu[i] << ",";
		}
		else
		{
			std::cout << bc_gpu[i] << "}" << std::endl;
		}
	}*/

	delete[] taken;

}

void single_gpu_streaming_computation_SOA(csr_graph &g, program_options op, float *bc_gpu, int *source_nodes, int max_threads_per_block, int number_of_SMs, std::set< std::pair<int,int> > removals_gpu, float &time_gpu_update, float &time_min_update_gpu, float &time_max_update_gpu)
{
	//Remove edges again so that we can stream on the GPU
	g.remove_edges(removals_gpu);
	std::cout << "SoA GPU Streaming: " << std::endl;
	bool firstiter = true;

	//Device Pointers
	float *bc_gpu_d, *delta_d;  
	int *C_d, *F_d, *d_d, *source_nodes_d;
	unsigned long long *sigma_d;
	//Allocate worst case scenario case 2 data structures so we don't need to malloc/free within a kernel
	int *touched_d;
	unsigned long long *sigma_hat_d;
	float *delta_hat_d;
	int *moved_d; //Could make this a bool with a possibly separate pitch to save some memory
	int *movement_d;
	int *taken_d; //Work stealing flag
	int *taken;
	int *Q_d, *Q2_d, *QQ_d; //Case 3 queues
	int *R_d;
	//Allocate 2D arrays so that each block has its own global data (because it won't all fit in shared)
	//Since d and delta are all the same size they will all have the same pitch.
	//This is NOT the case in general, so we're really exploiting the pitch size here.
	size_t pitch, pitch_sigma;
	dim3 dimBlock, dimGrid;
	cudaEvent_t start,end;
	//SoA pointers
	vertex_data *v_soa, *v_d;
	while(removals_gpu.size())
	{
		//Run full computation on the GPU so we have d, sigma, and delta
		if(firstiter)
		{
			//Reset bc scores
			for(int i=0; i<g.n; i++)
			{
				bc_gpu[i] = 0;
			}

			//Allocate and transfer data to the GPU
			checkCudaErrors(cudaMalloc((void**)&bc_gpu_d,sizeof(float)*g.n));
			checkCudaErrors(cudaMalloc((void**)&C_d,sizeof(int)*(g.m*2)));
			checkCudaErrors(cudaMalloc((void**)&F_d,sizeof(int)*(g.m*2)));
			if(op.approximate)
			{
				checkCudaErrors(cudaMalloc((void**)&source_nodes_d,sizeof(int)*op.k));
			}

			checkCudaErrors(cudaMemcpy(bc_gpu_d,bc_gpu,sizeof(float)*g.n,cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(C_d,g.C,sizeof(int)*(g.m*2),cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(F_d,g.F,sizeof(int)*(g.m*2),cudaMemcpyHostToDevice));
			if(op.approximate)
			{
				checkCudaErrors(cudaMemcpy(source_nodes_d,source_nodes,sizeof(int)*op.k,cudaMemcpyHostToDevice));
			}
			
			//Potentially dangerous quick fix:
			//Setting the number of SMs this way allows for reuse of the same kernels, which is nice because we know it works.
			//This is not good for performance, but in this case we're just generating results we can build on, so we don't care.
			//For large graphs in the exact case, this will blow up. But the n^2 storage might have blown things up anyway.	
			//This also takes advantage of the fact that k and n don't change as we stream the graph (because we only add edges).

			if(op.approximate)
			{
				dimGrid.x = op.k; //Use enough SMs to store all old data
				checkCudaErrors(cudaMallocPitch((void**)&d_d,&pitch,sizeof(int)*g.n,op.k));
				checkCudaErrors(cudaMallocPitch((void**)&sigma_d,&pitch_sigma,sizeof(unsigned long long)*g.n,op.k));
				checkCudaErrors(cudaMallocPitch((void**)&delta_d,&pitch,sizeof(float)*g.n,op.k));
			}
			else
			{
				dimGrid.x = g.n; //Use enough SMs to store all old data
				checkCudaErrors(cudaMallocPitch((void**)&d_d,&pitch,sizeof(int)*g.n,g.n));
				checkCudaErrors(cudaMallocPitch((void**)&sigma_d,&pitch_sigma,sizeof(unsigned long long)*g.n,g.n));
				checkCudaErrors(cudaMallocPitch((void**)&delta_d,&pitch,sizeof(float)*g.n,g.n));
			}

			//Set kernel dimensions
			dimBlock.x = max_threads_per_block; 
			dimBlock.y = 1;
			dimBlock.z = 1;
			//dimGrid.x = number_of_SMs;
			dimGrid.y = 1;
			dimGrid.z = 1;

			if(op.approximate)
			{
				bc_gpu_opt_approx<<<dimGrid,dimBlock>>>(bc_gpu_d,F_d,C_d,g.n,g.m,d_d,sigma_d,delta_d,pitch,pitch_sigma,source_nodes_d,op.k,0,op.k);
				checkCudaErrors(cudaPeekAtLastError()); //Check for kernel launch errors
			}
			else
			{
				bc_gpu_opt<<<dimGrid,dimBlock>>>(bc_gpu_d,F_d,C_d,g.n,g.m,d_d,sigma_d,delta_d,pitch,pitch_sigma,0,g.n);
				checkCudaErrors(cudaPeekAtLastError()); //Check for kernel launch errors
			}

			//No need to transfer back GPU results - we just want them on the device. The source nodes for the approx case won't change, so keep them on.
			checkCudaErrors(cudaFree(F_d));
			checkCudaErrors(cudaFree(C_d));
			
			firstiter = false;
			dimGrid.x = number_of_SMs; //Use optimal number of SMs

			//Allocate space for touched, sigma_hat, and delta_hat on the GPU - but only do this once
			v_soa = (vertex_data *) malloc(sizeof(vertex_data));
			checkCudaErrors(cudaMalloc((void**)&v_d,sizeof(vertex_data)));
			checkCudaErrors(cudaMalloc((void**)&touched_d,sizeof(int)*g.n*number_of_SMs));
			checkCudaErrors(cudaMalloc((void**)&sigma_hat_d,sizeof(unsigned long long)*g.n*number_of_SMs));
			checkCudaErrors(cudaMalloc((void**)&delta_hat_d,sizeof(float)*g.n*number_of_SMs));
			checkCudaErrors(cudaMalloc((void**)&moved_d,sizeof(int)*g.n*number_of_SMs));
			checkCudaErrors(cudaMalloc((void**)&movement_d,sizeof(int)*g.n*number_of_SMs));
			checkCudaErrors(cudaMalloc((void**)&Q_d,sizeof(int)*g.n*number_of_SMs));
			checkCudaErrors(cudaMalloc((void**)&Q2_d,sizeof(int)*g.n*number_of_SMs));
			checkCudaErrors(cudaMalloc((void**)&QQ_d,sizeof(int)*g.n*number_of_SMs));
			v_soa->touched = touched_d; //Point to device pointer in host struct
			v_soa->sigma_hat = sigma_hat_d;
			v_soa->delta_hat = delta_hat_d;
			v_soa->moved = moved_d;
			v_soa->movement = movement_d;
			v_soa->Q = Q_d;
			v_soa->Q2 = Q2_d;
			v_soa->QQ = QQ_d;
			checkCudaErrors(cudaMemcpy(v_d,v_soa,sizeof(vertex_data), cudaMemcpyHostToDevice));

			if(op.approximate)
			{
				checkCudaErrors(cudaMalloc((void**)&taken_d,sizeof(int)*op.k));
				taken = new int[op.k];
				for(int i=0; i<op.k; i++)
				{
					taken[i] = 0;
				}
			}
			else
			{
				checkCudaErrors(cudaMalloc((void**)&taken_d,sizeof(int)*g.n));
				taken = new int[g.n];
				for(int i=0; i<g.n; i++)
				{
					taken[i] = 0;
				}
			}
		}

		//Add edge
		int source = removals_gpu.begin()->first;
		int dest = removals_gpu.begin()->second;
		std::cout << "Inserting edge: (" << source << "," << dest << ")" << std::endl;
		g.reinsert_edge(source,dest);
		removals_gpu.erase(removals_gpu.begin());
		if(removals_gpu.find(std::make_pair(dest,source)) != removals_gpu.end())
		{
			removals_gpu.erase(std::make_pair(dest,source));
		}
		else
		{
			std::cerr << "Error reinserting edges: edge (" << source << "," << dest << ") found but edge (" << dest << "," << source << ") could not be found." << std::endl;
			exit(-1);
		}

		//Update on the GPU
		float prev_time = time_gpu_update;
	
		//Reallocate the updated graph
		graph_data *g_soa = (graph_data *) malloc(sizeof(graph_data));
		graph_data *g_d;
		checkCudaErrors(cudaMalloc((void**)&g_d,sizeof(graph_data))); //Allocate struct pointer
		checkCudaErrors(cudaMalloc((void**)&C_d,sizeof(int)*(g.m*2))); //Allocate struct members
		checkCudaErrors(cudaMalloc((void**)&F_d,sizeof(int)*(g.m*2)));
		checkCudaErrors(cudaMalloc((void**)&R_d,sizeof(int)*(g.n+1)));
		checkCudaErrors(cudaMemcpy(C_d,g.C,sizeof(int)*(g.m*2),cudaMemcpyHostToDevice)); //Copy struct members
		checkCudaErrors(cudaMemcpy(F_d,g.F,sizeof(int)*(g.m*2),cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(R_d,g.R,sizeof(int)*(g.n+1),cudaMemcpyHostToDevice));
		g_soa->C = C_d; //Point to device pointer in host struct
		g_soa->F = F_d;
		g_soa->R = R_d;
		checkCudaErrors(cudaMemcpy(g_d,g_soa,sizeof(graph_data), cudaMemcpyHostToDevice));
		
		if(op.approximate)
		{
			checkCudaErrors(cudaMemcpy(taken_d,taken,sizeof(int)*op.k,cudaMemcpyHostToDevice));
			pthread_t thread;
			start_clock(start,end);
			if(op.nvml)
			{
				start_power_sample(op,thread,10);
			}
			bc_gpu_update_edge_SOA<true><<<dimGrid,dimBlock>>>(bc_gpu_d,g_d,g.n,g.m,d_d,sigma_d,delta_d,v_d,taken_d,pitch,pitch_sigma,source_nodes_d,0,op.k,source,dest);
			checkCudaErrors(cudaPeekAtLastError()); //Check for kernel launch errors
			if(op.nvml)
			{
				float avg_power = end_power_sample(op,thread);
				std::cout << "Average power: " << avg_power << std::endl;
			}
			time_gpu_update += end_clock(start,end);
		}
		else
		{
			checkCudaErrors(cudaMemcpy(taken_d,taken,sizeof(int)*g.n,cudaMemcpyHostToDevice));
			pthread_t thread;
			start_clock(start,end);
			if(op.nvml)
			{
				start_power_sample(op,thread,10);
			}
			bc_gpu_update_edge_SOA<false><<<dimGrid,dimBlock>>>(bc_gpu_d,g_d,g.n,g.m,d_d,sigma_d,delta_d,v_d,taken_d,pitch,pitch_sigma,source_nodes_d,0,g.n,source,dest);
			checkCudaErrors(cudaPeekAtLastError()); //Check for kernel launch errors
			if(op.nvml)
			{
				float avg_power = end_power_sample(op,thread);
				std::cout << "Average power: " << avg_power << std::endl;
			}
			time_gpu_update += end_clock(start,end);
		}

		//No need to transfer back GPU results - we just want them on the device. The source nodes for the approx case won't change, so keep them on.
		checkCudaErrors(cudaFree(R_d));
		checkCudaErrors(cudaFree(F_d));
		checkCudaErrors(cudaFree(C_d));
		checkCudaErrors(cudaFree(g_d));
		free(g_soa);
		
		if(time_gpu_update-prev_time < time_min_update_gpu)
		{
			time_min_update_gpu = time_gpu_update-prev_time;
		}
		if(time_gpu_update-prev_time > time_max_update_gpu)
		{
			time_max_update_gpu = time_gpu_update-prev_time;
		}
	}
	std::cout << std::endl;

	//Now that we're done, copy back bc from the updated kernel and free remaining device memory
	checkCudaErrors(cudaMemcpy(bc_gpu,bc_gpu_d,sizeof(float)*g.n,cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(bc_gpu_d));
	checkCudaErrors(cudaFree(d_d));
	checkCudaErrors(cudaFree(sigma_d));
	checkCudaErrors(cudaFree(delta_d));
	if(op.approximate)
	{
		checkCudaErrors(cudaFree(source_nodes_d));
	}
	checkCudaErrors(cudaFree(touched_d));
	checkCudaErrors(cudaFree(sigma_hat_d));
	checkCudaErrors(cudaFree(delta_hat_d));
	checkCudaErrors(cudaFree(moved_d));
	checkCudaErrors(cudaFree(movement_d));
	checkCudaErrors(cudaFree(taken_d));
	checkCudaErrors(cudaFree(Q_d));
	checkCudaErrors(cudaFree(Q2_d));
	checkCudaErrors(cudaFree(QQ_d));
	checkCudaErrors(cudaFree(v_d));
}

void single_gpu_streaming_computation_AOS(csr_graph &g, program_options op, float *bc_gpu, int *source_nodes, int max_threads_per_block, int number_of_SMs, std::set< std::pair<int,int> > removals_gpu, float &time_gpu_update, float &time_min_update_gpu, float &time_max_update_gpu)
{
	//Remove edges again so that we can stream on the GPU
	g.remove_edges(removals_gpu);
	std::cout << "AoS GPU Streaming: " << std::endl;
	bool firstiter = true;

	//Device Pointers
	float *bc_gpu_d, *delta_d;  
	int *C_d, *F_d, *d_d, *source_nodes_d;
	unsigned long long *sigma_d;
	//Allocate worst case scenario case 2 data structures so we don't need to malloc/free within a kernel
	/*int *touched_d;
	unsigned long long *sigma_hat_d;
	float *delta_hat_d;
	int *moved_d; //Could make this a bool with a possibly separate pitch to save some memory
	int *movement_d;*/
	int *taken_d; //Work stealing flag
	int *taken;
	//int *Q_d, *Q2_d, *QQ_d; //Case 3 queues
	int *R_d;
	//Allocate 2D arrays so that each block has its own global data (because it won't all fit in shared)
	//Since d and delta are all the same size they will all have the same pitch.
	//This is NOT the case in general, so we're really exploiting the pitch size here.
	size_t pitch, pitch_sigma;
	dim3 dimBlock, dimGrid;
	cudaEvent_t start,end;

	//AoS pointer
	vertex_data_aos *v_d;
	while(removals_gpu.size())
	{
		//Run full computation on the GPU so we have d, sigma, and delta
		if(firstiter)
		{
			//Reset bc scores
			for(int i=0; i<g.n; i++)
			{
				bc_gpu[i] = 0;
			}

			//Allocate and transfer data to the GPU
			checkCudaErrors(cudaMalloc((void**)&bc_gpu_d,sizeof(float)*g.n));
			checkCudaErrors(cudaMalloc((void**)&C_d,sizeof(int)*(g.m*2)));
			checkCudaErrors(cudaMalloc((void**)&F_d,sizeof(int)*(g.m*2)));
			if(op.approximate)
			{
				checkCudaErrors(cudaMalloc((void**)&source_nodes_d,sizeof(int)*op.k));
			}

			checkCudaErrors(cudaMemcpy(bc_gpu_d,bc_gpu,sizeof(float)*g.n,cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(C_d,g.C,sizeof(int)*(g.m*2),cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(F_d,g.F,sizeof(int)*(g.m*2),cudaMemcpyHostToDevice));
			if(op.approximate)
			{
				checkCudaErrors(cudaMemcpy(source_nodes_d,source_nodes,sizeof(int)*op.k,cudaMemcpyHostToDevice));
			}
			
			//Potentially dangerous quick fix:
			//Setting the number of SMs this way allows for reuse of the same kernels, which is nice because we know it works.
			//This is not good for performance, but in this case we're just generating results we can build on, so we don't care.
			//For large graphs in the exact case, this will blow up. But the n^2 storage might have blown things up anyway.	
			//This also takes advantage of the fact that k and n don't change as we stream the graph (because we only add edges).

			if(op.approximate)
			{
				dimGrid.x = op.k; //Use enough SMs to store all old data
				checkCudaErrors(cudaMallocPitch((void**)&d_d,&pitch,sizeof(int)*g.n,op.k));
				checkCudaErrors(cudaMallocPitch((void**)&sigma_d,&pitch_sigma,sizeof(unsigned long long)*g.n,op.k));
				checkCudaErrors(cudaMallocPitch((void**)&delta_d,&pitch,sizeof(float)*g.n,op.k));
			}
			else
			{
				dimGrid.x = g.n; //Use enough SMs to store all old data
				checkCudaErrors(cudaMallocPitch((void**)&d_d,&pitch,sizeof(int)*g.n,g.n));
				checkCudaErrors(cudaMallocPitch((void**)&sigma_d,&pitch_sigma,sizeof(unsigned long long)*g.n,g.n));
				checkCudaErrors(cudaMallocPitch((void**)&delta_d,&pitch,sizeof(float)*g.n,g.n));
			}

			//Set kernel dimensions
			dimBlock.x = max_threads_per_block; 
			dimBlock.y = 1;
			dimBlock.z = 1;
			//dimGrid.x = number_of_SMs;
			dimGrid.y = 1;
			dimGrid.z = 1;

			if(op.approximate)
			{
				bc_gpu_opt_approx<<<dimGrid,dimBlock>>>(bc_gpu_d,F_d,C_d,g.n,g.m,d_d,sigma_d,delta_d,pitch,pitch_sigma,source_nodes_d,op.k,0,op.k);
				checkCudaErrors(cudaPeekAtLastError()); //Check for kernel launch errors
			}
			else
			{
				bc_gpu_opt<<<dimGrid,dimBlock>>>(bc_gpu_d,F_d,C_d,g.n,g.m,d_d,sigma_d,delta_d,pitch,pitch_sigma,0,g.n);
				checkCudaErrors(cudaPeekAtLastError()); //Check for kernel launch errors
			}

			//No need to transfer back GPU results - we just want them on the device. The source nodes for the approx case won't change, so keep them on.
			checkCudaErrors(cudaFree(F_d));
			checkCudaErrors(cudaFree(C_d));
			
			firstiter = false;
			dimGrid.x = number_of_SMs; //Use optimal number of SMs

			//Allocate space for touched, sigma_hat, and delta_hat on the GPU - but only do this once
			//Risky - using same pitch/pitch_sigma parameters. They do turn out to be correct, though.
			checkCudaErrors(cudaMalloc((void**)&v_d,sizeof(vertex_data_aos)*g.n*number_of_SMs));
			/*checkCudaErrors(cudaMallocPitch((void**)&touched_d,&pitch,sizeof(int)*g.n,number_of_SMs));
			checkCudaErrors(cudaMallocPitch((void**)&sigma_hat_d,&pitch_sigma,sizeof(unsigned long long)*g.n,number_of_SMs));
			checkCudaErrors(cudaMallocPitch((void**)&delta_hat_d,&pitch,sizeof(float)*g.n,number_of_SMs));
			checkCudaErrors(cudaMallocPitch((void**)&moved_d,&pitch,sizeof(int)*g.n,number_of_SMs));
			checkCudaErrors(cudaMallocPitch((void**)&movement_d,&pitch,sizeof(int)*g.n,number_of_SMs));
			checkCudaErrors(cudaMallocPitch((void**)&Q_d,&pitch,sizeof(int)*g.n,number_of_SMs));
			checkCudaErrors(cudaMallocPitch((void**)&Q2_d,&pitch,sizeof(int)*g.n,number_of_SMs));
			checkCudaErrors(cudaMallocPitch((void**)&QQ_d,&pitch,sizeof(int)*g.n,number_of_SMs));*/

			if(op.approximate)
			{
				checkCudaErrors(cudaMalloc((void**)&taken_d,sizeof(int)*op.k));
				taken = new int[op.k];
				for(int i=0; i<op.k; i++)
				{
					taken[i] = 0;
				}
			}
			else
			{
				checkCudaErrors(cudaMalloc((void**)&taken_d,sizeof(int)*g.n));
				taken = new int[g.n];
				for(int i=0; i<g.n; i++)
				{
					taken[i] = 0;
				}
			}
		}

		//Add edge
		int source = removals_gpu.begin()->first;
		int dest = removals_gpu.begin()->second;
		std::cout << "Inserting edge: (" << source << "," << dest << ")" << std::endl;
		g.reinsert_edge(source,dest);
		removals_gpu.erase(removals_gpu.begin());
		if(removals_gpu.find(std::make_pair(dest,source)) != removals_gpu.end())
		{
			removals_gpu.erase(std::make_pair(dest,source));
		}
		else
		{
			std::cerr << "Error reinserting edges: edge (" << source << "," << dest << ") found but edge (" << dest << "," << source << ") could not be found." << std::endl;
			exit(-1);
		}

		//Update on the GPU
		float prev_time = time_gpu_update;
	
		//Reallocate the updated graph: It would waste some space, but we can throw R into the graph_data_aos struct too.
		graph_data_aos *g_aos = (graph_data_aos *) malloc((g.m*2)*sizeof(graph_data_aos));
		for(int i=0; i<(2*g.m); i++)
		{
			g_aos[i].F = g.F[i];
			g_aos[i].C = g.C[i];
		}
		graph_data_aos *g_d;
		checkCudaErrors(cudaMalloc((void**)&g_d,(g.m*2)*sizeof(graph_data_aos))); //Allocate struct pointer
		checkCudaErrors(cudaMalloc((void**)&R_d,sizeof(int)*(g.n+1)));
		checkCudaErrors(cudaMemcpy(g_d,g_aos,sizeof(graph_data_aos)*(2*g.m),cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(R_d,g.R,sizeof(int)*(g.n+1),cudaMemcpyHostToDevice));
		
		if(op.approximate)
		{
			checkCudaErrors(cudaMemcpy(taken_d,taken,sizeof(int)*op.k,cudaMemcpyHostToDevice));
			pthread_t thread;
			start_clock(start,end);
			if(op.nvml)
			{
				start_power_sample(op,thread,10);
			}
			bc_gpu_update_edge_AOS<true><<<dimGrid,dimBlock>>>(bc_gpu_d,R_d,g_d,g.n,g.m,d_d,sigma_d,delta_d,v_d,taken_d,pitch,pitch_sigma,source_nodes_d,0,op.k,source,dest);
			checkCudaErrors(cudaPeekAtLastError()); //Check for kernel launch errors
			if(op.nvml)
			{
				float avg_power = end_power_sample(op,thread);
				std::cout << "Average power: " << avg_power << std::endl;
			}
			time_gpu_update += end_clock(start,end);
		}
		else
		{
			checkCudaErrors(cudaMemcpy(taken_d,taken,sizeof(int)*g.n,cudaMemcpyHostToDevice));
			pthread_t thread;
			start_clock(start,end);
			if(op.nvml)
			{
				start_power_sample(op,thread,10);
			}
			bc_gpu_update_edge_AOS<false><<<dimGrid,dimBlock>>>(bc_gpu_d,R_d,g_d,g.n,g.m,d_d,sigma_d,delta_d,v_d,taken_d,pitch,pitch_sigma,source_nodes_d,0,g.n,source,dest);
			checkCudaErrors(cudaPeekAtLastError()); //Check for kernel launch errors
			if(op.nvml)
			{
				float avg_power = end_power_sample(op,thread);
				std::cout << "Average power: " << avg_power << std::endl;
			}
			time_gpu_update += end_clock(start,end);
		}

		//No need to transfer back GPU results - we just want them on the device. The source nodes for the approx case won't change, so keep them on.
		checkCudaErrors(cudaFree(R_d));
		checkCudaErrors(cudaFree(g_d));
		free(g_aos);
		
		if(time_gpu_update-prev_time < time_min_update_gpu)
		{
			time_min_update_gpu = time_gpu_update-prev_time;
		}
		if(time_gpu_update-prev_time > time_max_update_gpu)
		{
			time_max_update_gpu = time_gpu_update-prev_time;
		}
	}
	std::cout << std::endl;

	//Now that we're done, copy back bc from the updated kernel and free remaining device memory
	checkCudaErrors(cudaMemcpy(bc_gpu,bc_gpu_d,sizeof(float)*g.n,cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(bc_gpu_d));
	checkCudaErrors(cudaFree(d_d));
	checkCudaErrors(cudaFree(sigma_d));
	checkCudaErrors(cudaFree(delta_d));
	if(op.approximate)
	{
		checkCudaErrors(cudaFree(source_nodes_d));
	}
	/*checkCudaErrors(cudaFree(touched_d));
	checkCudaErrors(cudaFree(sigma_hat_d));
	checkCudaErrors(cudaFree(delta_hat_d));
	checkCudaErrors(cudaFree(moved_d));
	checkCudaErrors(cudaFree(movement_d));
	checkCudaErrors(cudaFree(taken_d));
	checkCudaErrors(cudaFree(Q_d));
	checkCudaErrors(cudaFree(Q2_d));
	checkCudaErrors(cudaFree(QQ_d));*/
	checkCudaErrors(cudaFree(v_d));
}
