#include "util.cuh"
#include <getopt.h>

//Note: Times returned are in milliseconds
void start_clock(cudaEvent_t &start, cudaEvent_t &end)
{
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&end));
	checkCudaErrors(cudaEventRecord(start,0));
}

float end_clock(cudaEvent_t &start, cudaEvent_t &end)
{
	float time;
	checkCudaErrors(cudaEventRecord(end,0));
	checkCudaErrors(cudaEventSynchronize(end));
	checkCudaErrors(cudaEventElapsedTime(&time,start,end));
	time = time/(float)nreps;
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(end));

	return time;
}

void start_power_sample(program_options op, pthread_t &thread, long period)
{
	if(op.device != 1) //Could add an isTelsa flag, or use the NVML library directly to ensure that power can be measured from the GPU of interest
	{
		std::cerr << "Warning: Power can only be measured for Tesla GPUs." << std::endl;
	}
	else
	{
		//Spawn pthread for power measurement
		psample = new bool;
		*psample = true;
		pthread_create(&thread, NULL, power_sample, (void*)period);
		//std::cout << "Thread created." << std::endl;
		//std::cout << "Psample in main: " << *psample << std::endl;
	}
}

float end_power_sample(program_options op, pthread_t &thread)
{
	if(op.device == 1) //Again, hard coding the Tesla here.
	{
		cudaDeviceSynchronize();
		*psample = false;
		void* res;
		float avg_power;
		pthread_join(thread, (void**)&res);
		avg_power = *(float*)res;
		//std::cout << "Thread joined." << std::endl;
		delete psample;
		free(res); //Deleting void* is undefined, so we need to use malloc/free here. Gross.
		return avg_power;
	}

	return -1;
}

void parse_arguments(int argc, char **argv, program_options &op)
{
	int c;
	static struct option long_options[] =
	{
		{"infile",required_argument,0,'i'},
		{"print",no_argument,0,'p'},
		{"verify",no_argument,0,'v'},
		{"sequential",no_argument,0,'s'},
		{"source_nodes",required_argument,0,'k'},
		{"debug",no_argument,0,'d'},
		{"help",no_argument,0,'h'},
		{"gpu",required_argument,0,'g'},
		{"stream",required_argument,0,'t'},
		{"multi",no_argument,0,'m'},
		{"experiment",optional_argument,0,'e'},
		{"nvml",optional_argument,0,'n'},
		{"concurrency",required_argument,0,'c'},
		{"cpu Off",no_argument,0,'o'},
		{"export File",no_argument,0,'f'},
		{0,0,0,0} //Terminate with null
	};

	int option_index = 0;

	while((c = getopt_long(argc,argv,"c:de::fg:hi:k:mn::opst:v",long_options,&option_index)) != -1)
	{
		switch(c)
		{
			case 'c':
				op.node = atoi(optarg);
			break;

			case 'd':
				op.debug = true;		
			break;

			case 'e':
				op.experiment = true;
				op.result_file = optarg;
			break;

			case 'f':
				op.export_edge_graph = true;
			break;

			case 'g':
				op.device = atoi(optarg);	
			break;

			case 'h':
				std::cout << "Usage: " << argv[0] << " -i <graph input file> [optional arguments]" << std::endl << std::endl;
			        std::cout << "Options: " << std::endl;
				std::cout << "-p print BC scores to stdout" << std::endl;
			        std::cout << "-v verify GPU results with the CPU" << std::endl;
				std::cout << "-k <number of source nodes> Approximate BC using a given number of random source nodes" << std::endl;
				std::cout << "-t Streaming BC" << std::endl;
			       	std::cout << "-g <GPU> Choose which GPU to run the program on. Should be a number from 0 to n, where n is the number of GPUs in the system. This option is meant for single GPU execution only." << std::endl;
				std::cout << "-m multi-GPU algorithm" << std::endl;
			exit(0);

			case 'i':
				op.infile = optarg;
			break;

			case 'k':
				op.approximate = true;	
				op.k = atoi(optarg);
			break;

			case 'm': //Used to be a static multi-GPU computation - will now be a heterogeneous dynamic calculation.
				op.multi = true;
			break;

			case 'n':
				op.nvml = true;
				op.power_file = optarg;
			break;

			case 'o':
				op.no_cpu = true;
			break;

			case 'p':
				op.printBCscores = true;
			break;

			case 's':
				op.verify_all = true;	
			break;

			case 't':
				op.streaming = true;
				op.insertions = atoi(optarg);
			break;

			case 'v':
				op.verify_results = true;
			break;
		
			case '?': //Invalid argument: getopt will print the error message

			exit(-1);

			default: //Fatal error
				std::cerr << "Internal error parsing arguments." << std::endl;
			exit(-1);
		}
	}

	//Handle required command line options here
	//getopt ensures flags that need arguments have them, but does not check for required flags themselves
	if(op.infile == NULL)
	{
		std::cerr << "Command line error: Graph input file is required. Use the -i switch." << std::endl;
		exit(-1);
	}
	if(op.approximate && (op.k == -1))
	{
		std::cerr << "Command line error: Approximation requested but no number of source nodes given. Defaulting to 128." <<std::endl;
		op.k = 128;
	}
	if(op.streaming && (op.insertions == -1))
	{
		std::cerr << "Command line error: Streaming requested but no number of insertions given. Defaulting to 5." << std::endl;
		op.insertions = 5;
	}
	if((op.experiment) && (op.result_file == NULL))
	{
		std::cout << "Result file not given for experiments. Defaulting to stdout." << std::endl;
		std::cout << "Warning: This argument requires either -e<file> or --experiment=<file> syntax. No spacing is allowed. This is a known issue with getopt." << std::endl; //Sigh..
	}
	if((op.streaming) && (op.node == -1))
	{
		//Node vs. edge parallelism hasn't been decided...choose node by default
		op.node = 1;
	}
	if((op.node < - 1) || (op.node > 2))
	{
		std::cerr << "Command line error: Invalid value for node/edge parallelism. Choose 0 for edge-based, 1 for node-based, or 2 for hybrid." << std::endl;
		exit(-1);
	}
	if((op.nvml) && (op.power_file == NULL))
	{
		std::cout << "Output file not given for power measurements. Defaulting to stdout." << std::endl;
		std::cout << "Warning: This argument requires either -n<file> or --nvml=<file> syntax. No spacing is allowed." << std::endl;
	}
}

void verify(float *expected, float *actual, csr_graph g)
{
	double error = 0;
	double maxdiff = 0;
	for(int i=0; i<g.n; i++)
	{
		error += pow((expected[i] - actual[i]),2);
		if(fabs(expected[i] - actual[i]) > maxdiff)
		{
			maxdiff = fabs(expected[i] - actual[i]);
		}
	}
	//error = error/(double)g.n;
	error = sqrt(error);

	std::cout << "Maximum error: " << maxdiff << std::endl;
	std::cout << "Norm(bc-bc_gpu,2): " << error << std::endl;

	if(error < 1e-3)
	{
		std::cout << "Test Passed." << std::endl;
	}
	else
	{
		std::cout << "Test FAILED!" << std::endl;
	}	
}

//Check d, sigma, and delta for a given root node and check bc for all nodes. Useful for narrowing down where the problem might be.
void verify_all(float *bc_expected, float *bc_actual, int *d_expected, int *d_actual, int *sigma_expected, int *sigma_actual, float *delta_expected, float *delta_actual, csr_graph g)
{
	double error = 0;
	for(int i=0; i<g.n; i++)
	{
		error += pow((d_expected[i] - d_actual[i]),2);
	}
	error = error/(double)g.n;
	error = sqrt(error);

	std::cout << "RMS Error for d: " << error << std::endl;

	error = 0;
	for(int i=0; i<g.n; i++)
	{
		error += pow((sigma_expected[i] - sigma_actual[i]),2);
	}
	error = error/(double)g.n;
	error = sqrt(error);

	std::cout << "RMS Error for sigma: " << error << std::endl;

	error = 0;
	for(int i=0; i<g.n; i++)
	{
		error += pow((delta_expected[i] - delta_actual[i]),2);
	}
	error = error/(double)g.n;
	error = sqrt(error);

	std::cout << "RMS Error for delta: " << error << std::endl;
}

void choose_device(int &max_threads_per_block, int &number_of_SMs, int &choice)
{
	int count;
	checkCudaErrors(cudaGetDeviceCount(&count));
	std::cout << "Number of eligible devices: " << count << std::endl << std::endl;

	cudaDeviceProp prop;
	if((choice != -1) && (choice < count))
	{
		checkCudaErrors(cudaSetDevice(choice));
		checkCudaErrors(cudaGetDeviceProperties(&prop,choice));
	}
	else //Choose the best available device
	{
		int maxcc,bestdev;
		maxcc = 0;
		bestdev = 0;
		for(int i=0; i<count; i++)
		{
			checkCudaErrors(cudaGetDeviceProperties(&prop,i));
			if((prop.major + 0.1*prop.minor) > maxcc)
			{
				maxcc = prop.major + 0.1*prop.minor;
				bestdev = i;
			}
		}

		checkCudaErrors(cudaSetDevice(bestdev));
		checkCudaErrors(cudaGetDeviceProperties(&prop,bestdev));
	}

	std::cout << "Chosen Device: " << prop.name << std::endl;
	std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl << std::endl;
	std::cout << "Maximum Number of Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
	std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock << std::endl;
	if(prop.multiProcessorCount != 0)
	{
		std::cout << "Number of Streaming Multiprocessors: " << prop.multiProcessorCount << std::endl;
		number_of_SMs = prop.multiProcessorCount;
	}
	else
	{
		//Assume we're using Denali and hard code the number of SMs
		if(prop.minor == 1)
		{
			number_of_SMs = 7; //GTX 560
		}
		else
		{
			number_of_SMs = 14; //Tesla C2075
		}
	}
	std::cout << "Size of Global Memory: " << prop.totalGlobalMem << std::endl;
	//size_t heap_size = prop.totalGlobalMem/4;
	//checkCudaErrors(cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize));
	//checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size));
	//std::cout << "Size of the heap: " << heap_size << std::endl;

	std::cout << std::endl;
	max_threads_per_block = prop.maxThreadsPerBlock;
}

void print_devices()
{
	int count;
	checkCudaErrors(cudaGetDeviceCount(&count));
	for(int i=0; i<count; i++)
	{
		cudaDeviceProp prop;
		checkCudaErrors(cudaGetDeviceProperties(&prop,i));
		std::cout << "Device " << i << ": " << prop.name << std::endl;
		std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
	}
}

//Debugging routine to print intermediate data structures from both the CPU and GPU for direct comparison
void print_intermediates(int *d, int *d_gpu, int *sigma, int *sigma_gpu, float *delta, float *delta_gpu, float *bc_seq, float *bc_gpu, csr_graph g)
{
	std::cout << std::setprecision(9);

	if((d != NULL) && (d_gpu != NULL))
	{
		for(int i=0; i<g.n; i++)
		{
			std::cout << "d[" << i << "] = " << d[i] << std::endl;
		}
		for(int i=0; i<g.n; i++)
		{
			std::cout << "d_gpu[" << i << "] = " << d_gpu[i] << std::endl;
		}
	}

	if((sigma != NULL) && (sigma_gpu != NULL))
	{
		for(int i=0; i<g.n; i++)
		{
			std::cout << "sigma[" << i << "] = " << sigma[i] << std::endl;
		}
		for(int i=0; i<g.n; i++)
		{
			std::cout << "sigma_gpu[" << i << "] = " << sigma_gpu[i] << std::endl;
		}
	}

	if((delta != NULL) && (delta_gpu != NULL))
	{
		for(int i=0; i<g.n; i++)
		{
			std::cout << "delta[" << i << "] = " << delta[i] << std::endl;
		}
		for(int i=0; i<g.n; i++)
		{
			std::cout << "delta_gpu[" << i << "] = " << delta_gpu[i] << std::endl;
		}
	}

	if((bc_seq != NULL) && (bc_gpu != NULL))
	{
		for(int i=0; i<g.n; i++)
		{
			std::cout << "bc[" << i << "] = " << bc_seq[i] << std::endl;
		}
		for(int i=0; i<g.n; i++)
		{
			std::cout << "bc_gpu[" << i << "] = " << bc_gpu[i] << std::endl;
		}
	}
}

void remove_edges(csr_graph &g, std::set< std::pair<int,int> > &removals, program_options op)
{
	int removed = 0;
	while(removed < op.insertions)
	{
		int source = rand() % g.n;
		//Randomly choose a neighbor of the source and remove that edge
		int begin = g.R[source];
		int end = g.R[source+1];
		if((end - begin) == 0) //This vertex has no neighbors
		{
						continue;
		}
		int dest = g.C[begin + (rand() % (end - begin))];
		if(source == dest) //Can this actually happen?
		{
			std::cerr << "Self edge detected!" << std::endl;
			continue;
		}

		removals.insert(std::make_pair(source,dest));
		removals.insert(std::make_pair(dest,source));
		if(removals.size()/2 != (removed+1)) //Removed edges must be unique
		{
			continue;
		}

		removed++;
	}

	//Now remove this set of edges from the graph
	g.remove_edges(removals);
}

void *power_sample(void *period)
{
	checkNVMLErrors(nvmlInit());
	nvmlDevice_t nvml_dev;
	checkNVMLErrors(nvmlDeviceGetHandleByIndex(1,&nvml_dev)); //Hard coding the TeslaC0275 as device 1 to save some time. This is NOT portable.
	unsigned int power;
	unsigned int samples=0;
	float *avg_power = (float *) malloc(sizeof(float));
	*avg_power = 0;
	long samp_period = (long) period;
	//std::cout << "Sampling period: " << samp_period << " ms" << std::endl;

	while(*psample)
	{
		checkNVMLErrors(nvmlDeviceGetPowerUsage(nvml_dev,&power));
		samples++;
		*avg_power += power/(float)1000;
		usleep(samp_period*1000);
	}

	checkNVMLErrors(nvmlShutdown());

	*avg_power = *avg_power/(float)samples;
	//std::cout << "Average Power: " << avg_power << " W" << std::endl;
	//std::cout << "Number of samples: " << samples << std::endl;
	
	pthread_exit((void*)avg_power);
}
