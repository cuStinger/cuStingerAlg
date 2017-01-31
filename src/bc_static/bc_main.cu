#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>

#include <math.h>


#include "utils.hpp"
#include "update.hpp"
#include "cuStinger.hpp"

#include <getopt.h>

#include "algs.cuh"
#include "bc_static/bc.cuh"

using namespace cuStingerAlgs;

#define CUDA(call, ...) do {                        \
        cudaError_t _e = (call);                    \
        if (_e == cudaSuccess) break;               \
        fprintf(stdout,                             \
                "CUDA runtime error: %s (%d)\n",    \
                cudaGetErrorString(_e), _e);        \
        return -1;                                  \
    } while (0)

typedef void (*cus_kernel_call)(cuStinger& custing, void* func_meta_data);

void printUsageInfo(char **argv)
{
	std::cout << "Usage: " << argv[0];
	std::cout << " -i <graph input file> [optional arguments]";
	std::cout << std::endl << std::endl;

	std::cout << "Options: " << std::endl;

	std::cout << "-v                      \tVerbose. Prints debug output";
	std::cout << " to stdout" << std::endl;

	std::cout << "-k <# of src nodes>     \tApproximate BC using a given";
	std::cout << " number of random source nodes" << std::endl;

	std::cout << "-t <# of nodes to add>  \tStreaming BC" << std::endl;
	std::cout << std::endl;
}

typedef struct		
{		
	bool streaming;		
	bool approx;		
	// number of vertices used. If approx, set here via CLI.		
	// otherwise defaults to all vertices		
	int numRoots;		
	bool verbose;  // print debug info		
	int edgesToAdd;  // edges to add		
	char *infile;		
} program_options;		


program_options options;


void parse_arguments(int argc, char **argv)
{
	int c;
	static struct option long_options[] =
	{
		{"help", no_argument, 0, 'h'},
		{"infile", required_argument, 0, 'i'},
		{"source_nodes", required_argument, 0, 'k'},
		{"stream", required_argument, 0, 't'},  // arg is # of edges to insert
		{"verbose", no_argument, 0,'v'},
		{0,0,0,0} // Terminate with null
	};

	int option_index = 0;

	while((c = getopt_long(argc, argv, "c:de::fg:hi:k:mn::opst:v",
		long_options, &option_index)) != -1)
	{
		switch(c)
		{
			case 'i':
				options.infile = optarg;
			break;

			case 'k':
				options.numRoots = atoi(optarg);	
				options.approx = true;	
			break;

			case 't':
				options.edgesToAdd = atoi(optarg);
				options.streaming = true;
			break;

			case 'v':
				options.verbose = atoi(optarg);
			break;

			case 'h':
				printUsageInfo(argv);
				exit(0);
			break;

			default: //Fatal error
				std::cerr << "Internal error parsing arguments." << std::endl;
				printUsageInfo(argv);
				exit(-1);
		}
	}

	//Handle required command line options here
	if(options.infile == NULL)
	{
		std::cerr << "Command line error: Graph input file is required.";
		std::cerr << " Use the -i switch." << std::endl;
		printUsageInfo(argv);
		exit(-1);
	}
	if(options.approx && (options.numRoots == -1 || options.numRoots < 1))
	{
		std::cerr << "Command line error: Approximation requested but no";
		std::cerr << " number of source nodes given. Defaulting to 128.";
		std::cerr << std::endl;
		options.numRoots = 128;
	}
	if(options.streaming && (options.edgesToAdd == -1))
	{
		std::cerr << "Command line error: Streaming requested but no";
		std::cerr << " number of insertions given. Defaulting to 5.";
		std::cerr << std::endl;
		options.edgesToAdd = 5;
	}
}

void generateEdgeUpdates(length_t nv, length_t numEdges, vertexId_t* edgeSrc,
	vertexId_t* edgeDst)
{
		std::cout << "Edge Updates: " << std::endl;
	for(int32_t e=0; e<numEdges; e++) {
		edgeSrc[e] = rand()%nv;
		edgeDst[e] = rand()%nv;

		std::cout << "Edge: (" << edgeSrc[e] << ", " << edgeDst[e] << ")";
		std::cout << std::endl;
	}
}

// TODO: Implement later
// For RMAT edges
// typedef struct dxor128_env {
//   unsigned x,y,z,w;
// } dxor128_env_t;

// void rmat_edge (int64_t * iout, int64_t * jout, int SCALE, double A, double B, double C, double D, dxor128_env_t * env);

// void generateEdgeUpdatesRMAT(length_t nv, length_t numEdges, vertexId_t* edgeSrc, vertexId_t* edgeDst,double A, double B, double C, double D, dxor128_env_t * env){
// 	int64_t src,dst;
// 	int scale = (int)log2(double(nv));
// 	for(int32_t e=0; e<numEdges; e++){
// 		rmat_edge(&src,&dst,scale, A,B,C,D,env);
// 		edgeSrc[e] = src;
// 		edgeDst[e] = dst;

// 		std::cout << "Edge: (" << edgeSrc[e] << ", " << edgeDst[e] << ")";
// 		std::cout << std::endl;
// 	}
// }

void printcuStingerUtility(cuStinger custing)
{
	length_t used,allocated;

	used = custing.getNumberEdgesUsed();
	allocated = custing.getNumberEdgesAllocated();
	std::cout << "Used: " << used << "\tAllocated: " << allocated;
	std::cout << "\tRatio Used-To-Allocated: " << (float)used/(float)allocated << std::endl;
}


// typedef void (*cus_kernel_call)(cuStinger& custing, void* func_meta_data);

// void bc_static(cuStinger& custing, void* func_meta_data);


int main(const int argc, char **argv)
{
	parse_arguments(argc, argv);

	int device = 0;
    cudaSetDevice(device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);

	int max_threads_per_block = prop.maxThreadsPerBlock;
	int number_of_SMs = prop.multiProcessorCount;
 
    length_t nv, ne,*off;
    vertexId_t *adj;

	bool isDimacs = false;
	bool isSNAP = false;
	bool isRmat = false;
	string filename(options.infile);
	
	isDimacs = filename.find(".graph")==std::string::npos?false:true;
	isSNAP   = filename.find(".txt")==std::string::npos?false:true;
	isRmat 	 = filename.find("kron")==std::string::npos?false:true;

	if(isDimacs){
	    readGraphDIMACS(options.infile, &off, &adj, &nv, &ne, isRmat);
	}
	// TODO: FIX THIS
	// else if(isSNAP){
	//     readGraphSNAP(options.infile, &off, &adj, &nv, &ne);
	// }
	else{ 
		cout << "Unknown graph type" << endl;
		exit(0);
	}

	std::cout << "Vertices: " << nv << endl;
	std::cout << "Edges: " << ne << endl;

	// if not in approx mode, set numRoots to number of vertices
	if (!options.approx) {
		options.numRoots = nv;
	}

	std::cout << "Num Roots: " << options.numRoots << std::endl;

	cudaEvent_t ce_start,ce_stop;
	cuStinger custing(defaultInitAllocater,defaultUpdateAllocater);

	cuStingerInitConfig cuInit;
	cuInit.initState = eInitStateCSR;
	cuInit.maxNV = nv + 1;
	cuInit.useVWeight = false;
	cuInit.isSemantic = false;  // Use edge types and vertex types
	cuInit.useEWeight = false;
	// CSR data
	cuInit.csrNV  = nv;
	cuInit.csrNE = ne;
	cuInit.csrOff  = off;
	cuInit.csrAdj  = adj;
	cuInit.csrVW  = NULL;
	cuInit.csrEW = NULL;

	custing.initializeCuStinger(cuInit);

	// First, we'll add some random edges
	int numBatchEdges = options.edgesToAdd;

	std::cout << "Num edges: " << numBatchEdges << std::endl;

	BatchUpdateData bud(numBatchEdges, true);

	if (!isRmat) {
		generateEdgeUpdates(nv, numBatchEdges, bud.getSrc(),bud.getDst());
	}
	BatchUpdate bu(bud);

	// Print stats before insertions
	std::cout << "Before Insertions:" << std::endl;
	printcuStingerUtility(custing);

	// Insert these edges
	length_t allocs;
	custing.edgeInsertions(bu, allocs);
	std::cout << "After Insertions:" << std::endl;
	printcuStingerUtility(custing);


	// Now, delete them
	custing.edgeDeletions(bu);
	std::cout << "After Deletions:" << std::endl;
	printcuStingerUtility(custing);

	// Print custinger utility
	// printcuStingerUtility(custing);

	// cus_kernel_call call_kernel = bc_static;
	// call_kernel(custing, NULL);

	// call_kernel = connectComponentsMain;
	// call_kernel(custing,NULL);

	// call_kernel = connectComponentsMainLocal;
	// call_kernel(custing,NULL);

	// call_kernel = oneMoreMain;
	// call_kernel(custing,NULL);

	float *bc = new float[nv];

	StaticBC sbc;
	sbc.Init(custing);
	sbc.Reset();
	vertexId_t root = rand() % nv;
	sbc.setInputParameters(root);

	start_clock(ce_start, ce_stop);
	sbc.Run(custing);
	float totalTime = end_clock(ce_start, ce_stop);

	cout << "The number of levels          : " << sbc.getLevels() << endl;
	cout << "The number of elements found  : " << sbc.getElementsFound() << endl;
	cout << "Total time for connected-BFS : " << totalTime << endl; 

	sbc.Release();

	std::cout << "OUTCOME: " << std::endl;

	for (int k = 0; k < custing.nv; k++)
	{
		std::cout << "[ " << k  << " ]: " << bc[k] << std::endl;
	}

	// Free memory
	custing.freecuStinger();

	delete[] bc;

	free(off);
	free(adj);
    return 0;
}
