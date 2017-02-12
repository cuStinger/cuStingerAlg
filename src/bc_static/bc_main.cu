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
using namespace std;

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
	cout << "Usage: " << argv[0];
	cout << " -i <graph input file> [optional arguments]";
	cout << endl << endl;

	cout << "Options: " << endl;

	cout << "-v                      \tVerbose. Prints debug output";
	cout << " to stdout" << endl;

	cout << "-k <# of src nodes>     \tApproximate BC using a given";
	cout << " number of random source nodes" << endl;

	cout << "-t <# of nodes to add>  \tStreaming BC" << endl;
	cout << endl;
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
				cerr << "Internal error parsing arguments." << endl;
				printUsageInfo(argv);
				exit(-1);
		}
	}

	//Handle required command line options here
	if(options.infile == NULL)
	{
		cerr << "Command line error: Graph input file is required.";
		cerr << " Use the -i switch." << endl;
		printUsageInfo(argv);
		exit(-1);
	}
	if(options.approx && (options.numRoots == -1 || options.numRoots < 1))
	{
		cerr << "Command line error: Approximation requested but no";
		cerr << " number of source nodes given. Defaulting to 128.";
		cerr << endl;
		options.numRoots = 128;
	}
	if(options.streaming && (options.edgesToAdd == -1))
	{
		cerr << "Command line error: Streaming requested but no";
		cerr << " number of insertions given. Defaulting to 5.";
		cerr << endl;
		options.edgesToAdd = 5;
	}
}

void generateEdgeUpdates(length_t nv, length_t numEdges, vertexId_t* edgeSrc,
	vertexId_t* edgeDst)
{
		cout << "Edge Updates: " << endl;
	for(int32_t e=0; e<numEdges; e++) {
		edgeSrc[e] = rand()%nv;
		edgeDst[e] = rand()%nv;

		cout << "Edge: (" << edgeSrc[e] << ", " << edgeDst[e] << ")";
		cout << endl;
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

// 		cout << "Edge: (" << edgeSrc[e] << ", " << edgeDst[e] << ")";
// 		cout << endl;
// 	}
// }

void printcuStingerUtility(cuStinger custing)
{
	length_t used,allocated;

	used = custing.getNumberEdgesUsed();
	allocated = custing.getNumberEdgesAllocated();
	cout << "Used: " << used << "\tAllocated: " << allocated;
	cout << "\tRatio Used-To-Allocated: " << (float)used/(float)allocated << endl;
}


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

	isDimacs = filename.find(".graph")==string::npos?false:true;
	isSNAP   = filename.find(".txt")==string::npos?false:true;
	isRmat 	 = filename.find("kron")==string::npos?false:true;

	if(isDimacs){
	    readGraphDIMACS(options.infile, &off, &adj, &nv, &ne, isRmat);
	}
	// TODO: FIX THIS
	// else if(isSNAP){
	//     readGraphSNAP(options.infile, &off, &adj, &nv, &ne);
	// }
	else {
		cout << "Unknown graph type" << endl;
		exit(0);
	}

	cout << "Vertices: " << nv << endl;
	cout << "Edges: " << ne << endl;

	// if not in approx mode, set numRoots to number of vertices
	if (!options.approx) {
		options.numRoots = nv;
	}

	cout << "Num Roots: " << options.numRoots << endl;

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

	cout << "Num edges: " << numBatchEdges << endl;

	BatchUpdateData bud(numBatchEdges, true);

	if (!isRmat) {
		generateEdgeUpdates(nv, numBatchEdges, bud.getSrc(),bud.getDst());
	}
	BatchUpdate bu(bud);

	// Print stats before insertions
	cout << "Before Insertions:" << endl;
	printcuStingerUtility(custing);

	// Insert these edges
	length_t allocs;
	custing.edgeInsertions(bu, allocs);
	cout << "After Insertions:" << endl;
	printcuStingerUtility(custing);


	// Now, delete them
	custing.edgeDeletions(bu);
	cout << "After Deletions:" << endl;
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
	for (int k = 0; k < nv; k++)
	{
		bc[k] = 0;
	}

	vertexId_t root;
	int rootsVisited = 0;

	StaticBC sbc;
	sbc.Init(custing);
	sbc.Reset();

	vertexId_t* level = new vertexId_t[nv];
	long *sigma = new long[nv];
	float *delta = new float[nv];

	while (rootsVisited < options.numRoots)
	{
		printf("Iteration: %d of %d.\n", rootsVisited + 1, options.numRoots);

		// Get a rood node
		if (options.approx)
		{
			root = rand() % nv;
		} else
		{
			root = rootsVisited;
		}
		rootsVisited++;

		// Now, set the root and run
		sbc.setInputParameters(root);

		start_clock(ce_start, ce_stop);
		sbc.Run(custing);
		float totalTime = end_clock(ce_start, ce_stop);

		cout << "The number of levels          : " << sbc.getLevels() << endl;
		cout << "The number of elements found  : " << sbc.getElementsFound() << endl;
		cout << "Total time for connected-BFS : " << totalTime << endl;


		// Dependency accumulation
		// Walk back from the queue in reverse
		vertexQueue queue = sbc.hostBcStaticData.queue;
		queue.setQueueCurr(0);  // 0 is the position of the
		vertexId_t *start = queue.getQueue();
		vertexId_t *end = queue.getQueue() + queue.getQueueEnd() - 1;


		// Update host copies of level, sigma, delta
		copyArrayDeviceToHost(sbc.hostBcStaticData.level, level, nv, sizeof(vertexId_t));
		copyArrayDeviceToHost(sbc.hostBcStaticData.sigma, sigma, nv, sizeof(long));
		copyArrayDeviceToHost(sbc.hostBcStaticData.delta, delta, nv, sizeof(float));

		// vertexId_t* level = sbc.hostBcStaticData.level;
		// long *sigma = sbc.hostBcStaticData.sigma;
		// float *delta = sbc.hostBcStaticData.delta;

		printf("Begin Dep accumulation\n");

		// Keep iterating backwards in the queue
		while (end >= start)
		{
			// Look at adjacencies for this vertex at end
			vertexId_t w = *end;
			printf("Looking at all neighbors of vertex %d\n", w);
			// length_t numNeighbors = (custing.getHostVertexData()->used)[w];
			// if (numNeighbors > 0)
			// {
			// 	// Get adjacency list
			// 	cuStinger::cusEdgeData *adj = (custing.getHostVertexData()->adj)[w];
			// 	for(int k = 0; k < numNeighbors; k++)
			// 	{
			// 		// neighbord v of w from the adjacency list
			// 		vertexId_t v = adj->dst[k];
			// 		// if depth is less than depth of w
			// 		if (level[v] == level[w] + 1)
			// 		{
			// 			printf("{} is a neighbor of {} at depth +1\n", v, w);
			// 			delta[v] += (delta[v] / delta[w]) * (1 + delta[w]);
			// 		}
			// 	}
			// }

			// Now, put values into bc[]
			if (w != root)
			{
				bc[w] += delta[w];
			}

			end--;
		}

		// Now, reset the queue
		printf("Done with iteration. Reset queue\n");
		sbc.Reset();
	}

	// Release only once all iterations are done.
	sbc.Release();

	cout << "OUTCOME: " << endl;

	for (int k = 0; k < nv; k++)
	{
		cout << "[ " << k  << " ]: " << bc[k] << endl;
	}

	// Free memory
	custing.freecuStinger();

	delete[] bc;
	delete[] level;
	delete[] sigma;
	delete[] delta;

	free(off);
	free(adj);

    return 0;
}
