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
#include "static_betweenness_centrality/bc.cuh"

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
				options.verbose = true;
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

int main(const int argc, char **argv)
{
	parse_arguments(argc, argv);

	int device = 0;
    cudaSetDevice(device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);

	bool isDimacs = false;
	bool isSNAP = false;
	bool isRmat = false;
	length_t nv, ne, *off;
	vertexId_t *adj;

	string filename(options.infile);

	isDimacs = filename.find(".graph")==string::npos?false:true;
	isSNAP   = filename.find(".txt")==string::npos?false:true;
	isRmat 	 = filename.find("kron")==string::npos?false:true;

	if(isDimacs){
	    readGraphDIMACS(options.infile, &off, &adj, &nv, &ne, isRmat);
	} else if(isSNAP){
	    readGraphSNAP(options.infile, &off, &adj, &nv, &ne);
	}
	else {
		cout << "Unknown graph type" << endl;
		exit(0);
	}

	// if not in approx mode, set numRoots to number of vertices
	if (!options.approx) {
		options.numRoots = nv;
	}

	if (options.verbose) {
		cout << "Vertices: " << nv << endl;
		cout << "Edges: " << ne << endl;
		cout << "Num Roots: " << options.numRoots << endl;
	}

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

	// Must free this memory afterwards
	float *bc = new float[nv];
	for (int k = 0; k < nv; k++)
	{
		bc[k] = 0;
	}

	StaticBC sbc(options.numRoots, bc);
	sbc.Init(custing);
	sbc.Reset();

	cudaEvent_t ce_start,ce_stop;
	start_clock(ce_start, ce_stop);

	sbc.Run(custing);

	float totalTime = end_clock(ce_start, ce_stop);
	cout << "Total time for Betweenness Centrality Computation: " << totalTime << endl;

	sbc.Reset();
	sbc.Release();

	if (options.verbose) {
		cout << "RESULTS: " << endl;

		for (int k = 0; k < nv; k++)
		{
			cout << "[ " << k  << " ]: " << bc[k] << endl;
		}
	}

	// Free memory
	custing.freecuStinger();

	free(off);
	free(adj);

	delete[] bc;

    return 0;
}
