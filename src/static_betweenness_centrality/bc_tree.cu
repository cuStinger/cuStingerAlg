#include "utils.hpp"
#include "update.hpp"
#include "cuStinger.hpp"

#include "algs.cuh"
#include "static_betweenness_centrality/bc_tree.cuh"

namespace cuStingerAlgs {

bcTree* createHostBcTree(length_t nv)
{
	bcTree *tree_h = new bcTree;
	tree_h->offsets = new float[nv];
	tree_h->nv = nv;
	return tree_h;
}

// user must call destroyHostBcTree before terminating program to free memory
// must provide number of vertices (nv) and pointer to host bcTree
bcTree* createDeviceBcTree(length_t nv, bcTree *tree_h)
{
	int size = sizeof(bcTree);
	size += 2 * nv * sizeof(vertexId_t);  // d and sigma
	size += nv * sizeof(float);  // delta

	char *starting_point = (char *) allocDeviceArray(1, size);
	bcTree *tree_d = (bcTree*) starting_point;

	// pointer arithmetic for d, sigma, delta pointers
	// these are actual memory locations on the device for the arrays
	char *d = starting_point + sizeof(bcTree);  // start where tree_d ends
	char *sigma = d + nv * sizeof(vertexId_t);  // start where d ends
	char *delta = sigma + nv * sizeof(vertexId_t);  // start where sigma ends

	tree_h->d = (vertexId_t *) d;
	tree_h->sigma = (vertexId_t *) sigma;
	tree_h->delta = (float *) delta;

	copyArrayHostToDevice(tree_h, tree_d, 1, sizeof(bcTree));

	return tree_d;
}

void destroyDeviceBcTree(bcTree* tree_d)
{
	freeDeviceArray(tree_d);
}

void destroyHostBcTree(bcTree* tree_h)
{
	delete[] tree_h->offsets;
	delete tree_h;
}

}  // namespace