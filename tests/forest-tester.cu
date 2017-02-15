/********************************************************
 * Forest
 ********************************************************/

#include "utils.hpp"
#include "update.hpp"
#include "cuStinger.hpp"

#include "algs.cuh"

#include <stdio.h>
#include "forest-tester.h"

using namespace cuStingerAlgs;


bcTree* createTreeHost(vertexId_t root, int64_t numVertices) {
	// allocate space for host bctree
	bcTree* newTree = (bcTree*) allocHostArray(1, sizeof(bcTree));
	newTree->NV = numVertices;
	newTree->root = root;

	newTree->d = (vertexId_t*) allocHostArray(numVertices, sizeof(vertexId_t));
	newTree->sigma = (int64_t*) allocHostArray(numVertices, sizeof(int64_t));
	newTree->delta = (float*) allocHostArray(numVertices, sizeof(float));
	
	return newTree;
}

bcTree* createTreeDevice(vertexId_t root, int64_t numVertices) {
	bcTree* newTree_h = createTreeHost(root, numVertices);
	bcTree* newTree_d = copyTreeHostToDevice(newTree_h);
	destroyTreeHost(newTree_h);
	return newTree_d;
}

// may remove -- ask Oded if needed
// bcTree* createTreeDevice(int64_t numVertices) {
// 	bcTree* newTree_h = createTreeHost(numVertices);

// 	bcV* vArr_h = newTree_h->vArr;
// 	bcV* vArr_d = (bcV*) allocDeviceArray(numVertices, sizeof(bcV));
// 	copyArrayHostToDevice(newTree_h->vArr, vArr_d, numVertices, sizeof(bcV));
// 	newTree_h->vArr = vArr_d;

// 	bcTree* newTree_d = (bcTree*) allocDeviceArray(1, sizeof(bcTree));

// 	copyArrayHostToDevice(newTree_h, newTree_d, 1, sizeof(bcTree));

// 	newTree_h->vArr = vArr_h;
// 	destroyTreeHost(newTree_h);

// 	return newTree_d;
// }

bcTree* copyTreeHostToDevice(bcTree* tree_h) {
	// holds onto pointers of host arrays
	vertexId_t* d_h = tree_h->d;
	int64_t* sigma_h = tree_h->sigma;
	float* delta_h = tree_h->delta;

	int64_t numVertices = tree_h->NV;

	// copies host arrays to device arrays
	vertexId_t* d_d = (vertexId_t*) allocDeviceArray(numVertices, sizeof(vertexId_t));
	int64_t* sigma_d = (int64_t*) allocDeviceArray(numVertices, sizeof(int64_t));
	float* delta_d = (float*) allocDeviceArray(numVertices, sizeof(float));

	copyArrayHostToDevice(tree_h->d, d_d, numVertices, sizeof(vertexId_t));
	copyArrayHostToDevice(tree_h->sigma, sigma_d, numVertices, sizeof(int64_t));
	copyArrayHostToDevice(tree_h->delta, delta_d, numVertices, sizeof(float));


	// sets host array pointers to device arrays
	tree_h->d = d_d;
	tree_h->sigma = sigma_d;
	tree_h->delta = delta_d;

	// copies host bctree to device bctree
	bcTree* newTree_d = (bcTree*) allocDeviceArray(1, sizeof(bcTree));
	copyArrayHostToDevice(tree_h, newTree_d, 1, sizeof(bcTree));

	// sets back pointers of host arrays to original
	tree_h->d = d_h;
	tree_h->sigma = sigma_h;
	tree_h->delta = delta_h;

	return newTree_d;
}

bcTree* copyTreeDeviceToHost(bcTree* tree_d) {
	// copies device bctree to host bctree
	bcTree* newTree_h = (bcTree*) allocHostArray(1, sizeof(bcTree));
	copyArrayDeviceToHost(tree_d, newTree_h, 1, sizeof(bcTree));

	// copies device arrays to host arrays
	vertexId_t* d_h = (vertexId_t*) allocHostArray(newTree_h->NV, sizeof(vertexId_t));
	int64_t* sigma_h = (int64_t*) allocHostArray(newTree_h->NV, sizeof(int64_t));
	float* delta_h = (float*) allocHostArray(newTree_h->NV, sizeof(float));
	
	copyArrayDeviceToHost(newTree_h->d, d_h, newTree_h->NV, sizeof(vertexId_t));
	copyArrayDeviceToHost(newTree_h->sigma, sigma_h, newTree_h->NV, sizeof(int64_t));
	copyArrayDeviceToHost(newTree_h->delta, delta_h, newTree_h->NV, sizeof(float));

	newTree_h->d = d_h;
	newTree_h->sigma = sigma_h;
	newTree_h->delta = delta_h;

	return newTree_h;
}

void destroyTreeHost(bcTree* tree_h) {
	freeHostArray(tree_h->d);
	freeHostArray(tree_h->sigma);
	freeHostArray(tree_h->delta);
    freeHostArray(tree_h);
}

void destroyTreeDevice(bcTree* tree_d) {
	// copies device bctree to host bctree
	bcTree* newTree_h = (bcTree*) allocHostArray(1, sizeof(bcTree));
	copyArrayDeviceToHost(tree_d, newTree_h, 1, sizeof(bcTree));

	// frees device arrays
	freeDeviceArray(newTree_h->d);
	freeDeviceArray(newTree_h->sigma);
	freeDeviceArray(newTree_h->delta);

	freeHostArray(newTree_h);
	freeDeviceArray(tree_d);
}

int main(int argc, char** argv) {
	// test
	int64_t numVertices = 3;
	vertexId_t root = 5;

	bcTree* tree_h = createTreeHost(root, numVertices);

	for (int i = 0; i < numVertices; i++) {
		tree_h->d[i] = i;
		tree_h->sigma[i] = 10 + i;
		tree_h->delta[i] = 100 + i;
	}

	bcTree* newTree_d = copyTreeHostToDevice(tree_h);

	bcTree* newTree_h = copyTreeDeviceToHost(newTree_d);

	printf("%ld\n", newTree_h->NV);
	printf("%d\n", newTree_h->root);

	printf("Printing d\n");
	for (int i = 0; i < numVertices; i++) {
		printf("%d\n", newTree_h->d[i]);
	}

	printf("Printing sigma\n");
	for (int i = 0; i < numVertices; i++) {
		printf("%ld\n", newTree_h->sigma[i]);
	}

	printf("Printing delta\n");
	for (int i = 0; i < numVertices; i++) {
		printf("%f\n", newTree_h->delta[i]);
	}
}