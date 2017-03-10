#ifndef STATIC_BC_TREE
#define STATIC_BC_TREE

#include "algs.cuh"

namespace cuStingerAlgs {

typedef struct {
    length_t nv;
    vertexId_t root;
    vertexId_t currLevel;

    vertexId_t* d;  // depth of each vertex
    vertexId_t* sigma;
    float* delta;

    vertexQueue queue;

    // offsets storees ending position of each frontier in the queue.
    // Used during dependency accumulation. Host only
    float* offsets;
} bcTree;


typedef struct {
    bcTree *forest;
    length_t nv;
    length_t numRoots;
} bcForest;


bcTree* createHostBcTree(length_t nv);

// user must call destroyHostBcTree before terminating program to free memory
// must provide number of vertices (nv) and pointer to host bcTree
bcTree* createDeviceBcTree(length_t nv, bcTree *tree_h);

void destroyDeviceBcTree(bcTree* tree_d);

void destroyHostBcTree(bcTree* tree_h);

}  // namespace

#endif