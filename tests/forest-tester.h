/********************************************************
 * forest.h
 ********************************************************/

typedef struct {
    int64_t NV;
    vertexId_t root;
    vertexId_t* d;
    int64_t* sigma;
    float* delta;
} bcTree;

typedef bcTree* bcTreePtr;

typedef struct {
    bcTreePtr* forest;
    int64_t NV;
} bcForest;

typedef bcForest* bcForestPtr;



bcTree* createTreeHost(int64_t numVertices);
bcTree* createTreeDevice(int64_t numVertices);
void destroyTreeHost(bcTree* tree_h);
void destroyTreeDevice(bcTree* tree_d);
bcTree* copyTreeHostToDevice(bcTree* tree_h);
