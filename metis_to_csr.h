#ifndef METIS_TO_CSR
#define METIS_TO_CSR

#include <iostream>
#include <string>
#include <set>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <cstdlib>

class csr_graph
{
public:
	void print_offset_array();
	void print_edge_array();
	void print_from_array();
	void print_adjacency_list();
	void remove_edges(std::set< std::pair<int,int> >);
	void reinsert_edge(int src, int dst);

	int *C; //Array of edges
	int *R; //Array of offsets
	int *F; //Array of where edges originate from (used for edge-based parallelism)
	int n; //Number of vertices
	int m; //Number of edges
	bool weighted; //Is the graph weighted or unweighted?
	bool directed; //Is the graph directed or undirected?
};

csr_graph parse_metis(char *file);

#endif
