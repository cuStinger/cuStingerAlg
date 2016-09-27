#include "metis_to_csr.h"

bool is_number(const std::string& s)
{
    std::string::const_iterator it = s.begin();
    while (it != s.end() && std::isdigit(*it)) ++it;
    return !s.empty() && it == s.end();
}

void csr_graph::print_offset_array()
{
	std::cout << "R = [" << R[0];
	for(int i=1; i<=n; i++)
	{
		std::cout << ", " << R[i];
	}
	std::cout << "]" << std::endl;
}

void csr_graph::print_edge_array()
{
	std::cout << "C = [" << C[0];
	for(int i=1; i<(m)*2; i++)
	{
		std::cout << ", " << C[i];
	}
	std::cout << "]" << std::endl;
}

void csr_graph::print_from_array()
{
	std::cout << "F = [" << F[0];
	for(int i=1; i<(m)*2; i++)
	{
		std::cout << ", " << F[i];
	}
	std::cout << "]" << std::endl;
}

void csr_graph::print_adjacency_list()
{
	std::cout << "Edge lists for each vertex: " << std::endl;

	for(int i=0; i<n; i++)
	{
		int begin = R[i];
		int end = R[i+1];
		for(int j=begin; j<end; j++)
		{
			if(j==begin)
			{
				std::cout << i << " | " << C[j];
			}
			else
			{
				std::cout << ", " << C[j];
			}
		}
		if(begin == end) //Single, unconnected node
		{
			std::cout << i << " | ";
		}
		std::cout << std::endl;
	}
}

csr_graph parse_metis(char *file)
{
	csr_graph g;

	//Get n,m
	std::ifstream metis(file,std::ifstream::in);
	std::string line;
	bool firstline = true;
	int current_node = 0;
	int current_edge = 0;

	if(!metis.good())
	{
		std::cerr << "Error opening graph file." << std::endl;
		exit(-1);
	}

	while(std::getline(metis,line))
	{
		if(line[0] == '%')
		{
			continue;
		}

		std::vector<std::string> splitvec;
		boost::split(splitvec,line, boost::is_any_of(" \t"), boost::token_compress_on); //Now tokenize

		//If the last element is a space or tab itself, erase it
		if(!is_number(splitvec[splitvec.size()-1]))
		{
			splitvec.erase(splitvec.end()-1);
		}

		if(firstline)
		{
			g.n = stoi(splitvec[0]);
			g.m = stoi(splitvec[1]);
			if(splitvec.size() > 3)
			{
				std::cerr << "Error: Weighted graphs are not yet supported." << std::endl;
				exit(-2);
			}
			else if((splitvec.size() == 3) && (stoi(splitvec[2]) != 0))
			{
				std::cerr << "Error: Weighted graphs are not yet supported." << std::endl;
				exit(-2);
			}
			firstline = false;
			g.weighted = false;
			g.directed = false;
			g.R = new int[g.n+1];
			g.C = new int[2*g.m];
			g.F = new int[2*g.m];
			g.R[0] = 0;
			current_node++;
		}
		else
		{
			//Count the number of edges that this vertex has and add that to the most recent value in R
			g.R[current_node] = splitvec.size()+g.R[current_node-1];
			for(unsigned i=0; i<splitvec.size(); i++)
			{
				/*std::cout << "splitvec[i].compare(\" \") " << splitvec[i].compare(" ") << std::endl;
				std::cout << "splitvec[i].compare(0,1,\" \") " << splitvec[i].compare(0,1," ") << std::endl;
				std::cout << "splitvec[i].size() " << splitvec[i].size() << std::endl;
				std::cout << "12345" << splitvec[i] << "54321" << std::endl;*/
				//coPapersDBLP uses a space to mark the beginning of each line, so we'll account for that here
				if(!is_number(splitvec[i]))
				{
					/*for(unsigned j=0;j<splitvec.size(); j++)
					{
						std::cerr << "splitvec[" << j << "] = " << splitvec[j] << std::endl;
					}*/
					//Need to adjust g.R
					g.R[current_node]--;
					continue;
				}
				//Store the neighbors in C
				//METIS graphs are indexed by one, but for our convenience we represent them as if
				//they were zero-indexed
				g.C[current_edge] = stoi(splitvec[i])-1; 
				g.F[current_edge] = current_node-1;
				current_edge++;
			}
			current_node++;
		}
	}

	return g;
}

void csr_graph::remove_edges(std::set< std::pair<int,int> > removals)
{
	//Throw arrays into temporary vectors
	std::vector<int> tempC(C,C+(2*m));
	delete[] C;
	std::vector<int> tempF(F,F+(2*m));
	delete[] F;

	int removed_edges = removals.size()/2;

	while(removals.size())
	{
		int source = removals.begin()->first;
		int dest = removals.begin()->second;
		removals.erase(removals.begin());

		int index_removed = -1;
		for(unsigned i=0; i<tempC.size(); i++)
		{
			if((tempC[i] == source) && (tempF[i] == dest))
			{
				tempC.erase(tempC.begin()+i);
				tempF.erase(tempF.begin()+i);
				index_removed = i;
				break;
			}
		}

		if(index_removed == -1)
		{
			std::cerr << "Error removing edges - couldn't find edge (" << source << "," << dest << ")" << std::endl;
			exit(-1);
		}

		for(int i=0; i<n+1; i++)
		{
			if(R[i] > index_removed)
			{
				R[i]--;
			}
		}
	}

	m = m-removed_edges; //Note that n doesn't change - only edges are removed

	C = new int[2*m];
	F = new int [2*m];
	for(int i=0; i<2*m; i++)
	{
		C[i] = tempC[i];
		F[i] = tempF[i];
	}
}

//This function takes source and destination vertices and inserts edges (src,dst) and (dst,src) into the graph.
void csr_graph::reinsert_edge(int src, int dst)
{
	//Throw arrays into temporary vectors
	std::vector<int> tempC(C,C+(2*m));
	delete[] C;
	std::vector<int> tempF(F,F+(2*m));
	delete[] F;

	//Insert (src,dst)
	int end = R[src+1];
	tempC.insert(tempC.begin()+end,dst);
	tempF.insert(tempF.begin()+end,src);
	for(int i=src+1; i<n+1; i++)
	{
		R[i]++;
	}
	//Insert (dst,src)
	end = R[dst+1];
	tempC.insert(tempC.begin()+end,src);
	tempF.insert(tempF.begin()+end,dst);
	for(int i=dst+1; i<n+1; i++)
	{
		R[i]++;
	}

	m = m+1; //Update the number of edges in the graph. Again, the number of vertices does not change
	
	C = new int[2*m];
	F = new int[2*m];
	for(int i=0; i<2*m; i++)
	{
		C[i] = tempC[i];
		F[i] = tempF[i];
	}
}
