#include "sequential.h"

float* bc(csr_graph g, bool print_results, int *d_check, int *sigma_check, float *delta_check)
{
	float *bc = new float[g.n];
	for(int i=0; i<g.n; i++)
	{
		bc[i] = 0;
	}

	for(int i=0; i<g.n; i++)
	{
		std::stack<int> S;
		std::list<int> *P = new std::list<int>[g.n]; //Array of linked lists
		int *sigma = new int[g.n];
		for(int j=0; j<g.n; j++)
		{
			sigma[j] = 0;
		}
		sigma[i] = 1;
		int *d = new int[g.n];
		for(int j=0; j<g.n; j++)
		{
			d[j] = INT_MAX;
		}
		d[i] = 0;
		std::queue<int> Q;
		Q.push(i);

		while(!Q.empty())
		{
			int v = Q.front();
			Q.pop();
			S.push(v);
			int begin = g.R[v];
			int end = g.R[v+1];
			for(int j=begin; j<end; j++) //For each neighbor of v...
			{		
				int w = g.C[j];
				if(d[w] < 0)
				{
					Q.push(w);
					d[w] = d[v] + 1;
				}
				if(d[w] == (d[v] + 1))
				{
					sigma[w] = sigma[w] + sigma[v];
					P[w].push_back(v);
				} 
			}
		}

		float *delta = new float[g.n];
		for(int j=0; j<g.n; j++)
		{
			delta[j] = 0;
		}

		while(!S.empty())
		{
			int w = S.top();
			S.pop();
			for(std::list<int>::iterator j=P[w].begin(), e=P[w].end(); j!=e; ++j)
			{
				int v = *j;
				delta[v] = delta[v] + (sigma[v]/(float)sigma[w])*(1 + delta[w]);
			}
			if(w != i)
			{
				bc[w] = bc[w] + delta[w];
			}
		}

		if(i==3)
		{
			/*for(int j=0; j<g.n; j++)
			{
				std::cout << "d[" << i << "][" << j << "] = " << d[j] << std::endl;
			}
			for(int j=0; j<g.n; j++)
			{
				std::cout << "sigma[" << i << "][" << j << "] = " << sigma[j] << std::endl;	
			}
			for(int j=0; j<g.n; j++)
			{
				std::cout << "delta[" << i << "][" << j << "] = " << delta[j] << std::endl;
			}*/
			for(int j=0; j<g.n; j++)
			{
				d_check[j] = d[j];
				sigma_check[j] = sigma[j];
				delta_check[j] = delta[j];
			}
		}
	
		delete[] P;
		delete[] sigma;
		delete[] d;
		delete[] delta;
	}

	//o
	//If the graph is unweighted, divide the scores by 2 to not double count edges.
	//For now, I am NOT normalizing the results, but doing so is a trivial change.
	for(int i=0; i<g.n; i++)
	{
		bc[i] = bc[i]/(float)2; //Divide by 2 since the graph is unweighted
		bc[i] = bc[i]/(float)((g.n-1)*(g.n-2)); //Now normalize
	}

	if(print_results)
	{
		std::ofstream ofs("results.csv",std::ofstream::out);
		for(int i=0; i<g.n; i++)
		{
			if(i == g.n-1)
			{
				ofs << bc[i] << std::endl;
			}
			else
			{
				ofs << bc[i] << ",";
			}
		}	
	}

	return bc;
}

float* bc_no_parents(csr_graph g, bool print_results, bool streaming, std::vector< std::vector<int> > &d_old, std::vector< std::vector<unsigned long long> > &sigma_old, std::vector< std::vector<float> > &delta_old, int start, int end)
{
	float *bc = new float[g.n];
	for(int i=0; i<g.n; i++)
	{
		bc[i] = 0;
	}

	if(streaming)
	{
		//Need a copy of BFS depths for each source node - n^2 space in the exact case
		d_old.resize(g.n);
		sigma_old.resize(g.n);
		delta_old.resize(g.n);
		for(int i=0; i<g.n; i++)
		{
			d_old[i].resize(g.n);
			sigma_old[i].resize(g.n);
			delta_old[i].resize(g.n);
		}
	}

	for(int i=start; i<end; i++)
	{
		std::stack<int> S;
		unsigned long long *sigma = new unsigned long long[g.n];
		for(int j=0; j<g.n; j++)
		{
			sigma[j] = 0;
		}
		sigma[i] = 1;
		int *d = new int[g.n];
		for(int j=0; j<g.n; j++)
		{
			d[j] = INT_MAX;
		}
		d[i] = 0;
		std::queue<int> Q;
		Q.push(i);

		while(!Q.empty())
		{
			int v = Q.front();
			Q.pop();
			S.push(v);
			int begin = g.R[v];
			int end = g.R[v+1];
			for(int j=begin; j<end; j++) //For each neighbor of v...
			{		
				int w = g.C[j];
				if(d[w] == INT_MAX)
				{
					Q.push(w);
					d[w] = d[v] + 1;
				}
				if(d[w] == (d[v] + 1))
				{
					sigma[w] = sigma[w] + sigma[v];
				} 
			}
		}

		float *delta = new float[g.n];
		for(int j=0; j<g.n; j++)
		{
			delta[j] = 0;
		}

		while(!S.empty())
		{
			int w = S.top();
			S.pop();

			//Iterate through predecessors of w
			for(int j=g.R[w]; j<g.R[w+1]; j++) //For each edge of w...
			{
				int v = g.C[j];
				if(d[w] == (d[v] + 1)) //If w is a predecessor of v
				{
					delta[v] = delta[v] + (sigma[v]/(float)sigma[w])*(1 + delta[w]);
				}
			}
			if(w != i)
			{
				bc[w] = bc[w] + delta[w];
			}
		}

		if(streaming)
		{
			d_old[i].assign(d,d+g.n);
			sigma_old[i].assign(sigma,sigma+g.n);
			delta_old[i].assign(delta,delta+g.n);
		}

		delete[] sigma;
		delete[] d;
		delete[] delta;
	}

	//If the graph is unweighted, divide the scores by 2 to not double count edges.
	//If we're not streaming, wait until the end to normalize
	if(!streaming)
	{
		for(int i=0; i<g.n; i++)
		{
			bc[i] = bc[i]/(float)2;
			bc[i] = bc[i]/(float)((g.n-1)*(g.n-2)); //Now normalize
		}
	}

	if(print_results)
	{
		//TODO: Let this accept a file for output
		//std::ofstream ofs("results.csv",std::ofstream::out);
		for(int i=0; i<g.n; i++)
		{
			if(i == g.n-1)
			{
				std::cout << bc[i] << std::endl;
			}
			else
			{
				std::cout << bc[i] << ",";
			}
		}	
	}

	return bc;
}

float* bc_no_parents_approx(csr_graph g, int k, int *source_nodes, bool print_results, bool streaming, std::vector< std::vector<int> > &d_old, std::vector< std::vector<unsigned long long> > &sigma_old, std::vector< std::vector<float> > &delta_old, int start, int end)
{
	float *bc = new float[g.n];
	for(int i=0; i<g.n; i++)
	{
		bc[i] = 0;
	}

	if(streaming)
	{
		//Need a copy of BFS depths for each source node - O(kn) space in the approximate case
		d_old.resize(k);
		sigma_old.resize(k);
		delta_old.resize(k);
		for(int i=0; i<k; i++)
		{
			d_old[i].resize(g.n);
			sigma_old[i].resize(g.n);
			delta_old[i].resize(g.n);
		}
	}

	for(int m=start; m<end; m++)
	{
		int i = source_nodes[m];
		std::stack<int> S;
		unsigned long long *sigma = new unsigned long long[g.n];
		for(int j=0; j<g.n; j++)
		{
			sigma[j] = 0;
		}
		sigma[i] = 1;
		int *d = new int[g.n];
		for(int j=0; j<g.n; j++)
		{
			d[j] = INT_MAX;
		}
		d[i] = 0;
		std::queue<int> Q;
		Q.push(i);

		while(!Q.empty())
		{
			int v = Q.front();
			Q.pop();
			S.push(v);
			int begin = g.R[v];
			int end = g.R[v+1];
			for(int j=begin; j<end; j++) //For each neighbor of v...
			{		
				int w = g.C[j];
				if(d[w] == INT_MAX)
				{
					Q.push(w);
					d[w] = d[v] + 1;
				}
				if(d[w] == (d[v] + 1))
				{
					sigma[w] = sigma[w] + sigma[v];
				} 
			}
		}

		float *delta = new float[g.n];
		for(int j=0; j<g.n; j++)
		{
			delta[j] = 0;
		}

		while(!S.empty())
		{
			int w = S.top();
			S.pop();

			//Iterate through predecessors of w
			for(int j=g.R[w]; j<g.R[w+1]; j++) //For each edge of w...
			{
				int v = g.C[j];
				if(d[w] == (d[v] + 1)) //If w is a predecessor of v
				{
					delta[v] = delta[v] + (sigma[v]/(float)sigma[w])*(1 + delta[w]);
				}
			}
			if(w != i)
			{
				bc[w] = bc[w] + delta[w];
			}
		}

		if(streaming) //Note: preserving order of root nodes is mandatory!
		{
			d_old[m].assign(d,d+g.n);
			sigma_old[m].assign(sigma,sigma+g.n);
			delta_old[m].assign(delta,delta+g.n);
		}

		delete[] sigma;
		delete[] d;
		delete[] delta;
	}

	//If the graph is unweighted, divide the scores by 2 to not double count edges.
	if(!streaming) //If we're streaming, wait until the end to normalize
	{
		for(int i=0; i<g.n; i++)
		{
			bc[i] = bc[i]/(float)2;
			bc[i] = bc[i]/(float)((g.n-1)*(g.n-2)); //Now normalize
		}
	}

	if(print_results)
	{
		//TODO: Let this accept a file for output
		//std::ofstream ofs("results.csv",std::ofstream::out);
		for(int i=0; i<g.n; i++)
		{
			if(i == g.n-1)
			{
				std::cout << bc[i] << std::endl;
			}
			else
			{
				std::cout << bc[i] << ",";
			}
		}	
	}

	return bc;
}

void bc_update_edge(csr_graph g, bool approx, int k, int *source_nodes, int src, int dst, std::vector< std::vector<int> > &d_old, std::vector< std::vector<unsigned long long> > &sigma_old, std::vector< std::vector<float> > &delta_old, float *bc, std::vector<unsigned long long> &case_stats, std::vector< std::vector<unsigned int> > &nodes_touched, bool debug)
{
	if(approx)
	{
		for(int j=0; j<k; j++)
		{
			int i = source_nodes[j];
			
			//Figure out which insertion case we're dealing with
			int src_level = d_old[j][src];
			int dst_level = d_old[j][dst];

			if((src_level == INT_MAX) && (dst_level == INT_MAX)) 
			{
				//Case 5: u and v are in a separate connected component from s. Even with the new edge, there are no shortest paths from s to u or s to v. Special case of case 1?
				if(debug)
				{
					std::cout << "Root " << i << " - Case 5: No update needed." << std::endl;
				}
				case_stats[4]++;
			}	
			else if(((src_level == INT_MAX) && (dst_level != INT_MAX)) || ((src_level != INT_MAX) && (dst_level == INT_MAX)))
			{
				//Case 4: Either u or v (but not both) is in the same connected component as s. Special case of case 3?
				if(debug)
				{	
					std::cout << "Root " << i << " - Case 4: Update required." << std::endl;
				}
				if(src_level == INT_MAX)
				{
					int distance = src_level - dst_level - 1;
					non_adjacent_level_insertion(g,approx,k,src,dst,d_old[j],sigma_old[j],delta_old[j],bc,i,distance,nodes_touched[1],false);
				}
				else
				{
					int distance = dst_level - src_level - 1;
					non_adjacent_level_insertion(g,approx,k,dst,src,d_old[j],sigma_old[j],delta_old[j],bc,i,distance,nodes_touched[1],false);
				}
				//recompute_root(g,d_old[j],sigma_old[j],delta_old[j],bc,i);
				case_stats[3]++;
			}		
			else if(src_level == dst_level)
			{
				//Case 1: u and v are in the same level of s's BFS tree - no work needs to be done
				if(debug)
				{
					std::cout << "Root " << i << " - Case 1: No update needed." << std::endl;
				}
				case_stats[0]++;
			}
			else if(abs(src_level - dst_level) == 1)
			{
				//Case 2: u and v reside in adjacent levels
				if(debug)
				{
					std::cout << "Root " << i << " - Case 2: Update required." << std::endl;
				}
				
				//Start traversal at the vertex further from the root
				if(src_level > dst_level)
				{
					adjacent_level_insertion(g,src,dst,d_old[j],sigma_old[j],delta_old[j],bc,i,nodes_touched[0],false);
				}
				else //dst is further from the root
				{
					adjacent_level_insertion(g,dst,src,d_old[j],sigma_old[j],delta_old[j],bc,i,nodes_touched[0],false);
				}
				case_stats[1]++;
			}
			else if(abs(src_level - dst_level) > 1)
			{
				//Case 3: u and v reside in non-adjacent levels
				if(debug)
				{
					std::cout << "Root " << i << " - Case 3: Update required." << std::endl;
				}
				int distance = abs(src_level - dst_level) - 1;
				if(src_level > dst_level)
				{
					non_adjacent_level_insertion(g,approx,k,src,dst,d_old[j],sigma_old[j],delta_old[j],bc,i,distance,nodes_touched[1],false);
				}
				else
				{
					non_adjacent_level_insertion(g,approx,k,dst,src,d_old[j],sigma_old[j],delta_old[j],bc,i,distance,nodes_touched[1],false);
				}
				//For now, do a full recomputation for this scenario to help debug case 2
				//recompute_root(g,d_old[j],sigma_old[j],delta_old[j],bc,i);
				case_stats[2]++;
			}
			else
			{
				//Error?
				std::cerr << "Error: Case unaccounted for." << std::endl;
				std::cerr << "Root: " << i << " Source distance: " << src_level << "Destination distance: " << dst_level << std::endl;
				exit(-1);
			}
		}
	}
	else
	{
		for(int i=0; i<g.n; i++)
		{
			//Figure out which insertion case we're dealing with
			int src_level = d_old[i][src];
			int dst_level = d_old[i][dst];

			if((src_level == INT_MAX) && (dst_level == INT_MAX)) 
			{
				//Case 5: u and v are in a separate connected component from s. Even with the new edge, there are no shortest paths from s to u or s to v. Special case of case 1?
				if(debug)
				{
				//	std::cout << "Root " << i << " - Case 5: No update needed." << std::endl;
				}
				case_stats[4]++;
			}	
			else if(((src_level == INT_MAX) && (dst_level != INT_MAX)) || ((src_level != INT_MAX) && (dst_level == INT_MAX)))
			{
				//Case 4: Either u or v (but not both) is in the same connected component as s. Special case of case 3? 
				if(debug)
				{
				//	std::cout << "Root " << i << " - Case 4: Update required." << std::endl;
				}
				if(src_level == INT_MAX)
				{
					int distance = src_level - dst_level - 1;
					non_adjacent_level_insertion(g,approx,k,src,dst,d_old[i],sigma_old[i],delta_old[i],bc,i,distance,nodes_touched[1],false);
				}
				else
				{
					int distance = dst_level - src_level - 1;
					non_adjacent_level_insertion(g,approx,k,dst,src,d_old[i],sigma_old[i],delta_old[i],bc,i,distance,nodes_touched[1],false);
				}
				//recompute_root(g,d_old[i],sigma_old[i],delta_old[i],bc,i);
				case_stats[3]++;
			}		
			else if(src_level == dst_level)
			{
				//Case 1: u and v are in the same level of s's BFS tree - no work needs to be done
				if(debug)
				{
				//	std::cout << "Root " << i << " - Case 1: No update needed." << std::endl;
				}
				case_stats[0]++;
			}
			else if(abs(src_level - dst_level) == 1)
			{
				//Case 2: u and v reside in adjacent levels
				if(debug)
				{
				//	std::cout << "Root " << i << " - Case 2: Update required." << std::endl;
				}
				
				//Start traversal at the vertex further from the root
				if(src_level > dst_level)
				{
					adjacent_level_insertion(g,src,dst,d_old[i],sigma_old[i],delta_old[i],bc,i,nodes_touched[0],false);
				}
				else //dst is further from the root
				{
					adjacent_level_insertion(g,dst,src,d_old[i],sigma_old[i],delta_old[i],bc,i,nodes_touched[0],false);
				}
				case_stats[1]++;
			}
			else if(abs(src_level - dst_level) > 1)
			{
				//Case 3: u and v reside in non-adjacent levels
				if(debug)
				{
					std::cout << "Root " << i << " - Case 3: Update required." << std::endl;
				}
				int distance = abs(src_level - dst_level) - 1;
				if(src_level > dst_level)
				{
					non_adjacent_level_insertion(g,approx,k,src,dst,d_old[i],sigma_old[i],delta_old[i],bc,i,distance,nodes_touched[1],false);
				}
				else
				{
					non_adjacent_level_insertion(g,approx,k,dst,src,d_old[i],sigma_old[i],delta_old[i],bc,i,distance,nodes_touched[1],false);
				}
				//For now, do a full recomputation for this scenario to help debug case 2
				//recompute_root(g,d_old[i],sigma_old[i],delta_old[i],bc,i);
				case_stats[2]++;
			}
			else
			{
				//Error?
				std::cerr << "Error: Case unaccounted for." << std::endl;
				std::cerr << "Root: " << i << " Source distance: " << src_level << "Destination distance: " << dst_level << std::endl;
				exit(-1);
			}
		}
	}
}

void adjacent_level_insertion(csr_graph g, int u_low, int u_high, std::vector<int> &d_old, std::vector<unsigned long long> &sigma_old, std::vector<float> &delta_old, float *bc, int i, std::vector<unsigned int> &nodes_touched, bool hetero)
{
	//Note that i is the current root
	std::vector<int> t(g.n,0); //0 = Not Touched, 1 = Up, -1 = Down
	t[u_low] = -1;
	std::vector<int> dP(g.n,0); //Number of new shortest paths
	dP[u_low] = sigma_old[u_high];
	//Can instead initialize this when needed during the BFS
	std::vector<unsigned long long> sigma_hat(sigma_old);
	sigma_hat[u_low] += dP[u_low];	
	std::queue<int> Q;
	Q.push(u_low);
	std::vector< std::queue<int> > QQ(g.n); //Vector of queues. One queue for each level.
	QQ[d_old[u_low]].push(u_low);
	while(!Q.empty())
	{
		int v = Q.front();
		Q.pop();

		int begin = g.R[v];
		int end = g.R[v+1];
		for(int j=begin; j<end; j++) //For each neighbor of v...
		{		
			int w = g.C[j];
			if(d_old[w] == (d_old[v] + 1))
			{
				if(t[w] == 0)
				{
					Q.push(w);
					QQ[d_old[w]].push(w);
					t[w] = -1;
					dP[w] = dP[v];
				}
				else
				{
					dP[w] += dP[v];
				}
				sigma_hat[w] += dP[v];
			}
		}
	}
	std::vector<float> delta_hat(g.n,0);
	int level = g.n-1; //Worst case distance from the root is (n-1) hops. Make this max{d_old[w]}?
	while(level > 0)
	{
		while(!QQ[level].empty())
		{
			int w = QQ[level].front();
			QQ[level].pop();

			int begin = g.R[w];
			int end = g.R[w+1];
			for(int j=begin; j<end; j++) //For each neighbor of w...
			{
				int v = g.C[j];
				if(d_old[w] == (d_old[v] + 1)) //If v is a predecessor of w
				{
					if(t[v] == 0)
					{
						QQ[level-1].push(v);
						t[v] = 1;
						delta_hat[v] = delta_old[v];
					}
					delta_hat[v] += (sigma_hat[v]/(float)sigma_hat[w])*(1 + delta_hat[w]);
					if((t[v] == 1) && ((v != u_high) || (w != u_low)))
					{
						delta_hat[v] -= (sigma_old[v]/(float)sigma_old[w])*(1 + delta_old[w]);	
					}
					if(!hetero) //Let the heterogeneous implementation handle this itself
					{
						if(w != i)
						{
							bc[w] = bc[w] + delta_hat[w] - delta_old[w];
						}
					}
				}
			}
		}
		level--;
	}
	int touched=0;
	for(int j=0; j<g.n; j++)
	{
		sigma_old[j] = sigma_hat[j];
		if(t[j] != 0)
		{
			delta_old[j] = delta_hat[j];
			touched++;
		}
	}
	if(!hetero)
	{
		nodes_touched.push_back(touched);
	}

	/*
	std::cout << "Updated BC scores: " << std::endl;
	for(int j=0; j<g.n; j++)
	{
		if(j == g.n-1)
		{
			std::cout << bc[j] << std::endl;
		}
		else
		{
			std::cout << bc[j] << ",";
		}
	}
	*/	
}

void non_adjacent_level_insertion(csr_graph g, bool approx, int k, int u_low, int u_high, std::vector<int> &d_old, std::vector<unsigned long long> &sigma_old, std::vector<float> &delta_old, float *bc, int i, int distance, std::vector<unsigned int> &nodes_touched, bool hetero)
{
	//Note that i is the current root
	std::vector<int> t(g.n,0); //0 = Not Touched, 1 = Up, -1 = Down, 2 = Up when it didn't move
	t[u_low] = -1;
	std::vector<int> dP(g.n,0); //Number of new shortest paths
	dP[u_low] = sigma_old[u_high];
	std::vector<bool> moved(g.n,false); //Did this vertex move up?
	moved[u_low] = true;
	std::vector<int> movement(g.n,0); //How far did this vertex move up?
	movement[u_low] = distance;
	//Can instead initialize this when needed during the BFS
	std::vector<unsigned long long> sigma_hat(sigma_old);
	sigma_hat[u_low] = dP[u_low];	
	std::queue<int> Q;
	Q.push(u_low);
	std::vector< std::queue<int> > QQ; //Vector of queues. One queue for each level.
	for(int j=0; j<g.n; j++)
	{
		QQ.push_back(std::queue<int>());
	}
	//QQ[d_old[i][u_low]].push(u_low);
	
	while(!Q.empty())
	{
		int v = Q.front();
		Q.pop();

		int begin = g.R[v];
		int end = g.R[v+1];
		for(int j=begin; j<end; j++) //For each neighbor of v...
		{		
			int w = g.C[j];
			if(t[w] == 0)
			{
				int computed_distance = (movement[v] - movement[w]) - (d_old[v] - d_old[w] + 1);
				t[w] = -1;
			        if(computed_distance > 0) //Adjacent vertex should be moved
			        {
				        dP[w] = dP[v];
					sigma_hat[w] = dP[v];
					moved[w] = true;
					movement[w] = computed_distance;
					Q.push(w);
			        }
			        else if (computed_distance == 0) //Vertex will not move
			        {
				        dP[w] += dP[v];
					sigma_hat[w] += dP[v];
					Q.push(w);
			        }
				else //Else, the number of SPs from the root to this vertex does not change	       
				{
					t[w] = 0;
				}
			}
			else if(t[w] == -1)
			{
				int computed_distance = (movement[v] - movement[w]) - (d_old[v] - d_old[w] + 1);
				if(computed_distance >= 0)
				{
					sigma_hat[w] += dP[v];
					dP[w] += dP[v];				
				}
			}
		}

		d_old[v] -= movement[v]; //Adjust value of d
		QQ[d_old[v]].push(v);
	}

	/*std::cout << "Old delta values: ";
	for(int j=0;j<g.n;j++)
	{
		if(j == (g.n-1))
		{
			std::cout << delta_old[j] << std::endl;
		}
		else
		{
			std::cout << delta_old[j] << ",";
		}
	}*/

	std::vector<float> delta_hat(g.n,0);
	int level = g.n-1; //Worst case distance from the root is (n-1) hops
	while(level > 0)
	{
		while(!QQ[level].empty())
		{
			int w = QQ[level].front();
		//	std::cout << "QQ[" << level << "] = " << w << " popped." << std::endl;
			QQ[level].pop();

			int begin = g.R[w];
			int end = g.R[w+1];
			for(int j=begin; j<end; j++) //For each neighbor of w...
			{
				int v = g.C[j];
				if(d_old[v] == d_old[w]-1)
				{
					if((t[v] == 0))
					{
						delta_hat[v] = delta_old[v];
						t[v] = 1;
						QQ[level-1].push(v);
					}
					delta_hat[v] += (sigma_hat[v]/(float)sigma_hat[w])*(1 + delta_hat[w]);
					if((t[v] > 0) && ((v != u_high) || (w != u_low)))
					{
						delta_hat[v] -= (sigma_old[v]/(float)sigma_old[w])*(1 + delta_old[w]);
					}
				}
				else if((d_old[v] == d_old[w]) && (moved[w]) && (!moved[v]))
				{
					//if((t[v] == 0) || (t[v] == -1))
					if(t[v] == 0)
					{
						delta_hat[v] = delta_old[v];
						t[v] = 2;
						QQ[level].push(v);
					}
					delta_hat[v] -= (sigma_old[v]/(float)sigma_old[w])*(1 + delta_old[w]);
				}
			}

			if(!hetero)
			{
				if(w != i)
				{	
					bc[w] = bc[w] + delta_hat[w] - delta_old[w];
				}
			}
		}
		level--;
	}
	int touched=0;
	for(int j=0; j<g.n; j++)
	{
		sigma_old[j] = sigma_hat[j];
		if(t[j] != 0)
		{
			delta_old[j] = delta_hat[j];
			touched++;
		}
	}
	if(!hetero)
	{
		nodes_touched.push_back(touched);
	}
	/*std::cout << "Updated BC scores: " << std::endl;
	for(int j=0; j<g.n; j++)
	{
		if(j == g.n-1)
		{
			std::cout << bc[j] << std::endl;
		}
		else
		{
			std::cout << bc[j] << ",";
		}
	}*/	
}

void recompute_root(csr_graph g, std::vector<int> &d_old, std::vector<unsigned long long> &sigma_old, std::vector<float> &delta_old, float *bc, int i)
{
	for(int j=0;j<g.n;j++)
	{
		d_old[j] = -1;
		sigma_old[j] = 0;
	}
	d_old[i] = 0; 
	sigma_old[i] = 1;
	std::queue<int> Q;
	Q.push(i);
	std::stack<int> S;

	while(!Q.empty())
	{
		int v = Q.front();
		Q.pop();
		S.push(v);
		int begin = g.R[v];
		int end = g.R[v+1];
		for(int j=begin; j<end; j++)
		{
			int w = g.C[j];
			if(d_old[w] == -1)
			{
				Q.push(w);
				d_old[w] = d_old[v]+1;
			}
			if(d_old[w] == (d_old[v]+1))
			{
				sigma_old[w] += sigma_old[v];
			}	
		}
	}

	for(int j=0;j<g.n;j++)
	{
		if(j!=i)
		{
			bc[j] -= delta_old[j];
		}
		delta_old[j] = 0;
	}
	
	while(!S.empty())
	{
		int w = S.top();
		S.pop();
		int begin = g.R[w];
		int end = g.R[w+1];
		for(int j=begin; j<end; j++)
		{
			int v = g.C[j];
			if(d_old[w] == (d_old[v] + 1))
			{
				delta_old[v] += (sigma_old[v]/(float)sigma_old[w])*(1 + delta_old[w]);
			}
		}
		if(w != i)
		{
			bc[w] += delta_old[w];
		}
	}
}

//void heterogeneous_update(csr_graph g, bool approx, int k, int src, int dst, std::vector<int> &d_old, std::vector<unsigned long long> &sigma_old, std::vector<float> &delta_old, int root)
void *heterogeneous_update(void *arg)
{
	thread_data *t;
	t = (thread_data*) arg;
	for(int i = t->start; i < t->end; i++)
	{
		int root;
		if(t->approx)
		{
			root = source_nodes[i];
		}
		else
		{
			root = i;
		}

		int src_level = d_gpu_v[i][t->src];
		int dst_level = d_gpu_v[i][t->dst];
		std::vector<unsigned int> nodes_touched; //dummy variable since we can't pass NULL

		if(abs(src_level-dst_level) == 0)
		{
			return 0;
		}
		else if((src_level == INT_MAX) || (dst_level == INT_MAX))
		{
			//Case 4
			int distance = abs(src_level - dst_level) - 1;
			if(src_level == INT_MAX)
			{
				non_adjacent_level_insertion(t->g,t->approx,t->k,t->src,t->dst,d_gpu_v[i],sigma_gpu_v[i],delta_gpu_v[i],NULL,root,distance,nodes_touched,true);
			}
			else
			{
				non_adjacent_level_insertion(t->g,t->approx,t->k,t->dst,t->src,d_gpu_v[i],sigma_gpu_v[i],delta_gpu_v[i],NULL,root,distance,nodes_touched,true);
			}
		}
		else if(abs(src_level-dst_level) == 1)
		{
			//Case 2
			if(src_level > dst_level)
			{
				adjacent_level_insertion(t->g,t->src,t->dst,d_gpu_v[i],sigma_gpu_v[i],delta_gpu_v[i],NULL,root,nodes_touched,true);
			}
			else //dst is further from the root
			{
				adjacent_level_insertion(t->g,t->dst,t->src,d_gpu_v[i],sigma_gpu_v[i],delta_gpu_v[i],NULL,root,nodes_touched,true);
			}
		}
		else
		{
			//Case 3
			int distance = abs(src_level - dst_level) - 1;
			if(src_level > dst_level)
			{
				non_adjacent_level_insertion(t->g,t->approx,t->k,t->src,t->dst,d_gpu_v[i],sigma_gpu_v[i],delta_gpu_v[i],NULL,root,distance,nodes_touched,true);
			}
			else
			{
				non_adjacent_level_insertion(t->g,t->approx,t->k,t->dst,t->src,d_gpu_v[i],sigma_gpu_v[i],delta_gpu_v[i],NULL,root,distance,nodes_touched,true);
			}
		}
	}

	return 0;
}
