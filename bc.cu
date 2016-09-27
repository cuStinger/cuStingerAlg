//System includes
#include <iostream>
#include <cstdlib>
#include <stack>
#include <queue>
#include <set>
#include <cmath>
#include <getopt.h>
#include <nvml.h>
#include <numeric>

//Project includes
#include "util.cuh"
#include "metis_to_csr.h"
#include "sequential.h"
#include "kernels.cuh"
#include "gpu_setup.cuh"

//Global variables for PThread sections of code
bool *psample;
std::vector< std::vector<int> > d_gpu_v;
std::vector< std::vector<unsigned long long> > sigma_gpu_v;
std::vector< std::vector<float> > delta_gpu_v;
int *source_nodes;

int main(int argc, char **argv)
{
	int runtime_version;
	cudaRuntimeGetVersion(&runtime_version);
	std::cout << "CUDA Version: " << runtime_version << std::endl;
	//Parse program arguments
	program_options op;
	parse_arguments(argc, argv, op);

	//If we want to export the edge graph, do so and exit
	if(op.export_edge_graph)
	{
		csr_graph g = parse_metis(op.infile);
		std::string basename = std::string(op.infile).substr(0,std::string(op.infile).length()-6); //Assuming '.graph' extension
		basename.append(".edge");
		std::ofstream outfile;
		outfile.open(basename.c_str(), std::ofstream::out | std::ofstream::trunc);
		outfile << g.n << " " << g.m << std::endl;
		for(int k=0; k<2*g.m; k++)
		{
			if(g.F[k] <= g.C[k]) //Not sure how to handle self edges here. We'll include them for now.
			{
				outfile << g.F[k] << " " << g.C[k] << std::endl;
			}
		}
		return 0;	
	}

	//Choose the GPU by best compute capability or user input
	int max_threads_per_block, number_of_SMs;
	choose_device(max_threads_per_block, number_of_SMs, op.device);

	if(op.debug)
	{
		max_threads_per_block = 256;
	}

	//Parse the input graph into CSR/COO form
	csr_graph g = parse_metis(op.infile);
	std::cout << "Number of vertices: " << g.n << std::endl;
	std::cout << "Number of edges: " << g.m << std::endl;

	//If streaming, remove some number of edges at random
	std::set< std::pair<int,int> > removals; //Need unique edge removals
	if(op.streaming)
	{
		remove_edges(g,removals,op);
		if(op.approximate)
		{
			std::cout << "O(KN) memory requirement: " << sizeof(unsigned long long)*g.n*op.k + sizeof(float)*g.n*op.k + sizeof(int)*g.n*op.k << std::endl;
		}
		else
		{
			std::cout << "O(N^2) memory requirement: " << sizeof(unsigned long long)*g.n*g.n + sizeof(float)*g.n*g.n + sizeof(int)*g.n*g.n << std::endl;
		}
	}

	//If approximating, determine which source nodes to use at random
	if((op.approximate) && (op.k > g.n))
	{
		std::cerr << "Error: Number of source nodes to approximate is greater than the number of nodes in the graph." << std::endl;
		return -2;
	}
	if(op.approximate)
	{
		source_nodes = new int[op.k];
		bool *already_used = new bool[g.n];
		for(int i=0; i<g.n; i++)
		{
			already_used[i] = false;
		}
		for(int i=0; i<op.k; i++)
		{
			int temp_source = rand() % g.n;
			if(already_used[temp_source] == true)
			{
				i--;
				continue;
			}
			source_nodes[i] = temp_source;
			already_used[temp_source] = true;
		}
		delete[] already_used;
	}

	//Sequential with no parents
	//If necessary, let this return d, sigma, and delta
	float *bc_seq_no_parents;
	cudaEvent_t start, end;
	float time_no_parents;
	std::vector< std::vector<int> > d_old;
	std::vector< std::vector<unsigned long long> > sigma_old;
	std::vector< std::vector<float> > delta_old;
	if(op.approximate)
	{
		start_clock(start,end);
		bc_seq_no_parents = bc_no_parents_approx(g,op.k,source_nodes,op.printBCscores,op.streaming,d_old,sigma_old,delta_old,0,op.k);
		time_no_parents = end_clock(start,end);
	}
	else
	{
		start_clock(start,end);
		bc_seq_no_parents = bc_no_parents(g,op.printBCscores,op.streaming,d_old,sigma_old,delta_old,0,g.n);
		time_no_parents = end_clock(start,end);
	}

	//////////////////////
	//Single-GPU Algorithm
	//////////////////////
	float *bc_gpu; //Host result pointer

	bc_gpu = new float[g.n];
	for(int i=0; i<g.n; i++)
	{
		bc_gpu[i] = 0;
	}

	float time_gpu_opt;
	if(!op.streaming)
	{
        	time_gpu_opt = single_gpu_full_computation(g,op,number_of_SMs,max_threads_per_block,bc_gpu,source_nodes,false);
	}

	/////////////////////
	//Multi-GPU Algorithm (deprecated)
	/////////////////////
	/*float time_gpu_multi;
	if(op.multi) 
	{
		//Reset BC scores and validate the multi-GPU algorithm against the CPU
		for(int i=0; i<g.n; i++)
		{
			bc_gpu[i] = 0;
		}

		time_gpu_multi = multi_gpu_full_computation(g,op,bc_gpu,source_nodes);
	}*/
	//////////////////////
	//Streaming Algorithms
	//////////////////////
	std::vector<unsigned long long> case_stats(5,0); //Get some info on the distribution of cases
	std::vector< std::vector<unsigned int> > nodes_touched(2);
	float time_seq_update = 0;
	float time_min_update = INT_MAX;
	float time_max_update = 0;
	std::pair<int,int> min_update_edge;
	std::pair<int,int> max_update_edge;

	/*float time_gpu_update = 0;
	float time_min_update_gpu = INT_MAX;
	float time_max_update_gpu = 0;

	float time_gpu_update_soa=0;
	float time_min_update_gpu_soa=INT_MAX;
	float time_max_update_gpu_soa=0;	

	float time_gpu_update_aos=0;
	float time_min_update_gpu_aos=INT_MAX;
	float time_max_update_gpu_aos=0;*/	

	float time_gpu_update_edge=0;
	float time_min_update_gpu_edge=INT_MAX;
	float time_max_update_gpu_edge=0;	

	float time_gpu_update_node=0;
	float time_min_update_gpu_node=INT_MAX;
	float time_max_update_gpu_node=0;	
	
	float time_gpu_update_hyb=0;
	float time_min_update_gpu_hyb=INT_MAX;
	float time_max_update_gpu_hyb=0;

	float time_heterogeneous_update=0;
	float time_min_update_hetero=INT_MAX;
	float time_max_update_hetero=0;
	float time_for_accumulation=0;
	float time_CPU=0;
	
	float time_gpu_recomputation = 0;
	std::pair<int,int> min_update_gpu_edge;
	std::pair<int,int> max_update_gpu_edge;
	std::pair<int,int> min_update_gpu_node;
	std::pair<int,int> max_update_gpu_node;
	std::pair<int,int> min_update_gpu_hyb;
	std::pair<int,int> max_update_gpu_hyb;
	std::pair<int,int> min_update_hetero;
	std::pair<int,int> max_update_hetero;

	//Debugging intermediates
		
	if(op.streaming)
	{
		std::set< std::pair<int,int> > removals_gpu = removals; //Make a copy to do CPU/GPU streaming separately
		std::cout << std::endl;
		if(!op.no_cpu)
		{
			std::cout << "CPU Streaming: " << std::endl;
		}
		while(removals.size())
		{
			int source = removals.begin()->first;
			int dest = removals.begin()->second;
			g.reinsert_edge(source,dest);
			removals.erase(removals.begin());
			if(removals.find(std::make_pair(dest,source)) != removals.end())
			{
				removals.erase(std::make_pair(dest,source));
			}
			else
			{
				std::cerr << "Error reinserting edges: edge (" << source << "," << dest << ") found but edge (" << dest << "," << source << ") could not be found." << std::endl;
				return -1;
			}

			//Update on the CPU
			if(!op.no_cpu)
			{
				std::vector<unsigned long long> old_cases(5);
				std::copy(case_stats.begin(),case_stats.end(),old_cases.begin());
				start_clock(start,end);
				bc_update_edge(g,op.approximate,op.k,source_nodes,source,dest,d_old,sigma_old,delta_old,bc_seq_no_parents,case_stats,nodes_touched,op.debug);
				float prev_time = time_seq_update;
				time_seq_update += end_clock(start,end);
				std::cout << "Inserting edge: (" << source << "," << dest << ") - Breakdown of cases: (";
				for(int i=0; i<5; i++)
				{
					if(i==4)
					{
						std::cout << case_stats[i]-old_cases[i] << ")" << std::endl;
					}
					else
					{
						std::cout << case_stats[i]-old_cases[i] << ",";
					}

				}
				if(time_seq_update-prev_time < time_min_update)
				{
					time_min_update = time_seq_update - prev_time;
					min_update_edge.first = source;
					min_update_edge.second = dest;
				}
				if(time_seq_update-prev_time > time_max_update)
				{
					time_max_update = time_seq_update - prev_time;
					max_update_edge.first = source;
					max_update_edge.second = dest;
				}
			}
		}
		std::cout << std::endl;

		//Naive streaming kernel	
		//single_gpu_streaming_computation(g,op,bc_gpu,source_nodes,max_threads_per_block,number_of_SMs,removals_gpu,time_gpu_update,time_min_update_gpu,time_max_update_gpu,min_update_gpu_edge,max_update_gpu_edge,d_gpu_v,sigma_gpu_v,delta_gpu_v,false,false);

		//SoA kernel
		//single_gpu_streaming_computation_SOA(g,op,bc_gpu,source_nodes,max_threads_per_block,number_of_SMs,removals_gpu,time_gpu_update_soa,time_min_update_gpu_soa,time_max_update_gpu_soa);

		//AoS kernel
		//single_gpu_streaming_computation_AOS(g,op,bc_gpu,source_nodes,max_threads_per_block,number_of_SMs,removals_gpu,time_gpu_update_aos,time_min_update_gpu_aos,time_max_update_gpu_aos);
		
		if(op.node == 0)
		{
			//Optimized edge-based streaming kernel	
			single_gpu_streaming_computation(g,op,bc_gpu,source_nodes,max_threads_per_block,number_of_SMs,removals_gpu,time_gpu_update_edge,time_min_update_gpu_edge,time_max_update_gpu_edge,min_update_gpu_edge,max_update_gpu_edge,d_gpu_v,sigma_gpu_v,delta_gpu_v,true,0);
		}
		else if(op.node == 1)
		{
			//Optimized node-based streaming kernel	
			single_gpu_streaming_computation(g,op,bc_gpu,source_nodes,max_threads_per_block,number_of_SMs,removals_gpu,time_gpu_update_node,time_min_update_gpu_node,time_max_update_gpu_node,min_update_gpu_node,max_update_gpu_node,d_gpu_v,sigma_gpu_v,delta_gpu_v,true,1);
		}
		else //FIXME: Deprecated
		{
			//Hybrid: Case 2 is node based and Case 3 is edge based
			single_gpu_streaming_computation(g,op,bc_gpu,source_nodes,max_threads_per_block,number_of_SMs,removals_gpu,time_gpu_update_hyb,time_min_update_gpu_hyb,time_max_update_gpu_hyb,min_update_gpu_hyb,max_update_gpu_hyb,d_gpu_v,sigma_gpu_v,delta_gpu_v,true,2);
		}

		if(op.multi)
		{
			heterogeneous_streaming_computation(g,op,bc_gpu,source_nodes,max_threads_per_block,number_of_SMs,removals_gpu,time_heterogeneous_update,time_min_update_hetero,time_max_update_hetero,min_update_hetero,max_update_hetero,time_for_accumulation,time_CPU);
		}
		
		//Hybrid edge-node kernel: edge for BFS, node for depend
		//single_gpu_streaming_computation(g,op,bc_gpu,source_nodes,max_threads_per_block,number_of_SMs,removals_gpu,time_gpu_update_hyb,time_min_update_gpu_hyb,time_max_update_gpu_hyb,min_update_gpu_hyb,max_update_gpu_hyb,d_gpu_v,sigma_gpu_v,delta_gpu_v,true,2);

		//GPU full recomputation
		float *bc_gpu_recomp = new float[g.n];
		for(int i=0; i<g.n; i++)
		{
			bc_gpu_recomp[i] = 0;
		}
		time_gpu_recomputation = single_gpu_full_computation(g,op,number_of_SMs,max_threads_per_block,bc_gpu_recomp,source_nodes,true);
		delete[] bc_gpu_recomp;

		//Print CPU-based statistics here	
		/*if(!op.no_cpu)
		{	
			std::cout << "Distribution of cases: " << std::endl;
			for(int i=0; i<5; i++)
			{
				std::cout << "Case " << i+1 << ": " << case_stats[i] << std::endl;
			}
			std::cout << "Percentage of nodes touched for Case 2: " << std::endl;
			std::cout << "Average: " << (std::accumulate(nodes_touched[0].begin(),nodes_touched[0].end(),0)/(float)nodes_touched[0].size())/(float)g.n << std::endl;
			std::cout << "Maximum: " << (*std::max_element(nodes_touched[0].begin(),nodes_touched[0].end()))/(float)g.n << std::endl;
			std::cout << "Minimum: " << (*std::min_element(nodes_touched[0].begin(),nodes_touched[0].end()))/(float)g.n << std::endl;
			std::cout << "Percentage of nodes touched for Cases 3/4: " << std::endl;
			std::cout << "Average: " << (std::accumulate(nodes_touched[1].begin(),nodes_touched[1].end(),0)/(float)nodes_touched[1].size())/(float)g.n << std::endl;
			std::cout << "Maximum: " << (*std::max_element(nodes_touched[1].begin(),nodes_touched[1].end()))/(float)g.n << std::endl;
			std::cout << "Minimum: " << (*std::min_element(nodes_touched[1].begin(),nodes_touched[1].end()))/(float)g.n << std::endl;

			std::string basename = std::string(op.infile).substr(0,std::string(op.infile).length()-6); //Assuming '.graph' extension
			basename.append("_touched_nodes.csv");
			std::ofstream tfile;
			tfile.open(basename.c_str(), std::ofstream::out | std::ofstream::trunc);
			for(int i=0; i<nodes_touched[0].size(); i++)
			{
				if(i == nodes_touched[0].size()-1)
				{
					tfile << (nodes_touched[0][i]/(float)g.n)*100 << std::endl;
				}
				else
				{
					tfile << (nodes_touched[0][i]/(float)g.n)*100 << ",";
				}
			}
			for(int i=0; i<nodes_touched[1].size(); i++)
			{
				if(i == nodes_touched[1].size()-1)
				{
					tfile << (nodes_touched[1][i]/(float)g.n)*100 << std::endl;
				}
				else
				{
					tfile << (nodes_touched[1][i]/(float)g.n)*100 << ",";
				}
			}

			tfile.close();
			return 0;	
		}*/
	}

	for(int i=0; i<g.n; i++) //Divide results by 2 for unweighted graphs
	{
		bc_gpu[i] = bc_gpu[i]/(float)2;	//Divide by 2 for unweighted graphs
		bc_gpu[i] = bc_gpu[i]/(float)((g.n-1)*(g.n-2)); //Now normalize
	}

	std::cout << std::setprecision(9);
	float time_seq_recompute;
	if(op.verify_results)
	{
		if(op.streaming)
		{
			std::vector< std::vector<int> >tmp1;
			std::vector< std::vector<unsigned long long> >tmp2;
			std::vector< std::vector<float> >tmp3;
			float *bc_after_insertions;
			start_clock(start,end);
		        if(op.approximate)
			{
				bc_after_insertions = bc_no_parents_approx(g,op.k,source_nodes,false,true,tmp1,tmp2,tmp3,0,op.k);
			}
			else
			{
				bc_after_insertions = bc_no_parents(g,false,true,tmp1,tmp2,tmp3,0,g.n);
			}
			time_seq_recompute = end_clock(start,end);
			for(int i=0; i<g.n; i++)
			{
				bc_after_insertions[i] = bc_after_insertions[i]/(float)2;
				bc_after_insertions[i] = bc_after_insertions[i]/(float)((g.n-1)*(g.n-2));
				bc_seq_no_parents[i] = bc_seq_no_parents[i]/(float)2;
				bc_seq_no_parents[i] = bc_seq_no_parents[i]/(float)((g.n-1)*(g.n-2));
			}
			//std::cout << "CPU verification: " << std::endl;
			//verify(bc_seq_no_parents,bc_after_insertions,g);
			//std::cout << "GPU verification: " << std::endl;
			verify(bc_gpu,bc_after_insertions,g);
			if(op.debug)
			{
				std::cout << std::endl;
				//stream_debug_print<float*>(g,bc_after_insertions,bc_seq_no_parents,"BC");
				/*std::cout << "Full computation: " << std::endl;
				for(int i=0; i<g.n; i++)
				{
					if(i == (g.n-1))
					{
						std::cout << bc_after_insertions[i] << std::endl;
					}	
					else
					{
						std::cout << bc_after_insertions[i] << ",";
					}
				}
				std::cout << "CPU update: " << std::endl;
				for(int i=0; i<g.n; i++)
				{
					if(i == (g.n-1))
					{
						std::cout << bc_seq_no_parents[i] << std::endl;
					}	
					else
					{
						std::cout << bc_seq_no_parents[i] << ",";
					}
				}*/
				/*for(int i=0; i<g.n; i++)
				{
					std::cout << "i = " << i << std::endl;
					stream_debug_print<std::vector<int> >(g,tmp1[i],d_old[i],"d");
					stream_debug_print<std::vector<unsigned long long> >(g,tmp2[i],sigma_old[i],"sigma");
					stream_debug_print<std::vector<float> >(g,tmp3[i],delta_old[i],"delta");
				}*/
				/*std::cout << "GPU update: " << std::endl;
				for(int i=0; i<g.n; i++)
				{
					if(i == (g.n-1))
					{
						std::cout << bc_gpu[i] << std::endl;
					}	
					else
					{
						std::cout << bc_gpu[i] << ",";
					}
				}*/
				if(g.n < 2000)
				{
					for(int i=0; i<g.n; i++)
					{
						std::cout << "i = " << i << std::endl;
						stream_debug_print<std::vector<int> >(g,tmp1[i],d_gpu_v[i],"d");
						stream_debug_print<std::vector<unsigned long long> >(g,tmp2[i],sigma_gpu_v[i],"sigma");
						stream_debug_print<std::vector<float> >(g,tmp3[i],delta_gpu_v[i],"delta");
					}
				}
			}
			delete[] bc_after_insertions;
		}
		else
		{
			verify(bc_seq_no_parents,bc_gpu,g);
			std::cout << "Time for sequential algorithm with no parents: " << (time_no_parents)/(float)1000 << " s" << std::endl;
		}
	}
	if(!op.streaming)
	{
		std::cout << "Time for optimized GPU algorithm: " << (time_gpu_opt)/(float)1000 << " s" << std::endl;
	}
	if(op.streaming)
	{
		float update_time = (time_seq_update)/(float)1000;
		float gpu_recomp = time_gpu_recomputation/(float)1000;
		std::cout << std::endl;
		if(!op.no_cpu)
		{
			std::cout << "Time for CPU update: " << update_time  << " s" << std::endl;
			std::cout << "Fastest update ("<< min_update_edge.first << "," << min_update_edge.second << "): " << (time_min_update)/(float)1000 << " s" << std::endl;
			std::cout << "Slowest update (" << max_update_edge.first << "," << max_update_edge.second << "): " << (time_max_update)/(float)1000 << " s" << std::endl;
			if(op.verify_results)
			{
				std::cout << "Time for CPU recomputation: " << (time_seq_recompute)/(float)1000 << " s" << std::endl;
			}
			std::cout << "Updates per second: " << op.insertions/update_time << std::endl;
			std::cout << std::endl;
		}

		std::cout << "Time for GPU recomputation: " << gpu_recomp << " s" << std::endl;
		std::cout << std::endl;

		/*float gpu_update_time = (time_gpu_update)/(float)1000;
		std::cout << "Time for GPU update: " << gpu_update_time << " s" << std::endl;
		std::cout << "Fastest update: " << (time_min_update_gpu)/(float)1000 << " s" << std::endl;
		std::cout << "Slowest update: " << (time_max_update_gpu)/(float)1000 << " s" << std::endl;
		std::cout << "Updates per second: " << op.insertions/gpu_update_time << std::endl;
		std::cout << "Average Speedup over GPU Recomputation: " << gpu_recomp/(gpu_update_time/(float)op.insertions) << "" << std::endl;
		std::cout << "Speedup over CPU streaming: " << update_time/gpu_update_time << std::endl;
		std::cout << std::endl;
		
		float gpu_update_time_soa = (time_gpu_update_soa)/(float)1000;
		std::cout << "Time for SoA GPU update: " << gpu_update_time_soa << " s" << std::endl;
		std::cout << "Fastest update: " << (time_min_update_gpu_soa)/(float)1000 << " s" << std::endl;
		std::cout << "Slowest update: " << (time_max_update_gpu_soa)/(float)1000 << " s" << std::endl;
		std::cout << "Updates per second: " << op.insertions/gpu_update_time_soa << std::endl;
		std::cout << "Average Speedup over GPU Recomputation: " << gpu_recomp/(gpu_update_time_soa/(float)op.insertions) << "" << std::endl;
		std::cout << "Speedup over CPU streaming: " << update_time/gpu_update_time_soa << std::endl;
		std::cout << std::endl;
		
		float gpu_update_time_aos = (time_gpu_update_aos)/(float)1000;
		std::cout << "Time for AoS GPU update: " << gpu_update_time_aos << " s" << std::endl;
		std::cout << "Fastest update: " << (time_min_update_gpu_aos)/(float)1000 << " s" << std::endl;
		std::cout << "Slowest update: " << (time_max_update_gpu_aos)/(float)1000 << " s" << std::endl;
		std::cout << "Updates per second: " << op.insertions/gpu_update_time_aos << std::endl;
		std::cout << "Average Speedup over GPU Recomputation: " << gpu_recomp/(gpu_update_time_aos/(float)op.insertions) << "" << std::endl;
		std::cout << "Speedup over CPU streaming: " << update_time/gpu_update_time_aos << std::endl;
		std::cout << std::endl;*/

		if(op.node == 0)
		{
			float gpu_update_time_edge = (time_gpu_update_edge)/(float)1000;
			float gpu_update_time_min_edge = (time_min_update_gpu_edge)/(float)1000;
			float gpu_update_time_max_edge = (time_max_update_gpu_edge)/(float)1000;
			std::cout << "Time for Edge-based GPU update: " << gpu_update_time_edge << " s" << std::endl;
			std::cout << "Fastest update (" << min_update_gpu_edge.first << "," << min_update_gpu_edge.second << "): " << (time_min_update_gpu_edge)/(float)1000 << " s" << std::endl;
			std::cout << "Slowest update (" << max_update_gpu_edge.first << "," << max_update_gpu_edge.second << "): " << (time_max_update_gpu_edge)/(float)1000 << " s" << std::endl;
			std::cout << "Updates per second: " << op.insertions/gpu_update_time_edge << std::endl;
			std::cout << "Average Speedup over GPU Recomputation: " << gpu_recomp/(gpu_update_time_edge/(float)op.insertions) << "" << std::endl;
			std::cout << "Fastest Speedup over GPU Recomputation: " << gpu_recomp/gpu_update_time_min_edge << std::endl;
			std::cout << "Slowest Speedup over GPU Recomputation: " << gpu_recomp/gpu_update_time_max_edge << std::endl;
			if(!op.no_cpu)
			{
				std::cout << "Speedup over CPU streaming: " << update_time/gpu_update_time_edge << std::endl;
			}
			std::cout << std::endl;
		}
		else if(op.node == 1)
		{	
			float gpu_update_time_node = (time_gpu_update_node)/(float)1000;
			float gpu_update_time_min_node = (time_min_update_gpu_node)/(float)1000;
			float gpu_update_time_max_node = (time_max_update_gpu_node)/(float)1000;
			std::cout << "Time for Node-based GPU update: " << gpu_update_time_node << " s" << std::endl;
			std::cout << "Fastest update (" << min_update_gpu_node.first << "," << min_update_gpu_node.second << "): " << (time_min_update_gpu_node)/(float)1000 << " s" << std::endl;
			std::cout << "Slowest update (" << max_update_gpu_node.first << "," << max_update_gpu_node.second << "): " << (time_max_update_gpu_node)/(float)1000 << " s" << std::endl;
			std::cout << "Updates per second: " << op.insertions/gpu_update_time_node << std::endl;
			std::cout << "Average Speedup over GPU Recomputation: " << gpu_recomp/(gpu_update_time_node/(float)op.insertions) << "" << std::endl;
			std::cout << "Fastest Speedup over GPU Recomputation: " << gpu_recomp/gpu_update_time_min_node << std::endl;
			std::cout << "Slowest Speedup over GPU Recomputation: " << gpu_recomp/gpu_update_time_max_node << std::endl;
			if(!op.no_cpu)
			{
				std::cout << "Speedup over CPU streaming: " << update_time/gpu_update_time_node << std::endl;
			}
			std::cout << std::endl;
		}
		else
		{
			float gpu_update_time_hyb = (time_gpu_update_hyb)/(float)1000;
			float gpu_update_time_min_hyb = (time_min_update_gpu_hyb)/(float)1000;
			float gpu_update_time_max_hyb = (time_max_update_gpu_hyb)/(float)1000;
			std::cout << "Time for Hybrid GPU update: " << gpu_update_time_hyb << " s" << std::endl;
			std::cout << "Fastest update (" << min_update_gpu_hyb.first << "," << min_update_gpu_hyb.second << "): " << (time_min_update_gpu_hyb)/(float)1000 << " s" << std::endl;
			std::cout << "Slowest update (" << max_update_gpu_hyb.first << "," << max_update_gpu_hyb.second << "): " << (time_max_update_gpu_hyb)/(float)1000 << " s" << std::endl;
			std::cout << "Updates per second: " << op.insertions/gpu_update_time_hyb << std::endl;
			std::cout << "Average Speedup over GPU Recomputation: " << gpu_recomp/(gpu_update_time_hyb/(float)op.insertions) << "" << std::endl;
			std::cout << "Fastest Speedup over GPU Recomputation: " << gpu_recomp/gpu_update_time_min_hyb << std::endl;
			std::cout << "Slowest Speedup over GPU Recomputation: " << gpu_recomp/gpu_update_time_max_hyb << std::endl;
			if(!op.no_cpu)
			{
				std::cout << "Speedup over CPU streaming: " << update_time/gpu_update_time_hyb << std::endl;
			}
			std::cout << std::endl;
		}

		if(op.multi)
		{
			float heterogeneous_update_time = time_heterogeneous_update/(float)1000;
			float heterogeneous_update_time_min = time_min_update_hetero/(float)1000;
			float heterogeneous_update_time_max = time_max_update_hetero/(float)1000;
			float heterogeneous_accumulation_time = time_for_accumulation/(float)1000;
			float CPU_time = time_CPU/(float)1000;
			std::cout << "Time for Heterogeneous update: " << heterogeneous_update_time  << " s" << std::endl;
			std::cout << "Time for Accumulation: " << heterogeneous_accumulation_time << " s" << std::endl;
			std::cout << "Time for CPU portion of update: " << CPU_time << " s" << std::endl;
			/*std::cout << "Fastest update (" << min_update_hetero.first << "," << min_update_hetero.second << "): " << heterogeneous_update_time_min << " s" << std::endl;
			std::cout << "Slowest update (" << max_update_hetero.first << "," << max_update_hetero.second << "): " << heterogeneous_update_time_max << " s" << std::endl;
			float gpu_update_time_node = (time_gpu_update_node)/(float)1000;
			std::cout << "Speedup over single GPU: " << gpu_update_time_node/heterogeneous_update_time << std::endl;
			if(!op.no_cpu)
			{
				std::cout << "Speedup over CPU streaming: " << update_time/heterogeneous_update_time << std::endl;
			}*/
		}
	}
	if(op.verify_results) //This only ever gets allocated when the function is called
	{
		delete[] bc_seq_no_parents;
	}	
	if(op.approximate)
	{
		delete[] source_nodes;
	}
	delete[] bc_gpu;

	delete[] g.R; //Deallocate CSR representation
	delete[] g.C;
	delete[] g.F;

	return 0;
}

