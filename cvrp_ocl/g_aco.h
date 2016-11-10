//
//  ocl_aco.hpp
//  cvrp_ocl
//
//  Created by 孙晓奇 on 2016/11/7.
//  Copyright © 2016年 xiaoqi.sxq. All rights reserved.
//

#ifndef g_aco_h
#define g_aco_h

#include <stdio.h>
#include "cpu/problem.h"
#include "g_type.h"

class g_ACO
{
public:
    g_ACO(OpenclEnv &env, Problem &instance);
    ~g_ACO();
    void run_aco_iteration(void);
    
    void construct_solutions(void);
    void local_search(void);
    void update_statistics(void);
    void pheromone_update(void);
    
private:
    OpenclEnv& env;
    Problem &instance;
    int num_node;             // number of node
    int n_ants;               // number of ants
    int max_tour_sz;          // max tour size
    
    // memory objects
    cl_mem solutions_mem;
    cl_mem solution_lens_mem;    // tour length of each solution
    cl_mem demands_mem;          // demands of each node
    cl_mem distance_mem;         // distance array
    cl_mem pheromone_mem;
    cl_mem total_info_mem;
    cl_mem nn_list_mem;
    cl_mem seed_mem;            // seeds for all ants
    
    void create_memory_objects(void);
};
#endif /* g_aco_h */
