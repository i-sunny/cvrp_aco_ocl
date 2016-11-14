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
#include "../problem.h"
#include "g_type.h"
#include "../antColony.h"

class g_ACO {
public:
    g_ACO(OpenclEnv &env, Problem &instance);
    ~g_ACO();
    void init_aco();
    void exit_aco(void);
    void run_aco_iteration(void);
    
    void construct_solutions(void);
    void local_search(void);
    void update_statistics(void);
    void pheromone_update(void);
    
    void pheromone_evaporation(void);
    void ras_update(void);
    void pheromone_disturbance(void);
    void compute_total_info(void);
    void update_pheromone_weighted(AntStruct *a, int weight);
    
    void copy_solution_from_to(AntStruct *a1, AntStruct *a2);
    void update_best_so_far_from_mem(void);    // best-so-far from device to host memory
    void update_best_so_far_to_mem(void);      // best-so-far from host to device memory
    
private:
    OpenclEnv& env;
    Problem& instance;
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
