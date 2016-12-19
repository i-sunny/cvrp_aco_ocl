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
    
    void pheromone_init(float trail_0);
    void pheromone_evaporation(void);
    void pheromone_deposit(void);
    void compute_total_info(void);
    void update_pheromone_weighted(AntStruct *a, int weight);
    
    void copy_solution_from_to(AntStruct *a1, AntStruct *a2);
    void update_best_so_far_from_device(void);    // best-so-far from device to host memory
    void update_best_so_far_to_device(void);      // best-so-far from host to device memory
    
private:
    OpenclEnv& env;
    Problem& instance;
    int num_node;             // number of node
    int n_ants;               // number of ants
    int max_tour_sz;          // max tour size
    
    int cs_grp_size;          // work-group size of construct_solutions kernel
    int cs_num_grps;          // work-group number of construct_solutions kernel
    
    float cs_exec_time;
    
    // memory objects
    cl_mem solutions_mem;
    cl_mem solution_lens_mem;    // tour length of each solution
    cl_mem update_flag_mem;      // mark if we need to update best-so-far solution after an iteration
    
    cl_mem demands_mem;          // demands of each node
    cl_mem distance_mem;         // distance array
    cl_mem pheromone_mem;
    cl_mem total_info_mem;
    cl_mem nn_list_mem;
    cl_mem seed_mem;             // seeds for all ants
    cl_mem bsf_records_mem;      // record all best-so-far solutions information
    cl_mem num_bsf_mem;          // counter of best-so-far solutions records
    cl_mem elite_ids_mem;        // ids of solutions of all `ras_rank` elite ants
    
    cl_mem best_result_val;      // especially for find_best_solution()
    cl_mem best_result_idx;      // especially for find_best_solution()
    
    void create_memory_objects(void);
    void find_best_solution(void);
    void update_best_so_far(void);
    void get_elites(void);
    double event_runtime(cl_event& event);
};

#endif /* g_aco_h */

