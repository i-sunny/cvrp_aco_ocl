//
//  ocl_aco.cpp
//  cvrp_ocl
//
//  Created by 孙晓奇 on 2016/11/7.
//  Copyright © 2016年 xiaoqi.sxq. All rights reserved.
//

#include <iostream>
#include <math.h>
#include <assert.h>
#include <limits.h>

#include "g_aco.h"
#include "g_sa.h"
#include "../utilities.h"
#include "../vrpHelper.h"
#include "../io.h"
#include "../timer.h"

using namespace std;

///
//  Macros
//
#define check_error(a, b) check_error_file_line(a, b, __FILE__ , __LINE__)

void check_error_file_line(int err_num, int expected, const char* file, const int line_number);


g_ACO::g_ACO(OpenclEnv &env, Problem &instance): env(env), instance(instance)
{
    num_node = instance.num_node;
    n_ants = instance.n_ants;
    max_tour_sz = 2 * num_node;

}

g_ACO::~g_ACO()
{
    clReleaseMemObject(solutions_mem);
    clReleaseMemObject(solution_lens_mem);
    clReleaseMemObject(demands_mem);
    clReleaseMemObject(distance_mem);
    clReleaseMemObject(pheromone_mem);
    clReleaseMemObject(total_info_mem);
    clReleaseMemObject(nn_list_mem);
    clReleaseMemObject(seed_mem);
}

void g_ACO::init_aco()
{
    /*----------- (1) init parameters --------------*/
    instance.best_so_far_time = elapsed_time(VIRTUAL);
    
    /* Initialize variables concerning statistics etc. */
    instance.iteration   = 0;
    instance.best_so_far_ant->tour_length = INFINITY;
    
    instance.iter_stagnate_cnt = 0;
    instance.best_stagnate_cnt = 0;
    //!! g_ACO中需要对 iteration_best_ant分配一块空间
    instance.iteration_best_ant = &instance.ants[0];
    
    
    /*---------- (2) create memory objects ---------*/
    create_memory_objects();
    
    
    /*-------- (3.1) initialize the pheromone -------*/
    cl_int err_num;
    cl_kernel& pheromone_init = env.get_kernel(kernel_t::pheromone_init);
    float trail_0 = 0.5;
    
    // 1. set kernel arguments
    err_num = clSetKernelArg(pheromone_init, 0, sizeof(float), &trail_0);
    err_num |= clSetKernelArg(pheromone_init, 1, sizeof(cl_mem), &pheromone_mem);
    err_num |= clSetKernelArg(pheromone_init, 2, sizeof(cl_mem), &total_info_mem);
    err_num |= clSetKernelArg(pheromone_init, 3, sizeof(cl_mem), &distance_mem);
    check_error(err_num, CL_SUCCESS);
    
    size_t global_work_size[1] = {static_cast<size_t>(num_node * num_node)};
    size_t local_work_size[1] = {1};
    
    // 2. queue the kernel up for executeion
    err_num = clEnqueueNDRangeKernel(env.commandQueue, pheromone_init,
                                     1, NULL, global_work_size, local_work_size,
                                     0, NULL, NULL);
    check_error(err_num, CL_SUCCESS);
    
    
    /*-------- (3.2) initialize the pheromone -------*/
    // 第一次迭代用于设置一个合适的 pheromone init trail
    construct_solutions();
    if (instance.ls_flag) {
        local_search();
    }
    update_statistics();
    
    trail_0 =  1.0f / ((rho) * instance.best_so_far_ant->tour_length);
    // 1. set kernel arguments
    err_num = clSetKernelArg(pheromone_init, 0, sizeof(float), &trail_0);
    err_num |= clSetKernelArg(pheromone_init, 1, sizeof(cl_mem), &pheromone_mem);
    err_num |= clSetKernelArg(pheromone_init, 2, sizeof(cl_mem), &total_info_mem);
    err_num |= clSetKernelArg(pheromone_init, 3, sizeof(cl_mem), &distance_mem);
    check_error(err_num, CL_SUCCESS);
    
    // 2. queue the kernel up for executeion
    err_num = clEnqueueNDRangeKernel(env.commandQueue, pheromone_init,
                                     1, NULL, global_work_size, local_work_size,
                                     0, NULL, NULL);
    check_error(err_num, CL_SUCCESS);
    
    instance.iteration++;
    
}

void g_ACO::exit_aco(void)
{
    // copy best-so-far ant from opencl memory objects
    update_best_so_far_from_mem();
}

void g_ACO::run_aco_iteration()
{
    
    construct_solutions();
    
    if (instance.ls_flag) {
        local_search();
    }
    
    update_statistics();

    pheromone_update();
    
    if (sa_flag) {
        if ((instance.pid == 0 && instance.best_stagnate_cnt >= instance.num_node) ||
            (instance.pid != 0 && instance.best_stagnate_cnt >= 30))
        {
            // first, update the best-so-far ant from opencl memory objects
            update_best_so_far_from_mem();
            
            // run sa process
            SimulatedAnnealing *annealer = new SimulatedAnnealing(&instance, this, 5.0, 0.97, MAX(instance.num_node * 4, 250), 50);
            annealer->run();
            instance.best_stagnate_cnt = 0;
            
            delete annealer;
        }
    }
}

/*
 * solution construction phase
 */
void g_ACO::construct_solutions(void)
{
    cl_int err_num;
    const int grp_size = env.maxWorkGroupSize / 4;
//    const int num_grps = (ceil)(1.0 * n_ants / grp_size);
    
    size_t global_work_size[1] = {static_cast<size_t>(n_ants)};
    size_t local_work_size[1] = {1};
    cl_kernel& construct_solution = env.get_kernel(kernel_t::construct_solution);
    
    // 1. set kernel arguments
    err_num = clSetKernelArg(construct_solution, 0, sizeof(cl_int), &(instance.vehicle_capacity));
    err_num |= clSetKernelArg(construct_solution, 1, sizeof(float), &(instance.max_distance));
    err_num |= clSetKernelArg(construct_solution, 2, sizeof(float), &(instance.service_time));
    err_num |= clSetKernelArg(construct_solution, 3, sizeof(cl_mem), &seed_mem);
    err_num |= clSetKernelArg(construct_solution, 4, sizeof(cl_mem), &distance_mem);
    err_num |= clSetKernelArg(construct_solution, 5, sizeof(cl_mem), &demands_mem);
    err_num |= clSetKernelArg(construct_solution, 6, sizeof(cl_mem), &total_info_mem);
    err_num |= clSetKernelArg(construct_solution, 7, sizeof(cl_mem), &solutions_mem);
    err_num |= clSetKernelArg(construct_solution, 8, sizeof(cl_mem), &solution_lens_mem);
    check_error(err_num, CL_SUCCESS);
    
    // 2. queue the kernel up for executeion
    err_num = clEnqueueNDRangeKernel(env.commandQueue, construct_solution,
                                     1, NULL, global_work_size, local_work_size,
                                     0, NULL, NULL);
    check_error(err_num, CL_SUCCESS);
    
    /* ------ for debug ------- */
//    int *result = new int[max_tour_sz * n_ants];
//    err_num = clEnqueueReadBuffer(env.commandQueue, solutions_mem, CL_TRUE, 0, sizeof(int) * max_tour_sz * n_ants, result, 0, NULL, NULL);
//    check_error(err_num, CL_SUCCESS);
//
//    int i, j, beg, end;
//    AntStruct *ant;
//    
//    for (i = 0; i < n_ants; i++) {
//        ant = &instance.ants[i];
//        beg = i * max_tour_sz;
//        end = (i + 1) * max_tour_sz;
//        ant->tour_size = result[end - 1];
//        for (j = 0; j < ant->tour_size; j++) {
//            ant->tour[j] = result[j + beg];
//        }
//        ant->tour_length = compute_tour_length(&instance, ant->tour, ant->tour_size);
//        DEBUG(assert(check_solution(&instance, ant->tour, ant->tour_size));)
//        print_solution(&instance, ant->tour, ant->tour_size);
//    }
//    
//    delete[] result;
}

/*
 * local search phase
 */
void g_ACO::local_search(void)
{
    cl_int err_num;
    cl_kernel& local_search = env.get_kernel(kernel_t::local_search);
    
    // 1. set kernel arguments
    err_num = clSetKernelArg(local_search, 0, sizeof(cl_mem), &seed_mem);
    err_num |= clSetKernelArg(local_search, 1, sizeof(cl_int), &instance.nn_ls);
    err_num |= clSetKernelArg(local_search, 2, sizeof(cl_mem), &nn_list_mem);
    err_num |= clSetKernelArg(local_search, 3, sizeof(cl_mem), &distance_mem);
    err_num |= clSetKernelArg(local_search, 4, sizeof(cl_mem), &solutions_mem);
    err_num |= clSetKernelArg(local_search, 5, sizeof(cl_mem), &solution_lens_mem);
    err_num |= clSetKernelArg(local_search, 6, sizeof(cl_mem), &demands_mem);
    err_num |= clSetKernelArg(local_search, 7, sizeof(cl_int), &instance.vehicle_capacity);
    err_num |= clSetKernelArg(local_search, 8, sizeof(float), &instance.max_distance);
    err_num |= clSetKernelArg(local_search, 9, sizeof(float), &instance.service_time);
    check_error(err_num, CL_SUCCESS);
    
    size_t global_work_size[1] = {static_cast<size_t>(n_ants)};
    size_t local_work_size[1] = {1};
    
    // 2. queue the kernel up for executeion
    err_num = clEnqueueNDRangeKernel(env.commandQueue, local_search,
                                     1, NULL, global_work_size, local_work_size,
                                     0, NULL, NULL);
    check_error(err_num, CL_SUCCESS);
    
    /* ------ for debug ------- */
//    // 3. get ant solutions from result
//    int *result = new int[max_tour_sz * n_ants];
//    err_num = clEnqueueReadBuffer(env.commandQueue, solutions_mem, CL_TRUE, 0, sizeof(int) * max_tour_sz * n_ants, result, 0, NULL, NULL);
//    check_error(err_num, CL_SUCCESS);
//    
//    int i, j, beg, end;
//    AntStruct *ant;
//    
//    printf("\n---- after alcal search -----\n");
//    for (i = 0; i < n_ants; i++) {
//        ant = &instance.ants[i];
//        beg = i * max_tour_sz;
//        end = (i + 1) * max_tour_sz;
//        ant->tour_size = result[end - 1];
//        for (j = 0; j < ant->tour_size; j++) {
//            ant->tour[j] = result[j + beg];
//        }
//        ant->tour_length = compute_tour_length(&instance, ant->tour, ant->tour_size);
//        DEBUG(assert(check_solution(&instance, ant->tour, ant->tour_size));)
//        print_solution(&instance, ant->tour, ant->tour_size);
//    }
//    
//    delete[] result;
}

/*
 * pheromone update phase
 */
void g_ACO::pheromone_update(void)
{
    /*--- (a) pheromone evaporation ---*/
    pheromone_evaporation();
    
    /*--- (b) pheromone update ---*/
    
    // kernel 中已经实现扰动
    if (instance.iter_stagnate_cnt >= 5) {
        printf("pid %d start pheromone disturbance: iter %d, best_stagnate %d, iter_stagnate %d\n",
               instance.pid, instance.iteration, instance.best_stagnate_cnt, instance.iter_stagnate_cnt);
        
        pheromone_disturbance();
        instance.iter_stagnate_cnt -= 2;
    } else {
        ras_update();
    }
    
    /*--- (c) compute total info ---*/
    compute_total_info();
}


void g_ACO::pheromone_evaporation(void)
{
    cl_int err_num;
    size_t global_work_size[1] = {static_cast<size_t>(num_node * num_node)};
    size_t local_work_size[1] = {1};
    
    
    cl_kernel& pheromone_evaporation = env.get_kernel(kernel_t::pheromone_evaporation);
    // 1. set kernel arguments
    err_num = clSetKernelArg(pheromone_evaporation, 0, sizeof(cl_mem), &pheromone_mem);
    check_error(err_num, CL_SUCCESS);
    
    // 2. queue the kernel up for executeion
    err_num = clEnqueueNDRangeKernel(env.commandQueue, pheromone_evaporation,
                                     1, NULL, global_work_size, local_work_size,
                                     0, NULL, NULL);
    check_error(err_num, CL_SUCCESS);
}


void g_ACO::ras_update(void)
{
    cl_int err_num;
    size_t global_work_size[1] = {1};
    size_t local_work_size[1] = {1};
    

    cl_kernel& ras_update = env.get_kernel(kernel_t::ras_update);
    // 1. set kernel arguments
    err_num = clSetKernelArg(ras_update, 0, sizeof(cl_mem), &solutions_mem);
    err_num |= clSetKernelArg(ras_update, 1, sizeof(cl_mem), &solution_lens_mem);
    err_num |= clSetKernelArg(ras_update, 2, sizeof(cl_mem), &pheromone_mem);
    check_error(err_num, CL_SUCCESS);
    
    // 2. queue the kernel up for executeion
    err_num = clEnqueueNDRangeKernel(env.commandQueue, ras_update,
                                     1, NULL, global_work_size, local_work_size,
                                     0, NULL, NULL);
    check_error(err_num, CL_SUCCESS);
}


void g_ACO::pheromone_disturbance(void)
{
    cl_int err_num;
    size_t global_work_size[1] = {1};
    size_t local_work_size[1] = {1};
    
    
    cl_kernel& pheromone_disturbance = env.get_kernel(kernel_t::pheromone_disturbance);
    // 1. set kernel arguments
    err_num = clSetKernelArg(pheromone_disturbance, 0, sizeof(cl_mem), &pheromone_mem);
    check_error(err_num, CL_SUCCESS);
    
    // 2. queue the kernel up for executeion
    err_num = clEnqueueNDRangeKernel(env.commandQueue, pheromone_disturbance,
                                     1, NULL, global_work_size, local_work_size,
                                     0, NULL, NULL);
    check_error(err_num, CL_SUCCESS);
}

void g_ACO::compute_total_info(void)
{
    cl_int err_num;
    size_t global_work_size[1] = {static_cast<size_t>(num_node * num_node)};
    size_t local_work_size[1] = {1};
    
    
    cl_kernel& compute_total_info = env.get_kernel(kernel_t::compute_total_info);
    // 1. set kernel arguments
    err_num = clSetKernelArg(compute_total_info, 0, sizeof(cl_mem), &pheromone_mem);
    err_num |= clSetKernelArg(compute_total_info, 1, sizeof(cl_mem), &total_info_mem);
    err_num |= clSetKernelArg(compute_total_info, 2, sizeof(cl_mem), &distance_mem);
    check_error(err_num, CL_SUCCESS);
    
    // 2. queue the kernel up for executeion
    err_num = clEnqueueNDRangeKernel(env.commandQueue, compute_total_info,
                                     1, NULL, global_work_size, local_work_size,
                                     0, NULL, NULL);
    check_error(err_num, CL_SUCCESS);
}

/*
 * update pheromone with weight
 * number of threads: 1
 */
void g_ACO::update_pheromone_weighted(AntStruct *a, int weight)
{
    cl_int err_num;
    size_t global_work_size[1] = {1};
    size_t local_work_size[1] = {1};

    cl_mem tour_mem = clCreateBuffer(env.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(int) * a->tour_size, a->tour, &err_num);
    check_error(err_num, CL_SUCCESS);
    
    cl_kernel& update_pheromone_weighted = env.get_kernel(kernel_t::update_pheromone_weighted);
    // 1. set kernel arguments
    err_num = clSetKernelArg(update_pheromone_weighted, 0, sizeof(cl_mem), &pheromone_mem);
    err_num |= clSetKernelArg(update_pheromone_weighted, 1, sizeof(cl_mem), &tour_mem);
    err_num |= clSetKernelArg(update_pheromone_weighted, 2, sizeof(cl_int), &a->tour_size);
    err_num |= clSetKernelArg(update_pheromone_weighted, 3, sizeof(float), &a->tour_length);
    err_num |= clSetKernelArg(update_pheromone_weighted, 4, sizeof(cl_int), &weight);
    check_error(err_num, CL_SUCCESS);
    
    // 2. queue the kernel up for executeion
    err_num = clEnqueueNDRangeKernel(env.commandQueue, update_pheromone_weighted,
                                     1, NULL, global_work_size, local_work_size,
                                     0, NULL, NULL);
    check_error(err_num, CL_SUCCESS);
    
    clReleaseMemObject(tour_mem);
}

/*
 * update statistics
 * number of threads: 1
 */
void g_ACO::update_statistics(void)
{
    cl_int err_num;
    size_t global_work_size[1] ={1};
    size_t local_work_size[1] = {1};
    
    
    cl_kernel& update_statistics = env.get_kernel(kernel_t::update_statistics);
    // 1. set kernel arguments
    err_num = clSetKernelArg(update_statistics, 0, sizeof(cl_mem), &solutions_mem);
    err_num |= clSetKernelArg(update_statistics, 1, sizeof(cl_mem), &solution_lens_mem);
    check_error(err_num, CL_SUCCESS);
    
    // 2. queue the kernel up for executeion
    err_num = clEnqueueNDRangeKernel(env.commandQueue, update_statistics,
                                     1, NULL, global_work_size, local_work_size,
                                     0, NULL, NULL);
    check_error(err_num, CL_SUCCESS);

    // 3. get iter-best solution from buffer
    // get iter-best solution id
    int iter_best_id;
    err_num = clEnqueueReadBuffer(env.commandQueue, solutions_mem, CL_TRUE,
                                   sizeof(int) * max_tour_sz * (n_ants + 1),
                                   sizeof(int) * 1, &iter_best_id, 0, NULL, NULL);
    //debug
//    int *result = new int[max_tour_sz];
//    err_num |= clEnqueueReadBuffer(env.commandQueue, solutions_mem, CL_TRUE,
//                                  sizeof(int) * max_tour_sz * iter_best_id,
//                                  sizeof(int) * max_tour_sz, result, 0, NULL, NULL);
////    print_solution(&instance, result, result[max_tour_sz-1]);
    
    float iter_best_len;
    err_num |= clEnqueueReadBuffer(env.commandQueue, solution_lens_mem, CL_TRUE,
                                  sizeof(float) * iter_best_id,
                                  sizeof(float) * 1, &iter_best_len, 0, NULL, NULL);
    check_error(err_num, CL_SUCCESS);
    
    // 4. update statistics
    float best_so_far_len = instance.best_so_far_ant->tour_length;
    instance.iteration_best_ant->tour_length = iter_best_len;
    if (instance.pid == 0) {
        write_iter_report(&instance);
    }
    
    if (iter_best_len - best_so_far_len < -EPSILON) {
        // better solution found
        instance.best_stagnate_cnt = 0;
        
        instance.best_so_far_time = elapsed_time( VIRTUAL );
        // !!only copy tour length
        instance.best_so_far_ant->tour_length = iter_best_len;
        
        instance.best_solution_iter = instance.iteration;
        if (instance.pid == 0) {
            write_best_so_far_report(&instance);
        }
    } else {
        instance.best_stagnate_cnt++;
        if (fabs(instance.last_iter_solution - iter_best_len) < EPSILON) {
            instance.iter_stagnate_cnt++;
        } else {
            instance.iter_stagnate_cnt = 0;
        }
    }
    
    instance.last_iter_solution = iter_best_len;
    
//    delete[] result;
}


/*
 * create memory objects
 */
void g_ACO::create_memory_objects(void)
{
    int i, j, k;
    cl_context context = env.context;
    cl_int err_num;
    
    /*
     * solutions memory object: 
     * all ants solution（id: [0, n_ants-1] + best-so-far solution (id: n_ants) + iter-best solution (id: n_ants+1)
     * 注意：最后的 iter-best solution 只存本次迭代最优解的所在下标id: [0, n_ants-1], 因此只需要sizeof(int) * 1 的内存
     * 其余每个解分配内存: sizeof(int) * max_tour_sz
     */
    solutions_mem = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(int) * (max_tour_sz * (n_ants + 1) + 1), NULL, &err_num);
    check_error(err_num, CL_SUCCESS);
    
    /*
     * solution lens memory object: lens of all ants solutions + best-so-far solution
     * 注意： 需要初始化 best-so-far solution length
     */
    solution_lens_mem = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                       sizeof(float) * (n_ants + 1), NULL, &err_num);
    float best_len = INFINITY;
    err_num |= clEnqueueWriteBuffer(env.commandQueue, solution_lens_mem, CL_TRUE,
                                    sizeof(float) * n_ants, sizeof(float) * 1, &best_len, 0, NULL, NULL);
    check_error(err_num, CL_SUCCESS);
    
    // demand memory object
    int *tmp_demands = new int[num_node];
    for (i = 0; i < num_node; i++) {
        tmp_demands[i] = instance.nodeptr[i].demand;
    }
    demands_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(int) * num_node, tmp_demands, &err_num);
    check_error(err_num, CL_SUCCESS);
    
    // distance 2d-martix trans to 1d-array memory object
    k = 0;
    float *tmp_distance = new float[num_node * num_node];
    for (i = 0; i < num_node; i++) {
        for (j = 0; j < num_node; j++) {
            tmp_distance[k++] = instance.distance[i][j];
        }
    }
    distance_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(float) * num_node * num_node, tmp_distance, &err_num);
    check_error(err_num, CL_SUCCESS);
    
    // pheromone memory object
    pheromone_mem = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * num_node * num_node, NULL, &err_num);
    check_error(err_num, CL_SUCCESS);
    
    // total_info memory object
    total_info_mem = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(float) * num_node * num_node, NULL, &err_num);
    check_error(err_num, CL_SUCCESS);
    
    // nn_list memory object
    k = 0;
    int *tmp_nn_list = new int[num_node * instance.nn_ls];
    for (i = 0; i < num_node; i++) {
        for (j = 0; j < instance.nn_ls; j++) {
            tmp_nn_list[k++] = instance.nn_list[i][j];
        }
    }
    nn_list_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(int) * num_node * instance.nn_ls, tmp_nn_list, &err_num);
    check_error(err_num, CL_SUCCESS);
    
    // seed memory
    int *tmp_seed = new int[n_ants];
    for (i = 0; i < n_ants; i++) {
        tmp_seed[i] = (int)time(NULL) + i * (i + ran01(&instance.rnd_seed)) * n_ants;
    }
    seed_mem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                 sizeof(int) * n_ants, tmp_seed, &err_num);
    check_error(err_num, CL_SUCCESS);
    
    delete[] tmp_demands;
    delete[] tmp_distance;
    delete[] tmp_seed;
    delete[] tmp_nn_list;
}


/*
 * update host best-so-far solution from device memory
 */
void g_ACO::update_best_so_far_from_mem(void)
{
    cl_int err_num;
    int *result = new int[max_tour_sz];
    err_num = clEnqueueReadBuffer(env.commandQueue, solutions_mem, CL_TRUE,
                                  sizeof(int) * max_tour_sz * n_ants,
                                  sizeof(int) * max_tour_sz, result, 0, NULL, NULL);
    check_error(err_num, CL_SUCCESS);
    
    float best_len;
    err_num = clEnqueueReadBuffer(env.commandQueue, solution_lens_mem, CL_TRUE,
                                  sizeof(float) * n_ants,
                                  sizeof(float) * 1, &best_len, 0, NULL, NULL);
    check_error(err_num, CL_SUCCESS);
    
    // update best-so-far-ant
    AntStruct *ant = instance.best_so_far_ant;
    ant->tour_size = result[max_tour_sz-1];
    DEBUG(assert(fabs(best_len - ant->tour_length) < EPSILON);)
    ant->tour_length = best_len;
    for (int i = 0; i < ant->tour_size; i++) {
        ant->tour[i] = result[i];
    }
    
    delete[] result;
}

/*
 * update host best-so-far solution to device memory
 * number of threads: 1
 */
void g_ACO::update_best_so_far_to_mem(void)
{
    cl_int err_num;
    size_t global_work_size[1] ={1};
    size_t local_work_size[1] = {1};
    
    AntStruct *best_so_far = instance.best_so_far_ant;
    
    cl_mem tour_mem = clCreateBuffer(env.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     sizeof(int) * best_so_far->tour_size, best_so_far->tour, &err_num);
    check_error(err_num, CL_SUCCESS);
    
    cl_kernel& update_best_so_far_to_mem = env.get_kernel(kernel_t::update_best_so_far_to_mem);
    // 1. set kernel arguments
    err_num = clSetKernelArg(update_best_so_far_to_mem, 0, sizeof(cl_mem), &tour_mem);
    err_num |= clSetKernelArg(update_best_so_far_to_mem, 1, sizeof(cl_int), &best_so_far->tour_size);
    err_num |= clSetKernelArg(update_best_so_far_to_mem, 2, sizeof(float), &best_so_far->tour_length);
    err_num |= clSetKernelArg(update_best_so_far_to_mem, 3, sizeof(cl_mem), &solutions_mem);
    err_num |= clSetKernelArg(update_best_so_far_to_mem, 4, sizeof(cl_mem), &solution_lens_mem);
    check_error(err_num, CL_SUCCESS);
    
    // 2. queue the kernel up for executeion
    err_num = clEnqueueNDRangeKernel(env.commandQueue, update_best_so_far_to_mem,
                                     1, NULL, global_work_size, local_work_size,
                                     0, NULL, NULL);
    check_error(err_num, CL_SUCCESS);
    
    clReleaseMemObject(tour_mem);
}

/*
 FUNCTION:       copy solution from ant a1 into ant a2
 */
void g_ACO::copy_solution_from_to(AntStruct *a1, AntStruct *a2)
{
    int   i;
    
    a2->tour_length = a1->tour_length;
    a2->tour_size = a1->tour_size;
    for ( i = 0 ; i < a1->tour_size ; i++ ) {
        a2->tour[i] = a1->tour[i];
    }
}

/*
 * Check for error condition and exit if found.  Print file and line number
 * of error. (from NVIDIA SDK)
 */
void check_error_file_line(int err_num, int expected, const char* file, const int line_number)
{
    if (err_num != expected)
    {
        cerr << "Line " << line_number << " in File " << file << endl;
        exit(1);
    }
}

