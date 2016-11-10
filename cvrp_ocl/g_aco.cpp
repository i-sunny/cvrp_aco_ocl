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
#include "g_aco.h"
#include "cpu/utilities.h"
#include "problem.h"
#include "vrpHelper.h"
#include "cpu/io.h"
#include "cpu/timer.h"

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
    
    create_memory_objects();
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

void g_ACO::run_aco_iteration()
{
    construct_solutions();
    if (instance.ls_flag) {
        local_search();
    }
    update_statistics();
    
    pheromone_update();
}

/*
 * solution construction phase
 */
void g_ACO::construct_solutions(void)
{
    cl_int err_num;
    cl_kernel& construct_solution = env.get_kernel(kernel_t::construct_solution);
    
    // 1. set kernel arguments
    err_num = clSetKernelArg(construct_solution, 0, sizeof(int), &num_node);
    err_num |= clSetKernelArg(construct_solution, 1, sizeof(int), &(instance.vehicle_capacity));
    err_num |= clSetKernelArg(construct_solution, 2, sizeof(double), &(instance.max_distance));
    err_num |= clSetKernelArg(construct_solution, 3, sizeof(double), &(instance.service_time));
    err_num |= clSetKernelArg(construct_solution, 4, sizeof(cl_mem), &seed_mem);
    err_num |= clSetKernelArg(construct_solution, 5, sizeof(cl_mem), &distance_mem);
    err_num |= clSetKernelArg(construct_solution, 6, sizeof(cl_mem), &demands_mem);
    err_num |= clSetKernelArg(construct_solution, 7, sizeof(cl_mem), &total_info_mem);
    err_num |= clSetKernelArg(construct_solution, 8, sizeof(cl_mem), &solutions_mem);
    err_num |= clSetKernelArg(construct_solution, 9, sizeof(cl_mem), &solution_lens_mem);
    check_error(err_num, CL_SUCCESS);
    
    size_t global_work_size[1] = {static_cast<size_t>(n_ants)};
    size_t local_work_size[1] = {1};
    
    // 2. queue the kernel up for executeion
    err_num = clEnqueueNDRangeKernel(env.commandQueue, construct_solution,
                                     1, NULL, global_work_size, local_work_size,
                                     0, NULL, NULL);
    check_error(err_num, CL_SUCCESS);
    int *result = new int[max_tour_sz * n_ants];
    err_num = clEnqueueReadBuffer(env.commandQueue, solutions_mem, CL_TRUE, 0, sizeof(int) * max_tour_sz * n_ants, result, 0, NULL, NULL);
    check_error(err_num, CL_SUCCESS);
    
    //--- debug ---
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
    //--- debug ---
}

/*
 * local search phase
 */
void g_ACO::local_search(void)
{
    cl_int err_num;
    cl_kernel& local_search = env.get_kernel(kernel_t::local_search);
    
    // 1. set kernel arguments
    err_num = clSetKernelArg(local_search, 0, sizeof(int), &num_node);
    err_num |= clSetKernelArg(local_search, 1, sizeof(cl_mem), &seed_mem);
    err_num |= clSetKernelArg(local_search, 2, sizeof(int), &instance.nn_ls);
    err_num |= clSetKernelArg(local_search, 3, sizeof(cl_mem), &nn_list_mem);
    err_num |= clSetKernelArg(local_search, 4, sizeof(cl_mem), &distance_mem);
    err_num |= clSetKernelArg(local_search, 5, sizeof(cl_mem), &solutions_mem);
    err_num |= clSetKernelArg(local_search, 6, sizeof(cl_mem), &solution_lens_mem);
    check_error(err_num, CL_SUCCESS);
    
    size_t global_work_size[1] = {static_cast<size_t>(n_ants)};
    size_t local_work_size[1] = {1};
    
    // 2. queue the kernel up for executeion
    err_num = clEnqueueNDRangeKernel(env.commandQueue, local_search,
                                     1, NULL, global_work_size, local_work_size,
                                     0, NULL, NULL);
    check_error(err_num, CL_SUCCESS);
    
    int *result = new int[max_tour_sz * n_ants];
    err_num = clEnqueueReadBuffer(env.commandQueue, solutions_mem, CL_TRUE, 0, sizeof(int) * max_tour_sz * n_ants, result, 0, NULL, NULL);
    check_error(err_num, CL_SUCCESS);
    
    // 3. get ant solutions from result
//    int i, j, beg, end;
//    AntStruct *ant;
//    
//    //--- debug ---
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
    //--- debug ---
}

/*
 * pheromone update phase
 */
void g_ACO::pheromone_update(void)
{
    cl_int err_num;
    size_t global_work_size[1];
    size_t local_work_size[1] = {1};
    
    /*--- (a) pheromone evaporation ---*/
    cl_kernel& pheromone_evaporation = env.get_kernel(kernel_t::pheromone_evaporation);
    // 1. set kernel arguments
    err_num = clSetKernelArg(pheromone_evaporation, 0, sizeof(cl_mem), &pheromone_mem);
    check_error(err_num, CL_SUCCESS);
    
    // 2. queue the kernel up for executeion
    global_work_size[0] = num_node * num_node;
    err_num = clEnqueueNDRangeKernel(env.commandQueue, pheromone_evaporation,
                                     1, NULL, global_work_size, local_work_size,
                                     0, NULL, NULL);
    check_error(err_num, CL_SUCCESS);
    
    
    /*--- (b) pheromone update ---*/
    cl_kernel& pheromone_update = env.get_kernel(kernel_t::pheromone_update);
    // 1. set kernel arguments
    err_num = clSetKernelArg(pheromone_update, 0, sizeof(int), &num_node);
    err_num |= clSetKernelArg(pheromone_update, 1, sizeof(int), &n_ants);
    err_num |= clSetKernelArg(pheromone_update, 2, sizeof(cl_mem), &solutions_mem);
    err_num |= clSetKernelArg(pheromone_update, 3, sizeof(cl_mem), &solution_lens_mem);
    err_num |= clSetKernelArg(pheromone_update, 4, sizeof(cl_mem), &pheromone_mem);
    err_num |= clSetKernelArg(pheromone_update, 5, sizeof(int), &instance.iter_stagnate_cnt);
    check_error(err_num, CL_SUCCESS);
    
    // 2. queue the kernel up for executeion
    global_work_size[0] = 1;
    err_num = clEnqueueNDRangeKernel(env.commandQueue, pheromone_update,
                                     1, NULL, global_work_size, local_work_size,
                                     0, NULL, NULL);
    check_error(err_num, CL_SUCCESS);
    
    // kernel 中已经实现扰动
    if (instance.iter_stagnate_cnt >= 5) {
        instance.iter_stagnate_cnt -= 2;
    }
    
    /*--- (c) compute total info ---*/
    cl_kernel& compute_total_info = env.get_kernel(kernel_t::compute_total_info);
    // 1. set kernel arguments
    err_num = clSetKernelArg(compute_total_info, 0, sizeof(cl_mem), &pheromone_mem);
    err_num |= clSetKernelArg(compute_total_info, 1, sizeof(cl_mem), &total_info_mem);
    err_num |= clSetKernelArg(compute_total_info, 2, sizeof(cl_mem), &distance_mem);
    check_error(err_num, CL_SUCCESS);
    
    // 2. queue the kernel up for executeion
    global_work_size[0] = num_node * num_node;
    err_num = clEnqueueNDRangeKernel(env.commandQueue, compute_total_info,
                                     1, NULL, global_work_size, local_work_size,
                                     0, NULL, NULL);
    check_error(err_num, CL_SUCCESS);
}

/*
 * update statistics
 */
void g_ACO::update_statistics(void)
{
    cl_int err_num;
    size_t global_work_size[1] ={1};
    size_t local_work_size[1] = {1};
    
    
    cl_kernel& update_statistics = env.get_kernel(kernel_t::update_statistics);
    // 1. set kernel arguments
    err_num = clSetKernelArg(update_statistics, 0, sizeof(int), &num_node);
    err_num |= clSetKernelArg(update_statistics, 1, sizeof(int), &n_ants);
    err_num |= clSetKernelArg(update_statistics, 2, sizeof(cl_mem), &solutions_mem);
    err_num |= clSetKernelArg(update_statistics, 3, sizeof(cl_mem), &solution_lens_mem);
    check_error(err_num, CL_SUCCESS);
    
    // 2. queue the kernel up for executeion
    err_num = clEnqueueNDRangeKernel(env.commandQueue, update_statistics,
                                     1, NULL, global_work_size, local_work_size,
                                     0, NULL, NULL);
    check_error(err_num, CL_SUCCESS);

    // 3. get last solutions: iter-best solution from buffer
    int *result = new int[max_tour_sz];
    err_num = clEnqueueReadBuffer(env.commandQueue, solutions_mem, CL_TRUE,
                                  sizeof(int) * max_tour_sz * (n_ants + 1),
                                  sizeof(int) * max_tour_sz, result, 0, NULL, NULL);
    check_error(err_num, CL_SUCCESS);
    
    double iter_best_len;
    err_num = clEnqueueReadBuffer(env.commandQueue, solution_lens_mem, CL_TRUE,
                                  sizeof(double) * (n_ants + 1),
                                  sizeof(double) * 1, &iter_best_len, 0, NULL, NULL);
    check_error(err_num, CL_SUCCESS);
    
    
    // 4. update statistics
    double best_so_far_len = instance.best_so_far_ant->tour_length;
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
        if (fabs(instance.last_iter_solution - instance.iteration_best_ant->tour_length) < EPSILON) {
            instance.iter_stagnate_cnt++;
        } else {
            instance.iter_stagnate_cnt = 0;
        }
    }
    
    instance.last_iter_solution = iter_best_len;
}


/*
 * create memory objects
 */
void g_ACO::create_memory_objects(void)
{
    int i, j, k;
    cl_context context = env.context;
    cl_int err_num;
    
    // solutions memory object: all ants solution + best-so-far solution + iter-best solution
    solutions_mem = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(int) * max_tour_sz * (n_ants + 2), NULL, &err_num);
    check_error(err_num, CL_SUCCESS);
    
    // solution lens memory object: lens of all ants solutions + best-so-far solution + iter-best solution
    double *tmp_lens = new double[n_ants + 2];
    tmp_lens[n_ants] = instance.best_so_far_ant->tour_length;
    solution_lens_mem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                       sizeof(double) * (n_ants + 2), tmp_lens, &err_num);
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
    double *tmp_distance = new double[num_node * num_node];
    for (i = 0; i < num_node; i++) {
        for (j = 0; j < num_node; j++) {
            tmp_distance[k++] = instance.distance[i][j];
        }
    }
    distance_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(double) * num_node * num_node, tmp_distance, &err_num);
    check_error(err_num, CL_SUCCESS);
    
    // pheromone memory object
    k = 0;
    double *tmp_pheromone = new double[num_node * num_node];
    for (i = 0; i < num_node; i++) {
        for (j = 0; j < num_node; j++) {
            tmp_pheromone[k++] = instance.pheromone[i][j];
        }
    }
    pheromone_mem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   sizeof(double) * num_node * num_node, tmp_pheromone, &err_num);
    check_error(err_num, CL_SUCCESS);
    
    // total_info memory object
    k = 0;
    double *tmp_total_info = new double[num_node * num_node];
    for (i = 0; i < num_node; i++) {
        for (j = 0; j < num_node; j++) {
            tmp_total_info[k++] = instance.total_info[i][j];
        }
    }
    total_info_mem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                    sizeof(double) * num_node * num_node, tmp_total_info, &err_num);
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
    

    delete[] tmp_lens;
    delete[] tmp_demands;
    delete[] tmp_pheromone;
    delete[] tmp_total_info;
    delete[] tmp_distance;
    delete[] tmp_seed;
    delete[] tmp_nn_list;
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

