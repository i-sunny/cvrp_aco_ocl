/*********************************
 
 OCL-ACO: ACO for solving CVRP using OpenCL
 
 Created by 孙晓奇 on 2016/11/11.
 Copyright © 2016年 xiaoqi.sxq. All rights reserved.
 
 Program's name: cvrp_ocl
 Purpose: implementation of procedures for ants' behaviour
 
 email: sunxq1991@gmail.com
 
 *********************************/

#include <iostream>
#include <math.h>
#include <assert.h>
#include <limits.h>

#include "g_aco.h"
#include "../utilities.h"
#include "../vrpHelper.h"
#include "../io.h"
#include "../timer.h"

using namespace std;

#define PROFILE(x)   x

#define check_error(a, b) check_error_file_line(a, b, __FILE__ , __LINE__)

void check_error_file_line(int err_num, int expected, const char* file, const int line_number);


g_ACO::g_ACO(OpenclEnv &env, Problem &instance): env(env), instance(instance)
{
    num_node = instance.num_node;
    n_ants = instance.n_ants;
    max_tour_sz = 2 * num_node;
    
    cs_grp_size = env.maxWorkGroupSize / 2;
    cs_num_grps = n_ants;
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
    clReleaseMemObject(bsf_records_mem);
    clReleaseMemObject(num_bsf_mem);
    clReleaseMemObject(elite_ids_mem);
    clReleaseMemObject(best_result_val);
    clReleaseMemObject(best_result_idx);
    clReleaseMemObject(update_flag_mem);
}

void g_ACO::init_aco()
{
    /* 1. init parameters */
    instance.best_so_far_time = elapsed_time(REAL);
    
    /* Initialize variables concerning statistics etc. */
    instance.iteration   = 0;
    instance.best_so_far_ant->tour_length = INFINITY;
    
    instance.iter_stagnate_cnt = 0;
    instance.best_stagnate_cnt = 0;
    
    /* 2. create memory objects */
    create_memory_objects();
    
    /* 3. initialize the pheromone */
    float trail_0 = 0.5;
    pheromone_init(trail_0);
    
    // 第一次迭代用于设置一个合适的 pheromone init trail
    construct_solutions();
//    if (instance.ls_flag) {
//        local_search();
//    }
    update_statistics();
    update_best_so_far_from_device();
    trail_0 =  1.0f / ((rho) * instance.best_so_far_ant->tour_length);
    pheromone_init(trail_0);
    
    instance.iteration++;
}

void g_ACO::exit_aco(void)
{
    // copy best-so-far ant from opencl memory objects
    update_best_so_far_from_device();
    
    // get best-so-far records
    cl_int err_num;
    // first, get the number of records
    int num_bsf;
    err_num = clEnqueueReadBuffer(env.commandQueue, num_bsf_mem, CL_TRUE, 0,
                                  sizeof(int) * 1, &num_bsf, 0, NULL, NULL);
    // then, get the records
    BestSolutionInfo *records = new BestSolutionInfo[num_bsf];
    err_num |= clEnqueueReadBuffer(env.commandQueue, bsf_records_mem, CL_TRUE, 0,
                                   sizeof(BestSolutionInfo) * num_bsf, records, 0, NULL, NULL);
    check_error(err_num, CL_SUCCESS);
    for (int i = 0; i < num_bsf; i++) {
        write_best_report(records[i]);
    }
    instance.best_so_far_time = records[num_bsf-1].time;
    
    free(records);
}

/*
 * aco iteration
 */
void g_ACO::run_aco_iteration()
{
    
    construct_solutions();
    
//    if (instance.ls_flag) {
//        local_search();
//    }
    
    update_statistics();

    pheromone_update();
}

/*
 * solution construction phase
 * 采用data parallelism
 * 参考文献: Enhancing data parallelism for Ant Colony Optimization on GPUs
 */
void g_ACO::construct_solutions(void)
{
    cl_int err_num;
    cl_event event;
    
    // global work size must be divisable by the local work size
    size_t global_work_size[1] = {static_cast<size_t>(cs_grp_size * cs_num_grps)};
    size_t local_work_size[1] = {static_cast<size_t>(cs_grp_size)};
    
    cl_kernel& construct_solution = env.get_kernel(kernel_t::construct_solution);
    
    err_num = clSetKernelArg(construct_solution, 0, sizeof(int), &(instance.vehicle_capacity));
    err_num |= clSetKernelArg(construct_solution, 1, sizeof(float), &(instance.max_distance));
    err_num |= clSetKernelArg(construct_solution, 2, sizeof(float), &(instance.service_time));
    err_num |= clSetKernelArg(construct_solution, 3, sizeof(cl_mem), &seed_mem);
    err_num |= clSetKernelArg(construct_solution, 4, sizeof(cl_mem), &distance_mem);
    err_num |= clSetKernelArg(construct_solution, 5, sizeof(cl_mem), &demands_mem);
    err_num |= clSetKernelArg(construct_solution, 6, sizeof(cl_mem), &total_info_mem);
    err_num |= clSetKernelArg(construct_solution, 7, sizeof(cl_mem), &solutions_mem);
    err_num |= clSetKernelArg(construct_solution, 8, sizeof(cl_mem), &solution_lens_mem);
    err_num |= clSetKernelArg(construct_solution, 9, sizeof(float) * num_node, NULL);       // local memory for a colomn distance[*][0]
    err_num |= clSetKernelArg(construct_solution, 10, sizeof(int) * num_node, NULL);        // local memory for demands
    err_num |= clSetKernelArg(construct_solution, 11, sizeof(float) * num_node, NULL);      // local memory for a colomn distance[current][*]
    err_num |= clSetKernelArg(construct_solution, 12, sizeof(float) * num_node, NULL);      // local memory for a colomn total_info[current][*]
    err_num |= clSetKernelArg(construct_solution, 13, sizeof(bool) * num_node, NULL);       // local memory for visited
    err_num |= clSetKernelArg(construct_solution, 14, sizeof(bool) * num_node, NULL);       // local memory for candidate
    err_num |= clSetKernelArg(construct_solution, 15, sizeof(float) * num_node, NULL);      // local memory for prob_selection
    err_num |= clSetKernelArg(construct_solution, 16, sizeof(int) * num_node, NULL);        // local memory for scratch_idx
    check_error(err_num, CL_SUCCESS);

    err_num = clEnqueueNDRangeKernel(env.commandQueue, construct_solution,
                                     1, NULL, global_work_size, local_work_size,
                                     0, NULL, &event);
    check_error(err_num, CL_SUCCESS);
    
    PROFILE(printf("\n[construct solutions] %f ms\n", event_runtime(event));)
}

/*
 * local search phase
 */
void g_ACO::local_search(void)
{
    cl_int err_num;
    cl_kernel& local_search = env.get_kernel(kernel_t::local_search);
    
    err_num = clSetKernelArg(local_search, 0, sizeof(cl_int), &instance.nn_ls);
    err_num |= clSetKernelArg(local_search, 1, sizeof(cl_mem), &nn_list_mem);
    err_num |= clSetKernelArg(local_search, 2, sizeof(cl_mem), &distance_mem);
    err_num |= clSetKernelArg(local_search, 3, sizeof(cl_mem), &solutions_mem);
    err_num |= clSetKernelArg(local_search, 4, sizeof(cl_mem), &solution_lens_mem);
    err_num |= clSetKernelArg(local_search, 5, sizeof(cl_mem), &demands_mem);
    err_num |= clSetKernelArg(local_search, 6, sizeof(cl_int), &instance.vehicle_capacity);
    err_num |= clSetKernelArg(local_search, 7, sizeof(float), &instance.max_distance);
    err_num |= clSetKernelArg(local_search, 8, sizeof(float), &instance.service_time);
    check_error(err_num, CL_SUCCESS);
    
    size_t global_work_size[1] = {static_cast<size_t>(n_ants)};
    size_t local_work_size[1] = {1};

    err_num = clEnqueueNDRangeKernel(env.commandQueue, local_search,
                                     1, NULL, global_work_size, local_work_size,
                                     0, NULL, NULL);
    check_error(err_num, CL_SUCCESS);
    
    /* ------ debug check solutions ------- */
//    int *result = new int[max_tour_sz * n_ants];
//    err_num = clEnqueueReadBuffer(env.commandQueue, solutions_mem, CL_TRUE, 0,
//                                  sizeof(int) * max_tour_sz * n_ants, result, 0, NULL, NULL);
//    check_error(err_num, CL_SUCCESS);
//    
//    int i, j, beg, end;
//    AntStruct *ant;
//    for (i = 0; i < n_ants; i++) {
//        ant = &instance.ants[i];
//        beg = i * max_tour_sz;
//        end = (i + 1) * max_tour_sz;
//        ant->tour_size = result[end - 1];
//        for (j = 0; j < ant->tour_size; j++) {
//            ant->tour[j] = result[j + beg];
//        }
//        ant->tour_length = compute_tour_length(&instance, ant->tour, ant->tour_size);
//        assert(check_solution(&instance, ant->tour, ant->tour_size));
//        print_solution(&instance, ant->tour, ant->tour_size);
//    }
//    delete[] result;
    
}

/*
 * pheromone update phase
 */
void g_ACO::pheromone_update(void)
{
    /* 1. pheromone evaporation */
    pheromone_evaporation();
    
    /* 2. pheromone deposit */
    pheromone_deposit();
    
    /* 3.compute total info ---*/
    compute_total_info();
}

void g_ACO::pheromone_init(float trail_0)
{
    cl_int err_num;
    size_t global_work_size[1] = {static_cast<size_t>(num_node * num_node)};
    size_t local_work_size[1] = {1};
    
    cl_kernel& pheromone_init = env.get_kernel(kernel_t::pheromone_init);
    
    err_num = clSetKernelArg(pheromone_init, 0, sizeof(float), &trail_0);
    err_num |= clSetKernelArg(pheromone_init, 1, sizeof(cl_mem), &pheromone_mem);
    err_num |= clSetKernelArg(pheromone_init, 2, sizeof(cl_mem), &total_info_mem);
    err_num |= clSetKernelArg(pheromone_init, 3, sizeof(cl_mem), &distance_mem);
    check_error(err_num, CL_SUCCESS);
    
    err_num = clEnqueueNDRangeKernel(env.commandQueue, pheromone_init,
                                     1, NULL, global_work_size, local_work_size,
                                     0, NULL, NULL);
    check_error(err_num, CL_SUCCESS);
}

/*
 * number of threads: num_node * num_node
 */
void g_ACO::pheromone_evaporation(void)
{
    cl_int err_num;
    cl_event event;
    size_t global_work_size[1] = {static_cast<size_t>(num_node * num_node)};
    
    cl_kernel& pheromone_evaporation = env.get_kernel(kernel_t::pheromone_evaporation);
    // 1. set kernel arguments
    err_num = clSetKernelArg(pheromone_evaporation, 0, sizeof(cl_mem), &pheromone_mem);
    check_error(err_num, CL_SUCCESS);
    
    // 2. queue the kernel up for executeion
    err_num = clEnqueueNDRangeKernel(env.commandQueue, pheromone_evaporation,
                                     1, NULL, global_work_size, NULL,
                                     0, NULL, &event);
    check_error(err_num, CL_SUCCESS);
    
    PROFILE(printf("[pheromone evaporation] %f ms\n", event_runtime(event));)
}

/*
 * pheromone deposit using rank-AS strategy
 */
void g_ACO::pheromone_deposit(void)
{
    /* first, get elites */
    get_elites();
    
    /* then, pheromone_deposit */
    cl_int err_num;
    cl_event event;
    const int grp_size = env.maxWorkGroupSize;
    const int num_grp = ras_ranks;
    size_t global_work_size[1] = {static_cast<size_t>(grp_size * num_grp)};
    size_t local_work_size[1] = {static_cast<size_t>(grp_size)};
    
    cl_kernel& pheromone_deposit = env.get_kernel(kernel_t::pheromone_deposit);
    
    err_num = clSetKernelArg(pheromone_deposit, 0, sizeof(cl_mem), &elite_ids_mem);
    err_num |= clSetKernelArg(pheromone_deposit, 1, sizeof(cl_mem), &solutions_mem);
    err_num |= clSetKernelArg(pheromone_deposit, 2, sizeof(cl_mem), &solution_lens_mem);
    err_num |= clSetKernelArg(pheromone_deposit, 3, sizeof(cl_mem), &pheromone_mem);
    err_num |= clSetKernelArg(pheromone_deposit, 4, sizeof(float) * max_tour_sz, NULL);
    check_error(err_num, CL_SUCCESS);
    
    err_num = clEnqueueNDRangeKernel(env.commandQueue, pheromone_deposit,
                                     1, NULL, global_work_size, local_work_size,
                                     0, NULL, &event);
    check_error(err_num, CL_SUCCESS);
    
    PROFILE(printf("[pheromone deposit] %f ms\n", event_runtime(event));)
}


void g_ACO::get_elites(void)
{
    cl_int err_num;
    cl_event event;
    
    size_t global_work_size[1] = {1};
    size_t local_work_size[1] = {1};
    
    cl_kernel& get_elites = env.get_kernel(kernel_t::get_elites);
    
    
    err_num = clSetKernelArg(get_elites, 0, sizeof(cl_mem), &solution_lens_mem);
    err_num |= clSetKernelArg(get_elites, 1, sizeof(cl_mem), &elite_ids_mem);
    check_error(err_num, CL_SUCCESS);
    
    err_num = clEnqueueNDRangeKernel(env.commandQueue, get_elites,
                                     1, NULL, global_work_size, local_work_size,
                                     0, NULL, &event);
    check_error(err_num, CL_SUCCESS);
    
    PROFILE(printf("[get elites] %f ms\n", event_runtime(event));)
}


/*
 * the total_info computation is performed apart from the tour construction kernel, 
 * being included in a different kernel which is executed right before the tour construction
 *
 * number of threads: num_node * num_node
 */
void g_ACO::compute_total_info(void)
{
    cl_int err_num;
    cl_event event;
    size_t global_work_size[1] = {static_cast<size_t>(num_node * num_node)};

    cl_kernel& compute_total_info = env.get_kernel(kernel_t::compute_total_info);
    
    err_num = clSetKernelArg(compute_total_info, 0, sizeof(cl_mem), &pheromone_mem);
    err_num |= clSetKernelArg(compute_total_info, 1, sizeof(cl_mem), &total_info_mem);
    err_num |= clSetKernelArg(compute_total_info, 2, sizeof(cl_mem), &distance_mem);
    check_error(err_num, CL_SUCCESS);
    
    err_num = clEnqueueNDRangeKernel(env.commandQueue, compute_total_info,
                                     1, NULL, global_work_size, NULL,
                                     0, NULL, &event);
    check_error(err_num, CL_SUCCESS);
    
    PROFILE(printf("[compute total info] %f ms\n", event_runtime(event));)
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
    
    err_num = clSetKernelArg(update_pheromone_weighted, 0, sizeof(cl_mem), &pheromone_mem);
    err_num |= clSetKernelArg(update_pheromone_weighted, 1, sizeof(cl_mem), &tour_mem);
    err_num |= clSetKernelArg(update_pheromone_weighted, 2, sizeof(cl_int), &a->tour_size);
    err_num |= clSetKernelArg(update_pheromone_weighted, 3, sizeof(float), &a->tour_length);
    err_num |= clSetKernelArg(update_pheromone_weighted, 4, sizeof(cl_int), &weight);
    check_error(err_num, CL_SUCCESS);
    
    err_num = clEnqueueNDRangeKernel(env.commandQueue, update_pheromone_weighted,
                                     1, NULL, global_work_size, local_work_size,
                                     0, NULL, NULL);
    check_error(err_num, CL_SUCCESS);
    
    clReleaseMemObject(tour_mem);
}

/*
 * update statistics
 */
void g_ACO::update_statistics(void)
{
    // Blocks until all previously queued commands in a command-queue have completed.
    clFinish(env.commandQueue);
    
    // find iter-best solution
    find_best_solution();
    
    // update and record best-so-far solution if better solution found
    update_best_so_far();
}

void g_ACO::find_best_solution(void)
{
    cl_int err_num;
    cl_event event1, event2;
    
    // phase 1 -- find best solution in each group
    const int grp_size = env.maxWorkGroupSize / 4;
    const int num_grps = ceil(1.0 * n_ants / grp_size);
    
    size_t global_work_size[1] = {static_cast<size_t>(grp_size * num_grps)};
    size_t local_work_size[1] = {static_cast<size_t>(grp_size)};
    
    cl_kernel& best_phase_0 = env.get_kernel(kernel_t::best_solution_phase_0);
    
    err_num = clSetKernelArg(best_phase_0, 0, sizeof(int), &n_ants);
    err_num |= clSetKernelArg(best_phase_0, 1, sizeof(cl_mem), &solution_lens_mem);
    err_num |= clSetKernelArg(best_phase_0, 2, sizeof(float) * grp_size, NULL);    // local memory for solution_lens[]
    err_num |= clSetKernelArg(best_phase_0, 3, sizeof(int) * grp_size, NULL);      // local memory for solution index
    err_num |= clSetKernelArg(best_phase_0, 4, sizeof(cl_mem), &best_result_val);
    err_num |= clSetKernelArg(best_phase_0, 5, sizeof(cl_mem), &best_result_idx);
    check_error(err_num, CL_SUCCESS);
    
    err_num = clEnqueueNDRangeKernel(env.commandQueue, best_phase_0,
                                     1, NULL, global_work_size, local_work_size,
                                     0, NULL, &event1);
    check_error(err_num, CL_SUCCESS);
    
    
    // phase 2 - find best among all groups
    cl_kernel& best_phase_1 = env.get_kernel(kernel_t::best_solution_phase_1);
    
    err_num = clSetKernelArg(best_phase_1, 0, sizeof(int), &num_grps);
    err_num |= clSetKernelArg(best_phase_1, 1, sizeof(cl_mem), &solutions_mem);
    err_num |= clSetKernelArg(best_phase_1, 2, sizeof(cl_mem), &solution_lens_mem);
    err_num |= clSetKernelArg(best_phase_1, 3, sizeof(cl_mem), &best_result_val);
    err_num |= clSetKernelArg(best_phase_1, 4, sizeof(cl_mem), &best_result_idx);
    err_num |= clSetKernelArg(best_phase_1, 5, sizeof(cl_mem), &update_flag_mem);
    check_error(err_num, CL_SUCCESS);
    
    global_work_size[0] ={1};
    local_work_size[0] = {1};
    err_num = clEnqueueNDRangeKernel(env.commandQueue, best_phase_1,
                                     1, NULL, global_work_size, local_work_size,
                                     0, NULL, &event2);
    check_error(err_num, CL_SUCCESS);
    
    PROFILE(printf("[find best solution] p1: %f ms, p2: %f ms\n",
                   event_runtime(event1), event_runtime(event2));)
    
    // debug
//    int idx;
//    err_num = clEnqueueReadBuffer(env.commandQueue, solutions_mem, CL_TRUE,
//                                  sizeof(int) * max_tour_sz * (n_ants + 1),
//                                  sizeof(int), &idx, 0, NULL, NULL);
//    check_error(err_num, CL_SUCCESS);
//    
//    int *result = new int[max_tour_sz];
//    err_num = clEnqueueReadBuffer(env.commandQueue, solutions_mem, CL_TRUE,
//                                  sizeof(int) * max_tour_sz * idx,
//                                  sizeof(int) * max_tour_sz, result, 0, NULL, NULL);
//    check_error(err_num, CL_SUCCESS);
//    
//    AntStruct *ant = &instance.ants[0];
//    ant->tour_size = result[max_tour_sz - 1];
//    for (int i = 0; i < ant->tour_size; i++) {
//        ant->tour[i] = result[i];
//    }
//    ant->tour_length = compute_tour_length(&instance, ant->tour, ant->tour_size);
//    assert(check_solution(&instance, ant->tour, ant->tour_size));
////    print_solution(&instance, ant->tour, ant->tour_size);
//    delete[] result;
}

/*
 * update best-so-far solution if better solution found
 * this function should be called after find_best_solution()
 * 注意: kernel进入队列后, 按照FIFO顺序执行, 因此实际被执行时间很可能晚于
 * kernel入队列时间, 所以需要首先调用clFinish() 保证之前命令执行完成，然后才计时
 */
void g_ACO::update_best_so_far(void)
{
    float time = elapsed_time(REAL);
    cl_int err_num;
    cl_event event;
    cl_kernel& update_best_so_far = env.get_kernel(kernel_t::update_best_so_far);
    
    err_num = clSetKernelArg(update_best_so_far, 0, sizeof(cl_mem), &update_flag_mem);
    err_num |= clSetKernelArg(update_best_so_far, 1, sizeof(cl_mem), &solutions_mem);
    err_num |= clSetKernelArg(update_best_so_far, 2, sizeof(cl_mem), &solution_lens_mem);
    err_num |= clSetKernelArg(update_best_so_far, 3, sizeof(cl_mem), &bsf_records_mem);
    err_num |= clSetKernelArg(update_best_so_far, 4, sizeof(cl_mem), &num_bsf_mem);
    err_num |= clSetKernelArg(update_best_so_far, 5, sizeof(float), &time);
    err_num |= clSetKernelArg(update_best_so_far, 6, sizeof(int), &instance.iteration);
    check_error(err_num, CL_SUCCESS);
    
    size_t global_work_size[1] ={static_cast<size_t>(max_tour_sz)};   // 保证大于等于 iter-best solution size 即可
    // 不指定局部维度,OpenCL实现可以选择局部维度
    err_num = clEnqueueNDRangeKernel(env.commandQueue, update_best_so_far,
                                     1, NULL, global_work_size, NULL,
                                     0, NULL, &event);
    check_error(err_num, CL_SUCCESS);
    
    PROFILE(printf("[update best so far] %f ms\n", event_runtime(event));)
}

/*
 * update host best-so-far solution from device memory
 */
void g_ACO::update_best_so_far_from_device(void)
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

    ant->tour_length = best_len;
    for (int i = 0; i < ant->tour_size; i++) {
        ant->tour[i] = result[i];
    }
    delete[] result;
    DEBUG(assert(fabs(best_len - compute_tour_length(&instance, ant->tour, ant->tour_size)) < EPSILON);)
}

/*
 * update host best-so-far solution to device memory
 * number of threads: 1
 */
void g_ACO::update_best_so_far_to_device(void)
{
    cl_int err_num;
    size_t global_work_size[1] ={1};
    size_t local_work_size[1] = {1};
    
    AntStruct *best_so_far = instance.best_so_far_ant;
    
    cl_mem tour_mem = clCreateBuffer(env.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     sizeof(int) * best_so_far->tour_size, best_so_far->tour, &err_num);
    check_error(err_num, CL_SUCCESS);
    
    cl_kernel& update_best_so_far_to_device = env.get_kernel(kernel_t::update_best_so_far_to_device);
    // 1. set kernel arguments
    err_num = clSetKernelArg(update_best_so_far_to_device, 0, sizeof(cl_mem), &tour_mem);
    err_num |= clSetKernelArg(update_best_so_far_to_device, 1, sizeof(cl_int), &best_so_far->tour_size);
    err_num |= clSetKernelArg(update_best_so_far_to_device, 2, sizeof(float), &best_so_far->tour_length);
    err_num |= clSetKernelArg(update_best_so_far_to_device, 3, sizeof(cl_mem), &solutions_mem);
    err_num |= clSetKernelArg(update_best_so_far_to_device, 4, sizeof(cl_mem), &solution_lens_mem);
    check_error(err_num, CL_SUCCESS);
    
    // 2. queue the kernel up for executeion
    err_num = clEnqueueNDRangeKernel(env.commandQueue, update_best_so_far_to_device,
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

/*
 * 统计事件执行时间，单位:毫秒
 */
double g_ACO::event_runtime(cl_event& event) {
    cl_int err_num;
    cl_ulong ev_start_time = (cl_ulong) 0;
    cl_ulong ev_end_time = (cl_ulong) 0;
    size_t return_bytes;
    
    clFinish(env.commandQueue);
    err_num = clWaitForEvents(1, &event);
    
    err_num |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &ev_start_time, &return_bytes);
    err_num |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &ev_end_time, &return_bytes);
    check_error(err_num, CL_SUCCESS);
    
    double run_time = (double)(ev_end_time - ev_start_time);
    return run_time*1.0e-6;
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
    
    // update_flag object
    update_flag_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(bool), NULL, &err_num);
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
    int seed_size = cs_grp_size * cs_num_grps;
    int *tmp_seed = new int[seed_size];
    for (i = 0; i < seed_size; i++) {
        tmp_seed[i] = (int)time(NULL) + (i%n_ants) * ((i%n_ants) + ran01(&instance.rnd_seed)) * n_ants;
    }
    seed_mem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              sizeof(int) * seed_size, tmp_seed, &err_num);
    check_error(err_num, CL_SUCCESS);
    
    /*
     * best-so-far solutions information memory, at most 4096 records
     */
    bsf_records_mem = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                     sizeof(BestSolutionInfo) * 4096, NULL, &err_num);
    check_error(err_num, CL_SUCCESS);
    
    int tmp_num = 0;
    num_bsf_mem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                 sizeof(int) * 1, &tmp_num, &err_num);
    check_error(err_num, CL_SUCCESS);
    
    /*
     * store all `ras_rank` elite ants' ids
     */
    elite_ids_mem = clCreateBuffer(env.context, CL_MEM_READ_WRITE,
                                   sizeof(int) * ras_ranks, NULL, &err_num);
    check_error(err_num, CL_SUCCESS);
    
    /*
     * FIXME: hard code `num_grps`
     * especially for find_best_solution()
     */
    const int grp_size = env.maxWorkGroupSize / 4;
    const int num_grps = ceil(1.0 * n_ants / grp_size);
    best_result_val = clCreateBuffer(env.context, CL_MEM_READ_WRITE,
                                     sizeof(float) * num_grps, NULL, &err_num);
    check_error(err_num, CL_SUCCESS);
    best_result_idx = clCreateBuffer(env.context, CL_MEM_READ_WRITE,
                                     sizeof(int) * num_grps, NULL, &err_num);
    check_error(err_num, CL_SUCCESS);
    
    
    delete[] tmp_demands;
    delete[] tmp_distance;
    delete[] tmp_seed;
    delete[] tmp_nn_list;
}




