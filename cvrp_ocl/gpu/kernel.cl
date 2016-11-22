//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define TRUE true
#define FALSE false

#define EPSILON     0.001f
#define DEBUG(x)    x
/****** parameters ******/
#define RHO         0.1f
#define ALPHA       1.0f
#define BETA        2.0f
#define RAS_RANKS   6
/************************/


/*************************************************
                inline functions
 *************************************************/

#define IA 16807
#define IM 2147483647
#define AM (1.0f/IM)
#define IQ 127773
#define IR 2836

#ifndef NUM_NODE
#define NUM_NODE  1000
#endif

#ifndef N_ANTS
#define N_ANTS  NUM_NODE
#endif

#define MAX_TOUR_SZ  (2 * NUM_NODE)

struct BestSolutionInfo {
    int     iter;
    float   length;
    float   time;
};

inline float ran01(int *idum )
{
    int k;
    float ans;
    
    k =(*idum)/IQ;
    *idum = IA * (*idum - k * IQ) - IR * k;
    if (*idum < 0 ) *idum += IM;
    ans = AM * (*idum);
    return ans;
}

/*
 * In OpenCL there is only atomic_add or atomic_mul for ints, and not for floats.
 * http://simpleopencl.blogspot.com/2013/05/atomic-operations-and-floats-in-opencl.html
 */
inline void atomic_add_global(volatile global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

inline float compute_tour_length(__global float *distance, int *tour, int tour_size)
{
    int      i;
    float   tour_length = 0;
    
    for ( i = 0 ; i < tour_size-1; i++ ) {
        tour_length += distance[tour[i] * NUM_NODE + tour[i+1]];
    }
    return tour_length;
}

/******** print ********/
inline void print_solution(__global float *distance, int *tour, int tour_size)
{
    int   i;
    
    printf("\n--------\n");
    
    printf("Ant soulution:");
    for( i = 0 ; i < tour_size ; i++ ) {
        if (!i%25) {
            printf("\n");
        }
        printf("%d ", tour[i]);
    }
    printf("\n");
    printf("Tour length = %f", compute_tour_length(distance, tour, tour_size));
    
    printf("\n--------\n\n");
}

inline void print_single_route(__global float *distance, int *route, int route_size)
{
    int   i;
    
    printf("\n--------\n");
    
    printf("Route: ");
    for( i = 0 ; i < route_size ; i++ ) {
        if (i!= 0 && !i%25) {
            printf("\n");
        }
        printf("%d ", route[i]);
    }
    printf("\n");
    printf("Route length = %f", compute_tour_length(distance, route, route_size));
    
    printf("\n--------\n");
}

/******* construct solutions *********/

inline int choose_best_next(float *total_info_wrk, bool *candidate)
{
    int node, next_node;
    float   value_best;
    
    next_node = NUM_NODE;
    value_best = -1.0f;             /* values in total matrix are always >= 0.0 */
    for (node = 0; node < NUM_NODE; node++) {
        if(candidate[node] == TRUE) {
            if (total_info_wrk[node] > value_best) {
                next_node = node;
                value_best = total_info_wrk[node];
            }
        }
    }
    return next_node;
}

/*
 * FUNCTION:    Choose for an ant probabilistically a next node among all
 * unvisited and possible nodes in the current node's candidate list.
 */
inline int choose_and_move_to_next(int current_node, int *rnd_seed,
                                   float *total_info_wrk, bool *candidate)
{
    int i;
    float partial_sum = 0.0f, sum_prob = 0.0f;
    float rnd;
    
    for (i = 0 ; i < NUM_NODE; i++) {
        sum_prob += candidate[i] * total_info_wrk[i];
    }
    
    if (sum_prob <= 0.0f) {
        return choose_best_next(total_info_wrk, candidate);
        printf("oops! we have a bug!");
    } else {
        /* chose one according to the selection probabilities */
        rnd = ran01(rnd_seed) * sum_prob;
        
        i = 0;
        partial_sum = candidate[i] * total_info_wrk[i];
        while (partial_sum <= rnd) {
            i++;
            partial_sum += candidate[i] * total_info_wrk[i];
            if(i == NUM_NODE) {
                // This may very rarely happen because of rounding if rnd is close to 1.
                DEBUG(printf("omg! It happens!\n");)
                return choose_best_next(total_info_wrk, candidate);
            }
        }
        return i;
    }
}

/******** local search ********/

inline void two_opt_single_route(int *tour, int rbeg, int rend,
                                 bool *dlb, bool *route_node_map, int *tour_node_pos,
                                 __global float *distance,
                                 __global int *nn_list, const int nn_ls,
                                 float *dist_tour_wrk)
{
    int n1, n2;                            /* nodes considered for an exchange */
    int s_n1, s_n2;                        /* successor nodes of n1 and n2     */
    int p_n1, p_n2;                        /* predecessor nodes of n1 and n2   */
    int pos_n1, pos_n2;                    /* positions of nodes n1, n2        */
    int num_route_node = rend - rbeg + 1;  /* number of nodes in a single route(depot只计一次) */
    
    int i, j, h, l;
    bool imp_flag;
    int help;
    int h1=0, h2=0, h3=0, h4=0;
    float radius;             /* radius of nn-search */
    float gain = 0;
    
    /*-------------- private memory 优化 ---------------*/
    /* 
     * distance[n1][*], distance[s_n1][*] and distance[p_n1][*]
     * 在循环中被反复使用，因此拷贝至私有内存提高访问速度
     */
    float   dist_n1_wrk[NUM_NODE];
    float   dist_pn1_wrk[NUM_NODE];
    float   dist_sn1_wrk[NUM_NODE];
    
//    print_single_route(distance, tour + rbeg, num_route_node+1);
    
    imp_flag = TRUE;
    while (imp_flag) {

        imp_flag = FALSE;
        
        for (l = 0; l < num_route_node; l++) {
            
            /* the neighbourhood is scanned in sequence order */
            pos_n1 = rbeg + l;
            n1 = tour[pos_n1];
            if (dlb[n1]) {
                continue;
            }
            s_n1 = pos_n1 == rend ? tour[rbeg] : tour[pos_n1+1];
            
            // [优化-使用私有内存]
            for(int k = 0; k < NUM_NODE; k++) {
                dist_n1_wrk[k] = distance[n1 * NUM_NODE + k];
                dist_sn1_wrk[k] = distance[s_n1 * NUM_NODE + k];
            }
            
            radius = dist_n1_wrk[s_n1];
            
            /* First search for n1's nearest neighbours, use successor of n1 */
            for ( h = 0; h < nn_ls; h++ ) {
                n2 = nn_list[n1 * nn_ls + h];     /* exchange partner, determine its position */
                if (route_node_map[n2] == FALSE) {
                    /* 该点不在本route中 */
                    continue;
                }
                if (radius - dist_n1_wrk[n2] > EPSILON) {
                    pos_n2 = tour_node_pos[n2];
                    s_n2 = pos_n2 == rend ? tour[rbeg] : tour[pos_n2+1];
                    // gain = -radis + dist(n1,n2) + dis(s_n1, s_n2) - dis(n2, s_n2)
                    gain =  - radius + dist_n1_wrk[n2] + dist_sn1_wrk[s_n2] - dist_tour_wrk[pos_n2];
                    if (gain < -EPSILON) {
                        h1 = n1; h2 = s_n1; h3 = n2; h4 = s_n2;
//                        printf("b. %d %d %d %d %d\n", h1, h2, h3, h4, route_node_map[n2]);
                        goto exchange2opt;
                    }
                }
                else break;
            }
            
            
            /* Search one for next c1's h-nearest neighbours, use predecessor n1 */
            p_n1 = pos_n1 == rbeg ? tour[rend] : tour[pos_n1-1];
            
            for(int k = 0; k < NUM_NODE; k++) {
                dist_pn1_wrk[k] = distance[p_n1 * NUM_NODE + k];
            }
            
            radius = dist_pn1_wrk[n1];
            for ( h = 0; h < nn_ls; h++ ) {
                n2 = nn_list[n1 * nn_ls + h];    /* exchange partner, determine its position */
                if (route_node_map[n2] == FALSE) {
                    /* 该点不在本route中 */
                    continue;
                }
                if (radius - dist_n1_wrk[n2] > EPSILON) {
                    pos_n2 = tour_node_pos[n2];
                    p_n2 = pos_n2 == rbeg ? tour[rend] : tour[pos_n2-1];
                    
                    if ( p_n2 == n1 || p_n1 == n2)
                        continue;
                    // gain = -radis + dis(n1,n2) + dis(p_n1, p_n2) - dis(p_n2, n2)
                    gain =  - radius + dist_n1_wrk[n2] + dist_pn1_wrk[p_n2] - dist_tour_wrk[pos_n2-1];
                    if ( gain < -EPSILON ) {
                        h1 = p_n1; h2 = n1; h3 = p_n2; h4 = n2;
//                        printf("b. %d %d %d %d %d\n", h1, h2, h3, h4, route_node_map[n2]);
                        goto exchange2opt;
                    }
                }
                else break;
            }
            /* No exchange */
            dlb[n1] = TRUE;
            continue;
            
        exchange2opt:
            imp_flag = TRUE;
            dlb[h1] = FALSE; dlb[h2] = FALSE;
            dlb[h3] = FALSE; dlb[h4] = FALSE;
            /* Now perform move */
            if ( tour_node_pos[h3] < tour_node_pos[h1] ) {
                help = h1; h1 = h3; h3 = help;
                help = h2; h2 = h4; h4 = help;
            }
            
            /* !! update dist_tour_wrk[] between pos(h1) and pos(h3) */
            float tmp;
            i = tour_node_pos[h1];
            j = tour_node_pos[h3];
            dist_tour_wrk[i] = distance[h1 * NUM_NODE + h3];   // dis(h1,h3)
            dist_tour_wrk[j] = distance[h2 * NUM_NODE + h4];   // dis(h2,h4)
            i++; j--;
            while(i < j) {
                tmp = dist_tour_wrk[i];
                dist_tour_wrk[i] = dist_tour_wrk[j];
                dist_tour_wrk[j] = tmp;
                i++; j--;
            }
            
            /* reverse inner part from pos[h2] to pos[h3] */
            i = tour_node_pos[h2];
            j = tour_node_pos[h3];
            while (i < j) {
                n1 = tour[i];
                n2 = tour[j];
                tour[i] = n2;
                tour[j] = n1;
                tour_node_pos[n1] = j;
                tour_node_pos[n2] = i;
                i++; j--;
            }

            // debug
//            printf("after ls. [%d %d] gain %f\n", h2, h3, gain);
//            print_single_route(distance, tour + rbeg, num_route_node+1);
        }
    }
}

inline void two_opt_solution(int *tour, int tour_size, __global float *distance,
                             __global int *nn_list, const int nn_ls,
                             float  *dist_tour_wrk)
{
    /* vector containing don't look bits */
    bool dlb[NUM_NODE];
    /* mark for all nodes in a single route */
    bool route_node_map[NUM_NODE];
    /* positions of nodes in tour */
    int tour_node_pos[NUM_NODE];
    
    int route_beg = 0;
    int i, j;
    
    for (i = 0 ; i < NUM_NODE; i++) {
        dlb[i] = FALSE;
        route_node_map[i] = FALSE;
    }
    
    for (j = 0; j < tour_size; j++) {
        tour_node_pos[tour[j]] = j;
    }
    
    for (i = 1; i < tour_size; i++) {
        // 2-opt a single route from tour
        if (tour[i] == 0) {
            tour_node_pos[0] = route_beg;
            two_opt_single_route(tour, route_beg, i-1, dlb,
                                 route_node_map, tour_node_pos,
                                 distance, nn_list, nn_ls, dist_tour_wrk);
            
            for (j = 0; j < NUM_NODE; j++) {
                route_node_map[j] = FALSE;
            }
            route_beg = i;
            
        } else {
            route_node_map[tour[i]] = TRUE;
        }
    }
}


/*********************************************************
                      kernels
 ********************************************************/

/*
 * construct the solution for an ant
 * number of threads: N_ANTS
 */
__kernel void construct_solution(const int capacity, const float max_dist, const float serv_time,
                                 __global int *rnd_seeds, __global float *distance, __global int *demands,
                                 __global float *total_info, __global int *solutions, __global float *solution_lens,
                                 __local float *dist0_wrk, __local int *demands_wrk)
{
    //    DEBUG(printf("begin constructing solution\n");)
    int     gid = get_global_id(0);
    int     lid = get_local_id(0);
    int     nloc = get_local_size(0);
    
    /*-------------- local memory 优化 ---------------*/
    /* 
     * 复制第0列 distance[*][0] 为局部内存
     * 注意：需要在 if(gid < N_ANTS) 之外，保证 all work-items execute barrier(),
     * 否则将永久阻塞在此.
     */
    for(int i = lid; i < NUM_NODE; i +=nloc) {
        dist0_wrk[i] = distance[i * NUM_NODE];
        demands_wrk[i] = demands[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);  // !! blocks until all work-items execute this function
    
    if (gid < N_ANTS)
    {
        __global int  *tour = solutions + MAX_TOUR_SZ * gid;
        int     tour_size;
        
        bool    visited[NUM_NODE];
        bool    candidate[NUM_NODE];
        
        int     visited_cnt;            /* count of visited node by this ant */
        int     path_load;              /* 单次从depot出发的送货量 */
        int     next_node, current_node;
        int     i, candidate_cnt, step;
        float   path_dist, tour_dist;

        /*-------------- private memory 优化 ---------------*/
        /* 复制i行 distance[i][*] 为私有内存 */
        float   dist_wrk[NUM_NODE];
        
        /* 复制i行 total_info[i][*] 为私有内存 */
        float   total_info_wrk[NUM_NODE];
        
        /* 复制 rnd_seed[gid] 为私有内存 */
        int     rnd_seed = rnd_seeds[gid];
        
    
        /* Mark all nodes as unvisited */
        for(i = 0; i < NUM_NODE; i++) {
            visited[i] = FALSE;
        }
        
        path_load = 0;
        path_dist = 0;
        tour_dist = 0;
        step = 0;
        
        // init ant place
        visited[0] = TRUE;
        tour[step] = 0;
        visited_cnt = 1;
        current_node = 0;
        
        while (visited_cnt < NUM_NODE) {
            step++;
            candidate_cnt = 0;
            
            /* [优化-使用私有内存]
             * 将 distance[current_node][*] 一行的数据复制到私有内存， 提高读取速度
             */
            for(i = 0; i < NUM_NODE; i++) {
                candidate[i] = FALSE;   // 初始化所有满足配送条件的点
                dist_wrk[i] = distance[current_node * NUM_NODE + i];
                total_info_wrk[i] = total_info[current_node * NUM_NODE + i];
            }
            
            for(i = 0; i < NUM_NODE; i++) {
                if (visited[i] == FALSE &&
                    path_load + demands_wrk[i] <= capacity &&
                    path_dist + (dist_wrk[i] + serv_time) + dist0_wrk[i] <= max_dist) {
                    candidate[i] = TRUE;
                    candidate_cnt++;
                }
            }
            
            /*
             1)如果满足配送条件, 则蚂蚁回到depot，重新开始新的路径
             2）否则，选择下一个配送点
             */
            if (candidate_cnt == 0) {
                next_node = 0;
                path_load = 0;
                path_dist = 0;
            } else {
                next_node = choose_and_move_to_next(current_node, &rnd_seed, total_info_wrk, candidate);
                path_load += demands_wrk[next_node];
                path_dist += dist_wrk[next_node] + serv_time;
                visited_cnt++;
            }
            visited[next_node] = TRUE;
            tour[step] = next_node;
            current_node = next_node;
            tour_dist += dist_wrk[next_node];
        }
        
        // 最后回到depot
        step++;
        tour[step] = 0;
        tour_size = step + 1;
        tour_dist += dist0_wrk[tour[step-1]];
        tour[MAX_TOUR_SZ-1] = tour_size;
        
        solution_lens[gid] = tour_dist;
//        solution_lens[gid] = compute_tour_length(distance, tour, tour_size);
//        printf("len:%f \n", solution_lens[gid]);
        
        // !!更新种子至 global memory
        rnd_seeds[gid] = rnd_seed;
    }
}


/*
 * local search
 * number of threads: N_ANTS
 */
__kernel void local_search(const int nn_ls, __global int *nn_list,
                           __global float *distance, __global int *solutions,
                           __global float *solution_lens, __global int *demands,
                           const int capacity, const float max_dist, const float serv_time)
{
    int     gid = get_global_id(0);
    __global int *tour = solutions + MAX_TOUR_SZ * gid;
    int     tour_size = tour[MAX_TOUR_SZ-1];
    int     i;
    
    /*-------------- private memory 优化 ---------------*/
    int     tour_wrk[MAX_TOUR_SZ];
    /* 将tour路径上相邻两点距离的拷贝至私有内存
     * 注意：tour_wrk[i]中下标i与tour[i]对应，而不是与tour中的node对应
     */
    float   dist_tour_wrk[MAX_TOUR_SZ];
    
    for(i = 0; i < tour_size; i++) {
        tour_wrk[i] = tour[i];    // 将tour global memory 拷贝至私有内存
    }
    
    for (i = 0; i < tour_size-1; i++) {
        dist_tour_wrk[i] = distance[tour_wrk[i] * NUM_NODE + tour_wrk[i+1]];
    }
    
    // 1) 2-opt local search
    two_opt_solution(tour_wrk, tour_size, distance, nn_list, nn_ls, dist_tour_wrk);
    
    
    // !!update global memory
    for(int i = 0; i < tour_size; i++) {
        tour[i] = tour_wrk[i];
    }
    
    float len = 0;
    for (i = 0; i < tour_size-1; i++) {
        len += dist_tour_wrk[i];
    }
    solution_lens[gid] = len;
}


/*
 * update best so far solution after an iteration if better solution found
 * number of threads: MAX_TOUR_SZ (>= tour_size) 即可
 */
__kernel void update_best_so_far(__global bool *update_flag,
                                 __global int *solutions, __global float *solution_lens,
                                 __global struct BestSolutionInfo *bsf_records, __global int *num_bsf,
                                 float time, int iteration)
{
    if(*update_flag)
    {
        int gid = get_global_id(0);
        __global int *best_tour, *iter_best_tour;
        
        /*
         * after call kernel best_solution_phase_1(),
         * we have already got the iter-best solution
         */
        int idx = solutions[MAX_TOUR_SZ * (N_ANTS + 1)];
        iter_best_tour = solutions + MAX_TOUR_SZ * idx;
        int tour_size = iter_best_tour[MAX_TOUR_SZ - 1];
        float len = solution_lens[idx];

        // update best-so-far solution
        if (gid < tour_size) {
            best_tour = solutions + MAX_TOUR_SZ * N_ANTS;
            best_tour[gid] = iter_best_tour[gid];
            
            if (gid == 0) {
                best_tour[MAX_TOUR_SZ - 1] = tour_size;
                solution_lens[N_ANTS] = len;
            }
        }
        
        // record this best-so-far solution information
        if(gid == 0) {
            int pos = *num_bsf;
            bsf_records[pos].iter = iteration;
            bsf_records[pos].length = len;
            bsf_records[pos].time = time;
            DEBUG(printf("Best: iter: %d len: %f time: %.2f\n", iteration, len, time);)
            *num_bsf = ++pos;
        }
    }
}


/*
 * Finds the best solution with minimum length
 * http://developer.amd.com/resources/articles-whitepapers/opencl-optimization-case-study-simple-reductions/
 * Step 1 : reduce phase, find best solution for each group
 */
__kernel void best_solution_phase_0(int n_solutions, __global float* solution_lens,
                                    __local float* scratch_val, __local int* scratch_idx,
                                   __global float* result_val,  __global int* result_idx)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    
    // Load data into local memory
    if (gid < n_solutions) {
        scratch_val[lid] = solution_lens[gid];
    } else {
        // Infinity is the identity element for the min operation
        scratch_val[lid] = INFINITY;
    }
    scratch_idx[lid] = gid;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Apply reduction
    for (int offset = get_local_size(0) / 2; offset > 0; offset /= 2)
    {
        if (lid < offset) {
            float other = scratch_val[lid + offset];
            float mine = scratch_val[lid];
            if (other < mine) {
                scratch_val[lid] = other;
                scratch_idx[lid] = scratch_idx[lid + offset];
            }
            
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        int grp_id = get_group_id(0);
        result_val[grp_id] = scratch_val[0];
        result_idx[grp_id] = scratch_idx[0];
    }
}

/*
 * Finds the best solution with minimum length
 * Step 2: get best soltion from all work-groups' best solutions
 * Note: It assumes that 'length' is very small (and so doesn't incur into
 *       another reduction)
 * number of threads: 1
 */
__kernel void best_solution_phase_1(int num_grps,
                                    __global int *solutions, __global float *solution_lens,
                                    __global float *result_val, __global int *result_idx,
                                    __global bool *update_flag)
{

    float minval = result_val[0];
    int minidx = result_idx[0];
    
    for (int i = 1; i < num_grps; i++) {
        float x = result_val[i];
        if (x < minval) {
            minval = x;
            minidx = result_idx[i];
        }
    }
    result_idx[0] = minidx;
    
    // update iter-best solution id
    solutions[MAX_TOUR_SZ * (N_ANTS + 1)] = minidx;
    // mark if we need to update best-so-far solution
    if(minval - solution_lens[N_ANTS] < -EPSILON) {
        *update_flag = TRUE;
    } else {
        *update_flag = FALSE;
    }
}


/*
 * pheromone init
 * number of threads: NUM_NODE * NUM_NODE
 */
__kernel void pheromone_init(float initial_trail, __global float *pheromone,
                             __global float *total_info, __global float *distance)
{
    int gid = get_global_id(0);
    
    pheromone[gid] = initial_trail;
    total_info[gid] = pow((float)pheromone[gid], (float)ALPHA) * pow((float)(1.0f/(distance[gid]+0.1f)), (float)BETA);
}

/*
 * pheromone evaporation
 * number of threads: NUM_NODE * NUM_NODE
 */
__kernel void pheromone_evaporation(__global float *pheromone)
{
    //    printf("begin pheromone_evaporation...\n");
    int gid = get_global_id(0);
    
    /* Simulate the pheromone evaporation of all pheromones */
    pheromone[gid] = (1 - RHO) * pheromone[gid];
}

/*
 * update pheromone weighted
 * number of threads: 1
 */
__kernel void update_pheromone_weighted(__global float *pheromone,
                                        __global int *tour, int tour_size,
                                        float tour_length, int weight)
{
    int      i, j, h;
    float   d_tau;
    
    d_tau = weight / tour_length;
    for (i = 0; i < tour_size - 1; i++) {
        j = tour[i];
        h = tour[i+1];
        pheromone[j * NUM_NODE + h] += d_tau;
    }
}

/*
 * get elite ants
 * number of threads: 1
 */
__kernel void get_elites(__global float *solution_lens, __global int *elite_ids)
{
    int i, k, target;
    float help_b[N_ANTS], b;

    // first, best-so-far solution
    elite_ids[0] = N_ANTS;
    
    // then, get elite ants ids
    for (k = 0; k < N_ANTS; k++) {
        help_b[k] = solution_lens[k];
    }
    for (i = 0; i < RAS_RANKS - 1; i++) {
        b = help_b[0];
        target = 0;
        for (k = 0; k < N_ANTS; k++) {
            if (help_b[k] < b) {
                b = help_b[k];
                target = k;
            }
        }
        help_b[target] = INFINITY;
        elite_ids[i+1] = target;
    }
}

/*
 * work_group_size = env.maxWorkGroupSize / 4
 * num_grp = number of elite ants + best-so-far ant = RAS_RANKS
 */
__kernel void pheromone_deposit(__global int *elite_ids,
                                __global int *solutions, __global float *solution_lens,
                                __global float *pheromone, __local int *tour_wrk)
{
    int lid = get_local_id(0);
    int grp_id = get_group_id(0);   // 0 to RAS_RANKS - 1
    int tid = elite_ids[grp_id];
    int lsz = get_local_size(0);
    
    __global int *tour = solutions + tid * MAX_TOUR_SZ;
    int tour_size = tour[MAX_TOUR_SZ-1];
    float tour_length = solution_lens[tid];
    
    // [优化]局部内存优化
    for(int i = lid; i < tour_size; i+=lsz) {
        tour_wrk[i] = tour[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // update pheromone int this tour
    int weight = RAS_RANKS - grp_id;
    float d_tau = 1.0f * weight / tour_length;

    for(int i = lid; i < tour_size - 1; i+=lsz) {
        int j = tour_wrk[i];
        int h = tour_wrk[i+1];
        // !!多只蚂蚁同时对同一块内存写数据，需要原子操作
        atomic_add_global(&(pheromone[j * NUM_NODE + h]), d_tau);
    }
}

/*
 * compute total info
 * number of threads: NUM_NODE * NUM_NODE
 */
__kernel void compute_total_info(__global float *pheromone, __global float *total_info, __global float *distance)
{
    int gid = get_global_id(0);
    
    /* Compute combined information pheromone times heuristic info after
     the pheromone update for ACO algorithm */
    total_info[gid] = pow((float)pheromone[gid], (float)ALPHA) * pow((float)(1.0f/(distance[gid]+0.1f)), (float)BETA);
}

/*
 * update host best-so-far solution to device memory
 * number of threads: 1
 */
__kernel void update_best_so_far_to_device(__global int *from_tour, int tour_size, float tour_length,
                                        __global int *solutions, __global float *solution_lens)
{
    __global int *best_tour = solutions + MAX_TOUR_SZ * N_ANTS;
    
    for(int i = 0; i < tour_size; i++) {
        best_tour[i] = from_tour[i];
    }
    best_tour[MAX_TOUR_SZ-1] = tour_size;
    
    solution_lens[N_ANTS] = tour_length;
}



