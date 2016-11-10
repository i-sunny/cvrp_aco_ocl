#pragma OPENCL EXTENSION cl_khr_fp64 : enable


#define TRUE true
#define FALSE false

#define EPSILON 0.001

/****** parameters ******/
#define RHO         0.1
#define ALPHA       1.0
#define BETA        2.0
#define RAS_RANKS   6
#define DELTA       0.7
/************************/


/*************************************************
                inline functions
 *************************************************/

#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836

inline double ran01(__global int *idum )
{
    int k;
    double ans;
    
    k =(*idum)/IQ;
    *idum = IA * (*idum - k * IQ) - IR * k;
    if (*idum < 0 ) *idum += IM;
    ans = AM * (*idum);
    return ans;
}

inline double compute_tour_length(int num_node, __global double *distance,
                                  __global int *tour, int tour_size)
{
    int      i;
    double   tour_length = 0;
    
    for ( i = 0 ; i < tour_size-1; i++ ) {
        tour_length += distance[tour[i] * num_node + tour[i+1]];
    }
    return tour_length;
}

/******** print ********/
inline void print_solution(int num_node, __global double *distance, __global int *tour, int tour_size)
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
    printf("Tour length = %f", compute_tour_length(num_node, distance, tour, tour_size));
    
    printf("\n--------\n\n");
}

inline void print_single_route(int num_node, __global double *distance, __global int *route, int route_size)
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
    printf("Route length = %f", compute_tour_length(num_node, distance, route, route_size));
    
    printf("\n--------\n");
}

/******* construct solutions *********/
inline int choose_best_next(int phase, int num_node, int *tour,
                            __global double *total_info, bool *visited, bool *demand_meet)
{
    int node, current_node, next_node;
    double   value_best;
    
    next_node = num_node;
    current_node = tour[phase-1];
    value_best = -1.;             /* values in total matrix are always >= 0.0 */
    for ( node = 0 ; node < num_node ; node++ ) {
        if (visited[node]) {
            ; /* node already visited, do nothing */
        } else if(demand_meet[node] == FALSE) {
            ;  /* 该点不满足要求 */
        } else {
            if ( total_info[current_node * num_node + node] > value_best ) {
                next_node = node;
                value_best = total_info[current_node * num_node + node];
            }
        }
    }
    tour[phase] = next_node;
    visited[next_node] = TRUE;
    
    return next_node;
}

inline int choose_and_move_to_next(int phase, int num_node, __global int *rnd_seed, int *tour,
                                   __global double *total_info, double *prob_ptr,
                                   bool *visited, bool *demand_meet)
{
    int i;
    int current_node;
    double partial_sum = 0.0, sum_prob = 0.0;
    double rnd;
    
    current_node = tour[phase-1];           /* current_node node of ant k */
    for (i = 0 ; i < num_node; i++) {
        if (visited[i]) {
            prob_ptr[i] = 0.0;   /* node already visited */
        } else if(demand_meet[i] == FALSE) {
            prob_ptr[i] = 0.0;  /* 该点不满足要求 */
        } else {
            prob_ptr[i] = total_info[current_node * num_node + i];
            sum_prob += prob_ptr[i];
        }
    }

    
    if (sum_prob <= 0.0) {
        /* All nodes from the candidate set are tabu */
        return choose_best_next(phase, num_node, tour, total_info, visited, demand_meet);
        printf("------error-------\n");
    } else {
        /* chose one according to the selection probabilities */
        rnd = ran01(rnd_seed);
        rnd *= sum_prob;
        
        i = 0;
        partial_sum = prob_ptr[i];

        while (partial_sum < rnd) {
            i++;
            partial_sum += prob_ptr[i];
        }
        tour[phase] = i;
        visited[i] = TRUE;
        
        return i;
    }
}

/******** local search ********/
inline void generate_random_permutation(int n, int *r, __global int *rnd_seed)
{
    int  i, help, node, tot_assigned = 0;
    double    rnd;

    for ( i = 0 ; i < n; i++){
        r[i] = i;
    }
    for ( i = 0 ; i < n ; i++ ) {
        /* find (randomly) an index for a free unit */
        rnd  = ran01(rnd_seed);
        node = (int) (rnd  * (n - tot_assigned));

        help = r[i];
        r[i] = r[i+node];
        r[i+node] = help;
        tot_assigned++;
    }
}

inline void two_opt_single_route(int num_node, __global int *tour, int rbeg, int rend,
                                 bool *dlb, bool *route_node_map, int *tour_node_pos,
                                 __global int *rnd_seed, __global double *distance,
                                 __global int *nn_list, int nn_ls)
{
    int n1, n2;                            /* nodes considered for an exchange */
    int s_n1, s_n2;                        /* successor nodes of n1 and n2     */
    int p_n1, p_n2;                        /* predecessor nodes of n1 and n2   */
    int pos_n1, pos_n2;                    /* positions of nodes n1, n2        */
    int num_route_node = rend - rbeg + 1;  /* number of nodes in a single route(depot只计一次) */
    
    int i, j, h, l;
    int improvement_flag, help, n_improves = 0, n_exchanges = 0;
    int h1=0, h2=0, h3=0, h4=0;
    double radius;             /* radius of nn-search */
    double gain = 0;
    int random_vector[num_route_node];
    
//    print_single_route(num_node, distance, tour + rbeg, num_route_node+1);
    
    improvement_flag = TRUE;
    generate_random_permutation(num_route_node, random_vector, rnd_seed);
    
    while ( improvement_flag ) {

        improvement_flag = FALSE;
        
        for (l = 0 ; l < num_route_node; l++) {
            
            /* the neighbourhood is scanned in random order */
            pos_n1 = rbeg + random_vector[l];
            n1 = tour[pos_n1];
            if (dlb[n1])
                continue;
            
            s_n1 = pos_n1 == rend ? tour[rbeg] : tour[pos_n1+1];
            radius = distance[n1 * num_node + s_n1];
            
            /* First search for n1's nearest neighbours, use successor of n1 */
            for ( h = 0; h < nn_ls; h++ ) {
                n2 = nn_list[n1 * nn_ls + h];     /* exchange partner, determine its position */
                if (route_node_map[n2] == FALSE) {
                    /* 该点不在本route中 */
                    continue;
                }
                if (radius - distance[n1 * num_node + n2] > EPSILON) {
                    pos_n2 = tour_node_pos[n2];
                    s_n2 = pos_n2 == rend ? tour[rbeg] : tour[pos_n2+1];
                    gain =  - radius + distance[n1 * num_node + n2] +
                    distance[s_n1 * num_node + s_n2] - distance[n2 * num_node + s_n2];
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
            radius = distance[p_n1 * num_node + n1];
            for ( h = 0 ; h < nn_ls ; h++ ) {
                n2 = nn_list[n1 * nn_ls + h];    /* exchange partner, determine its position */
                if (route_node_map[n2] == FALSE) {
                    /* 该点不在本route中 */
                    continue;
                }
                if ( radius - distance[n1 * num_node + n2] > EPSILON) {
                    pos_n2 = tour_node_pos[n2];
                    p_n2 = pos_n2 == rbeg ? tour[rend] : tour[pos_n2-1];
                    
                    if ( p_n2 == n1 || p_n1 == n2)
                        continue;
                    gain =  - radius + distance[n1 * num_node + n2] +
                    distance[p_n1 * num_node + p_n2] - distance[p_n2 * num_node + n2];
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
            n_exchanges++;
            improvement_flag = TRUE;
            dlb[h1] = FALSE; dlb[h2] = FALSE;
            dlb[h3] = FALSE; dlb[h4] = FALSE;
            /* Now perform move */
            if ( tour_node_pos[h3] < tour_node_pos[h1] ) {
                help = h1; h1 = h3; h3 = help;
                help = h2; h2 = h4; h4 = help;
            }
            /* reverse inner part from pos[h2] to pos[h3] */
            i = tour_node_pos[h2]; j = tour_node_pos[h3];
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
//            print_single_route(num_node, distance, tour + rbeg, num_route_node+1);
        }
        if ( improvement_flag ) {
            n_improves++;
        }
    }
}

/******** pheromone update ********/
inline void update_pheromone_weighted(int num_node, __global double *pheromone,
                                      __global int *tour, int tour_size,
                                      double tour_length, int weight)
{
    int      i, j, h;
    double   d_tau;

    d_tau = weight / tour_length;
    for (i = 0; i < tour_size - 1; i++) {
        j = tour[i];
        h = tour[i+1];
        pheromone[j * num_node + h] += d_tau;
    }
}

/*
 * 蚁群停滞时，加入扰动跳出局部最优解
 */
inline void pheromone_disturbance(int num_node, __global double *pheromone)
{
//    printf("begin pheromone disturbance...\n");
    
    int i;
    double mean_pheromone = 0.0;
    int sz = num_node * num_node;
    
    for (i = 0; i < sz; i++) {
        mean_pheromone += pheromone[i];
    }
    mean_pheromone = mean_pheromone / sz;
    
    for (i = 0; i < sz; i++) {
        pheromone[i] = (1- DELTA) * mean_pheromone + DELTA * pheromone[i];
    }
}

inline int find_best(int n_ants, __global double *solution_lens)
{
    double   min;
    int   k, k_min;
    
    min = solution_lens[0];
    k_min = 0;
    for(k = 1; k < n_ants; k++) {
        if(solution_lens[k] < min ) {
            min = solution_lens[k];
            k_min = k;
        }
    }
    return k_min;
}

/*********************************************************
                      kernels
 ********************************************************/

/*
 * construct the solution for an ant
* number of threads: n_ants
 */
__kernel void construct_solution(int num_node, int capacity, double max_dist, double serv_time,
                                 __global int *rnd_seeds, __global double *distance, __global int *demand,
                                 __global double *total_info, __global int *solutions, __global double *solution_lens)
{
    int     gid = get_global_id(0);
    int     max_tour_sz = 2 * num_node;
    int     beg = max_tour_sz * gid, end = max_tour_sz * (gid + 1);
    int     tour[max_tour_sz];
    int     tour_size;
    
    bool    visited[num_node];
    bool    demand_meet[num_node];
    double  prob_of_selection[num_node];
    
    int     visited_node_cnt = 0;   /* count of visited node by this ant */
    int     path_load;              /* 单次从depot出发的送货量 */
    int     next_node, current_node;
    int     i, demand_meet_cnt, step;
    double  path_distance;
    
    /* Mark all nodes as unvisited */
    tour_size = 0;
    for(i = 0; i < num_node; i++) {
        visited[i] = FALSE;
    }
    
    path_load = 0;
    path_distance = 0;
    step = 0;
    // init ant place
    tour[step] = 0;
    visited[0] = TRUE;
    
    while (visited_node_cnt < num_node - 1) {
        current_node = tour[step];
        step++;
        
        /* 查看所有可以派送的点 */
        demand_meet_cnt = 0;
        for (i = 0; i < num_node; i++) {
            demand_meet[i] = FALSE;
        }
        for(i = 0; i < num_node; i++) {
            if (visited[i] == FALSE
                && path_load + demand[i] <= capacity
                && path_distance + (distance[current_node * num_node + i] + serv_time) + distance[i * num_node + 0] <= max_dist) {
                demand_meet[i] = TRUE;
                demand_meet_cnt++;
            }
        }
        
        /*
         1)如果没有可行的配送点,则蚂蚁回到depot，重新开始新的路径
         2）否则，选择下一个配送点
         */
        if (demand_meet_cnt == 0) {
            path_load = 0;
            path_distance = 0;
            tour[step] = 0;
            visited[0] = TRUE;
        } else {
            next_node = choose_and_move_to_next(step, num_node, (rnd_seeds + gid), tour, total_info,
                                                prob_of_selection, visited, demand_meet);
            path_load += demand[next_node];
            path_distance += distance[current_node * num_node + next_node] + serv_time;
            visited_node_cnt++;
        }
    }
    
    // 最后回到depot
    step++;
    tour[step] = tour[0];
    tour_size = step + 1;
    
    // return solution
    for (i = 0; i < tour_size; i++) {
        solutions[i+beg] = tour[i];
    }
    solutions[end-1] = tour_size;
    
    solution_lens[gid] = compute_tour_length(num_node, distance, solutions + beg, tour_size);
//    printf("len:%lf \n", solution_lens[gid]);
}

/*
 * local search
 * number of threads: n_ants
 */
__kernel void local_search(int num_node, __global int *rnd_seeds,
                           int nn_ls, __global int *nn_list,
                           __global double *distance, __global int *solutions,
                           __global double *solution_lens)
{
    int     gid = get_global_id(0);
    int     max_tour_sz = 2 * num_node;
    int     beg = max_tour_sz * gid, end = max_tour_sz * (gid + 1);
    __global int *tour = solutions + beg;
    int     tour_size = solutions[end-1];
    
    /* vector containing don't look bits */
    bool dlb[num_node];
    /* mark for all nodes in a single route */
    bool route_node_map[num_node];
    /* positions of nodes in tour */
    int tour_node_pos[num_node];
    
    int route_beg = 0;
    int i, j;
    
    for (i = 0 ; i < num_node; i++) {
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
            two_opt_single_route(num_node, tour, route_beg, i-1, dlb,
                                 route_node_map, tour_node_pos, (rnd_seeds + gid),
                                 distance, nn_list, nn_ls);
            
            for (j = 0; j < num_node; j++) {
                route_node_map[j] = FALSE;
            }
            route_beg = i;
            
        } else {
            route_node_map[tour[i]] = TRUE;
        }
    }
    
    solution_lens[gid] = compute_tour_length(num_node, distance, tour, tour_size);
//    printf("len:%lf \n", solution_lens[gid]);
}

/*
 * update statistics 
 * number of threads: 1 
 */
__kernel void update_statistics(int num_node, int n_ants, __global int *solutions, __global double *solution_lens)
{
    __global int *best_tour, *iter_best_tour, *tmp_tour;
    int     tmp_tour_size;
    int     i, idx, max_tour_sz = 2 * num_node;
    
    
    // find the best solution in an iteration
    idx = find_best(n_ants, solution_lens);
    tmp_tour = solutions + max_tour_sz * idx;
    tmp_tour_size = tmp_tour[max_tour_sz - 1];
    
    // get iter best solution
    iter_best_tour = solutions + max_tour_sz * (n_ants + 1);
    for (i = 0; i < tmp_tour_size; i++) {
        iter_best_tour[i] = tmp_tour[i];
    }
    iter_best_tour[max_tour_sz - 1] = tmp_tour_size;
    solution_lens[n_ants+1] = solution_lens[idx];
    
    // upfate best-so-far solution
    best_tour = solutions + max_tour_sz * n_ants;
    if(solution_lens[idx] - solution_lens[n_ants] < -EPSILON){
        // get better solution
        for (i = 0; i < tmp_tour_size; i++) {
            best_tour[i] = iter_best_tour[i];
        }
        best_tour[max_tour_sz - 1] = tmp_tour_size;
        solution_lens[n_ants] = solution_lens[idx];
    }
}

/*
 * pheromone evaporation
 * number of threads: num_node * num_node
 */
__kernel void pheromone_evaporation(__global double *pheromone)
{
    int gid = get_global_id(0);
    
//    printf("begin pheromone_evaporation...\n");
    
    /* Simulate the pheromone evaporation of all pheromones */
    pheromone[gid] = (1 - RHO) * pheromone[gid];
    
}

/*
 * pheromone update
 * number of threads: 1
 */
__kernel void pheromone_update(int num_node, int n_ants,
                               __global int *solutions, __global double *solution_lens,
                               __global double *pheromone, int iter_stagnate_cnt)
{
    __global int *tour, *best_tour;
    int tour_size, best_tour_size;
    int max_tour_sz = 2 * num_node;
    
    int i, k, b, target;
    double help_b[n_ants];
    
    if (iter_stagnate_cnt >= 5)
    {
        pheromone_disturbance(num_node, pheromone);
//        iter_stagnate_cnt -= 2;
    } else {
        /* apply the pheromone deposit */
        
        for (k = 0; k < n_ants; k++) {
            help_b[k] = solution_lens[k];
        }
        for (i = 0; i < RAS_RANKS - 1; i++) {
            b = help_b[0];
            target = 0;
            for (k = 0; k < n_ants; k++) {
                if (help_b[k] < b) {
                    b = help_b[k];
                    target = k;
                }
            }
            help_b[target] = LONG_MAX;
            tour = solutions + max_tour_sz * k;
            tour_size = tour[max_tour_sz - 1];
            update_pheromone_weighted(num_node, pheromone, tour, tour_size, solution_lens[k], RAS_RANKS-i-1);
        }
        // best so far tour store at the end of solutions memory
        best_tour = solutions + max_tour_sz * n_ants;
        best_tour_size = best_tour[max_tour_sz - 1];
        update_pheromone_weighted(num_node, pheromone, best_tour, best_tour_size, solution_lens[n_ants], RAS_RANKS);
    }
}

/*
 * compute total info
 * number of threads: num_node * num_node
 */
__kernel void compute_total_info(__global double *pheromone, __global double *total_info, __global double *distance)
{
    int gid = get_global_id(0);
    
    /* Compute combined information pheromone times heuristic info after
     the pheromone update for ACO algorithm */
    total_info[gid] = pow((float)pheromone[gid], (float)ALPHA) * pow((float)(1.0/(distance[gid]+0.1)), (float)BETA);
}