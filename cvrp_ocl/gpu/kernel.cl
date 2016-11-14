//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define TRUE true
#define FALSE false

#define EPSILON 0.001f
#define DEBUG(x)   x
/****** parameters ******/
#define RHO         0.1f
#define ALPHA       1.0f
#define BETA        2.0f
#define RAS_RANKS   6
#define DELTA       0.7f
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

inline float ran01(__global int *idum )
{
    int k;
    float ans;
    
    k =(*idum)/IQ;
    *idum = IA * (*idum - k * IQ) - IR * k;
    if (*idum < 0 ) *idum += IM;
    ans = AM * (*idum);
    return ans;
}

inline float compute_tour_length(__global float *distance,
                                  __global int *tour, int tour_size)
{
    int      i;
    float   tour_length = 0;
    
    for ( i = 0 ; i < tour_size-1; i++ ) {
        tour_length += distance[tour[i] * NUM_NODE + tour[i+1]];
    }
    return tour_length;
}

/******** print ********/
inline void print_solution(__global float *distance, __global int *tour, int tour_size)
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

inline void print_single_route(__global float *distance, __global int *route, int route_size)
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
inline int choose_best_next(int phase, __global int *tour,__global float *total_info,
                            bool *visited, bool *demand_meet)
{
    int node, current_node, next_node;
    float   value_best;
    
    next_node = NUM_NODE;
    current_node = tour[phase-1];
    value_best = -1.;             /* values in total matrix are always >= 0.0 */
    for ( node = 0 ; node < NUM_NODE ; node++ ) {
        if (visited[node]) {
            ; /* node already visited, do nothing */
        } else if(demand_meet[node] == FALSE) {
            ;  /* 该点不满足要求 */
        } else {
            if ( total_info[current_node * NUM_NODE + node] > value_best ) {
                next_node = node;
                value_best = total_info[current_node * NUM_NODE + node];
            }
        }
    }
    tour[phase] = next_node;
    visited[next_node] = TRUE;
    
    return next_node;
}

inline int choose_and_move_to_next(int phase, __global int *rnd_seed, __global int *tour,
                                   __global float *total_info, float *prob_ptr,
                                   bool *visited, bool *demand_meet)
{
    int i;
    int current_node;
    float partial_sum = 0.0f, sum_prob = 0.0f;
    float rnd;
    
    current_node = tour[phase-1];           /* current_node node of ant k */
    for (i = 0 ; i < NUM_NODE; i++) {
        if (visited[i]) {
            prob_ptr[i] = 0.0f;   /* node already visited */
        } else if(demand_meet[i] == FALSE) {
            prob_ptr[i] = 0.0f;  /* 该点不满足要求 */
        } else {
            prob_ptr[i] = total_info[current_node * NUM_NODE + i];
            sum_prob += prob_ptr[i];
        }
    }

    
    if (sum_prob <= 0.0f) {
        /* All nodes from the candidate set are tabu */
        return choose_best_next(phase, tour, total_info, visited, demand_meet);
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
//inline void generate_random_permutation(int n, int *r, __global int *rnd_seed)
//{
//    int  i, help, node, tot_assigned = 0;
//    float    rnd;
//
//    for ( i = 0 ; i < n; i++){
//        r[i] = i;
//    }
//    for ( i = 0 ; i < n ; i++ ) {
//        /* find (randomly) an index for a free unit */
//        rnd  = ran01(rnd_seed);
//        node = (int) (rnd  * (n - tot_assigned));
//
//        help = r[i];
//        r[i] = r[i+node];
//        r[i+node] = help;
//        tot_assigned++;
//    }
//}

inline void two_opt_single_route(__global int *tour, int rbeg, int rend,
                                 bool *dlb, bool *route_node_map, int *tour_node_pos,
                                 __global int *rnd_seed, __global float *distance,
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
    float radius;             /* radius of nn-search */
    float gain = 0;
//    int random_vector[num_route_node];
    
//    print_single_route(NUM_NODE, distance, tour + rbeg, num_route_node+1);
    
    improvement_flag = TRUE;
//    generate_random_permutation(num_route_node, random_vector, rnd_seed);

    while ( improvement_flag ) {

        improvement_flag = FALSE;
        
        for (l = 0 ; l < num_route_node; l++) {
            
            /* 1) the neighbourhood is scanned in random order */
//            pos_n1 = rbeg + random_vector[l];
            /* 2)the neighbourhood is scanned in sequence order */
            pos_n1 = rbeg + l;
            n1 = tour[pos_n1];
            if (dlb[n1]) {
                continue;
            }
            s_n1 = pos_n1 == rend ? tour[rbeg] : tour[pos_n1+1];
            radius = distance[n1 * NUM_NODE + s_n1];
            
            /* First search for n1's nearest neighbours, use successor of n1 */
            for ( h = 0; h < nn_ls; h++ ) {
                n2 = nn_list[n1 * nn_ls + h];     /* exchange partner, determine its position */
                if (route_node_map[n2] == FALSE) {
                    /* 该点不在本route中 */
                    continue;
                }
                if (radius - distance[n1 * NUM_NODE + n2] > EPSILON) {
                    pos_n2 = tour_node_pos[n2];
                    s_n2 = pos_n2 == rend ? tour[rbeg] : tour[pos_n2+1];
                    gain =  - radius + distance[n1 * NUM_NODE + n2] +
                    distance[s_n1 * NUM_NODE + s_n2] - distance[n2 * NUM_NODE + s_n2];
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
            radius = distance[p_n1 * NUM_NODE + n1];
            for ( h = 0; h < nn_ls; h++ ) {
                n2 = nn_list[n1 * nn_ls + h];    /* exchange partner, determine its position */
                if (route_node_map[n2] == FALSE) {
                    /* 该点不在本route中 */
                    continue;
                }
                if ( radius - distance[n1 * NUM_NODE + n2] > EPSILON) {
                    pos_n2 = tour_node_pos[n2];
                    p_n2 = pos_n2 == rbeg ? tour[rend] : tour[pos_n2-1];
                    
                    if ( p_n2 == n1 || p_n1 == n2)
                        continue;
                    gain =  - radius + distance[n1 * NUM_NODE + n2] +
                    distance[p_n1 * NUM_NODE + p_n2] - distance[p_n2 * NUM_NODE + n2];
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
//            print_single_route(NUM_NODE, distance, tour + rbeg, num_route_node+1);
        }
        if ( improvement_flag ) {
            n_improves++;
        }
    }
}

inline void two_opt_solution(__global int *tour, int tour_size, __global float *distance,
                             __global int *rnd_seed, __global int *nn_list, int nn_ls)
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
                                 route_node_map, tour_node_pos, rnd_seed,
                                 distance, nn_list, nn_ls);
            
            for (j = 0; j < NUM_NODE; j++) {
                route_node_map[j] = FALSE;
            }
            route_beg = i;
            
        } else {
            route_node_map[tour[i]] = TRUE;
        }
    }
}

/*
 * The swap operation selects two customers at random and
 * then swaps these two customers in their positions.
 */
inline void swap(__global int *tour, int tour_size,
                 __global float *distance, __global int *demand,
                 int capacity, float max_dist, float service_time)
{
    /* array of single route load */
    int route_load[NUM_NODE-1];
    /* array of single route distance */
    float route_dist[NUM_NODE-1];
    int beg;
    int load = 0, load1 = 0, load2 = 0;
    float dist = 0, dist1 = 0, dist2 = 0;

    int i = 0, j = 0, k = 0;
    float gain = 0;
    int n1, p_n1, s_n1, n2, p_n2, s_n2;
    int p1 = 0, p2 = 0;     /* path idx of node n1 and n2 */
    
    
    // 1) step 1: 获取load/distance array
    load = 0;
    dist = 0;
    k = 0;
    beg = 0;
    for (i = 1; i < tour_size; i++) {
        load += demand[tour[i]];
        dist += distance[tour[i-1] * NUM_NODE + tour[i]];
        
        if (tour[i] == 0) {
            route_load[k] = load;
            route_dist[k] = dist + service_time * (i - beg - 1);

            k++;
            load = 0;
            dist = 0;
            beg = i;
        }
    }
    
    // 2）step 2: swap
    for (i = 1; i < tour_size; i++) {
        n1 = tour[i];
        if (n1 == 0) {
            p1++;
            continue;
        }
        p_n1 = tour[i-1];
        s_n1 = tour[i+1];
        
        
        p2 = p1;
        for (j = i+1; j < tour_size; j++) {
            n2 = tour[j];
            if (n2 == 0) {
                p2++;
                continue;
            }
            p_n2 = tour[j-1];
            s_n2 = tour[j+1];
            
            // calulate gain
            if (j == i + 1) {
                gain = -(distance[p_n1 * NUM_NODE + n1] + distance[n2 * NUM_NODE + s_n2])
                + (distance[p_n1 * NUM_NODE + n2] + distance[n1 * NUM_NODE + s_n2]);
            } else {
                gain = -(distance[p_n1 * NUM_NODE + n1] + distance[n1 * NUM_NODE + s_n1]
                         + distance[p_n2 * NUM_NODE + n2] + distance[n2 * NUM_NODE + s_n2])
                + (distance[p_n1 * NUM_NODE + n2] + distance[n2 * NUM_NODE + s_n1]
                   + distance[p_n2 * NUM_NODE + n1] + distance[n1 * NUM_NODE + s_n2]);
            }
            if (gain < -EPSILON) {
                
                // node n1 and n2 not in the same route
                if (p1 != p2) {
                    load1 = route_load[p1] - demand[n1] + demand[n2];
                    load2 = route_load[p2] - demand[n2] + demand[n1];
                    
                    dist1 = route_dist[p1] - (distance[p_n1 * NUM_NODE + n1] + distance[n1 * NUM_NODE + s_n1])
                    + (distance[p_n1 * NUM_NODE + n2] + distance[n2 * NUM_NODE + s_n1]);
                    
                    dist2 = route_dist[p2] - (distance[p_n2 * NUM_NODE + n2] + distance[n2 * NUM_NODE + s_n2])
                    + (distance[p_n2 * NUM_NODE + n1] + distance[n1 * NUM_NODE + s_n2]);
                    
                    if ((load1 > capacity || load2 > capacity) || (dist1 > max_dist || dist2 > max_dist)) {
                        continue;
                    }
                } else {
                    load1 = route_load[p1];
                    load2 = load1;
                    
                    dist1 = route_dist[p1] + gain;   // gain < 0, so dont need to check max_distance limit
                    dist2 = dist1;
                }
                
                
                tour[i] = n2;
                tour[j] = n1;
                
                route_load[p1] = load1;
                route_load[p2] = load2;
                
                route_dist[p1] = dist1;
                route_dist[p2] = dist2;
                
                i--;
                //                print_solution(instance, tour, tour_size);
                break;
            }
        }
    }
}

/******** pheromone update ********/
//inline void update_pheromone_weighted(__global float *pheromone,
//                                      __global int *tour, int tour_size,
//                                      float tour_length, int weight)
//{
//    int      i, j, h;
//    float   d_tau;
//
//    d_tau = weight / tour_length;
//    for (i = 0; i < tour_size - 1; i++) {
//        j = tour[i];
//        h = tour[i+1];
//        pheromone[j * NUM_NODE + h] += d_tau;
//    }
//}


inline int find_best(__global float *solution_lens)
{
    float   min;
    int   k, k_min;
    
    min = solution_lens[0];
    k_min = 0;
    for(k = 1; k < N_ANTS; k++) {
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
* number of threads: N_ANTS
 */
__kernel void construct_solution(int capacity, float max_dist, float serv_time,
                                 __global int *rnd_seeds, __global float *distance, __global int *demand,
                                 __global float *total_info, __global int *solutions, __global float *solution_lens)
{
//    DEBUG(printf("begin constructing solution\n");)
    
    int     gid = get_global_id(0);
    int     max_tour_sz = 2 * NUM_NODE;
    __global int  *tour = solutions + max_tour_sz * gid;
    int     tour_size;
    
    bool    visited[NUM_NODE];
    bool    demand_meet[NUM_NODE];
    float  prob_of_selection[NUM_NODE];
    
    int     visited_node_cnt = 0;   /* count of visited node by this ant */
    int     path_load;              /* 单次从depot出发的送货量 */
    int     next_node, current_node;
    int     i, demand_meet_cnt, step;
    float  path_distance;
    
    /* Mark all nodes as unvisited */
    tour_size = 0;
    for(i = 0; i < NUM_NODE; i++) {
        visited[i] = FALSE;
    }
    
    path_load = 0;
    path_distance = 0;
    step = 0;
    // init ant place
    tour[step] = 0;
    visited[0] = TRUE;
    
    while (visited_node_cnt < NUM_NODE - 1) {
        current_node = tour[step];
        step++;
        
        /* 查看所有可以派送的点 */
        demand_meet_cnt = 0;
        for (i = 0; i < NUM_NODE; i++) {
            demand_meet[i] = FALSE;
        }
        for(i = 0; i < NUM_NODE; i++) {
            if (visited[i] == FALSE
                && path_load + demand[i] <= capacity
                && path_distance + (distance[current_node * NUM_NODE + i] + serv_time) + distance[i * NUM_NODE + 0] <= max_dist) {
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
            next_node = choose_and_move_to_next(step, (rnd_seeds + gid), tour, total_info,
                                                prob_of_selection, visited, demand_meet);
            path_load += demand[next_node];
            path_distance += distance[current_node * NUM_NODE + next_node] + serv_time;
            visited_node_cnt++;
        }
    }
    
    // 最后回到depot
    step++;
    tour[step] = tour[0];
    tour_size = step + 1;

    tour[max_tour_sz-1] = tour_size;
    
    solution_lens[gid] = compute_tour_length(distance, tour, tour_size);
//    printf("len:%lf \n", solution_lens[gid]);
}

/*
 * local search
 * number of threads: N_ANTS
 */
__kernel void local_search(__global int *rnd_seeds, int nn_ls, __global int *nn_list,
                           __global float *distance, __global int *solutions, __global float *solution_lens,
                           __global int *demand, int capacity, float max_dist, float serv_time)
{
    int     gid = get_global_id(0);
    int     max_tour_sz = 2 * NUM_NODE;
    __global int *tour = solutions + max_tour_sz * gid;
    int     tour_size = tour[max_tour_sz-1];
    
    // 1) 2-opt local search
    two_opt_solution(tour, tour_size, distance, rnd_seeds + gid, nn_list, nn_ls);
    
    // 2) swap - exchange 2 nodes in a tour
    swap(tour, tour_size, distance, demand, capacity, max_dist, serv_time);
    
    solution_lens[gid] = compute_tour_length(distance, tour, tour_size);
//    printf("len:%lf \n", solution_lens[gid]);
}

/*
 * update statistics 
 * number of threads: 1 
 */
__kernel void update_statistics(__global int *solutions, __global float *solution_lens)
{
    __global int *best_tour, *iter_best_tour;
    int     tour_size;
    int     i, idx, max_tour_sz = 2 * NUM_NODE;
    
    
    // find the iter-best solution
    idx = find_best(solution_lens);
    iter_best_tour = solutions + max_tour_sz * idx;
    tour_size = iter_best_tour[max_tour_sz - 1];
 
    // update iter-best solution id
    solutions[max_tour_sz * (N_ANTS + 1)] = idx;
    
    // update best-so-far solution
    best_tour = solutions + max_tour_sz * N_ANTS;
    if(solution_lens[idx] - solution_lens[N_ANTS] < -EPSILON){
        // get better solution
        for (i = 0; i < tour_size; i++) {
            best_tour[i] = iter_best_tour[i];
        }
        best_tour[max_tour_sz - 1] = tour_size;
        solution_lens[N_ANTS] = solution_lens[idx];
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
    total_info[gid] = pow((float)pheromone[gid], (float)ALPHA) * pow((float)(1.0/(distance[gid]+0.1)), (float)BETA);
}

/*
 * pheromone evaporation
 * number of threads: NUM_NODE * NUM_NODE
 */
__kernel void pheromone_evaporation(__global float *pheromone)
{
    int gid = get_global_id(0);
    
//    printf("begin pheromone_evaporation...\n");
    
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
 * pheromone update
 * number of threads: 1
 */
__kernel void ras_update(__global int *solutions, __global float *solution_lens,
                         __global float *pheromone)
{
    __global int *tour, *best_tour;
    int tour_size, best_tour_size;
    int max_tour_sz = 2 * NUM_NODE;
    
    int i, k, target;
    float help_b[N_ANTS], b;


    /* apply the pheromone deposit */
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
        tour = solutions + max_tour_sz * target;
        tour_size = tour[max_tour_sz - 1];
        update_pheromone_weighted(pheromone, tour, tour_size, solution_lens[target], RAS_RANKS-i-1);
    }
    
    // best so far tour store at the end of solutions memory
    best_tour = solutions + max_tour_sz * N_ANTS;
    best_tour_size = best_tour[max_tour_sz - 1];
    update_pheromone_weighted(pheromone, best_tour, best_tour_size, solution_lens[N_ANTS], RAS_RANKS);
}

/*
 * 蚁群停滞时，加入扰动跳出局部最优解
 *  number of threads: 1
 */
__kernel void pheromone_disturbance(__global float *pheromone)
{
//    printf("begin pheromone disturbance...\n");
    int i;
    float mean_pheromone = 0.0;
    int sz = NUM_NODE * NUM_NODE;
    
    for (i = 0; i < sz; i++) {
        mean_pheromone += pheromone[i];
    }
    mean_pheromone = mean_pheromone / sz;

    for (i = 0; i < sz; i++) {
        pheromone[i] = (1- DELTA) * mean_pheromone + DELTA * pheromone[i];
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
    total_info[gid] = pow((float)pheromone[gid], (float)ALPHA) * pow((float)(1.0/(distance[gid]+0.1)), (float)BETA);
}

/*
 * update host best-so-far solution to device memory
 * number of threads: 1
 */
__kernel void update_best_so_far_to_mem(__global int *from_tour, int tour_size, float tour_length,
                                        __global int *solutions, __global float *solution_lens)
{
    int max_tour_sz = 2 * NUM_NODE;
    __global int *best_tour = solutions + max_tour_sz * N_ANTS;
    
    for(int i = 0; i < tour_size; i++) {
        best_tour[i] = from_tour[i];
    }
    best_tour[max_tour_sz-1] = tour_size;
    
    solution_lens[N_ANTS] = tour_length;
}
