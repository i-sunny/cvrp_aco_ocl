//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define TRUE true
#define FALSE false

#define EPSILON     0.001f
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

#define MAX_TOUR_SZ  (2 * NUM_NODE)

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

    
    /* chose one according to the selection probabilities */
    rnd = ran01(rnd_seed) * sum_prob;
    
    i = 0;
    partial_sum = candidate[i] * total_info_wrk[i];
    while (partial_sum < rnd) {
        i++;
        partial_sum += candidate[i] * total_info_wrk[i];
    }
    
    return i;
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

/*
 * The swap operation selects two customers at random and
 * then swaps these two customers in their positions.
 */
inline void swap(int *tour, int tour_size,
                 __global float *distance, __global int *demands,
                 const int capacity, const float max_dist, const float service_time)
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
        load += demands[tour[i]];
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
                    load1 = route_load[p1] - demands[n1] + demands[n2];
                    load2 = route_load[p2] - demands[n2] + demands[n1];
                    
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
        visited_cnt = 0;
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
        current_node = tour[step];
        
        while (visited_cnt < NUM_NODE - 1) {
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
    
    /* 
     * 2) swap - exchange 2 nodes in a tour
     * 暂时不使用swap()操作, 且swap()操作没有做内存优化
     * 注意: 由于没有使用dist_tour_wrk对swap()进行优化,
     * 因此如果直接使用没有内存优化的swap(), 则计算solution length的方式应该是：
     * compute_tour_length(distance, tour_wrk, tour_size)
     */
//    swap(tour_wrk, tour_size, distance, demands, capacity, max_dist, serv_time);
    
    // !!update global memory
    for(int i = 0; i < tour_size; i++) {
        tour[i] = tour_wrk[i];
    }
    
    float len = 0;
    for (i = 0; i < tour_size-1; i++) {
        len += dist_tour_wrk[i];
    }
    solution_lens[gid] = len;
    
//    solution_lens[gid] = compute_tour_length(distance, tour_wrk, tour_size);
//    printf("len:%f \n", solution_lens[gid]);
    
}

/*
 * update statistics 
 * number of threads: 1 
 */
__kernel void update_statistics(__global int *solutions, __global float *solution_lens)
{
    __global int *best_tour, *iter_best_tour;
    int     tour_size;
    int     i, idx;
    
    
    // find the iter-best solution
    idx = find_best(solution_lens);
    iter_best_tour = solutions + MAX_TOUR_SZ * idx;
    tour_size = iter_best_tour[MAX_TOUR_SZ - 1];
 
    // update iter-best solution id
    solutions[MAX_TOUR_SZ * (N_ANTS + 1)] = idx;
    
    // update best-so-far solution
    best_tour = solutions + MAX_TOUR_SZ * N_ANTS;
    if(solution_lens[idx] - solution_lens[N_ANTS] < -EPSILON){
        // get better solution
        for (i = 0; i < tour_size; i++) {
            best_tour[i] = iter_best_tour[i];
        }
        best_tour[MAX_TOUR_SZ - 1] = tour_size;
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
    total_info[gid] = pow((float)pheromone[gid], (float)ALPHA) * pow((float)(1.0f/(distance[gid]+0.1f)), (float)BETA);
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
        tour = solutions + MAX_TOUR_SZ * target;
        tour_size = tour[MAX_TOUR_SZ - 1];
        update_pheromone_weighted(pheromone, tour, tour_size, solution_lens[target], RAS_RANKS-i-1);
    }
    
    // best so far tour store at the end of solutions memory
    best_tour = solutions + MAX_TOUR_SZ * N_ANTS;
    best_tour_size = best_tour[MAX_TOUR_SZ - 1];
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
    float mean_pheromone = 0.0f;
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
    total_info[gid] = pow((float)pheromone[gid], (float)ALPHA) * pow((float)(1.0f/(distance[gid]+0.1f)), (float)BETA);
}

/*
 * update host best-so-far solution to device memory
 * number of threads: 1
 */
__kernel void update_best_so_far_to_mem(__global int *from_tour, int tour_size, float tour_length,
                                        __global int *solutions, __global float *solution_lens)
{
    __global int *best_tour = solutions + MAX_TOUR_SZ * N_ANTS;
    
    for(int i = 0; i < tour_size; i++) {
        best_tour[i] = from_tour[i];
    }
    best_tour[MAX_TOUR_SZ-1] = tour_size;
    
    solution_lens[N_ANTS] = tour_length;
}
