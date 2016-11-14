/*********************************
 
 Ant Colony Optimization algorithms (AS, ACS, EAS, RAS, MMAS, BWAS) for CVRP
 
 Created by 孙晓奇 on 2016/10/8.
 Copyright © 2016年 xiaoqi.sxq. All rights reserved.
 
 Program's name: acovrp
 Purpose: local search routines
 
 email: sunxq1991@gmail.com
 
 *********************************/

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <limits.h>
#include <assert.h>

#include "localSearch.h"
#include "utilities.h"
#include "vrpHelper.h"
#include "io.h"

LocalSearch::LocalSearch(Problem *instance) {
    this->instance = instance;
    
    ants = instance->ants;
    n_ants = instance->n_ants;
    ls_flag = instance->ls_flag;
    num_node = instance->num_node;
    nn_list = instance->nn_list;
    nn_ls = instance->nn_ls;
    dlb_flag = instance->dlb_flag;
    distance = instance->distance;
}

/*
 FUNCTION:       manage the local search phase; apply local search to ALL ants; in
 dependence of ls_flag one of 2-opt local search is chosen.
 INPUT:          none
 OUTPUT:         none
 (SIDE)EFFECTS:  all ants of the colony have locally optimal tours
 COMMENTS:       typically, best performance is obtained by applying local search
 to all ants. It is known that some improvements (e.g. convergence
 speed towards high quality solutions) may be obtained for some
 ACO algorithms by applying local search to only some of the ants.
 Overall best performance is typcially obtained by using 3-opt.
 */
void LocalSearch::do_local_search(void)
{
    int k;
    
    TRACE ( printf("apply local search to all ants\n"); );
    
    for ( k = 0 ; k < n_ants ; k++ ) {
        //debug
//        printf("\n--Before local search:");
//        if(check_solution(instance, ants[k].tour, ants[k].tour_size)) {
//            print_solution(instance, ants[k].tour, ants[k].tour_size);
//        }
        
        if (ls_flag) {
             // 1) 2-opt local search
            two_opt_solution(ants[k].tour, ants[k].tour_size);
            
            //2) swap - exchange 2 nodes in the tour
            swap(ants[k].tour, ants[k].tour_size);
            
            ants[k].tour_length = compute_tour_length(instance, ants[k].tour, ants[k].tour_size);
        }
        
        //debug
//        printf("\n--After local search:");
        DEBUG(assert(check_solution(instance, ants[k].tour, ants[k].tour_size));)
    }
}

/*
 * apply loacal search to a single ant
 */
void LocalSearch::do_local_search(AntStruct *ant)
{
    two_opt_solution(ant->tour, ant->tour_size);
    ant->tour_length = compute_tour_length(instance, ant->tour, ant->tour_size);
}

/*
 FUNCTION:       generate a random permutation of the integers 0 .. n-1
 INPUT:          length of the array
 OUTPUT:         pointer to the random permutation
 (SIDE)EFFECTS:  the array holding the random permutation is allocated in this
 function. Don't forget to free again the memory!
 COMMENTS:       only needed by the local search procedures
 */
int * LocalSearch::generate_random_permutation( int n )
{
   int  i, help, node, tot_assigned = 0;
   double    rnd;
   int  *r;

   r = (int *)malloc(n * sizeof(int));

   for ( i = 0 ; i < n; i++) 
     r[i] = i;

   for ( i = 0 ; i < n ; i++ ) {
     /* find (randomly) an index for a free unit */ 
     rnd  = ran01 ( &instance->rnd_seed );
     node = (int) (rnd  * (n - tot_assigned)); 
     assert( i + node < n );
     help = r[i];
     r[i] = r[i+node];
     r[i+node] = help;
     tot_assigned++;
   }
   return r;
}

/*
 FUNCTION:       2-opt all routes of an ant's solution.
 INPUT:          tour, an ant's solution
                 depotId
 OUTPUT:         none
 */
void LocalSearch::two_opt_solution(int *tour, int tour_size)
{
    int *dlb;               /* vector containing don't look bits */
    int *route_node_map;    /* mark for all nodes in a single route */
    int *tour_node_pos;     /* positions of nodes in tour */

    int route_beg = 0;
    
    dlb = (int *)malloc(num_node * sizeof(int));
    for (int i = 0 ; i < num_node; i++) {
        dlb[i] = FALSE;
    }
    
    route_node_map =  (int *)malloc(num_node * sizeof(int));
    for (int j = 0; j < num_node; j++) {
        route_node_map[j] = FALSE;
    }
    
    tour_node_pos =  (int *)malloc(num_node * sizeof(int));
    for (int j = 0; j < tour_size; j++) {
        tour_node_pos[tour[j]] = j;
    }
    
    for (int i = 1; i < tour_size; i++) {
        // 2-opt a single route from tour
        if (tour[i] == 0) {
            tour_node_pos[0] = route_beg;
            two_opt_single_route(tour, route_beg, i-1, dlb, route_node_map, tour_node_pos);
            
            for (int j = 0; j < num_node; j++) {
                route_node_map[j] = FALSE;
            }
            route_beg = i;
            
        } else {
            route_node_map[tour[i]] = TRUE;
        }
    }
    
    free( dlb );
    free(route_node_map);
    free(tour_node_pos);
}

/*
 FUNCTION:       2-opt a single route from an ant's solution.
                 This heuristic is applied separately to each
 of the vehicle routes built by an ant.
 INPUT:          rbeg, route的起始位置
                 rend, route的结束位置(包含rend处的点, rend处不为0)
 OUTPUT:         none
 COMMENTS:       the neighbourhood is scanned in random order
 */
void LocalSearch::two_opt_single_route(int *tour, int rbeg, int rend,
                          int *dlb, int *route_node_map, int *tour_node_pos)
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
    int *random_vector;

    // debug
//    print_single_route(instance, tour + rbeg, num_route_node+1);

    improvement_flag = TRUE;
    random_vector = generate_random_permutation(num_route_node);

    while ( improvement_flag ) {

        improvement_flag = FALSE;

        for (l = 0 ; l < num_route_node; l++) {

            /* the neighbourhood is scanned in random order */
            pos_n1 = rbeg + random_vector[l];
            n1 = tour[pos_n1];
            if (dlb_flag && dlb[n1])
                continue;
            
            s_n1 = pos_n1 == rend ? tour[rbeg] : tour[pos_n1+1];
            radius = distance[n1][s_n1];
            /* First search for c1's nearest neighbours, use successor of n1 */
            for ( h = 0 ; h < nn_ls ; h++ ) {
                n2 = nn_list[n1][h]; /* exchange partner, determine its position */
                if (route_node_map[n2] == FALSE) {
                    /* 该点不在本route中 */
                    continue;
                }
                if (radius - distance[n1][n2] > EPSILON) {
                    pos_n2 = tour_node_pos[n2];
                    s_n2 = pos_n2 == rend ? tour[rbeg] : tour[pos_n2+1];
                    gain =  - radius + distance[n1][n2] +
                            distance[s_n1][s_n2] - distance[n2][s_n2];
                    if ( gain < -EPSILON ) {
                        h1 = n1; h2 = s_n1; h3 = n2; h4 = s_n2;
                        goto exchange2opt;
                    }
                }
                else break;
            }
            
            /* Search one for next c1's h-nearest neighbours, use predecessor n1 */
            p_n1 = pos_n1 == rbeg ? tour[rend] : tour[pos_n1-1];
            radius = distance[p_n1][n1];
            for ( h = 0 ; h < nn_ls ; h++ ) {
                n2 = nn_list[n1][h];  /* exchange partner, determine its position */
                if (route_node_map[n2] == FALSE) {
                    /* 该点不在本route中 */
                    continue;
                }
                if ( radius - distance[n1][n2] > EPSILON) {
                    pos_n2 = tour_node_pos[n2];
                    p_n2 = pos_n2 == rbeg ? tour[rend] : tour[pos_n2-1];
                    
                    if ( p_n2 == n1 || p_n1 == n2)
                        continue;
                    gain =  - radius + distance[n1][n2] +
                            distance[p_n1][p_n2] - distance[p_n2][n2];
                    if ( gain < -EPSILON ) {
                        h1 = p_n1; h2 = n1; h3 = p_n2; h4 = n2;
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
//            printf("after ls. pid %d gain %f\n", instance->pid, gain);
//            print_single_route(instance, tour + rbeg, num_route_node+1);
        }
        if ( improvement_flag ) {
            n_improves++;
        }
    }
    free( random_vector );
}


/*
 * The swap operation selects two customers at random and 
 * then swaps these two customers in their positions.
 */
void LocalSearch::swap(int *tour, int tour_size)
{
    /* array of single route load */
    int *route_load = new int[num_node-1];
    /* array of single route distance */
    double *route_dist = new double[num_node-1];
    int beg;
    int load = 0, load1 = 0, load2 = 0;
    double dist = 0, dist1 = 0, dist2 = 0;
    
    Point *nodes = instance->nodeptr;
    
    int i = 0, j = 0, k = 0;
    double gain = 0;
    int n1, p_n1, s_n1, n2, p_n2, s_n2;
    int p1 = 0, p2 = 0;     /* path idx of node n1 and n2 */
    
    
    // 1) step 1: 获取load/distance array
    load = 0;
    dist = 0;
    k = 0;
    beg = 0;
    for (i = 1; i < tour_size; i++) {
        load += nodes[tour[i]].demand;
        dist += distance[tour[i-1]][tour[i]];
        
        if (tour[i] == 0) {
            route_load[k] = load;
            route_dist[k] = dist + instance->service_time * (i - beg - 1);
            DEBUG(assert(route_dist[k] == compute_route_length(instance, tour + beg, i - beg + 1) + instance->service_time * (i - beg - 1));)
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
        
        DEBUG(assert(n1 > 0);)
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
                gain = -(distance[p_n1][n1] + distance[n2][s_n2]) + (distance[p_n1][n2] + distance[n1][s_n2]);
            } else {
                gain = -(distance[p_n1][n1] + distance[n1][s_n1] + distance[p_n2][n2] + distance[n2][s_n2])
                +(distance[p_n1][n2] + distance[n2][s_n1] + distance[p_n2][n1] + distance[n1][s_n2]);
            }
            if (gain < -EPSILON) {
                
                // node n1 and n2 not in the same route
                if (p1 != p2) {
                    load1 = route_load[p1] - nodes[n1].demand + nodes[n2].demand;
                    load2 = route_load[p2] - nodes[n2].demand + nodes[n1].demand;
                    
                    dist1 = route_dist[p1] - (distance[p_n1][n1] + distance[n1][s_n1]) + (distance[p_n1][n2] + distance[n2][s_n1]);
                    dist2 = route_dist[p2] - (distance[p_n2][n2] + distance[n2][s_n2]) + (distance[p_n2][n1] + distance[n1][s_n2]);
                    
                    if ((load1 > instance->vehicle_capacity || load2 > instance->vehicle_capacity)
                        || (dist1 > instance->max_distance || dist2 > instance->max_distance)) {
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
                
                DEBUG(assert(check_solution(instance, tour, tour_size));)
                
    //                print_solution(instance, tour, tour_size);
                break;
            }
        }
    }
    delete[] route_load;
    delete[] route_dist;
}

