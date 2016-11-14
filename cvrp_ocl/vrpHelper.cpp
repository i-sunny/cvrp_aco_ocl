/*********************************
Ant Colony Optimization algorithms (AS, ACS, EAS, RAS, MMAS) for CVRP

Created by 孙晓奇 on 2016/10/8.
Copyright © 2016年 xiaoqi.sxq. All rights reserved.

Program's name: acovrp
Purpose: vrp related procedures, distance computation, neighbour lists

email: sunxq1991@gmail.com

*********************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <assert.h>
#include <vector>

#include "vrpHelper.h"
#include "utilities.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846264
#endif

double distance(Point *nodeptr, int i, int j, DistanceTypeEnum type);
double round_distance (Point *nodeptr, int i, int j);
int ceil_distance (Point *nodeptr, int i, int j);
int geo_distance (Point *nodeptr, int i, int j);
int att_distance (Point *nodeptr, int i, int j);


static double dtrunc (double x)
{
    int k;

    k = (int) x;
    x = (double) k;
    return x;
}


/*
 * 统一的距离计算入口
 */
double distance(Point *nodeptr, int i, int j, DistanceTypeEnum type)
{
    
    switch(type) {
        case DIST_EUC_2D: return round_distance(nodeptr, i, j);
        case DIST_CEIL_2D: return ceil_distance(nodeptr, i, j);
        case DIST_GEO: return geo_distance(nodeptr, i, j);
        case DIST_ATT:  return att_distance(nodeptr, i, j);
        default: return round_distance(nodeptr, i, j);
    }
}

/*
      FUNCTION: the following four functions implement different ways of 
                computing distances for VRPLIB instances
      INPUT:    two node indices
      OUTPUT:   distance between the two nodes
*/

double round_distance (Point *nodeptr, int i, int j)
/*    
      FUNCTION: compute Euclidean distances between two nodes rounded to next 
                integer for VRPLIB instances
      INPUT:    two node indices
      OUTPUT:   distance between the two nodes
      COMMENTS: for the definition of how to compute this distance see VRPLIB
*/
{
    double xd = nodeptr[i].x - nodeptr[j].x;
    double yd = nodeptr[i].y - nodeptr[j].y;
    double r  = sqrt(xd*xd + yd*yd);

    return r;
}

int ceil_distance (Point *nodeptr, int i, int j)
/*    
      FUNCTION: compute ceiling distance between two nodes rounded to next 
                integer for VRPLIB instances
      INPUT:    two node indices
      OUTPUT:   distance between the two nodes
      COMMENTS: for the definition of how to compute this distance see VRPLIB
*/
{
    double xd = nodeptr[i].x - nodeptr[j].x;
    double yd = nodeptr[i].y - nodeptr[j].y;
    double r  = sqrt(xd*xd + yd*yd);

    return (int)(ceil (r));
}

int geo_distance (Point *nodeptr, int i, int j)
/*    
      FUNCTION: compute geometric distance between two nodes rounded to next 
                integer for VRPLIB instances
      INPUT:    two node indices
      OUTPUT:   distance between the two nodes
      COMMENTS: adapted from concorde code
                for the definition of how to compute this distance see VRPLIB
*/
{
    double deg, min;
    double lati, latj, longi, longj;
    double q1, q2, q3;
    int dd;
    double x1 = nodeptr[i].x, x2 = nodeptr[j].x, 
	y1 = nodeptr[i].y, y2 = nodeptr[j].y;

    deg = dtrunc (x1);
    min = x1 - deg;
    lati = M_PI * (deg + 5.0 * min / 3.0) / 180.0;
    deg = dtrunc (x2);
    min = x2 - deg;
    latj = M_PI * (deg + 5.0 * min / 3.0) / 180.0;

    deg = dtrunc (y1);
    min = y1 - deg;
    longi = M_PI * (deg + 5.0 * min / 3.0) / 180.0;
    deg = dtrunc (y2);
    min = y2 - deg;
    longj = M_PI * (deg + 5.0 * min / 3.0) / 180.0;

    q1 = cos (longi - longj);
    q2 = cos (lati - latj);
    q3 = cos (lati + latj);
    dd = (int) (6378.388 * acos (0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0);
    return dd;

}

int att_distance (Point *nodeptr, int i, int j)
/*    
      FUNCTION: compute ATT distance between two nodes rounded to next 
                integer for VRPLIB instances
      INPUT:    two node indices
      OUTPUT:   distance between the two nodes
      COMMENTS: for the definition of how to compute this distance see VRPLIB
*/
{
    double xd = nodeptr[i].x - nodeptr[j].x;
    double yd = nodeptr[i].y - nodeptr[j].y;
    double rij = sqrt ((xd * xd + yd * yd) / 10.0);
    double tij = dtrunc (rij);
    int dij;

    if (tij < rij)
        dij = (int) tij + 1;
    else
        dij = (int) tij;
    return dij;
}

double **compute_distances(Problem *instance)
/*    
      FUNCTION: computes the matrix of all intercity distances
      INPUT:    none
      OUTPUT:   pointer to distance matrix, has to be freed when program stops
*/
{
    int     i, j;
    double     **matrix;
    int num_node = instance->num_node;

    if((matrix = (double **)malloc(sizeof(double) * num_node * num_node +
                                     sizeof(double *) * num_node)) == NULL){
        fprintf(stderr,"Out of memory, exit.");
        exit(1);
    }

    for ( i = 0 ; i < num_node ; i++ ) {
        matrix[i] = (double *)(matrix + num_node) + i*num_node;
        for ( j = 0  ; j < num_node ; j++ ) {
            matrix[i][j] = distance(instance->nodeptr, i, j, instance->dis_type);
        }
    }
    return matrix;
}



int ** compute_nn_lists (Problem *instance)
/*    
      FUNCTION: computes nearest neighbor lists of depth nn for each node
      INPUT:    none
      OUTPUT:   pointer to the nearest neighbor lists
*/
{
    int i, node, nn;
    double *distance_vector;
    int *help_vector;
    int **m_nnear;
    int num_node = instance->num_node;
 
    TRACE ( printf("\n computing nearest neighbor lists, "); )

    nn = MAX(instance->nn_ls, instance->nn_ants);
    if ( nn >= num_node) {
        nn = num_node - 1;
    }
    DEBUG ( assert( num_node > nn ); )
    
    TRACE ( printf("nn = %ld ... \n",nn); ) 

    if((m_nnear = (int **)malloc(sizeof(int) * num_node * nn
                                      + num_node * sizeof(int *))) == NULL){
        exit(EXIT_FAILURE);
    }
    distance_vector = (double *)calloc(num_node, sizeof(double));
    help_vector = (int *)calloc(num_node, sizeof(int));
 
    for ( node = 0 ; node < num_node ; node++ ) {  /* compute cnd-sets for all node */
        m_nnear[node] = (int *)(m_nnear + num_node) + node * nn;

        for ( i = 0 ; i < num_node ; i++ ) {  /* Copy distances from nodes to the others */
            distance_vector[i] = instance->distance[node][i];
            help_vector[i] = i;
        }
        distance_vector[node] = INFINITY;  /* node is not nearest neighbour */
        distance_vector[0] = INFINITY;     /* depot点需要排除在 nearest neighbour之外 */
        sort2(distance_vector, help_vector, 0, num_node-1);
        for ( i = 0 ; i < nn ; i++ ) {
            m_nnear[node][i] = help_vector[i];
        }
    }
    free(distance_vector);
    free(help_vector);
    TRACE ( printf("\n    .. done\n"); )
    return m_nnear;
}


/*
 FUNCTION: compute the tour length of tour tour
 INPUT:    pointer to tour tour, tour size tour_size
 OUTPUT:   tour length of tour t
 */
double compute_tour_length(Problem *instance, int *tour, int tour_size)
{
    int      i;
    double   tour_length = 0;
  
    for ( i = 0 ; i < tour_size-1; i++ ) {
        tour_length += instance->distance[tour[i]][tour[i+1]];
    }
    return tour_length;
}

/*
 FUNCTION: compute the route length of route
 */
double compute_route_length(Problem *instance, int *route, int route_size)
{
    int      i;
    double   route_length = 0;
    
    for ( i = 0 ; i < route_size-1; i++ ) {
        route_length += instance->distance[route[i]][route[i+1]];
    }
    return route_length;
}

/*
 * FUNCTION:    Determine for each route of this solution the center gravity
 * INPUT:       tour, soulution point
 *              i.e. toute = [0,1,4,2,0]
 */
void compute_route_centers(Problem *instance, int *tour, const vector<RouteCenter *>& centers)
{
    Point *nodeptr = instance->nodeptr;
    Point *cp;
    
    int i, j;
    for (i = 0; i < centers.size(); i++) {
        cp = new Point();
        cp->x = 0;
        cp->y = 0;
        for (j = centers[i]->beg; j < centers[i]->end; j++) {
            cp->demand += nodeptr[tour[j]].demand;
        }
        for (j = centers[i]->beg; j < centers[i]->end; j++) {
            // weighted by demand
            cp->x += nodeptr[tour[j]].x * nodeptr[tour[j]].demand * 1.0 / cp->demand;
            cp->y += nodeptr[tour[j]].y * nodeptr[tour[j]].demand * 1.0 / cp->demand;
        }
        centers[i]->coord = cp;
    }
}

