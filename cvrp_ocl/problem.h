/*********************************
 Ant Colony Optimization algorithms (AS, ACS, EAS, RAS, MMAS) for CVRP
 
 Created by 孙晓奇 on 2016/10/8.
 Copyright © 2016年 xiaoqi.sxq. All rights reserved.
 
 Program's name: acovrp
 Purpose: vrp problem model, problem init/exit, solution check
 
 email: sunxq1991@gmail.com
 
 *********************************/

#ifndef Problem_h
#define Problem_h

#include <stdio.h>
#include <bitset>
#include <vector>

using namespace std;

#define LINE_BUF_LEN     255

/*************** global variables *********************/

extern int g_master_problem_iteration_num;   /* 每次外循环，主问题蚁群的迭代的次数 */
extern int g_sub_problem_iteration_num;      /* 每次外循环，子问题蚁群的迭代次数 */

extern float rho;           /* parameter for evaporation */
extern float alpha;         /* importance of trail */
extern float beta;          /* importance of heuristic evaluate */
extern int ras_ranks;   /* additional parameter for rank-based version of ant system */
extern bool sa_flag;

/****************** data struct ***********************/
enum DistanceTypeEnum {
    DIST_EUC_2D, DIST_CEIL_2D, DIST_GEO, DIST_ATT
};

struct Point {
    float x;
    float y;
    int demand;     /* 每个配送点需求 */
};

struct RouteCenter {
    int beg;                   /* route在tour中的开始位置,tour[beg] = 0(depot) */
    int end;                   /* route在tour中的结束位置,tour[end] = 0(depot) */
    Point   *coord;                 /* center point coord */
    short angle;                    /* 重心与depot的角度 -180~180 */
};

struct Route {
    Route(int beg_, int end_, int load_, float dist_):beg(beg_), end(end_), load(load_), dist(dist_){}
    int beg;                   /* route在tour中的开始位置,tour[beg] = 0(depot) */
    int end;                   /* route在tour中的结束位置,tour[end] = 0(depot) */
    int load;                  /* route load */
    float dist;                    /* route distance */
};

/* best-so-far soution information */
struct BestSolutionInfo {
    int     iter;           // iteration of the best-so-far solution
    float   length;         // tour length of solution
    float   time;           // elapsed time
};

/*
 * ant tour consists of some single route.
 * a single route begin with depot(0) and end with depot(0),
 * i.e. tour = [0,1,4,2,0,5,3,0] toute1 = [0,1,4,2,0] route2 = [0,5,3,0]
 */
struct AntStruct {
    int     *tour;
    int     tour_size;     /* 路程中经过的点个数（depot可能出现多次） */
    float   tour_length;   /* 车辆路程行走长度 */
    bool    *visited;      /* mark nodes visited status */
    bool    *candidate;    /** 所有可配送的点(单次route中，目前车辆可以仍可配送的点) */
};


struct Problem {
    Problem(short id): pid(id){}
    
    short         pid;                      /* 主问题id = 0, 子问题id递增 */
    char          name[LINE_BUF_LEN];      	 /* instance name */
    char          edge_weight_type[LINE_BUF_LEN];  /* selfexplanatory */
    DistanceTypeEnum dis_type;               /* 用于决定使用哪种距离方式 */
    float        optimum;                /* optimal tour length if known, otherwise a bound */
    int      num_node;               /* number of nodes, depot included, numd_node = 1(depot) + num of target nodes*/
    Point         *nodeptr;               /* array of structs containing coordinates of nodes */
    float        **distance;             /* distance matrix: distance[i][j] gives distance
                                           between node i und j */
    int      **nn_list;              /* nearest neighbor list; contains for each node i a
                                           sorted list of n_near nearest neighbors */
    int      vehicle_capacity;       /* 车辆最大装载量 */
    float        max_distance;           /* 最大行驶距离 */
    float        service_time;           /*  service time needed for node */
    
    
    /*----- local search -----*/
    bool ls_flag;          /* indicates whether and which local search is used */
    int nn_ls;            /* maximal depth of nearest neighbour lists used in the
                                local search */
    bool dlb_flag;         /* flag indicating whether don't look bits are used. I recommend
                                to always use it if local search is applied */
    
    int iteration;           /* counter of number iterations */
    int max_iteration;       /* maximum number of iterations */
    
    /*----- ant info -----*/
    AntStruct *ants;                   /* this (array of) struct will hold the colony */
    AntStruct *best_so_far_ant;        /* struct that contains the best-so-far ant */
    AntStruct *iteration_best_ant;     /* 当前迭代表现最好的蚂蚁 */
    
    float   **pheromone;               /* pheromone matrix, one entry for each arc */
    float   **total_info;              /* combination of pheromone and heuristic information */
    
    float   *prob_of_selection;        /* 依概率选择下一个node */
    
    int n_ants;                    /* number of ants */
    int nn_ants;                   /* length of nearest neighbor lists for the ants'
                                         solution construction */
    
    float   max_runtime;               /* maximal allowed run time */
    float   best_so_far_time;          /* 当前最优解出现的时间 */
    int best_solution_iter;        /* iteration in which best solution is found */
    
    int rnd_seed;                  /* 用于生成随机数的种子 */
    
    float last_iter_solution;          /* 上一次迭代的解 */
    int iter_stagnate_cnt;         /* 迭代停滞计数器，记录解迭代解停滞的次数 */
    int best_stagnate_cnt;         /* 最优解停滞计数器，最优解未更新的次数 */
    
    /*----- sub problem only -----*/
    int num_subs;                  /* 仅用于parallel版本蚁群, 子问题个数 */
    vector<int> real_nodes;        /* 仅用于sub-problem, sub nodes需要重新编号，因此需要额外数组记录nodes真实的编号 */
    float   **best_pheromone;          /* 仅用于sub-problem, 记录sub 获得best_so_far_solution时候的信息素
                                           用于sub 迭代结束时更新至master */
};

void init_problem(Problem *instance);
void exit_problem(Problem *instance);
void init_sub_problem(Problem *master, Problem *sub);
void exit_sub_problem(Problem *sub);
bool check_solution(Problem *instance, int *tour, int tour_size);
bool check_route(Problem *instance, int *tour, int rbeg, int rend);

#endif /* Problem_h */
