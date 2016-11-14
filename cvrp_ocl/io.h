/*********************************
Ant Colony Optimization algorithms (AS, ACS, EAS, RAS, MMAS) for CVRP

Created by 孙晓奇 on 2016/10/8.
Copyright © 2016年 xiaoqi.sxq. All rights reserved.

Program's name: acovrp
Purpose: mainly input / output / statistic routines

email: sunxq1991@gmail.com

*********************************/

#ifndef inout_h
#define inout_h

#include <stdio.h>
#include "problem.h"
#include "move.h"


void read_instance_file(Problem *instance, const char *vrp_file_name);
char* parse_commandline (int argc, char *argv []);

void print_solution(Problem *instance, int *tour, int tour_size);
void print_single_route(Problem *instance, int *route, int route_size);
void print_problem_decompositon(const vector<Problem *>& subs);
void print_probabilities(Problem *instance);
void print_distance(Problem *instance);
void print_pheromone(Problem *instance);
void print_total_info(Problem *instance);
void print_solution_to_file(Problem *instance, FILE *file, int *tour, int tour_size);

void init_report(Problem *instance, int ntry);
void exit_report(Problem *instance, int ntry);
void write_best_so_far_report(Problem *instance);
void write_iter_report(Problem *instance);
void write_anneal_report(Problem *instance, AntStruct *ant, Move *move);

#endif /* ants_h */
