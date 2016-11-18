/*********************************
 Ant Colony Optimization algorithms (AS, ACS, EAS, RAS, MMAS) for CVRP
 
 Created by 孙晓奇 on 2016/10/8.
 Copyright © 2016年 xiaoqi.sxq. All rights reserved.
 
 Program's name: acovrp
 Purpose: some additional useful procedures
 
 email: sunxq1991@gmail.com
 
 *********************************/

#ifndef   utilities_h
#define   utilities_h

#define MAXIMUM_NO_TRIES      100

const float EPSILON = 0.001f;

#define TRUE  1
#define FALSE 0

/* general macros */

#define MAX(x,y)        ((x)>=(y)?(x):(y))
#define MIN(x,y)        ((x)<=(y)?(x):(y))

#define DEBUG( x ) x

#define TRACE( x )

/* constants for a random number generator, for details see numerical recipes in C */

#define IA 16807
#define IM 2147483647
#define AM (1.0f/IM)
#define IQ 127773
#define IR 2836
#define MASK 123459876

float mean ( int *values, int max);

float meanr ( float *values, int max );

float std_deviation ( int *values, int i, float mean );

float std_deviationr ( float *values, int i, float mean );

int best_of_vector ( int *values, int i );

int worst_of_vector ( int *values, int i );

void swap ( int v[], int i, int j );

void sort ( int v[], int left, int right );

float quantil ( int vector[], float q, int numbers );

void swap2(float v[], int v2[], int i, int j);

void sort2(float v[], int v2[], int left, int right);

float ran01 ( int *idum );

int random_number ( int *idum );

int ** generate_int_matrix( int n, int m);

float ** generate_float_matrix( int n, int m);

void swap(int *i, int *j);

#endif
