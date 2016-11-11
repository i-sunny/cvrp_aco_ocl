//
//  g_type.hpp
//  cvrp_ocl
//
//  Created by 孙晓奇 on 2016/10/17.
//  Copyright © 2016年 xiaoqi.sxq. All rights reserved.
//

#ifndef g_type_hpp
#define g_type_hpp


#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

enum class kernel_t
{
    construct_solution,
    local_search,
    pheromone_init,
    pheromone_evaporation,
    ras_update,
    pheromone_disturbance,
    update_pheromone_weighted,
    compute_total_info,
    update_statistics,
    update_best_so_far_to_mem,
    LENGTH = update_best_so_far_to_mem+1
};

class OpenclEnv
{
private:
    cl_kernel krnl_table[static_cast<int>(kernel_t::LENGTH)];
    
    static cl_command_queue createCommandQueue(cl_context context, cl_device_id *device);
    static cl_program createProgram(cl_context context, cl_device_id device, const char* filename);
    static std::string readFile(const char *filename);
    
public:
    cl_platform_id platformId;
    cl_context context;
    cl_device_id deviceId;
    cl_command_queue commandQueue;
    cl_program program;
    int numComputeUnits;
    
    OpenclEnv();
    ~OpenclEnv();
    cl_kernel& get_kernel(kernel_t id);
};

#endif /* g_type_hpp */
