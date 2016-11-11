//
//  g_type.cpp
//  cvrp_ocl
//
//  Created by 孙晓奇 on 2016/10/17.
//  Copyright © 2016年 xiaoqi.sxq. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "g_type.h"

using namespace std;

static const char *kernelSrcPath { "/Users/sunny/Desktop/cvrp_proj/src/cvrp_aco_ocl/cvrp_ocl/kernel.cl" };

OpenclEnv::OpenclEnv()
{
    cl_int errNum;
    cl_uint numPlatforms;
    size_t size;
    
    // 选择opencl platform
    errNum = clGetPlatformIDs(1, &platformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0) {
        cerr << "Failed to find OpenCL platforms." << endl;
        exit(EXIT_FAILURE);
    }
    cl_context_properties cps[3] = {
        CL_CONTEXT_PLATFORM,
        cl_context_properties(platformId),
        0
    };
    
    // create opencl context on the platform
    context = clCreateContextFromType(cps, CL_DEVICE_TYPE_CPU, NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS) {
        cout << "Could not create GPU context" << endl;
    }
    
    // create a command queue on the first device available
    commandQueue = createCommandQueue(context, &deviceId);
    if (commandQueue == NULL) {
        exit(EXIT_FAILURE);
    }
    
    // create opencl program from kernel.cl source
    program = createProgram(context, deviceId, kernelSrcPath);
    if (program == NULL) {
        exit(EXIT_FAILURE);
    }
    
    // get numComputeUnits
    clGetDeviceInfo(deviceId, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &numComputeUnits, &size);
    
    krnl_table[static_cast<int>(kernel_t::construct_solution)] = clCreateKernel(program, "construct_solution", NULL);
    krnl_table[static_cast<int>(kernel_t::local_search)] = clCreateKernel(program, "local_search", NULL);
    krnl_table[static_cast<int>(kernel_t::pheromone_init)] = clCreateKernel(program, "pheromone_init", NULL);
    krnl_table[static_cast<int>(kernel_t::pheromone_evaporation)] = clCreateKernel(program, "pheromone_evaporation", NULL);
    krnl_table[static_cast<int>(kernel_t::ras_update)] = clCreateKernel(program, "ras_update", NULL);
    krnl_table[static_cast<int>(kernel_t::pheromone_disturbance)] = clCreateKernel(program, "pheromone_disturbance", NULL);
    krnl_table[static_cast<int>(kernel_t::update_pheromone_weighted)] = clCreateKernel(program, "update_pheromone_weighted", NULL);
    krnl_table[static_cast<int>(kernel_t::compute_total_info)] = clCreateKernel(program, "compute_total_info", NULL);
    krnl_table[static_cast<int>(kernel_t::update_statistics)] = clCreateKernel(program, "update_statistics", NULL);
    krnl_table[static_cast<int>(kernel_t::update_best_so_far_to_mem)] = clCreateKernel(program, "update_best_so_far_to_mem", NULL);
}

OpenclEnv::~OpenclEnv()
{
    clReleaseCommandQueue(commandQueue);
    for (int i = 0; i < static_cast<int>(kernel_t::LENGTH); i++) {
        clReleaseKernel(krnl_table[i]);
    }
    clReleaseProgram(program);
    clReleaseContext(context);
}

cl_command_queue OpenclEnv::createCommandQueue(cl_context context, cl_device_id *device)
{
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue commandQueue =  NULL;
    size_t deviceBufferSize = -1;
    
    // the the size of devices buffer
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS ) {
        cerr << "Fail call to clGetContextInfo()" << endl;
        return NULL;
    }
    
    if (deviceBufferSize <= 0) {
        cerr << "No device available." << endl;
        return NULL;
    }
    
    // allocate memory for the device buffer
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    if (errNum != CL_SUCCESS) {
        cerr << "Failed to get device IDs" << endl;
        return NULL;
    }
    
    // choose the first available device.
    commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
    if (commandQueue == NULL) {
        cerr << "Failed to create commandQueue from device 0" << endl;
        return NULL;
    }
    
    *device = devices[0];
    delete [] devices;
    
    return commandQueue;
    
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program OpenclEnv::createProgram(cl_context context, cl_device_id device, const char* filename)
{
    cl_int errNum;
    cl_program program;
    
    std::string srcStdStr = readFile(filename);
    const char *srcStr = srcStdStr.c_str();
    
    program = clCreateProgramWithSource(context, 1, (const char**)&srcStr, NULL, NULL);
    if (program == NULL)
    {
        std::cerr << "Failed to create CL program from source." << std::endl;
        return NULL;
    }
    
    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);
        
        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }
    
    return program;
}

std::string OpenclEnv::readFile(const char *filename)
{
    std::ifstream kernelFile(filename, std::ios::in);
    if (!kernelFile.is_open())
    {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return NULL;
    }
    
    std::ostringstream oss;
    oss << kernelFile.rdbuf();
    std::string content = oss.str();
    
    return content;
}

cl_kernel& OpenclEnv:: get_kernel(kernel_t id)
{
    return krnl_table[static_cast<int>(id)];
}