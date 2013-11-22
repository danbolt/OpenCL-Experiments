//
//  main.cpp
//  OpenCL-Experiment
//
//  Created by Daniel Savage on 11/22/2013.
//  Copyright (c) 2013 FrostTree Games. All rights reserved.
//

#include <cstdio>

#include <OpenCL/OpenCL.h>

//apparently xCode has generated this
#include "mykernel.cl.h"

//hard coding the number of values to test for convenience
#define NUM_VALUES 1024

int main(int argc, const char * argv[])
{
    int i;
    char deviceName[128];
    
    // try to get the dispatch queue for the GPU
    dispatch_queue_t queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_GPU, NULL);
    
    // in the event that the system does not have an OpenCL GPU, we can use the CPU instead
    if (queue == NULL)
    {
        queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_CPU, NULL);
    }
    
    // let's print some data on the device we're usng!　かわいいです！
    cl_device_id gpu = gcl_get_device_id_with_dispatch_queue(queue);
    clGetDeviceInfo(gpu, CL_DEVICE_NAME, 128, deviceName, NULL);
    fprintf(stdout, "Created a dispatch queue using the %s\n", deviceName);
    
    // let's hardcode some handy test data that's easy to understand
    float* test_in = (float*)malloc(sizeof(cl_float) * NUM_VALUES);
    for (i = 0; i < NUM_VALUES; i++)
    {
        test_in[i] = (cl_float)i;
    }
    
    // Once the computation using the CL is done, we'll need space in RAM for the output
    float* test_out = (float*)malloc(sizeof(cl_float) * NUM_VALUES);
    
    // Now we're going to allocate the buffers again in the OpenCL device's memory space
    // CL_MEM_COPY_HOST_PTR will copy the values of test_in to mem_in
    void* mem_in = gcl_malloc(sizeof(cl_float) * NUM_VALUES, test_in, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void* mem_out = gcl_malloc(sizeof(cl_float) * NUM_VALUES, NULL, CL_MEM_WRITE_ONLY);
    
    // DISPATCH THE KERNEL PROGRAM
    dispatch_sync(queue, ^{
        
        //workgroup size, I think
        size_t wgs;
        
        //information on sizing of dimensions
        gcl_get_kernel_block_workgroup_info(square_kernel, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wgs), &wgs, NULL);
        cl_ndrange range = {
            1,
            {0, 0, 0},
            {NUM_VALUES, 0, 0},
            {wgs, 0, 0}
        };
        
        // call the kernel
        square_kernel(&range, (cl_float*)mem_in, (cl_float*)mem_out);
        
        //copy the output into memory
        gcl_memcpy(test_out, mem_out, sizeof(cl_float) * NUM_VALUES);
    });
    
    // let's try printing the results
    for (i = 0; i < NUM_VALUES; i++)
    {
        printf("%f : %f\n", test_in[i], test_out[i]);
    }
    
    gcl_free(mem_in);
    gcl_free(mem_out);
    
    free(test_in);
    free(test_out);
    
    dispatch_release(queue);
    
    return 0;
}

