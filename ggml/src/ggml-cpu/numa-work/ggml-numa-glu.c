
#include "ggml-numa-glu.h"
#include "ggml-numa-operation-dispatch.h"
#include "ggml-numa-coordinator.h"
#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "ggml.h"
#include "vec.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <pthread.h>

#ifdef __linux__
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#endif

// Specialized work function for GLU operations with NUMA-aware chunking
enum ggml_status ggml_numa_work_function_glu_chunk(void * work_context, struct ggml_compute_params * params) {
    if (!work_context || !params) {
        GGML_LOG_ERROR("GLU work function: Invalid parameters\n");
        return GGML_STATUS_FAILED;
    }
    
    ggml_numa_dispatcher_work_context_t * ctx = (ggml_numa_dispatcher_work_context_t *)work_context;
    
    if (!ctx->operation) {
        GGML_LOG_ERROR("GLU work function: Operation is NULL\n");
        return GGML_STATUS_FAILED;
    }
    
    GGML_LOG_DEBUG("GLU work function: operation=%p, type=%d, elements=%ld\n", 
                   (void*)ctx->operation, ctx->operation->op, ggml_nelements(ctx->operation));
    
    // Check source tensors - GLU can have one or two source tensors
    const struct ggml_tensor * src0 = ctx->operation->src[0];
    const struct ggml_tensor * src1 = ctx->operation->src[1];
    
    if (!src0) {
        GGML_LOG_ERROR("GLU work function: Missing primary source tensor\n");
        return GGML_STATUS_FAILED;
    }
    
    // Check SOURCE tensor data with comprehensive validation
    void *src0_data = ggml_get_data(src0);
    if (!src0_data) {
        GGML_LOG_ERROR("GLU work function: Source tensor data is NULL\n");
        return GGML_STATUS_FAILED;
    }
    
    // Validate input data for NaN/inf values
    float *data0 = (float*)src0_data;
    const int64_t src0_elements = ggml_nelements(src0);
    
    for (int64_t i = 0; i < src0_elements; i++) {
        NUMA_ASSERT(isfinite(data0[i]), "GLU: Found NaN/inf in src0 data at index %d: %f", (int)i, (double)data0[i]);
    }
    
    GGML_LOG_DEBUG("GLU work function: First few SOURCE0 values: %.6f %.6f %.6f %.6f\n", 
                   data0[0], data0[1], data0[2], data0[3]);

    // Check second source tensor if present
    if (src1) {
        void *src1_data = ggml_get_data(src1);
        if (src1_data) {
            float *data1 = (float*)src1_data;
            
            // Validate second input tensor for NaN/inf values
            const int64_t src1_elements = ggml_nelements(src1);
            for (int64_t i = 0; i < src1_elements; i++) {
                NUMA_ASSERT(isfinite(data1[i]), "GLU: Found NaN/inf in src1 data at index %d: %f", (int)i, (double)data1[i]);
            }
            
            GGML_LOG_DEBUG("GLU work function: First few SOURCE1 values: %.6f %.6f %.6f %.6f\n", 
                           data1[0], data1[1], data1[2], data1[3]);
        } else {
            GGML_LOG_WARN("GLU work function: Second source tensor data is NULL\n");
        }
    } else {
        GGML_LOG_DEBUG("GLU work function: Single source tensor mode (split tensor)\n");
    }    // Check destination tensor (should start as zeros or uninitialized)
    void *dst_data = ggml_get_data(ctx->operation);
    if (dst_data) {
        float *data = (float*)dst_data;
        GGML_LOG_DEBUG("GLU work function: First few DESTINATION values (before): %.6f %.6f %.6f %.6f\n", 
                       data[0], data[1], data[2], data[3]);
    } else {
        GGML_LOG_ERROR("GLU work function: Destination tensor data is NULL\n");
        return GGML_STATUS_FAILED;
    }
    
    GGML_LOG_DEBUG("Executing GLU chunk work function with %d threads on NUMA node (current CPU: %d)\n", 
                   params->nth, sched_getcpu());
    
    // For GLU operations, we need to call the kernel directly but with single-threaded parameters
    // to avoid conflicts with the NUMA coordinator threading model
    // Set up single-threaded compute params to process all data on this NUMA node
    struct ggml_compute_params single_thread_params = {
        .ith = 0,                  // Process all data (thread index 0)
        .nth = 1,                  // Single thread (total threads = 1)
        .wsize = params->wsize,    // Use coordinator's work buffer
        .wdata = params->wdata,    // Use coordinator's work buffer
        .threadpool = NULL         // No threadpool conflicts
    };
    
    GGML_LOG_DEBUG("GLU work function: Calling ggml_compute_forward_glu with single_thread_params (ith=%d, nth=%d)\n",
                   single_thread_params.ith, single_thread_params.nth);
    
    // Call the GLU mathematical kernel directly with single-threaded params
    ggml_compute_forward_glu(&single_thread_params, ctx->operation);
    
    // Add memory barrier to ensure all writes are visible before returning
    __sync_synchronize();
    
    // Check output values
    if (dst_data) {
        float *data = (float*)dst_data;
        GGML_LOG_DEBUG("GLU work function: First few DESTINATION values (after): %.6f %.6f %.6f %.6f\n", 
                       data[0], data[1], data[2], data[3]);
    }
    
    GGML_LOG_DEBUG("Successfully executed GLU chunk work function\n");
    
    return GGML_STATUS_SUCCESS;
}
