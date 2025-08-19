
#include "ggml-numa-rope.h"
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

// Specialized work function for ROPE operations with NUMA-aware chunking
enum ggml_status ggml_numa_work_function_rope_chunk(void * work_context, struct ggml_compute_params * params) {
    if (!work_context || !params) {
        GGML_LOG_ERROR("ROPE work function: Invalid parameters\n");
        return GGML_STATUS_FAILED;
    }
    
    ggml_numa_dispatcher_work_context_t * ctx = (ggml_numa_dispatcher_work_context_t *)work_context;
    
    if (!ctx->operation) {
        GGML_LOG_ERROR("ROPE work function: Operation is NULL\n");
        return GGML_STATUS_FAILED;
    }
    
    GGML_LOG_DEBUG("ROPE work function: operation=%p, type=%d, elements=%ld\n", 
                   (void*)ctx->operation, ctx->operation->op, ggml_nelements(ctx->operation));
    
    // Check SOURCE tensor data (src[0] is where ROPE reads input from)
    if (ctx->operation->src[0]) {
        void *src_data = ggml_get_data(ctx->operation->src[0]);
        if (src_data) {
            float *data = (float*)src_data;
            GGML_LOG_DEBUG("ROPE work function: First few SOURCE values: %.6f %.6f %.6f %.6f\n", 
                           data[0], data[1], data[2], data[3]);
            
            // STRICT VALIDATION: Check first 16 elements for NaN/inf corruption
            int64_t check_count = ggml_nelements(ctx->operation->src[0]);
            check_count = (check_count > 16) ? 16 : check_count;
            for (int64_t i = 0; i < check_count; i++) {
                NUMA_ASSERT(isfinite(data[i]), "ROPE: Found NaN/inf in src[0] data at index %d: %f", (int)i, (double)data[i]);
            }
        } else {
            GGML_LOG_WARN("ROPE work function: src[0] data is NULL\n");
        }
    } else {
        GGML_LOG_ERROR("ROPE work function: src[0] is NULL\n");
        return GGML_STATUS_FAILED;
    }
    
    // Check destination tensor (should start as zeros)
    void *dst_data = ggml_get_data(ctx->operation);
    if (dst_data) {
        float *data = (float*)dst_data;
        GGML_LOG_DEBUG("ROPE work function: First few DESTINATION values (before): %.6f %.6f %.6f %.6f\n", 
                       data[0], data[1], data[2], data[3]);
    }
    
    GGML_LOG_DEBUG("Executing ROPE chunk work function with %d threads on NUMA node (current CPU: %d)\n", 
                   params->nth, sched_getcpu());
    
    // For ROPE operations, we need to call the kernel directly but with single-threaded parameters
    // to avoid conflicts with the NUMA coordinator threading model
    // Set up single-threaded compute params to process all data on this NUMA node
    struct ggml_compute_params single_thread_params = {
        .ith = 0,                  // Process all data (thread index 0)
        .nth = 1,                  // Single thread (total threads = 1)
        .wsize = params->wsize,    // Use coordinator's work buffer
        .wdata = params->wdata,    // Use coordinator's work buffer
        .threadpool = NULL         // No threadpool conflicts
    };
    
    GGML_LOG_DEBUG("ROPE work function: Calling ggml_compute_forward_rope with single_thread_params (ith=%d, nth=%d)\n",
                   single_thread_params.ith, single_thread_params.nth);
    
    // Call the ROPE mathematical kernel directly with single-threaded params
    ggml_compute_forward_rope(&single_thread_params, ctx->operation);
    
    // Add memory barrier to ensure all writes are visible before returning
    __sync_synchronize();
    
    // Check output values and validate for corruption
    if (dst_data) {
        float *data = (float*)dst_data;
        GGML_LOG_DEBUG("ROPE work function: First few DESTINATION values (after): %.6f %.6f %.6f %.6f\n", 
                       data[0], data[1], data[2], data[3]);
        
        // CRITICAL VALIDATION: Check ROPE output for NaN/inf corruption 
        int64_t output_check_count = ggml_nelements(ctx->operation);
        output_check_count = (output_check_count > 32) ? 32 : output_check_count; // Check first 32 elements
        for (int64_t i = 0; i < output_check_count; i++) {
            NUMA_ASSERT(isfinite(data[i]), "ROPE: Generated corrupted output at index %d: %f", (int)i, (double)data[i]);
        }
        GGML_LOG_DEBUG("🔍 ROPE output validation: %ld elements checked, all finite\n", output_check_count);
    }
    
    GGML_LOG_DEBUG("Successfully executed ROPE chunk work function\n");
    
    return GGML_STATUS_SUCCESS;
}