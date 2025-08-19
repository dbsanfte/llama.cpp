#include "ggml-numa-softmax.h"
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

// Specialized work function for SOFT_MAX operations
enum ggml_status ggml_numa_work_function_soft_max(void * work_context, struct ggml_compute_params * params) {
    if (!work_context || !params) {
        return GGML_STATUS_FAILED;
    }
    
    ggml_numa_dispatcher_work_context_t * ctx = (ggml_numa_dispatcher_work_context_t *)work_context;
    
    GGML_LOG_DEBUG("Executing SOFT_MAX work function with %d threads\n", params->nth);
    
    // For SOFT_MAX operations, we need to call the kernel directly but with single-threaded parameters
    // to avoid conflicts with the NUMA coordinator threading model
    // Set up single-threaded compute params to process all data on this NUMA node
    struct ggml_compute_params single_thread_params = {
        .ith = 0,                  // Process all data (thread index 0)
        .nth = 1,                  // Single thread (total threads = 1)
        .wsize = params->wsize,    // Use coordinator's work buffer
        .wdata = params->wdata,    // Use coordinator's work buffer
        .threadpool = NULL         // No threadpool conflicts
    };
    
    // Call the SOFT_MAX mathematical kernel directly with single-threaded params
    ggml_compute_forward_soft_max(&single_thread_params, ctx->operation);
    
    GGML_LOG_DEBUG("Successfully executed SOFT_MAX work function\n");
    
    return GGML_STATUS_SUCCESS;
}

// Specialized work function for SOFT_MAX operations with NUMA-aware data parallel chunking
enum ggml_status ggml_numa_work_function_soft_max_chunk(void * work_context, struct ggml_compute_params * params) {
    if (!work_context || !params) {
        return GGML_STATUS_FAILED;
    }
    
    ggml_numa_dispatcher_work_context_t * ctx = (ggml_numa_dispatcher_work_context_t *)work_context;
    struct ggml_tensor * dst = ctx->operation;

    NUMA_ASSERT(dst && dst->src[0]);
    
    GGML_LOG_DEBUG("🔍 SOFT_MAX chunk: wsize=%zu, wdata=%p\n", params->wsize, params->wdata);
    GGML_LOG_DEBUG("🔍 SOFT_MAX chunk: dst=%p, src=%p\n", (void*)dst, (void*)dst->src[0]);
    
    const struct ggml_tensor * src = dst->src[0];
    
    GGML_LOG_DEBUG("🔍 Tensor dimensions: dst=(%ld,%ld,%ld,%ld), src=(%ld,%ld,%ld,%ld)\n",
           dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], 
           src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
    
    // NUMA-aware data slicing: each NUMA node processes a portion of the rows
    // SOFT_MAX is typically applied row-wise (along ne[0] dimension)
    // Each row can be processed independently, so distribute rows across NUMA nodes
    
    int current_numa_node = ggml_numa_get_current_node();
    int total_numa_nodes = ggml_numa_coordinator_get_num_nodes();
    
    const int64_t ne01 = dst->ne[1];  // Number of rows to process
    const int64_t ne02 = dst->ne[2];  // Batch dimension 2
    const int64_t ne03 = dst->ne[3];  // Batch dimension 3
    
    // Calculate which rows this NUMA node should process
    int64_t rows_per_node = ne01 / total_numa_nodes;
    int64_t extra_rows = ne01 % total_numa_nodes;
    
    // Calculate start and end rows for this NUMA node within each batch
    int64_t start_row = current_numa_node * rows_per_node;
    int64_t end_row = start_row + rows_per_node;
    
    // Distribute any extra rows to the first few nodes
    if (current_numa_node < extra_rows) {
        start_row += current_numa_node;
        end_row += current_numa_node + 1;
    } else {
        start_row += extra_rows;
        end_row += extra_rows;
    }
    
    // Ensure we don't exceed bounds
    if (end_row > ne01) end_row = ne01;
    if (start_row >= ne01) {
        // This NUMA node has no work to do
        GGML_LOG_DEBUG("🔍 NUMA node %d has no work (start_row=%ld >= ne01=%ld)\n", 
                       current_numa_node, start_row, ne01);
        return GGML_STATUS_SUCCESS;
    }
    
    GGML_LOG_DEBUG("🔍 NUMA node %d (of %d): processing rows %ld to %ld (of %ld total per batch)\n", 
                   current_numa_node, total_numa_nodes, start_row, end_row - 1, ne01);
    
    // Process each batch and each row in this NUMA node's assigned range
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = start_row; i01 < end_row; i01++) {
                
                GGML_LOG_DEBUG("🔍 Processing row i01=%ld, i02=%ld, i03=%ld\n", i01, i02, i03);
                
                // Get pointers to the source and destination data for this specific row
                const char * src_base = (const char*)ggml_get_data(src);
                char * dst_base = (char*)ggml_get_data(dst);
                
                const size_t src_offset = i01*src->nb[1] + i02*src->nb[2] + i03*src->nb[3];
                const size_t dst_offset = i01*dst->nb[1] + i02*dst->nb[2] + i03*dst->nb[3];
                
                const float * src_row = (const float*)(src_base + src_offset);
                float * dst_row = (float*)(dst_base + dst_offset);
                
                const int64_t ne00 = dst->ne[0];  // Row width
                
                // Apply softmax to this row using SIMD-optimized reference implementation
                // First pass: find maximum value
                float max_val = -INFINITY;
                ggml_vec_max_f32(ne00, &max_val, src_row);
                
                // Second pass: compute exponentials and sum using optimized vector function
                ggml_float sum = ggml_vec_soft_max_f32(ne00, dst_row, src_row, max_val);
                
                // Third pass: normalize using optimized vector scale
                const ggml_float sum_inv = 1.0 / sum;
                ggml_vec_scale_f32(ne00, dst_row, (float)sum_inv);
                
                GGML_LOG_DEBUG("🔍 Processed row with max_val=%.6f, sum=%.6f\n", max_val, sum);
            }
        }
    }
    
    GGML_LOG_DEBUG("🔍 SOFT_MAX chunk processing completed successfully\n");
    
    return GGML_STATUS_SUCCESS;
}