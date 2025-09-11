/**
 * @file rms_norm.c
 * @brief NUMA kernel for RMS_NORM operation (Root Mean Square normalization)
 * 
 * @author David Sanftenberg
 * @date 2025-09-09
 * 
 * This kernel implements the GGML_OP_RMS_NORM operation for NUMA-aware execution.
 * RMS_NORM performs Root Mean Square normalization on input tensors, computing
 * the mean of squares for each row and scaling by 1/sqrt(mean + eps).
 * 
 * Mathematical Formula:
 * - For each row x[i] of length ne00:
 *   1. mean = sum(x[i]^2) / ne00
 *   2. y[i] = x[i] / sqrt(mean + eps)
 * 
 * Key characteristics:
 * - Row-wise reduction operation (processes ne00 elements per row)
 * - Independent computation per row (ne01 rows total)
 * - SIMD-optimized scaling using ggml_vec_scale_f32
 * - Optimal for NUMA row-slicing strategy
 */

#include "rms_norm.h"
#include "numa-kernels.h"
#include "ggml-numa-shared.h"
#include "../ggml-vec-numa.h"
#include <math.h>
#include <string.h>

/**
 * @brief Execute RMS_NORM operation with NUMA optimization
 * 
 * Performs Root Mean Square normalization on input tensor rows.
 * Each row is processed independently: compute mean of squares,
 * then scale by 1/sqrt(mean + eps).
 * 
 * @param work_context Pointer to the destination tensor (cast from ggml_tensor*)
 * @param params Compute parameters containing thread information
 * @return GGML_STATUS_SUCCESS on success, GGML_STATUS_FAILED on error
 */
enum ggml_status ggml_numa_kernel_rms_norm_execute(void * work_context, struct ggml_compute_params * params) {
    NUMA_ASSERT(work_context != NULL, "Work context cannot be null");
    NUMA_ASSERT(params != NULL, "Compute params cannot be null");
    
    struct ggml_tensor * dst = (struct ggml_tensor *)work_context;
    struct ggml_tensor * src0 = dst->src[0];
    
    NUMA_ASSERT(src0 != NULL, "Source tensor cannot be null");
    NUMA_ASSERT(dst->op == GGML_OP_RMS_NORM, "Expected RMS_NORM operation");
    NUMA_ASSERT(ggml_are_same_shape(src0, dst), "Source and destination must have same shape");
    NUMA_ASSERT(src0->nb[0] == sizeof(float), "Only F32 tensors supported");
    
    // Extract epsilon parameter from operation
    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    NUMA_ASSERT(eps >= 0.0f, "Epsilon must be non-negative");
    
    // === COMPOSABLE KERNEL SETUP USING NEW BUILDING BLOCKS ===
    NUMA_ROWWISE_KERNEL_SETUP(ctx, dst, params, dst_data, float);
    
    // Get source data using building block
    NUMA_GET_SOURCE_POINTER(src_data, dst, 0, float);
    
    // Extract tensor dimensions
    const int64_t ne00 = dst->ne[0];  // Elements per row
    
    // Calculate strides from source tensor (obtained via building block)
    const size_t nb01 = dst->src[0]->nb[1];
    const size_t nb02 = dst->src[0]->nb[2];
    const size_t nb03 = dst->src[0]->nb[3];
    const size_t nb1 = dst->nb[1];
    const size_t nb2 = dst->nb[2];
    const size_t nb3 = dst->nb[3];
    
    // Note: dst_data from NUMA_ROWWISE_KERNEL_SETUP is ready to use
    
    // 4D nested loop processing using NUMA rowwise pattern: outer loops (i03, i02) process all elements,
    // inner loop (i01) is distributed across threads using NUMA slice context
    NUMA_4D_ROWWISE_LOOP(dst, ctx, {
        // Calculate row pointers using tensor_data for NUMA-aware access
        const float * x = (const float *)((const char *)tensor_data(dst->src[0]) + i01*nb01 + i02*nb02 + i03*nb03);
        float * y = (float *)((char *)tensor_data(dst) + i01*nb1 + i02*nb2 + i03*nb3);
        
        // First pass: compute sum of squares for this row
        ggml_float sum = 0.0;
        for (int64_t i00 = 0; i00 < ne00; i00++) {
            sum += (ggml_float)(x[i00] * x[i00]);
        }
        
        // Compute mean and normalization scale
        const float mean = sum / ne00;
        const float scale = 1.0f / sqrtf(mean + eps);
        
        // Verify scale is valid (catches NaN/inf issues early)
        NUMA_ASSERT(scale > 0.0f && isfinite(scale), "Invalid normalization scale computed");
        
        // Second pass: copy input and apply scaling with SIMD optimization
        memcpy(y, x, ne00 * sizeof(float));
        ggml_vec_scale_f32(ne00, y, scale);
    });
    
    // End barrier for consistent thread synchronization
    NUMA_BARRIER_AUTO(ctx);
    
    NUMA_LOG_TRACE("RMS_NORM processed rows %zu-%zu with thread %d/%d on NUMA node %d", 
                   ctx.thread_start, ctx.thread_end,
                   ctx.thread_id, ctx.total_threads, ctx.numa_node);
    
    return GGML_STATUS_SUCCESS;
}

// ============================================================================ 
// Complete kernel implementation using shared macros
// ============================================================================

NUMA_KERNEL_REGISTER_METADATA(
    rms_norm,                               // kernel name
    GGML_OP_RMS_NORM,                      // operation type
    "NUMA RMS_NORM Kernel (Row-wise Reduction)",  // kernel description
    512,                                   // single_single threshold (smaller for reductions)
    2048,                                 // single_multi threshold (128K elements)  
    ggml_numa_kernel_rms_norm_execute      // execution function
)


