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
    const int64_t ne01 = dst->ne[1];  // Number of rows (distributed across threads)
    const int64_t ne02 = dst->ne[2];  // Outer dimension (full processing per thread)
    const int64_t ne03 = dst->ne[3];  // Outermost dimension (full processing per thread)
    
    // Calculate strides from source tensor (obtained via building block)
    const size_t nb01 = dst->src[0]->nb[1];
    const size_t nb02 = dst->src[0]->nb[2];
    const size_t nb03 = dst->src[0]->nb[3];
    const size_t nb1 = dst->nb[1];
    const size_t nb2 = dst->nb[2];
    const size_t nb3 = dst->nb[3];
    
    // Note: dst_data from NUMA_ROWWISE_KERNEL_SETUP is ready to use
    
    // 3D nested loop processing: outer loops (i03, i02) process all elements,
    // inner loop (i01) is distributed across threads using NUMA slice context
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (size_t i01 = ctx.thread_start; i01 < ctx.thread_end; i01++) {
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
            }
        }
    }
    
    // End barrier for consistent thread synchronization
    NUMA_BARRIER_AUTO(ctx);
    
    NUMA_LOG_TRACE("RMS_NORM processed rows %zu-%zu with thread %d/%d on NUMA node %d", 
                   ctx.thread_start, ctx.thread_end,
                   ctx.thread_id, ctx.total_threads, ctx.numa_node);
    
    return GGML_STATUS_SUCCESS;
}

/**
 * @brief Query optimal execution strategy for RMS_NORM operation
 * 
 * RMS_NORM benefits from different strategies based on tensor size:
 * - Small tensors: Single-thread to avoid threading overhead
 * - Medium tensors: Multi-thread single-node for cache locality
 * - Large tensors: Data-parallel for maximum throughput
 * 
 * @param tensor Target tensor for RMS normalization
 * @return Optimal execution strategy based on tensor size
 */
ggml_numa_execution_strategy_t ggml_numa_kernel_rms_norm_query(const struct ggml_tensor * tensor) {
    // Calculate total elements for strategy selection (hot path - must be fast)
    const size_t total_elements = ggml_nelements(tensor);
    
    // Get cache entry for RMS_NORM operation using direct lookup
    const ggml_numa_kernel_cache_entry_t * cache_entry = ggml_numa_lookup_kernel_direct(GGML_OP_RMS_NORM);
    
    if (!cache_entry || !cache_entry->supported) {
        return NUMA_STRATEGY_SINGLE_THREAD;  // Fallback to single thread strategy
    }
    
    // Use unified strategy selection macro for consistent behavior
    ggml_numa_execution_strategy_t selected_strategy;
    NUMA_SELECT_STRATEGY_FROM_CACHE(cache_entry, total_elements, selected_strategy);
    
    // Debug logging (controlled by environment variable)
    const char* op_name = cache_entry && cache_entry->kernel_name[0] ? cache_entry->kernel_name : "NUMA RMS_NORM";
    NUMA_LOG_DEBUG("RMS_NORM query: %zu elements -> strategy %d (%s)\n", 
                   total_elements, selected_strategy, op_name);
    
    return selected_strategy;
}

/**
 * @brief Calculate work buffer size for RMS_NORM operation
 * 
 * RMS_NORM requires no additional work buffers since it processes
 * data in-place with direct memory access patterns.
 * 
 * @param tensor Target tensor (unused - no work buffer needed)
 * @param total_numa_nodes Total NUMA nodes (unused)
 * @param total_threads Total threads (unused)
 * @return 0 (no work buffer required)
 */
size_t ggml_numa_kernel_rms_norm_work_buffer_calc(const struct ggml_tensor * tensor, int total_numa_nodes, int total_threads) {
    NUMA_ASSERT(tensor != NULL, "Tensor cannot be null");
    NUMA_ASSERT(total_numa_nodes > 0, "Total NUMA nodes must be positive");
    NUMA_ASSERT(total_threads > 0, "Total threads must be positive");
    
    // RMS_NORM processes data in-place, no additional work buffers needed
    return 0;
}

/**
 * @brief Register RMS_NORM kernel with NUMA system
 * 
 * Configures registration info for the RMS_NORM operation with
 * reduction-specific thresholds and row-wise processing strategy.
 * 
 * @return Fully populated registration info structure
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_rms_norm_register(void) {
    ggml_numa_kernel_registration_info_t info = {0};
    
    info.op_type = GGML_OP_RMS_NORM;
    info.supported = true;
    info.kernel_name = "NUMA RMS_NORM Kernel (Row-wise Reduction)";
    
    // Strategy thresholds for reduction operation
    // Use smaller thresholds than element-wise ops due to reduction overhead
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_SINGLE] = 2048;      // Single thread below 2K elements
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_MULTI] = 131072;     // Multi-thread below 128K elements
    // Above 128K elements: data-parallel strategy
    info.strategy_array.valid = true;
    
    // Function pointers for different strategies
    info.work_funcs.single_single_fn = ggml_numa_kernel_rms_norm_execute;
    info.work_funcs.single_multi_fn = ggml_numa_kernel_rms_norm_execute;
    info.work_funcs.data_parallel_fn = ggml_numa_kernel_rms_norm_execute;
    info.work_funcs.valid = true;
    
    // Query function pointer for direct dispatch
    info.query_fn = (void*)ggml_numa_kernel_rms_norm_query;
    
    // Work buffer calculation function pointer
    info.work_buffer_calc_fn = (void*)ggml_numa_kernel_rms_norm_work_buffer_calc;
    
    // No aggregation functions needed (row-wise processing, no cross-NUMA reduction)
    info.agg_funcs.single_single_fn = NULL;
    info.agg_funcs.single_multi_fn = NULL; 
    info.agg_funcs.data_parallel_fn = NULL;
    info.agg_funcs.valid = false;
    
    return info;
}
