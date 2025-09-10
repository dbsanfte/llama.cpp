/**
 * @file soft_max.c  
 * @brief NUMA SOFT_MAX (Softmax Activation) Kernel Implementation
 * @author David Sanftenberg
 * 
 * ============================================================================
 * NUMA KERNEL: SOFT_MAX (Softmax Activation)
 * ============================================================================
 * 
 * Operation: y = softmax(x) with ALiBi attention bias support
 * 
 * Pattern: Reduction operation with row-wise processing
 * Features: 
 * - Numerical stability (max subtraction)
 * - ALiBi attention bias support with scale/max_bias
 * - Optional mask tensor handling (F16/F32)
 * - Shared memory optimization for direct result writes
 * - SIMD-optimized vector operations with ggml_vec functions
 * 
 * Implementation: Hybrid approach with composable macros for setup/validation
 * and custom row-wise processing for mathematical correctness
 */

#include "soft_max.h"
#include "numa-kernels.h"
#include "../ggml-numa-shared.h"
#include "../ggml-numa-openmp-coordinator.h"
#include "../vec.h"
#include "../ops.h"
#include "ggml.h"
#include <math.h>

// Forward declaration of F32 implementation
static enum ggml_status ggml_numa_kernel_soft_max_f32_execute(void * work_context, struct ggml_compute_params * params);

/**
 * @brief Main SOFT_MAX kernel execute function - dispatches to type-specific implementations
 * @param work_context Pointer to tensor being processed  
 * @param params Compute parameters from coordinator
 * @return Status of the computation
 */
enum ggml_status ggml_numa_kernel_soft_max_execute(void * work_context, struct ggml_compute_params * params) {
    struct ggml_tensor * dst = (struct ggml_tensor *)work_context;
    
    // Dispatch to type-specific implementation
    switch (dst->type) {
        case GGML_TYPE_F32:
            return ggml_numa_kernel_soft_max_f32_execute(work_context, params);
        default:
            // For now, only support F32 types (like ADD kernel pattern)
            // Return FAILED to allow graceful fallback to reference implementation
            return GGML_STATUS_FAILED;
    }
}

/**
 * @brief SOFT_MAX kernel implementation for F32 tensors using hybrid approach
 * @param work_context Pointer to tensor being processed
 * @param params Compute parameters from coordinator
 * @return Status of the computation
 */
static enum ggml_status ggml_numa_kernel_soft_max_f32_execute(void * work_context, struct ggml_compute_params * params) {
    struct ggml_tensor * dst = (struct ggml_tensor *)work_context;
    
    // Use composable macros for setup and validation (hybrid approach)
    NUMA_INIT_CONTEXT(ctx, dst, params);
    NUMA_VALIDATE_INPUTS(dst, params);
    
    // Get NUMA execution context from thread-local variables
    extern __thread bool ggml_numa_is_data_parallel_execution;
    extern __thread void * ggml_numa_shared_result_tensor_data;
    
    // Extract tensor parameters for SOFT_MAX operation
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];  // Optional mask tensor
    
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2]; 
    const int64_t ne03 = src0->ne[3];
    
    // Row-wise processing: total rows to process
    const int64_t nr = ne01 * ne02 * ne03;
    
    // NUMA-aware row slicing for data-parallel execution
    // First slice by NUMA nodes, then by threads within each node
    int64_t numa_start_row = 0, numa_end_row = nr;
    if (ggml_numa_is_data_parallel_execution) {
        // Get NUMA execution context from thread-local variables
        extern __thread int ggml_current_numa_node;
        extern __thread int ggml_numa_total_nodes_for_data_parallel;
        
        // First-level slicing: divide rows across NUMA nodes
        const int64_t rows_per_node = nr / ggml_numa_total_nodes_for_data_parallel;
        numa_start_row = ggml_current_numa_node * rows_per_node;
        numa_end_row = (ggml_current_numa_node == ggml_numa_total_nodes_for_data_parallel - 1) ? 
                       nr : numa_start_row + rows_per_node;
        
        NUMA_LOG_DEBUG("SOFT_MAX data-parallel: NUMA node %d processing rows %ld-%ld (total=%ld)", 
                      ggml_current_numa_node, numa_start_row, numa_end_row, nr);
    }
    
    // Second-level slicing: divide NUMA node's rows across threads  
    const int64_t numa_rows = numa_end_row - numa_start_row;
    const int64_t rows_per_thread = (numa_rows + ctx.total_threads - 1) / ctx.total_threads;
    const int64_t ir0 = numa_start_row + (ctx.thread_id * rows_per_thread);
    const int64_t ir1 = MIN(ir0 + rows_per_thread, numa_end_row);
    
    NUMA_LOG_TRACE("SOFT_MAX F32: thread %d/%d processing rows %ld to %ld (total rows=%ld)", 
                   ctx.thread_id, ctx.total_threads, ir0, ir1, nr);
    
    if (ir0 >= ir1) {
        NUMA_LOG_TRACE("SOFT_MAX F32: thread %d has no work, participating in barrier", ctx.thread_id);
        NUMA_BARRIER_AUTO(ctx);  // Still participate in barriers even with no work
        return GGML_STATUS_SUCCESS;
    }
    
    // Extract SOFT_MAX operation parameters
    float scale     = 1.0f;
    float max_bias  = 0.0f; 
    
    // Get operation parameters if available (ALiBi support)
    const float * op_params = (const float *)dst->op_params;
    if (op_params) {
        scale    = op_params[0];
        max_bias = op_params[1];
    }
    
    // ALiBi calculations - matching reference implementation
    const uint32_t n_head      = ne02;
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));
    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    
    // Access data using tensor_data() like ROPE
    void * dst_base = ggml_numa_shared_result_tensor_data ? 
                      ggml_numa_shared_result_tensor_data : 
                      tensor_data(dst);
    const void * src0_base = tensor_data(src0);
    const void * src1_base = src1 ? tensor_data(src1) : NULL;
    
    float * dst_data = (float *)dst_base;
    const float * src0_data = (const float *)src0_base;
    
    NUMA_LOG_TRACE("SOFT_MAX F32: thread %d processing %ld rows, dst=%p, src0=%p", 
                   ctx.thread_id, ir1 - ir0, dst_data, src0_data);
    
    // Process assigned rows ir0 to ir1
    for (int64_t i = ir0; i < ir1; i++) {
        // Calculate row indices
        const int64_t i03_val = i / (ne02 * ne01);
        const int64_t i02_val = (i - i03_val * ne02 * ne01) / ne01;
        const int64_t i01_val = i - i03_val * ne02 * ne01 - i02_val * ne01;
        
        // Calculate data pointers for this row
        const float * sp = src0_data + i03_val * src0->nb[3]/sizeof(float) + 
                                      i02_val * src0->nb[2]/sizeof(float) +
                                      i01_val * src0->nb[1]/sizeof(float);
        
        float * dp = dst_data + i03_val * dst->nb[3]/sizeof(float) +
                               i02_val * dst->nb[2]/sizeof(float) +
                               i01_val * dst->nb[1]/sizeof(float);
        
        if (i == ir0) {  // Log first row for debugging
            NUMA_LOG_TRACE("SOFT_MAX F32: thread %d first row: i=%ld, indices=[%ld,%ld,%ld,%ld], sp=%p, dp=%p", 
                           ctx.thread_id, i, i01_val, i02_val, i03_val, 0L, sp, dp);
            NUMA_LOG_TRACE("SOFT_MAX F32: thread %d first row input[0-3]: [%.6f, %.6f, %.6f, %.6f]", 
                           ctx.thread_id, sp[0], sp[1], sp[2], sp[3]);
        }
        
        // Handle ALiBi bias and mask if present
        const char * mask_ptr = NULL;
        if (src1_base) {
            mask_ptr = (const char *)src1_base + i01_val * src1->nb[1] + 
                                                 i02_val * src1->nb[2] + 
                                                 i03_val * src1->nb[3];
        }
        
        // Calculate ALiBi slope for this head (matching reference implementation)
        const uint32_t h = i02_val; // head index
        const float slope = (max_bias > 0.0f) ? 
                           (h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1)) : 
                           1.0f;
        
        // Copy source to destination using SIMD (matching reference)
        ggml_vec_cpy_f32(ne00, dp, sp);
        
        // Apply scale
        if (scale != 1.0f) {
            ggml_vec_scale_f32(ne00, dp, scale);
        }
        
        // Apply mask if present with ALiBi slope
        if (mask_ptr) {
            if (src1->type == GGML_TYPE_F16) {
                const ggml_fp16_t * mask_f16 = (const ggml_fp16_t *)mask_ptr;
                for (int j = 0; j < ne00; j++) {
                    dp[j] += slope * ggml_fp16_to_fp32(mask_f16[j]);
                }
            } else if (src1->type == GGML_TYPE_F32) {
                const float * mask_f32 = (const float *)mask_ptr;
                for (int j = 0; j < ne00; j++) {
                    dp[j] += slope * mask_f32[j];
                }
            }
        }
        
        // Compute softmax with numerical stability using SIMD-optimized function
        // Step 1: Find maximum for numerical stability
        float max_val = dp[0];
        for (int j = 1; j < ne00; j++) {
            if (dp[j] > max_val) {
                max_val = dp[j];
            }
        }
        
        // Step 2: Use SIMD-optimized ggml_vec_soft_max_f32 for exp and sum computation
        // This is critical for numerical precision - the reference implementation uses this
        ggml_float sum = ggml_vec_soft_max_f32(ne00, dp, dp, max_val);
        
        // Step 3: Normalize by sum using SIMD-optimized scaling
        if (sum > 0.0f) {
            ggml_vec_scale_f32(ne00, dp, 1.0f / sum);
        }
        
        if (i == ir0) {  // Log first row result for debugging
            NUMA_LOG_TRACE("SOFT_MAX F32: thread %d first row result[0-3]: [%.6f, %.6f, %.6f, %.6f] (max=%.6f, sum=%.6f)", 
                           ctx.thread_id, dp[0], dp[1], dp[2], dp[3], max_val, (double)sum);
        }
    }
    
    // CRITICAL: All threads must reach barrier for data-parallel synchronization
    NUMA_BARRIER_AUTO(ctx);
    
    return GGML_STATUS_SUCCESS;
}

/**
 * @brief Calculate work buffer size for SOFT_MAX operation
 * @param tensor The destination tensor for SOFT_MAX operation
 * @param total_numa_nodes Total number of NUMA nodes participating
 * @param total_threads Total number of threads participating
 * @return Total work buffer size needed for all threads
 */
size_t ggml_numa_kernel_soft_max_work_buffer_calc(const struct ggml_tensor * tensor, int total_numa_nodes, int total_threads) {
    // Each thread needs a work buffer for one row (ne00 elements + cache line alignment)
    const size_t ne00 = tensor->ne[0];
    const size_t cache_line_size_f32 = 16;  // CACHE_LINE_SIZE_F32 approximation
    const size_t per_thread_buffer = (ne00 + cache_line_size_f32) * sizeof(float);
    
    // Return TOTAL work buffer size for ALL threads
    return per_thread_buffer * total_threads;
}

/**
 * @brief Query function for SOFT_MAX NUMA kernel strategy selection
 * @param tensor Tensor to be processed
 * @return Recommended execution strategy based on tensor size
 */
ggml_numa_execution_strategy_t ggml_numa_kernel_soft_max_query(const struct ggml_tensor * tensor) {
    // Calculate total elements for strategy selection (hot path - must be fast)
    size_t total_elements = ggml_nelements(tensor);
    
    // Get cache entry for this operation (O(1) lookup)
    const ggml_numa_kernel_cache_entry_t * cache_entry = ggml_numa_lookup_kernel_direct(GGML_OP_SOFT_MAX);
    
    // Use shared macro for unified strategy selection
    ggml_numa_execution_strategy_t selected_strategy;
    NUMA_SELECT_STRATEGY_FROM_CACHE(cache_entry, total_elements, selected_strategy);
    
    // Debug logging for operation analysis
    const char* strategy_name = (selected_strategy == NUMA_STRATEGY_SINGLE_THREAD) ? "(Single/Single)" :
                               (selected_strategy == NUMA_STRATEGY_SINGLE_NODE) ? "(Single/Multi)" :
                               (selected_strategy == NUMA_STRATEGY_DATA_PARALLEL) ? "(Data Parallel)" : "(Unknown)";
    
    NUMA_LOG_DEBUG("NUMA SOFT_MAX %s", strategy_name);
    
    // Return strategy only - executor gets everything else from cache
    return selected_strategy;
}

/**
 * @brief Register SOFT_MAX kernel with NUMA system
 * @return Registration information for SOFT_MAX kernel
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_soft_max_register(void) {
    ggml_numa_kernel_registration_info_t info = {0};
    
    info.op_type = GGML_OP_SOFT_MAX;
    info.supported = true;
    info.kernel_name = "NUMA SOFT_MAX Kernel";
    
    // Strategy thresholds optimized for reduction operations
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_SINGLE] = 1024;      // Single thread below 1K elements
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_MULTI] = 65536;      // Multi-thread below 64K elements
    // Above 64K elements: data-parallel strategy
    info.strategy_array.valid = true;
    
    // Function pointers for different strategies
    info.work_funcs.single_single_fn = ggml_numa_kernel_soft_max_execute;
    info.work_funcs.single_multi_fn = ggml_numa_kernel_soft_max_execute;
    info.work_funcs.data_parallel_fn = ggml_numa_kernel_soft_max_execute;
    info.work_funcs.valid = true;
    
    // Query function pointer - enables direct dispatch without switch statements
    info.query_fn = (void*)ggml_numa_kernel_soft_max_query;
    
    // Work buffer calculation function pointer
    info.work_buffer_calc_fn = (void*)ggml_numa_kernel_soft_max_work_buffer_calc;
    
    // SOFT_MAX doesn't need aggregation functions - operations are in-place
    info.agg_funcs.single_single_fn = NULL;
    info.agg_funcs.single_multi_fn = NULL; 
    info.agg_funcs.data_parallel_fn = NULL;
    info.agg_funcs.valid = false;
    
    return info;
}
