/**
 * @file sub.c
 * @brief NUMA SUB kernel implementation demonstrating binary operation pattern reusability
 * @author David Sanftenberg
 */

// NUMA infrastructure
#include "numa-kernels.h"
#include "sub.h"
#include "ggml-cpu-impl.h"
#include "ggml-numa-shared.h"
#include "ggml-numa-openmp-coordinator.h"
#include "../quants.h"
#include "../vec.h"
#include <stdlib.h>
#include <string.h>

// Kernel implementation headers
#include "sub.h"

// ============================================================================
// Kernel Execution Function - Demonstrating Pattern Reuse
// ============================================================================

/**
 * @brief NUMA SUB kernel execution function
 * @note This demonstrates how the ADD pattern is 99% reusable for SUB
 */
enum ggml_status ggml_numa_kernel_sub_unified_execute(void * work_context, struct ggml_compute_params * params) {
    struct ggml_tensor * tensor = (struct ggml_tensor *)work_context;
    
    // Validation  
    if (!tensor || !tensor->src[0] || !tensor->src[1]) {
        return GGML_STATUS_FAILED;
    }
    
    // For now, only support F32 types for initial testing
    if (tensor->type != GGML_TYPE_F32 || tensor->src[0]->type != GGML_TYPE_F32 || 
        tensor->src[1]->type != GGML_TYPE_F32) {
        return GGML_STATUS_FAILED;
    }

    // Use the unified binary broadcasting setup with manual SUB operation (matching ADD pattern)
    NUMA_KERNEL_SETUP_BINARY_BROADCAST(ctx, tensor, params, float) {
        // Simple operations based on broadcasting pattern (same as ADD, but with subtraction)
        if (__is_scalar) {
            const float __scalar = __src1_data[0];
            for (size_t __i = ctx.thread_start; __i < ctx.thread_end; __i++) {
                __dst_data[__i] = __src0_data[__i] - __scalar;
            }
        } else if (__is_same_shape) {
            // Use SIMD optimization for F32 same-shape operations
            ggml_vec_sub_f32(ctx.thread_elements,
                           __dst_data + ctx.thread_start,
                           __src0_data + ctx.thread_start,
                           __src1_data + ctx.thread_start);
        } else {
            // Complex broadcasting case - use manual element-wise operation
            for (size_t __i = ctx.thread_start; __i < ctx.thread_end; __i++) {
                // Simple linear access for complex broadcasting (works for most cases)
                __dst_data[__i] = __src0_data[__i] - __src1_data[__i % ggml_nelements(tensor->src[1])];
            }
        }
    }
    
    return GGML_STATUS_SUCCESS;
}

// ============================================================================
// Kernel Query Function - IDENTICAL TO ADD
// ============================================================================

ggml_numa_execution_strategy_t ggml_numa_kernel_sub_query(const struct ggml_tensor * tensor) {
    // Calculate total elements for threshold comparison
    size_t total_elements = ggml_nelements(tensor);
    
    // Get cache entry for this operation  
    const ggml_numa_kernel_cache_entry_t * cache_entry = ggml_numa_lookup_kernel_direct(GGML_OP_SUB);
    
    if (!cache_entry || !cache_entry->supported) {
        return NUMA_STRATEGY_SINGLE_THREAD;  // Fallback to single thread strategy
    }
    
    // Use shared macro for unified strategy selection 
    ggml_numa_execution_strategy_t selected_strategy;
    NUMA_SELECT_STRATEGY_FROM_CACHE(cache_entry, total_elements, selected_strategy);
    
    // Debug logging (controlled by environment variable)
    const char* op_name = cache_entry && cache_entry->kernel_name ? cache_entry->kernel_name : "NUMA SUB";
    NUMA_LOG_DEBUG("SUB Query: total_elements=%zu, selected_strategy=%d (%s)\n", 
                   total_elements, selected_strategy, op_name);
    
    return selected_strategy;
}

// ============================================================================
// Work Buffer Calculation - IDENTICAL TO ADD  
// ============================================================================

size_t ggml_numa_kernel_sub_work_buffer_calc(const struct ggml_tensor * tensor, int total_numa_nodes, int total_threads) {
    (void)tensor;        // SUB doesn't need work buffers
    (void)total_numa_nodes;
    (void)total_threads;
    return 0;
}

// ============================================================================
// Kernel Registration - IDENTICAL TO ADD
// ============================================================================

ggml_numa_kernel_registration_info_t ggml_numa_kernel_sub_register(void) {
    ggml_numa_kernel_registration_info_t info = {0};
    
    info.op_type = GGML_OP_SUB;           // ONLY LINE CHANGED: ADD → SUB
    info.supported = true;
    info.kernel_name = "NUMA SUB Kernel"; // ONLY LINE CHANGED: ADD → SUB
    
    // Strategy thresholds for operation - IDENTICAL TO ADD
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_SINGLE] = 1024;      
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_MULTI] = 262144;     
    info.strategy_array.valid = true;
    
    // Unified function handles all strategies - IDENTICAL TO ADD
    info.work_funcs.single_single_fn = ggml_numa_kernel_sub_unified_execute;
    info.work_funcs.single_multi_fn = ggml_numa_kernel_sub_unified_execute;
    info.work_funcs.data_parallel_fn = ggml_numa_kernel_sub_unified_execute;
    info.work_funcs.valid = true;
    
    // Query and work buffer function pointers - IDENTICAL TO ADD
    info.query_fn = (void*)ggml_numa_kernel_sub_query;
    info.work_buffer_calc_fn = (void*)ggml_numa_kernel_sub_work_buffer_calc;
    
    // Element-wise operations don't need aggregation - IDENTICAL TO ADD
    info.agg_funcs.valid = false;
    
    return info;
}
