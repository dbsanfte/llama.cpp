/**
 * @file add.c
 * @brief NUMA-aware ADD kernel implementation using refined macro system
 * @author David Sanftenberg
 */

#include "numa-kernels.h"
#include "add.h"
#include "ggml-cpu-impl.h"
#include "ggml-numa-shared.h"
#include "ggml-numa-openmp-coordinator.h"
#include "../quants.h"
#include "../vec.h"
#include <stdlib.h>
#include <string.h>

// ============================================================================
// ADD Kernel Implementation (F32 Only for Testing Refined Macros)
// ============================================================================

/**
 * @brief Unified ADD kernel execution using refined broadcasting macro system
 */
enum ggml_status ggml_numa_kernel_add_unified_execute(void * work_context, struct ggml_compute_params * params) {
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

    // Use the new unified binary broadcasting setup and operation
    NUMA_KERNEL_SETUP_BINARY_BROADCAST(ctx, tensor, params, float) {
        // Simple operations based on broadcasting pattern
        if (__is_scalar) {
            const float __scalar = __src1_data[0];
            for (size_t __i = ctx.thread_start; __i < ctx.thread_end; __i++) {
                __dst_data[__i] = __src0_data[__i] + __scalar;
            }
        } else if (__is_same_shape) {
            // Use SIMD optimization for F32 same-shape operations
            ggml_vec_add_f32(ctx.thread_elements,
                           __dst_data + ctx.thread_start,
                           __src0_data + ctx.thread_start,
                           __src1_data + ctx.thread_start);
        } else {
            // Complex broadcasting case - use reusable macro
            NUMA_COMPLEX_BROADCAST_LOOP(ctx, tensor, float, val0 + val1);
        }
    }
    
    return GGML_STATUS_SUCCESS;
}

// ============================================================================
// Kernel Query Function
// ============================================================================

ggml_numa_execution_strategy_t ggml_numa_kernel_add_query(const struct ggml_tensor * tensor) {
    // Calculate total elements for strategy selection (hot path - must be fast)
    const size_t total_elements = ggml_nelements(tensor);
    
    // Get cache entry for ADD operation using direct lookup
    const ggml_numa_kernel_cache_entry_t * cache_entry = ggml_numa_lookup_kernel_direct(GGML_OP_ADD);
    
    if (!cache_entry || !cache_entry->supported) {
        return NUMA_STRATEGY_SINGLE_THREAD;  // Fallback to single thread strategy
    }
    
    // Use unified strategy selection macro for consistent behavior
    ggml_numa_execution_strategy_t selected_strategy;
    NUMA_SELECT_STRATEGY_FROM_CACHE(cache_entry, total_elements, selected_strategy);
    
    // Debug logging (controlled by environment variable)
    const char* op_name = cache_entry && cache_entry->kernel_name ? cache_entry->kernel_name : "NUMA ADD";
    NUMA_LOG_DEBUG("ADD query: %zu elements -> strategy %d (%s)\n", 
                   total_elements, selected_strategy, op_name);
    
    return selected_strategy;
}

// ============================================================================
// Work Buffer Calculation Function
// ============================================================================

size_t ggml_numa_kernel_add_work_buffer_calc(const struct ggml_tensor * tensor, 
                                            int total_numa_nodes, int total_threads) {
    // ADD operations don't need work buffers for F32-only implementation
    (void)tensor;
    (void)total_numa_nodes;
    (void)total_threads;
    return 0;
}

// ============================================================================
// Kernel Registration Function
// ============================================================================

ggml_numa_kernel_registration_info_t ggml_numa_kernel_add_register(void) {
    ggml_numa_kernel_registration_info_t info = {0};
    
    info.op_type = GGML_OP_ADD;
    info.supported = true;
    info.kernel_name = "NUMA ADD Kernel";
    
    // Strategy thresholds for ADD operation
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_SINGLE] = 1024;      // Single thread below 1K elements
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_MULTI] = 262144;     // Multi-thread below 256K elements
    // Above 256K elements: data-parallel strategy
    info.strategy_array.valid = true;
    
    // Function pointers for different strategies (unified function handles all)
    info.work_funcs.single_single_fn = ggml_numa_kernel_add_unified_execute;
    info.work_funcs.single_multi_fn = ggml_numa_kernel_add_unified_execute;
    info.work_funcs.data_parallel_fn = ggml_numa_kernel_add_unified_execute;
    info.work_funcs.valid = true;
    
    // Query function pointer for direct dispatch
    info.query_fn = (void*)ggml_numa_kernel_add_query;
    
    // Work buffer calculation function pointer
    info.work_buffer_calc_fn = (void*)ggml_numa_kernel_add_work_buffer_calc;
    
    // ADD operations don't need aggregation functions
    info.agg_funcs.single_single_fn = NULL;
    info.agg_funcs.single_multi_fn = NULL; 
    info.agg_funcs.data_parallel_fn = NULL;
    info.agg_funcs.valid = false;
    
    return info;
}
