/**
 * @file div.c
 * @brief NUMA-aware element-wise division kernel implementation using shared broadcasting macros
 * 
 * This kernel demonstrates 99% code reuse from ADD kernel using the shared macro framework.
 * Only the arithmetic operation changes: val0 / val1 instead of val0 + val1.
 * 
 * @author David Sanftenberg
 */

#include "numa-kernels.h"
#include "div.h"
#include "ggml-cpu-impl.h"
#include "ggml-numa-shared.h"
#include "ggml-numa-openmp-coordinator.h"
#include "../quants.h"
#include "../vec.h"
#include <stdlib.h>
#include <string.h>

/**
 * @brief Execute F32 element-wise division using NUMA coordinator and macro-based broadcasting
 * 
 * Uses NUMA_COMPLEX_BROADCAST_LOOP macro for all coordinate calculation and broadcasting logic.
 * This provides 99% code reuse with ADD kernel - only the arithmetic operation differs.
 * 
 * @param work_context Tensor context (struct ggml_tensor*)
 * @param params Compute parameters with NUMA threading info
 * @return GGML_STATUS_SUCCESS on completion, GGML_STATUS_FAILED on error
 */
enum ggml_status ggml_numa_kernel_div_execute(void * work_context, struct ggml_compute_params * params) {
    struct ggml_tensor * tensor = (struct ggml_tensor *)work_context;
    
    // Input validation
    if (!tensor || !tensor->src[0] || !tensor->src[1]) {
        NUMA_LOG_DEBUG("DIV kernel: NULL tensor or source inputs\n");
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
                __dst_data[__i] = __src0_data[__i] / __scalar;  // DIV operation
            }
        } else if (__is_same_shape) {
            // Use element-wise division for F32 same-shape operations
            for (size_t __i = 0; __i < ctx.thread_elements; __i++) {
                __dst_data[ctx.thread_start + __i] = __src0_data[ctx.thread_start + __i] / __src1_data[ctx.thread_start + __i];
            }
        } else {
            // Complex broadcasting case - use reusable macro
            NUMA_COMPLEX_BROADCAST_LOOP(ctx, tensor, float, val0 / val1);
        }
    }
    
    return GGML_STATUS_SUCCESS;
}

/**
 * @brief Query function for DIV kernel - determines optimal execution strategy
 * 
 * Uses cache-based threshold lookup with shared macro for consistent strategy selection.
 * Strategy selection logic is identical to ADD kernel - only operation differs.
 * 
 * @param tensor Input tensor for strategy analysis
 * @return Execution strategy recommendation
 */
ggml_numa_execution_strategy_t ggml_numa_kernel_div_query(const struct ggml_tensor * tensor) {
    // Calculate total elements for strategy selection (hot path - must be fast)
    const size_t total_elements = ggml_nelements(tensor);
    
    // Get cache entry for DIV operation using direct lookup
    const ggml_numa_kernel_cache_entry_t * cache_entry = ggml_numa_lookup_kernel_direct(GGML_OP_DIV);
    
    if (!cache_entry || !cache_entry->supported) {
        return NUMA_STRATEGY_SINGLE_THREAD;  // Fallback to single thread strategy
    }
    
    // Use unified strategy selection macro for consistent behavior
    ggml_numa_execution_strategy_t selected_strategy;
    NUMA_SELECT_STRATEGY_FROM_CACHE(cache_entry, total_elements, selected_strategy);
    
    // Debug logging (controlled by environment variable)
    const char* op_name = cache_entry && cache_entry->kernel_name ? cache_entry->kernel_name : "NUMA DIV";
    NUMA_LOG_DEBUG("DIV query: %zu elements -> strategy %d (%s)\n", 
                   total_elements, selected_strategy, op_name);
    
    return selected_strategy;
}

/**
 * @brief Calculate work buffer requirements for DIV kernel
 * 
 * Element-wise DIV operation processes data in-place with minimal intermediate storage.
 * No additional work buffer space required beyond input/output tensors.
 * 
 * @param tensor Input tensor for analysis
 * @param total_numa_nodes Number of NUMA nodes participating
 * @param total_threads Total thread count across all nodes
 * @return Work buffer size in bytes (0 for DIV)
 */
size_t ggml_numa_kernel_div_work_buffer_calc(const struct ggml_tensor * tensor, int total_numa_nodes, int total_threads) {
    // DIV operation is purely element-wise with no intermediate storage
    // No additional work buffer space required
    (void)tensor;
    (void)total_numa_nodes;
    (void)total_threads;
    
    return 0;
}

/**
 * @brief Register DIV kernel with NUMA system using shared infrastructure
 * 
 * Registration info is nearly identical to ADD kernel with same thresholds and patterns.
 * Demonstrates how shared macro framework enables trivial kernel addition.
 * 
 * @return Kernel registration information for NUMA system
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_div_register(void) {
    ggml_numa_kernel_registration_info_t info = {0};
    
    info.op_type = GGML_OP_DIV;
    info.supported = true;
    info.kernel_name = "NUMA DIV Kernel";
    
    // Strategy thresholds identical to ADD kernel
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_SINGLE] = 1024;      // Single thread below 1K elements
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_MULTI] = 262144;     // Multi-thread below 256K elements
    // Above 256K elements: data-parallel strategy
    info.strategy_array.valid = true;
    
    // Unified function handles all strategies (identical to ADD)
    info.work_funcs.single_single_fn = ggml_numa_kernel_div_execute;
    info.work_funcs.single_multi_fn = ggml_numa_kernel_div_execute;
    info.work_funcs.data_parallel_fn = ggml_numa_kernel_div_execute;
    info.work_funcs.valid = true;
    
    // Query and work buffer function pointers (same pattern as ADD)
    info.query_fn = (void*)ggml_numa_kernel_div_query;
    info.work_buffer_calc_fn = (void*)ggml_numa_kernel_div_work_buffer_calc;
    
    // Element-wise operations don't need aggregation (same as ADD)
    info.agg_funcs.valid = false;
    
    return info;
}
