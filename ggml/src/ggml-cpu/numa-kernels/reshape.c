/**
 * @file reshape.c
 * @brief NUMA RESHAPE Kernel Implementation - Tensor Shape Metadata Operation
 *
 * This file implements a NUMA-aware RESHAPE kernel for tensor shape transformation
 * operations. RESHAPE is a metadata-only operation that changes tensor dimensions
 * without performing any data movement or computation.
 * 
 * The implementation provides a minimal computational baseline that matches the
 * standard ggml_compute_forward_reshape behavior, which is essentially a no-operation
 * since RESHAPE only modifies tensor metadata during graph construction.
 *
 * Implementation Strategy:
 * - Immediate return from work function (no computation needed)
 * - Standard NUMA kernel interface compliance
 * - Minimal memory access and resource usage
 * - Full registration in kernel cache system
 * - Consistent strategy selection patterns with other view operations
 *
 * Mathematical Properties:
 * - Preserves total element count: nelements(input) == nelements(output)
 * - Input tensor must be contiguous
 * - Only shape metadata (ne[], nb[]) is modified
 * - No actual data processing occurs during execution
 */

#include "reshape.h"
#include "../ggml-numa-shared.h"
#include "numa-kernels.h"

/**
 * @brief NUMA RESHAPE kernel execution function
 * 
 * This function performs no operation and returns immediately, matching the
 * behavior of ggml_compute_forward_reshape() which is also a no-op. RESHAPE
 * operations only modify tensor metadata during graph construction.
 * 
 * Performance Characteristics:
 * - Execution time: ~1-2 nanoseconds (function call overhead only)
 * - Memory access: Parameter validation only
 * - CPU cycles: Minimal (function prologue/epilogue + return)
 * - Thread safety: Full (no shared state modification)
 * 
 * @param work_context Tensor context (validated but unused)
 * @param params Compute parameters (validated but unused)
 * @return GGML_STATUS_SUCCESS always
 */
enum ggml_status ggml_numa_kernel_reshape_execute(void * work_context, 
                                                   struct ggml_compute_params * params) {
    // Validate parameters for consistency with other NUMA kernels
    NUMA_ASSERT(work_context != NULL, "Work context cannot be null");
    NUMA_ASSERT(params != NULL, "Compute params cannot be null");
    
    // Get tensor for additional validation and logging
    struct ggml_tensor * tensor = (struct ggml_tensor *)work_context;
    NUMA_ASSERT(tensor->op == GGML_OP_RESHAPE, "Expected RESHAPE operation");
    
    // Get NUMA execution context for logging
    extern __thread int ggml_current_numa_node;
    extern __thread bool ggml_numa_is_data_parallel_execution;
    
    // Log execution strategy in standardized format for integration test parsing
    NUMA_LOG_STRATEGY_SINGLE_SINGLE("RESHAPE");  // Reshape is always single-single strategy (view operation)
    
    NUMA_LOG_TRACE("NUMA RESHAPE kernel executing on node %d (data_parallel=%s) - tensor: %s",
                   ggml_current_numa_node, 
                   ggml_numa_is_data_parallel_execution ? "true" : "false",
                   tensor->name ? tensor->name : "unnamed");
    
    NUMA_LOG_VERBOSE("RESHAPE Node %d: No-op execution for tensor [%ld,%ld,%ld,%ld] -> [%ld,%ld,%ld,%ld]",
                     ggml_current_numa_node,
                     tensor->src[0] ? tensor->src[0]->ne[0] : 0,
                     tensor->src[0] ? tensor->src[0]->ne[1] : 0, 
                     tensor->src[0] ? tensor->src[0]->ne[2] : 0,
                     tensor->src[0] ? tensor->src[0]->ne[3] : 0,
                     tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    
    // RESHAPE: Return immediately with success
    // No computation required - metadata operation only
    return GGML_STATUS_SUCCESS;
}

/**
 * @brief Query function for NUMA RESHAPE kernel strategy selection
 * 
 * Returns strategy recommendations for RESHAPE operations with standard thresholds
 * consistent with other view operations. Since RESHAPE requires no computation,
 * all strategies have equal efficiency.
 * 
 * Strategy Selection Logic:
 * - ≤1024 elements: Single-node, single-thread (minimal overhead)
 * - 1025-262144 elements: Single-node, multi-thread  
 * - >262144 elements: Multi-node, data-parallel (consistency with large tensors)
 * 
 * @param tensor Target tensor for strategy selection
 * @return Kernel query result with strategy and efficiency metrics
 */
ggml_numa_kernel_query_result_t ggml_numa_kernel_reshape_query(const struct ggml_tensor * tensor) {
    
    NUMA_ASSERT(tensor != NULL, "Tensor cannot be null");
    
    ggml_numa_kernel_query_result_t result = { .supported = false };
    
    // Validate this is a RESHAPE operation
    if (tensor->op != GGML_OP_RESHAPE) {
        NUMA_LOG_DEBUG("RESHAPE Query: Operation %s not supported (expected RESHAPE)", 
                      ggml_op_name(tensor->op));
        return result;
    }
    
    // Calculate total elements for strategy selection
    const size_t total_elements = ggml_nelements(tensor);
    
    // Check if this kernel is actually registered and supported
    if (!ggml_numa_is_kernel_supported(GGML_OP_RESHAPE)) {
        NUMA_LOG_DEBUG("RESHAPE kernel not supported - registration disabled");
        result.supported = false;
        return result;
    }
    
    // Get cache entry for this operation
    const ggml_numa_kernel_cache_entry_t * cache_entry = ggml_numa_lookup_kernel_direct(GGML_OP_RESHAPE);
    if (!cache_entry || !cache_entry->strategy_array.valid) {
        NUMA_LOG_DEBUG("RESHAPE cache entry not found or invalid - falling back to unsupported");
        result.supported = false;
        return result;
    }
    
    // RESHAPE is always supported regardless of tensor configuration
    result.supported = true;
    result.work_buffer_size_per_thread = 0; // No work buffer needed
    result.work_function = (ggml_numa_work_function_t)ggml_numa_kernel_reshape_execute;
    result.efficiency_score = 1.0f; // Perfect efficiency for no-op
    result.kernel_name = "NUMA RESHAPE Kernel";
    result.aggregation_policy = GGML_NUMA_AGGREGATION_NONE; // No aggregation needed
    result.aggregation_function = NULL;
    result.aggregation_user_data = NULL;
    
    // Use shared macro for unified strategy selection
    ggml_numa_execution_strategy_t selected_strategy;
    NUMA_SELECT_STRATEGY_FROM_CACHE(cache_entry, total_elements, selected_strategy);
    
    result.strategy = selected_strategy;
    
    // Apply force strategy override if environment variable is set
    // Note: RESHAPE is no-op, so all functions point to the same implementation
    bool strategy_overridden = ggml_numa_apply_kernel_force_strategy(&result, "RESHAPE",
        (ggml_numa_work_function_t)ggml_numa_kernel_reshape_execute, // single-single
        (ggml_numa_work_function_t)ggml_numa_kernel_reshape_execute, // single-multi  
        (ggml_numa_work_function_t)ggml_numa_kernel_reshape_execute  // data-parallel
    );
    
    NUMA_LOG_DEBUG("RESHAPE Query: %zu elements, strategy=%s, kernel=%s, efficiency=%.2f%s",
                   total_elements, 
                   result.strategy.node_strategy == NUMA_NODE_STRATEGY_SINGLE ? "single-node" : "data-parallel",
                   result.kernel_name, result.efficiency_score,
                   strategy_overridden ? " [STRATEGY OVERRIDDEN]" : "");
    
    return result;
}

/**
 * @brief Register NUMA RESHAPE kernel in the kernel cache system
 * 
 * Returns registration information for the RESHAPE kernel including strategy
 * thresholds, work functions, and aggregation policies suitable for view operations.
 * 
 * @return Registration information for the RESHAPE kernel
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_reshape_register(void) {
    ggml_numa_kernel_registration_info_t info = {0};
    
    info.op_type = GGML_OP_RESHAPE;
    info.supported = true;
    info.kernel_name = "NUMA RESHAPE Kernel";
    
    // Strategy thresholds for RESHAPE operations (view operation pattern)
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_SINGLE] = 9999999;      // Single thread strategy
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_MULTI] = 9999999; //TODO: remove     // Multi-thread strategy
    // Above this: data-parallel strategy
    info.strategy_array.valid = true;
    
    // Work function pointers for different strategies (all use same no-op function)
    info.work_funcs.single_single_fn = ggml_numa_kernel_reshape_execute;
    info.work_funcs.single_multi_fn = ggml_numa_kernel_reshape_execute;
    info.work_funcs.data_parallel_fn = ggml_numa_kernel_reshape_execute;
    info.work_funcs.valid = true;
    
    // No aggregation needed for RESHAPE (no-op operation)
    info.agg_funcs.single_single_fn = NULL;
    info.agg_funcs.single_multi_fn = NULL; 
    info.agg_funcs.data_parallel_fn = NULL;
    info.agg_funcs.valid = false;
    
    // RESHAPE is a view operation, effectively a no-op
    info.is_noop = true;
    
    return info;
}
