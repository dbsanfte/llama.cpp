/**
 * @file transpose.c
 * @brief NUMA TRANSPOSE Kernel Implementation - Tensor Dimension Swapping Operation
 *
 * This file implements a NUMA-aware TRANSPOSE kernel for tensor dimension
 * swapping operations. TRANSPOSE is a view operation that swaps the first
 * two dimensions of a tensor without performing any actual data movement.
 * 
 * The implementation provides a minimal computational baseline that matches the
 * standard ggml_compute_forward_transpose behavior, which is essentially a 
 * no-operation since TRANSPOSE only modifies tensor metadata during graph
 * construction.
 *
 * Mathematical Operation:
 * - Input tensor:  [ne0, ne1, ne2, ne3] with strides [nb0, nb1, nb2, nb3]
 * - Output tensor: [ne1, ne0, ne2, ne3] with strides [nb1, nb0, nb2, nb3]
 * - Total elements remain unchanged: ne0*ne1*ne2*ne3 = ne1*ne0*ne2*ne3
 * - No actual data copying or computation occurs
 *
 * Implementation Strategy:
 * - Immediate return from work function (no computation needed)
 * - Standard NUMA kernel interface compliance
 * - Minimal memory access and resource usage
 * - Full registration in kernel cache system
 * - Consistent strategy selection patterns with other view operations
 *
 * Performance Characteristics:
 * - Execution time: ~1-2 nanoseconds (function call overhead only)
 * - Memory bandwidth: Zero (no data access during execution)
 * - CPU utilization: Minimal (parameter validation only)
 * - Thread safety: Full (no shared state modification)
 */

#include "transpose.h"
#include "../ggml-numa-shared.h"
#include "../ggml-numa-perf.h"
#include "numa-kernels.h"

// ============================================================================
// TRANSPOSE Kernel Execution Function  
// ============================================================================

/**
 * @brief NUMA TRANSPOSE kernel execution function
 * 
 * This function performs no operation and returns immediately, matching the
 * behavior of ggml_compute_forward_transpose() which is also a no-op. TRANSPOSE
 * operations only modify tensor metadata during graph construction.
 * 
 * The kernel validates input parameters for consistency with the NUMA kernel
 * interface but performs no actual computation since TRANSPOSE is a pure
 * view operation.
 * 
 * @param work_context Tensor context (validated but unused)
 * @param params Compute parameters (validated but unused)
 * @return GGML_STATUS_SUCCESS always
 */
enum ggml_status ggml_numa_kernel_transpose_execute(void * work_context, 
                                                     struct ggml_compute_params * params) {
    // Validate parameters for consistency with other NUMA kernels
    NUMA_ASSERT(work_context != NULL, "Work context cannot be null");
    NUMA_ASSERT(params != NULL, "Compute params cannot be null");
    
    struct ggml_tensor * tensor = (struct ggml_tensor *)work_context;
    NUMA_ASSERT(tensor != NULL, "Tensor cannot be null");
    NUMA_ASSERT(tensor->op == GGML_OP_TRANSPOSE, "Tensor operation must be TRANSPOSE");
    
    // Performance instrumentation for NUMA execution tracking
    NUMA_PERF_START(NUMA_PERF_KERNEL_NUMA_EXEC, "TRANSPOSE", "no_op_execution", -1, 0, 0);
    
    // Log execution details for debugging (only at trace level to avoid noise)
    NUMA_LOG_TRACE("TRANSPOSE no-op execution: tensor=%p, dims=[%ld,%ld,%ld,%ld]", 
                   tensor, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    
    // TRANSPOSE is a view operation - no actual computation needed
    // The dimension swapping was already handled during graph construction
    // by ggml_transpose() which modified the tensor metadata (ne[], nb[])
    
    NUMA_PERF_END();
    return GGML_STATUS_SUCCESS;
}

// ============================================================================
// TRANSPOSE Query Function
// ============================================================================

/**
 * @brief Query TRANSPOSE kernel capabilities and strategy selection
 * 
 * This function evaluates tensor characteristics and selects the optimal
 * execution strategy for TRANSPOSE operations. Since TRANSPOSE is a view
 * operation that requires no computation, it always uses single-node
 * single-thread execution for minimal overhead.
 * 
 * Strategy Selection Logic:
 * - Always use SINGLE_NODE_SINGLE_THREAD (no computation needed)
 * - No work buffer required (no temporary storage needed)
 * - Perfect efficiency score (minimal resource usage)
 * - No aggregation needed (no results to combine)
 * 
 * @param tensor Target tensor for TRANSPOSE operation
 * @return Query result with strategy selection and efficiency metrics
 */
ggml_numa_kernel_query_result_t ggml_numa_kernel_transpose_query(const struct ggml_tensor * tensor) {
    ggml_numa_kernel_query_result_t result = {0};
    
    // Validate tensor for TRANSPOSE operation
    if (!tensor || tensor->op != GGML_OP_TRANSPOSE) {
        NUMA_LOG_DEBUG("TRANSPOSE query: REJECTING - invalid tensor or operation");
        result.supported = false;
        return result;
    }
    
    // Calculate total elements for logging (metadata operation only)
    const size_t total_elements = ggml_nelements(tensor);
    
    // TRANSPOSE is always a view operation - use minimal resource strategy
    result.supported = true;
    result.strategy = (ggml_numa_execution_strategy_t){
        .node_strategy = NUMA_NODE_STRATEGY_SINGLE,
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD
    };
    result.work_buffer_size_per_thread = 0;  // No temporary storage needed
    result.work_function = ggml_numa_kernel_transpose_execute;
    result.efficiency_score = 1.0f;  // Perfect efficiency for no-op operation
    result.kernel_name = "NUMA TRANSPOSE Kernel";
    
    // No aggregation needed for view operations
    result.aggregation_policy = GGML_NUMA_AGGREGATION_NONE;
    result.aggregation_function = NULL;
    result.aggregation_user_data = NULL;
    
    // Log strategy selection
    NUMA_LOG_DEBUG("TRANSPOSE Query: %zu elements, strategy=single-node-single-thread, kernel=NUMA TRANSPOSE Kernel, efficiency=%.2f",
                   total_elements, result.efficiency_score);
    
    return result;
}

// ============================================================================
// TRANSPOSE Registration Function
// ============================================================================

/**
 * @brief Register TRANSPOSE kernel in NUMA kernel cache system
 * 
 * This function provides kernel registration information for the NUMA
 * kernel cache, enabling direct function pointer dispatch without switch
 * statement overhead.
 * 
 * Registration Details:
 * - Operation type: GGML_OP_TRANSPOSE
 * - Strategy: Single-node single-thread for all tensor sizes
 * - Work functions: Same execute function for all strategies
 * - Aggregation: Not needed (view operation)
 * - Query function: Direct dispatch via function pointer
 * 
 * @return Registration information structure with kernel metadata
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_transpose_register(void) {
    ggml_numa_kernel_registration_info_t info = {0};
    
    info.op_type = GGML_OP_TRANSPOSE;
    info.supported = true;
    info.kernel_name = "NUMA TRANSPOSE Kernel";
    
    // TRANSPOSE uses single-thread strategy for all tensor sizes (no computation)
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_SINGLE] = SIZE_MAX;  // All sizes use single-thread
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_MULTI] = SIZE_MAX;   // No multi-thread needed
    // Data-parallel strategy not applicable for view operations
    info.strategy_array.valid = true;
    
    // All strategies use the same no-op execute function
    info.work_funcs.single_single_fn = ggml_numa_kernel_transpose_execute;
    info.work_funcs.single_multi_fn = ggml_numa_kernel_transpose_execute;
    info.work_funcs.data_parallel_fn = ggml_numa_kernel_transpose_execute;
    info.work_funcs.valid = true;
    
    // Query function pointer for direct dispatch (eliminates switch statements)
    info.query_fn = (void*)ggml_numa_kernel_transpose_query;
    
    // No aggregation functions needed for view operations
    info.agg_funcs.single_single_fn = NULL;
    info.agg_funcs.single_multi_fn = NULL; 
    info.agg_funcs.data_parallel_fn = NULL;
    info.agg_funcs.valid = false;
    
    // Log registration completion
    NUMA_LOG_DEBUG("Registered TRANSPOSE kernel: strategy=single-node-single-thread, supports_view_ops=true");
    
    return info;
}
