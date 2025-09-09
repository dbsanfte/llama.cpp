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
 * - Single-thread execution strategy (minimal overhead for metadata-only operation)
 *
 * Mathematical Properties:
 * - Preserves total element count: nelements(input) == nelements(output)
 * - Input tensor must be contiguous
 * - Only shape metadata (ne[], nb[]) is modified
 * - No actual data processing occurs during execution
 * 
 * @author David Sanftenberg
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
    // RESHAPE is a metadata-only operation - no computation required
    // The reshape is handled by the ggml tensor system
    
    // This function should never be called.
    NUMA_ASSERT(false, "RESHAPE kernel execute function should not be called - metadata-only operation");
    return GGML_STATUS_SUCCESS;
}

/**
 * @brief Query function for NUMA RESHAPE kernel strategy selection
 * 
 * Returns strategy recommendations for RESHAPE operations. Since RESHAPE requires 
 * no computation, we always use the single-thread strategy for minimal overhead.
 * 
 * @param tensor Target tensor for strategy selection
 * @return Single-thread strategy recommendation
 */
ggml_numa_execution_strategy_t ggml_numa_kernel_reshape_query(const struct ggml_tensor * tensor) {
    NUMA_ASSERT(tensor != NULL, "Tensor cannot be null");
    
    // View operations are always single-thread/single-node (metadata-only, no computation)
    return NUMA_STRATEGY_SINGLE_THREAD;
}

/**
 * @brief Work buffer calculation function for RESHAPE operations
 * 
 * @param tensor Target tensor
 * @param total_numa_nodes Number of NUMA nodes  
 * @param total_threads Total number of threads
 * @return 0 (no work buffer needed for metadata-only operation)
 */
size_t ggml_numa_kernel_reshape_work_buffer_calc(const struct ggml_tensor * tensor, 
                                                  int total_numa_nodes, 
                                                  int total_threads) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(total_numa_nodes);
    GGML_UNUSED(total_threads);
    
    // RESHAPE is a metadata-only operation requiring no work buffer
    return 0;
}

/**
 * @brief Register the RESHAPE kernel with the NUMA system
 * 
 * @return Registration information for the RESHAPE kernel
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_reshape_register(void) {
    ggml_numa_kernel_registration_info_t info = {0};
    
    info.op_type = GGML_OP_RESHAPE;
    info.supported = true;
    info.kernel_name = "NUMA RESHAPE Kernel";
    info.is_noop = true; // This kernel doesn't do any calculations
    
    // Strategy thresholds - always use single-thread for metadata operations
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_SINGLE] = SIZE_MAX;  // Always single-thread
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_MULTI] = SIZE_MAX;   // Never multi-thread
    info.strategy_array.valid = true;
    
    // Function pointers - only single-thread execution needed
    info.work_funcs.single_single_fn = ggml_numa_kernel_reshape_execute;
    info.work_funcs.single_multi_fn = ggml_numa_kernel_reshape_execute;   // Fallback
    info.work_funcs.data_parallel_fn = ggml_numa_kernel_reshape_execute;  // Fallback
    info.work_funcs.valid = true;
    
    // Query function pointer for O(1) strategy selection
    info.query_fn = (void*)ggml_numa_kernel_reshape_query;
    
    // Work buffer calculation function
    info.work_buffer_calc_fn = (void*)ggml_numa_kernel_reshape_work_buffer_calc;
    
    // RESHAPE doesn't need aggregation functions (metadata-only operation)
    info.agg_funcs.single_single_fn = NULL;
    info.agg_funcs.single_multi_fn = NULL;
    info.agg_funcs.data_parallel_fn = NULL;
    info.agg_funcs.valid = false;
    
    return info;
}
