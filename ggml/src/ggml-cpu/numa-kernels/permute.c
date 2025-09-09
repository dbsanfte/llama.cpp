/**
 * @file permute.c
 * @brief NUMA kernel for PERMUTE operation (metadata-only tensor dimension permutation)
 * 
 * @author David Sanftenberg
 * @date 2025-09-09
 * 
 * This kernel implements the GGML_OP_PERMUTE operation for NUMA-aware execution.
 * PERMUTE is a metadata-only operation that reorders tensor dimensions according
 * to a specified permutation. Like the reference implementation, this kernel
 * performs no actual data movement - the permutation is handled at the tensor
 * metadata level by the ggml tensor system.
 * 
 * Key characteristics:
 * - Zero computational overhead (no-op execution)
 * - Metadata-only transformation
 * - Always uses single-thread/single-node strategy
 * - Compatible with all tensor data types
 */

#include "permute.h"
#include "ggml-numa-shared.h"
#include <stddef.h>

/**
 * @brief Execute PERMUTE operation (no-op implementation)
 * 
 * PERMUTE is a metadata-only operation that reorders tensor dimensions.
 * The actual permutation is handled by the ggml tensor system at the
 * metadata level, so this kernel performs no computation.
 * 
 * @param work_context Pointer to the destination tensor (cast from ggml_tensor*)
 * @param params Compute parameters (unused for no-op)
 * @return GGML_STATUS_SUCCESS always (no-op operations cannot fail)
 */
enum ggml_status ggml_numa_kernel_permute_execute(void * work_context, struct ggml_compute_params * params) {
    // PERMUTE is a metadata-only operation - no computation required
    // The dimension permutation is handled by the ggml tensor system
    
    // This function should never be called.
    NUMA_ASSERT(false, "PERMUTE kernel execute function should not be called - metadata-only operation");
    return GGML_STATUS_SUCCESS;
}

/**
 * @brief Query optimal execution strategy for PERMUTE operation
 * 
 * PERMUTE is always executed using single-thread/single-node strategy
 * since it's a metadata-only operation with zero computational cost.
 * 
 * @param tensor Target tensor for permutation (used for validation)
 * @return NUMA_STRATEGY_SINGLE_THREAD always
 */
ggml_numa_execution_strategy_t ggml_numa_kernel_permute_query(const struct ggml_tensor * tensor) {
    NUMA_ASSERT(tensor != NULL, "Tensor cannot be null");
    
    // Metadata-only operations always use single-thread strategy for minimal overhead
    return NUMA_STRATEGY_SINGLE_THREAD;
}

/**
 * @brief Calculate work buffer size for PERMUTE operation
 * 
 * PERMUTE requires no work buffers since it's a metadata-only operation.
 * 
 * @param tensor Target tensor (unused for no-op)
 * @param total_numa_nodes Total NUMA nodes (unused for no-op)
 * @param total_threads Total threads (unused for no-op)
 * @return 0 (no work buffer needed)
 */
size_t ggml_numa_kernel_permute_work_buffer_calc(const struct ggml_tensor * tensor, int total_numa_nodes, int total_threads) {
    NUMA_ASSERT(tensor != NULL, "Tensor cannot be null");
    NUMA_ASSERT(total_numa_nodes > 0, "Total NUMA nodes must be positive");
    NUMA_ASSERT(total_threads > 0, "Total threads must be positive");
    
    // Metadata-only operations require no work buffers
    return 0;
}

/**
 * @brief Register PERMUTE kernel with NUMA system
 * 
 * Configures registration info for the PERMUTE operation with metadata-only
 * characteristics and single-thread execution strategy.
 * 
 * @return Fully populated registration info structure
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_permute_register(void) {
    ggml_numa_kernel_registration_info_t info = {0};
    
    info.op_type = GGML_OP_PERMUTE;
    info.supported = true;
    info.kernel_name = "NUMA PERMUTE Kernel (Metadata-Only)";
    info.is_noop = true; // This kernel doesn't do any calculations
    
    // Strategy thresholds for metadata-only operation
    // Use SIZE_MAX to ensure single-thread strategy for all tensor sizes
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_SINGLE] = SIZE_MAX;
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_MULTI] = SIZE_MAX;
    // All sizes use single-thread strategy (metadata-only)
    info.strategy_array.valid = true;
    
    // Function pointers - all strategies use same no-op implementation
    info.work_funcs.single_single_fn = ggml_numa_kernel_permute_execute;
    info.work_funcs.single_multi_fn = ggml_numa_kernel_permute_execute;
    info.work_funcs.data_parallel_fn = ggml_numa_kernel_permute_execute;
    info.work_funcs.valid = true;
    
    // Query function pointer for direct dispatch
    info.query_fn = (void*)ggml_numa_kernel_permute_query;
    
    // Work buffer calculation function pointer
    info.work_buffer_calc_fn = (void*)ggml_numa_kernel_permute_work_buffer_calc;
    
    // No aggregation functions needed for metadata-only operations
    info.agg_funcs.single_single_fn = NULL;
    info.agg_funcs.single_multi_fn = NULL; 
    info.agg_funcs.data_parallel_fn = NULL;
    info.agg_funcs.valid = false;
    
    return info;
}
