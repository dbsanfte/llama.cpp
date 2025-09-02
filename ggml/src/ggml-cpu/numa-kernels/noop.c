/**
 * @file noop.c
 * @brief NUMA NOOP Kernel Implementation - Performance Testing Operation
 *
 * This file implements a NUMA-aware NOOP (No Operation) kernel for performance
 * testing and benchmarking the NUMA kernel dispatch system overhead.
 * 
 * The implementation provides a minimal computational baseline that allows
 * measurement of the pure dispatch overhead introduced by the NUMA coordinator,
 * executor, and kernel registration system compared to direct ggml-cpu execution.
 *
 * Implementation Strategy:
 * - Immediate return from work function (no computation)
 * - Standard NUMA kernel interface compliance
 * - Minimal memory access and resource usage
 * - Full registration in kernel cache system
 * - Consistent strategy selection patterns
 *
 * Benchmarking Usage:
 * Compare execution time of GGML_OP_NUMA_NOOP (this kernel) vs 
 * GGML_OP_NUMA_FALLBACK_NOOP (ggml-cpu.c implementation) to measure
 * the performance difference between NUMA dispatch and direct dispatch.
 */

#include "noop.h"
#include "../ggml-numa-shared.h"
#include "numa-kernels.h"

/**
 * @brief NUMA NOOP kernel execution function
 * 
 * This function performs no operation and returns immediately. It follows the
 * standard NUMA kernel work function interface but does zero computational work.
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
enum ggml_status ggml_numa_kernel_noop_execute(void * work_context, 
                                                struct ggml_compute_params * params) {
    // Validate parameters for consistency with other NUMA kernels
    NUMA_ASSERT(work_context != NULL, "Work context cannot be null");
    NUMA_ASSERT(params != NULL, "Compute params cannot be null");
    
    // Get NUMA execution context for logging consistency
    extern __thread int ggml_current_numa_node;
    extern __thread bool ggml_numa_is_data_parallel_execution;
    
    NUMA_LOG_TRACE("NUMA NOOP kernel executing on node %d (data_parallel=%s)",
                   ggml_current_numa_node, 
                   ggml_numa_is_data_parallel_execution ? "true" : "false");
    
    // NOOP: Return immediately with success
    // This measures pure NUMA dispatch overhead without computation
    return GGML_STATUS_SUCCESS;
}

/**
 * @brief Query function for NUMA NOOP kernel strategy selection
 * 
 * Returns strategy recommendations for NOOP operations with custom thresholds
 * for performance testing different execution strategies. Since NOOP requires
 * no computation, all strategies have equal efficiency.
 * 
 * Strategy Selection Logic (Custom for Testing):
 * - ≤256 elements: Single-node, single-thread
 * - 257-512 elements: Single-node, multi-thread  
 * - 513-1024 elements: Multi-node, data-parallel
 * - >1024 elements: Multi-node, data-parallel (large)
 * 
 * @param tensor Target tensor for strategy selection
 * @return Kernel query result with strategy and efficiency metrics
 */
ggml_numa_kernel_query_result_t ggml_numa_kernel_noop_query(
    const struct ggml_tensor * tensor) {
    
    NUMA_ASSERT(tensor != NULL, "Tensor cannot be null");
    
    // Calculate total elements from tensor
    size_t total_elements = ggml_nelements(tensor);
    
    ggml_numa_kernel_query_result_t result = { .supported = false };
    
    // Validate this is a NUMA_NOOP operation
    if (tensor->op != GGML_OP_NUMA_NOOP) {
        return result;
    }
    
    // NOOP is always supported regardless of tensor configuration
    result.supported = true;
    result.work_buffer_size_per_thread = 0; // No work buffer needed
    result.work_function = (ggml_numa_work_function_t)ggml_numa_kernel_noop_execute;
    result.efficiency_score = 1.0f; // Perfect efficiency for NOOP
    result.aggregation_policy = GGML_NUMA_AGGREGATION_NONE; // No aggregation needed
    result.aggregation_function = NULL;
    result.aggregation_user_data = NULL;
    
    // Strategy selection using custom thresholds for testing
    if (total_elements <= 256) {
        // Small tensors: Single-node, single-thread
        result.strategy.node_strategy = NUMA_NODE_STRATEGY_SINGLE;
        result.strategy.on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD;
        result.kernel_name = "NUMA NOOP Kernel (Single-Single)";
    } else if (total_elements <= 512) {
        // Medium tensors: Single-node, multi-thread
        result.strategy.node_strategy = NUMA_NODE_STRATEGY_SINGLE;
        result.strategy.on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD;
        result.kernel_name = "NUMA NOOP Kernel (Single-Multi)";
    } else if (total_elements <= 1024) {
        // Large tensors: Data-parallel across nodes
        result.strategy.node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL;
        result.strategy.on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD;
        result.kernel_name = "NUMA NOOP Kernel (Data-Parallel)";
    } else {
        // Very large tensors: Data-parallel across nodes (large)
        result.strategy.node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL;
        result.strategy.on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD;
        result.kernel_name = "NUMA NOOP Kernel (Data-Parallel-Large)";
    }
    
    NUMA_LOG_DEBUG("NOOP Query: %zu elements → %s, efficiency=%.2f",
                   total_elements, result.kernel_name, result.efficiency_score);
    
    return result;
}

/**
 * @brief Calculate work buffer size for NOOP operations
 * 
 * NOOP operations require no work buffer since they perform no computation.
 * 
 * @param tensor Target tensor (unused for NOOP)
 * @return 0 (no work buffer required)
 */
size_t ggml_numa_kernel_noop_calculate_work_buffer_size(const struct ggml_tensor * tensor) {
    GGML_UNUSED(tensor);
    
    // NOOP requires no work buffer
    return 0;
}

/**
 * @brief Register NUMA NOOP kernels with the kernel cache system
 * 
 * Registers the NOOP kernel for GGML_OP_NUMA_NOOP operations using the direct
 * array cache system. Sets up custom strategy thresholds for performance testing
 * different execution modes.
 * 
 * Registration Details:
 * - Operation: GGML_OP_NUMA_NOOP
 * - Thresholds: 256 (single→multi), 512 (multi→parallel), 1024 (parallel+)
 * - Strategy Testing:
 *   - ≤256 elements: Single node / Single thread
 *   - 257-512 elements: Single node / Multi thread  
 *   - 513-1024 elements: Multi node / Data parallel
 *   - >1024 elements: Multi node / Data parallel (large)
 * - Work functions: Same function for all strategies (NOOP is strategy-agnostic)
 * - Aggregation: None required (NOOP produces no intermediate results)
 */
void ggml_numa_register_noop_kernels(void) {
    ggml_numa_kernel_registration_info_t info = {
        .op_type = GGML_OP_NUMA_NOOP,
        .strategy_array = {
            .thresholds = {256, 512, 1024},  // Custom thresholds for testing execution strategies
            .valid = true
        },
        .work_funcs = {
            .single_single_fn = ggml_numa_kernel_noop_execute,
            .single_multi_fn = ggml_numa_kernel_noop_execute,
            .data_parallel_fn = ggml_numa_kernel_noop_execute,
            .valid = true
        },
        .agg_funcs = {
            .valid = false  // NOOP requires no aggregation
        },
        .kernel_name = "NUMA NOOP Kernel",
        .supported = true,
        .is_noop = true  // NOOP kernel - skip coordinator dispatch overhead
    };
    
    // Register with direct array cache system
    ggml_numa_register_kernel_strategy(info.op_type, &info.strategy_array, 
                                       &info.work_funcs, &info.agg_funcs, ggml_numa_kernel_noop_query, info.supported, info.is_noop);
    
    NUMA_LOG_DEBUG("✅ Registered NUMA NOOP Kernel (for performance testing)");
}
