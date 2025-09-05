/**
 * @file noop.h
 * @brief Header for NUMA-aware NOOP kernel implementation
 * 
 * This header defines the interface for the NUMA NOOP kernel, which provides
 * a minimal no-operation implementation for performance testing and benchmarking.
 * The kernel follows the standard NUMA kernel interface patterns while performing
 * minimal computation to isolate NUMA system overhead.
 * 
 * Key features:
 * - Minimal computational overhead for pure NUMA system measurement
 * - Support for all three NUMA execution strategies
 * - Standard kernel interface compliance for consistent integration
 * - Debugging and validation capabilities for NUMA execution flows
 * 
 * @author David Sanftenberg
 * @date 2024
 */

#ifndef GGML_NUMA_KERNEL_NOOP_H
#define GGML_NUMA_KERNEL_NOOP_H

#include "../ggml-numa-shared.h"
#include "numa-kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Unified NOOP kernel execution function
 * 
 * Provides a minimal no-operation implementation that follows the standard
 * NUMA kernel execution pattern. Performs basic validation and returns
 * immediately, making it ideal for measuring pure NUMA system overhead
 * without computational complexity.
 * 
 * Supports all three NUMA execution strategies:
 * - Single-thread/single-node: Minimal overhead measurement
 * - Multi-thread/single-node: Threading overhead measurement  
 * - Data-parallel/multi-node: Full NUMA distribution overhead measurement
 * 
 * @param work_context Pointer to the tensor being processed (cast from ggml_tensor*)
 * @param params Compute parameters containing thread information and work data
 * @return GGML_STATUS_SUCCESS on successful completion
 * 
 * @note This function intentionally performs minimal work to isolate NUMA
 *       system overhead from computational complexity
 */
enum ggml_status ggml_numa_kernel_noop_unified_execute(void * work_context, struct ggml_compute_params * params);

/**
 * @brief Kernel registration function for NUMA NOOP operations
 * 
 * Provides registration information for the NOOP kernel, configuring it
 * for all three NUMA execution strategies with thresholds optimized for
 * performance testing scenarios.
 * 
 * Strategy configuration:
 * - Below 1K elements: Single-thread execution (minimal overhead)
 * - 1K-256K elements: Multi-thread single-node (threading overhead)
 * - Above 256K elements: Data-parallel execution (NUMA distribution overhead)
 * 
 * @return Registration information structure with function pointers and thresholds
 * 
 * @note The NOOP kernel does not require aggregation functions as it performs
 *       no meaningful computation that needs to be combined across threads
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_noop_register(void);

/**
 * @brief Query function for NUMA NOOP kernel strategy selection
 * 
 * Determines the optimal execution strategy for NOOP operations based on
 * tensor size and system configuration. Uses unified strategy selection
 * to ensure consistent behavior across all kernels.
 * 
 * Strategy selection based on element count thresholds makes this ideal
 * for measuring overhead at different computational scales.
 * 
 * @param tensor The tensor to be processed (used for size calculation)
 * @return Query result containing selected strategy and execution parameters
 * 
 * @note Returns minimal operation count (1) to represent pure overhead measurement
 */
ggml_numa_kernel_query_result_t ggml_numa_kernel_noop_query(const struct ggml_tensor * tensor);

/**
 * @brief Work buffer calculation function for NOOP operations
 * 
 * NOOP operations require no work buffers as they perform no meaningful
 * computation. This function returns zero to indicate no additional
 * memory allocation is needed.
 * 
 * @param tensor The tensor being processed (unused for NOOP)
 * @param total_numa_nodes Total number of NUMA nodes (unused for NOOP)  
 * @param total_threads Total number of threads (unused for NOOP)
 * @return Zero indicating no work buffer memory required
 * 
 * @note Maintains standard kernel interface while indicating no memory overhead
 */
size_t ggml_numa_kernel_noop_work_buffer_calc(const struct ggml_tensor * tensor, int total_numa_nodes, int total_threads);

#ifdef __cplusplus
}
#endif

#endif // GGML_NUMA_KERNEL_NOOP_H
