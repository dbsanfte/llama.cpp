/**
 * @file ggml-numa-simple-coordinator.h
 * @brief Simple NUMA Coordination System
 * 
 * This module provides a streamlined NUMA coordination system designed for
 * high-performance tensor operations on multi-socket systems. It focuses on
 * simplicity and efficiency while providing essential NUMA-aware features.
 * 
 * Key Features:
 * - Per-NUMA-node threadpool management with optimal CPU binding
 * - Simple, efficient operation execution without complex async overhead
 * - Data-parallel execution across NUMA nodes with automatic load balancing
 * - Work buffer management and memory allocation optimization
 * - Resource cleanup and lifecycle management
 * 
 * Design Philosophy:
 * - No work groups or complex async coordination
 * - Minimal interface for maximum performance
 * - Focus on thread locality and memory affinity
 * - Direct execution model with predictable performance
 * 
 * @author David Sanftenberg
 * @date 2025
 */

#pragma once

#include "ggml.h"
#include "numa-kernels/numa-kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize simple NUMA coordinator with per-node threadpools
 * 
 * Sets up the NUMA coordination infrastructure with dedicated threadpools
 * for each NUMA node. Each threadpool is optimized for CPU binding and
 * memory affinity to maximize performance on multi-socket systems.
 * 
 * Initialization includes:
 * - Detection of NUMA topology and available nodes
 * - Creation of per-node threadpools with optimal thread counts
 * - CPU binding and memory affinity configuration
 * - Work buffer allocation and management setup
 * - Resource tracking and cleanup preparation
 * 
 * @param tpp Threadpool parameters for optimal CPU binding and thread configuration
 * @return true on successful initialization, false on failure
 */
bool ggml_numa_simple_coordinator_init(struct ggml_threadpool_params * tpp);

/**
 * @brief Cleanup simple NUMA coordinator and free threadpools
 * 
 * Performs comprehensive cleanup of all NUMA coordination resources:
 * - Shuts down all per-node threadpools gracefully
 * - Releases work buffers and memory allocations
 * - Frees NUMA topology information
 * - Resets internal state for clean restart capability
 */
void ggml_numa_simple_coordinator_cleanup(void);

/**
 * @brief Execute work function on single NUMA node
 * 
 * Executes a NUMA kernel function on a specific target node using that
 * node's dedicated threadpool. Provides optimal performance for operations
 * that benefit from node-local execution.
 * 
 * Execution features:
 * - Thread binding to target NUMA node
 * - Memory allocation on target node
 * - Work buffer management and reuse
 * - Performance monitoring and statistics
 * 
 * @param work_function The NUMA kernel function to execute
 * @param work_context Context (typically tensor) for the work
 * @param target_node Target NUMA node (0-based index)
 * @param work_size Size of work buffer required (0 if none needed)
 * @return GGML_STATUS_SUCCESS on success, error code on failure
 */
enum ggml_status ggml_numa_simple_coordinator_execute_single_node(
    ggml_numa_work_function_t work_function,
    void * work_context,
    int target_node,
    size_t work_size);

/**
 * @brief Execute work function across all NUMA nodes with data-parallel strategy
 * 
 * Executes a NUMA kernel function across all available NUMA nodes in parallel,
 * automatically distributing work and managing inter-node coordination.
 * Optimal for large operations that benefit from parallel processing.
 * 
 * Data-parallel features:
 * - Automatic work distribution across nodes
 * - Load balancing based on node capabilities
 * - Synchronized execution with barrier synchronization
 * - Result aggregation when required
 * - Memory locality optimization
 * 
 * @param work_function The NUMA kernel function to execute
 * @param work_context Context (typically tensor) for the work
 * @param work_size Size of work buffer required per thread (0 if none needed)
 * @param aggregation_policy Policy for result aggregation across nodes
 * @param aggregation_function Optional function to aggregate results from multiple nodes (NULL if not needed)
 * @param aggregation_user_data Context for aggregation function (NULL if not needed)
 * @return GGML_STATUS_SUCCESS on success, error code on failure
 */
enum ggml_status ggml_numa_simple_coordinator_execute_data_parallel(
    ggml_numa_work_function_t work_function,
    void * work_context,
    size_t work_size,
    ggml_numa_aggregation_policy_t aggregation_policy,
    ggml_numa_aggregation_function_t aggregation_function,
    void * aggregation_user_data);

/**
 * @brief Get number of available NUMA nodes in the system
 * 
 * Returns the total number of NUMA nodes detected and configured
 * during coordinator initialization. Each node has been set up with
 * dedicated threadpools and resource management.
 * 
 * @return Number of NUMA nodes (>= 1), or 0 if coordinator not initialized
 */
int ggml_numa_simple_coordinator_get_num_nodes(void);

/**
 * @brief Check if NUMA coordinator has been successfully initialized
 * 
 * Verifies that the coordinator infrastructure is properly set up
 * and ready for work execution across NUMA nodes.
 * 
 * @return true if coordinator is initialized and ready, false otherwise
 */
bool ggml_numa_simple_coordinator_is_initialized(void);

/**
 * @brief Get current NUMA node for the calling thread
 * 
 * Determines which NUMA node the current thread is bound to or
 * executing on. Useful for NUMA-aware memory allocation and
 * work distribution decisions.
 * 
 * @return Current NUMA node ID (0-based), or -1 if detection fails
 */
int ggml_numa_get_current_node(void);

/**
 * @brief Get the dedicated fallback threadpool
 * 
 * Returns the fallback threadpool that is used when NUMA-specific
 * execution is not available or fails. The fallback threadpool
 * is typically bound to NUMA node 0 and provides compatibility.
 * 
 * @return Fallback threadpool bound to NUMA node 0, or NULL if not initialized
 */
struct ggml_threadpool * ggml_numa_simple_coordinator_get_fallback_threadpool(void);

/**
 * @brief Get the fallback threadpool thread count
 * 
 * Returns the number of threads available in the fallback threadpool.
 * This count is used for work distribution when falling back from
 * NUMA-aware execution.
 * 
 * @return Number of threads in the fallback threadpool, or 1 if not initialized
 */
int ggml_numa_simple_coordinator_get_fallback_thread_count(void);

/**
 * @brief Get or allocate persistent fallback work buffer with auto-growing capability
 * 
 * Provides a persistent work buffer for fallback operations that automatically
 * grows as needed to accommodate larger work sizes. The buffer is reused
 * across multiple operations to minimize allocation overhead.
 * 
 * Buffer management features:
 * - Automatic size growth when needed
 * - Persistent allocation for reuse
 * - Thread-safe access and allocation
 * - Memory alignment for performance
 * 
 * @param needed_size Required buffer size in bytes
 * @return Pointer to fallback work buffer of at least needed_size, or NULL on allocation failure
 */
void * ggml_numa_simple_coordinator_get_fallback_work_buffer(size_t needed_size);

/**
 * @brief Get current fallback work buffer size
 * 
 * Returns the current size of the allocated fallback work buffer.
 * This size may be larger than the last requested size due to
 * buffer growth policies.
 * 
 * @return Current buffer size in bytes, or 0 if no buffer is allocated
 */
size_t ggml_numa_simple_coordinator_get_fallback_work_buffer_size(void);

/**
 * @brief Validate NUMA thread binding with hard failure on mismatch
 * 
 * Performs strict validation that the current thread is bound to the
 * expected NUMA node. This is a debugging and correctness verification
 * function that will abort the program if binding is incorrect.
 * 
 * Validation features:
 * - Hard failure (abort) on binding mismatch
 * - Detailed error messages with thread context
 * - Optional validation (pass -1 to skip)
 * - Integration with NUMA debugging system
 * 
 * @param expected_node Expected NUMA node (-1 to skip validation)
 * @param thread_type Description of thread type for error messages
 * @param thread_id Thread identifier for error messages
 */
void ggml_numa_simple_coordinator_assert_thread_binding(int expected_node, const char* thread_type, int thread_id);

/**
 * @brief Thread-local flag indicating data-parallel execution mode
 * 
 * This thread-local variable controls how NUMA kernels process data:
 * - When true: Kernels should slice data across NUMA nodes for parallel processing
 * - When false: Kernels should process the entire dataset on current node
 * 
 * The flag is automatically set by the coordinator based on strategy selection
 * and allows kernels to adapt their behavior for optimal performance.
 * 
 * Usage in kernels:
 * ```c
 * if (ggml_numa_is_data_parallel_execution) {
 *     // Calculate NUMA slice bounds and process partial data
 *     size_t numa_start = calculate_node_slice_start();
 *     size_t numa_end = calculate_node_slice_end();
 *     process_data_slice(numa_start, numa_end);
 * } else {
 *     // Process entire dataset on current node
 *     process_entire_dataset();
 * }
 * ```
 */
extern __thread bool ggml_numa_is_data_parallel_execution;

/**
 * @brief Thread-local pointer to shared result tensor data for data-parallel operations
 * 
 * When set to a non-NULL value, this pointer indicates that all NUMA nodes
 * should write their results directly to this shared memory location rather
 * than to their local tensor copies. This eliminates the need for result
 * aggregation and provides optimal performance for data-parallel operations.
 * 
 * Memory layout and usage:
 * - Points to the final tensor's data buffer with proper NUMA allocation
 * - Each node writes to its designated slice of the shared memory
 * - Eliminates copy overhead and aggregation latency
 * - Maintains memory locality through NUMA-aware allocation
 * 
 * Kernel usage pattern:
 * ```c
 * float * dst_data;
 * if (ggml_numa_shared_result_tensor_data != NULL) {
 *     // Use shared memory for direct writes (optimal path)
 *     dst_data = (float *)ggml_numa_shared_result_tensor_data;
 * } else {
 *     // Fall back to local tensor data (compatibility path)
 *     dst_data = (float *)tensor_data(tensor);
 * }
 * 
 * // Write results directly to final location
 * process_and_write_results(dst_data + numa_offset, numa_size);
 * ```
 * 
 * @note This optimization is particularly effective for element-wise operations
 *       and large tensors where aggregation overhead would be significant.
 */
extern __thread void * ggml_numa_shared_result_tensor_data;

#ifdef __cplusplus
}
#endif
