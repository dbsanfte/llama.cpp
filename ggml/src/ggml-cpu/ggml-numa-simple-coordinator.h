/**
 * @file ggml-numa-simple-coordinator.h
 * @brief Simple NUMA coordination system header
 * 
 * Minimal interface for NUMA coordination:
 * - Threadpool management
 * - Simple operation execution
 * - No work groups, no async complexity
 */

#pragma once

#include "ggml.h"
#include "numa-kernels/numa-kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize simple NUMA coordinator with per-node threadpools
 * @param tpp Threadpool parameters for optimal CPU binding  
 * @return true on success, false on failure
 */
bool ggml_numa_simple_coordinator_init(struct ggml_threadpool_params * tpp);

/**
 * Cleanup simple NUMA coordinator and free threadpools
 */
void ggml_numa_simple_coordinator_cleanup(void);

/**
 * Execute work function on single NUMA node
 * @param work_function The kernel function to execute
 * @param work_context Context (tensor) for the work
 * @param target_node Target NUMA node (0-based)
 * @param work_size Size of work buffer required (0 if none needed)
 * @return GGML status
 */
enum ggml_status ggml_numa_simple_coordinator_execute_single_node(
    ggml_numa_work_function_t work_function,
    void * work_context,
    int target_node,
    size_t work_size);

/**
 * Execute work function across all NUMA nodes with data-parallel strategy
 * @param work_function The kernel function to execute
 * @param work_context Context (tensor) for the work
 * @param work_size Size of work buffer required per thread (0 if none needed)
 * @return GGML status
 */
enum ggml_status ggml_numa_simple_coordinator_execute_data_parallel(
    ggml_numa_work_function_t work_function,
    void * work_context,
    size_t work_size);

/**
 * Get number of available NUMA nodes
 * @return Number of NUMA nodes, or 0 if not initialized
 */
int ggml_numa_simple_coordinator_get_num_nodes(void);

/**
 * Check if coordinator is initialized
 * @return true if initialized, false otherwise
 */
bool ggml_numa_simple_coordinator_is_initialized(void);

/**
 * NUMA node detection functions (moved from complex coordinator)
 */
void ggml_numa_set_virtual_node(int node);
int ggml_numa_get_current_node(void);

/**
 * Get the dedicated fallback threadpool
 * @return Fallback threadpool bound to NUMA node 0, or NULL if not initialized
 */
struct ggml_threadpool * ggml_numa_simple_coordinator_get_fallback_threadpool(void);

/**
 * Get the fallback thread count
 * @return Number of threads in the fallback threadpool, or 1 if not initialized
 */
int ggml_numa_simple_coordinator_get_fallback_thread_count(void);

/**
 * NUMA thread binding validation (hard failure)
 * @param expected_node Expected NUMA node (-1 to skip validation)
 * @param thread_type Description of thread type for error messages
 * @param thread_id Thread identifier for error messages
 */
void ggml_numa_simple_coordinator_assert_thread_binding(int expected_node, const char* thread_type, int thread_id);

/**
 * Thread-local flag indicating data-parallel execution mode
 * When true, kernels should slice data across NUMA nodes
 * When false, kernels should process entire dataset
 */
extern __thread bool ggml_numa_is_data_parallel_execution;

#ifdef __cplusplus
}
#endif
