/**
 * @file ggml-numa-openmp-coordinator.h
 * @brief OpenMP-based NUMA coordinator interface for clean threading architecture
 * 
 * This replaces the ggml_threadpool-based coordinator with a clean OpenMP implementation
 * that properly supports kernels expecting multi-thread execution patterns.
 * 
 * @author David Sanftenberg
 */

#ifndef GGML_NUMA_OPENMP_COORDINATOR_H
#define GGML_NUMA_OPENMP_COORDINATOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdbool.h>

// Forward declarations
struct ggml_tensor;
struct ggml_compute_params;

/**
 * @brief CPU mask for thread affinity control
 * 
 * Simple wrapper around system CPU set for cross-platform compatibility.
 */
typedef struct {
    void * cpu_set_ptr;             ///< Pointer to platform-specific CPU set (cpu_set_t on Linux)
    int max_cpus;                   ///< Maximum number of CPUs supported
    bool valid;                     ///< Whether the CPU mask is valid
} ggml_numa_cpu_mask_t;

/**
 * @brief OpenMP-based NUMA coordinator execution strategies
 * 
 * Clean three-strategy execution model using OpenMP parallel regions
 * instead of ggml_threadpool dispatch threads.
 */
typedef enum {
    NUMA_OPENMP_STRATEGY_SINGLE_THREAD = 0,    ///< Single thread, single NUMA node
    NUMA_OPENMP_STRATEGY_SINGLE_NODE,          ///< Multi-thread, single NUMA node  
    NUMA_OPENMP_STRATEGY_DATA_PARALLEL,       ///< Multi-thread, multi-NUMA node
    NUMA_OPENMP_STRATEGY_COUNT
} ggml_numa_openmp_strategy_t;

/**
 * @brief OpenMP coordinator configuration
 * 
 * Simplified configuration using OpenMP thread binding instead of
 * complex ggml_threadpool management.
 */
typedef struct {
    int total_numa_nodes;           ///< Total NUMA nodes available
    int threads_per_node;           ///< Threads to use per NUMA node
    bool numa_available;            ///< Whether NUMA is available on system
    bool initialized;               ///< Whether coordinator is initialized
} ggml_numa_openmp_config_t;

/**
 * @brief Work function signature for OpenMP execution
 * 
 * Clean interface that matches kernel expectations:
 * - Called once per thread with unique ith values
 * - OpenMP handles thread creation and synchronization
 * - No dispatch thread vs worker thread conflicts
 */
typedef enum ggml_status (*ggml_numa_openmp_work_fn_t)(
    void * work_context,
    struct ggml_compute_params * params
);

/**
 * @brief Initialize OpenMP-based NUMA coordinator
 * 
 * Sets up OpenMP thread binding and NUMA node configuration.
 * Much simpler than ggml_threadpool initialization.
 * 
 * @return True if initialization successful, false otherwise
 */
bool ggml_numa_openmp_coordinator_init(void);

/**
 * @brief Initialize OpenMP coordinator with CPU mask
 * 
 * Allows user-specified CPU selection for NUMA-aware execution.
 * 
 * @param cpu_mask Bitmask of CPUs to use (NULL for auto-detection)
 * @param total_threads Total number of threads to use (0 for auto-detection)
 * @return True if initialization successful, false otherwise
 */
bool ggml_numa_openmp_coordinator_init_with_mask(const ggml_numa_cpu_mask_t * cpu_mask, int total_threads);

/**
 * @brief Get OpenMP coordinator configuration
 * 
 * @return Current coordinator configuration
 */
ggml_numa_openmp_config_t ggml_numa_openmp_coordinator_get_config(void);

/**
 * @brief Get number of NUMA nodes from coordinator
 * 
 * @return Number of NUMA nodes available
 */
int ggml_numa_openmp_coordinator_get_num_nodes(void);

/**
 * @brief Execute work using single-thread strategy
 * 
 * Runs work function once on target NUMA node using OpenMP.
 * Clean single-thread execution without threading overhead.
 * 
 * @param tensor Target tensor for execution
 * @param work_fn Work function to execute
 * @param target_numa_node NUMA node to run on (0-based)
 * @return GGML_STATUS_SUCCESS on success, error otherwise
 */
enum ggml_status ggml_numa_openmp_execute_single_thread(
    struct ggml_tensor * tensor,
    ggml_numa_openmp_work_fn_t work_fn,
    int target_numa_node,
    size_t work_buffer_size
);

/**
 * @brief Execute work using single-node multi-thread strategy
 * 
 * Uses OpenMP parallel region on single NUMA node.
 * All threads share memory locality, work function called once per thread.
 * 
 * @param tensor Target tensor for execution
 * @param work_fn Work function to execute  
 * @param target_numa_node NUMA node to run on (0-based)
 * @param n_threads Number of threads to use
 * @return GGML_STATUS_SUCCESS on success, error otherwise
 */
enum ggml_status ggml_numa_openmp_execute_single_node(
    struct ggml_tensor * tensor,
    ggml_numa_openmp_work_fn_t work_fn,
    int target_numa_node,
    int n_threads,
    size_t work_buffer_size
);

/**
 * @brief Execute work using data-parallel multi-node strategy
 * 
 * Uses nested OpenMP parallel regions across NUMA nodes.
 * Maximum parallelism with proper NUMA data slicing.
 * 
 * @param tensor Target tensor for execution
 * @param work_fn Work function to execute
 * @return GGML_STATUS_SUCCESS on success, error otherwise
 */
enum ggml_status ggml_numa_openmp_execute_data_parallel(
    struct ggml_tensor * tensor,
    ggml_numa_openmp_work_fn_t work_fn,
    size_t work_buffer_size
);

/**
 * @brief Clean shutdown of OpenMP coordinator
 * 
 * Much simpler than ggml_threadpool cleanup - OpenMP handles thread lifecycle.
 */
void ggml_numa_openmp_coordinator_shutdown(void);

/**
 * @brief Create CPU mask from CPU indices
 * 
 * @param cpu_indices Array of CPU indices to include in mask
 * @param num_cpus Number of CPU indices in array
 * @return CPU mask pointer or NULL on error
 */
ggml_numa_cpu_mask_t * ggml_numa_create_cpu_mask(const int * cpu_indices, int num_cpus);

/**
 * @brief Free CPU mask
 * 
 * @param mask CPU mask to free
 */
void ggml_numa_free_cpu_mask(ggml_numa_cpu_mask_t * mask);

/**
 * @brief Get fallback threadpool for legacy ggml operations
 * 
 * Some ggml operations (like MUL_MAT) expect a valid threadpool for barrier synchronization.
 * This provides a minimal threadpool structure that can handle those operations.
 * 
 * @return Pointer to fallback threadpool or NULL if not available
 */
struct ggml_threadpool * ggml_numa_openmp_get_fallback_threadpool(void);

#ifdef __cplusplus
}
#endif

#endif // GGML_NUMA_OPENMP_COORDINATOR_H
