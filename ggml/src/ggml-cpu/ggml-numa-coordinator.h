/**
 * NUMA 3-Tier Coordinator Architecture Header
 * 
 * Flow: Main Thread → Coordinator Threads → NUMA Node Threadpools
 */

#ifndef GGML_NUMA_COORDINATOR_H
#define GGML_NUMA_COORDINATOR_H

#include "ggml.h"
// Note: Minimal includes to avoid path issues from tests
// Full implementation includes are in the .c file

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct ggml_numa_coordinator_manager;
struct ggml_work_item;
struct ggml_work_queue;
struct ggml_coordinator_thread;
struct ggml_compute_params;  // Forward declaration to avoid include path issues

/**
 * Memory management strategy for the NUMA coordinator
 */
enum ggml_numa_memory_strategy {
    GGML_NUMA_STRATEGY_AUTO,        // Adaptive strategy selection based on workload (default)
    GGML_NUMA_STRATEGY_MATRIX_REDUCTION,  // Reduce matrix dimensions to fit memory (better scaling measurement)
    GGML_NUMA_STRATEGY_CHUNKED_PROCESSING, // Process in chunks with full matrices (better throughput)
    GGML_NUMA_STRATEGY_HYBRID       // Dynamic switching based on runtime conditions
};

/**
 * CPU cache hierarchy information
 */
struct ggml_numa_cache_info {
    int64_t l1_cache_size;           // L1 cache size in bytes (per core)
    int64_t l2_cache_size;           // L2 cache size in bytes (per core) 
    int64_t l3_cache_size;           // L3 cache size in bytes (shared)
    int cache_line_size;             // Cache line size in bytes (usually 64)
    int l3_sharing_cores;            // Number of cores sharing L3 cache
    bool cache_detection_successful; // Whether cache detection succeeded
};

/**
 * Workload characteristics for strategy selection (enhanced with cache awareness)
 */
struct ggml_numa_workload_info {
    int64_t matrix_dim;              // Matrix dimension (for MxM operations)
    int batch_size;                  // Batch size being processed
    int64_t available_memory_gb;     // Available system memory in GB
    bool prioritize_scaling_accuracy; // Whether to prioritize batch scaling measurement over throughput
    enum ggml_numa_memory_strategy user_override; // User-specified strategy override
    struct ggml_numa_cache_info cache_info; // CPU cache hierarchy information
};

/**
 * Create NUMA coordinator manager with 3-tier architecture
 * 
 * @param n_threads Total number of threads to distribute across NUMA nodes
 * @param force_multi_socket Force multi-socket mode even without NUMA
 * @return Manager instance or NULL on failure
 */
struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_new(int n_threads, bool force_multi_socket);

/**
 * Create NUMA coordinator manager with threadpool parameters (supports CPU/NUMA masks)
 * 
 * @param tpp Threadpool parameters including CPU masks and NUMA settings
 * @return Manager instance or NULL on failure
 */
struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_new_with_params(const struct ggml_threadpool_params * tpp);

/**
 * Get or create global NUMA coordinator manager (singleton pattern)
 * 
 * @param n_threads Total number of threads to distribute across NUMA nodes
 * @param force_multi_socket Force multi-socket mode even without NUMA
 * @return Manager instance or NULL on failure
 */
struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_get_global(int n_threads, bool force_multi_socket);

/**
 * Get or create global NUMA coordinator manager with parameters (singleton pattern)
 * 
 * @param tpp Threadpool parameters including CPU masks and NUMA settings
 * @return Manager instance or NULL on failure
 */
struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_get_global_with_params(const struct ggml_threadpool_params * tpp);

/**
 * Free NUMA coordinator manager (hierarchical cleanup)
 * 
 * @param mgr Manager to free
 */
void ggml_numa_coordinator_manager_free(struct ggml_numa_coordinator_manager * mgr);

/**
 * Set cgraph for all NUMA nodes (each gets full copy)
 * 
 * @param mgr Manager instance
 * @param master_cgraph Master cgraph to copy to each NUMA node
 * @return 0 on success, -1 on failure
 */
int ggml_numa_coordinator_manager_set_cgraph(struct ggml_numa_coordinator_manager * mgr, 
                                            const struct ggml_cgraph * master_cgraph);

/**
 * Start coordinator threads (called after cgraph is set)
 * 
 * @param mgr Manager instance
 * @return 0 on success, -1 on failure
 */
int ggml_numa_coordinator_manager_start(struct ggml_numa_coordinator_manager * mgr);

/**
 * Submit work to coordinator manager
 * 
 * @param mgr Manager instance
 * @param tensor Tensor to process
 * @param numa_node_hint Preferred NUMA node (-1 for automatic)
 * @return Work ID on success, -1 on failure
 */
int ggml_numa_coordinator_manager_submit_work(struct ggml_numa_coordinator_manager * mgr,
                                              struct ggml_tensor * tensor,
                                              int numa_node_hint);

/**
 * Submit tensor with data parallelism across multiple NUMA nodes
 * This function splits the tensor into chunks and distributes them across available NUMA nodes
 * for parallel processing, then integrates the results.
 * 
 * @param mgr Manager instance
 * @param tensor Tensor to process with data parallelism
 * @return Work group ID on success, -1 on failure
 */
int ggml_numa_coordinator_manager_submit_data_parallel_work(struct ggml_numa_coordinator_manager * mgr,
                                                            struct ggml_tensor * tensor);

/**
 * Submit computation graph with data parallelism
 * This function analyzes the graph and applies data parallelism where beneficial
 * 
 * @param mgr Manager instance
 * @param cgraph Computation graph to process
 * @return 0 on success, -1 on failure
 */
int ggml_numa_coordinator_manager_compute_graph(struct ggml_numa_coordinator_manager * mgr,
                                               struct ggml_cgraph * cgraph);

/**
 * Wait for all work to complete
 * 
 * @param mgr Manager instance
 * @return 0 on success, -1 on failure
 */
int ggml_numa_coordinator_manager_wait_for_completion(struct ggml_numa_coordinator_manager * mgr);

/**
 * Wait for a specific work group to complete (used for data parallel work)
 * 
 * @param mgr Manager instance
 * @param work_group_id Work group ID returned by submit_data_parallel_work
 * @return 0 on success, -1 on failure
 */
int ggml_numa_coordinator_manager_wait_for_work_group(struct ggml_numa_coordinator_manager * mgr, int work_group_id);

/**
 * Progress callback function type
 * 
 * @param work_id ID of completed work item
 * @param numa_node NUMA node that processed the work
 * @param tensor Tensor that was processed
 * @param user_data User-provided data pointer
 */
typedef void (*ggml_numa_progress_callback_t)(int work_id, int numa_node, struct ggml_tensor * tensor, void * user_data);

/**
 * Set progress callback for work completion notifications
 * 
 * @param mgr Manager instance
 * @param callback Callback function (NULL to disable)
 * @param user_data User data pointer passed to callback
 * @return 0 on success, -1 on failure
 */
int ggml_numa_coordinator_manager_set_progress_callback(struct ggml_numa_coordinator_manager * mgr,
                                                        ggml_numa_progress_callback_t callback,
                                                        void * user_data);

/**
 * Get performance statistics
 * 
 * @param mgr Manager instance
 * @param numa_node NUMA node to query (-1 for overall stats)
 * @return Statistics structure
 */
struct ggml_numa_perf_stats {
    int64_t total_work_items;
    int64_t total_processing_time_us;
    int64_t average_processing_time_us;
    double throughput_items_per_sec;
};

struct ggml_numa_perf_stats ggml_numa_coordinator_manager_get_stats(struct ggml_numa_coordinator_manager * mgr, int numa_node);

/**
 * Main NUMA-aware graph computation function
 * This is the primary integration point that replaces standard ggml_graph_compute
 * when NUMA coordination is beneficial
 * 
 * @param cgraph Computation graph to execute
 * @param n_threads Number of threads for computation
 * @return GGML_STATUS_SUCCESS on success, GGML_STATUS_FAILED on failure
 */
enum ggml_status ggml_numa_graph_compute(struct ggml_cgraph * cgraph, int n_threads);

/**
 * NUMA-aware graph computation with virtual NUMA support for testing
 * 
 * @param cgraph Computation graph to execute
 * @param n_threads Number of threads for computation
 * @param force_virtual_numa Force virtual NUMA mode even without hardware NUMA
 * @return GGML_STATUS_SUCCESS on success, GGML_STATUS_FAILED on failure
 */
enum ggml_status ggml_numa_graph_compute_with_virtual(struct ggml_cgraph * cgraph, int n_threads, bool force_virtual_numa);

/**
 * Set memory management strategy for the coordinator
 * 
 * @param mgr Manager instance
 * @param strategy Strategy to use (AUTO for adaptive selection)
 * @return 0 on success, -1 on failure
 */
int ggml_numa_coordinator_manager_set_strategy(struct ggml_numa_coordinator_manager * mgr, enum ggml_numa_memory_strategy strategy);

/**
 * Get current memory management strategy
 * 
 * @param mgr Manager instance
 * @return Current strategy
 */
enum ggml_numa_memory_strategy ggml_numa_coordinator_manager_get_strategy(struct ggml_numa_coordinator_manager * mgr);

/**
 * Get the number of NUMA nodes that the coordinator is managing
 * 
 * @param mgr Manager instance
 * @return Number of NUMA nodes (1 if NUMA not available or mgr is NULL)
 */
int ggml_numa_coordinator_manager_get_numa_nodes(struct ggml_numa_coordinator_manager * mgr);

//
// Coordinator Interface for Dispatcher
// These functions provide controlled access to coordinator resources
// without exposing internal implementation details
//

/**
 * Get NUMA threadpool for a specific NUMA node
 * @param manager Manager instance
 * @param numa_node Target NUMA node (0-based)
 * @return Threadpool pointer or NULL if invalid
 */
struct ggml_threadpool * ggml_numa_coordinator_get_threadpool(struct ggml_numa_coordinator_manager * manager, int numa_node);

/**
 * Get thread count for a specific NUMA node
 * @param manager Manager instance  
 * @param numa_node Target NUMA node (0-based)
 * @return Thread count or -1 if invalid
 */
int ggml_numa_coordinator_get_thread_count(struct ggml_numa_coordinator_manager * manager, int numa_node);

/**
 * Ensure adequate work buffer for a coordinator
 * @param manager Manager instance
 * @param numa_node Target NUMA node (0-based)
 * @param required_size Required buffer size in bytes
 * @return true if buffer is adequate, false on failure
 */
bool ggml_numa_coordinator_ensure_work_buffer(struct ggml_numa_coordinator_manager * manager, int numa_node, size_t required_size);

/**
 * Get work buffer pointer for a coordinator
 * @param manager Manager instance
 * @param numa_node Target NUMA node (0-based)
 * @return Work buffer pointer or NULL if invalid
 */
void * ggml_numa_coordinator_get_work_buffer(struct ggml_numa_coordinator_manager * manager, int numa_node);

/**
 * Get work buffer size for a coordinator
 * @param manager Manager instance
 * @param numa_node Target NUMA node (0-based) 
 * @return Work buffer size or 0 if invalid
 */
size_t ggml_numa_coordinator_get_work_buffer_size(struct ggml_numa_coordinator_manager * manager, int numa_node);

/**
 * Execute operation using graph-based approach with full parallelization
 * This handles complex operations like MUL_MAT that benefit from graph execution
 * @param manager Manager instance
 * @param operation Operation tensor
 * @param numa_node Target NUMA node for execution
 * @return GGML_STATUS_SUCCESS on success, GGML_STATUS_FAILED on failure
 */
enum ggml_status ggml_numa_coordinator_execute_graph_operation(
    struct ggml_numa_coordinator_manager * manager, 
    struct ggml_tensor * operation, 
    int numa_node
);
int ggml_numa_coordinator_manager_get_num_nodes(struct ggml_numa_coordinator_manager * mgr);

/**
 * Choose optimal memory management strategy based on workload characteristics
 * This function implements the adaptive strategy selection logic based on A/B test results
 * 
 * @param workload Workload characteristics
 * @return Recommended strategy
 */
enum ggml_numa_memory_strategy ggml_numa_choose_strategy(const struct ggml_numa_workload_info * workload);

/**
 * Submit tensor with adaptive strategy selection
 * This extends the data parallel work submission with intelligent strategy selection
 * 
 * @param mgr Manager instance
 * @param tensor Tensor to process
 * @param workload Workload characteristics for strategy selection
 * @return Work group ID on success, -1 on failure
 */
int ggml_numa_coordinator_manager_submit_adaptive_work(struct ggml_numa_coordinator_manager * mgr,
                                                       struct ggml_tensor * tensor,
                                                       const struct ggml_numa_workload_info * workload);

/**
 * Detect CPU cache hierarchy information
 * This function reads cache information from the system and populates the cache_info structure
 * 
 * @param cache_info Output structure to populate with cache information
 * @return 0 on success, -1 on failure
 */
int ggml_numa_detect_cache_info(struct ggml_numa_cache_info * cache_info);

/**
 * Calculate optimal matrix tile size based on cache characteristics
 * This function determines the best matrix tile dimensions for cache-friendly operations
 * 
 * @param cache_info Cache hierarchy information
 * @param element_size Size of each matrix element in bytes
 * @param cache_level Target cache level (1 for L1, 2 for L2, 3 for L3)
 * @return Optimal tile dimension (for square tiles)
 */
int64_t ggml_numa_optimal_tile_size(const struct ggml_numa_cache_info * cache_info, 
                                     int element_size, 
                                     int cache_level);

/**
 * Calculate cache-aware chunk size for matrix operations
 * This function determines optimal chunk boundaries based on L3 cache sharing
 * 
 * @param cache_info Cache hierarchy information
 * @param matrix_dim Matrix dimension
 * @param batch_size Batch size
 * @param element_size Size of each element
 * @return Optimal chunk size
 */
int64_t ggml_numa_cache_aware_chunk_size(const struct ggml_numa_cache_info * cache_info,
                                          int64_t matrix_dim,
                                          int batch_size,
                                          int element_size);

/**
 * Get list of active NUMA nodes from coordinator manager
 * 
 * @param mgr Manager instance (NULL for global singleton)
 * @param nodes Output array to fill with active node IDs
 * @param max_nodes Maximum number of nodes to return
 * @return Number of active nodes, or -1 on error
 */
int ggml_numa_coordinator_get_active_nodes(struct ggml_numa_coordinator_manager * mgr, int * nodes, int max_nodes);

#ifdef __cplusplus
}
#endif

#endif // GGML_NUMA_COORDINATOR_H
