/**
 * NUMA 3-Tier Coordinator Architecture Header
 * 
 * Flow: Main Thread → Coordinator Threads → NUMA Node Threadpools
 */

#ifndef GGML_NUMA_COORDINATOR_H
#define GGML_NUMA_COORDINATOR_H

#include "ggml.h"
#include "ggml-impl.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct ggml_numa_coordinator_manager;
struct ggml_work_item;
struct ggml_work_queue;
struct ggml_coordinator_thread;

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

#ifdef __cplusplus
}
#endif

#endif // GGML_NUMA_COORDINATOR_H
