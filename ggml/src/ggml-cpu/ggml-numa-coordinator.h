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
 * Wait for all work to complete
 * 
 * @param mgr Manager instance
 * @return 0 on success, -1 on failure
 */
int ggml_numa_coordinator_manager_wait_for_completion(struct ggml_numa_coordinator_manager * mgr);

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
