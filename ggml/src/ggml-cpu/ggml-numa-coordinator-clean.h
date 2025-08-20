/**
 * NUMA Coordinator - Simplified Header
 * 
 * This provides the interface for the simplified NUMA coordinator that bridges
 * to the NUMA executor system.
 */

#ifndef GGML_NUMA_COORDINATOR_H
#define GGML_NUMA_COORDINATOR_H

#include "ggml.h"
#include "ggml-cpu-impl.h"  // For ggml_compute_params structure

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct ggml_numa_coordinator_manager;
struct ggml_threadpool_params;
struct ggml_cgraph;

/**
 * Memory management strategy for the NUMA coordinator
 */
enum ggml_numa_memory_strategy {
    GGML_NUMA_STRATEGY_AUTO,                   // Adaptive strategy selection (default)
    GGML_NUMA_STRATEGY_MATRIX_REDUCTION,      // Reduce matrix dimensions for better scaling
    GGML_NUMA_STRATEGY_CHUNKED_PROCESSING,    // Process in chunks for better throughput
    GGML_NUMA_STRATEGY_HYBRID                 // Dynamic switching
};

// Bridge functions - core interface
struct ggml_numa_coordinator_manager * ggml_numa_coordinator_get_manager(void);
enum ggml_status ggml_numa_intercept_operation(struct ggml_tensor * tensor, const struct ggml_compute_params * params);
int ggml_numa_coordinator_get_fallback_thread_count(struct ggml_numa_coordinator_manager * mgr);

// Manager lifecycle
struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_new(int n_threads);
struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_new_with_params(const struct ggml_threadpool_params * tpp);
struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_get_global(int n_threads);
struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_get_global_with_params(const struct ggml_threadpool_params * tpp);
struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_get_existing(void);
void ggml_numa_coordinator_manager_free(struct ggml_numa_coordinator_manager * mgr);
void ggml_numa_coordinator_manager_free_global(void);

// Strategy management
int ggml_numa_coordinator_manager_set_strategy(struct ggml_numa_coordinator_manager * mgr, enum ggml_numa_memory_strategy strategy);
enum ggml_numa_memory_strategy ggml_numa_coordinator_manager_get_strategy(struct ggml_numa_coordinator_manager * mgr);

// Threadpool management  
struct ggml_threadpool * ggml_numa_coordinator_get_fallback_threadpool(struct ggml_numa_coordinator_manager * mgr);

// Graph computation (simplified stubs)
int ggml_numa_coordinator_manager_set_cgraph(struct ggml_numa_coordinator_manager * mgr, struct ggml_cgraph * cgraph);
int ggml_numa_coordinator_manager_start(struct ggml_numa_coordinator_manager * mgr);
int ggml_numa_coordinator_manager_compute_graph(struct ggml_numa_coordinator_manager * mgr, struct ggml_cgraph * cgraph);
enum ggml_status ggml_numa_coordinator_manager_wait_for_completion(struct ggml_numa_coordinator_manager * mgr);

// Node management
int ggml_numa_coordinator_manager_get_num_nodes(struct ggml_numa_coordinator_manager * mgr);
int ggml_numa_coordinator_get_num_nodes(void);
void ggml_numa_coordinator_manager_reset_status(struct ggml_numa_coordinator_manager * mgr);

#ifdef __cplusplus
}
#endif

#endif // GGML_NUMA_COORDINATOR_H
