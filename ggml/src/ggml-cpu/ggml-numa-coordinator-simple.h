/**
 * Simple NUMA Coordinator - Lightweight threadpool management and barriers
 * 
 * Replaces the over-complex coordinator with simple requirements:
 * 1. Manage NUMA threadpool lifetimes (start/stop)
 * 2. Direct dispatch single operations to NUMA nodes  
 * 3. Simple barrier synchronization to wait for completion
 * 4. No work groups, queues, or async integration - just direct execution
 */

#pragma once

#include "ggml.h"
#include "ggml-cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct ggml_threadpool;
struct ggml_compute_params;

// Simple work function signature - takes work context and compute params
typedef enum ggml_status (*ggml_numa_work_function_simple_t)(void * work_context, struct ggml_compute_params * params);

// Simple NUMA coordinator - one threadpool per NUMA node + basic synchronization
struct ggml_numa_coordinator_simple {
    int num_numa_nodes;                              // Number of NUMA nodes
    struct ggml_threadpool ** numa_threadpools;      // Array of NUMA threadpools
    int * numa_thread_counts;                        // Number of threads per NUMA node
    
    // Simple barrier synchronization
    pthread_mutex_t completion_mutex;
    pthread_cond_t completion_cond;
    atomic_int pending_work_items;                   // Number of work items still running
    atomic_bool coordinator_active;                  // Coordinator is active
};

// ============================================================================
// Core API - Simple threadpool management and direct dispatch
// ============================================================================

/**
 * Initialize simple NUMA coordinator with one threadpool per NUMA node
 */
struct ggml_numa_coordinator_simple * ggml_numa_coordinator_simple_init(void);

/**
 * Shutdown coordinator and clean up threadpools
 */
void ggml_numa_coordinator_simple_free(struct ggml_numa_coordinator_simple * coordinator);

/**
 * Get the singleton instance (create if needed)
 */
struct ggml_numa_coordinator_simple * ggml_numa_coordinator_simple_get(void);

/**
 * Direct execution: submit work function to specific NUMA node or all nodes
 * @param coordinator Simple coordinator instance
 * @param work_function Function to execute 
 * @param work_context Context passed to work function
 * @param target_numa_node NUMA node to execute on (-1 = all nodes for data-parallel)
 * @return 0 on success, -1 on error
 */
enum ggml_status ggml_numa_coordinator_simple_execute(
    struct ggml_numa_coordinator_simple * coordinator,
    ggml_numa_work_function_simple_t work_function,
    void * work_context,
    int target_numa_node);

/**
 * Get number of NUMA nodes available
 */
int ggml_numa_coordinator_simple_get_num_nodes(struct ggml_numa_coordinator_simple * coordinator);

#ifdef __cplusplus
}
#endif
