/**
 * NUMA Coordinator - Direct Simple Implementation
 * 
 * This coordinator provides direct NUMA kernel dispatch without
 * the complex 3-tier architecture. Main function is to intercept
 * operations and call NUMA kernels directly.
 */

#include "ggml-numa-coordinator.h"
#include "ggml-numa-shared.h"             // Shared NUMA logging and utilities
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"  // For ggml_compute_params structure

#ifdef __linux__
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>
#include <stdatomic.h>
#include <unistd.h>

#ifndef GGML_NUMA_MAX_NODES
#define GGML_NUMA_MAX_NODES 8
#endif

// Bridge function to intercept operations and route directly to NUMA kernels
enum ggml_status ggml_numa_intercept_operation(struct ggml_tensor * tensor, 
                                               const struct ggml_compute_params * params) {
    if (!tensor || !params) {
        return GGML_STATUS_FAILED;
    }
    
    // Direct kernel dispatch based on operation type
    switch (tensor->op) {
        case GGML_OP_ADD:
            {
                // For ADD operations with sufficient size, use direct NUMA dispatch
                if (ggml_nelements(tensor) >= 100000) {
                    // Import the NUMA ADD kernel function
                    extern enum ggml_status ggml_numa_kernel_add_execute(struct ggml_tensor * tensor, struct ggml_cplan * cplan);
                    
                    // Create a simple cplan for the kernel
                    struct ggml_cplan cplan = {
                        .n_threads = params->nth,
                        .work_size = params->wsize,
                        .work_data = params->wdata,
                        .threadpool = params->threadpool
                    };
                    
                    // Direct call to NUMA ADD kernel - this bypasses all the complex coordination
                    enum ggml_status result = ggml_numa_kernel_add_execute(tensor, &cplan);
                    
                    if (result == GGML_STATUS_SUCCESS) {
                        NUMA_COORD_LOG_INFO(-1, "🚀 Direct NUMA ADD kernel completed successfully (bypassed complex coordinator)");
                        return GGML_STATUS_SUCCESS;
                    } else {
                        NUMA_COORD_LOG_DEBUG(-1, "⚠️ Direct NUMA ADD kernel failed, falling back to CPU");
                        return GGML_STATUS_FAILED;
                    }
                }
                break;
            }
        default:
            // Other operations not implemented in direct coordinator yet
            break;
    }
    
    // Operation not supported by direct coordinator, let CPU backend handle it
    return GGML_STATUS_FAILED;
}

#include "ggml-numa-coordinator.h"
#include "ggml-numa-shared.h"             // Shared NUMA logging and utilities
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"  // For ggml_compute_params structure

#ifdef __linux__
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>
#include <stdatomic.h>
#include <unistd.h>

#ifndef GGML_NUMA_MAX_NODES
#define GGML_NUMA_MAX_NODES 8
#endif

// Direct work item for simple dispatch
struct ggml_numa_work_item_direct {
    struct ggml_tensor * tensor;
    struct ggml_compute_params params;
    int numa_node;
    enum ggml_status * result;
    atomic_int * pending_count;
    pthread_mutex_t * completion_mutex;
    pthread_cond_t * completion_cond;
};

// Simple coordinator manager - direct threadpool management
struct ggml_numa_coordinator_manager {
    // Configuration
    int num_numa_nodes;                   // Number of NUMA nodes in the system
    int total_threads;                    // Total threads across all NUMA nodes
    struct ggml_threadpool_params threadpool_params; // Threadpool configuration
    
    // Direct NUMA threadpools
    struct ggml_threadpool ** numa_threadpools; // One threadpool per NUMA node
    int * numa_thread_counts;            // Thread count per NUMA node
    
    // Simple synchronization
    pthread_mutex_t completion_mutex;
    pthread_cond_t completion_cond;
    atomic_int pending_work_items;
    
    // Global state
    atomic_bool manager_active;           // Manager is active
    struct ggml_threadpool * fallback_threadpool; // Fallback non-NUMA threadpool
    
    // Strategy settings (simplified)
    enum ggml_numa_memory_strategy memory_strategy; // Memory management strategy
};

// Global state
static struct ggml_numa_coordinator_manager * g_global_coordinator_manager = NULL;
static pthread_mutex_t g_coordinator_mutex = PTHREAD_MUTEX_INITIALIZER;

// Direct worker function that executes tensor operations on specific NUMA node
static void ggml_numa_direct_worker_func(void * arg) {
    struct ggml_numa_work_item_direct * work_item = (struct ggml_numa_work_item_direct *)arg;
    
#ifdef __linux__
    // Set CPU affinity to target NUMA node
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    
    if (numa_available() >= 0) {
        struct bitmask * numa_node_mask = numa_allocate_cpumask();
        numa_node_to_cpus(work_item->numa_node, numa_node_mask);
        
        for (int cpu = 0; cpu < numa_num_possible_cpus(); cpu++) {
            if (numa_bitmask_isbitset(numa_node_mask, cpu)) {
                CPU_SET(cpu, &cpu_set);
            }
        }
        
        sched_setaffinity(0, sizeof(cpu_set), &cpu_set);
        numa_free_cpumask(numa_node_mask);
    }
#endif
    
    // Execute tensor operation directly using existing CPU backend
    *(work_item->result) = ggml_compute_forward(work_item->tensor, &work_item->params);
    
    // Signal completion using simple barrier
    pthread_mutex_lock(work_item->completion_mutex);
    atomic_fetch_sub(work_item->pending_count, 1);
    pthread_cond_broadcast(work_item->completion_cond);
    pthread_mutex_unlock(work_item->completion_mutex);
}

//
// Core implementation - direct NUMA dispatch
//

// Bridge function to intercept operations and route directly to NUMA kernels
enum ggml_status ggml_numa_intercept_operation(struct ggml_tensor * tensor, 
                                               const struct ggml_compute_params * params) {
    if (!tensor || !params) {
        return GGML_STATUS_FAILED;
    }
    
    struct ggml_numa_coordinator_manager * mgr = ggml_numa_coordinator_get_manager();
    if (!mgr || !atomic_load(&mgr->manager_active)) {
        // Fallback to normal CPU execution
        return GGML_STATUS_FAILED;
    }
    
    // Direct kernel dispatch based on operation type
    switch (tensor->op) {
        case GGML_OP_ADD:
            {
                // For ADD operations with sufficient size, use direct NUMA dispatch
                if (mgr->num_numa_nodes > 1 && ggml_nelements(tensor) >= 100000) {
                    // Import the NUMA ADD kernel function
                    extern enum ggml_status ggml_numa_kernel_add_execute(struct ggml_tensor * tensor, struct ggml_cplan * cplan);
                    
                    // Create a simple cplan for the kernel
                    struct ggml_cplan cplan = {
                        .n_threads = params->nth,
                        .work_size = params->wsize,
                        .work_data = params->wdata,
                        .threadpool = params->threadpool
                    };
                    
                    // Direct call to NUMA ADD kernel - this bypasses all the complex coordination
                    enum ggml_status result = ggml_numa_kernel_add_execute(tensor, &cplan);
                    
                    if (result == GGML_STATUS_SUCCESS) {
                        NUMA_COORD_LOG_INFO(-1, "🚀 Direct NUMA ADD kernel completed successfully (bypassed complex coordinator)");
                        return GGML_STATUS_SUCCESS;
                    } else {
                        NUMA_COORD_LOG_DEBUG(-1, "⚠️ Direct NUMA ADD kernel failed, falling back to CPU");
                        return GGML_STATUS_FAILED;
                    }
                }
                break;
            }
        default:
            // Other operations not implemented in direct coordinator yet
            break;
    }
    
    // Operation not supported by direct coordinator, let CPU backend handle it
    return GGML_STATUS_FAILED;
}

//
// Bridge and manager lifecycle functions
//

// Bridge function to get the coordinator manager (required by CPU backend)
struct ggml_numa_coordinator_manager * ggml_numa_coordinator_get_manager(void) {
    if (g_global_coordinator_manager == NULL) {
        // Create a minimal manager if none exists
        return ggml_numa_coordinator_manager_new(1); // Use 1 thread as default
    }
    return g_global_coordinator_manager;
}

// Bridge function to get fallback thread count
int ggml_numa_coordinator_get_fallback_thread_count(struct ggml_numa_coordinator_manager * mgr) {
    if (!mgr) {
        return 1; // Safe default
    }
    return mgr->total_threads;
}

struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_new(int n_threads) {
    struct ggml_threadpool_params tpp = {
        .n_threads = n_threads,
        .prio = 0,
        .poll = 50,
        .strict_cpu = false,
        .paused = false,
    };
    return ggml_numa_coordinator_manager_new_with_params(&tpp);
}

struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_new_with_params(const struct ggml_threadpool_params * tpp) {
    if (!tpp) {
        return NULL;
    }
    
    struct ggml_numa_coordinator_manager * mgr = malloc(sizeof(struct ggml_numa_coordinator_manager));
    if (!mgr) {
        return NULL;
    }
    
    // Initialize NUMA detection
#ifdef __linux__
    if (numa_available() < 0) {
        mgr->num_numa_nodes = 1;
    } else {
        mgr->num_numa_nodes = numa_num_configured_nodes();
        if (mgr->num_numa_nodes <= 0) {
            mgr->num_numa_nodes = 1;
        }
    }
#else
    mgr->num_numa_nodes = 1;
#endif
    
    // Initialize basic configuration
    mgr->total_threads = tpp->n_threads;
    mgr->threadpool_params = *tpp;
    atomic_store(&mgr->manager_active, true);
    mgr->fallback_threadpool = NULL;
    mgr->memory_strategy = GGML_NUMA_STRATEGY_AUTO;
    
    // Initialize synchronization
    pthread_mutex_init(&mgr->completion_mutex, NULL);
    pthread_cond_init(&mgr->completion_cond, NULL);
    atomic_init(&mgr->pending_work_items, 0);
    
    // Allocate NUMA threadpools
    mgr->numa_threadpools = calloc(mgr->num_numa_nodes, sizeof(struct ggml_threadpool *));
    mgr->numa_thread_counts = calloc(mgr->num_numa_nodes, sizeof(int));
    
    if (!mgr->numa_threadpools || !mgr->numa_thread_counts) {
        pthread_mutex_destroy(&mgr->completion_mutex);
        pthread_cond_destroy(&mgr->completion_cond);
        free(mgr->numa_threadpools);
        free(mgr->numa_thread_counts);
        free(mgr);
        return NULL;
    }
    
    // Create one threadpool per NUMA node
    int threads_per_node = mgr->total_threads / mgr->num_numa_nodes;
    if (threads_per_node < 1) threads_per_node = 1;
    
    for (int i = 0; i < mgr->num_numa_nodes; i++) {
        mgr->numa_thread_counts[i] = threads_per_node;
        
        struct ggml_threadpool_params node_params = *tpp;
        node_params.n_threads = threads_per_node;
        
        mgr->numa_threadpools[i] = ggml_threadpool_new(&node_params);
        if (!mgr->numa_threadpools[i]) {
            // Cleanup on failure
            for (int j = 0; j < i; j++) {
                ggml_threadpool_free(mgr->numa_threadpools[j]);
            }
            pthread_mutex_destroy(&mgr->completion_mutex);
            pthread_cond_destroy(&mgr->completion_cond);
            free(mgr->numa_threadpools);
            free(mgr->numa_thread_counts);
            free(mgr);
            return NULL;
        }
    }
    
    NUMA_COORD_LOG_INFO(-1, "🌟 Direct NUMA coordinator created with %d nodes, %d threads per node", 
                        mgr->num_numa_nodes, threads_per_node);
    
    return mgr;
}

// Global manager access functions
struct ggml_numa_coordinator_manager * ggml_get_global_coordinator_manager(int n_threads) {
    pthread_mutex_lock(&g_coordinator_mutex);
    if (!g_global_coordinator_manager) {
        g_global_coordinator_manager = ggml_numa_coordinator_manager_new(n_threads);
    }
    pthread_mutex_unlock(&g_coordinator_mutex);
    return g_global_coordinator_manager;
}

struct ggml_numa_coordinator_manager * ggml_get_global_coordinator_manager_with_params(const struct ggml_threadpool_params * tpp) {
    pthread_mutex_lock(&g_coordinator_mutex);
    if (!g_global_coordinator_manager) {
        g_global_coordinator_manager = ggml_numa_coordinator_manager_new_with_params(tpp);
    }
    pthread_mutex_unlock(&g_coordinator_mutex);
    return g_global_coordinator_manager;
}

struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_get_global(int n_threads) {
    return ggml_get_global_coordinator_manager(n_threads);
}

struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_get_global_with_params(const struct ggml_threadpool_params * tpp) {
    return ggml_get_global_coordinator_manager_with_params(tpp);
}

struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_get_existing(void) {
    return g_global_coordinator_manager;
}

void ggml_numa_coordinator_manager_free(struct ggml_numa_coordinator_manager * mgr) {
    if (!mgr) {
        return;
    }
    
    atomic_store(&mgr->manager_active, false);
    
    // Free NUMA threadpools
    if (mgr->numa_threadpools) {
        for (int i = 0; i < mgr->num_numa_nodes; i++) {
            if (mgr->numa_threadpools[i]) {
                ggml_threadpool_free(mgr->numa_threadpools[i]);
            }
        }
        free(mgr->numa_threadpools);
    }
    
    if (mgr->numa_thread_counts) {
        free(mgr->numa_thread_counts);
    }
    
    if (mgr->fallback_threadpool) {
        ggml_threadpool_free(mgr->fallback_threadpool);
    }
    
    // Cleanup synchronization
    pthread_mutex_destroy(&mgr->completion_mutex);
    pthread_cond_destroy(&mgr->completion_cond);
    
    free(mgr);
    
    NUMA_COORD_LOG_INFO(-1, "Freed direct NUMA coordinator manager");
}

void ggml_numa_coordinator_manager_free_global(void) {
    pthread_mutex_lock(&g_coordinator_mutex);
    if (g_global_coordinator_manager) {
        ggml_numa_coordinator_manager_free(g_global_coordinator_manager);
        g_global_coordinator_manager = NULL;
    }
    pthread_mutex_unlock(&g_coordinator_mutex);
}

//
// Strategy management (simplified stubs)
//

int ggml_numa_coordinator_manager_set_strategy(struct ggml_numa_coordinator_manager * mgr, enum ggml_numa_memory_strategy strategy) {
    if (!mgr) {
        return -1;
    }
    mgr->memory_strategy = strategy;
    return 0;
}

enum ggml_numa_memory_strategy ggml_numa_coordinator_manager_get_strategy(struct ggml_numa_coordinator_manager * mgr) {
    if (!mgr) {
        return GGML_NUMA_STRATEGY_AUTO;
    }
    return mgr->memory_strategy;
}

//
// Fallback thread management
//

struct ggml_threadpool * ggml_numa_coordinator_get_fallback_threadpool(struct ggml_numa_coordinator_manager * mgr) {
    if (!mgr) {
        return NULL;
    }
    
    if (!mgr->fallback_threadpool) {
        mgr->fallback_threadpool = ggml_threadpool_new(&mgr->threadpool_params);
    }
    
    return mgr->fallback_threadpool;
}

//
// Stub functions for compatibility
//

int ggml_numa_coordinator_manager_set_cgraph(struct ggml_numa_coordinator_manager * mgr, struct ggml_cgraph * cgraph) {
    // Simplified: no-op
    (void)mgr;
    (void)cgraph;
    return 0;
}

int ggml_numa_coordinator_manager_start(struct ggml_numa_coordinator_manager * mgr) {
    // Simplified: no-op
    (void)mgr;
    return 0;
}

int ggml_numa_coordinator_manager_compute_graph(struct ggml_numa_coordinator_manager * mgr, struct ggml_cgraph * cgraph) {
    // Simplified: delegate to normal CPU computation
    (void)mgr;
    (void)cgraph;
    return 0;
}

enum ggml_status ggml_numa_coordinator_manager_wait_for_completion(struct ggml_numa_coordinator_manager * mgr) {
    // Simplified: no-op
    (void)mgr;
    return GGML_STATUS_SUCCESS;
}

int ggml_numa_coordinator_manager_get_num_nodes(struct ggml_numa_coordinator_manager * mgr) {
    if (!mgr) {
        return 1;
    }
    return mgr->num_numa_nodes;
}

int ggml_numa_coordinator_get_num_nodes(void) {
#ifdef __linux__
    if (numa_available() >= 0) {
        return numa_max_node() + 1;
    }
#endif
    return 1; // Non-NUMA system
}

//
// Fallback thread management
//

struct ggml_threadpool * ggml_numa_coordinator_get_fallback_threadpool(struct ggml_numa_coordinator_manager * mgr) {
    if (!mgr) {
        return NULL;
    }
    
    if (!mgr->fallback_threadpool) {
        mgr->fallback_threadpool = ggml_threadpool_new(&mgr->threadpool_params);
    }
    
    return mgr->fallback_threadpool;
}

//
// Strategy management (simplified stubs)
//

int ggml_numa_coordinator_manager_set_strategy(struct ggml_numa_coordinator_manager * mgr, enum ggml_numa_memory_strategy strategy) {
    if (!mgr) {
        return -1;
    }
    mgr->memory_strategy = strategy;
    return 0;
}

enum ggml_numa_memory_strategy ggml_numa_coordinator_manager_get_strategy(struct ggml_numa_coordinator_manager * mgr) {
    if (!mgr) {
        return GGML_NUMA_STRATEGY_AUTO;
    }
    return mgr->memory_strategy;
}

//
// Stub functions for compatibility with CPU backend
//

int ggml_numa_coordinator_manager_set_cgraph(struct ggml_numa_coordinator_manager * mgr, struct ggml_cgraph * cgraph) {
    // Direct approach: no-op for individual operations
    (void)mgr;
    (void)cgraph;
    return 0;
}

int ggml_numa_coordinator_manager_start(struct ggml_numa_coordinator_manager * mgr) {
    // Direct approach: no-op since threadpools are always ready
    (void)mgr;
    return 0;
}

int ggml_numa_coordinator_manager_compute_graph(struct ggml_numa_coordinator_manager * mgr, struct ggml_cgraph * cgraph) {
    // Direct approach: we handle individual operations, not entire graphs
    (void)mgr;
    (void)cgraph;
    return 0;
}

enum ggml_status ggml_numa_coordinator_manager_wait_for_completion(struct ggml_numa_coordinator_manager * mgr) {
    // Direct approach: operations are synchronous
    (void)mgr;
    return GGML_STATUS_SUCCESS;
}

void ggml_numa_coordinator_manager_reset_status(struct ggml_numa_coordinator_manager * mgr) {
    // Direct approach: no persistent status to reset
    (void)mgr;
}
