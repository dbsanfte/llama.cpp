/**
 * NUMA Coordinator - Simplified Bridge to NUMA Executor
 * 
 * This coordinator provides a minimal bridge to the NUMA executor system.
 * The old work-items and dispatcher system has been removed.
 * We only provide the bridge functions needed for compatibility.
 */

#include "ggml-numa-coordinator.h"
#include "ggml-numa-shared.h"             // Shared NUMA logging and utilities
#include "ggml-numa-executor.h"           // New NUMA executor
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"  // For ggml_compute_params structure

#ifdef __linux__
#include <numa.h>
#include <numaif.h>
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>
#include <stdatomic.h>

#ifndef GGML_NUMA_MAX_NODES
#define GGML_NUMA_MAX_NODES 8
#endif

// Simple coordinator manager for the bridge functions
struct ggml_numa_coordinator_manager {
    // Configuration
    int num_numa_nodes;                   // Number of NUMA nodes in the system
    int total_threads;                    // Total threads across all NUMA nodes
    struct ggml_threadpool_params threadpool_params; // Threadpool configuration
    
    // Global state
    atomic_bool manager_active;           // Manager is active
    struct ggml_threadpool * fallback_threadpool; // Fallback non-NUMA threadpool
    
    // Strategy settings (simplified)
    enum ggml_numa_memory_strategy memory_strategy; // Memory management strategy
};

// Global state
static struct ggml_numa_coordinator_manager * g_global_coordinator_manager = NULL;
static pthread_mutex_t g_coordinator_mutex = PTHREAD_MUTEX_INITIALIZER;

//
// Bridge functions - these provide the interface that the CPU backend expects
//

// Bridge function to get the coordinator manager (required by CPU backend)
struct ggml_numa_coordinator_manager * ggml_numa_coordinator_get_manager(void) {
    if (g_global_coordinator_manager == NULL) {
        // Create a minimal manager if none exists
        return ggml_numa_coordinator_manager_new(1); // Use 1 thread as default
    }
    return g_global_coordinator_manager;
}

// Bridge function to intercept operations and route to executor
enum ggml_status ggml_numa_intercept_operation(struct ggml_tensor * tensor, 
                                               const struct ggml_compute_params * params) {
    if (!tensor || !params) {
        return GGML_STATUS_FAILED;
    }
    
    // Route to the NUMA executor
    return ggml_numa_executor_execute_operation(tensor, params->nth);
}

// Bridge function to get fallback thread count
int ggml_numa_coordinator_get_fallback_thread_count(struct ggml_numa_coordinator_manager * mgr) {
    if (!mgr) {
        return 1; // Safe default
    }
    return mgr->total_threads;
}

//
// Manager lifecycle functions
//

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
    
    // Initialize simple configuration
    mgr->num_numa_nodes = 1; // Simplified: assume single node
    mgr->total_threads = tpp->n_threads;
    mgr->threadpool_params = *tpp;
    atomic_store(&mgr->manager_active, true);
    mgr->fallback_threadpool = NULL;
    mgr->memory_strategy = GGML_NUMA_STRATEGY_AUTO;
    
    NUMA_COORD_LOG_INFO(-1, "Created simplified NUMA coordinator manager with %d threads", tpp->n_threads);
    
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
    
    if (mgr->fallback_threadpool) {
        ggml_threadpool_free(mgr->fallback_threadpool);
    }
    
    free(mgr);
    
    NUMA_COORD_LOG_INFO(-1, "Freed NUMA coordinator manager");
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

void ggml_numa_coordinator_manager_reset_status(struct ggml_numa_coordinator_manager * mgr) {
    // Simplified: no-op
    (void)mgr;
}
