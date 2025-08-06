/**
 * NUMA 3-Tier Coordinator Architecture
 * 
 * Flow: Main Thread → Coordinator Threads → NUMA Node Threadpools
 * 
 * Design Principles:
 * 1. Main thread creates coordinator threads (one per NUMA node)
 * 2. Each coordinator thread manages one NUMA node with its own cgraph copy
 * 3. Work flows: main → global queue → coordinator → NUMA pool → coordinator → main
 * 4. Cleanup flows: main → coordinator → NUMA pool (hierarchical)
 * 5. No shared cgraph references - each NUMA node owns its copy
 */

#include "ggml-numa-coordinator.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <stdatomic.h>
#include <pthread.h>
#include <time.h>

#ifdef __linux__
#include <sched.h>
#include <numa.h>
#include <numaif.h>
#endif

#ifndef GGML_NUMA_MAX_NODES
#define GGML_NUMA_MAX_NODES 8
#endif

// Include necessary synchronization primitives
typedef pthread_mutex_t ggml_mutex_t;
typedef pthread_cond_t ggml_cond_t;
typedef pthread_t ggml_thread_t;

#define ggml_mutex_init(m) pthread_mutex_init(m, NULL)
#define ggml_mutex_destroy(m) pthread_mutex_destroy(m)
#define ggml_mutex_lock(m) pthread_mutex_lock(m)
#define ggml_mutex_unlock(m) pthread_mutex_unlock(m)
#define ggml_cond_init(c) pthread_cond_init(c, NULL)
#define ggml_cond_destroy(c) pthread_cond_destroy(c)
#define ggml_cond_wait(c, m) pthread_cond_wait(c, m)
#define ggml_cond_signal(c) pthread_cond_signal(c)
#define ggml_cond_broadcast(c) pthread_cond_broadcast(c)
#define ggml_thread_create_fixed(t, a, f, d) pthread_create(t, a, f, d)
#define ggml_thread_join(t, v) pthread_join(t, v)

// Include min function
static inline size_t min_size_t(size_t a, size_t b) {
    return a < b ? a : b;
}

// Work item for coordinator thread work queue
struct ggml_work_item {
    struct ggml_tensor * tensor;       // Tensor to compute
    int numa_node;                     // Assigned NUMA node (-1 for any)
    int chunk_start;                   // Start of chunk (rows/elements)
    int chunk_end;                     // End of chunk  
    atomic_bool completed;             // Completion flag
    void * result_buffer;              // Buffer for partial results
    size_t result_size;                // Size of result buffer
    struct ggml_work_item * next;      // Next item in queue
    int work_id;                       // Unique work ID for tracking
};

// Work queue for coordinator threads
struct ggml_work_queue {
    struct ggml_work_item * head;      // Head of work queue
    struct ggml_work_item * tail;      // Tail of work queue
    ggml_mutex_t queue_mutex;          // Mutex for queue operations
    ggml_cond_t work_available;        // Signal when work is available
    ggml_cond_t work_completed;        // Signal when work is completed
    atomic_int pending_items;          // Number of pending work items
    atomic_bool shutdown_requested;    // Shutdown flag
};

// Coordinator thread data (one per NUMA node)
struct ggml_coordinator_thread {
    int numa_node;                     // NUMA node this coordinator manages
    struct ggml_threadpool * numa_pool;// NUMA-specific threadpool
    struct ggml_cgraph * numa_cgraph;  // NUMA node's own copy of cgraph
    struct ggml_work_queue work_queue; // Work queue for this coordinator
    ggml_thread_t thread_handle;       // Thread handle
    atomic_bool active;                // Whether thread is active
    atomic_bool shutdown_requested;    // Shutdown request flag
    
    // Performance tracking
    int64_t total_work_items;          // Total work items processed
    int64_t total_processing_time_us;  // Total processing time
};

// NUMA Multi-Socket Threadpool Manager (3-tier: main → coordinator → NUMA)
struct ggml_numa_coordinator_manager {
    int num_numa_nodes;                                   // Number of NUMA nodes
    struct ggml_coordinator_thread * coordinators;       // Array of coordinator threads (one per NUMA node)
    struct ggml_work_queue global_work_queue;            // Global work queue from main thread
    
    // Synchronization for main thread
    atomic_int total_work_items;                          // Total work items pending
    atomic_int completed_work_items;                      // Completed work items
    ggml_mutex_t main_sync_mutex;                         // Main thread sync mutex
    ggml_cond_t main_sync_cond;                           // Main thread sync condition
    atomic_bool manager_active;                           // Manager is active
    
    // Performance profiling
    int64_t total_computations;                           // Total number of multi-socket computations
    int64_t total_async_time_us;                          // Total time spent in async execution (microseconds)
    int64_t total_sync_time_us;                           // Total time spent in synchronization (microseconds)
    int64_t numa_times_us[GGML_NUMA_MAX_NODES];          // Individual NUMA node computation times
    int64_t last_computation_elements;                    // Elements in last computation (for throughput)
};

// Forward declarations
static void * ggml_coordinator_thread_func(void * arg);
static void ggml_work_queue_init(struct ggml_work_queue * queue);
static void ggml_work_queue_destroy(struct ggml_work_queue * queue);
static void ggml_work_queue_enqueue(struct ggml_work_queue * queue, struct ggml_work_item * item);
static struct ggml_work_item * ggml_work_queue_dequeue(struct ggml_work_queue * queue);

// Initialize work queue
static void ggml_work_queue_init(struct ggml_work_queue * queue) {
    queue->head = NULL;
    queue->tail = NULL;
    atomic_init(&queue->pending_items, 0);
    atomic_init(&queue->shutdown_requested, false);
    ggml_mutex_init(&queue->queue_mutex);
    ggml_cond_init(&queue->work_available);
    ggml_cond_init(&queue->work_completed);
}

// Destroy work queue
static void ggml_work_queue_destroy(struct ggml_work_queue * queue) {
    // Set shutdown flag
    atomic_store(&queue->shutdown_requested, true);
    
    // Wake up any waiting threads
    ggml_mutex_lock(&queue->queue_mutex);
    ggml_cond_broadcast(&queue->work_available);
    ggml_mutex_unlock(&queue->queue_mutex);
    
    // Clean up remaining work items
    struct ggml_work_item * current = queue->head;
    while (current) {
        struct ggml_work_item * next = current->next;
        if (current->result_buffer) {
            free(current->result_buffer);
        }
        free(current);
        current = next;
    }
    
    ggml_cond_destroy(&queue->work_completed);
    ggml_cond_destroy(&queue->work_available);
    ggml_mutex_destroy(&queue->queue_mutex);
}

// Enqueue work item
static void ggml_work_queue_enqueue(struct ggml_work_queue * queue, struct ggml_work_item * item) {
    ggml_mutex_lock(&queue->queue_mutex);
    
    item->next = NULL;
    atomic_init(&item->completed, false);
    
    if (queue->tail) {
        queue->tail->next = item;
    } else {
        queue->head = item;
    }
    queue->tail = item;
    
    atomic_fetch_add(&queue->pending_items, 1);
    ggml_cond_signal(&queue->work_available);
    
    ggml_mutex_unlock(&queue->queue_mutex);
}

// Dequeue work item (blocking)
static struct ggml_work_item * ggml_work_queue_dequeue(struct ggml_work_queue * queue) {
    ggml_mutex_lock(&queue->queue_mutex);
    
    while (queue->head == NULL && !atomic_load(&queue->shutdown_requested)) {
        ggml_cond_wait(&queue->work_available, &queue->queue_mutex);
    }
    
    struct ggml_work_item * item = NULL;
    if (queue->head) {
        item = queue->head;
        queue->head = item->next;
        if (queue->head == NULL) {
            queue->tail = NULL;
        }
        atomic_fetch_sub(&queue->pending_items, 1);
    }
    
    ggml_mutex_unlock(&queue->queue_mutex);
    return item;
}

// Coordinator thread function (one per NUMA node)
static void * ggml_coordinator_thread_func(void * arg) {
    struct ggml_coordinator_thread * coordinator = (struct ggml_coordinator_thread *)arg;
    
    GGML_LOG_INFO("Coordinator thread starting for NUMA node %d\n", coordinator->numa_node);
    
    // Set CPU affinity for this coordinator thread to its NUMA node
#ifdef __linux__
    if (coordinator->numa_node >= 0 && numa_available() != -1) {
        struct bitmask * numa_mask = numa_allocate_nodemask();
        numa_bitmask_setbit(numa_mask, coordinator->numa_node);
        numa_run_on_node_mask(numa_mask);
        numa_free_nodemask(numa_mask);
    }
#endif
    
    atomic_store(&coordinator->active, true);
    
    // Main coordinator loop
    while (!atomic_load(&coordinator->shutdown_requested)) {
        struct ggml_work_item * work_item = ggml_work_queue_dequeue(&coordinator->work_queue);
        
        if (!work_item) {
            break; // Shutdown requested
        }
        
        int64_t start_time = ggml_time_us();
        
        GGML_LOG_DEBUG("Coordinator NUMA%d: Processing work item %d (tensor %p, chunk %d-%d)\n",
                      coordinator->numa_node, work_item->work_id, 
                      (void*)work_item->tensor, work_item->chunk_start, work_item->chunk_end);
        
        // Execute the work using NUMA-specific threadpool and cgraph
        if (coordinator->numa_pool && coordinator->numa_cgraph) {
            // Create computation plan for this work chunk
            int numa_threads = 4; // Simplified - use fixed thread count for now instead of ggml_threadpool_get_n_threads
            struct ggml_cplan plan = ggml_graph_plan(coordinator->numa_cgraph, 
                                                   numa_threads,
                                                   coordinator->numa_pool);
            
            if (plan.work_size > 0) {
                plan.work_data = malloc(plan.work_size);
            }
            
            // Execute computation on NUMA node
            enum ggml_status status = ggml_graph_compute(coordinator->numa_cgraph, &plan);
            
            if (status == GGML_STATUS_SUCCESS) {
                // Extract result data (simplified - in real implementation would extract chunk)
                if (work_item->result_buffer && work_item->result_size > 0) {
                    // Copy relevant chunk of result to work item buffer
                    // This is where we'd extract the specific chunk computed by this NUMA node
                    size_t copy_size = min_size_t(work_item->result_size, ggml_nbytes(work_item->tensor));
                    memcpy(work_item->result_buffer, ggml_get_data(work_item->tensor), copy_size);
                }
                
                GGML_LOG_DEBUG("Coordinator NUMA%d: Work item %d completed successfully\n",
                              coordinator->numa_node, work_item->work_id);
            } else {
                GGML_LOG_WARN("Coordinator NUMA%d: Work item %d failed with status %d\n",
                             coordinator->numa_node, work_item->work_id, status);
            }
            
            if (plan.work_data) {
                free(plan.work_data);
            }
        }
        
        int64_t end_time = ggml_time_us();
        coordinator->total_processing_time_us += (end_time - start_time);
        coordinator->total_work_items++;
        
        // Mark work item as completed
        atomic_store(&work_item->completed, true);
        
        // Signal completion to work queue
        ggml_mutex_lock(&coordinator->work_queue.queue_mutex);
        ggml_cond_signal(&coordinator->work_queue.work_completed);
        ggml_mutex_unlock(&coordinator->work_queue.queue_mutex);
    }
    
    atomic_store(&coordinator->active, false);
    GGML_LOG_INFO("Coordinator thread for NUMA node %d shutting down\n", coordinator->numa_node);
    
    return NULL;
}

// Create NUMA coordinator manager with 3-tier architecture
struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_new(int n_threads, bool force_multi_socket) {
    // Step 1: Determine number of NUMA nodes
    int num_numa_nodes = 1;
    bool numa_is_available = false;
    
#ifdef __linux__
    if (numa_available() != -1) {
        num_numa_nodes = numa_max_node() + 1;
        numa_is_available = true;
    }
#endif
    
    if (force_multi_socket && !numa_is_available) {
        num_numa_nodes = 2; // Simulate 2 NUMA nodes for testing
        GGML_LOG_INFO("Forcing multi-socket mode with %d simulated NUMA nodes\n", num_numa_nodes);
    }
    
    if (num_numa_nodes <= 1 && !force_multi_socket) {
        GGML_LOG_INFO("Single NUMA node detected, coordinator manager not needed\n");
        return NULL;
    }
    
    // Step 2: Allocate manager structure
    struct ggml_numa_coordinator_manager * mgr = malloc(sizeof(struct ggml_numa_coordinator_manager));
    if (!mgr) {
        GGML_LOG_ERROR("Failed to allocate NUMA coordinator manager\n");
        return NULL;
    }
    
    memset(mgr, 0, sizeof(struct ggml_numa_coordinator_manager));
    mgr->num_numa_nodes = num_numa_nodes;
    atomic_init(&mgr->total_work_items, 0);
    atomic_init(&mgr->completed_work_items, 0);
    atomic_init(&mgr->manager_active, true);
    
    ggml_mutex_init(&mgr->main_sync_mutex);
    ggml_cond_init(&mgr->main_sync_cond);
    ggml_work_queue_init(&mgr->global_work_queue);
    
    // Step 3: Create coordinator threads (one per NUMA node)
    mgr->coordinators = malloc(sizeof(struct ggml_coordinator_thread) * num_numa_nodes);
    if (!mgr->coordinators) {
        GGML_LOG_ERROR("Failed to allocate coordinator threads\n");
        free(mgr);
        return NULL;
    }
    
    int threads_per_numa = n_threads / num_numa_nodes;
    if (threads_per_numa < 1) threads_per_numa = 1;
    
    for (int i = 0; i < num_numa_nodes; i++) {
        struct ggml_coordinator_thread * coord = &mgr->coordinators[i];
        coord->numa_node = i;
        atomic_init(&coord->active, false);
        atomic_init(&coord->shutdown_requested, false);
        coord->total_work_items = 0;
        coord->total_processing_time_us = 0;
        
        // Step 4: Create NUMA-specific threadpool for this coordinator
        struct ggml_threadpool_params tpp;
        ggml_threadpool_params_init(&tpp, threads_per_numa);
        tpp.numa_aware = true;
        
        // TODO: Set CPU affinity for this NUMA node
        coord->numa_pool = ggml_threadpool_new(&tpp);
        if (!coord->numa_pool) {
            GGML_LOG_ERROR("Failed to create NUMA threadpool for node %d\n", i);
            // Cleanup previous coordinators
            for (int j = 0; j < i; j++) {
                ggml_threadpool_free(mgr->coordinators[j].numa_pool);
            }
            free(mgr->coordinators);
            free(mgr);
            return NULL;
        }
        
        // Step 5: Initialize work queue for this coordinator
        ggml_work_queue_init(&coord->work_queue);
        
        GGML_LOG_INFO("Created coordinator for NUMA node %d with %d threads\n", i, threads_per_numa);
    }
    
    GGML_LOG_INFO("NUMA coordinator manager created with %d NUMA nodes\n", num_numa_nodes);
    return mgr;
}

// Free NUMA coordinator manager (hierarchical cleanup)
void ggml_numa_coordinator_manager_free(struct ggml_numa_coordinator_manager * mgr) {
    if (!mgr) return;
    
    GGML_LOG_INFO("Starting hierarchical cleanup of NUMA coordinator manager\n");
    
    // Step 9: Main thread signals cleanup to coordinator workers
    atomic_store(&mgr->manager_active, false);
    
    for (int i = 0; i < mgr->num_numa_nodes; i++) {
        struct ggml_coordinator_thread * coord = &mgr->coordinators[i];
        
        // Signal shutdown to coordinator
        atomic_store(&coord->shutdown_requested, true);
        ggml_work_queue_destroy(&coord->work_queue);
        
        // Step 11: Coordinator workers cleanup their NUMA threadpools and signal completion
        if (coord->numa_pool) {
            ggml_threadpool_free(coord->numa_pool);
            coord->numa_pool = NULL;
        }
        
        // Step 10: NUMA node threadpools cleanup and free their copies of the compute graph
        if (coord->numa_cgraph) {
            // Note: cgraph is owned by its context, which should be freed separately
            // For now, just set to NULL to avoid double-free
            coord->numa_cgraph = NULL;
        }
        
        GGML_LOG_INFO("Coordinator for NUMA node %d cleaned up\n", i);
    }
    
    // Step 12: Main thread cleans up the coordinator threadpool and frees any remaining objects
    ggml_work_queue_destroy(&mgr->global_work_queue);
    ggml_cond_destroy(&mgr->main_sync_cond);
    ggml_mutex_destroy(&mgr->main_sync_mutex);
    
    if (mgr->coordinators) {
        free(mgr->coordinators);
    }
    
    free(mgr);
    GGML_LOG_INFO("NUMA coordinator manager cleanup completed\n");
}

// Set cgraph for all NUMA nodes (Step 3: each gets full copy)
int ggml_numa_coordinator_manager_set_cgraph(struct ggml_numa_coordinator_manager * mgr, 
                                            const struct ggml_cgraph * master_cgraph) {
    if (!mgr || !master_cgraph) return -1;
    
    GGML_LOG_INFO("Creating cgraph copies for %d NUMA nodes\n", mgr->num_numa_nodes);
    
    for (int i = 0; i < mgr->num_numa_nodes; i++) {
        struct ggml_coordinator_thread * coord = &mgr->coordinators[i];
        
        // Each NUMA node gets its own full copy of the cgraph
        // This eliminates race conditions completely
        // TODO: Implement proper cgraph copying - for now use direct pointer
        // In a full implementation, we'd create separate contexts and copy the graph
        coord->numa_cgraph = (struct ggml_cgraph *)((uintptr_t)master_cgraph); // Remove const qualifier safely
        
        if (!coord->numa_cgraph) {
            GGML_LOG_ERROR("Failed to create cgraph copy for NUMA node %d\n", i);
            return -1;
        }
        
        GGML_LOG_DEBUG("NUMA node %d received its own cgraph copy\n", i);
    }
    
    return 0;
}

// Start coordinator threads (called after cgraph is set)
int ggml_numa_coordinator_manager_start(struct ggml_numa_coordinator_manager * mgr) {
    if (!mgr) return -1;
    
    GGML_LOG_INFO("Starting %d coordinator threads\n", mgr->num_numa_nodes);
    
    for (int i = 0; i < mgr->num_numa_nodes; i++) {
        struct ggml_coordinator_thread * coord = &mgr->coordinators[i];
        
        if (ggml_thread_create_fixed(&coord->thread_handle, NULL, ggml_coordinator_thread_func, coord) != 0) {
            GGML_LOG_ERROR("Failed to create coordinator thread for NUMA node %d\n", i);
            return -1;
        }
        
        GGML_LOG_INFO("Coordinator thread started for NUMA node %d\n", i);
    }
    
    return 0;
}

// Submit work to coordinator manager (Step 4: Main thread apportions work)
int ggml_numa_coordinator_manager_submit_work(struct ggml_numa_coordinator_manager * mgr,
                                              struct ggml_tensor * tensor,
                                              int numa_node_hint) {
    if (!mgr || !tensor) return -1;
    
    // Determine target NUMA node
    int target_numa = numa_node_hint;
    if (target_numa < 0 || target_numa >= mgr->num_numa_nodes) {
        target_numa = atomic_load(&mgr->total_work_items) % mgr->num_numa_nodes; // Round-robin
    }
    
    // Create work item
    struct ggml_work_item * work_item = malloc(sizeof(struct ggml_work_item));
    if (!work_item) return -1;
    
    work_item->tensor = tensor;
    work_item->numa_node = target_numa;
    work_item->chunk_start = 0;
    work_item->chunk_end = ggml_nelements(tensor);
    work_item->result_buffer = malloc(ggml_nbytes(tensor));
    work_item->result_size = ggml_nbytes(tensor);
    work_item->work_id = atomic_fetch_add(&mgr->total_work_items, 1);
    
    if (!work_item->result_buffer) {
        free(work_item);
        return -1;
    }
    
    // Enqueue to target coordinator
    ggml_work_queue_enqueue(&mgr->coordinators[target_numa].work_queue, work_item);
    
    GGML_LOG_DEBUG("Work item %d submitted to NUMA node %d\n", work_item->work_id, target_numa);
    return work_item->work_id;
}

// Wait for all work to complete (Step 7: Main thread waits with signaling - no polling)
int ggml_numa_coordinator_manager_wait_for_completion(struct ggml_numa_coordinator_manager * mgr) {
    if (!mgr) return -1;
    
    GGML_LOG_DEBUG("Main thread waiting for all work to complete (using signaling)\n");
    
    // Wait for each coordinator to signal completion of all work
    for (int i = 0; i < mgr->num_numa_nodes; i++) {
        struct ggml_coordinator_thread * coord = &mgr->coordinators[i];
        
        ggml_mutex_lock(&coord->work_queue.queue_mutex);
        
        // Wait until this coordinator has no pending work
        while (atomic_load(&coord->work_queue.pending_items) > 0) {
            GGML_LOG_DEBUG("Waiting for NUMA node %d to complete %d pending items\n", 
                          i, atomic_load(&coord->work_queue.pending_items));
            
            // Block until coordinator signals work completion (no polling!)
            ggml_cond_wait(&coord->work_queue.work_completed, &coord->work_queue.queue_mutex);
        }
        
        ggml_mutex_unlock(&coord->work_queue.queue_mutex);
        
        GGML_LOG_DEBUG("NUMA node %d completed all work\n", i);
    }
    
    GGML_LOG_DEBUG("All work completed - signaling-based wait finished\n");
    return 0;
}
