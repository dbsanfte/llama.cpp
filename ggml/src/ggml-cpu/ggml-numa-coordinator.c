/**
 * NUMA 3-Tier Coordinator Architecture
 * 
 * Flow: Main Thread → Coordinator Threads → NUMA Node Threadpools
 * 
 * Design Principles:
 * 1. Main thread creates coordinator threads (one per NUMA node)
 * 2. Each coordinator thread manages one NUMA node with its own threadpool
 * 3. Work flows: main → global queue → coordinator → NUMA pool → coordinator → main
 * 4. Cleanup flows: main → coordinator → NUMA pool (hierarchical)
 */

#include "ggml-numa-coordinator.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"

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
#define ggml_thread_create(t, a, f, d) pthread_create(t, a, f, d)
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
    struct ggml_context * numa_ctx;    // Context for this NUMA node's cgraph copy
    struct ggml_work_queue work_queue; // Work queue for this coordinator
    ggml_thread_t thread_handle;       // Thread handle
    atomic_bool active;                // Whether thread is active
    atomic_bool shutdown_requested;    // Shutdown request flag
    atomic_bool thread_created;        // Whether thread was actually created
    
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
    atomic_bool threads_started;                          // Whether coordinator threads have been started
    
    // Performance profiling
    int64_t total_computations;                           // Total number of multi-socket computations
    int64_t total_async_time_us;                          // Total time spent in async execution (microseconds)
    int64_t total_sync_time_us;                           // Total time spent in synchronization (microseconds)
    int64_t numa_times_us[GGML_NUMA_MAX_NODES];          // Individual NUMA node computation times
    int64_t last_computation_elements;                    // Elements in last computation (for throughput)
};

// Global singleton coordinator manager - persists for program lifetime
static struct ggml_numa_coordinator_manager * g_global_coordinator_manager = NULL;
static ggml_mutex_t g_coordinator_init_mutex = PTHREAD_MUTEX_INITIALIZER;

// Forward declarations
static void * ggml_coordinator_thread_func(void * arg);
static void ggml_work_queue_init(struct ggml_work_queue * queue);
static void ggml_work_queue_destroy(struct ggml_work_queue * queue);
static void ggml_work_queue_enqueue(struct ggml_work_queue * queue, struct ggml_work_item * item);
static struct ggml_work_item * ggml_work_queue_dequeue(struct ggml_work_queue * queue);

// Global coordinator management functions
static struct ggml_numa_coordinator_manager * ggml_get_global_coordinator_manager(int n_threads, bool force_multi_socket);
static void ggml_register_program_exit_cleanup(void);

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
        // Don't decrement pending_items here - do it when work is actually completed
    }
    
    ggml_mutex_unlock(&queue->queue_mutex);
    return item;
}

// Coordinator thread function (one per NUMA node)
static void * ggml_coordinator_thread_func(void * arg) {
    if (!arg) {
        GGML_LOG_ERROR("FATAL: Thread received NULL state pointer\n");
        return NULL;
    }
    
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
            // Use a default thread count since we can't access the threadpool's internal structure
            int numa_threads = 4; // Default, should be configurable
            struct ggml_cplan plan = ggml_graph_plan(coordinator->numa_cgraph, numa_threads, coordinator->numa_pool);
            
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
        
        // Decrement pending items counter now that work is actually completed
        atomic_fetch_sub(&coordinator->work_queue.pending_items, 1);
        
        // Signal completion to work queue
        ggml_mutex_lock(&coordinator->work_queue.queue_mutex);
        ggml_cond_signal(&coordinator->work_queue.work_completed);
        ggml_mutex_unlock(&coordinator->work_queue.queue_mutex);
        
        // Free the work item and its result buffer (prevent memory leak)
        if (work_item->result_buffer) {
            free(work_item->result_buffer);
        }
        free(work_item);
    }
    
    atomic_store(&coordinator->active, false);
    GGML_LOG_INFO("Coordinator thread for NUMA node %d shutting down\n", coordinator->numa_node);
    
    return NULL;
}

// Get or create the global singleton coordinator manager
static struct ggml_numa_coordinator_manager * ggml_get_global_coordinator_manager(int n_threads, bool force_multi_socket) {
    // Thread-safe singleton initialization
    ggml_mutex_lock(&g_coordinator_init_mutex);
    
    if (g_global_coordinator_manager == NULL) {
        GGML_LOG_INFO("Creating global singleton 3-tier NUMA coordinator manager\n");
        g_global_coordinator_manager = ggml_numa_coordinator_manager_new(n_threads, force_multi_socket);
        
        if (g_global_coordinator_manager) {
            // Register cleanup function to run at program exit
            ggml_register_program_exit_cleanup();
            GGML_LOG_INFO("Global coordinator manager created and registered for program exit cleanup\n");
        }
    }
    
    ggml_mutex_unlock(&g_coordinator_init_mutex);
    return g_global_coordinator_manager;
}

// Program exit cleanup function
static void ggml_cleanup_global_coordinator_at_exit(void) {
    if (g_global_coordinator_manager) {
        GGML_LOG_INFO("Program exit: cleaning up global coordinator manager\n");
        ggml_numa_coordinator_manager_free(g_global_coordinator_manager);
        g_global_coordinator_manager = NULL;
    }
}

// Register cleanup to happen at program exit
static void ggml_register_program_exit_cleanup(void) {
    static bool cleanup_registered = false;
    if (!cleanup_registered) {
        atexit(ggml_cleanup_global_coordinator_at_exit);
        cleanup_registered = true;
    }
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
    atomic_init(&mgr->threads_started, false);
    
    ggml_mutex_init(&mgr->main_sync_mutex);
    ggml_cond_init(&mgr->main_sync_cond);
    ggml_work_queue_init(&mgr->global_work_queue);
    
    // Step 3: Create coordinator threads (one per NUMA node)
    mgr->coordinators = malloc(sizeof(struct ggml_coordinator_thread) * num_numa_nodes);
    if (!mgr->coordinators) {
        GGML_LOG_ERROR("Failed to allocate coordinator threads\n");
        ggml_work_queue_destroy(&mgr->global_work_queue);
        ggml_cond_destroy(&mgr->main_sync_cond);
        ggml_mutex_destroy(&mgr->main_sync_mutex);
        free(mgr);
        return NULL;
    }
    
    // Initialize coordinator array memory
    memset(mgr->coordinators, 0, sizeof(struct ggml_coordinator_thread) * num_numa_nodes);
    
    int threads_per_numa = n_threads / num_numa_nodes;
    if (threads_per_numa < 1) threads_per_numa = 1;
    
    for (int i = 0; i < num_numa_nodes; i++) {
        struct ggml_coordinator_thread * coord = &mgr->coordinators[i];
        coord->numa_node = i;
        atomic_init(&coord->active, false);
        atomic_init(&coord->shutdown_requested, false);
        atomic_init(&coord->thread_created, false);
        coord->total_work_items = 0;
        coord->total_processing_time_us = 0;
        
        // Step 4: Create NUMA-specific threadpool for this coordinator
        struct ggml_threadpool_params tpp;
        ggml_threadpool_params_init(&tpp, threads_per_numa);
        tpp.numa_aware = false; // CRITICAL: Disable coordinator recursion - we ARE the coordinator
        tpp.force_multi_socket = false; // Don't create nested coordinators
        
        // Note: CPU affinity will be set in the coordinator thread function
        // when it starts, using numa_run_on_node_mask()
        
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

// Get the global singleton coordinator manager (create if needed)
struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_get_global(int n_threads, bool force_multi_socket) {
    return ggml_get_global_coordinator_manager(n_threads, force_multi_socket);
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
        
        // Also signal shutdown to the work queue
        atomic_store(&coord->work_queue.shutdown_requested, true);
        
        // Wake up any sleeping coordinator threads
        ggml_mutex_lock(&coord->work_queue.queue_mutex);
        ggml_cond_broadcast(&coord->work_queue.work_available);
        ggml_mutex_unlock(&coord->work_queue.queue_mutex);
        
        // Wait for coordinator thread to finish (only if thread was actually created)
        if (atomic_load(&coord->thread_created)) {
            GGML_LOG_DEBUG("Waiting for coordinator thread %d to finish\n", i);
            ggml_thread_join(coord->thread_handle, NULL);
        }
        
        // Clean up work queue AFTER thread has finished - just clean resources, no signaling
        struct ggml_work_item * current = coord->work_queue.head;
        while (current) {
            struct ggml_work_item * next = current->next;
            if (current->result_buffer) {
                free(current->result_buffer);
            }
            free(current);
            current = next;
        }
        ggml_cond_destroy(&coord->work_queue.work_completed);
        ggml_cond_destroy(&coord->work_queue.work_available);
        ggml_mutex_destroy(&coord->work_queue.queue_mutex);
        
        // Step 11: Coordinator workers cleanup their NUMA threadpools and signal completion
        if (coord->numa_pool) {
            GGML_LOG_DEBUG("Freeing NUMA threadpool for coordinator %d\n", i);
            ggml_threadpool_free(coord->numa_pool);
            coord->numa_pool = NULL;
            GGML_LOG_DEBUG("NUMA threadpool for coordinator %d freed\n", i);
        }
        
        // Step 10: NUMA node threadpools cleanup and clear their cgraph references  
        if (coord->numa_cgraph) {
            GGML_LOG_DEBUG("Clearing cgraph reference for NUMA node %d\n", i);
            // Just clear the reference - the original cgraph is owned by the caller
            coord->numa_cgraph = NULL;
        }
        
        // No separate contexts to free since we're using references
        if (coord->numa_ctx) {
            GGML_LOG_DEBUG("Freeing context for NUMA node %d\n", i);
            ggml_free(coord->numa_ctx);
            coord->numa_ctx = NULL;
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

// Set cgraph for all NUMA nodes (Step 3: each gets reference to the original)
int ggml_numa_coordinator_manager_set_cgraph(struct ggml_numa_coordinator_manager * mgr, 
                                            const struct ggml_cgraph * master_cgraph) {
    if (!mgr || !master_cgraph) return -1;
    
    GGML_LOG_INFO("Setting cgraph reference for %d NUMA nodes\n", mgr->num_numa_nodes);
    
    // Clean up any existing cgraph contexts from previous computations
    for (int i = 0; i < mgr->num_numa_nodes; i++) {
        struct ggml_coordinator_thread * coord = &mgr->coordinators[i];
        
        if (coord->numa_cgraph) {
            GGML_LOG_DEBUG("Cleaning up previous cgraph for NUMA node %d\n", i);
            coord->numa_cgraph = NULL; // Will be freed with the context
        }
        
        if (coord->numa_ctx) {
            GGML_LOG_DEBUG("Cleaning up previous context for NUMA node %d\n", i);
            ggml_free(coord->numa_ctx);
            coord->numa_ctx = NULL;
        }
    }
    
    // Each NUMA node gets a reference to the same original cgraph
    // This ensures all coordinators work with the same tensors and memory
    for (int i = 0; i < mgr->num_numa_nodes; i++) {
        struct ggml_coordinator_thread * coord = &mgr->coordinators[i];
        
        // Store reference to the original cgraph (no copying needed)
        coord->numa_cgraph = (struct ggml_cgraph *)master_cgraph; // Cast away const
        coord->numa_ctx = NULL; // No separate context needed
        
        GGML_LOG_DEBUG("NUMA node %d received cgraph reference (cgraph=%p)\n", 
                      i, (void*)coord->numa_cgraph);
    }
    
    return 0;
}

// Start coordinator threads (called after cgraph is set) - safe for multiple calls
int ggml_numa_coordinator_manager_start(struct ggml_numa_coordinator_manager * mgr) {
    if (!mgr) return -1;
    
    // Check if threads are already started
    if (atomic_load(&mgr->threads_started)) {
        GGML_LOG_DEBUG("Coordinator threads already started, skipping\n");
        return 0;
    }
    
    GGML_LOG_INFO("Starting %d coordinator threads\n", mgr->num_numa_nodes);
    
    for (int i = 0; i < mgr->num_numa_nodes; i++) {
        struct ggml_coordinator_thread * coord = &mgr->coordinators[i];
        
        // Validate coordinator before thread creation
        if (!coord->numa_pool) {
            GGML_LOG_ERROR("Coordinator %d has no NUMA pool, cannot start thread\n", i);
            return -1;
        }
        
        if (!coord->numa_cgraph) {
            GGML_LOG_ERROR("Coordinator %d has no cgraph, cannot start thread\n", i);
            return -1;
        }
        
        GGML_LOG_DEBUG("Creating thread for coordinator %d (ptr=%p)\n", i, (void*)coord);
        
        if (ggml_thread_create(&coord->thread_handle, NULL, ggml_coordinator_thread_func, coord) != 0) {
            GGML_LOG_ERROR("Failed to create coordinator thread for NUMA node %d\n", i);
            return -1;
        }
        
        // Mark thread as successfully created
        atomic_store(&coord->thread_created, true);
        
        GGML_LOG_INFO("Coordinator thread started for NUMA node %d\n", i);
    }
    
    // Mark threads as started
    atomic_store(&mgr->threads_started, true);
    
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

// Submit computation graph to coordinator manager
int ggml_numa_coordinator_manager_compute_graph(struct ggml_numa_coordinator_manager * mgr,
                                               struct ggml_cgraph * cgraph) {
    if (!mgr || !cgraph) return -1;
    
    GGML_LOG_INFO("Submitting computation graph with %d nodes to coordinator manager\n", cgraph->n_nodes);
    
    // Set the cgraph for all coordinators (they will use their own copies)
    int result = ggml_numa_coordinator_manager_set_cgraph(mgr, cgraph);
    if (result != 0) {
        GGML_LOG_ERROR("Failed to set cgraph for coordinators\n");
        return -1;
    }
    
    // Start coordinator threads if not already started
    result = ggml_numa_coordinator_manager_start(mgr);
    if (result != 0) {
        GGML_LOG_ERROR("Failed to start coordinator threads\n");
        return -1;
    }
    
    // For simplicity, submit each node as a separate work item
    // In a more sophisticated implementation, this would analyze dependencies
    // and submit work in proper order
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if (node) {
            int work_id = ggml_numa_coordinator_manager_submit_work(mgr, node, -1);
            if (work_id < 0) {
                GGML_LOG_WARN("Failed to submit work for cgraph node %d\n", i);
            }
        }
    }
    
    // Wait for all work to complete
    result = ggml_numa_coordinator_manager_wait_for_completion(mgr);
    if (result != 0) {
        GGML_LOG_ERROR("Failed to wait for computation completion\n");
        return -1;
    }
    
    GGML_LOG_INFO("Computation graph completed successfully\n");
    return 0;
}

// Wait for all work to complete (Step 7: Main thread polls for completion)
int ggml_numa_coordinator_manager_wait_for_completion(struct ggml_numa_coordinator_manager * mgr) {
    if (!mgr) return -1;
    
    GGML_LOG_DEBUG("Main thread waiting for all work to complete\n");
    
    // Check all coordinators for pending work
    bool all_complete = false;
    while (!all_complete) {
        all_complete = true;
        
        for (int i = 0; i < mgr->num_numa_nodes; i++) {
            struct ggml_coordinator_thread * coord = &mgr->coordinators[i];
            int pending = atomic_load(&coord->work_queue.pending_items);
            
            if (pending > 0) {
                all_complete = false;
                break;
            }
        }
        
        if (!all_complete) {
            usleep(100); // Sleep 100 microseconds
        }
    }
    
    GGML_LOG_DEBUG("All work completed\n");
    
    // Add a small delay to ensure coordinator threads are truly idle
    // This prevents race conditions when switching to a new cgraph immediately
    usleep(1000); // 1ms delay
    
    return 0;
}

// Get performance statistics
struct ggml_numa_perf_stats ggml_numa_coordinator_manager_get_stats(struct ggml_numa_coordinator_manager * mgr, int numa_node) {
    struct ggml_numa_perf_stats stats = {0};
    
    if (!mgr) return stats;
    
    if (numa_node >= 0 && numa_node < mgr->num_numa_nodes) {
        // Get stats for specific NUMA node
        struct ggml_coordinator_thread * coord = &mgr->coordinators[numa_node];
        stats.total_work_items = coord->total_work_items;
        stats.total_processing_time_us = coord->total_processing_time_us;
        
        if (coord->total_work_items > 0) {
            stats.average_processing_time_us = coord->total_processing_time_us / coord->total_work_items;
            if (coord->total_processing_time_us > 0) {
                stats.throughput_items_per_sec = (double)coord->total_work_items * 1000000.0 / coord->total_processing_time_us;
            }
        }
    } else {
        // Get aggregated stats for all NUMA nodes
        for (int i = 0; i < mgr->num_numa_nodes; i++) {
            struct ggml_coordinator_thread * coord = &mgr->coordinators[i];
            stats.total_work_items += coord->total_work_items;
            stats.total_processing_time_us += coord->total_processing_time_us;
        }
        
        if (stats.total_work_items > 0) {
            stats.average_processing_time_us = stats.total_processing_time_us / stats.total_work_items;
            if (stats.total_processing_time_us > 0) {
                stats.throughput_items_per_sec = (double)stats.total_work_items * 1000000.0 / stats.total_processing_time_us;
            }
        }
    }
    
    return stats;
}
