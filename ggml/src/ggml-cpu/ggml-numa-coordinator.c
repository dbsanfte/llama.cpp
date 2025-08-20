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
#include "ggml-numa-executor.h"           // New NUMA executor
#include "ggml-numa-work-shared.h"         // Shared logging macros and utilities
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"  // For ggml_compute_params structure
#include "ops.h"  // For ggml_compute_forward_* functions
#include "binary-ops.h"  // For binary operation functions

#ifdef __linux__
#include <sched.h>
#include <numa.h>
#include <numaif.h>
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <stdatomic.h>
#include <pthread.h>
#include <time.h>
#include <math.h>

// Thread-local storage for virtual NUMA node (for simulated environments)
static __thread int g_virtual_numa_node = -1;

// Functions to manage virtual NUMA node for testing
void ggml_numa_set_virtual_node(int node) {
    g_virtual_numa_node = node;
}

int ggml_numa_get_current_node(void) {
    if (numa_available() >= 0) {
        // Get current CPU and determine its NUMA node
        int current_cpu = sched_getcpu();
        if (current_cpu >= 0) {
            int current_node = numa_node_of_cpu(current_cpu);
            if (current_node >= 0) {
                return current_node;
            }
        }
    }

    // Fallback: return node 0 if detection fails
    return 0;
}

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
#define ggml_cond_timedwait(c, m, t) pthread_cond_timedwait(c, m, t)
#define ggml_cond_signal(c) pthread_cond_signal(c)
#define ggml_cond_broadcast(c) pthread_cond_broadcast(c)
#define ggml_thread_create(t, a, f, d) pthread_create(t, a, f, d)
#define ggml_thread_join(t, v) pthread_join(t, v)

// Include min function
static inline size_t min_size_t(size_t a, size_t b) {
    return a < b ? a : b;
}

// New graph-level NUMA coordination structures

// Operation assignment for NUMA scheduling
struct ggml_numa_operation_assignment {
    struct ggml_tensor * operation;     // Complete operation to execute
    int assigned_numa_node;             // Which NUMA node handles it (-1 for unassigned)
    int64_t priority;                   // Scheduling priority (higher = earlier)
    struct ggml_tensor ** dependencies; // Operations this depends on
    int num_dependencies;               // Number of dependencies
    atomic_bool dependencies_ready;     // All dependencies completed
    atomic_bool completed;              // This operation completed
    int64_t estimated_cost;             // Estimated computation cost
    int64_t memory_requirement;         // Memory required (bytes)
};

// Graph scheduler for NUMA coordination
struct ggml_numa_graph_scheduler {
    struct ggml_numa_operation_assignment * assignments; // Operation assignments
    int num_operations;                    // Number of operations in graph
    int max_operations;                    // Maximum operations capacity
    ggml_mutex_t scheduler_mutex;          // Synchronization for scheduler
    ggml_cond_t operations_completed_cond; // Condition variable for operation completion
    struct ggml_cgraph * original_graph;   // Reference to original compute graph
    int num_numa_nodes;                    // Number of NUMA nodes available
    atomic_int completed_operations;       // Number of completed operations
    atomic_bool scheduler_completed;       // All operations completed
    
    // Load balancing state
    int64_t numa_load[GGML_NUMA_MAX_NODES];   // Current load per NUMA node
    int64_t numa_memory[GGML_NUMA_MAX_NODES]; // Memory usage per NUMA node
};

// Work item for graph-level operation execution (simplified from chunk-based)
struct ggml_work_item {
    // Generic execution - function pointer approach
    ggml_numa_work_function_t work_function;    // Function to execute (provided by dispatcher)
    void * work_context;                        // Context data for the function (provided by dispatcher)
    
    // Legacy operation field for compatibility (TODO: remove when dispatcher is fully updated)
    struct ggml_tensor * operation;             // Complete operation (not chunk) - DEPRECATED
    
    // Work item metadata
    int assigned_numa_node;                     // Target NUMA node
    struct ggml_tensor ** dependencies;         // Operations to wait for
    int num_dependencies;                       // Dependency count
    atomic_bool dependencies_ready;             // All dependencies completed
    atomic_bool completed;                      // This operation completed
    struct ggml_work_item * next;               // Next item in queue
    int work_id;                                // Unique work ID for tracking
    size_t required_work_buffer_size;           // Required work buffer size in bytes
    ggml_numa_execution_strategy_t execution_strategy;  // How to execute this operation
};

// Work group for data parallel operations
struct ggml_work_group {
    int group_id;                         // Unique group ID
    struct ggml_tensor * original_tensor; // Original tensor being processed
    struct ggml_work_item ** chunks;      // Array of work items (one per chunk)
    int num_chunks;                       // Number of chunks
    atomic_int completed_chunks;          // Number of completed chunks
    atomic_bool group_completed;          // Whether entire group is completed
    struct ggml_tensor * result_tensor;   // Integrated result tensor
    int64_t split_dimension;              // Which dimension was split (0=rows, 1=cols, etc.)
    
    // Lock-free linked list fields
    struct ggml_work_group * volatile atomic_next; // Next work group in active list
    atomic_bool in_active_list;                    // Whether this group is in the active list
    atomic_int ref_count;                          // Reference count for safe memory management
    
    // Synchronization for waiting on group completion
    ggml_mutex_t completion_mutex;  // Mutex for completion waiting
    ggml_cond_t completion_cond;    // Condition variable for completion
};

// Work group pool for memory allocation optimization
struct ggml_work_group_pool {
    struct ggml_work_group * pool_storage;     // Pre-allocated work group storage
    struct ggml_work_group ** free_list;       // Array of pointers to free groups
    int pool_size;                             // Total number of groups in pool
    atomic_int free_count;                     // Number of available groups
    ggml_mutex_t pool_mutex;                   // Mutex for pool operations
    int64_t total_allocations;                 // Performance counter: total allocations
    int64_t pool_hits;                         // Performance counter: successful pool gets
    int64_t pool_misses;                       // Performance counter: pool exhaustion events
};

// Work group tracking in manager (lock-free version)
struct ggml_work_group_tracker {
    struct ggml_work_group * volatile active_list_head; // Lock-free linked list head
    atomic_int next_group_id;                           // Next group ID to assign
    struct ggml_work_group_pool pool;                   // Pre-allocated work group pool
    
    // Statistics for lock-free operations
    atomic_long lockfree_list_adds;                     // Number of groups added to list
    atomic_long lockfree_list_removes;                  // Number of groups removed from list
    atomic_long lockfree_scan_cycles;                   // Number of integration scan cycles
    
    // Legacy fields for compatibility (to be removed)
    struct ggml_work_group ** groups;  // Array of active work groups (DEPRECATED)
    int max_groups;                    // Maximum number of groups (DEPRECATED)
    ggml_mutex_t groups_mutex;         // Mutex for group operations (DEPRECATED)
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
    int numa_node;                                  // NUMA node this coordinator manages
    int n_threads;                                  // Number of threads in this coordinator's pool
    struct ggml_threadpool * numa_pool;             // NUMA-specific threadpool
    const struct ggml_cgraph * numa_cgraph;         // NUMA node's own copy of cgraph (const since read-only)
    struct ggml_context * numa_ctx;                 // Context for this NUMA node's cgraph copy
    struct ggml_work_queue work_queue;              // Work queue for this coordinator
    ggml_thread_t thread_handle;                    // Thread handle
    atomic_bool active;                             // Whether thread is active
    atomic_bool shutdown_requested;                 // Shutdown request flag
    atomic_bool thread_created;                     // Whether thread was actually created
    struct ggml_numa_coordinator_manager * manager; // Reference to parent manager for callbacks
    
    // Performance tracking
    int64_t total_work_items;          // Total work items processed
    int64_t total_processing_time_us;  // Total processing time
};

// NUMA Multi-Socket Threadpool Manager (3-tier: main → coordinator → NUMA)
struct ggml_numa_coordinator_manager {
    int num_numa_nodes;                                   // Number of NUMA nodes
    struct ggml_coordinator_thread * coordinators;        // Array of coordinator threads (one per NUMA node)
    struct ggml_work_queue global_work_queue;             // Global work queue from main thread
    
    // Asynchronous integration system
    ggml_thread_t integration_thread;                     // Background integration thread
    atomic_bool integration_thread_active;               // Integration thread active flag
    atomic_bool integration_shutdown_requested;          // Shutdown flag for integration thread
    ggml_mutex_t integration_mutex;                       // Mutex for integration operations
    ggml_cond_t integration_work_available;               // Signal when work groups need integration
    
    // Synchronization for main thread
    atomic_int total_work_items;                          // Total work items pending
    atomic_int completed_work_items;                      // Completed work items
    ggml_mutex_t main_sync_mutex;                         // Main thread sync mutex
    ggml_cond_t main_sync_cond;                           // Main thread sync condition
    atomic_bool manager_active;                           // Manager is active
    atomic_bool threads_started;                          // Whether coordinator threads have been started
    
    // Data parallelism support
    struct ggml_work_group_tracker work_groups;           // Work group tracking
    
    // Progress callback system
    ggml_numa_progress_callback_t progress_callback;      // User progress callback function
    void * progress_callback_user_data;                   // User data for callback
    
    // Performance profiling
    int64_t total_computations;                           // Total number of multi-socket computations
    int64_t total_async_time_us;                          // Total time spent in async execution (microseconds)
    int64_t total_sync_time_us;                           // Total time spent in synchronization (microseconds)
    int64_t numa_times_us[GGML_NUMA_MAX_NODES];           // Individual NUMA node computation times
    int64_t last_computation_elements;                    // Elements in last computation (for throughput)
    
    // Memory management strategy
    enum ggml_numa_memory_strategy memory_strategy;       // Current memory management strategy
    
    // Work status tracking for critical error handling
    atomic_int last_work_status;                          // Last work execution status (ggml_status)
    ggml_mutex_t status_mutex;                           // Mutex for status updates
    ggml_mutex_t strategy_mutex;                          // Mutex for strategy changes
    
    // Fallback threadpool for simple CPU operations
    struct ggml_threadpool * fallback_threadpool;         // Simple threadpool for CPU fallbacks (NUMA node 0)
    int fallback_thread_count;                           // Number of threads in fallback pool
};

// Global singleton coordinator manager - persists for program lifetime
static struct ggml_numa_coordinator_manager * g_global_coordinator_manager = NULL;
static ggml_mutex_t g_coordinator_init_mutex = PTHREAD_MUTEX_INITIALIZER;

// Global threadpool cache for reuse across manager instances
#define MAX_CACHED_THREADPOOLS 8
struct ggml_threadpool_cache_entry {
    struct ggml_threadpool * pool;
    int n_threads;
    int numa_node;
    bool in_use;
    int64_t created_time_us;
    int64_t last_used_time_us;
    int reuse_count;
};

struct ggml_threadpool_cache {
    struct ggml_threadpool_cache_entry entries[MAX_CACHED_THREADPOOLS];
    int cache_size;
    ggml_mutex_t cache_mutex;
    int64_t total_requests;
    int64_t cache_hits;
    int64_t cache_misses;
    bool initialized;
};

static struct ggml_threadpool_cache g_threadpool_cache = {0};
static void ggml_threadpool_cache_init(void);
static struct ggml_threadpool * ggml_threadpool_cache_get(int n_threads, int numa_node);
static void ggml_threadpool_cache_return(struct ggml_threadpool * pool, int n_threads, int numa_node);
static void ggml_threadpool_cache_cleanup(void);
static void ggml_threadpool_cache_print_stats(void);

// Lock-free work group list operations
static void ggml_work_group_list_add(struct ggml_work_group_tracker * tracker, struct ggml_work_group * group);
static bool ggml_work_group_list_remove(struct ggml_work_group_tracker * tracker, struct ggml_work_group * group);
static void ggml_work_group_list_scan_and_integrate(struct ggml_work_group_tracker * tracker);
static void ggml_work_group_ref_inc(struct ggml_work_group * group);
static bool ggml_work_group_ref_dec_and_free(struct ggml_work_group_tracker * tracker, struct ggml_work_group * group);
static void ggml_work_group_tracker_print_lockfree_stats(struct ggml_work_group_tracker * tracker);

// Forward declarations
static void * ggml_coordinator_thread_func(void * arg);
static void * ggml_integration_thread_func(void * arg);  // New async integration thread
static void ggml_work_queue_init(struct ggml_work_queue * queue);
static void ggml_work_queue_destroy(struct ggml_work_queue * queue);
static void ggml_work_queue_enqueue(struct ggml_work_queue * queue, struct ggml_work_item * item);
static struct ggml_work_item * ggml_work_queue_dequeue(struct ggml_work_queue * queue);

// Work group management functions
static void ggml_work_group_tracker_init(struct ggml_work_group_tracker * tracker);
static void ggml_work_group_tracker_destroy(struct ggml_work_group_tracker * tracker);
static struct ggml_work_group * ggml_work_group_create(struct ggml_work_group_tracker * tracker, struct ggml_tensor * tensor, int num_chunks);
static void ggml_work_group_free(struct ggml_work_group_tracker * tracker, struct ggml_work_group * group);
static int ggml_work_group_check_completion(struct ggml_work_group * group);
// static int ggml_operation_split_for_numa(struct ggml_tensor * tensor, int num_numa_nodes, struct ggml_work_item *** out_chunks); // Unused for now

// Work group pool management functions
static void ggml_work_group_pool_init(struct ggml_work_group_pool * pool, int pool_size);
static void ggml_work_group_pool_destroy(struct ggml_work_group_pool * pool);
static struct ggml_work_group * ggml_work_group_pool_get(struct ggml_work_group_pool * pool);
static void ggml_work_group_pool_return(struct ggml_work_group_pool * pool, struct ggml_work_group * group);
static void ggml_work_group_pool_print_stats(struct ggml_work_group_pool * pool);

// Async work group completion functions
int ggml_numa_coordinator_manager_check_work_group_completion(struct ggml_numa_coordinator_manager * mgr, int work_group_id);

// Global coordinator management functions
static struct ggml_numa_coordinator_manager * ggml_get_global_coordinator_manager(int n_threads);
static struct ggml_numa_coordinator_manager * ggml_get_global_coordinator_manager_with_params(const struct ggml_threadpool_params * tpp);
static void ggml_register_program_exit_cleanup(void);

// New graph-level NUMA coordination functions
static struct ggml_numa_graph_scheduler * ggml_numa_create_graph_scheduler(
    struct ggml_cgraph * graph,
    int num_numa_nodes
);

static void ggml_numa_free_graph_scheduler(struct ggml_numa_graph_scheduler * scheduler);

static int ggml_numa_assign_operations_to_nodes(
    struct ggml_numa_graph_scheduler * scheduler
);

static enum ggml_status ggml_numa_execute_assigned_operations(
    struct ggml_numa_coordinator_manager * mgr,
    struct ggml_numa_graph_scheduler * scheduler
);

static enum ggml_status ggml_numa_node_execute_operation(
    struct ggml_coordinator_thread * coordinator,
    const struct ggml_work_item * work_item
);

static bool ggml_numa_should_coordinate(
    struct ggml_cgraph * cgraph,
    int n_threads
);

// Operation-level NUMA parallelization using proper GGML compute functions
static struct ggml_numa_coordinator_manager * ggml_get_global_coordinator_manager(int n_threads);
static struct ggml_numa_coordinator_manager * ggml_get_global_coordinator_manager_with_params(const struct ggml_threadpool_params * tpp);
static void ggml_register_program_exit_cleanup(void);

// Operation-specific NUMA parallelization using proper GGML compute functions
// Execute complete operations on NUMA nodes using GGML's optimized functions

// Per-thread work buffer management functions
static void * ggml_numa_get_partitioned_work_buffer(size_t required_size_per_thread, int numa_node, int n_threads);
static void ggml_numa_free_thread_work_buffers(void);

static enum ggml_status ggml_numa_node_execute_operation(
    struct ggml_coordinator_thread * coordinator,
    const struct ggml_work_item * work_item
) {
    NUMA_ASSERT(coordinator);
    NUMA_ASSERT(work_item);
    NUMA_ASSERT(work_item->work_function);

    NUMA_COORD_LOG_DEBUG(coordinator->numa_node, "executing generic work function with node_strategy=%d, on_node_strategy=%d", 
                    (int)work_item->execution_strategy.node_strategy,
                    (int)work_item->execution_strategy.on_node_strategy);
        
    // Set up compute parameters with partitioned work buffer 
    // Allocate buffer large enough for all threads, each thread gets its own partition
    int n_threads = (work_item->execution_strategy.on_node_strategy == NUMA_ON_NODE_STRATEGY_SINGLE_THREAD) ? 1 : coordinator->n_threads;
    void * work_buffer = NULL;
    if (work_item->required_work_buffer_size > 0) {
        work_buffer = ggml_numa_get_partitioned_work_buffer(work_item->required_work_buffer_size, coordinator->numa_node, n_threads);
        if (!work_buffer) {
            NUMA_COORD_LOG_ERROR(coordinator->numa_node, "Failed to allocate partitioned work buffer (%zu bytes per thread, %d threads)", 
                            work_item->required_work_buffer_size, n_threads);
            return GGML_STATUS_FAILED;
        }
    }
    
    struct ggml_compute_params params = {
        .ith = 0,
        .nth = n_threads,
        .wsize = work_item->required_work_buffer_size,
        .wdata = work_buffer,
        .threadpool = (work_item->execution_strategy.on_node_strategy == NUMA_ON_NODE_STRATEGY_SINGLE_THREAD) ? NULL : coordinator->numa_pool
    };
    
    // Execute the work function provided by the dispatcher
    // The coordinator has no knowledge of what this function does - it's completely generic
    NUMA_COORD_LOG_DEBUG(coordinator->numa_node, "About to execute work item %p with context %p", 
                    (void*)work_item, work_item->work_context);
    NUMA_COORD_LOG_DEBUG(coordinator->numa_node, "About to call work_function %p with context %p", 
                    (void*)work_item->work_function, work_item->work_context);
    
    // DEBUGGING: Validate function pointer and context before calling
    NUMA_COORD_LOG_DEBUG(coordinator->numa_node, "PRE-CALL: work_function=%p, work_context=%p, params=%p", 
                    (void*)work_item->work_function, work_item->work_context, (void*)&params);
    NUMA_COORD_LOG_DEBUG(coordinator->numa_node, "PRE-CALL: params.ith=%d, params.nth=%d, params.wsize=%zu", 
                    params.ith, params.nth, params.wsize);
    
    // Set virtual NUMA node for testing purposes (thread-local storage)
    ggml_numa_set_virtual_node(coordinator->numa_node);
    
    enum ggml_status status = work_item->work_function(work_item->work_context, &params);
    
    // This should never be reached if function crashes
    NUMA_COORD_LOG_DEBUG(coordinator->numa_node, "Work function returned status %d", (int)status);
    
    // Store work status for all results to ensure latest status is preserved
    ggml_mutex_lock(&coordinator->manager->status_mutex);
    atomic_store(&coordinator->manager->last_work_status, (int)status);
    ggml_mutex_unlock(&coordinator->manager->status_mutex);
    
    if (status == GGML_STATUS_SUCCESS) {
        GGML_LOG_DEBUG("NUMA%d: Successfully executed work function\n", coordinator->numa_node);
    } else {
        GGML_LOG_ERROR("NUMA%d: Work function failed with status %d\n", coordinator->numa_node, (int)status);
    }
    
    return status;
}

// NOTE: Work buffer size calculation removed from coordinator.
// Buffer sizes should be calculated by the dispatcher and passed to the coordinator.
// The coordinator's job is purely to execute work, not to calculate requirements.

// Work buffer management - ensure coordinator has sufficient NUMA-local work buffer
// Per-thread work buffer management using thread-local storage
__thread void * ggml_thread_work_buffer = NULL;
__thread size_t ggml_thread_work_buffer_size = 0;
__thread int ggml_thread_work_buffer_numa_node = -1;

// Partitioned work buffer management - allocate large buffer and partition per thread
static void * ggml_numa_get_partitioned_work_buffer(size_t required_size_per_thread, int numa_node, int n_threads) {
    size_t total_required_size = required_size_per_thread * n_threads;
    
    // Check if current thread buffer is sufficient for all threads
    if (ggml_thread_work_buffer && 
        ggml_thread_work_buffer_size >= total_required_size && 
        ggml_thread_work_buffer_numa_node == numa_node) {
        return ggml_thread_work_buffer;
    }
    
    // Free existing buffer if it exists
    if (ggml_thread_work_buffer) {
        GGML_LOG_DEBUG("NUMA%d: Growing partitioned work buffer from %zu to %zu bytes (%d threads)\n", 
                       numa_node, ggml_thread_work_buffer_size, total_required_size, n_threads);
        numa_free(ggml_thread_work_buffer, ggml_thread_work_buffer_size);
        ggml_thread_work_buffer = NULL;
        ggml_thread_work_buffer_size = 0;
    } else {
        GGML_LOG_DEBUG("NUMA%d: Allocating partitioned work buffer of %zu bytes (%zu per thread, %d threads)\n", 
                       numa_node, total_required_size, required_size_per_thread, n_threads);
    }
    
    // Allocate new NUMA-local buffer for all threads
    ggml_thread_work_buffer = numa_alloc_onnode(total_required_size, numa_node);
    if (!ggml_thread_work_buffer) {
        GGML_LOG_ERROR("NUMA%d: Failed to allocate NUMA-local partitioned work buffer of size %zu\n", 
                       numa_node, total_required_size);
        ggml_thread_work_buffer_size = 0;
        ggml_thread_work_buffer_numa_node = -1;
        return NULL;
    }
    
    ggml_thread_work_buffer_size = total_required_size;
    ggml_thread_work_buffer_numa_node = numa_node;
    GGML_LOG_DEBUG("NUMA%d: Successfully allocated %zu bytes partitioned NUMA-local work buffer\n", 
                   numa_node, total_required_size);
    return ggml_thread_work_buffer;
}

static void ggml_numa_free_thread_work_buffers(void) {
    if (ggml_thread_work_buffer) {
        GGML_LOG_DEBUG("NUMA%d: Freeing thread work buffer (%zu bytes)\n", 
                       ggml_thread_work_buffer_numa_node, ggml_thread_work_buffer_size);
        numa_free(ggml_thread_work_buffer, ggml_thread_work_buffer_size);
        ggml_thread_work_buffer = NULL;
        ggml_thread_work_buffer_size = 0;
        ggml_thread_work_buffer_numa_node = -1;
    }
}

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
        // Note: No longer need to free result_buffer - operations work on full tensors
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
        
        GGML_LOG_ERROR("🔧 DEQUEUE: Retrieved work item %p with context %p\n", 
                       (void*)item, item ? item->work_context : NULL);
    }
    
    ggml_mutex_unlock(&queue->queue_mutex);
    return item;
}

// Initialize work group tracker
static void ggml_work_group_tracker_init(struct ggml_work_group_tracker * tracker) {
    // Initialize lock-free fields
    tracker->active_list_head = NULL;  // Direct assignment instead of atomic_init for volatile pointer
    atomic_init(&tracker->lockfree_list_adds, 0);
    atomic_init(&tracker->lockfree_list_removes, 0);
    atomic_init(&tracker->lockfree_scan_cycles, 0);
    
    // Legacy initialization for compatibility
    tracker->max_groups = 64; // Support up to 64 concurrent work groups
    tracker->groups = malloc(sizeof(struct ggml_work_group *) * tracker->max_groups);
    memset(tracker->groups, 0, sizeof(struct ggml_work_group *) * tracker->max_groups);
    atomic_init(&tracker->next_group_id, 1);
    ggml_mutex_init(&tracker->groups_mutex);
    
    // Initialize work group pool with same capacity as tracker
    ggml_work_group_pool_init(&tracker->pool, tracker->max_groups);
}

// Destroy work group tracker
static void ggml_work_group_tracker_destroy(struct ggml_work_group_tracker * tracker) {
    if (!tracker || !tracker->groups) return;
    
    ggml_mutex_lock(&tracker->groups_mutex);
    
    // Clean up any remaining work groups
    for (int i = 0; i < tracker->max_groups; i++) {
        if (tracker->groups[i]) {
            ggml_work_group_free(tracker, tracker->groups[i]);
            tracker->groups[i] = NULL;
        }
    }
    
    ggml_mutex_unlock(&tracker->groups_mutex);
    ggml_mutex_destroy(&tracker->groups_mutex);
    
    // Print lock-free statistics before cleanup
    ggml_work_group_tracker_print_lockfree_stats(tracker);
    
    free(tracker->groups);
    tracker->groups = NULL;
    
    // Clean up work group pool
    ggml_work_group_pool_destroy(&tracker->pool);
}

// ================================================================================================
// Work Group Pool Management - Memory Allocation Optimization
// ================================================================================================

// Initialize work group pool
static void ggml_work_group_pool_init(struct ggml_work_group_pool * pool, int pool_size) {
    if (!pool || pool_size <= 0) return;
    
    pool->pool_size = pool_size;
    atomic_init(&pool->free_count, pool_size);
    pool->total_allocations = 0;
    pool->pool_hits = 0;
    pool->pool_misses = 0;
    
    // Pre-allocate storage for work groups
    pool->pool_storage = malloc(sizeof(struct ggml_work_group) * pool_size);
    pool->free_list = malloc(sizeof(struct ggml_work_group *) * pool_size);
    
    if (!pool->pool_storage || !pool->free_list) {
        GGML_LOG_ERROR("Failed to allocate work group pool storage\n");
        free(pool->pool_storage);
        free(pool->free_list);
        pool->pool_storage = NULL;
        pool->free_list = NULL;
        atomic_store(&pool->free_count, 0);
        return;
    }
    
    // Initialize all groups and add to free list
    for (int i = 0; i < pool_size; i++) {
        memset(&pool->pool_storage[i], 0, sizeof(struct ggml_work_group));
        pool->free_list[i] = &pool->pool_storage[i];
    }
    
    ggml_mutex_init(&pool->pool_mutex);
    GGML_LOG_DEBUG("Initialized work group pool with %d pre-allocated groups\n", pool_size);
}

// Destroy work group pool
static void ggml_work_group_pool_destroy(struct ggml_work_group_pool * pool) {
    if (!pool) return;
    
    ggml_work_group_pool_print_stats(pool);
    
    ggml_mutex_destroy(&pool->pool_mutex);
    free(pool->pool_storage);
    free(pool->free_list);
    
    pool->pool_storage = NULL;
    pool->free_list = NULL;
    atomic_store(&pool->free_count, 0);
}

// Get work group from pool (fast path)
static struct ggml_work_group * ggml_work_group_pool_get(struct ggml_work_group_pool * pool) {
    if (!pool || !pool->pool_storage) return NULL;
    
    pool->total_allocations++;
    
    ggml_mutex_lock(&pool->pool_mutex);
    
    int free_count = atomic_load(&pool->free_count);
    if (free_count <= 0) {
        ggml_mutex_unlock(&pool->pool_mutex);
        pool->pool_misses++;
        GGML_LOG_WARN("Work group pool exhausted, falling back to malloc\n");
        return NULL; // Fall back to malloc
    }
    
    // Get group from free list
    struct ggml_work_group * group = pool->free_list[free_count - 1];
    pool->free_list[free_count - 1] = NULL;
    atomic_fetch_sub(&pool->free_count, 1);
    
    ggml_mutex_unlock(&pool->pool_mutex);
    
    pool->pool_hits++;
    
    // Reset group state
    memset(group, 0, sizeof(struct ggml_work_group));
    return group;
}

// Return work group to pool (fast path)
static void ggml_work_group_pool_return(struct ggml_work_group_pool * pool, struct ggml_work_group * group) {
    if (!pool || !group || !pool->pool_storage) return;
    
    // Verify this group belongs to our pool
    bool belongs_to_pool = (group >= pool->pool_storage && 
                           group < pool->pool_storage + pool->pool_size);
    
    if (!belongs_to_pool) {
        GGML_LOG_DEBUG("Work group not from pool, using regular free\n");
        return; // This group was malloc'd, not from pool
    }
    
    ggml_mutex_lock(&pool->pool_mutex);
    
    int free_count = atomic_load(&pool->free_count);
    if (free_count >= pool->pool_size) {
        ggml_mutex_unlock(&pool->pool_mutex);
        GGML_LOG_ERROR("Work group pool corruption: too many free groups\n");
        return;
    }
    
    // Return to free list
    pool->free_list[free_count] = group;
    atomic_fetch_add(&pool->free_count, 1);
    
    ggml_mutex_unlock(&pool->pool_mutex);
}

// Print pool performance statistics
static void ggml_work_group_pool_print_stats(struct ggml_work_group_pool * pool) {
    if (!pool) return;
    
    double hit_rate = 0.0;
    if (pool->total_allocations > 0) {
        hit_rate = (double)pool->pool_hits / (double)pool->total_allocations * 100.0;
    }
    
    GGML_LOG_INFO("Work Group Pool Stats: %ld total allocations, %ld hits (%.1f%%), %ld misses, %d free\n",
                  pool->total_allocations, pool->pool_hits, hit_rate, pool->pool_misses, 
                  atomic_load(&pool->free_count));
}

// ================================================================================================
// Threadpool Cache Management - Persistent Threadpool Optimization
// ================================================================================================

// Initialize global threadpool cache
static void ggml_threadpool_cache_init(void) {
    if (g_threadpool_cache.initialized) return;
    
    memset(&g_threadpool_cache, 0, sizeof(g_threadpool_cache));
    ggml_mutex_init(&g_threadpool_cache.cache_mutex);
    g_threadpool_cache.initialized = true;
    
    GGML_LOG_DEBUG("Initialized threadpool cache with %d slots\n", MAX_CACHED_THREADPOOLS);
}

// Get threadpool from cache or create new one
static struct ggml_threadpool * ggml_threadpool_cache_get(int n_threads, int numa_node) {
    if (!g_threadpool_cache.initialized) {
        ggml_threadpool_cache_init();
    }
    
    g_threadpool_cache.total_requests++;
    
    ggml_mutex_lock(&g_threadpool_cache.cache_mutex);
    
    // Look for matching cached threadpool
    for (int i = 0; i < MAX_CACHED_THREADPOOLS; i++) {
        struct ggml_threadpool_cache_entry * entry = &g_threadpool_cache.entries[i];
        
        if (entry->pool && !entry->in_use && 
            entry->n_threads == n_threads && entry->numa_node == numa_node) {
            
            // Found matching cached threadpool
            entry->in_use = true;
            entry->last_used_time_us = ggml_time_us();
            entry->reuse_count++;
            
            ggml_mutex_unlock(&g_threadpool_cache.cache_mutex);
            
            g_threadpool_cache.cache_hits++;
            GGML_LOG_DEBUG("Threadpool cache HIT: reusing pool for %d threads, numa %d (reuse count: %d)\n", 
                          n_threads, numa_node, entry->reuse_count);
            
            return entry->pool;
        }
    }
    
    ggml_mutex_unlock(&g_threadpool_cache.cache_mutex);
    
    // Cache miss - need to create new threadpool
    g_threadpool_cache.cache_misses++;
    GGML_LOG_DEBUG("Threadpool cache MISS: creating new pool for %d threads, numa %d\n", n_threads, numa_node);
    
    return NULL; // Signal caller to create new threadpool
}

// Return threadpool to cache
static void ggml_threadpool_cache_return(struct ggml_threadpool * pool, int n_threads, int numa_node) {
    if (!pool || !g_threadpool_cache.initialized) return;
    
    ggml_mutex_lock(&g_threadpool_cache.cache_mutex);
    
    // Find the entry for this threadpool
    for (int i = 0; i < MAX_CACHED_THREADPOOLS; i++) {
        struct ggml_threadpool_cache_entry * entry = &g_threadpool_cache.entries[i];
        
        if (entry->pool == pool && entry->in_use) {
            entry->in_use = false;
            entry->last_used_time_us = ggml_time_us();
            
            ggml_mutex_unlock(&g_threadpool_cache.cache_mutex);
            GGML_LOG_DEBUG("Threadpool returned to cache: %d threads, numa %d\n", 
                          entry->n_threads, entry->numa_node);
            return;
        }
    }
    
    // Not found in cache - try to add new entry
    for (int i = 0; i < MAX_CACHED_THREADPOOLS; i++) {
        struct ggml_threadpool_cache_entry * entry = &g_threadpool_cache.entries[i];
        
        if (!entry->pool) {
            // Empty slot found
            entry->pool = pool;
            entry->n_threads = n_threads;  // ✅ Store the actual thread count
            entry->numa_node = numa_node;   // ✅ Store the actual NUMA node
            entry->in_use = false;
            entry->created_time_us = ggml_time_us();
            entry->last_used_time_us = ggml_time_us();
            entry->reuse_count = 0;
            
            ggml_mutex_unlock(&g_threadpool_cache.cache_mutex);
            GGML_LOG_DEBUG("Threadpool added to cache in slot %d\n", i);
            return;
        }
    }
    
    ggml_mutex_unlock(&g_threadpool_cache.cache_mutex);
    
    // Cache full - free the threadpool
    GGML_LOG_DEBUG("Threadpool cache full, freeing threadpool\n");
    ggml_threadpool_free(pool);
}

// Print threadpool cache statistics
static void ggml_threadpool_cache_print_stats(void) {
    if (!g_threadpool_cache.initialized) return;
    
    double hit_rate = 0.0;
    if (g_threadpool_cache.total_requests > 0) {
        hit_rate = (double)g_threadpool_cache.cache_hits / (double)g_threadpool_cache.total_requests * 100.0;
    }
    
    int active_pools = 0;
    for (int i = 0; i < MAX_CACHED_THREADPOOLS; i++) {
        if (g_threadpool_cache.entries[i].pool) active_pools++;
    }
    
    GGML_LOG_INFO("Threadpool Cache Stats: %ld requests, %ld hits (%.1f%%), %ld misses, %d cached pools\n",
                  g_threadpool_cache.total_requests, g_threadpool_cache.cache_hits, hit_rate,
                  g_threadpool_cache.cache_misses, active_pools);
}

// Cleanup threadpool cache
static void ggml_threadpool_cache_cleanup(void) {
    if (!g_threadpool_cache.initialized) return;
    
    // Skip stats printing and complex cleanup to avoid segfaults during program termination
    // Most cleanup happens naturally when threads exit
    
    // Just mark as uninitialized - avoid complex mutex operations during exit
    g_threadpool_cache.initialized = false;
    
    // Skip mutex destruction and threadpool freeing during exit cleanup
    // These operations can cause segfaults when C runtime is being torn down
}

// ============================================================================
// Lock-free Work Group List Operations
// ============================================================================

// Increment reference count for work group (for memory safety)
static void ggml_work_group_ref_inc(struct ggml_work_group * group) {
    if (group) {
        atomic_fetch_add(&group->ref_count, 1);
    }
}

// Decrement reference count and free if zero (for memory safety)
static bool ggml_work_group_ref_dec_and_free(struct ggml_work_group_tracker * tracker, struct ggml_work_group * group) {
    if (!group) return false;
    
    int old_ref = atomic_fetch_sub(&group->ref_count, 1);
    if (old_ref == 1) {
        // Reference count reached zero, safe to free
        GGML_LOG_DEBUG("Freeing work group %d (ref count reached 0)\n", group->group_id);
        
        // Clean up work group
        if (group->chunks) {
            free(group->chunks);
            group->chunks = NULL;
        }
        
        ggml_cond_destroy(&group->completion_cond);
        ggml_mutex_destroy(&group->completion_mutex);
        
        // Return to pool or free
        ggml_work_group_pool_return(&tracker->pool, group);
        return true;
    }
    return false;
}

// Add work group to lock-free active list
static void ggml_work_group_list_add(struct ggml_work_group_tracker * tracker, struct ggml_work_group * group) {
    if (!tracker || !group) return;
    
    ggml_work_group_ref_inc(group); // Increment ref count for list membership
    atomic_store(&group->in_active_list, true);
    
    // Lock-free insertion at head using compare-and-swap
    struct ggml_work_group * old_head;
    do {
        old_head = atomic_load(&tracker->active_list_head);
        group->atomic_next = old_head;
    } while (!atomic_compare_exchange_weak(&tracker->active_list_head, &old_head, group));
    
    atomic_fetch_add(&tracker->lockfree_list_adds, 1);
    GGML_LOG_DEBUG("Added work group %d to lock-free active list\n", group->group_id);
}

// Remove work group from lock-free active list (called when completed)
static bool ggml_work_group_list_remove(struct ggml_work_group_tracker * tracker, struct ggml_work_group * group) {
    if (!tracker || !group) return false;
    
    // Mark as not in list to prevent double removal
    bool was_in_list = atomic_exchange(&group->in_active_list, false);
    if (!was_in_list) {
        return false; // Already removed
    }
    
    // Note: For simplicity, we don't actually remove from the linked list here
    // The integration thread will skip groups where in_active_list is false
    // This avoids complex lock-free list removal with ABA problems
    
    atomic_fetch_add(&tracker->lockfree_list_removes, 1);
    GGML_LOG_DEBUG("Marked work group %d for removal from active list\n", group->group_id);
    
    // Decrement reference count (will be freed when ref count reaches 0)
    ggml_work_group_ref_dec_and_free(tracker, group);
    
    return true;
}

// Lock-free scan of active work groups and integrate completed ones
static void ggml_work_group_list_scan_and_integrate(struct ggml_work_group_tracker * tracker) {
    if (!tracker) return;
    
    atomic_fetch_add(&tracker->lockfree_scan_cycles, 1);
    
    // Walk the lock-free list without any locks
    struct ggml_work_group * current = atomic_load(&tracker->active_list_head);
    
    while (current) {
        struct ggml_work_group * next = atomic_load(&current->atomic_next);
        
        // Skip if group was marked for removal
        if (!atomic_load(&current->in_active_list)) {
            current = next;
            continue;
        }
        
        // Skip if already completed
        if (atomic_load(&current->group_completed)) {
            current = next;
            continue;
        }
        
        // Check if all chunks in this group are completed
        int completed_chunks = atomic_load(&current->completed_chunks);
        if (completed_chunks >= current->num_chunks) {
            GGML_LOG_DEBUG("Lock-free integration: Work group %d ready (%d/%d chunks)\n", 
                          current->group_id, completed_chunks, current->num_chunks);
            
            // Perform asynchronous integration (still need mutex for completion signaling)
            ggml_mutex_lock(&current->completion_mutex);
            
            if (!atomic_load(&current->group_completed)) {
                // Mark as completed
                atomic_store(&current->group_completed, true);
                
                // Signal any threads waiting on this work group
                ggml_cond_broadcast(&current->completion_cond);
                
                GGML_LOG_DEBUG("Lock-free integration: Work group %d integrated and completed\n", current->group_id);
                
                // Remove from active list
                ggml_work_group_list_remove(tracker, current);
            }
            
            ggml_mutex_unlock(&current->completion_mutex);
        }
        
        current = next;
    }
}

// Print lock-free statistics
static void ggml_work_group_tracker_print_lockfree_stats(struct ggml_work_group_tracker * tracker) {
    if (!tracker) return;
    
    long adds = atomic_load(&tracker->lockfree_list_adds);
    long removes = atomic_load(&tracker->lockfree_list_removes);
    long scans = atomic_load(&tracker->lockfree_scan_cycles);
    
    GGML_LOG_INFO("Lock-free Work Group Stats: %ld adds, %ld removes, %ld scan cycles\n", 
                  adds, removes, scans);
}

// Create new work group
static struct ggml_work_group * ggml_work_group_create(struct ggml_work_group_tracker * tracker, struct ggml_tensor * tensor, int num_chunks) {
    if (!tracker || !tensor || num_chunks <= 0) return NULL;
    
    // Try to get work group from pool first (fast path)
    struct ggml_work_group * group = ggml_work_group_pool_get(&tracker->pool);
    
    // Fall back to malloc if pool is exhausted (slow path)
    if (!group) {
        group = malloc(sizeof(struct ggml_work_group));
        if (!group) return NULL;
        memset(group, 0, sizeof(struct ggml_work_group));
        GGML_LOG_DEBUG("Work group allocation: using malloc fallback\n");
    }
    
    // Initialize work group
    group->group_id = atomic_fetch_add(&tracker->next_group_id, 1);
    group->original_tensor = tensor;
    group->num_chunks = num_chunks;
    atomic_init(&group->completed_chunks, 0);
    atomic_init(&group->group_completed, false);
    group->result_tensor = NULL;
    group->split_dimension = 0; // Default to row-wise split
    
    // Initialize lock-free fields
    group->atomic_next = NULL;
    atomic_init(&group->in_active_list, false);
    atomic_init(&group->ref_count, 1); // Start with reference count of 1
    
    // Initialize synchronization primitives
    ggml_mutex_init(&group->completion_mutex);
    ggml_cond_init(&group->completion_cond);
    
    // Allocate chunks array
    group->chunks = malloc(sizeof(struct ggml_work_item *) * num_chunks);
    if (!group->chunks) {
        // Return to pool if it came from there, otherwise free
        bool is_from_pool = (group >= tracker->pool.pool_storage && 
                            group < tracker->pool.pool_storage + tracker->pool.pool_size);
        if (is_from_pool) {
            ggml_work_group_pool_return(&tracker->pool, group);
        } else {
            free(group);
        }
        return NULL;
    }
    memset(group->chunks, 0, sizeof(struct ggml_work_item *) * num_chunks);
    
    // Add to lock-free active list (replaces mutex-based array storage)
    ggml_work_group_list_add(tracker, group);
    
    // Legacy storage for compatibility (TODO: remove after full migration)
    ggml_mutex_lock(&tracker->groups_mutex);
    bool stored = false;
    for (int i = 0; i < tracker->max_groups; i++) {
        if (tracker->groups[i] == NULL) {
            tracker->groups[i] = group;
            stored = true;
            break;
        }
    }
    ggml_mutex_unlock(&tracker->groups_mutex);
    
    if (!stored) {
        GGML_LOG_WARN("Legacy work group array full, but lock-free list succeeded\n");
    }
    
    GGML_LOG_DEBUG("Created work group %d with %d chunks for tensor %p\n", group->group_id, num_chunks, (void*)tensor);
    return group;
}

// Free work group
static void ggml_work_group_free(struct ggml_work_group_tracker * tracker, struct ggml_work_group * group) {
    if (!group) return;
    
    GGML_LOG_DEBUG("Freeing work group %d\n", group->group_id);
    
    // Free chunks (work items are freed by coordinator threads)
    if (group->chunks) {
        free(group->chunks);
        group->chunks = NULL;
    }
    
    // Clean up synchronization primitives
    ggml_mutex_destroy(&group->completion_mutex);
    ggml_cond_destroy(&group->completion_cond);
    
    // Note: We don't free result_tensor as it's typically owned by the caller
    // Note: We don't free original_tensor as it's owned by the caller
    
    // Return to pool if it came from there, otherwise free
    if (tracker) {
        ggml_work_group_pool_return(&tracker->pool, group);
        
        // Check if the return was successful (group was from pool)
        bool is_from_pool = (group >= tracker->pool.pool_storage && 
                            group < tracker->pool.pool_storage + tracker->pool.pool_size);
        if (!is_from_pool) {
            free(group); // It was malloc'd, so free it
        }
    } else {
        free(group); // No tracker available, assume it was malloc'd
    }
}

// Check work group completion and perform integration if needed
static int ggml_work_group_check_completion(struct ggml_work_group * group) {
    if (!group) return -1;
    
    // Simple implementation: check if all chunks are completed
    int completed_chunks = atomic_load(&group->completed_chunks);
    if (completed_chunks >= group->num_chunks) {
        // All chunks completed, mark group as completed
        if (!atomic_load(&group->group_completed)) {
            atomic_store(&group->group_completed, true);
            
            // Signal completion
            ggml_mutex_lock(&group->completion_mutex);
            ggml_cond_broadcast(&group->completion_cond);
            ggml_mutex_unlock(&group->completion_mutex);
        }
        return 1; // Successfully completed
    }
    
    return 0; // Still in progress
}

// Coordinator thread function (one per NUMA node) - Graph-level approach
static void * ggml_coordinator_thread_func(void * arg) {
    if (!arg) {
        GGML_LOG_ERROR("FATAL: Thread received NULL state pointer\n");
        return NULL;
    }
    
    struct ggml_coordinator_thread * coordinator = (struct ggml_coordinator_thread *)arg;
    
    GGML_LOG_INFO("Coordinator thread starting for NUMA node %d (graph-level)\n", coordinator->numa_node);
    
    // Set CPU affinity for this coordinator thread to its NUMA node
#ifdef __linux__
    if (coordinator->numa_node >= 0 && numa_available() != -1) {
        struct bitmask * numa_mask = numa_allocate_nodemask();
        numa_bitmask_setbit(numa_mask, coordinator->numa_node);
        numa_run_on_node_mask(numa_mask);
        numa_free_nodemask(numa_mask);
    }
#endif
    
    // Set thread-local NUMA node variable for tensor_data() access
    extern __thread int ggml_current_numa_node;
    ggml_current_numa_node = coordinator->numa_node;
    
    atomic_store(&coordinator->active, true);
    
    // Main coordinator loop - processes complete operations 
    while (!atomic_load(&coordinator->shutdown_requested) && 
           !atomic_load(&coordinator->work_queue.shutdown_requested)) {
        struct ggml_work_item * work_item = ggml_work_queue_dequeue(&coordinator->work_queue);
        
        if (!work_item) {
            // Either no work available or shutdown was requested during dequeue
            break;
        }
        
        int64_t start_time = ggml_time_us();
        
        // Execute complete operation using graph-level approach
        // This function handles both work_function (new approach) and operation (legacy approach)
        enum ggml_status status = ggml_numa_node_execute_operation(coordinator, work_item);
        
        int64_t end_time = ggml_time_us();
        coordinator->total_processing_time_us += (end_time - start_time);
        coordinator->total_work_items++;
        
        // Mark work item as completed
        atomic_store(&work_item->completed, true);
        
        // Check if this work item belongs to a work group and update completion
        if (coordinator->manager) {
            ggml_mutex_lock(&coordinator->manager->work_groups.groups_mutex);
            
            for (int g = 0; g < coordinator->manager->work_groups.max_groups; g++) {
                struct ggml_work_group * group = coordinator->manager->work_groups.groups[g];
                
                if (group) {
                    // Check if this work item belongs to this group
                    for (int c = 0; c < group->num_chunks; c++) {
                        if (group->chunks[c] == work_item) {
                            // This work item belongs to this group
                            int completed = atomic_fetch_add(&group->completed_chunks, 1) + 1;
                            GGML_LOG_DEBUG("Work group %d: chunk %d/%d completed\n", 
                                          group->group_id, completed, group->num_chunks);
                            
                            // Check if group is now complete
                            ggml_work_group_check_completion(group);
                            goto work_group_updated;
                        }
                    }
                }
            }

            work_group_updated:
            ggml_mutex_unlock(&coordinator->manager->work_groups.groups_mutex);
        }
        
        // Extract operation name for logging - try multiple sources
        const char * op_name = "unknown";
        if (work_item->operation) {
            op_name = ggml_op_name(work_item->operation->op);
        } else if (work_item->work_function) {
            // For work functions submitted directly, just use a generic name
            op_name = "function";
        }
        
        if (status != GGML_STATUS_SUCCESS) {
            GGML_LOG_WARN("Coordinator NUMA%d: Operation %s failed with status %d\n",
                         coordinator->numa_node, op_name, status);
        } else {
            GGML_LOG_DEBUG("Coordinator NUMA%d: Operation %s completed successfully\n",
                          coordinator->numa_node, op_name);
        }
        
        // Call progress callback if set
        if (coordinator->manager && coordinator->manager->progress_callback) {
            coordinator->manager->progress_callback(
                work_item->work_id,
                coordinator->numa_node, 
                work_item->operation,
                coordinator->manager->progress_callback_user_data
            );
        }
        
        // Decrement pending items counter
        atomic_fetch_sub(&coordinator->work_queue.pending_items, 1);
        
        // Signal completion to work queue
        ggml_mutex_lock(&coordinator->work_queue.queue_mutex);
        ggml_cond_signal(&coordinator->work_queue.work_completed);
        ggml_mutex_unlock(&coordinator->work_queue.queue_mutex);
        
        // Signal main thread's condition variable for manager-level waiting
        ggml_mutex_lock(&coordinator->manager->main_sync_mutex);
        ggml_cond_broadcast(&coordinator->manager->main_sync_cond);
        ggml_mutex_unlock(&coordinator->manager->main_sync_mutex);
        
        // Signal integration thread that work may be ready for integration
        ggml_mutex_lock(&coordinator->manager->integration_mutex);
        ggml_cond_signal(&coordinator->manager->integration_work_available);
        ggml_mutex_unlock(&coordinator->manager->integration_mutex);
        
        // Free the work item
        free(work_item);
    }
    
    // Mark thread as inactive before exit
    atomic_store(&coordinator->active, false);
    
    return NULL;
}

// Asynchronous integration thread function - handles work group completion and integration in background
static void * ggml_integration_thread_func(void * arg) {
    if (!arg) {
        GGML_LOG_ERROR("Integration thread: invalid argument\n");
        return NULL;
    }
    
    struct ggml_numa_coordinator_manager * mgr = (struct ggml_numa_coordinator_manager *)arg;
    
    GGML_LOG_INFO("Async integration thread starting\n");
    
    atomic_store(&mgr->integration_thread_active, true);
    
    // Main integration loop - runs continuously in background
    while (!atomic_load(&mgr->integration_shutdown_requested)) {
        
        // Lock-free list scan:
        ggml_work_group_list_scan_and_integrate(&mgr->work_groups);
        
        // Use a short timeout to periodically check for new work groups
        ggml_mutex_lock(&mgr->integration_mutex);
        
        // Double-check shutdown flag before waiting
        if (!atomic_load(&mgr->integration_shutdown_requested)) {
            struct timespec timeout;
            clock_gettime(CLOCK_REALTIME, &timeout);
            timeout.tv_nsec += 10000000; // 10ms timeout
            if (timeout.tv_nsec >= 1000000000) {
                timeout.tv_sec += 1;
                timeout.tv_nsec -= 1000000000;
            }
            ggml_cond_timedwait(&mgr->integration_work_available, &mgr->integration_mutex, &timeout);
        }
        
        ggml_mutex_unlock(&mgr->integration_mutex);
        
        // Final shutdown check after condition wait to ensure responsiveness
        if (atomic_load(&mgr->integration_shutdown_requested)) {
            break;
        }
    }
    
    // Mark thread as inactive before exit
    atomic_store(&mgr->integration_thread_active, false);
    
    return NULL;
}

// Get or create the global singleton coordinator manager
static struct ggml_numa_coordinator_manager * ggml_get_global_coordinator_manager(int n_threads) {
    // Thread-safe singleton initialization
    ggml_mutex_lock(&g_coordinator_init_mutex);
    
    if (g_global_coordinator_manager == NULL) {
        GGML_LOG_INFO("Creating global singleton NUMA coordinator manager\n");
        g_global_coordinator_manager = ggml_numa_coordinator_manager_new(n_threads);
        
        if (g_global_coordinator_manager) {
            // Register cleanup function to run at program exit
            ggml_register_program_exit_cleanup();
            GGML_LOG_INFO("Global coordinator manager created and registered for program exit cleanup\n");
        }
    }
    
    ggml_mutex_unlock(&g_coordinator_init_mutex);
    return g_global_coordinator_manager;
}

// Get or create the global singleton coordinator manager with parameters
static struct ggml_numa_coordinator_manager * ggml_get_global_coordinator_manager_with_params(const struct ggml_threadpool_params * tpp) {
    // Thread-safe singleton initialization
    ggml_mutex_lock(&g_coordinator_init_mutex);
    
    if (g_global_coordinator_manager == NULL) {
        GGML_LOG_INFO("Creating global singleton NUMA coordinator manager with custom parameters\n");
        g_global_coordinator_manager = ggml_numa_coordinator_manager_new_with_params(tpp);
        
        if (g_global_coordinator_manager) {
            // Initialize intelligent operation dispatcher
            // Register cleanup function to run at program exit
            ggml_register_program_exit_cleanup();
            GGML_LOG_INFO("Global coordinator manager created with parameters and registered for program exit cleanup\n");
        }
    }
    
    ggml_mutex_unlock(&g_coordinator_init_mutex);
    return g_global_coordinator_manager;
}

// Program exit cleanup function
static void ggml_cleanup_global_coordinator_at_exit(void) {
    // During atexit(), C runtime is being torn down - keep cleanup minimal and safe
    // Most cleanup is already handled by thread exits and normal shutdown
    
    if (g_global_coordinator_manager) {
        // Signal shutdown to any remaining threads (no logging to avoid segfaults)
        atomic_store(&g_global_coordinator_manager->manager_active, false);
        atomic_store(&g_global_coordinator_manager->integration_shutdown_requested, true);
        
        // Clear global reference without complex cleanup
        g_global_coordinator_manager = NULL;
    }
}

// Register cleanup to happen at program exit
static void ggml_register_program_exit_cleanup(void) {
    static bool cleanup_registered = false;
    if (!cleanup_registered) {
        // Skip atexit registration to avoid segfaults during program termination
        // The OS will clean up memory and threads when the program exits
        cleanup_registered = true;
    }
}

// Helper function to create hyperthreading-optimized CPU masks
// This assigns CPUs to avoid conflicts on the same physical core
static void create_optimal_cpu_masks(struct ggml_threadpool_params *tpp, int num_numa_nodes) {
    if (!tpp || num_numa_nodes < 1) return;  // Fixed: Allow single node optimization
    
    // Check if we have a custom CPU mask set
    bool has_custom_mask = false;
    int available_cpus[GGML_MAX_N_THREADS];
    int cpu_count = 0;
    GGML_UNUSED(available_cpus); // May be set but not used in all code paths
    
    for (int cpu = 0; cpu < GGML_MAX_N_THREADS; cpu++) {
        if (tpp->cpumask[cpu]) {
            available_cpus[cpu_count++] = cpu;
            has_custom_mask = true;
        }
    }
    
    // If no custom mask is set, create an optimal one based on system topology
    if (!has_custom_mask) {
        GGML_LOG_INFO("🔧 Creating NUMA-aware CPU assignment based on real topology...\n");
        
        // Handle auto-detection (-1) case
        int target_threads = tpp->n_threads;
        if (target_threads <= 0) {
            // Auto-detect: use all available logical CPUs
            target_threads = numa_num_configured_cpus();

            GGML_LOG_INFO("   Auto-detected %d threads (was %d)\n", target_threads, tpp->n_threads);
            tpp->n_threads = target_threads; // Update the threadpool params
        }
        
        GGML_LOG_INFO("   Target: %d total threads across %d NUMA nodes\n", target_threads, num_numa_nodes);
        
        // Clear the mask first
        memset(tpp->cpumask, false, sizeof(tpp->cpumask));
        cpu_count = 0;
        
        // Use real NUMA topology instead of hardcoded assumptions
        int total_cpus = numa_num_configured_cpus();
        
        // For systems with gaps in CPU numbering, we need to scan higher
        int cpu_scan_limit = GGML_MAX_N_THREADS;
        

        // Find the actual highest CPU number on the system
        int max_cpu_found = 0;
        for (int node = 0; node < num_numa_nodes; node++) {
            struct bitmask* test_cpus = numa_allocate_cpumask();
            if (numa_node_to_cpus(node, test_cpus) == 0) {
                for (int cpu = 0; cpu < GGML_MAX_N_THREADS; cpu++) {
                    if (numa_bitmask_isbitset(test_cpus, cpu) && cpu > max_cpu_found) {
                        max_cpu_found = cpu;
                    }
                }
            }
            numa_free_cpumask(test_cpus);
        }
        cpu_scan_limit = max_cpu_found + 1;
        
        GGML_LOG_INFO("   CPU scanning range: 0 to %d (total CPUs: %d, max CPU number: %d)\n", 
                     cpu_scan_limit - 1, total_cpus, max_cpu_found);
        int threads_per_node = target_threads / num_numa_nodes;
        
        GGML_LOG_INFO("Distributing %d threads across %d NUMA nodes (%d per node)\n", 
                     target_threads, num_numa_nodes, threads_per_node);
        
        for (int node = 0; node < num_numa_nodes; node++) {
            struct bitmask* node_cpus = numa_allocate_cpumask();
            if (numa_node_to_cpus(node, node_cpus) == 0) {
                int node_cpu_list[GGML_MAX_N_THREADS];
                int node_cpu_count = 0;
                
                // Collect all CPUs for this node
                for (int cpu = 0; cpu < cpu_scan_limit; cpu++) {
                    if (numa_bitmask_isbitset(node_cpus, cpu)) {
                        node_cpu_list[node_cpu_count++] = cpu;
                    }
                }
                
                // Assign threads to this node intelligently
                int assigned = 0;
                int target_threads_node = (node == num_numa_nodes - 1) ? 
                    (target_threads - (threads_per_node * (num_numa_nodes - 1))) : threads_per_node;
                
                GGML_LOG_INFO("NUMA node %d: has %d CPUs, assigning %d threads\n", 
                             node, node_cpu_count, target_threads_node);
                
                // Debug: Log which CPUs are available for this node
                char available_cpu_str[512] = {0};
                int pos = 0;
                for (int i = 0; i < node_cpu_count && pos < 500; i++) {
                    pos += snprintf(available_cpu_str + pos, sizeof(available_cpu_str) - pos, 
                                   "%s%d", i > 0 ? "," : "", node_cpu_list[i]);
                }
                GGML_LOG_INFO("   Available CPUs for node %d: [%s]\n", node, available_cpu_str);
                
                // Sort CPUs by core topology: first physical cores, then hyperthreads
                int primary_cpus[GGML_MAX_N_THREADS];
                int hyperthread_cpus[GGML_MAX_N_THREADS];
                int primary_count = 0, hyperthread_count = 0;
                
                for (int i = 0; i < node_cpu_count; i++) {
                    int cpu = node_cpu_list[i];
                    
                    // Try to detect primary vs hyperthread by checking core_id
                    // For most Intel systems, CPU 0,2,4... or similar patterns are primary
                    // But we need a more robust detection method
                    char thread_siblings_path[256];
                    snprintf(thread_siblings_path, sizeof(thread_siblings_path), 
                             "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
                    
                    FILE* siblings_file = fopen(thread_siblings_path, "r");
                    bool is_primary = true; // Default assumption
                    
                    if (siblings_file) {
                        char siblings_line[256];
                        if (fgets(siblings_line, sizeof(siblings_line), siblings_file)) {
                            // Parse thread siblings (e.g., "0,56" or "28,84")
                            // The first number in the list is typically the primary thread
                            int first_sibling = -1;
                            sscanf(siblings_line, "%d", &first_sibling);
                            is_primary = (cpu == first_sibling);
                        }
                        fclose(siblings_file);
                    }
                    
                    if (is_primary) {
                        primary_cpus[primary_count++] = cpu;
                    } else {
                        hyperthread_cpus[hyperthread_count++] = cpu;
                    }
                }
                
                GGML_LOG_INFO("   Primary cores: %d, Hyperthreads: %d\n", 
                             primary_count, hyperthread_count);
                
                // First pass: assign primary cores
                for (int i = 0; i < primary_count && assigned < target_threads_node; i++) {
                    int cpu = primary_cpus[i];
                    tpp->cpumask[cpu] = true;
                    available_cpus[cpu_count++] = cpu;
                    assigned++;
                    GGML_LOG_INFO("   Assigned primary CPU %d to node %d\n", cpu, node);
                }
                
                // Second pass: assign hyperthreads if needed
                for (int i = 0; i < hyperthread_count && assigned < target_threads_node; i++) {
                    int cpu = hyperthread_cpus[i];
                    tpp->cpumask[cpu] = true;
                    available_cpus[cpu_count++] = cpu;
                    assigned++;
                    GGML_LOG_INFO("   Assigned hyperthread CPU %d to node %d\n", cpu, node);
                }
            }
            numa_free_cpumask(node_cpus);
        }
        
        GGML_LOG_INFO("    Created NUMA-aware CPU mask with %d CPUs total across %d nodes\n", cpu_count, num_numa_nodes);
        
        // Debug: Show final global CPU mask
        char final_mask_str[1024] = {0};
        int final_pos = 0;
        for (int cpu = 0; cpu < GGML_MAX_N_THREADS && final_pos < 1000; cpu++) {
            if (tpp->cpumask[cpu]) {
                final_pos += snprintf(final_mask_str + final_pos, sizeof(final_mask_str) - final_pos, 
                                     "%s%d", final_pos > 0 ? "," : "", cpu);
            }
        }
        GGML_LOG_INFO("    Final global CPU mask: [%s]\n", final_mask_str);
    } else {
        GGML_LOG_INFO("    Using custom CPU mask with %d CPUs\n", cpu_count);
    }
}

// Create NUMA coordinator manager
struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_new(int n_threads) {
    // Create basic threadpool parameters and delegate to new function
    struct ggml_threadpool_params tpp;
    ggml_threadpool_params_init(&tpp, n_threads);
    
    return ggml_numa_coordinator_manager_new_with_params(&tpp);
}

// Create NUMA coordinator manager with threadpool parameters (supports CPU/NUMA masks)
struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_new_with_params(const struct ggml_threadpool_params * tpp) {
    if (!tpp) {
        GGML_LOG_ERROR("Invalid threadpool parameters\n");
        return NULL;
    }
    
    // Step 1: Determine number of NUMA nodes
    int num_numa_nodes = 1;
    bool numa_is_available = false;
    
    if (numa_available() != -1) {
        num_numa_nodes = numa_max_node() + 1;
        numa_is_available = true;
    }
    
    // === COMPREHENSIVE COORDINATOR SETUP LOGGING ===
    GGML_LOG_INFO("================================================================================\n");
    GGML_LOG_INFO("                     NUMA Coordinator Initialization\n");
    GGML_LOG_INFO("================================================================================\n");
    GGML_LOG_INFO("    Number of NUMA nodes requested: %d\n", num_numa_nodes);
    GGML_LOG_INFO("    Hardware NUMA available: %s\n", numa_is_available ? "YES" : "NO");
    GGML_LOG_INFO("    Total threads to distribute: %d\n", tpp->n_threads);
    GGML_LOG_INFO("    Threads per NUMA node: %d\n", tpp->n_threads / num_numa_nodes);
    GGML_LOG_INFO("    Strict CPU placement: %s\n", tpp->strict_cpu ? "STRICT_CPU" : "DEFAULT");
    GGML_LOG_INFO("    CPU mask enforcement: %s\n", 
                  tpp->cpumask[0] ? "ENABLED" : "DISABLED");
    
    // Log master CPU mask details if available
    if (tpp->cpumask[0]) {
        char cpu_list[512] = {0};
        int pos = 0;
        for (int cpu = 0; cpu < GGML_MAX_N_THREADS && pos < 500; cpu++) {
            if (tpp->cpumask[cpu]) {
                pos += snprintf(cpu_list + pos, sizeof(cpu_list) - pos, "%d,", cpu);
            }
        }
        if (pos > 0) cpu_list[pos - 1] = '\0'; // Remove trailing comma
        GGML_LOG_INFO("    Master CPU mask: [%s]\n", cpu_list);
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
    
    // Initialize async integration system
    atomic_init(&mgr->integration_thread_active, false);
    atomic_init(&mgr->integration_shutdown_requested, false);
    ggml_mutex_init(&mgr->integration_mutex);
    ggml_cond_init(&mgr->integration_work_available);
    
    // Initialize progress callback system
    mgr->progress_callback = NULL;
    mgr->progress_callback_user_data = NULL;
    
    // Initialize memory management strategy
    mgr->memory_strategy = GGML_NUMA_STRATEGY_AUTO;  // Default to adaptive strategy
    ggml_mutex_init(&mgr->strategy_mutex);
    
    // Initialize work status tracking
    atomic_init(&mgr->last_work_status, GGML_STATUS_SUCCESS);
    ggml_mutex_init(&mgr->status_mutex);
    
    // Log memory strategy information
    const char* strategy_name = "UNKNOWN";
    switch(mgr->memory_strategy) {
        case GGML_NUMA_STRATEGY_AUTO: strategy_name = "AUTO"; break;
        case GGML_NUMA_STRATEGY_MATRIX_REDUCTION: strategy_name = "MATRIX_REDUCTION"; break;
        case GGML_NUMA_STRATEGY_CHUNKED_PROCESSING: strategy_name = "CHUNKED_PROCESSING"; break;
        case GGML_NUMA_STRATEGY_HYBRID: strategy_name = "HYBRID"; break;
        default: strategy_name = "UNKNOWN"; break;
    }
    GGML_LOG_INFO("    Memory strategy: %s\n", strategy_name);
    
    ggml_mutex_init(&mgr->main_sync_mutex);
    ggml_cond_init(&mgr->main_sync_cond);
    ggml_work_queue_init(&mgr->global_work_queue);
    
    // Initialize work group tracker for data parallelism
    ggml_work_group_tracker_init(&mgr->work_groups);
    
    // Step 2.5: Create dedicated fallback threadpool for CPU operations
    // This is a simple threadpool on NUMA node 0 for fallback operations
    mgr->fallback_thread_count = 1;  // Start with single thread
    GGML_LOG_INFO("🔧 Creating fallback threadpool with %d thread(s) on NUMA node 0\n", mgr->fallback_thread_count);
    
    struct ggml_threadpool_params fallback_params = ggml_threadpool_params_default(mgr->fallback_thread_count);
    // Set CPU mask to NUMA node 0 CPUs if available
    if (numa_available() >= 0) {
        // Clear the CPU mask first
        for (int i = 0; i < GGML_MAX_N_THREADS; i++) {
            fallback_params.cpumask[i] = false;
        }
        // Set only NUMA node 0 CPUs
        struct bitmask * node0_mask = numa_allocate_cpumask();
        if (numa_node_to_cpus(0, node0_mask) >= 0) {
            for (int cpu = 0; cpu < numa_num_possible_cpus() && cpu < GGML_MAX_N_THREADS; cpu++) {
                if (numa_bitmask_isbitset(node0_mask, cpu)) {
                    fallback_params.cpumask[cpu] = true;
                }
            }
        }
        numa_free_cpumask(node0_mask);
    }
    fallback_params.strict_cpu = false;  // Allow OS to manage scheduling
    
    mgr->fallback_threadpool = ggml_threadpool_new(&fallback_params);
    if (!mgr->fallback_threadpool) {
        GGML_LOG_WARN("Failed to create fallback threadpool, fallback operations will be single-threaded\n");
        mgr->fallback_thread_count = 0;
    } else {
        GGML_LOG_INFO("✅ Fallback threadpool created successfully\n");
    }
    
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
    
    // NOTE: We'll calculate threads_per_numa after optimized CPU mask creation
    
    // Step 3.5: Create optimal CPU masks to avoid hyperthreading conflicts
    struct ggml_threadpool_params optimized_tpp = *tpp;  // Copy original parameters
    if (num_numa_nodes >= 1) {  // Fixed: Always create optimal masks, even for single node
        GGML_LOG_INFO("================================================================================\n");
        GGML_LOG_INFO("                     Creating Optimal CPU Masks\n");
        GGML_LOG_INFO("================================================================================\n");
        GGML_LOG_INFO("   Input: %d total threads, %d NUMA nodes\n", tpp->n_threads, num_numa_nodes);
        
        // Show original CPU mask if any
        bool has_original_mask = false;
        char orig_mask_str[512] = {0};
        int orig_pos = 0;
        for (int cpu = 0; cpu < GGML_MAX_N_THREADS && orig_pos < 500; cpu++) {
            if (tpp->cpumask[cpu]) {
                orig_pos += snprintf(orig_mask_str + orig_pos, sizeof(orig_mask_str) - orig_pos, 
                                   "%s%d", orig_pos > 0 ? "," : "", cpu);
                has_original_mask = true;
            }
        }
        GGML_LOG_INFO("   Original CPU mask: %s\n", has_original_mask ? orig_mask_str : "(none)");
        
        create_optimal_cpu_masks(&optimized_tpp, num_numa_nodes);
        
        GGML_LOG_INFO("================================================================================\n");
    }
    
    // Now calculate threads per NUMA node using the optimized thread count
    int threads_per_numa = optimized_tpp.n_threads / num_numa_nodes;
    if (threads_per_numa < 1) threads_per_numa = 1;
    
    GGML_LOG_INFO("Creating NUMA coordinator with %d threads distributed across %d NUMA nodes (%d threads per node)\n", 
                  optimized_tpp.n_threads, num_numa_nodes, threads_per_numa);
    
    for (int i = 0; i < num_numa_nodes; i++) {
        GGML_LOG_INFO("================================================================================\n");
        GGML_LOG_INFO("                Creating Coordinator for NUMA Node %d\n", i);
        GGML_LOG_INFO("================================================================================\n");
        
        struct ggml_coordinator_thread * coord = &mgr->coordinators[i];
        coord->numa_node = i;
        coord->n_threads = threads_per_numa;  // Store thread count for compute params
        atomic_init(&coord->active, false);
        atomic_init(&coord->shutdown_requested, false);
        atomic_init(&coord->thread_created, false);
        coord->total_work_items = 0;
        coord->total_processing_time_us = 0;
        coord->manager = mgr; // Reference to parent manager for callbacks
        
        // Step 4: Create NUMA-specific threadpool parameters for this coordinator
        struct ggml_threadpool_params numa_tpp = optimized_tpp; // Use optimized parameters
        numa_tpp.n_threads = threads_per_numa;
        numa_tpp.numa_aware = false; // CRITICAL: Disable coordinator recursion - we ARE the coordinator
        
        // Step 5: Apply NUMA-specific CPU mask filtering
        GGML_LOG_INFO("   Processing NUMA node %d in REAL NUMA mode (hardware node exists)\n", i);

        // Real NUMA: Filter the optimized CPU mask to only include CPUs from this NUMA node
        GGML_LOG_INFO("Filtering CPU mask for NUMA node %d (real NUMA mode)\n", i);
        
        struct bitmask* node_cpus = numa_allocate_cpumask();
        if (numa_node_to_cpus(i, node_cpus) == 0) {
            // Debug: Show optimized mask before filtering
            char opt_mask_str[1024] = {0};  // Increased size for more CPUs
            int opt_pos = 0;
            for (int cpu = 0; cpu < GGML_MAX_N_THREADS && opt_pos < 1000; cpu++) {
                if (optimized_tpp.cpumask[cpu]) {
                    opt_pos += snprintf(opt_mask_str + opt_pos, sizeof(opt_mask_str) - opt_pos, 
                                        "%s%d", opt_pos > 0 ? "," : "", cpu);
                }
            }
            GGML_LOG_INFO("   Optimized mask before filtering: [%s]\n", opt_mask_str);
            
            // Debug: Show what CPUs belong to this NUMA node  
            char node_mask_str[1024] = {0};  // Increased size for more CPUs
            int node_pos = 0;
            for (int cpu = 0; cpu < GGML_MAX_N_THREADS && node_pos < 1000; cpu++) {
                if (numa_bitmask_isbitset(node_cpus, cpu)) {
                    node_pos += snprintf(node_mask_str + node_pos, sizeof(node_mask_str) - node_pos, 
                                        "%s%d", node_pos > 0 ? "," : "", cpu);
                }
            }
            GGML_LOG_INFO("   CPUs belonging to NUMA node %d: [%s]\n", i, node_mask_str);
            
            // Create NUMA-filtered CPU mask
            memset(numa_tpp.cpumask, false, sizeof(numa_tpp.cpumask));
            bool has_numa_cpus = false;
            
            for (int cpu = 0; cpu < GGML_MAX_N_THREADS; cpu++) {
                // Include CPU if it's both in the optimized mask AND on this NUMA node
                if (optimized_tpp.cpumask[cpu] && numa_bitmask_isbitset(node_cpus, cpu)) {
                    numa_tpp.cpumask[cpu] = true;
                    has_numa_cpus = true;
                }
            }
            
            if (!has_numa_cpus) {
                // Debug: Show what filtering failed to find
                char filtered_mask_str[1024] = {0};  // Increased size for more CPUs
                int filtered_pos = 0;
                for (int cpu = 0; cpu < GGML_MAX_N_THREADS && filtered_pos < 1000; cpu++) {
                    if (numa_tpp.cpumask[cpu]) {
                        filtered_pos += snprintf(filtered_mask_str + filtered_pos, sizeof(filtered_mask_str) - filtered_pos, 
                                                "%s%d", filtered_pos > 0 ? "," : "", cpu);
                    }
                }
                GGML_LOG_WARN("   ❌ NUMA node %d: no intersection found between optimized mask and node CPUs\n", i);
                GGML_LOG_WARN("   ❌ Filtered result would be: [%s] (empty)\n", filtered_mask_str);
                GGML_LOG_WARN("NUMA node %d: no NUMA-local CPUs found in optimized mask, using original mask\n", i);
                memcpy(numa_tpp.cpumask, optimized_tpp.cpumask, sizeof(numa_tpp.cpumask));
            } else {
                // Debug: Show successful filtering result
                char filtered_mask_str[1024] = {0};  // Increased size for more CPUs
                int filtered_pos = 0;
                int filtered_count = 0;
                for (int cpu = 0; cpu < GGML_MAX_N_THREADS && filtered_pos < 1000; cpu++) {
                    if (numa_tpp.cpumask[cpu]) {
                        filtered_pos += snprintf(filtered_mask_str + filtered_pos, sizeof(filtered_mask_str) - filtered_pos, 
                                                "%s%d", filtered_pos > 0 ? "," : "", cpu);
                        filtered_count++;
                    }
                }
                GGML_LOG_INFO("   ✅ NUMA node %d: successfully filtered to %d CPUs: [%s]\n", i, filtered_count, filtered_mask_str);
                GGML_LOG_INFO("NUMA node %d: filtered optimized CPU mask to NUMA-local CPUs\n", i);
            }
        } else {
            GGML_LOG_WARN("NUMA node %d: failed to get node CPUs, using optimized mask\n", i);
            memcpy(numa_tpp.cpumask, optimized_tpp.cpumask, sizeof(numa_tpp.cpumask));
        }
        numa_free_cpumask(node_cpus);

        // Step 6: Create NUMA threadpool with cache optimization
        coord->numa_pool = ggml_threadpool_cache_get(numa_tpp.n_threads, i);
        if (!coord->numa_pool) {
            // Cache miss - create new threadpool
            coord->numa_pool = ggml_threadpool_new(&numa_tpp);
            if (!coord->numa_pool) {
                GGML_LOG_ERROR("Failed to create NUMA threadpool for node %d\n", i);
               
                // Cleanup previous coordinators  
                for (int j = 0; j < i; j++) {
                    ggml_threadpool_cache_return(mgr->coordinators[j].numa_pool, 
                                               mgr->coordinators[j].n_threads, 
                                               mgr->coordinators[j].numa_node);
                }
                free(mgr->coordinators);
                free(mgr);
                return NULL;
            }
            GGML_LOG_DEBUG("Created new threadpool for NUMA node %d (%d threads)\n", i, numa_tpp.n_threads);
        } else {
            GGML_LOG_DEBUG("Reused cached threadpool for NUMA node %d (%d threads)\n", i, numa_tpp.n_threads);
        }
        
        // Step 7: Initialize work queue for this coordinator
        ggml_work_queue_init(&coord->work_queue);
        
        // Step 8: Per-thread work buffers are now handled automatically
        GGML_LOG_INFO("    NUMA node %d: initialized per-thread work buffer system\n", i);
        
        // Enhanced logging with complete CPU assignment details
        char final_cpu_summary[256] = {0};
        int summary_pos = 0;
        bool has_mask = false;
        
        for (int cpu = 0; cpu < GGML_MAX_N_THREADS; cpu++) {
            if (numa_tpp.cpumask[cpu]) {
                if (!has_mask) {
                    summary_pos += snprintf(final_cpu_summary + summary_pos, 
                                          sizeof(final_cpu_summary) - summary_pos, "CPUs:");
                    has_mask = true;
                }
                summary_pos += snprintf(final_cpu_summary + summary_pos, 
                                      sizeof(final_cpu_summary) - summary_pos, "%d,", cpu);
            }
        }
        
        if (has_mask && summary_pos > 0) {
            final_cpu_summary[summary_pos - 1] = '\0';  // Remove trailing comma
        } else {
            snprintf(final_cpu_summary, sizeof(final_cpu_summary), "default affinity");
        }
        
        GGML_LOG_INFO("    Coordinator NUMA node %d: %d threads, %s\n", 
                      i, threads_per_numa, final_cpu_summary);
        GGML_LOG_INFO("✅ NUMA node %d coordinator created successfully with %d threads\n", i, threads_per_numa);
        GGML_LOG_INFO("================================================================================\n");
    }
    
    // === FINAL COMPREHENSIVE SUMMARY ===
    GGML_LOG_INFO("--------------------------------------------------------------------------------\n");
    GGML_LOG_INFO("                    NUMA Coordinator Initialization Complete\n");
    GGML_LOG_INFO("--------------------------------------------------------------------------------\n");
    GGML_LOG_INFO("    Status: %d NUMA coordinators created and configured\n", num_numa_nodes);
    
    // Calculate actual total threads distributed by summing from coordinators
    int total_distributed_threads = 0;
    for (int i = 0; i < num_numa_nodes; i++) {
        total_distributed_threads += mgr->coordinators[i].n_threads;
    }
    int avg_threads_per_node = total_distributed_threads / num_numa_nodes;
    
    GGML_LOG_INFO("    Total threads distributed: %d (avg %d per node)\n", 
                  total_distributed_threads, avg_threads_per_node);
    
    // Get current strategy name for final summary
    const char* final_strategy_name = "UNKNOWN";
    switch(mgr->memory_strategy) {
        case GGML_NUMA_STRATEGY_AUTO: final_strategy_name = "AUTO"; break;
        case GGML_NUMA_STRATEGY_MATRIX_REDUCTION: final_strategy_name = "MATRIX_REDUCTION"; break;
        case GGML_NUMA_STRATEGY_CHUNKED_PROCESSING: final_strategy_name = "CHUNKED_PROCESSING"; break;
        case GGML_NUMA_STRATEGY_HYBRID: final_strategy_name = "HYBRID"; break;
        default: final_strategy_name = "UNKNOWN"; break;
    }
    
    GGML_LOG_INFO("    Memory strategy: %s\n", final_strategy_name);
    GGML_LOG_INFO("    Thread binding: %s\n", 
                  tpp->cpumask[0] ? "CPU affinity enforced" : "Default OS scheduling");
    GGML_LOG_INFO("    Manager state: ACTIVE and ready for work distribution\n");
    GGML_LOG_INFO("\n");
    GGML_LOG_INFO("    Coordinator Details:\n");
    
    // Log individual coordinator summary with thread binding details
    for (int i = 0; i < num_numa_nodes; i++) {
        struct ggml_coordinator_thread * coord = &mgr->coordinators[i];
        const char* thread_status = atomic_load(&coord->thread_created) ? "CREATED" : "READY";
        GGML_LOG_INFO("      Node %d: CoordPtr=%p, Status=%s, NUMA=%s, Workers=%d\n", 
                      i, 
                      (void*)coord,
                      thread_status,
                      numa_is_available ? "HARDWARE" : "VIRTUAL",
                      coord->n_threads);  // Use actual thread count from coordinator
    }
    GGML_LOG_INFO("================================================================================\n");
    
    // Start the async integration thread for work group completion handling
    GGML_LOG_INFO("Starting async integration thread for background work group completion\n");
    if (ggml_thread_create(&mgr->integration_thread, NULL, ggml_integration_thread_func, mgr) != 0) {
        GGML_LOG_ERROR("Failed to create async integration thread\n");
        // Integration thread failure is not critical - manager can still work synchronously
    } else {
        GGML_LOG_INFO("✅ Async integration thread started successfully\n");
    }
    
    return mgr;
}

// Get the global singleton coordinator manager (create if needed)
struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_get_global(int n_threads) {
    return ggml_get_global_coordinator_manager(n_threads);
}

// Get the global singleton coordinator manager with parameters (create if needed)
struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_get_global_with_params(const struct ggml_threadpool_params * tpp) {
    return ggml_get_global_coordinator_manager_with_params(tpp);
}

// Get existing global singleton coordinator manager (no creation)
struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_get_existing(void) {
    return g_global_coordinator_manager;
}

// Free NUMA coordinator manager (hierarchical cleanup)
void ggml_numa_coordinator_manager_free(struct ggml_numa_coordinator_manager * mgr) {
    if (!mgr) return;
    
    // Step 8.5: Stop async integration thread first with graceful shutdown
    atomic_store(&mgr->integration_shutdown_requested, true);
    
    // Signal integration thread to wake up and check shutdown flag
    ggml_mutex_lock(&mgr->integration_mutex);
    ggml_cond_signal(&mgr->integration_work_available);
    ggml_mutex_unlock(&mgr->integration_mutex);
    
    // Wait for integration thread to finish if it was started
    if (atomic_load(&mgr->integration_thread_active)) {
        // Wait up to 1 second for graceful shutdown
        struct timespec timeout;
        clock_gettime(CLOCK_REALTIME, &timeout);
        timeout.tv_sec += 1; // 1 second timeout
        
        // Check if thread acknowledges shutdown by clearing the active flag
        bool graceful_shutdown = false;
        while (atomic_load(&mgr->integration_thread_active)) {
            struct timespec current_time;
            clock_gettime(CLOCK_REALTIME, &current_time);
            
            if (current_time.tv_sec > timeout.tv_sec || 
                (current_time.tv_sec == timeout.tv_sec && current_time.tv_nsec > timeout.tv_nsec)) {
                // Timeout reached
                break;
            }
            
            // Brief sleep to avoid busy waiting
            struct timespec sleep_time = { 0, 1000000 }; // 1ms
            nanosleep(&sleep_time, NULL);
        }
        
        if (!atomic_load(&mgr->integration_thread_active)) {
            graceful_shutdown = true;
        }
        
        // Skip join if graceful shutdown - thread has already exited cleanly
        if (!graceful_shutdown) {
            // Only join if thread didn't respond gracefully
            ggml_thread_join(mgr->integration_thread, NULL);
        }
    }
    
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
    }
    
    // Give threads a brief moment to complete their logging and shutdown gracefully
    struct timespec shutdown_delay = { 0, 10000000 }; // 10ms
    nanosleep(&shutdown_delay, NULL);
    
    // Now join with all coordinator threads (with timeout to prevent deadlock)
    for (int i = 0; i < mgr->num_numa_nodes; i++) {
        struct ggml_coordinator_thread * coord = &mgr->coordinators[i];
        
        // Wait for coordinator thread to finish (only if thread was actually created)
        if (atomic_load(&coord->thread_created)) {
            // Wait up to 100ms for thread to exit gracefully
            bool thread_exited = false;
            for (int attempt = 0; attempt < 10; attempt++) {
                if (!atomic_load(&coord->active)) {
                    thread_exited = true;
                    break;
                }
                struct timespec wait_delay = { 0, 10000000 }; // 10ms
                nanosleep(&wait_delay, NULL);
            }
            
            if (thread_exited) {
                // Thread exited gracefully, safe to join
                int join_result = ggml_thread_join(coord->thread_handle, NULL);
                if (join_result != 0) {
                    // Join failed, but thread is inactive so continue
                }
            } else {
                // Thread didn't exit gracefully within timeout - skip join to avoid deadlock
                // This prevents hanging but may leave some resources uncleaned
            }
        }
    }
    
    // Clean up resources for all coordinators after all threads have joined
    for (int i = 0; i < mgr->num_numa_nodes; i++) {
        struct ggml_coordinator_thread * coord = &mgr->coordinators[i];
        
        // Clean up work queue AFTER thread has finished - no result buffers in graph-level approach
        struct ggml_work_item * current = coord->work_queue.head;
        while (current) {
            struct ggml_work_item * next = current->next;
            // Note: No result_buffer to free in graph-level approach
            free(current);
            current = next;
        }
        ggml_cond_destroy(&coord->work_queue.work_completed);
        ggml_cond_destroy(&coord->work_queue.work_available);
        ggml_mutex_destroy(&coord->work_queue.queue_mutex);
        
        // Per-thread work buffers are automatically cleaned up when threads exit
        GGML_LOG_DEBUG("Per-thread work buffers for NUMA node %d will be cleaned automatically\n", i);
        
        // Step 11: Coordinator workers return NUMA threadpools to cache for reuse
        if (coord->numa_pool) {
            GGML_LOG_DEBUG("Returning NUMA threadpool for coordinator %d to cache\n", i);
            ggml_threadpool_cache_return(coord->numa_pool, coord->n_threads, coord->numa_node);
            coord->numa_pool = NULL;
            GGML_LOG_DEBUG("NUMA threadpool for coordinator %d returned to cache\n", i);
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
    
    // Clean up fallback threadpool
    if (mgr->fallback_threadpool) {
        GGML_LOG_INFO("🔧 Cleaning up fallback threadpool\n");
        ggml_threadpool_free(mgr->fallback_threadpool);
        mgr->fallback_threadpool = NULL;
        mgr->fallback_thread_count = 0;
    }
    
    ggml_work_group_tracker_destroy(&mgr->work_groups);
    ggml_work_queue_destroy(&mgr->global_work_queue);
    ggml_cond_destroy(&mgr->main_sync_cond);
    ggml_mutex_destroy(&mgr->main_sync_mutex);
    ggml_mutex_destroy(&mgr->strategy_mutex);
    
    // Clean up async integration system
    ggml_cond_destroy(&mgr->integration_work_available);
    ggml_mutex_destroy(&mgr->integration_mutex);
    
    if (mgr->coordinators) {
        free(mgr->coordinators);
    }
    
    free(mgr);
    ggml_threadpool_cache_print_stats();
}

// Free global NUMA coordinator manager (for backend cleanup)
void ggml_numa_coordinator_manager_free_global(void) {
    ggml_mutex_lock(&g_coordinator_init_mutex);
    
    if (g_global_coordinator_manager) {
        GGML_LOG_INFO("NUMA coordinator shutting down...\n");
        
        // During backend cleanup, use fast shutdown to avoid hanging
        // Signal shutdown to all coordinator threads
        atomic_store(&g_global_coordinator_manager->manager_active, false);
        atomic_store(&g_global_coordinator_manager->integration_shutdown_requested, true);
        
        for (int i = 0; i < g_global_coordinator_manager->num_numa_nodes; i++) {
            struct ggml_coordinator_thread * coord = &g_global_coordinator_manager->coordinators[i];
            
            // Signal shutdown to coordinator
            atomic_store(&coord->shutdown_requested, true);
            atomic_store(&coord->work_queue.shutdown_requested, true);
            
            // Wake up any sleeping coordinator threads
            ggml_mutex_lock(&coord->work_queue.queue_mutex);
            ggml_cond_broadcast(&coord->work_queue.work_available);
            ggml_mutex_unlock(&coord->work_queue.queue_mutex);
        }
        
        // Give threads a brief moment to exit gracefully
        struct timespec shutdown_delay = { 0, 50000000 }; // 50ms
        nanosleep(&shutdown_delay, NULL);
        
        // Don't do full cleanup during backend free to avoid deadlocks
        // Just clear the global reference - threads will exit on their own
        g_global_coordinator_manager = NULL;
    }
    
    ggml_mutex_unlock(&g_coordinator_init_mutex);
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
        
        // Store reference to the original cgraph (no cast needed with const field)
        coord->numa_cgraph = master_cgraph; 
        coord->numa_ctx = NULL; // No separate context needed
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
        
        // Note: numa_cgraph is only required for graph-level processing
        // Individual operation dispatch through the dispatcher can work without it
        if (!coord->numa_cgraph) {
            GGML_LOG_DEBUG("Coordinator %d has no cgraph - will only handle individual operations\n", i);
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
    
    // RACE CONDITION FIX: Wait for all coordinator threads to become active
    // This prevents work submission before threads are ready to process
    GGML_LOG_DEBUG("Waiting for all coordinator threads to become active...\n");
    for (int i = 0; i < mgr->num_numa_nodes; i++) {
        struct ggml_coordinator_thread * coord = &mgr->coordinators[i];
        
        // Polling wait with short delays to avoid hanging
        int wait_attempts = 0;
        const int max_wait_attempts = 1000; // 1 second total
        
        while (!atomic_load(&coord->active) && wait_attempts < max_wait_attempts) {
            usleep(1000); // 1ms delay
            wait_attempts++;
        }
        
        if (!atomic_load(&coord->active)) {
            GGML_LOG_ERROR("Coordinator thread %d failed to become active after 1 second\n", i);
            return -1;
        }
        
        GGML_LOG_DEBUG("Coordinator thread %d is now active\n", i);
    }
    
    // Mark threads as started
    atomic_store(&mgr->threads_started, true);
    
    GGML_LOG_INFO("All %d coordinator threads are active and ready\n", mgr->num_numa_nodes);
    
    return 0;
}

// Submit work to coordinator manager (Step 4: Main thread apportions work)
int ggml_numa_coordinator_manager_submit_work(struct ggml_numa_coordinator_manager * mgr,
                                              struct ggml_tensor * tensor,
                                              int numa_node_hint,
                                              ggml_numa_execution_strategy_t execution_strategy,
                                              size_t required_buffer_size) {
    if (!mgr || !tensor) return -1;
    
    // Ensure coordinator threads are started and have a cgraph before submitting work
    int start_result = ggml_numa_coordinator_manager_start(mgr);
    if (start_result != 0) {
        GGML_LOG_ERROR("Failed to start coordinator threads\n");
        return -1;
    }
    
    // Determine target NUMA node
    int target_numa = numa_node_hint;
    if (target_numa < 0 || target_numa >= mgr->num_numa_nodes) {
        if (mgr->num_numa_nodes > 0) {
            target_numa = atomic_load(&mgr->total_work_items) % mgr->num_numa_nodes; // Round-robin
        } else {
            // No NUMA nodes available - this is a critical error
            GGML_LOG_ERROR("Cannot submit work: no NUMA coordinators available (num_numa_nodes=%d)\n", mgr->num_numa_nodes);
            return -1;
        }
    }
    
    // Additional safety check - ensure target NUMA node is valid
    if (target_numa < 0 || target_numa >= mgr->num_numa_nodes || !mgr->coordinators) {
        GGML_LOG_ERROR("Invalid target NUMA node %d (available: 0-%d, coordinators=%p)\n", 
                       target_numa, mgr->num_numa_nodes - 1, (void*)mgr->coordinators);
        return -1;
    }
    
    // Create work item for graph-level operation
    struct ggml_work_item * work_item = malloc(sizeof(struct ggml_work_item));
    if (!work_item) return -1;
    
    // Use provided work buffer size (calculated by dispatcher)
    // No need to calculate - dispatcher already did the work!
    
    // Set up graph-level work item fields
    // NEW APPROACH: Function pointer (not set yet - dispatcher will need to be updated)
    work_item->work_function = NULL;     // No function pointer yet - using legacy approach
    work_item->work_context = NULL;      // No context yet
    
    // LEGACY APPROACH: Operation-based (current dispatcher still uses this)
    work_item->operation = tensor;           // Complete operation to execute
    work_item->assigned_numa_node = target_numa;
    work_item->dependencies = NULL;          // No dependencies for now
    work_item->num_dependencies = 0;
    atomic_init(&work_item->dependencies_ready, true); // Ready to execute
    atomic_init(&work_item->completed, false);
    work_item->next = NULL;
    work_item->work_id = atomic_fetch_add(&mgr->total_work_items, 1);
    work_item->required_work_buffer_size = required_buffer_size;
    work_item->execution_strategy = execution_strategy;  // How to execute this operation
    
    ggml_work_queue_enqueue(&mgr->coordinators[target_numa].work_queue, work_item);
    
    return work_item->work_id;
}

// Submit work function with generic function pointer (NEW APPROACH) 
int ggml_numa_coordinator_manager_submit_work_function(struct ggml_numa_coordinator_manager * mgr,
                                                       ggml_numa_work_function_t work_function,
                                                       void * work_context,
                                                       int numa_node_hint,
                                                       ggml_numa_execution_strategy_t execution_strategy,
                                                       size_t required_buffer_size) {
    if (!mgr || !work_function) return -1;
    
    // Reset work status to success before submitting new work
    ggml_mutex_lock(&mgr->status_mutex);
    atomic_store(&mgr->last_work_status, GGML_STATUS_SUCCESS);
    ggml_mutex_unlock(&mgr->status_mutex);
    
    // Ensure coordinator threads are started before submitting work
    int start_result = ggml_numa_coordinator_manager_start(mgr);
    if (start_result != 0) {
        GGML_LOG_ERROR("Failed to start coordinator threads for function work\n");
        return -1;
    }
    
    // Determine target NUMA node(s) based on execution strategy
    if (execution_strategy.node_strategy == NUMA_NODE_STRATEGY_DATA_PARALLEL) {
        // Data parallel: submit work to ALL available NUMA nodes
        GGML_LOG_DEBUG("Data parallel execution: submitting to all %d NUMA nodes\n", mgr->num_numa_nodes);
        
        int first_work_id = -1;
        
        // Submit identical work to each NUMA node
        for (int i = 0; i < mgr->num_numa_nodes; i++) {
            // Create work item for this NUMA node
            struct ggml_work_item * work_item = malloc(sizeof(struct ggml_work_item));
            if (!work_item) {
                GGML_LOG_ERROR("Failed to allocate work item for NUMA node %d in data parallel execution\n", i);
                return -1;
            }
            
            // Set up function-based work item
            work_item->work_function = work_function;
            work_item->work_context = work_context;
            work_item->operation = NULL;  // Using function pointer approach
            
            // Work item metadata
            work_item->assigned_numa_node = i;
            work_item->dependencies = NULL;
            work_item->num_dependencies = 0;
            atomic_init(&work_item->dependencies_ready, true);
            atomic_init(&work_item->completed, false);
            work_item->next = NULL;
            work_item->work_id = atomic_fetch_add(&mgr->total_work_items, 1);
            work_item->required_work_buffer_size = required_buffer_size;
            work_item->execution_strategy = execution_strategy;
            
            // Submit to this NUMA node's coordinator
            ggml_work_queue_enqueue(&mgr->coordinators[i].work_queue, work_item);
            
            GGML_LOG_ERROR("🔧 SUBMIT: Created work item %p with context %p for NUMA %d\n", 
                           (void*)work_item, work_context, i);
            GGML_LOG_ERROR("🔧 SUBMIT: Enqueued work item %p to NUMA %d (data parallel)\n", 
                           (void*)work_item, i);
            
            if (first_work_id < 0) {
                first_work_id = work_item->work_id;  // Return the first work ID
            }
        }
        
        GGML_LOG_DEBUG("Submitted data parallel work to %d NUMA nodes (first ID: %d)\n", 
                       mgr->num_numa_nodes, first_work_id);
        return first_work_id;
        
    } else {
        // Single node execution: use existing logic
        int target_numa = numa_node_hint;
        if (target_numa < 0 || target_numa >= mgr->num_numa_nodes) {
            if (mgr->num_numa_nodes > 0) {
                // Default to node 0 for single node strategy when no hint specified
                target_numa = 0;
            } else {
                GGML_LOG_ERROR("Cannot submit function work: no NUMA coordinators available\n");
                return -1;
            }
        }
        
        // Additional safety check - ensure target NUMA node is valid
        if (target_numa < 0 || target_numa >= mgr->num_numa_nodes || !mgr->coordinators) {
            GGML_LOG_ERROR("Invalid target NUMA node %d for function work (available: 0-%d)\n", 
                           target_numa, mgr->num_numa_nodes - 1);
            return -1;
        }
        
        // Create work item for single-node execution
        struct ggml_work_item * work_item = malloc(sizeof(struct ggml_work_item));
        if (!work_item) return -1;
        
        // Set up function-based work item
        work_item->work_function = work_function;
        work_item->work_context = work_context;
        work_item->operation = NULL;  // Using function pointer approach
        
        GGML_LOG_ERROR("🔧 SUBMIT: Created work item %p with context %p\n", 
                       (void*)work_item, work_context);
        
        // Work item metadata
        work_item->assigned_numa_node = target_numa;
        work_item->dependencies = NULL;
        work_item->num_dependencies = 0;
        atomic_init(&work_item->dependencies_ready, true);
        atomic_init(&work_item->completed, false);
        work_item->next = NULL;
        work_item->work_id = atomic_fetch_add(&mgr->total_work_items, 1);
        work_item->required_work_buffer_size = required_buffer_size;
        work_item->execution_strategy = execution_strategy;
        
        // Submit to target coordinator
        ggml_work_queue_enqueue(&mgr->coordinators[target_numa].work_queue, work_item);
        
        GGML_LOG_ERROR("🔧 SUBMIT: Enqueued work item %p to NUMA %d\n", 
                       (void*)work_item, target_numa);
        
        GGML_LOG_DEBUG("Submitted generic function work (ID: %d) to NUMA node %d\n", 
                       work_item->work_id, target_numa);
        
        return work_item->work_id;
    }
}

// Submit tensor for data parallel processing across multiple NUMA nodes
int ggml_numa_coordinator_manager_submit_data_parallel_work(struct ggml_numa_coordinator_manager * mgr,
                                                            struct ggml_tensor * tensor) {
    if (!mgr || !tensor || mgr->num_numa_nodes <= 1) {
        return -1;
    }
    
    GGML_LOG_DEBUG("Submitting data parallel work for tensor with %ld elements\n", ggml_nelements(tensor));
    
    // Create work group for data parallelism
    int num_chunks = mgr->num_numa_nodes; // One chunk per NUMA node
    struct ggml_work_group * group = ggml_work_group_create(&mgr->work_groups, tensor, num_chunks);
    if (!group) {
        GGML_LOG_ERROR("Failed to create work group for data parallel work\n");
        return -1;
    }
    
    // Create work items for each chunk (one per NUMA node)
    for (int i = 0; i < num_chunks; i++) {
        struct ggml_work_item * work_item = malloc(sizeof(struct ggml_work_item));
        if (!work_item) {
            GGML_LOG_ERROR("Failed to allocate work item for chunk %d\n", i);
            
            // Clean up partially created work group
            // First free any already created work items
            for (int j = 0; j < i; j++) {
                if (group->chunks[j]) {
                    free(group->chunks[j]);
                    group->chunks[j] = NULL;
                }
            }
            
            // Then free the work group itself
            ggml_work_group_free(&mgr->work_groups, group);
            return -1;
        }
        
        // Set up work item for complete operation (simplified from chunk-based)
        // NEW APPROACH: Function pointer (not set yet - dispatcher will need to be updated)
        work_item->work_function = NULL;     // No function pointer yet - using legacy approach
        work_item->work_context = NULL;      // No context yet
        
        // LEGACY APPROACH: Operation-based (current dispatcher still uses this)
        work_item->operation = tensor;           // Each NUMA node processes complete operation
        work_item->assigned_numa_node = i;       // Assign to specific NUMA node
        work_item->dependencies = NULL;
        work_item->num_dependencies = 0;
        atomic_init(&work_item->dependencies_ready, true);
        atomic_init(&work_item->completed, false);
        work_item->next = NULL;
        work_item->work_id = atomic_fetch_add(&mgr->total_work_items, 1);
        work_item->required_work_buffer_size = 0; // No buffer size info - let fallback handle it
        
        // Store work item in group
        group->chunks[i] = work_item;
        
        // Submit work item to assigned NUMA node with safety checks
        if (i >= mgr->num_numa_nodes || !mgr->coordinators) {
            GGML_LOG_ERROR("Cannot submit to NUMA node %d: invalid or not available\n", i);
            
            // Clean up the work group on error
            for (int j = 0; j <= i; j++) {
                if (group->chunks[j]) {
                    free(group->chunks[j]);
                    group->chunks[j] = NULL;
                }
            }
            ggml_work_group_free(&mgr->work_groups, group);
            return -1;
        }
        
        ggml_work_queue_enqueue(&mgr->coordinators[i].work_queue, work_item);
        
        GGML_LOG_DEBUG("Submitted work item %d to NUMA node %d for group %d\n", 
                       work_item->work_id, i, group->group_id);
    }
    
    GGML_LOG_DEBUG("Created data parallel work group %d with %d chunks\n", group->group_id, num_chunks);
    return group->group_id;
}

// Submit computation graph to coordinator manager
int ggml_numa_coordinator_manager_compute_graph(struct ggml_numa_coordinator_manager * mgr,
                                               struct ggml_cgraph * cgraph) {
    if (!mgr || !cgraph) return -1;
    
    GGML_LOG_INFO("Submitting computation graph with %d nodes to coordinator manager (%d NUMA nodes)\n", 
                  cgraph->n_nodes, mgr->num_numa_nodes);
    
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
    
    // Process each node in the computation graph
    // Delegate all operation-specific logic to the dispatcher
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if (!node) continue;
        
        GGML_LOG_DEBUG("Node %d: Processing operation %s (%ld elements) via dispatcher\n", 
                       i, ggml_op_name(node->op), ggml_nelements(node));
        
        // Create context for the dispatcher - use same approach as existing dispatcher code
        ggml_numa_work_context_t context = {
            .total_elements = ggml_nelements(node),
            .element_size = ggml_element_size(node),
            .numa_nodes = mgr->num_numa_nodes,
            .threads_per_node = mgr->num_numa_nodes > 0 ? mgr->coordinators[0].n_threads : 1,
            .l3_cache_size = 8ULL * 1024 * 1024,  // 8MB default
            .memory_bandwidth = 100ULL * 1024 * 1024 * 1024  // 100GB/s default
        };
        
        // Copy tensor dimensions
        for (int j = 0; j < GGML_MAX_DIMS && j < 4; j++) {
            context.ne[j] = node->ne[j];
        }
        context.n_dims = ggml_n_dims(node);
        
        // Delegate to dispatcher - it will handle all operation-specific decisions:
        // - Operation type analysis
        // - Data parallelism vs single-node decision  
        // - Work function selection
        // - Execution strategy optimization
        enum ggml_status dispatch_result = ggml_numa_dispatch_operation(mgr, node, &context);
        
        if (dispatch_result != GGML_STATUS_SUCCESS) {
            GGML_LOG_ERROR("Failed to dispatch operation %s for cgraph node %d (status: %d)\n", 
                           ggml_op_name(node->op), i, dispatch_result);
            return -1;
        }
        
        GGML_LOG_DEBUG("Node %d: Successfully dispatched operation %s\n", i, ggml_op_name(node->op));
    }
    
    // Wait for any remaining single-node work to complete
    result = ggml_numa_coordinator_manager_wait_for_completion(mgr);
    if (result != 0) {
        GGML_LOG_ERROR("Failed to wait for computation completion\n");
        return -1;
    }
    
    GGML_LOG_INFO("Computation graph completed successfully with data parallelism\n");
    return 0;
}

// Wait for all work to complete (Step 7: Main thread waits with condition variables)
enum ggml_status ggml_numa_coordinator_manager_wait_for_completion(struct ggml_numa_coordinator_manager * mgr) {
    if (!mgr) return GGML_STATUS_FAILED;
    
    GGML_LOG_DEBUG("Waiting for all work to complete using condition variables\n");
    
    ggml_mutex_lock(&mgr->main_sync_mutex);
    
    // Wait until all coordinators have no pending work
    while (true) {
        bool all_complete = true;
        
        for (int i = 0; i < mgr->num_numa_nodes; i++) {
            struct ggml_coordinator_thread * coord = &mgr->coordinators[i];
            int pending = atomic_load(&coord->work_queue.pending_items);
            
            if (pending > 0) {
                all_complete = false;
                break;
            }
        }
        
        if (all_complete) {
            break;
        }
        
        // RACE CONDITION FIX: Double-check condition before waiting to avoid lost signals
        // This prevents the classic race where work completes between checking and waiting
        GGML_LOG_DEBUG("Work still pending, entering wait...\n");
        
        // Wait on condition variable instead of sleeping
        // The coordinator threads will signal this condition when they complete work
        ggml_cond_wait(&mgr->main_sync_cond, &mgr->main_sync_mutex);
        
        // After being signaled, we'll loop back and check the condition again
        GGML_LOG_DEBUG("Received signal, re-checking work completion...\n");
    }
    
    ggml_mutex_unlock(&mgr->main_sync_mutex);
    
    GGML_LOG_DEBUG("All work completed\n");
    
    // Check if any work failed and return the status
    ggml_mutex_lock(&mgr->status_mutex);
    enum ggml_status final_status = (enum ggml_status)atomic_load(&mgr->last_work_status);
    ggml_mutex_unlock(&mgr->status_mutex);
    
    return final_status;
}

// Non-blocking check for work group completion (replaces blocking wait)
int ggml_numa_coordinator_manager_check_work_group_completion(struct ggml_numa_coordinator_manager * mgr, int work_group_id) {
    if (!mgr || work_group_id <= 0) return -1;
    
    struct ggml_work_group * target_group = NULL;
    
    // Find the work group
    ggml_mutex_lock(&mgr->work_groups.groups_mutex);
    for (int i = 0; i < mgr->work_groups.max_groups; i++) {
        struct ggml_work_group * group = mgr->work_groups.groups[i];
        if (group && group->group_id == work_group_id) {
            target_group = group;
            break;
        }
    }
    ggml_mutex_unlock(&mgr->work_groups.groups_mutex);
    
    if (!target_group) {
        GGML_LOG_DEBUG("Work group %d not found (may have been cleaned up)\n", work_group_id);
        return 1; // Assume completed if not found
    }
    
    // Non-blocking check for completion
    return atomic_load(&target_group->group_completed) ? 1 : 0;
}

// Wait for a specific work group to complete (used for data parallel work)
// NOTE: This is now mainly used for final synchronization, async integration handles most cases
int ggml_numa_coordinator_manager_wait_for_work_group(struct ggml_numa_coordinator_manager * mgr, int work_group_id) {
    if (!mgr || work_group_id <= 0) return -1;
    
    GGML_LOG_DEBUG("Final synchronization wait for work group %d\n", work_group_id);
    
    // First try non-blocking check
    int completion_status = ggml_numa_coordinator_manager_check_work_group_completion(mgr, work_group_id);
    if (completion_status == 1) {
        GGML_LOG_DEBUG("Work group %d already completed\n", work_group_id);
        return 0;
    } else if (completion_status == -1) {
        return -1; // Error case
    }
    
    struct ggml_work_group * target_group = NULL;
    
    // Find the work group for final synchronization
    ggml_mutex_lock(&mgr->work_groups.groups_mutex);
    for (int i = 0; i < mgr->work_groups.max_groups; i++) {
        struct ggml_work_group * group = mgr->work_groups.groups[i];
        if (group && group->group_id == work_group_id) {
            target_group = group;
            break;
        }
    }
    ggml_mutex_unlock(&mgr->work_groups.groups_mutex);
    
    if (!target_group) {
        GGML_LOG_DEBUG("Work group %d not found during final sync (may have been cleaned up)\n", work_group_id);
        return 0; // Assume completed if not found
    }
    
    // Final blocking wait for completion (should be rare due to async integration)
    ggml_mutex_lock(&target_group->completion_mutex);
    
    while (!atomic_load(&target_group->group_completed)) {
        // Brief timeout-based wait to allow async integration to complete
        struct timespec timeout;
        clock_gettime(CLOCK_REALTIME, &timeout);
        timeout.tv_nsec += 1000000; // 1ms timeout
        if (timeout.tv_nsec >= 1000000000) {
            timeout.tv_sec += 1;
            timeout.tv_nsec -= 1000000000;
        }
        
        int wait_result = ggml_cond_timedwait(&target_group->completion_cond, &target_group->completion_mutex, &timeout);
        if (wait_result != 0) {
            // Timeout - check if integration thread is handling it
            GGML_LOG_DEBUG("Work group %d: timeout in final sync, checking async integration status\n", work_group_id);
            break; // Exit and recheck completion status
        }
    }
    
    bool is_completed = atomic_load(&target_group->group_completed);
    ggml_mutex_unlock(&target_group->completion_mutex);
    
    if (is_completed) {
        GGML_LOG_DEBUG("Work group %d completed via async integration\n", work_group_id);
    } else {
        GGML_LOG_WARN("Work group %d: final sync timeout - async integration may still be processing\n", work_group_id);
    }
    
    // Clean up completed work group
    ggml_mutex_lock(&mgr->work_groups.groups_mutex);
    for (int i = 0; i < mgr->work_groups.max_groups; i++) {
        if (mgr->work_groups.groups[i] && mgr->work_groups.groups[i]->group_id == work_group_id) {
            ggml_work_group_free(&mgr->work_groups, mgr->work_groups.groups[i]);
            mgr->work_groups.groups[i] = NULL;
            break;
        }
    }
    ggml_mutex_unlock(&mgr->work_groups.groups_mutex);
    
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

// Set progress callback for work completion notifications
int ggml_numa_coordinator_manager_set_progress_callback(struct ggml_numa_coordinator_manager * mgr,
                                                        ggml_numa_progress_callback_t callback,
                                                        void * user_data) {
    if (!mgr) {
        GGML_LOG_ERROR("Invalid coordinator manager for setting progress callback\n");
        return -1;
    }
    
    mgr->progress_callback = callback;
    mgr->progress_callback_user_data = user_data;
    
    GGML_LOG_INFO("Progress callback %s\n", callback ? "enabled" : "disabled");
    return 0;
}

//
// New graph-level scheduler functions (Phase 2 implementation)
//

// Create a graph scheduler that assigns operations to NUMA nodes
static struct ggml_numa_graph_scheduler * ggml_numa_create_graph_scheduler(
    struct ggml_cgraph * graph,
    int num_numa_nodes
) {
    if (!graph || num_numa_nodes <= 0 || num_numa_nodes > GGML_NUMA_MAX_NODES) {
        GGML_LOG_ERROR("Invalid parameters for graph scheduler creation\n");
        return NULL;
    }
    
    struct ggml_numa_graph_scheduler * scheduler = malloc(sizeof(struct ggml_numa_graph_scheduler));
    if (!scheduler) {
        GGML_LOG_ERROR("Failed to allocate graph scheduler\n");
        return NULL;
    }
    
    memset(scheduler, 0, sizeof(struct ggml_numa_graph_scheduler));
    
    // Initialize scheduler fields
    scheduler->original_graph = graph;
    scheduler->num_numa_nodes = num_numa_nodes;
    scheduler->num_operations = graph->n_nodes;
    scheduler->max_operations = graph->n_nodes;
    atomic_init(&scheduler->completed_operations, 0);
    atomic_init(&scheduler->scheduler_completed, false);
    ggml_mutex_init(&scheduler->scheduler_mutex);
    ggml_cond_init(&scheduler->operations_completed_cond);
    
    // Initialize load balancing state
    for (int i = 0; i < num_numa_nodes; i++) {
        scheduler->numa_load[i] = 0;
        scheduler->numa_memory[i] = 0;
    }
    
    // Allocate operation assignments array (one per graph node)
    scheduler->assignments = malloc(sizeof(struct ggml_numa_operation_assignment) * graph->n_nodes);
    if (!scheduler->assignments) {
        GGML_LOG_ERROR("Failed to allocate operation assignments\n");
        ggml_mutex_destroy(&scheduler->scheduler_mutex);
        free(scheduler);
        return NULL;
    }
    
    // Initialize assignments
    for (int i = 0; i < graph->n_nodes; i++) {
        scheduler->assignments[i].operation = graph->nodes[i];
        scheduler->assignments[i].assigned_numa_node = -1; // Not assigned yet
        scheduler->assignments[i].memory_requirement = ggml_nbytes(graph->nodes[i]);
        scheduler->assignments[i].estimated_cost = ggml_nelements(graph->nodes[i]); // Simple estimate
        atomic_init(&scheduler->assignments[i].dependencies_ready, false);
        atomic_init(&scheduler->assignments[i].completed, false);
    }
    
    GGML_LOG_INFO("Created graph scheduler for %d operations across %d NUMA nodes\n", 
                  graph->n_nodes, num_numa_nodes);
    
    return scheduler;
}

// Free a graph scheduler
static void ggml_numa_free_graph_scheduler(struct ggml_numa_graph_scheduler * scheduler) {
    if (!scheduler) return;
    
    GGML_LOG_DEBUG("Freeing graph scheduler\n");
    
    if (scheduler->assignments) {
        free(scheduler->assignments);
    }
    
    ggml_mutex_destroy(&scheduler->scheduler_mutex);
    ggml_cond_destroy(&scheduler->operations_completed_cond);
    free(scheduler);
}

// Assign operations to NUMA nodes using load balancing
static int ggml_numa_assign_operations_to_nodes(
    struct ggml_numa_graph_scheduler * scheduler
) {
    if (!scheduler || !scheduler->original_graph) {
        GGML_LOG_ERROR("Invalid scheduler for operation assignment\n");
        return -1;
    }
    
    struct ggml_cgraph * graph = scheduler->original_graph;
    int num_nodes = graph->n_nodes;
    int num_numa_nodes = scheduler->num_numa_nodes;
    
    GGML_LOG_INFO("Assigning %d operations to %d NUMA nodes\n", num_nodes, num_numa_nodes);
    
    // Simple round-robin assignment for now - can be enhanced with load balancing
    for (int i = 0; i < num_nodes; i++) {
        struct ggml_tensor * operation = graph->nodes[i];
        
        // Calculate which NUMA node should handle this operation
        int assigned_node = -1;
        
        // Strategy 1: Round-robin for simplicity
        assigned_node = i % num_numa_nodes;
        
        // Strategy 2: Load balancing based on memory usage (future enhancement)
        // This could consider the memory requirements and current load per node
        
        // Assign the operation
        scheduler->assignments[i].assigned_numa_node = assigned_node;
        scheduler->numa_load[assigned_node] += scheduler->assignments[i].estimated_cost;
        scheduler->numa_memory[assigned_node] += scheduler->assignments[i].memory_requirement;
        
        GGML_LOG_DEBUG("Operation %d (%s) assigned to NUMA node %d (load: %ld ops, memory: %ld bytes)\n",
                       i, ggml_op_name(operation->op), assigned_node,
                       scheduler->numa_load[assigned_node], scheduler->numa_memory[assigned_node]);
    }
    
    // Log final load distribution
    for (int node = 0; node < num_numa_nodes; node++) {
        GGML_LOG_INFO("NUMA node %d: %ld operations, %ld bytes memory\n",
                      node, scheduler->numa_load[node], scheduler->numa_memory[node]);
    }
    
    return 0;
}

// Execute operations assigned to each NUMA node
static enum ggml_status ggml_numa_execute_assigned_operations(
    struct ggml_numa_coordinator_manager * mgr,
    struct ggml_numa_graph_scheduler * scheduler
) {
    if (!mgr || !scheduler || !scheduler->original_graph) {
        GGML_LOG_ERROR("Invalid parameters for executing assigned operations\n");
        return GGML_STATUS_FAILED;
    }
    
    GGML_LOG_INFO("Executing %d assigned operations using graph-level coordination\n", 
                  scheduler->num_operations);
    
    // Create work items for each operation and submit them to appropriate NUMA nodes
    for (int i = 0; i < scheduler->num_operations; i++) {
        struct ggml_numa_operation_assignment * assignment = &scheduler->assignments[i];
        
        if (assignment->assigned_numa_node < 0 || assignment->assigned_numa_node >= mgr->num_numa_nodes) {
            GGML_LOG_ERROR("Invalid NUMA node assignment for operation %d\n", i);
            continue;
        }
        
        // Create work item for this operation
        struct ggml_work_item * work_item = malloc(sizeof(struct ggml_work_item));
        if (!work_item) {
            GGML_LOG_ERROR("Failed to allocate work item for operation %d\n", i);
            continue;
        }
        
        // Set up work item for graph-level execution
        // NEW APPROACH: Function pointer (not set yet - dispatcher will need to be updated)
        work_item->work_function = NULL;     // No function pointer yet - using legacy approach
        work_item->work_context = NULL;      // No context yet
        
        // LEGACY APPROACH: Operation-based (current dispatcher still uses this)
        work_item->operation = assignment->operation;
        work_item->assigned_numa_node = assignment->assigned_numa_node;
        work_item->dependencies = NULL; // Simple for now - can add dependency tracking later
        work_item->num_dependencies = 0;
        atomic_init(&work_item->dependencies_ready, true); // Ready to execute
        atomic_init(&work_item->completed, false);
        work_item->next = NULL;
        work_item->work_id = i; // Use operation index as work ID
        work_item->required_work_buffer_size = 0; // No buffer size info - let fallback handle it
        
        // Submit to the assigned NUMA node with safety check
        if (assignment->assigned_numa_node >= 0 && assignment->assigned_numa_node < mgr->num_numa_nodes && mgr->coordinators) {
            ggml_work_queue_enqueue(&mgr->coordinators[assignment->assigned_numa_node].work_queue, work_item);
        } else {
            GGML_LOG_ERROR("Cannot submit work to invalid NUMA node %d (available nodes: %d)\n", 
                          assignment->assigned_numa_node, mgr->num_numa_nodes);
            free(work_item);
            continue;
        }
        
        GGML_LOG_DEBUG("Submitted operation %d (%s) to NUMA node %d\n",
                       i, ggml_op_name(assignment->operation->op), assignment->assigned_numa_node);
    }
    
    // Wait for all operations to complete using proper synchronization
    GGML_LOG_INFO("Waiting for all %d operations to complete...\n", scheduler->num_operations);
    
    // Use condition variables instead of busy waiting
    ggml_mutex_lock(&scheduler->scheduler_mutex);
    
    while (atomic_load(&scheduler->completed_operations) < scheduler->num_operations) {
        // Update completed operations count first
        int completed_count = 0;
        for (int i = 0; i < scheduler->num_operations; i++) {
            if (atomic_load(&scheduler->assignments[i].completed)) {
                completed_count++;
            }
        }
        atomic_store(&scheduler->completed_operations, completed_count);
        
        // Check if we're now complete
        if (completed_count >= scheduler->num_operations) {
            break;
        }
        
        // Wait on condition variable with timeout to periodically check for completion
        struct timespec timeout;
        clock_gettime(CLOCK_REALTIME, &timeout);
        timeout.tv_nsec += 1000000; // 1 millisecond timeout
        if (timeout.tv_nsec >= 1000000000) {
            timeout.tv_sec += 1;
            timeout.tv_nsec -= 1000000000;
        }
        
        // Use timed wait instead of indefinite wait
        ggml_cond_timedwait(&scheduler->operations_completed_cond, &scheduler->scheduler_mutex, &timeout);
    }
    
    ggml_mutex_unlock(&scheduler->scheduler_mutex);
    
    // Verify all operations are actually completed
    int final_completed_count = atomic_load(&scheduler->completed_operations);
    if (final_completed_count == scheduler->num_operations) {
        atomic_store(&scheduler->scheduler_completed, true);
        GGML_LOG_INFO("All %d operations completed successfully\n", final_completed_count);
        return GGML_STATUS_SUCCESS;
    } else {
        GGML_LOG_ERROR("Completion check failed: %d/%d operations completed\n", 
                       final_completed_count, scheduler->num_operations);
        return GGML_STATUS_FAILED;
    }
}

// ================================================================================================
// Memory Management Strategy Implementation
// ================================================================================================

int ggml_numa_coordinator_manager_set_strategy(struct ggml_numa_coordinator_manager * mgr, enum ggml_numa_memory_strategy strategy) {
    if (!mgr) {
        return -1;
    }
    
    ggml_mutex_lock(&mgr->strategy_mutex);
    mgr->memory_strategy = strategy;
    ggml_mutex_unlock(&mgr->strategy_mutex);
    
    GGML_LOG_DEBUG("NUMA coordinator strategy set to %d\n", strategy);
    return 0;
}

enum ggml_numa_memory_strategy ggml_numa_coordinator_manager_get_strategy(struct ggml_numa_coordinator_manager * mgr) {
    if (!mgr) {
        return GGML_NUMA_STRATEGY_AUTO;
    }
    
    ggml_mutex_lock(&mgr->strategy_mutex);
    enum ggml_numa_memory_strategy strategy = mgr->memory_strategy;
    ggml_mutex_unlock(&mgr->strategy_mutex);
    
    return strategy;
}

int ggml_numa_coordinator_manager_get_num_nodes(struct ggml_numa_coordinator_manager * mgr) {
    if (!mgr) {
        return 1; // Fallback to single node if no manager
    }
    
    return mgr->num_numa_nodes;
}

enum ggml_numa_memory_strategy ggml_numa_choose_strategy(const struct ggml_numa_workload_info * workload) {
    if (!workload) {
        return GGML_NUMA_STRATEGY_MATRIX_REDUCTION; // Safe default
    }
    
    // Honor user override if specified
    if (workload->user_override != GGML_NUMA_STRATEGY_AUTO) {
        return workload->user_override;
    }
    
    // Get cache information for cache-aware optimization
    struct ggml_numa_cache_info cache_info = {0};
    ggml_numa_detect_cache_info(&cache_info);
    
    // Adaptive strategy selection based on A/B test results and cache characteristics
    // Key findings from testing:
    // - Small matrices (≤512): Chunked processing wins by ~4%  
    // - Large matrices (≥768): Matrix reduction wins by ~4%
    // - Memory efficiency: Matrix reduction uses ~50% less memory
    // - Cache awareness: Consider L3 cache size for optimal chunk boundaries
    
    // If prioritizing scaling accuracy, always use matrix reduction
    if (workload->prioritize_scaling_accuracy) {
        return GGML_NUMA_STRATEGY_MATRIX_REDUCTION;
    }
    
    // Memory-constrained environments: prefer matrix reduction
    if (workload->available_memory_gb < 16) {
        return GGML_NUMA_STRATEGY_MATRIX_REDUCTION;
    }
    
    // Cache-aware strategy selection
    const size_t matrix_bytes = workload->matrix_dim * workload->matrix_dim * sizeof(float);
    const bool fits_in_l3 = cache_info.l3_cache_size > 0 && matrix_bytes <= (size_t)cache_info.l3_cache_size;
    const bool fits_in_l2 = cache_info.l2_cache_size > 0 && matrix_bytes <= (size_t)cache_info.l2_cache_size;
    
    // Very small matrices that fit in L2 cache: optimize for cache locality
    if (fits_in_l2 && workload->matrix_dim <= 256) {
        return GGML_NUMA_STRATEGY_CHUNKED_PROCESSING; // Better cache utilization
    }
    
    // Medium matrices that fit in L3 cache: consider cache sharing
    if (fits_in_l3) {
        if (cache_info.l3_sharing_cores > 1) {
            // Multiple cores share L3: matrix reduction minimizes cache conflicts
            return GGML_NUMA_STRATEGY_MATRIX_REDUCTION;
        } else {
            // Dedicated L3 per core: chunked processing can leverage full cache
            return GGML_NUMA_STRATEGY_CHUNKED_PROCESSING;
        }
    }
    
    // Large matrices exceeding L3 cache: fall back to empirical results
    if (workload->matrix_dim <= 512) {
        // Small matrices: chunked processing for better throughput
        return GGML_NUMA_STRATEGY_CHUNKED_PROCESSING;
    } else if (workload->matrix_dim >= 768) {
        // Large matrices: matrix reduction for better performance and memory efficiency
        return GGML_NUMA_STRATEGY_MATRIX_REDUCTION;
    } else {
        // Medium matrices: consider batch size and cache characteristics
        if (workload->batch_size >= 128 || (cache_info.l3_sharing_cores > 4)) {
            // Large batches or high cache contention: prefer matrix reduction for memory efficiency
            return GGML_NUMA_STRATEGY_MATRIX_REDUCTION;
        } else {
            // Small batches with good cache characteristics: chunked processing may be better
            return GGML_NUMA_STRATEGY_CHUNKED_PROCESSING;
        }
    }
}

static int64_t estimate_memory_usage_gb(const struct ggml_tensor * tensor, enum ggml_numa_memory_strategy strategy) {
    if (!tensor) return 0;
    
    int64_t elements = ggml_nelements(tensor);
    int64_t element_size = ggml_element_size(tensor);
    int64_t base_memory = elements * element_size;
    
    switch (strategy) {
        case GGML_NUMA_STRATEGY_MATRIX_REDUCTION:
            // Matrix reduction uses less memory due to smaller matrices
            return (base_memory * 3) / (1024 * 1024 * 1024); // 3x for intermediate results, convert to GB
            
        case GGML_NUMA_STRATEGY_CHUNKED_PROCESSING:
            // Chunked processing uses more memory for full-size matrices
            return (base_memory * 6) / (1024 * 1024 * 1024); // 6x for chunks and intermediates
            
        default:
            return (base_memory * 4) / (1024 * 1024 * 1024); // Conservative estimate
    }
}

int ggml_numa_coordinator_manager_submit_adaptive_work(struct ggml_numa_coordinator_manager * mgr,
                                                       struct ggml_tensor * tensor,
                                                       const struct ggml_numa_workload_info * workload) {
    if (!mgr || !tensor) {
        return -1;
    }
    
    // Get current strategy
    enum ggml_numa_memory_strategy current_strategy = ggml_numa_coordinator_manager_get_strategy(mgr);
    enum ggml_numa_memory_strategy chosen_strategy;
    
    if (current_strategy == GGML_NUMA_STRATEGY_AUTO) {
        // Use adaptive selection
        chosen_strategy = ggml_numa_choose_strategy(workload);
        GGML_LOG_DEBUG("Adaptive strategy selection chose: %d\n", chosen_strategy);
    } else {
        // Use fixed user-specified strategy
        chosen_strategy = current_strategy;
        GGML_LOG_DEBUG("Using fixed strategy: %d\n", chosen_strategy);
    }
    
    // Estimate memory usage for logging/monitoring
    int64_t estimated_memory = estimate_memory_usage_gb(tensor, chosen_strategy);
    GGML_LOG_DEBUG("Estimated memory usage: %ldGB for strategy %d\n", estimated_memory, chosen_strategy);
    
    // For now, delegate to the existing data parallel work submission
    // TODO: In future, implement strategy-specific processing logic here
    return ggml_numa_coordinator_manager_submit_data_parallel_work(mgr, tensor);
}

// ================================================================================================
// CPU Cache Detection and Optimization
// ================================================================================================

int ggml_numa_detect_cache_info(struct ggml_numa_cache_info * cache_info) {
    if (!cache_info) {
        return -1;
    }
    
    // Initialize with safe defaults
    cache_info->l1_cache_size = 32 * 1024;      // 32KB default
    cache_info->l2_cache_size = 256 * 1024;     // 256KB default  
    cache_info->l3_cache_size = 8 * 1024 * 1024; // 8MB default
    cache_info->cache_line_size = 64;           // 64 bytes default
    cache_info->l3_sharing_cores = 4;           // 4 cores sharing L3 default
    cache_info->cache_detection_successful = false;
    
#ifdef __linux__
    // Try to detect cache sizes from sysfs
    FILE* fp;
    char path[256];
    char line[128];
    long cache_size;
    
    // Detect L1 data cache size (CPU 0 as representative)
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu0/cache/index0/size");
    fp = fopen(path, "r");
    if (fp) {
        if (fgets(line, sizeof(line), fp)) {
            // Parse size like "32K" or "256K" 
            cache_size = strtol(line, NULL, 10);
            if (strstr(line, "K")) cache_size *= 1024;
            else if (strstr(line, "M")) cache_size *= 1024 * 1024;
            
            if (cache_size > 0 && cache_size < 1024 * 1024) { // Reasonable L1 size
                cache_info->l1_cache_size = cache_size;
            }
        }
        fclose(fp);
    }
    
    // Detect L2 cache size  
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu0/cache/index2/size");
    fp = fopen(path, "r");
    if (fp) {
        if (fgets(line, sizeof(line), fp)) {
            cache_size = strtol(line, NULL, 10);
            if (strstr(line, "K")) cache_size *= 1024;
            else if (strstr(line, "M")) cache_size *= 1024 * 1024;
            
            if (cache_size > 0 && cache_size < 16 * 1024 * 1024) { // Reasonable L2 size
                cache_info->l2_cache_size = cache_size;
            }
        }
        fclose(fp);
    }
    
    // Detect L3 cache size
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu0/cache/index3/size");
    fp = fopen(path, "r");
    if (fp) {
        if (fgets(line, sizeof(line), fp)) {
            cache_size = strtol(line, NULL, 10);
            if (strstr(line, "K")) cache_size *= 1024;
            else if (strstr(line, "M")) cache_size *= 1024 * 1024;
            
            if (cache_size > 0) { // Any reasonable L3 size
                cache_info->l3_cache_size = cache_size;
            }
        }
        fclose(fp);
    }
    
    // Detect cache line size
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size");
    fp = fopen(path, "r");
    if (fp) {
        if (fgets(line, sizeof(line), fp)) {
            int line_size = atoi(line);
            if (line_size >= 32 && line_size <= 128) { // Reasonable cache line size
                cache_info->cache_line_size = line_size;
            }
        }
        fclose(fp);
    }
    
    // Try to count cores sharing L3 cache
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu0/cache/index3/shared_cpu_list");
    fp = fopen(path, "r");
    if (fp) {
        if (fgets(line, sizeof(line), fp)) {
            // Count CPUs in shared list (e.g., "0-7" or "0,2,4,6")
            int sharing_cores = 1; // At least the current core
            char* ptr = line;
            while (*ptr) {
                if (*ptr == ',' || *ptr == '-') sharing_cores++;
                ptr++;
            }
            if (sharing_cores > 0 && sharing_cores <= 64) { // Reasonable sharing count
                cache_info->l3_sharing_cores = sharing_cores;
            }
        }
        fclose(fp);
    }
    
    cache_info->cache_detection_successful = true;
    
    GGML_LOG_DEBUG("Cache detection: L1=%ldKB, L2=%ldKB, L3=%ldMB, line=%dB, L3_sharing=%d\n",
                   cache_info->l1_cache_size / 1024,
                   cache_info->l2_cache_size / 1024, 
                   cache_info->l3_cache_size / (1024 * 1024),
                   cache_info->cache_line_size,
                   cache_info->l3_sharing_cores);
    
#else
    GGML_LOG_DEBUG("Cache detection not available on this platform, using defaults\n");
#endif
    
    return 0;
}

int64_t ggml_numa_optimal_tile_size(const struct ggml_numa_cache_info * cache_info, 
                                     int element_size, 
                                     int cache_level) {
    if (!cache_info || element_size <= 0) {
        return 64; // Safe default tile size
    }
    
    int64_t target_cache_size;
    switch (cache_level) {
        case 1: target_cache_size = cache_info->l1_cache_size; break;
        case 2: target_cache_size = cache_info->l2_cache_size; break;
        case 3: target_cache_size = cache_info->l3_cache_size; break;
        default: target_cache_size = cache_info->l2_cache_size; // Default to L2
    }
    
    // For matrix multiplication C = A * B, we need to fit:
    // - A tile: tile_size × tile_size × element_size
    // - B tile: tile_size × tile_size × element_size  
    // - C tile: tile_size × tile_size × element_size
    // Total: 3 × tile_size² × element_size
    //
    // We want: 3 × tile_size² × element_size ≤ target_cache_size
    // So: tile_size² ≤ target_cache_size / (3 × element_size)
    // Therefore: tile_size ≤ sqrt(target_cache_size / (3 × element_size))
    
    int64_t max_tile_elements = target_cache_size / (3 * element_size);
    int64_t tile_size = (int64_t)sqrt((double)max_tile_elements);
    
    // Round down to cache line boundary for better alignment
    int elements_per_cache_line = cache_info->cache_line_size / element_size;
    if (elements_per_cache_line > 0) {
        tile_size = (tile_size / elements_per_cache_line) * elements_per_cache_line;
    }
    
    // Ensure minimum and maximum reasonable sizes
    if (tile_size < 16) tile_size = 16;       // Minimum tile size
    if (tile_size > 2048) tile_size = 2048;   // Maximum tile size
    
    GGML_LOG_DEBUG("Optimal L%d tile size: %ld (cache=%ldKB, element_size=%d)\n", 
                   cache_level, tile_size, target_cache_size / 1024, element_size);
    
    return tile_size;
}

int64_t ggml_numa_cache_aware_chunk_size(const struct ggml_numa_cache_info * cache_info,
                                          int64_t matrix_dim,
                                          int batch_size,
                                          int element_size) {
    if (!cache_info || matrix_dim <= 0 || batch_size <= 0 || element_size <= 0) {
        return batch_size / 4; // Safe default: quarter batch size
    }
    
    // Calculate memory required for one matrix operation
    int64_t matrix_memory = matrix_dim * matrix_dim * element_size;
    int64_t batch_memory = matrix_memory * batch_size;
    GGML_UNUSED(batch_memory); // May be used for future batch optimization
    
    // Use L3 cache as the target for chunk sizing since it's shared across cores
    int64_t target_memory = cache_info->l3_cache_size;
    
    // We want chunks that can fit working set in L3 cache
    // Working set includes input matrices A, B and output matrix C
    // For batch processing: 3 × matrix_memory × chunk_size ≤ L3_cache_size
    int64_t max_chunk_size = target_memory / (3 * matrix_memory);
    
    if (max_chunk_size <= 0) {
        max_chunk_size = 1; // At least one item per chunk
    }
    if (max_chunk_size > batch_size) {
        max_chunk_size = batch_size; // Don't exceed batch size
    }
    
    // Align chunk size to L3 sharing boundaries for better cache utilization
    // If L3 is shared among N cores, try to make chunk_size divisible by N
    if (cache_info->l3_sharing_cores > 1 && max_chunk_size >= cache_info->l3_sharing_cores) {
        max_chunk_size = (max_chunk_size / cache_info->l3_sharing_cores) * cache_info->l3_sharing_cores;
    }
    
    if (max_chunk_size <= 0) {
        max_chunk_size = 1;
    }
    
    GGML_LOG_DEBUG("Cache-aware chunk size: %ld (matrix=%ldx%ld, batch=%d, L3=%ldMB)\n",
                   max_chunk_size, matrix_dim, matrix_dim, batch_size, 
                   cache_info->l3_cache_size / (1024 * 1024));
    
    return max_chunk_size;
}

/**
 * Get the NUMA nodes that the coordinator is actively using
 */
int ggml_numa_coordinator_get_active_nodes(struct ggml_numa_coordinator_manager * mgr, int * nodes, int max_nodes) {
    // If no manager specified, use the global singleton
    if (mgr == NULL) {
        mgr = g_global_coordinator_manager;
    }
    
    if (mgr == NULL || nodes == NULL || max_nodes <= 0) {
        return -1;
    }
    
    int count = 0;
    
    // Get the NUMA nodes from the coordinator threads
    for (int i = 0; i < mgr->num_numa_nodes && count < max_nodes; i++) {
        if (mgr->coordinators && mgr->coordinators[i].numa_node >= 0) {
            nodes[count++] = mgr->coordinators[i].numa_node;
        }
    }
    
    return count;
}

/**
 * Get the total number of NUMA nodes from the global coordinator manager
 * @return Number of NUMA nodes, or 1 if no coordinator is active
 */
int ggml_numa_coordinator_get_num_nodes(void) {
    if (g_global_coordinator_manager == NULL) {
        return 1; // Default to single node if no coordinator
    }
    return g_global_coordinator_manager->num_numa_nodes;
}

int ggml_numa_coordinator_manager_get_numa_nodes(struct ggml_numa_coordinator_manager * mgr) {
    if (!mgr) {
        return 1;  // Default to 1 if no manager available
    }
    return mgr->num_numa_nodes;
}

//
// Coordinator Interface Implementation
// These functions provide controlled access to coordinator resources for the dispatcher
//

struct ggml_threadpool * ggml_numa_coordinator_get_threadpool(struct ggml_numa_coordinator_manager * manager, int numa_node) {
    if (!manager || numa_node < 0 || numa_node >= manager->num_numa_nodes) {
        return NULL;
    }
    return manager->coordinators[numa_node].numa_pool;
}

int ggml_numa_coordinator_get_thread_count(struct ggml_numa_coordinator_manager * manager, int numa_node) {
    if (!manager || numa_node < 0 || numa_node >= manager->num_numa_nodes) {
        return -1;
    }
    return manager->coordinators[numa_node].n_threads;
}

bool ggml_numa_coordinator_ensure_work_buffer(struct ggml_numa_coordinator_manager * manager, int numa_node, size_t required_size) {
    if (!manager || numa_node < 0 || numa_node >= manager->num_numa_nodes) {
        return false;
    }
    // With per-thread work buffers, this function always succeeds
    // Individual threads will allocate their own partitioned buffers as needed
    return true;
}

void * ggml_numa_coordinator_get_work_buffer(struct ggml_numa_coordinator_manager * manager, int numa_node) {
    if (!manager || numa_node < 0 || numa_node >= manager->num_numa_nodes) {
        return NULL;
    }
    // With per-thread work buffers, return the current thread's buffer
    // This is mainly for legacy compatibility
    return ggml_numa_get_partitioned_work_buffer(0, numa_node, 1);
}

size_t ggml_numa_coordinator_get_work_buffer_size(struct ggml_numa_coordinator_manager * manager, int numa_node) {
    if (!manager || numa_node < 0 || numa_node >= manager->num_numa_nodes) {
        return 0;
    }
    // With per-thread work buffers, return the current thread's buffer size
    return ggml_thread_work_buffer_size;
}

enum ggml_status ggml_numa_coordinator_execute_graph_operation(
    struct ggml_numa_coordinator_manager * manager, 
    struct ggml_tensor * operation, 
    int numa_node) {
    
    if (!manager || !operation || numa_node < 0 || numa_node >= manager->num_numa_nodes) {
        return GGML_STATUS_FAILED;
    }
    
    struct ggml_coordinator_thread * coordinator = &manager->coordinators[numa_node];
    
    GGML_LOG_DEBUG("NUMA%d: Executing graph-based operation %s\n", 
                   numa_node, ggml_op_name(operation->op));
    
    // Create temporary context for graph computation
    struct ggml_context * temp_ctx = ggml_init((struct ggml_init_params) {
        .mem_size = 1024 * 1024, // 1MB should be sufficient for graph metadata
        .mem_buffer = NULL,
        .no_alloc = true, // Don't allocate tensor data, just metadata
    });
    
    if (!temp_ctx) {
        GGML_LOG_ERROR("NUMA%d: Failed to create temporary context for operation %s\n", 
                       numa_node, ggml_op_name(operation->op));
        return GGML_STATUS_FAILED;
    }
    
    // Create a computation graph containing just this operation
    struct ggml_cgraph * temp_graph = ggml_new_graph(temp_ctx);
    ggml_build_forward_expand(temp_graph, operation);
    
    // Create a computation plan using the coordinator's NUMA threadpool
    struct ggml_cplan cplan = ggml_graph_plan(temp_graph, coordinator->n_threads, coordinator->numa_pool);
    
    enum ggml_status status = GGML_STATUS_SUCCESS;
    
    // Ensure adequate work buffer if needed
    if (cplan.work_size > 0) {
        void* work_buffer = ggml_numa_get_partitioned_work_buffer(cplan.work_size, numa_node, 1);
        if (!work_buffer) {
            GGML_LOG_ERROR("NUMA%d: Failed to get partitioned work buffer of size %zu for operation %s\n", 
                          numa_node, cplan.work_size, ggml_op_name(operation->op));
            status = GGML_STATUS_FAILED;
        } else {
            cplan.work_data = work_buffer;
            GGML_LOG_DEBUG("NUMA%d: Using per-thread work buffer (%zu bytes) for %s\n", 
                           numa_node, cplan.work_size, ggml_op_name(operation->op));
        }
    }
    
    // Execute the graph using the computation plan
    if (status == GGML_STATUS_SUCCESS) {
        status = ggml_graph_compute(temp_graph, &cplan);
        
        if (status == GGML_STATUS_SUCCESS) {
            GGML_LOG_DEBUG("NUMA%d: Graph-based operation %s completed successfully\n", 
                           numa_node, ggml_op_name(operation->op));
        } else {
            GGML_LOG_ERROR("NUMA%d: Graph-based operation %s failed with status %d\n", 
                           numa_node, ggml_op_name(operation->op), status);
        }
    }
    
    // Clean up temporary context
    ggml_free(temp_ctx);
    
    return status;
}

// Reset work status for new work submission
void ggml_numa_coordinator_manager_reset_status(struct ggml_numa_coordinator_manager * mgr) {
    if (!mgr) return;
    
    ggml_mutex_lock(&mgr->status_mutex);
    atomic_store(&mgr->last_work_status, GGML_STATUS_SUCCESS);
    ggml_mutex_unlock(&mgr->status_mutex);
}

// ============================================================================
// Fallback Threadpool Implementation
// ============================================================================

struct ggml_threadpool * ggml_numa_coordinator_get_fallback_threadpool(struct ggml_numa_coordinator_manager * mgr) {
    if (!mgr) {
        // Use global singleton if no manager provided
        mgr = ggml_numa_coordinator_manager_get_existing();
        if (!mgr) {
            GGML_LOG_WARN("No NUMA coordinator manager available for fallback threadpool\n");
            return NULL;
        }
    }
    
    return mgr->fallback_threadpool;
}

int ggml_numa_coordinator_get_fallback_thread_count(struct ggml_numa_coordinator_manager * mgr) {
    if (!mgr) {
        // Use global singleton if no manager provided
        mgr = ggml_numa_coordinator_manager_get_existing();
        if (!mgr) {
            GGML_LOG_WARN("No NUMA coordinator manager available for fallback thread count\n");
            return 1;  // Default to single thread
        }
    }
    
    return mgr->fallback_thread_count > 0 ? mgr->fallback_thread_count : 1;
}

