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
#include "ggml-cpu-impl.h"  // For ggml_compute_params structure
#include "ops.h"  // For ggml_compute_forward_* functions
#include "binary-ops.h"  // For binary operation functions

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <stdatomic.h>
#include <pthread.h>
#include <time.h>
#include <math.h>

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
    struct ggml_tensor * operation;     // Complete operation (not chunk)
    int assigned_numa_node;             // Target NUMA node
    struct ggml_tensor ** dependencies; // Operations to wait for
    int num_dependencies;               // Dependency count
    atomic_bool dependencies_ready;     // All dependencies completed
    atomic_bool completed;              // This operation completed
    struct ggml_work_item * next;       // Next item in queue
    int work_id;                        // Unique work ID for tracking
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
    
    // Synchronization for waiting on group completion
    ggml_mutex_t completion_mutex;  // Mutex for completion waiting
    ggml_cond_t completion_cond;    // Condition variable for completion
};

// Work group tracking in manager
struct ggml_work_group_tracker {
    struct ggml_work_group ** groups;  // Array of active work groups
    int max_groups;                    // Maximum number of groups
    atomic_int next_group_id;          // Next group ID to assign
    ggml_mutex_t groups_mutex;         // Mutex for group operations
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
    struct ggml_cgraph * numa_cgraph;               // NUMA node's own copy of cgraph
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
    ggml_mutex_t strategy_mutex;                          // Mutex for strategy changes
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

// Work group management functions
static void ggml_work_group_tracker_init(struct ggml_work_group_tracker * tracker);
static void ggml_work_group_tracker_destroy(struct ggml_work_group_tracker * tracker);
static struct ggml_work_group * ggml_work_group_create(struct ggml_work_group_tracker * tracker, struct ggml_tensor * tensor, int num_chunks);
static void ggml_work_group_free(struct ggml_work_group * group);
static int ggml_work_group_check_completion(struct ggml_work_group * group);
// static int ggml_operation_split_for_numa(struct ggml_tensor * tensor, int num_numa_nodes, struct ggml_work_item *** out_chunks); // Unused for now

// Global coordinator management functions
static struct ggml_numa_coordinator_manager * ggml_get_global_coordinator_manager(int n_threads, bool force_multi_socket);
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
    struct ggml_tensor * operation
);

static bool ggml_numa_should_coordinate(
    struct ggml_cgraph * cgraph,
    int n_threads
);

// Operation-level NUMA parallelization using proper GGML compute functions
static struct ggml_numa_coordinator_manager * ggml_get_global_coordinator_manager(int n_threads, bool force_multi_socket);
static struct ggml_numa_coordinator_manager * ggml_get_global_coordinator_manager_with_params(const struct ggml_threadpool_params * tpp);
static void ggml_register_program_exit_cleanup(void);

// Operation-specific NUMA parallelization using proper GGML compute functions
// Execute complete operations on NUMA nodes using GGML's optimized functions
static enum ggml_status ggml_numa_node_execute_operation(
    struct ggml_coordinator_thread * coordinator,
    struct ggml_tensor * operation
) {
    if (!coordinator || !operation) return GGML_STATUS_FAILED;
    
    GGML_LOG_DEBUG("NUMA%d: executing complete operation %s\n", 
                   coordinator->numa_node, ggml_op_name(operation->op));
    
    // Set up compute parameters for this NUMA node
    struct ggml_compute_params params = {
        .ith = 0,
        .nth = coordinator->n_threads,  
        .wsize = 0,
        .wdata = NULL,
        .threadpool = coordinator->numa_pool  
    };
    
    // PROPER FIX: Execute MUL_MAT operations with full multi-threaded parallelization
    // Instead of single-threaded workaround, use the NUMA threadpool's parallel execution
    if (operation->op == GGML_OP_MUL_MAT) {
        GGML_LOG_DEBUG("NUMA%d: MUL_MAT operation - using full threadpool parallelization\n", 
                       coordinator->numa_node);
        
        // Use ggml_graph_compute with a single-node graph for this operation
        // This leverages GGML's existing parallel execution infrastructure
        struct ggml_context * temp_ctx = ggml_init((struct ggml_init_params) {
            .mem_size = 1024 * 1024, // 1MB should be sufficient for graph metadata
            .mem_buffer = NULL,
            .no_alloc = true, // Don't allocate tensor data, just metadata
        });
        
        if (!temp_ctx) {
            GGML_LOG_ERROR("NUMA%d: Failed to create temporary context for MUL_MAT\n", coordinator->numa_node);
            return GGML_STATUS_FAILED;
        }
        
        // Create a computation graph containing just this operation
        struct ggml_cgraph * temp_graph = ggml_new_graph(temp_ctx);
        ggml_build_forward_expand(temp_graph, operation);
        
        // Create a computation plan using the coordinator's NUMA threadpool
        struct ggml_cplan cplan = ggml_graph_plan(temp_graph, coordinator->n_threads, coordinator->numa_pool);
        
        // Execute the graph using the computation plan
        enum ggml_status status = ggml_graph_compute(temp_graph, &cplan);
        
        // Clean up temporary context
        ggml_free(temp_ctx);
        
        if (status == GGML_STATUS_SUCCESS) {
            GGML_LOG_DEBUG("NUMA%d: MUL_MAT operation completed with full parallelization\n", coordinator->numa_node);
        } else {
            GGML_LOG_ERROR("NUMA%d: MUL_MAT operation failed with status %d\n", coordinator->numa_node, status);
        }
        
        return status;
    }
    
    // Use GGML's optimized compute functions directly - no chunking
    enum ggml_status status = GGML_STATUS_SUCCESS;
    
    switch (operation->op) {
        // Basic arithmetic operations (high frequency, good NUMA candidates)
        case GGML_OP_ADD:
            ggml_compute_forward_add(&params, operation);
            break;
        case GGML_OP_ADD1:
            ggml_compute_forward_add1(&params, operation);
            break;
        case GGML_OP_SUB:
            ggml_compute_forward_sub(&params, operation);
            break;
        case GGML_OP_MUL:
            ggml_compute_forward_mul(&params, operation);
            break;
        case GGML_OP_DIV:
            ggml_compute_forward_div(&params, operation);
            break;
            
        // Matrix operations (highest impact for NUMA coordination)
        case GGML_OP_MUL_MAT:
            ggml_compute_forward_mul_mat(&params, operation);
            break;
        case GGML_OP_OUT_PROD:
            ggml_compute_forward_out_prod(&params, operation);
            break;
            
        // Normalization operations (common in transformers, memory-intensive)
        case GGML_OP_NORM:
            ggml_compute_forward_norm(&params, operation);
            break;
        case GGML_OP_RMS_NORM:
            ggml_compute_forward_rms_norm(&params, operation);
            break;
        case GGML_OP_RMS_NORM_BACK:
            ggml_compute_forward_rms_norm_back(&params, operation);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_compute_forward_group_norm(&params, operation);
            break;
        case GGML_OP_L2_NORM:
            ggml_compute_forward_l2_norm(&params, operation);
            break;
            
        // Activation functions (frequent in neural networks)
        case GGML_OP_UNARY:
            ggml_compute_forward_unary(&params, operation);
            break;
        case GGML_OP_SILU_BACK:
            ggml_compute_forward_silu_back(&params, operation);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_compute_forward_leaky_relu(&params, operation);
            break;
            
        // Attention and transformer-specific operations
        case GGML_OP_SOFT_MAX:
            ggml_compute_forward_soft_max(&params, operation);
            break;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_compute_forward_soft_max_ext_back(&params, operation);
            break;
        case GGML_OP_ROPE:
            ggml_compute_forward_rope(&params, operation);
            break;
        case GGML_OP_ROPE_BACK:
            ggml_compute_forward_rope_back(&params, operation);
            break;
            
        // Tensor manipulation (moderate impact, but common)
        case GGML_OP_CPY:
            ggml_compute_forward_cpy(&params, operation);
            break;
        case GGML_OP_DUP:
            ggml_compute_forward_dup(&params, operation);
            break;
        case GGML_OP_CONT:
            ggml_compute_forward_cont(&params, operation);
            break;
        case GGML_OP_RESHAPE:
            ggml_compute_forward_reshape(&params, operation);
            break;
        case GGML_OP_PERMUTE:
            ggml_compute_forward_permute(&params, operation);
            break;
        case GGML_OP_TRANSPOSE:
            ggml_compute_forward_transpose(&params, operation);
            break;
        case GGML_OP_VIEW:
            ggml_compute_forward_view(&params, operation);
            break;
        case GGML_OP_SCALE:
            ggml_compute_forward_scale(&params, operation);
            break;
        case GGML_OP_SET:
            ggml_compute_forward_set(&params, operation);
            break;
            
        // Row/tensor operations  
        case GGML_OP_GET_ROWS:
            ggml_compute_forward_get_rows(&params, operation);
            break;
        case GGML_OP_GET_ROWS_BACK:
            ggml_compute_forward_get_rows_back(&params, operation);
            break;
        case GGML_OP_SET_ROWS:
            ggml_compute_forward_set_rows(&params, operation);
            break;
            
        // Aggregation operations
        case GGML_OP_SUM:
            ggml_compute_forward_sum(&params, operation);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_compute_forward_sum_rows(&params, operation);
            break;
        case GGML_OP_MEAN:
            ggml_compute_forward_mean(&params, operation);
            break;
        case GGML_OP_ARGMAX:
            ggml_compute_forward_argmax(&params, operation);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_compute_forward_count_equal(&params, operation);
            break;
            
        // Convolution operations (high computational cost when present)
        case GGML_OP_CONV_2D:
            ggml_compute_forward_conv_2d(&params, operation);
            break;
        case GGML_OP_IM2COL:
            ggml_compute_forward_im2col(&params, operation);
            break;
        case GGML_OP_IM2COL_BACK:
            ggml_compute_forward_im2col_back_f32(&params, operation);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_compute_forward_conv_transpose_1d(&params, operation);
            break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            ggml_compute_forward_conv_transpose_2d(&params, operation);
            break;
        case GGML_OP_POOL_1D:
            ggml_compute_forward_pool_1d(&params, operation);
            break;
        case GGML_OP_POOL_2D:
            ggml_compute_forward_pool_2d(&params, operation);
            break;
            
        // Miscellaneous operations
        case GGML_OP_REPEAT:
            ggml_compute_forward_repeat(&params, operation);
            break;
        case GGML_OP_REPEAT_BACK:
            ggml_compute_forward_repeat_back(&params, operation);
            break;
        case GGML_OP_CONCAT:
            ggml_compute_forward_concat(&params, operation);
            break;
        case GGML_OP_CLAMP:
            ggml_compute_forward_clamp(&params, operation);
            break;
            
        default:
            GGML_LOG_DEBUG("NUMA%d: Operation %s using public fallback execution\n", 
                          coordinator->numa_node, ggml_op_name(operation->op));
            // Use the public fallback function for operations not specifically handled
            // This ensures operations are properly executed rather than silently skipped
            status = ggml_numa_fallback_execute_operation(operation, &params);
            if (status != GGML_STATUS_SUCCESS) {
                GGML_LOG_WARN("NUMA%d: Fallback execution failed for operation %s\n",
                             coordinator->numa_node, ggml_op_name(operation->op));
            }
            break;
    }
    
    if (status == GGML_STATUS_SUCCESS) {
        GGML_LOG_DEBUG("NUMA%d: Successfully executed operation %s\n", 
                       coordinator->numa_node, ggml_op_name(operation->op));
    } else {
        GGML_LOG_ERROR("NUMA%d: Failed to execute operation %s\n", 
                       coordinator->numa_node, ggml_op_name(operation->op));
    }
    
    return status;
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
    }
    
    ggml_mutex_unlock(&queue->queue_mutex);
    return item;
}

// Initialize work group tracker
static void ggml_work_group_tracker_init(struct ggml_work_group_tracker * tracker) {
    tracker->max_groups = 64; // Support up to 64 concurrent work groups
    tracker->groups = malloc(sizeof(struct ggml_work_group *) * tracker->max_groups);
    memset(tracker->groups, 0, sizeof(struct ggml_work_group *) * tracker->max_groups);
    atomic_init(&tracker->next_group_id, 1);
    ggml_mutex_init(&tracker->groups_mutex);
}

// Destroy work group tracker
static void ggml_work_group_tracker_destroy(struct ggml_work_group_tracker * tracker) {
    if (!tracker || !tracker->groups) return;
    
    ggml_mutex_lock(&tracker->groups_mutex);
    
    // Clean up any remaining work groups
    for (int i = 0; i < tracker->max_groups; i++) {
        if (tracker->groups[i]) {
            ggml_work_group_free(tracker->groups[i]);
            tracker->groups[i] = NULL;
        }
    }
    
    ggml_mutex_unlock(&tracker->groups_mutex);
    ggml_mutex_destroy(&tracker->groups_mutex);
    
    free(tracker->groups);
    tracker->groups = NULL;
}

// Create new work group
static struct ggml_work_group * ggml_work_group_create(struct ggml_work_group_tracker * tracker, struct ggml_tensor * tensor, int num_chunks) {
    if (!tracker || !tensor || num_chunks <= 0) return NULL;
    
    struct ggml_work_group * group = malloc(sizeof(struct ggml_work_group));
    if (!group) return NULL;
    
    memset(group, 0, sizeof(struct ggml_work_group));
    group->group_id = atomic_fetch_add(&tracker->next_group_id, 1);
    group->original_tensor = tensor;
    group->num_chunks = num_chunks;
    atomic_init(&group->completed_chunks, 0);
    atomic_init(&group->group_completed, false);
    group->result_tensor = NULL;
    group->split_dimension = 0; // Default to row-wise split
    
    // Initialize synchronization primitives
    ggml_mutex_init(&group->completion_mutex);
    ggml_cond_init(&group->completion_cond);
    
    // Allocate chunks array
    group->chunks = malloc(sizeof(struct ggml_work_item *) * num_chunks);
    if (!group->chunks) {
        free(group);
        return NULL;
    }
    memset(group->chunks, 0, sizeof(struct ggml_work_item *) * num_chunks);
    
    // Find empty slot in tracker and store group
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
        GGML_LOG_ERROR("Work group tracker full, cannot create new group\n");
        free(group->chunks);
        free(group);
        return NULL;
    }
    
    GGML_LOG_DEBUG("Created work group %d with %d chunks for tensor %p\n", group->group_id, num_chunks, (void*)tensor);
    return group;
}

// Free work group
static void ggml_work_group_free(struct ggml_work_group * group) {
    if (!group) return;
    
    GGML_LOG_DEBUG("Freeing work group %d\n", group->group_id);
    
    // Free chunks (work items are freed by coordinator threads)
    if (group->chunks) {
        free(group->chunks);
    }
    
    // Clean up synchronization primitives
    ggml_mutex_destroy(&group->completion_mutex);
    ggml_cond_destroy(&group->completion_cond);
    
    // Note: We don't free result_tensor as it's typically owned by the caller
    // Note: We don't free original_tensor as it's owned by the caller
    
    free(group);
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
    
    atomic_store(&coordinator->active, true);
    
    // Main coordinator loop - processes complete operations 
    while (!atomic_load(&coordinator->shutdown_requested)) {
        struct ggml_work_item * work_item = ggml_work_queue_dequeue(&coordinator->work_queue);
        
        if (!work_item) {
            break; // Shutdown requested
        }
        
        int64_t start_time = ggml_time_us();
        
        // Execute complete operation using graph-level approach
        enum ggml_status status = GGML_STATUS_FAILED;
        
        if (work_item->operation) {
            // Execute the complete operation assigned to this NUMA node
            status = ggml_numa_node_execute_operation(coordinator, work_item->operation);
        } else {
            GGML_LOG_ERROR("NUMA%d: Work item has no operation tensor\n", coordinator->numa_node);
        }
        
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
        
        if (status != GGML_STATUS_SUCCESS) {
            GGML_LOG_WARN("Coordinator NUMA%d: Operation %s failed with status %d\n",
                         coordinator->numa_node, 
                         work_item->operation ? ggml_op_name(work_item->operation->op) : "unknown",
                         status);
        } else {
            GGML_LOG_DEBUG("Coordinator NUMA%d: Operation %s completed successfully\n",
                          coordinator->numa_node,
                          work_item->operation ? ggml_op_name(work_item->operation->op) : "unknown");
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
        
        // Free the work item
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

// Get or create the global singleton coordinator manager with parameters
static struct ggml_numa_coordinator_manager * ggml_get_global_coordinator_manager_with_params(const struct ggml_threadpool_params * tpp) {
    // Thread-safe singleton initialization
    ggml_mutex_lock(&g_coordinator_init_mutex);
    
    if (g_global_coordinator_manager == NULL) {
        GGML_LOG_INFO("Creating global singleton 3-tier NUMA coordinator manager with custom parameters\n");
        g_global_coordinator_manager = ggml_numa_coordinator_manager_new_with_params(tpp);
        
        if (g_global_coordinator_manager) {
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

// Helper function to create hyperthreading-optimized CPU masks
// This assigns CPUs to avoid conflicts on the same physical core
static void create_optimal_cpu_masks(struct ggml_threadpool_params *tpp, int num_numa_nodes) {
    if (!tpp || num_numa_nodes < 2) return;
    
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
        GGML_LOG_INFO("   Target: %d total threads across %d NUMA nodes\n", tpp->n_threads, num_numa_nodes);
        
        // Clear the mask first
        memset(tpp->cpumask, false, sizeof(tpp->cpumask));
        cpu_count = 0;
        
#ifdef __linux__
        // Use real NUMA topology instead of hardcoded assumptions
        int total_cpus = numa_num_configured_cpus();
        int max_cpu_num = total_cpus - 1; // numa_num_configured_cpus() returns count, not max number
        
        // For systems with gaps in CPU numbering, we need to scan higher
        // Your system: 112 CPUs numbered 0-27,56-83,28-55,84-111  
        // So max CPU number is 111, but total_cpus is 112
        int cpu_scan_limit = GGML_MAX_N_THREADS;
        
#ifdef __linux__
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
#endif
        
        GGML_LOG_INFO("   CPU scanning range: 0 to %d (total CPUs: %d, max CPU number: %d)\n", 
                     cpu_scan_limit - 1, total_cpus, max_cpu_found);
        int threads_per_node = tpp->n_threads / num_numa_nodes;
        
        GGML_LOG_INFO("Distributing %d threads across %d NUMA nodes (%d per node)\n", 
                     tpp->n_threads, num_numa_nodes, threads_per_node);
        
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
                int target_threads = (node == num_numa_nodes - 1) ? 
                    (tpp->n_threads - (threads_per_node * (num_numa_nodes - 1))) : threads_per_node;
                
                GGML_LOG_INFO("NUMA node %d: has %d CPUs, assigning %d threads\n", 
                             node, node_cpu_count, target_threads);
                
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
                for (int i = 0; i < primary_count && assigned < target_threads; i++) {
                    int cpu = primary_cpus[i];
                    tpp->cpumask[cpu] = true;
                    available_cpus[cpu_count++] = cpu;
                    assigned++;
                    GGML_LOG_INFO("   Assigned primary CPU %d to node %d\n", cpu, node);
                }
                
                // Second pass: assign hyperthreads if needed
                for (int i = 0; i < hyperthread_count && assigned < target_threads; i++) {
                    int cpu = hyperthread_cpus[i];
                    tpp->cpumask[cpu] = true;
                    available_cpus[cpu_count++] = cpu;
                    assigned++;
                    GGML_LOG_INFO("   Assigned hyperthread CPU %d to node %d\n", cpu, node);
                }
            }
            numa_free_cpumask(node_cpus);
        }
#else
        // Fallback for non-Linux systems
        GGML_LOG_WARN("Non-Linux system: using simple round-robin CPU assignment\n");
        for (int i = 0; i < tpp->n_threads && i < GGML_MAX_N_THREADS; i++) {
            tpp->cpumask[i] = true;
            available_cpus[cpu_count++] = i;
        }
#endif
        
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

// Create NUMA coordinator manager with 3-tier architecture
struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_new(int n_threads, bool force_multi_socket) {
    // Create basic threadpool parameters and delegate to new function
    struct ggml_threadpool_params tpp;
    ggml_threadpool_params_init(&tpp, n_threads);
    tpp.force_multi_socket = force_multi_socket;
    
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
    
#ifdef __linux__
    if (numa_available() != -1) {
        num_numa_nodes = numa_max_node() + 1;
        numa_is_available = true;
    }
#endif
    
    if (tpp->force_multi_socket && !numa_is_available) {
        num_numa_nodes = 2; // Simulate 2 NUMA nodes for testing
        GGML_LOG_INFO("Forcing multi-socket mode with %d simulated NUMA nodes\n", num_numa_nodes);
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
    
    // Initialize progress callback system
    mgr->progress_callback = NULL;
    mgr->progress_callback_user_data = NULL;
    
    // Initialize memory management strategy
    mgr->memory_strategy = GGML_NUMA_STRATEGY_AUTO;  // Default to adaptive strategy
    ggml_mutex_init(&mgr->strategy_mutex);
    
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
    
    int threads_per_numa = tpp->n_threads / num_numa_nodes;
    if (threads_per_numa < 1) threads_per_numa = 1;
    
    GGML_LOG_INFO("Creating NUMA coordinator with %d threads distributed across %d NUMA nodes (%d threads per node)\n", 
                  tpp->n_threads, num_numa_nodes, threads_per_numa);
    
    // Step 3.5: Create optimal CPU masks to avoid hyperthreading conflicts
    struct ggml_threadpool_params optimized_tpp = *tpp;  // Copy original parameters
    if (num_numa_nodes > 1) {
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
        numa_tpp.force_multi_socket = false; // Don't create nested coordinators
        
        // Step 5: Apply NUMA-specific CPU mask filtering
        int real_numa_nodes = 1;
#ifdef __linux__
        if (numa_is_available) {
            real_numa_nodes = numa_max_node() + 1;
        }
#endif
        
        if (numa_is_available && i < real_numa_nodes) {
            GGML_LOG_INFO("   Processing NUMA node %d in REAL NUMA mode (hardware node exists)\n", i);
#ifdef __linux__
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
#endif
        } else {
            // Simulated NUMA (when i >= real_numa_nodes) or no NUMA: distribute CPUs evenly
            GGML_LOG_INFO("   Processing NUMA node %d in SIMULATED NUMA mode (real nodes: %d)\n", i, real_numa_nodes);
            bool has_cpu_mask = false;
            for (int cpu = 0; cpu < GGML_MAX_N_THREADS; cpu++) {
                if (optimized_tpp.cpumask[cpu]) {
                    has_cpu_mask = true;
                    break;
                }
            }
            
            if (has_cpu_mask) {
                // Count available CPUs in optimized mask
                int available_cpus[GGML_MAX_N_THREADS];
                int cpu_count = 0;
                
                for (int cpu = 0; cpu < GGML_MAX_N_THREADS; cpu++) {
                    if (optimized_tpp.cpumask[cpu]) {
                        available_cpus[cpu_count++] = cpu;
                    }
                }
                
                // Distribute CPUs round-robin across NUMA nodes
                memset(numa_tpp.cpumask, false, sizeof(numa_tpp.cpumask));
                bool has_assigned_cpus = false;
                int assigned_cpu_list[GGML_MAX_N_THREADS];
                int assigned_count = 0;
                
                for (int j = 0; j < cpu_count; j++) {
                    if ((j % num_numa_nodes) == i) {
                        numa_tpp.cpumask[available_cpus[j]] = true;
                        assigned_cpu_list[assigned_count++] = available_cpus[j];
                        has_assigned_cpus = true;
                    }
                }
                
                // Enhanced CPU assignment logging
                if (has_assigned_cpus) {
                    char cpu_list_str[512] = {0};
                    char hyperthreading_analysis[512] = {0};
                    int pos = 0;
                    int ht_conflicts = 0;
                    GGML_UNUSED(hyperthreading_analysis); // May not be used in all debug builds
                    
                    // Build CPU list string and analyze hyperthreading
                    for (int k = 0; k < assigned_count; k++) {
                        int cpu = assigned_cpu_list[k];
                        int physical_core = cpu / 2;  // Intel HT pattern: CPU 0,1->Core 0, CPU 2,3->Core 1
                        bool is_ht_sibling = (cpu % 2 == 1);
                        
                        pos += snprintf(cpu_list_str + pos, sizeof(cpu_list_str) - pos, 
                                       "%sCPU%d(Core%d%s)", 
                                       k > 0 ? "," : "", cpu, physical_core, 
                                       is_ht_sibling ? "-HT" : "");
                        
                        // Check for hyperthreading conflicts with other assigned CPUs
                        for (int l = k + 1; l < assigned_count; l++) {
                            int other_cpu = assigned_cpu_list[l];
                            if (cpu / 2 == other_cpu / 2) {  // Same physical core
                                ht_conflicts++;
                                break;
                            }
                        }
                    }
                    
                    GGML_LOG_INFO("NUMA node %d: assigned %d CPUs [%s] via round-robin\n", 
                                  i, assigned_count, cpu_list_str);
                    
                    if (ht_conflicts > 0) {
                        GGML_LOG_WARN("    NUMA node %d: WARNING - %d hyperthreading conflicts detected - may reduce performance\n", 
                                      i, ht_conflicts);
                    } else {
                        GGML_LOG_INFO("    NUMA node %d: OK - No hyperthreading conflicts - optimal CPU assignment\n", i);
                    }
                } else {
                    GGML_LOG_INFO("NUMA node %d: no specific CPUs assigned - using default affinity\n", i);
                }
            }
        }
        
        // Step 6: Create NUMA threadpool with filtered parameters
        coord->numa_pool = ggml_threadpool_new(&numa_tpp);
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
        
        // Step 7: Initialize work queue for this coordinator
        ggml_work_queue_init(&coord->work_queue);
        
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
    GGML_LOG_INFO("    Total threads distributed: %d (avg %d per node)\n", 
                  tpp->n_threads, tpp->n_threads / num_numa_nodes);
    
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
                      threads_per_numa);
    }
    GGML_LOG_INFO("================================================================================\n");
    
    return mgr;
}

// Get the global singleton coordinator manager (create if needed)
struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_get_global(int n_threads, bool force_multi_socket) {
    return ggml_get_global_coordinator_manager(n_threads, force_multi_socket);
}

// Get the global singleton coordinator manager with parameters (create if needed)
struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_get_global_with_params(const struct ggml_threadpool_params * tpp) {
    return ggml_get_global_coordinator_manager_with_params(tpp);
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
            ggml_thread_join(coord->thread_handle, NULL);
        }
        
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
    ggml_work_group_tracker_destroy(&mgr->work_groups);
    ggml_work_queue_destroy(&mgr->global_work_queue);
    ggml_cond_destroy(&mgr->main_sync_cond);
    ggml_mutex_destroy(&mgr->main_sync_mutex);
    ggml_mutex_destroy(&mgr->strategy_mutex);
    
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
        
        // Store reference to the original cgraph (properly handle const)
        // The coordinator threads only read the cgraph, so the const cast is safe
        coord->numa_cgraph = (struct ggml_cgraph *)master_cgraph; 
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
    
    // Ensure coordinator threads are started and have a cgraph before submitting work
    int start_result = ggml_numa_coordinator_manager_start(mgr);
    if (start_result != 0) {
        GGML_LOG_ERROR("Failed to start coordinator threads\n");
        return -1;
    }
    
    // Determine target NUMA node
    int target_numa = numa_node_hint;
    if (target_numa < 0 || target_numa >= mgr->num_numa_nodes) {
        target_numa = atomic_load(&mgr->total_work_items) % mgr->num_numa_nodes; // Round-robin
    }
    
    // Create work item for graph-level operation
    struct ggml_work_item * work_item = malloc(sizeof(struct ggml_work_item));
    if (!work_item) return -1;
    
    // Set up graph-level work item fields
    work_item->operation = tensor;           // Complete operation to execute
    work_item->assigned_numa_node = target_numa;
    work_item->dependencies = NULL;          // No dependencies for now
    work_item->num_dependencies = 0;
    atomic_init(&work_item->dependencies_ready, true); // Ready to execute
    atomic_init(&work_item->completed, false);
    work_item->next = NULL;
    work_item->work_id = atomic_fetch_add(&mgr->total_work_items, 1);
    
    // Enqueue to target coordinator
    ggml_work_queue_enqueue(&mgr->coordinators[target_numa].work_queue, work_item);
    
    return work_item->work_id;
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
            ggml_work_group_free(group);
            return -1;
        }
        
        // Set up work item for complete operation (simplified from chunk-based)
        work_item->operation = tensor;           // Each NUMA node processes complete operation
        work_item->assigned_numa_node = i;       // Assign to specific NUMA node
        work_item->dependencies = NULL;
        work_item->num_dependencies = 0;
        atomic_init(&work_item->dependencies_ready, true);
        atomic_init(&work_item->completed, false);
        work_item->next = NULL;
        work_item->work_id = atomic_fetch_add(&mgr->total_work_items, 1);
        
        // Store work item in group
        group->chunks[i] = work_item;
        
        // Submit work item to assigned NUMA node
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
    // Use data parallelism for suitable operations when multiple NUMA nodes are available
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if (!node) continue;
        
        // Analyze the operation to determine if data parallelism is beneficial
        bool use_data_parallel = false;
        if (mgr->num_numa_nodes > 1) {
            // Check if this operation can benefit from data parallelism
            switch (node->op) {
                case GGML_OP_MUL_MAT:
                case GGML_OP_MUL_MAT_ID:
                    // Matrix multiplication - excellent candidate for row-wise splitting
                    if (ggml_nelements(node) > 50000) { // Only for large matrices
                        use_data_parallel = true;
                        GGML_LOG_DEBUG("Node %d: Matrix operation (%s) using data parallelism (%ld elements)\n", 
                                       i, ggml_op_name(node->op), ggml_nelements(node));
                    }
                    break;
                    
                case GGML_OP_ADD:
                case GGML_OP_MUL:
                case GGML_OP_DIV:
                case GGML_OP_SUB:
                    // Element-wise operations - good for parallelism on large tensors
                    if (ggml_nelements(node) > 100000) {
                        use_data_parallel = true;
                        GGML_LOG_DEBUG("Node %d: Element-wise operation (%s) using data parallelism (%ld elements)\n", 
                                       i, ggml_op_name(node->op), ggml_nelements(node));
                    }
                    break;
                    
                case GGML_OP_UNARY:
                    // Unary operations (GELU, SILU, RELU, etc.) - good for parallelism on large tensors
                    if (ggml_nelements(node) > 50000) {
                        use_data_parallel = true;
                        GGML_LOG_DEBUG("Node %d: Unary operation (%s) using data parallelism (%ld elements)\n", 
                                       i, ggml_op_name(node->op), ggml_nelements(node));
                    }
                    break;
                    
                default: {
                    // For other operations, use simple size heuristic
                    int64_t tensor_elements = ggml_nelements(node);
                    int64_t tensor_bytes = ggml_nbytes(node);
                    int64_t min_elements_per_numa = 10000;
                    int64_t min_bytes_per_numa = 1024 * 1024; // 1MB
                    
                    if (tensor_elements > (min_elements_per_numa * mgr->num_numa_nodes) ||
                        tensor_bytes > (min_bytes_per_numa * mgr->num_numa_nodes)) {
                        use_data_parallel = true;
                        GGML_LOG_DEBUG("Node %d: Large tensor operation (%s) using data parallelism (%ld elements, %ld bytes)\n", 
                                       i, ggml_op_name(node->op), tensor_elements, tensor_bytes);
                    } else {
                        GGML_LOG_DEBUG("Node %d: Small operation (%s) using single-node processing (%ld elements, %ld bytes)\n", 
                                       i, ggml_op_name(node->op), tensor_elements, tensor_bytes);
                    }
                    break;
                }
            }
        }
        
        if (use_data_parallel) {
            // Submit with data parallelism
            int work_group_id = ggml_numa_coordinator_manager_submit_data_parallel_work(mgr, node);
            if (work_group_id < 0) {
                GGML_LOG_WARN("Failed to submit data parallel work for cgraph node %d, falling back to single-node\n", i);
                // Fallback to single-node processing
                int work_id = ggml_numa_coordinator_manager_submit_work(mgr, node, -1);
                if (work_id < 0) {
                    GGML_LOG_ERROR("Failed to submit work for cgraph node %d\n", i);
                    return -1;
                }
            } else {
                // Wait for data parallel work group to complete
                result = ggml_numa_coordinator_manager_wait_for_work_group(mgr, work_group_id);
                if (result != 0) {
                    GGML_LOG_ERROR("Data parallel work group %d failed for cgraph node %d\n", work_group_id, i);
                    return -1;
                }
            }
        } else {
            // Submit to single NUMA node
            int work_id = ggml_numa_coordinator_manager_submit_work(mgr, node, -1);
            if (work_id < 0) {
                GGML_LOG_WARN("Failed to submit work for cgraph node %d\n", i);
                return -1;
            }
        }
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
int ggml_numa_coordinator_manager_wait_for_completion(struct ggml_numa_coordinator_manager * mgr) {
    if (!mgr) return -1;
    
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
        
        // Wait on condition variable instead of sleeping
        // The coordinator threads will signal this condition when they complete work
        ggml_cond_wait(&mgr->main_sync_cond, &mgr->main_sync_mutex);
    }
    
    ggml_mutex_unlock(&mgr->main_sync_mutex);
    
    GGML_LOG_DEBUG("All work completed\n");
    return 0;
}

// Wait for a specific work group to complete (used for data parallel work)
int ggml_numa_coordinator_manager_wait_for_work_group(struct ggml_numa_coordinator_manager * mgr, int work_group_id) {
    if (!mgr || work_group_id <= 0) return -1;
    
    GGML_LOG_DEBUG("Waiting for work group %d to complete\n", work_group_id);
    
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
        GGML_LOG_ERROR("Work group %d not found\n", work_group_id);
        return -1;
    }
    
    // Wait for completion and integration using condition variable
    ggml_mutex_lock(&target_group->completion_mutex);
    
    while (!atomic_load(&target_group->group_completed)) {
        // Check if all chunks are completed and trigger integration if needed
        int completion_status = ggml_work_group_check_completion(target_group);
        if (completion_status == 1) {
            // Successfully completed and integrated
            break;
        } else if (completion_status == -1) {
            // Integration failed
            GGML_LOG_ERROR("Work group %d integration failed\n", work_group_id);
            ggml_mutex_unlock(&target_group->completion_mutex);
            return -1;
        }
        
        // Wait on condition variable instead of sleeping
        ggml_cond_wait(&target_group->completion_cond, &target_group->completion_mutex);
    }
    
    ggml_mutex_unlock(&target_group->completion_mutex);
    
    GGML_LOG_INFO("Work group %d completed successfully\n", work_group_id);
    
    // Clean up completed work group
    ggml_mutex_lock(&mgr->work_groups.groups_mutex);
    for (int i = 0; i < mgr->work_groups.max_groups; i++) {
        if (mgr->work_groups.groups[i] && mgr->work_groups.groups[i]->group_id == work_group_id) {
            ggml_work_group_free(mgr->work_groups.groups[i]);
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
        work_item->operation = assignment->operation;
        work_item->assigned_numa_node = assignment->assigned_numa_node;
        work_item->dependencies = NULL; // Simple for now - can add dependency tracking later
        work_item->num_dependencies = 0;
        atomic_init(&work_item->dependencies_ready, true); // Ready to execute
        atomic_init(&work_item->completed, false);
        work_item->next = NULL;
        work_item->work_id = i; // Use operation index as work ID
        
        // Submit to the assigned NUMA node
        ggml_work_queue_enqueue(&mgr->coordinators[assignment->assigned_numa_node].work_queue, work_item);
        
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

//
// GGML Integration Function - Main entry point from ggml-cpu.c
//

/**
 * Main GGML integration function - replaces standard graph computation with NUMA-aware version
 * This is the primary integration point that ggml-cpu.c calls instead of standard ggml_graph_compute
 * 
 * @param cgraph Computation graph to execute
 * @param n_threads Number of threads (used to determine if NUMA coordination is beneficial)
 * @return GGML_STATUS_SUCCESS on success, GGML_STATUS_FAILED on failure
 */
enum ggml_status ggml_numa_graph_compute(struct ggml_cgraph * cgraph, int n_threads) {
    if (!cgraph) {
        GGML_LOG_ERROR("Invalid cgraph for NUMA graph computation\n");
        return GGML_STATUS_FAILED;
    }
    
    // Determine if NUMA coordination would be beneficial
    if (!ggml_numa_should_coordinate(cgraph, n_threads)) {
        GGML_LOG_DEBUG("NUMA coordination not beneficial - using standard computation\n");
        // Fall back to standard GGML computation
        struct ggml_cplan plan = ggml_graph_plan(cgraph, n_threads, NULL);
        if (plan.work_size > 0) {
            plan.work_data = malloc(plan.work_size);
            if (!plan.work_data) {
                return GGML_STATUS_FAILED;
            }
        }
        
        enum ggml_status status = ggml_graph_compute(cgraph, &plan);
        
        if (plan.work_data) {
            free(plan.work_data);
        }
        
        return status;
    }
    
    GGML_LOG_INFO("Using NUMA-aware graph computation for %d operations with %d threads\n", 
                  cgraph->n_nodes, n_threads);
    
    // Get or create the global NUMA coordinator manager
    struct ggml_numa_coordinator_manager * mgr = ggml_numa_coordinator_manager_get_global(n_threads, false);
    if (!mgr) {
        GGML_LOG_ERROR("Failed to create NUMA coordinator manager\n");
        return GGML_STATUS_FAILED;
    }
    
    // Create graph scheduler
    struct ggml_numa_graph_scheduler * scheduler = ggml_numa_create_graph_scheduler(cgraph, mgr->num_numa_nodes);
    if (!scheduler) {
        GGML_LOG_ERROR("Failed to create graph scheduler\n");
        return GGML_STATUS_FAILED;
    }
    
    // Assign operations to NUMA nodes
    int assign_result = ggml_numa_assign_operations_to_nodes(scheduler);
    if (assign_result != 0) {
        GGML_LOG_ERROR("Failed to assign operations to NUMA nodes\n");
        ggml_numa_free_graph_scheduler(scheduler);
        return GGML_STATUS_FAILED;
    }
    
    // Execute the assigned operations
    enum ggml_status exec_result = ggml_numa_execute_assigned_operations(mgr, scheduler);
    
    // Clean up scheduler
    ggml_numa_free_graph_scheduler(scheduler);
    
    if (exec_result == GGML_STATUS_SUCCESS) {
        GGML_LOG_INFO("NUMA graph computation completed successfully\n");
    } else {
        GGML_LOG_ERROR("NUMA graph computation failed\n");
    }
    
    return exec_result;
}

// Determine if NUMA coordination would be beneficial for a given graph
static bool ggml_numa_should_coordinate(
    struct ggml_cgraph * cgraph,
    int n_threads
) {
    if (!cgraph) {
        return false;
    }
    
    // Check if NUMA is available at all
#ifdef __linux__
    if (numa_available() == -1) {
        GGML_LOG_DEBUG("NUMA not available, skipping coordination\n");
        return false;
    }
#else
    GGML_LOG_DEBUG("NUMA coordination not supported on this platform\n");
    return false;
#endif
    
    // Minimum requirements for NUMA coordination
    int min_operations_for_numa = 10;  // Need enough operations to distribute
    int min_threads_for_numa = 4;      // Need enough threads to make coordination worthwhile
    
    if (cgraph->n_nodes < min_operations_for_numa) {
        GGML_LOG_DEBUG("Too few operations (%d < %d) for NUMA coordination\n", 
                      cgraph->n_nodes, min_operations_for_numa);
        return false;
    }
    
    if (n_threads < min_threads_for_numa) {
        GGML_LOG_DEBUG("Too few threads (%d < %d) for NUMA coordination\n", 
                      n_threads, min_threads_for_numa);
        return false;
    }
    
    // Check if we have multiple NUMA nodes
    int num_numa_nodes = numa_max_node() + 1;
    if (num_numa_nodes <= 1) {
        GGML_LOG_DEBUG("Single NUMA node (%d), coordination not beneficial\n", num_numa_nodes);
        return false;
    }
    
    // Estimate if the computational load is large enough to justify NUMA coordination overhead
    int64_t total_elements = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (cgraph->nodes[i]) {
            total_elements += ggml_nelements(cgraph->nodes[i]);
        }
    }
    
    int64_t min_elements_for_numa = 100000; // Minimum computational load
    if (total_elements < min_elements_for_numa) {
        GGML_LOG_DEBUG("Computational load too small (%ld < %ld) for NUMA coordination\n", 
                      total_elements, min_elements_for_numa);
        return false;
    }
    
    GGML_LOG_INFO("NUMA coordination beneficial: %d operations, %d threads, %d NUMA nodes, %ld total elements\n",
                  cgraph->n_nodes, n_threads, num_numa_nodes, total_elements);
    
    return true;
}

// Public fallback execution function for operations not supported by NUMA coordinator
enum ggml_status ggml_numa_fallback_execute_operation(struct ggml_tensor * operation, const struct ggml_compute_params * params) {
    if (!operation) {
        GGML_LOG_ERROR("Invalid operation tensor for fallback execution\n");
        return GGML_STATUS_FAILED;
    }
    
    GGML_LOG_DEBUG("Fallback execution for operation %s\n", ggml_op_name(operation->op));
    
    // Create default single-threaded compute parameters if none provided
    struct ggml_compute_params fallback_params;
    if (params && operation->op != GGML_OP_MUL_MAT) {
        // Use provided params for non-MUL_MAT operations
        fallback_params = *params;
    } else {
        // For MUL_MAT operations, we need a minimal threadpool even in single-threaded mode
        // because MUL_MAT code expects valid threadpool for barriers and chunk management
        if (operation->op == GGML_OP_MUL_MAT && params && params->threadpool) {
            // Create single-threaded parameters but keep the threadpool
            // We'll temporarily set it to single-threaded mode
            fallback_params = (struct ggml_compute_params) {
                .ith = 0,
                .nth = 1,
                .wsize = 0,
                .wdata = NULL,
                .threadpool = params->threadpool
            };
        } else {
            // Default fallback for other operations or when no threadpool available
            fallback_params = (struct ggml_compute_params) {
                .ith = 0,
                .nth = 1,
                .wsize = 0,
                .wdata = NULL,
                .threadpool = NULL
            };
        }
    }
    
    // Execute the operation using GGML's standard compute functions
    // This provides a basic fallback for operations not handled by NUMA coordinator
    switch (operation->op) {
        // Add basic operations that have public compute functions
        case GGML_OP_ADD:
            ggml_compute_forward_add(&fallback_params, operation);
            break;
        case GGML_OP_MUL:
            ggml_compute_forward_mul(&fallback_params, operation);
            break;
        case GGML_OP_DUP:
            ggml_compute_forward_dup(&fallback_params, operation);
            break;
        case GGML_OP_CPY:
            ggml_compute_forward_cpy(&fallback_params, operation);
            break;
        case GGML_OP_CONT:
            ggml_compute_forward_cont(&fallback_params, operation);
            break;
        case GGML_OP_RESHAPE:
            ggml_compute_forward_reshape(&fallback_params, operation);
            break;
        case GGML_OP_VIEW:
            ggml_compute_forward_view(&fallback_params, operation);
            break;
        case GGML_OP_PERMUTE:
            ggml_compute_forward_permute(&fallback_params, operation);
            break;
        case GGML_OP_TRANSPOSE:
            ggml_compute_forward_transpose(&fallback_params, operation);
            break;
        case GGML_OP_NORM:
            ggml_compute_forward_norm(&fallback_params, operation);
            break;
        case GGML_OP_RMS_NORM:
            ggml_compute_forward_rms_norm(&fallback_params, operation);
            break;
        case GGML_OP_SUM:
            ggml_compute_forward_sum(&fallback_params, operation);
            break;
        case GGML_OP_MEAN:
            ggml_compute_forward_mean(&fallback_params, operation);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_compute_forward_soft_max(&fallback_params, operation);
            break;
        case GGML_OP_MUL_MAT:
            ggml_compute_forward_mul_mat(&fallback_params, operation);
            break;
            
        default:
            GGML_LOG_ERROR("Operation %s not available for fallback execution - no public compute function\n", 
                          ggml_op_name(operation->op));
            return GGML_STATUS_FAILED;
    }
    
    GGML_LOG_DEBUG("Fallback execution completed for operation %s\n", ggml_op_name(operation->op));
    return GGML_STATUS_SUCCESS;
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

