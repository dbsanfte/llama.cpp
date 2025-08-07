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
    int64_t chunk_start;               // Start of chunk (elements/rows depending on operation)
    int64_t chunk_end;                 // End of chunk  
    atomic_bool completed;             // Completion flag
    void * result_buffer;              // Buffer for partial results
    size_t result_size;                // Size of result buffer
    struct ggml_work_item * next;      // Next item in queue
    int work_id;                       // Unique work ID for tracking
    
    // Data parallelism support
    int work_group_id;                 // Work group this item belongs to (-1 for single work)
    int chunk_index;                   // Index of this chunk within work group (0-based)
    int total_chunks;                  // Total number of chunks in work group
    bool is_data_parallel;             // Whether this is part of data parallel work
    struct ggml_tensor * original_tensor; // Reference to original tensor before chunking
    
    // Operation-specific information
    enum ggml_op operation;            // The operation being performed
    bool is_element_wise;              // Whether operation can be split by elements
    bool is_row_wise;                  // Whether operation can be split by rows
};

// Work group for data parallel operations
struct ggml_work_group {
    int group_id;                      // Unique group ID
    struct ggml_tensor * original_tensor; // Original tensor being processed
    struct ggml_work_item ** chunks;   // Array of work items (one per chunk)
    int num_chunks;                    // Number of chunks
    atomic_int completed_chunks;       // Number of completed chunks
    atomic_bool group_completed;       // Whether entire group is completed
    struct ggml_tensor * result_tensor; // Integrated result tensor
    int64_t split_dimension;           // Which dimension was split (0=rows, 1=cols, etc.)
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
    int numa_node;                     // NUMA node this coordinator manages
    struct ggml_threadpool * numa_pool;// NUMA-specific threadpool
    struct ggml_cgraph * numa_cgraph;  // NUMA node's own copy of cgraph
    struct ggml_context * numa_ctx;    // Context for this NUMA node's cgraph copy
    struct ggml_work_queue work_queue; // Work queue for this coordinator
    ggml_thread_t thread_handle;       // Thread handle
    atomic_bool active;                // Whether thread is active
    atomic_bool shutdown_requested;    // Shutdown request flag
    atomic_bool thread_created;        // Whether thread was actually created
    struct ggml_numa_coordinator_manager * manager; // Reference to parent manager for callbacks
    
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
    
    // Data parallelism support
    struct ggml_work_group_tracker work_groups;          // Work group tracking
    
    // Progress callback system
    ggml_numa_progress_callback_t progress_callback;     // User progress callback function
    void * progress_callback_user_data;                  // User data for callback
    
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

// Work group management functions
static void ggml_work_group_tracker_init(struct ggml_work_group_tracker * tracker);
static void ggml_work_group_tracker_destroy(struct ggml_work_group_tracker * tracker);
static struct ggml_work_group * ggml_work_group_create(struct ggml_work_group_tracker * tracker, struct ggml_tensor * tensor, int num_chunks);
static void ggml_work_group_free(struct ggml_work_group * group);
static int ggml_work_group_check_completion(struct ggml_work_group * group);
static int ggml_operation_split_for_numa(struct ggml_tensor * tensor, int num_numa_nodes, struct ggml_work_item *** out_chunks);
static int ggml_tensor_integrate_results(struct ggml_work_group * group);

// Operation-specific NUMA chunk execution functions
static enum ggml_status ggml_numa_execute_elementwise_chunk(
    struct ggml_coordinator_thread * coordinator,
    struct ggml_tensor * tensor,
    int64_t chunk_start,
    int64_t chunk_end,
    void * result_buffer,
    size_t result_size
);

static enum ggml_status ggml_numa_execute_matmul_chunk(
    struct ggml_coordinator_thread * coordinator,
    struct ggml_tensor * tensor,
    int64_t chunk_start,
    int64_t chunk_end,
    void * result_buffer,
    size_t result_size
);

static enum ggml_status ggml_numa_execute_unary_chunk(
    struct ggml_coordinator_thread * coordinator,
    struct ggml_tensor * tensor,
    int64_t chunk_start,
    int64_t chunk_end,
    void * result_buffer,
    size_t result_size
);

static enum ggml_status ggml_numa_execute_softmax_chunk(
    struct ggml_coordinator_thread * coordinator,
    struct ggml_tensor * tensor,
    int64_t chunk_start,
    int64_t chunk_end,
    void * result_buffer,
    size_t result_size
);

// Global coordinator management functions
static struct ggml_numa_coordinator_manager * ggml_get_global_coordinator_manager(int n_threads, bool force_multi_socket);
static struct ggml_numa_coordinator_manager * ggml_get_global_coordinator_manager_with_params(const struct ggml_threadpool_params * tpp);
static void ggml_register_program_exit_cleanup(void);

// Operation-specific NUMA parallelization using proper GGML compute functions
static enum ggml_status ggml_numa_execute_operation_chunk(
    struct ggml_coordinator_thread * coordinator,
    struct ggml_tensor * tensor,
    int64_t chunk_start,
    int64_t chunk_end,
    void * result_buffer,
    size_t result_size
) {
    if (!coordinator || !tensor || !result_buffer) return GGML_STATUS_FAILED;
    
    GGML_LOG_DEBUG("NUMA%d: executing operation %s chunk %ld-%ld\n", 
                   coordinator->numa_node, ggml_op_name(tensor->op), chunk_start, chunk_end);
    
    enum ggml_status status = GGML_STATUS_FAILED;
    
    switch (tensor->op) {
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SUB:
            // Element-wise operations
            status = ggml_numa_execute_elementwise_chunk(coordinator, tensor, chunk_start, chunk_end, result_buffer, result_size);
            break;
            
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            // Matrix multiplication
            status = ggml_numa_execute_matmul_chunk(coordinator, tensor, chunk_start, chunk_end, result_buffer, result_size);
            break;
            
        case GGML_OP_UNARY:
            // Unary operations (GELU, SILU, RELU, etc.)
            status = ggml_numa_execute_unary_chunk(coordinator, tensor, chunk_start, chunk_end, result_buffer, result_size);
            break;
            
        case GGML_OP_SOFT_MAX:
            // Softmax
            status = ggml_numa_execute_softmax_chunk(coordinator, tensor, chunk_start, chunk_end, result_buffer, result_size);
            break;
            
        default:
            GGML_LOG_WARN("NUMA%d: Operation %s not supported for operation-level parallelism\n", 
                         coordinator->numa_node, ggml_op_name(tensor->op));
            return GGML_STATUS_FAILED;
    }
    
    if (status != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("NUMA%d: Failed to execute operation %s chunk %ld-%ld\n", 
                       coordinator->numa_node, ggml_op_name(tensor->op), chunk_start, chunk_end);
    }
    
    return status;
}

static enum ggml_status ggml_numa_execute_elementwise_chunk(
    struct ggml_coordinator_thread * coordinator,
    struct ggml_tensor * tensor,
    int64_t chunk_start,
    int64_t chunk_end,
    void * result_buffer,
    size_t result_size
) {
    if (!coordinator || !tensor || !result_buffer) return GGML_STATUS_FAILED;
    
    // For element-wise operations, we need to create a tensor view for the chunk
    // This is complex because we need to handle tensor slicing properly
    // For now, let's use the threadpool to compute the full tensor and extract the chunk
    
    GGML_LOG_DEBUG("NUMA%d: computing element-wise operation %s (elements %ld-%ld)\n", 
                   coordinator->numa_node, ggml_op_name(tensor->op), chunk_start, chunk_end);
    
    // Create compute parameters
    struct ggml_compute_params params = {
        .ith = 0,
        .nth = 1,
        .wsize = 0,
        .wdata = NULL,
        .threadpool = coordinator->numa_pool
    };
    
    // For this version, compute the full operation and extract the chunk
    // TODO: Optimize this to only compute the required chunk
    enum ggml_status status = GGML_STATUS_FAILED;
    
    switch (tensor->op) {
        case GGML_OP_ADD:
            ggml_compute_forward_add(&params, tensor);
            status = GGML_STATUS_SUCCESS;
            break;
        case GGML_OP_MUL:
            ggml_compute_forward_mul(&params, tensor);
            status = GGML_STATUS_SUCCESS;
            break;
        case GGML_OP_DIV:
            ggml_compute_forward_div(&params, tensor);
            status = GGML_STATUS_SUCCESS;
            break;
        case GGML_OP_SUB:
            ggml_compute_forward_sub(&params, tensor);
            status = GGML_STATUS_SUCCESS;
            break;
        default:
            GGML_LOG_ERROR("Unsupported element-wise operation: %s\n", ggml_op_name(tensor->op));
            return GGML_STATUS_FAILED;
    }
    
    if (status == GGML_STATUS_SUCCESS) {
        // Extract the chunk from the computed result
        size_t element_size = ggml_type_size(tensor->type);
        size_t chunk_bytes = (chunk_end - chunk_start) * element_size;
        size_t chunk_offset = chunk_start * element_size;
        
        if (chunk_bytes <= result_size) {
            memcpy(result_buffer, (char*)ggml_get_data(tensor) + chunk_offset, chunk_bytes);
        } else {
            GGML_LOG_ERROR("Chunk size %zu exceeds result buffer size %zu\n", chunk_bytes, result_size);
            status = GGML_STATUS_FAILED;
        }
    }
    
    return status;
}

static enum ggml_status ggml_numa_execute_matmul_chunk(
    struct ggml_coordinator_thread * coordinator,
    struct ggml_tensor * tensor,
    int64_t chunk_start,
    int64_t chunk_end,
    void * result_buffer,
    size_t result_size
) {
    if (!coordinator || !tensor || !result_buffer) return GGML_STATUS_FAILED;
    
    GGML_LOG_DEBUG("NUMA%d: computing matrix multiplication chunk (rows %ld-%ld)\n", 
                   coordinator->numa_node, chunk_start, chunk_end);
    
    // Create compute parameters
    struct ggml_compute_params params = {
        .ith = 0,
        .nth = 1,
        .wsize = 0,
        .wdata = NULL,
        .threadpool = coordinator->numa_pool
    };
    
    // For matrix multiplication, we would need to create tensor views for the chunk
    // This is complex and requires understanding the exact tensor layout
    // For now, compute full matrix and extract chunk
    
    enum ggml_status status = GGML_STATUS_FAILED;
    
    switch (tensor->op) {
        case GGML_OP_MUL_MAT:
            ggml_compute_forward_mul_mat(&params, tensor);
            status = GGML_STATUS_SUCCESS;
            break;
        default:
            GGML_LOG_ERROR("Unsupported matrix operation: %s\n", ggml_op_name(tensor->op));
            return GGML_STATUS_FAILED;
    }
    
    if (status == GGML_STATUS_SUCCESS) {
        // Extract the chunk (rows chunk_start to chunk_end)
        int64_t cols = tensor->ne[0];
        size_t element_size = ggml_type_size(tensor->type);
        size_t row_stride = cols * element_size;
        size_t chunk_bytes = (chunk_end - chunk_start) * row_stride;
        size_t chunk_offset = chunk_start * row_stride;
        
        if (chunk_bytes <= result_size) {
            memcpy(result_buffer, (char*)ggml_get_data(tensor) + chunk_offset, chunk_bytes);
        } else {
            GGML_LOG_ERROR("Matrix chunk size %zu exceeds result buffer size %zu\n", chunk_bytes, result_size);
            status = GGML_STATUS_FAILED;
        }
    }
    
    return status;
}

static enum ggml_status ggml_numa_execute_unary_chunk(
    struct ggml_coordinator_thread * coordinator,
    struct ggml_tensor * tensor,
    int64_t chunk_start,
    int64_t chunk_end,
    void * result_buffer,
    size_t result_size
) {
    if (!coordinator || !tensor || !result_buffer) return GGML_STATUS_FAILED;
    
    GGML_LOG_DEBUG("NUMA%d: computing unary operation chunk (elements %ld-%ld)\n", 
                   coordinator->numa_node, chunk_start, chunk_end);
    
    // Create compute parameters
    struct ggml_compute_params params = {
        .ith = 0,
        .nth = 1,
        .wsize = 0,
        .wdata = NULL,
        .threadpool = coordinator->numa_pool
    };
    
    // For unary operations, compute full tensor and extract chunk
    ggml_compute_forward_unary(&params, tensor);
    
    // Extract the chunk
    size_t element_size = ggml_type_size(tensor->type);
    size_t chunk_bytes = (chunk_end - chunk_start) * element_size;
    size_t chunk_offset = chunk_start * element_size;
    
    if (chunk_bytes <= result_size) {
        memcpy(result_buffer, (char*)ggml_get_data(tensor) + chunk_offset, chunk_bytes);
        return GGML_STATUS_SUCCESS;
    } else {
        GGML_LOG_ERROR("Unary chunk size %zu exceeds result buffer size %zu\n", chunk_bytes, result_size);
        return GGML_STATUS_FAILED;
    }
}

static enum ggml_status ggml_numa_execute_softmax_chunk(
    struct ggml_coordinator_thread * coordinator,
    struct ggml_tensor * tensor,
    int64_t chunk_start,
    int64_t chunk_end,
    void * result_buffer,
    size_t result_size
) {
    if (!coordinator || !tensor || !result_buffer) return GGML_STATUS_FAILED;
    
    GGML_LOG_DEBUG("NUMA%d: computing softmax chunk (rows %ld-%ld)\n", 
                   coordinator->numa_node, chunk_start, chunk_end);
    
    // Create compute parameters
    struct ggml_compute_params params = {
        .ith = 0,
        .nth = 1,
        .wsize = 0,
        .wdata = NULL,
        .threadpool = coordinator->numa_pool
    };
    
    // For softmax, each row is independent, so we can compute per-row chunks
    // For now, compute full tensor and extract chunk
    ggml_compute_forward_soft_max(&params, tensor);
    
    // Extract the chunk (rows chunk_start to chunk_end)
    int64_t cols = tensor->ne[0];
    size_t element_size = ggml_type_size(tensor->type);
    size_t row_stride = cols * element_size;
    size_t chunk_bytes = (chunk_end - chunk_start) * row_stride;
    size_t chunk_offset = chunk_start * row_stride;
    
    if (chunk_bytes <= result_size) {
        memcpy(result_buffer, (char*)ggml_get_data(tensor) + chunk_offset, chunk_bytes);
        return GGML_STATUS_SUCCESS;
    } else {
        GGML_LOG_ERROR("Softmax chunk size %zu exceeds result buffer size %zu\n", chunk_bytes, result_size);
        return GGML_STATUS_FAILED;
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
    
    // Note: We don't free result_tensor as it's typically owned by the caller
    // Note: We don't free original_tensor as it's owned by the caller
    
    free(group);
}

// Check if work group is completed and integrate results if so
static int ggml_work_group_check_completion(struct ggml_work_group * group) {
    if (!group) return -1;
    
    if (atomic_load(&group->group_completed)) {
        return 1; // Already completed
    }
    
    int completed = atomic_load(&group->completed_chunks);
    if (completed >= group->num_chunks) {
        // All chunks completed, integrate results
        int result = ggml_tensor_integrate_results(group);
        if (result == 0) {
            atomic_store(&group->group_completed, true);
            GGML_LOG_INFO("Work group %d completed and integrated\n", group->group_id);
            return 1;
        } else {
            GGML_LOG_ERROR("Failed to integrate results for work group %d\n", group->group_id);
            return -1;
        }
    }
    
    return 0; // Not yet completed
}

// Split operation for NUMA parallel processing (operation-aware)
static int ggml_operation_split_for_numa(struct ggml_tensor * tensor, int num_numa_nodes, struct ggml_work_item *** out_chunks) {
    if (!tensor || num_numa_nodes <= 0 || !out_chunks) return -1;
    
    // Determine how to split based on the operation type
    bool can_parallelize = false;
    bool split_by_elements = false;  // For element-wise ops
    bool split_by_rows = false;      // For matrix ops, softmax, etc.
    int64_t total_work_units = 0;
    
    switch (tensor->op) {
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SUB:
        case GGML_OP_UNARY:
            // Element-wise operations - split by elements
            can_parallelize = true;
            split_by_elements = true;
            total_work_units = ggml_nelements(tensor);
            GGML_LOG_DEBUG("Operation %s: splitting %ld elements across %d NUMA nodes\n", 
                           ggml_op_name(tensor->op), total_work_units, num_numa_nodes);
            break;
            
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            // Matrix multiplication - split by output rows
            can_parallelize = true;
            split_by_rows = true;
            total_work_units = tensor->ne[1]; // Number of rows in output matrix
            GGML_LOG_DEBUG("Operation %s: splitting %ld rows across %d NUMA nodes\n", 
                           ggml_op_name(tensor->op), total_work_units, num_numa_nodes);
            break;
            
        case GGML_OP_SOFT_MAX:
            // Softmax - split by rows (each row computed independently)
            can_parallelize = true;
            split_by_rows = true;
            total_work_units = tensor->ne[1]; // Number of rows
            GGML_LOG_DEBUG("Operation %s: splitting %ld rows across %d NUMA nodes\n", 
                           ggml_op_name(tensor->op), total_work_units, num_numa_nodes);
            break;
            
        default:
            GGML_LOG_WARN("Operation %s not supported for NUMA parallelization\n", ggml_op_name(tensor->op));
            return -1;
    }
    
    if (!can_parallelize || total_work_units < num_numa_nodes) {
        // Not enough work to split across all NUMA nodes
        if (total_work_units > 0 && total_work_units < num_numa_nodes) {
            num_numa_nodes = (int)total_work_units;
        } else if (!can_parallelize) {
            return -1;
        }
    }
    
    int64_t work_per_chunk = total_work_units / num_numa_nodes;
    int64_t remaining_work = total_work_units % num_numa_nodes;
    
    GGML_LOG_DEBUG("Splitting operation %s: %ld work units into %d chunks (%ld per chunk, %ld remaining)\n", 
                   ggml_op_name(tensor->op), total_work_units, num_numa_nodes, work_per_chunk, remaining_work);
    
    // Allocate array of work item pointers
    struct ggml_work_item ** chunks = malloc(sizeof(struct ggml_work_item *) * num_numa_nodes);
    if (!chunks) return -1;
    
    int64_t current_start = 0;
    for (int i = 0; i < num_numa_nodes; i++) {
        chunks[i] = malloc(sizeof(struct ggml_work_item));
        if (!chunks[i]) {
            // Cleanup on failure
            for (int j = 0; j < i; j++) {
                if (chunks[j]->result_buffer) free(chunks[j]->result_buffer);
                free(chunks[j]);
            }
            free(chunks);
            return -1;
        }
        
        memset(chunks[i], 0, sizeof(struct ggml_work_item));
        
        // Calculate chunk size (give extra work to first chunks if uneven split)
        int64_t chunk_work = work_per_chunk;
        if (i < remaining_work) {
            chunk_work++;
        }
        
        // Set up work item
        chunks[i]->tensor = tensor;
        chunks[i]->original_tensor = tensor;
        chunks[i]->numa_node = i; // Assign to specific NUMA node
        chunks[i]->chunk_start = current_start;
        chunks[i]->chunk_end = current_start + chunk_work;
        chunks[i]->chunk_index = i;
        chunks[i]->total_chunks = num_numa_nodes;
        chunks[i]->is_data_parallel = true;
        
        // Operation-specific information
        chunks[i]->operation = tensor->op;
        chunks[i]->is_element_wise = split_by_elements;
        chunks[i]->is_row_wise = split_by_rows;
        
        // Calculate result buffer size based on operation type
        size_t chunk_bytes = 0;
        if (split_by_elements) {
            // Element-wise: chunk_work elements
            chunk_bytes = chunk_work * ggml_type_size(tensor->type);
        } else if (split_by_rows) {
            // Row-wise: chunk_work rows × cols
            int64_t cols = tensor->ne[0];
            chunk_bytes = chunk_work * cols * ggml_type_size(tensor->type);
        }
        
        chunks[i]->result_buffer = malloc(chunk_bytes);
        chunks[i]->result_size = chunk_bytes;
        
        if (!chunks[i]->result_buffer) {
            // Cleanup on failure
            for (int j = 0; j <= i; j++) {
                if (chunks[j]->result_buffer) free(chunks[j]->result_buffer);
                free(chunks[j]);
            }
            free(chunks);
            return -1;
        }
        
        atomic_init(&chunks[i]->completed, false);
        
        if (split_by_elements) {
            GGML_LOG_DEBUG("Chunk %d: elements %ld-%ld (%ld elements, %zu bytes) -> NUMA %d\n", 
                           i, current_start, current_start + chunk_work - 1, chunk_work, chunk_bytes, i);
        } else if (split_by_rows) {
            GGML_LOG_DEBUG("Chunk %d: rows %ld-%ld (%ld rows, %zu bytes) -> NUMA %d\n", 
                           i, current_start, current_start + chunk_work - 1, chunk_work, chunk_bytes, i);
        }
        
        current_start += chunk_work;
    }
    
    *out_chunks = chunks;
    return num_numa_nodes;
}

// Integrate results from all chunks back into original tensor
static int ggml_tensor_integrate_results(struct ggml_work_group * group) {
    if (!group || !group->chunks || !group->original_tensor || group->num_chunks == 0) return -1;
    
    GGML_LOG_DEBUG("Integrating results for work group %d (%d chunks)\n", group->group_id, group->num_chunks);
    
    struct ggml_tensor * original = group->original_tensor;
    char * original_data = (char *)ggml_get_data(original);
    
    if (!original_data) {
        GGML_LOG_ERROR("Original tensor has no data pointer\n");
        return -1;
    }
    
    // Get operation type from first chunk
    struct ggml_work_item * first_chunk = group->chunks[0];
    if (!first_chunk) {
        GGML_LOG_ERROR("No chunks found in work group\n");
        return -1;
    }
    
    enum ggml_op operation = first_chunk->operation;
    bool is_element_wise = first_chunk->is_element_wise;
    bool is_row_wise = first_chunk->is_row_wise;
    
    size_t element_size = ggml_type_size(original->type);
    
    GGML_LOG_DEBUG("Integration for operation %s (element_wise=%d, row_wise=%d)\n", 
                   ggml_op_name(operation), is_element_wise, is_row_wise);
    
    // Verify all chunks are completed
    for (int i = 0; i < group->num_chunks; i++) {
        struct ggml_work_item * chunk = group->chunks[i];
        if (!chunk || !atomic_load(&chunk->completed)) {
            GGML_LOG_ERROR("Chunk %d not completed for work group %d\n", i, group->group_id);
            return -1;
        }
        
        if (!chunk->result_buffer) {
            GGML_LOG_ERROR("Chunk %d has no result buffer for work group %d\n", i, group->group_id);
            return -1;
        }
    }
    
    // Integrate based on operation type
    if (is_element_wise) {
        // Element-wise operations: concatenate element results
        GGML_LOG_DEBUG("Integrating element-wise operation\n");
        
        for (int i = 0; i < group->num_chunks; i++) {
            struct ggml_work_item * chunk = group->chunks[i];
            int64_t start_element = chunk->chunk_start;
            int64_t num_elements = chunk->chunk_end - chunk->chunk_start;
            
            size_t dest_offset = start_element * element_size;
            size_t chunk_size = num_elements * element_size;
            
            // Sanity checks
            if (dest_offset + chunk_size > ggml_nbytes(original)) {
                GGML_LOG_ERROR("Element-wise chunk %d would overflow: offset=%zu + size=%zu > tensor_size=%zu\n", 
                               i, dest_offset, chunk_size, ggml_nbytes(original));
                return -1;
            }
            
            GGML_LOG_DEBUG("Chunk %d: copying %zu bytes to offset %zu (elements %ld-%ld)\n", 
                           i, chunk_size, dest_offset, start_element, chunk->chunk_end - 1);
            
            memcpy(original_data + dest_offset, chunk->result_buffer, chunk_size);
        }
        
    } else if (is_row_wise) {
        // Row-wise operations: concatenate row results
        GGML_LOG_DEBUG("Integrating row-wise operation\n");
        
        int64_t cols = original->ne[0];
        size_t row_stride = cols * element_size;
        
        for (int i = 0; i < group->num_chunks; i++) {
            struct ggml_work_item * chunk = group->chunks[i];
            int64_t start_row = chunk->chunk_start;
            int64_t num_rows = chunk->chunk_end - chunk->chunk_start;
            
            size_t dest_offset = start_row * row_stride;
            size_t chunk_size = num_rows * row_stride;
            
            // Sanity checks
            if (dest_offset + chunk_size > ggml_nbytes(original)) {
                GGML_LOG_ERROR("Row-wise chunk %d would overflow: offset=%zu + size=%zu > tensor_size=%zu\n", 
                               i, dest_offset, chunk_size, ggml_nbytes(original));
                return -1;
            }
            
            GGML_LOG_DEBUG("Chunk %d: copying %zu bytes to offset %zu (rows %ld-%ld)\n", 
                           i, chunk_size, dest_offset, start_row, chunk->chunk_end - 1);
            
            memcpy(original_data + dest_offset, chunk->result_buffer, chunk_size);
        }
        
    } else {
        GGML_LOG_ERROR("Unknown integration pattern for operation %s\n", ggml_op_name(operation));
        return -1;
    }
    
    GGML_LOG_INFO("Successfully integrated %d chunks for operation %s\n", 
                  group->num_chunks, ggml_op_name(operation));
    return 0;
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
        
        // Execute the work using NUMA-specific threadpool and cgraph
        if (coordinator->numa_pool && coordinator->numa_cgraph) {
            enum ggml_status status = GGML_STATUS_FAILED;
            
            if (work_item->is_data_parallel) {
                // For data parallel work, execute operation-specific parallelism
                // Each NUMA node computes a portion of the specific operation
                
                status = ggml_numa_execute_operation_chunk(
                    coordinator, 
                    work_item->tensor, 
                    work_item->chunk_start, 
                    work_item->chunk_end,
                    work_item->result_buffer,
                    work_item->result_size
                );
                
            } else {
                // For non-data-parallel work, compute the entire cgraph
                // Use single-threaded execution at coordinator level - the threadpool handles parallelism
                int numa_threads = 1;
                
                struct ggml_cplan plan = ggml_graph_plan(coordinator->numa_cgraph, numa_threads, coordinator->numa_pool);
                
                if (plan.work_size > 0) {
                    plan.work_data = malloc(plan.work_size);
                }
                
                // Execute computation on NUMA node
                status = ggml_graph_compute(coordinator->numa_cgraph, &plan);
                
                if (status == GGML_STATUS_SUCCESS && work_item->result_buffer && work_item->result_size > 0) {
                    // Copy the entire result
                    size_t copy_size = min_size_t(work_item->result_size, ggml_nbytes(work_item->tensor));
                    memcpy(work_item->result_buffer, ggml_get_data(work_item->tensor), copy_size);
                }
                
                if (plan.work_data) {
                    free(plan.work_data);
                }
            }
            
            if (status != GGML_STATUS_SUCCESS) {
                GGML_LOG_WARN("Coordinator NUMA%d: Work item %d failed with status %d\n",
                             coordinator->numa_node, work_item->work_id, status);
            }
        }
        
        int64_t end_time = ggml_time_us();
        coordinator->total_processing_time_us += (end_time - start_time);
        coordinator->total_work_items++;
        
        // Mark work item as completed
        atomic_store(&work_item->completed, true);
        
        // Handle work group completion for data parallel work
        if (work_item->is_data_parallel && work_item->work_group_id >= 0) {
            // Find the work group and check if all chunks are completed
            ggml_mutex_lock(&coordinator->manager->work_groups.groups_mutex);
            
            for (int group_idx = 0; group_idx < coordinator->manager->work_groups.max_groups; group_idx++) {
                struct ggml_work_group * group = coordinator->manager->work_groups.groups[group_idx];
                if (group && group->group_id == work_item->work_group_id) {
                    // Increment completed chunks counter
                    int completed_chunks = atomic_fetch_add(&group->completed_chunks, 1) + 1;
                    
                    GGML_LOG_DEBUG("Work group %d: chunk %d completed (%d/%d)\n", 
                                   group->group_id, work_item->chunk_index, completed_chunks, group->num_chunks);
                    
                    // Check if all chunks are completed
                    if (completed_chunks >= group->num_chunks) {
                        GGML_LOG_INFO("Work group %d: all chunks completed, ready for integration\n", group->group_id);
                    }
                    break;
                }
            }
            
            ggml_mutex_unlock(&coordinator->manager->work_groups.groups_mutex);
        }
        
        // Call progress callback if set
        if (coordinator->manager && coordinator->manager->progress_callback) {
            coordinator->manager->progress_callback(
                work_item->work_id,
                coordinator->numa_node, 
                work_item->tensor,
                coordinator->manager->progress_callback_user_data
            );
        }
        
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
    
    for (int i = 0; i < num_numa_nodes; i++) {
        struct ggml_coordinator_thread * coord = &mgr->coordinators[i];
        coord->numa_node = i;
        atomic_init(&coord->active, false);
        atomic_init(&coord->shutdown_requested, false);
        atomic_init(&coord->thread_created, false);
        coord->total_work_items = 0;
        coord->total_processing_time_us = 0;
        coord->manager = mgr; // Reference to parent manager for callbacks
        
        // Step 4: Create NUMA-specific threadpool parameters for this coordinator
        struct ggml_threadpool_params numa_tpp = *tpp; // Copy base parameters
        numa_tpp.n_threads = threads_per_numa;
        numa_tpp.numa_aware = false; // CRITICAL: Disable coordinator recursion - we ARE the coordinator
        numa_tpp.force_multi_socket = false; // Don't create nested coordinators
        
        // Step 5: Apply NUMA-specific CPU mask filtering
        if (numa_is_available && i < GGML_NUMA_MAX_NODES) {
#ifdef __linux__
            // Filter the original CPU mask to only include CPUs from this NUMA node
            struct bitmask* node_cpus = numa_allocate_cpumask();
            if (numa_node_to_cpus(i, node_cpus) == 0) {
                // Create NUMA-filtered CPU mask
                memset(numa_tpp.cpumask, false, sizeof(numa_tpp.cpumask));
                bool has_numa_cpus = false;
                
                for (int cpu = 0; cpu < GGML_MAX_N_THREADS; cpu++) {
                    // Include CPU if it's both in the original mask AND on this NUMA node
                    if (tpp->cpumask[cpu] && numa_bitmask_isbitset(node_cpus, cpu)) {
                        numa_tpp.cpumask[cpu] = true;
                        has_numa_cpus = true;
                    }
                }
                
                if (!has_numa_cpus) {
                    GGML_LOG_WARN("NUMA node %d: no NUMA-local CPUs found in mask, using original mask\n", i);
                    memcpy(numa_tpp.cpumask, tpp->cpumask, sizeof(numa_tpp.cpumask));
                } else {
                    GGML_LOG_INFO("NUMA node %d: filtered CPU mask to NUMA-local CPUs\n", i);
                }
            } else {
                GGML_LOG_WARN("NUMA node %d: failed to get node CPUs, using original mask\n", i);
                memcpy(numa_tpp.cpumask, tpp->cpumask, sizeof(numa_tpp.cpumask));
            }
            numa_free_cpumask(node_cpus);
#endif
        } else {
            // No NUMA or simulated mode: divide CPU mask evenly
            bool has_cpu_mask = false;
            for (int cpu = 0; cpu < GGML_MAX_N_THREADS; cpu++) {
                if (tpp->cpumask[cpu]) {
                    has_cpu_mask = true;
                    break;
                }
            }
            
            if (has_cpu_mask) {
                // Count available CPUs in original mask
                int available_cpus[GGML_MAX_N_THREADS];
                int cpu_count = 0;
                
                for (int cpu = 0; cpu < GGML_MAX_N_THREADS; cpu++) {
                    if (tpp->cpumask[cpu]) {
                        available_cpus[cpu_count++] = cpu;
                    }
                }
                
                // Distribute CPUs round-robin across NUMA nodes
                memset(numa_tpp.cpumask, false, sizeof(numa_tpp.cpumask));
                bool has_assigned_cpus = false;
                
                for (int j = 0; j < cpu_count; j++) {
                    if ((j % num_numa_nodes) == i) {
                        numa_tpp.cpumask[available_cpus[j]] = true;
                        has_assigned_cpus = true;
                    }
                }
                
                GGML_LOG_INFO("NUMA node %d: assigned %s CPUs using round-robin distribution\n", 
                              i, has_assigned_cpus ? "specific" : "no specific");
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
        
        GGML_LOG_INFO("Created coordinator for NUMA node %d with %d threads (CPU mask: %s)\n", 
                      i, threads_per_numa, "NUMA-aware");
    }
    
    GGML_LOG_INFO("NUMA coordinator manager created with %d NUMA nodes\n", num_numa_nodes);
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
    ggml_work_group_tracker_destroy(&mgr->work_groups);
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
    
    // Initialize data parallelism fields for single-node work
    work_item->work_group_id = -1;           // Not part of a work group
    work_item->chunk_index = 0;              // Single chunk
    work_item->total_chunks = 1;             // Single chunk
    work_item->is_data_parallel = false;     // Not data parallel
    work_item->original_tensor = tensor;     // Same as tensor for single work
    
    if (!work_item->result_buffer) {
        free(work_item);
        return -1;
    }
    
    // Enqueue to target coordinator
    ggml_work_queue_enqueue(&mgr->coordinators[target_numa].work_queue, work_item);
    
    return work_item->work_id;
}

// Submit tensor with data parallelism across multiple NUMA nodes
int ggml_numa_coordinator_manager_submit_data_parallel_work(struct ggml_numa_coordinator_manager * mgr,
                                                            struct ggml_tensor * tensor) {
    if (!mgr || !tensor) return -1;
    
    // Only use data parallelism if we have multiple NUMA nodes
    if (mgr->num_numa_nodes <= 1) {
        GGML_LOG_DEBUG("Single NUMA node, falling back to regular work submission\n");
        return ggml_numa_coordinator_manager_submit_work(mgr, tensor, -1);
    }
    
    GGML_LOG_INFO("Submitting data parallel work for tensor %p across %d NUMA nodes\n", (void*)tensor, mgr->num_numa_nodes);
    
    // Ensure coordinator threads are started before submitting work
    int start_result = ggml_numa_coordinator_manager_start(mgr);
    if (start_result != 0) {
        GGML_LOG_ERROR("Failed to start coordinator threads for data parallel work\n");
        return -1;
    }
    
    // Data parallel work requires a proper cgraph context - it should not be called
    // for individual tensor submissions but only as part of compute_graph operations
    if (!mgr->coordinators[0].numa_cgraph) {
        GGML_LOG_ERROR("Data parallel work requires cgraph to be set first - use ggml_numa_coordinator_manager_compute_graph\n");
        return -1;
    }
    
    // Create work group for this tensor
    struct ggml_work_group * group = ggml_work_group_create(&mgr->work_groups, tensor, mgr->num_numa_nodes);
    if (!group) {
        GGML_LOG_ERROR("Failed to create work group for data parallel work\n");
        return -1;
    }
    
    // Split operation into chunks
    struct ggml_work_item ** chunks = NULL;
    int num_chunks = ggml_operation_split_for_numa(tensor, mgr->num_numa_nodes, &chunks);
    if (num_chunks <= 0 || !chunks) {
        GGML_LOG_ERROR("Failed to split tensor for NUMA processing\n");
        ggml_work_group_free(group);
        return -1;
    }
    
    // Store chunks in work group
    group->chunks = chunks;
    group->num_chunks = num_chunks;
    
    // Submit each chunk to its assigned NUMA node
    for (int i = 0; i < num_chunks; i++) {
        struct ggml_work_item * chunk = chunks[i];
        chunk->work_group_id = group->group_id;
        chunk->work_id = atomic_fetch_add(&mgr->total_work_items, 1);
        
        // Enqueue to the specific NUMA node coordinator
        int target_numa = chunk->numa_node;
        if (target_numa < 0 || target_numa >= mgr->num_numa_nodes) {
            target_numa = i % mgr->num_numa_nodes; // Fallback
        }
        
        GGML_LOG_DEBUG("Submitting chunk %d (work_id %d) to NUMA node %d\n", i, chunk->work_id, target_numa);
        ggml_work_queue_enqueue(&mgr->coordinators[target_numa].work_queue, chunk);
    }
    
    GGML_LOG_INFO("Submitted %d chunks for data parallel processing (work group %d)\n", num_chunks, group->group_id);
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

// Wait for all work to complete (Step 7: Main thread polls for completion)
int ggml_numa_coordinator_manager_wait_for_completion(struct ggml_numa_coordinator_manager * mgr) {
    if (!mgr) return -1;
    
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
    
    // Add a small delay to ensure coordinator threads are truly idle
    // This prevents race conditions when switching to a new cgraph immediately
    usleep(1000); // 1ms delay
    
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
    
    // Wait for completion and integration
    while (!atomic_load(&target_group->group_completed)) {
        // Check if all chunks are completed and trigger integration if needed
        int completion_status = ggml_work_group_check_completion(target_group);
        if (completion_status == 1) {
            // Successfully completed and integrated
            break;
        } else if (completion_status == -1) {
            // Integration failed
            GGML_LOG_ERROR("Work group %d integration failed\n", work_group_id);
            return -1;
        }
        
        // Sleep briefly to avoid busy waiting
        usleep(100); // 100 microseconds
    }
    
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

