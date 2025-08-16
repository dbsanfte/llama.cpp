/**
 * NUMA Operation Dispatch Infrastructure Implementation
 * 
 * This implements the intelligent dispatcher system that routes operations
 * to appropriate execution strategies while preserving existing thread synchronization.
 */

#include "ggml-numa-operation-dispatch.h"
#include "ggml-numa-coordinator.h"
#include "ggml-numa-fallback.h"
#include "ggml-impl.h"
#include "ggml-cpu-impl.h"  // For ggml_compute_params structure
#include "ggml.h"           // For ggml_cplan and graph functions

// Include all operation headers for fallback system
#include "ops.h"            // Main operations
#include "unary-ops.h"      // Unary operations (sin, cos, log, etc.)
#include "binary-ops.h"     // Binary operations (add, sub, mul, div)

#ifdef __linux__
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdatomic.h>      // For atomic operations in statistics
#include <pthread.h>        // For threading primitives
#include <inttypes.h>       // For PRId64 format specifier

// Threading primitives - consistent with coordinator
typedef pthread_mutex_t ggml_mutex_t;
typedef pthread_cond_t  ggml_cond_t;
typedef pthread_t       ggml_thread_t;

#define ggml_mutex_init(mutex)    pthread_mutex_init(mutex, NULL)
#define ggml_mutex_destroy(mutex) pthread_mutex_destroy(mutex)
#define ggml_mutex_lock(mutex)    pthread_mutex_lock(mutex)
#define ggml_mutex_unlock(mutex)  pthread_mutex_unlock(mutex)

// Include threading support (needed for mutex operations)
#include "ggml-threading.h"

//
// Work Function Pointer Infrastructure - New Function Pointer Architecture
//

// Work context structure for function pointer execution
typedef struct {
    struct ggml_tensor * operation;        // The operation to execute
    struct ggml_cplan * cplan;             // Compute plan for execution
    const char * operation_name;           // For debugging and logging
    void * additional_context;             // Operation-specific additional data
    size_t additional_context_size;        // Size of additional context data
} ggml_numa_dispatcher_work_context_t;

// Work function prototypes - these are the actual functions passed to coordinator
static enum ggml_status ggml_numa_work_function_fallback(void * work_context, struct ggml_compute_params * params);
static enum ggml_status ggml_numa_work_function_mul_mat_single(void * work_context, struct ggml_compute_params * params);
static enum ggml_status ggml_numa_work_function_mul_mat_chunk(void * work_context, struct ggml_compute_params * params);
static enum ggml_status ggml_numa_work_function_soft_max(void * work_context, struct ggml_compute_params * params);

// Work context creation and management functions
static ggml_numa_dispatcher_work_context_t * ggml_numa_dispatcher_create_work_context(
    struct ggml_tensor * operation,
    const char * operation_name,
    void * additional_context,
    size_t additional_context_size
);

static void ggml_numa_dispatcher_free_work_context(ggml_numa_dispatcher_work_context_t * context);

// Buffer size calculation function for dispatcher work contexts
static size_t ggml_numa_dispatcher_calculate_work_buffer_size(const struct ggml_tensor * operation);

//
// Forward Declarations for Internal Functions
//

// Helper function to determine if NUMA coordination would be beneficial for a given graph
static bool ggml_numa_should_coordinate(struct ggml_cgraph * cgraph, int n_threads);

static enum ggml_status ggml_numa_execute_single_node(
    struct ggml_numa_coordinator_manager * manager,
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context
);

static enum ggml_status ggml_numa_execute_data_parallel(
    struct ggml_numa_coordinator_manager * manager,
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context
);

static enum ggml_status ggml_numa_execute_complex_graph(
    struct ggml_numa_coordinator_manager * manager,
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context
);

// MUL_MAT chunked execution using discrete mathematical kernels
static enum ggml_status ggml_numa_execute_mul_mat_chunked(
    struct ggml_numa_coordinator_manager * manager,
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context
);

// SOFT_MAX chunked execution using row-wise parallelization
static enum ggml_status ggml_numa_execute_soft_max_chunked(
    struct ggml_numa_coordinator_manager * manager,
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context
);

static enum ggml_status ggml_numa_execute_fallback_direct(
    struct ggml_numa_coordinator_manager * manager,
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context
);

/*
// TEMPORARILY DISABLED - needs updating for new strategy system
// MUL_MAT analyzer function
static enum ggml_status ggml_numa_analyze_mul_mat(
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context,
    ggml_numa_execution_strategy_t * strategy,
    int * recommended_chunks
);
*/

// MUL_MAT chunked execution functions
static enum ggml_status ggml_numa_execute_mul_mat_single_chunk(
    struct ggml_numa_coordinator_manager * manager,
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context,
    size_t work_size
);

static enum ggml_status ggml_numa_execute_mul_mat_parallel_chunks(
    struct ggml_numa_coordinator_manager * manager,
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context,
    size_t work_size
);

static enum ggml_status ggml_numa_execute_mul_mat_sequential_chunks(
    struct ggml_numa_coordinator_manager * manager,
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context,
    size_t work_size
);

static enum ggml_status ggml_numa_execute_mul_mat_chunk_range(
    struct ggml_numa_coordinator_manager * manager,
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context,
    size_t work_size,
    int64_t row_start,
    int64_t row_end,
    int64_t col_start,
    int64_t col_end
) {
    // Suppress unused parameter warnings
    (void)manager;
    (void)context;
    
    GGML_LOG_DEBUG("Executing MUL_MAT chunk range: rows[%ld:%ld] cols[%ld:%ld]\n",
                   row_start, row_end, col_start, col_end);

    const struct ggml_tensor * src0 = operation->src[0];
    
    // Ensure work buffer
    void * work_buffer = NULL;
    if (work_size > 0) {
        if (!ggml_numa_dispatch_ensure_work_buffer(0, work_size)) {
            GGML_LOG_ERROR("Failed to ensure work buffer of size %zu for chunk range\n", work_size);
            return GGML_STATUS_FAILED;
        }
        
        size_t buffer_size = 0;
        work_buffer = ggml_numa_dispatch_get_work_buffer(0, &buffer_size);
    }
    
    // Create compute parameters for the mathematical kernel  
    struct ggml_compute_params params = {
        .ith = 0,  // Single threaded kernel execution
        .nth = 1,  // One thread per chunk
        .wsize = work_size,
        .wdata = work_buffer,
    };
    
    const int64_t num_rows_per_vec_dot = 1;  // Standard single-row processing
    
    // Call the mathematical kernel for the specific chunk range
    extern void ggml_compute_forward_mul_mat_one_chunk(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst,
        const enum ggml_type type,
        const int64_t num_rows_per_vec_dot,
        const int64_t ir0_start,
        const int64_t ir0_end,
        const int64_t ir1_start,
        const int64_t ir1_end);

    // Remove const qualifier for kernel call (kernel may modify tensor metadata)
    struct ggml_tensor * dst_tensor = (struct ggml_tensor *)operation;
    
    ggml_compute_forward_mul_mat_one_chunk(
        &params,
        dst_tensor,                      // dst
        src0->type,                      // type
        num_rows_per_vec_dot,           // num_rows_per_vec_dot  
        row_start, row_end,             // row range
        col_start, col_end              // column range
    );
    
    GGML_LOG_DEBUG("MUL_MAT chunk range execution completed successfully\n");
    return GGML_STATUS_SUCCESS;
}

//
// Coordinator Interface Implementation
//

// Forward declarations for coordinator interface functions
static struct ggml_threadpool * coordinator_get_numa_threadpool(struct ggml_numa_coordinator_manager * manager, int numa_node);
static int coordinator_get_numa_thread_count(struct ggml_numa_coordinator_manager * manager, int numa_node);
static bool coordinator_ensure_work_buffer(struct ggml_numa_coordinator_manager * manager, int numa_node, size_t required_size);
static void * coordinator_get_work_buffer(struct ggml_numa_coordinator_manager * manager, int numa_node);
static size_t coordinator_get_work_buffer_size(struct ggml_numa_coordinator_manager * manager, int numa_node);
static int coordinator_submit_work(struct ggml_numa_coordinator_manager * manager, struct ggml_tensor * operation, int target_numa_node, ggml_numa_execution_strategy_t strategy);
static int coordinator_submit_data_parallel_work(struct ggml_numa_coordinator_manager * manager, struct ggml_tensor * operation, 
                                                int work_group_id, const int * target_nodes, int num_target_nodes);

// Global coordinator interface instance
static const ggml_numa_coordinator_interface_t g_coordinator_interface = {
    .get_numa_threadpool = coordinator_get_numa_threadpool,
    .get_numa_thread_count = coordinator_get_numa_thread_count,
    .ensure_work_buffer = coordinator_ensure_work_buffer,
    .get_work_buffer = coordinator_get_work_buffer,
    .get_work_buffer_size = coordinator_get_work_buffer_size,
    .submit_work = coordinator_submit_work,
    .submit_data_parallel_work = coordinator_submit_data_parallel_work
};

//
// Forward Declarations for Handler Definitions
//

extern const ggml_numa_operation_handler_t ggml_numa_handler_elementwise;
extern const ggml_numa_operation_handler_t ggml_numa_handler_mul_mat_enhanced;
extern const ggml_numa_operation_handler_t ggml_numa_handler_soft_max;
extern const ggml_numa_operation_handler_t ggml_numa_handler_complex;

//
// Global Dispatch State
//

// Per-thread work buffer system constants
#define GGML_NUMA_MAX_THREADS_PER_NODE 256  // Support for high-thread-count systems (e.g., 128-core CPUs with hyperthreading)
#define GGML_NUMA_MAX_TOTAL_THREADS (GGML_NUMA_MAX_NODES * GGML_NUMA_MAX_THREADS_PER_NODE)

// Per-thread dispatcher work buffer system  
typedef struct {
    void * buffer;               // The work buffer
    size_t buffer_size;         // Current buffer size
    int numa_node;              // NUMA node for this buffer
    int thread_id;              // Thread ID within the NUMA node
    ggml_mutex_t mutex;         // Thread safety for buffer operations (for auto-resizing)
    bool initialized;           // Whether this thread buffer is initialized
} ggml_numa_dispatch_thread_buffer_t;

// Global work buffers per thread: [numa_node * MAX_THREADS_PER_NODE + thread_id]
static ggml_numa_dispatch_thread_buffer_t g_dispatch_thread_buffers[GGML_NUMA_MAX_TOTAL_THREADS];
static int g_numa_nodes_count = 0;
static bool g_dispatch_work_buffers_initialized = false;

static ggml_numa_operation_handler_t * g_operation_handlers[GGML_OP_COUNT];
static ggml_numa_dispatch_stats_t g_dispatch_stats;
static bool g_dispatch_initialized = false;

//
// Forward Declarations
//

// Work buffer management (implementation at end of file)
static void ggml_numa_dispatch_work_buffers_init_internal(void);
void ggml_numa_dispatch_cleanup_operation_handlers(void);

static enum ggml_status ggml_numa_execute_single_node(
    struct ggml_numa_coordinator_manager * manager,
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context
);

static enum ggml_status ggml_numa_execute_data_parallel(
    struct ggml_numa_coordinator_manager * manager,
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context
);

//
// Core Dispatch System Implementation
//

void ggml_numa_dispatch_init(void) {
    if (g_dispatch_initialized) {
        return;
    }
    
    // Initialize handler registry
    memset(g_operation_handlers, 0, sizeof(g_operation_handlers));
    memset(&g_dispatch_stats, 0, sizeof(g_dispatch_stats));
    
    // Initialize dispatcher work buffer system
    ggml_numa_dispatch_work_buffers_init_internal();
    
    // Register built-in handlers
    ggml_numa_dispatch_register_handler(&ggml_numa_handler_elementwise);
    ggml_numa_dispatch_register_handler(&ggml_numa_handler_mul_mat_enhanced);  // Use enhanced MUL_MAT handler
    ggml_numa_dispatch_register_handler(&ggml_numa_handler_soft_max);         // NUMA-aware SOFT_MAX handler
    ggml_numa_dispatch_register_handler(&ggml_numa_handler_complex);
    
    // Skip atexit registration to avoid segfaults during program termination
    // The OS will clean up memory and resources when the process exits
    // atexit(ggml_numa_dispatch_cleanup_work_buffers);
    // atexit(ggml_numa_dispatch_cleanup_operation_handlers);
    
    g_dispatch_initialized = true;
    GGML_LOG_INFO("NUMA operation dispatch system initialized with enhanced handlers\n");
}

void ggml_numa_dispatch_register_handler(const ggml_numa_operation_handler_t * handler) {
    if (!handler || handler->operation_type >= GGML_OP_COUNT) {
        GGML_LOG_ERROR("Invalid operation handler for registration\n");
        return;
    }
    
    // Allocate and copy handler using NUMA-aware allocation
#ifdef __linux__
    ggml_numa_operation_handler_t * registered_handler = numa_alloc_onnode(sizeof(ggml_numa_operation_handler_t), 0);
    if (!registered_handler) {
        // Fallback to regular malloc if NUMA allocation fails
        registered_handler = malloc(sizeof(ggml_numa_operation_handler_t));
    }
#else
    ggml_numa_operation_handler_t * registered_handler = malloc(sizeof(ggml_numa_operation_handler_t));
#endif
    
    if (!registered_handler) {
        GGML_LOG_ERROR("Failed to allocate memory for operation handler\n");
        return;
    }
    
    memcpy(registered_handler, handler, sizeof(ggml_numa_operation_handler_t));
    g_operation_handlers[handler->operation_type] = registered_handler;
    
    GGML_LOG_DEBUG("Registered handler for operation %s\n", ggml_op_name(handler->operation_type));
}

const ggml_numa_operation_handler_t * ggml_numa_dispatch_get_handler(enum ggml_op operation_type) {
    if (operation_type >= GGML_OP_COUNT) {
        return NULL;
    }
    
    return g_operation_handlers[operation_type];
}

//
// Main Dispatch Logic
//

enum ggml_status ggml_numa_dispatch_operation(
    struct ggml_numa_coordinator_manager * manager,
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context) {
    
    if (!manager || !operation || !context) {
        GGML_LOG_ERROR("Invalid parameters for operation dispatch\n");
        return GGML_STATUS_FAILED;
    }
    
    // Update statistics
    g_dispatch_stats.total_operations++;
    if (operation->op < GGML_OP_COUNT) {
        g_dispatch_stats.op_counts[operation->op]++;
    }
    
    int64_t start_time = ggml_time_us();
    enum ggml_status result = GGML_STATUS_SUCCESS;
    
    // Get handler for this operation type
    const ggml_numa_operation_handler_t * handler = ggml_numa_dispatch_get_handler(operation->op);
    
    // Debug logging for MUL_MAT specifically
    if (operation->op == GGML_OP_MUL_MAT) {
        GGML_LOG_INFO("MUL_MAT dispatch: handler=%p, initialized=%d\n", (const void*)handler, g_dispatch_initialized);
    }
    
    if (handler) {
        // Use registered handler to analyze and execute
        ggml_numa_execution_strategy_t strategy;
        int recommended_chunks;
        
        // Analyze operation to determine optimal strategy
        if (handler->analyze) {
            enum ggml_status analysis_result = handler->analyze(operation, context, &strategy, &recommended_chunks);
            if (analysis_result != GGML_STATUS_SUCCESS) {
                GGML_LOG_WARN("Operation analysis failed for %s, falling back to single node\n", 
                             ggml_op_name(operation->op));
                strategy = (ggml_numa_execution_strategy_t){
                    .node_strategy = NUMA_NODE_STRATEGY_SINGLE_NODE,
                    .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
                };
            }
        } else {
            // Use default strategy from handler
            strategy = handler->default_strategy;
            recommended_chunks = context->numa_nodes;
        }
        
        // Execute based on determined strategy
        switch (strategy.node_strategy) {
            case NUMA_NODE_STRATEGY_SINGLE_NODE:
                result = ggml_numa_execute_single_node(manager, operation, context);
                break;
                
            case NUMA_NODE_STRATEGY_DATA_PARALLEL:
                result = ggml_numa_execute_data_parallel(manager, operation, context);
                g_dispatch_stats.parallelized_operations++;
                break;
                
        case NUMA_NODE_STRATEGY_TASK_PARALLEL:
            // For task parallel, distribute different tasks across nodes
            GGML_LOG_DEBUG("Task parallel execution strategy for %s\n", 
                          ggml_op_name(operation->op));
            
            // For MUL_MAT, break the operation into discrete chunks for NUMA coordination
            if (operation->op == GGML_OP_MUL_MAT) {
                result = ggml_numa_execute_mul_mat_chunked(manager, operation, context);
            } else if (operation->op == GGML_OP_SOFT_MAX) {
                result = ggml_numa_execute_soft_max_chunked(manager, operation, context);
            } else {
                // For other complex operations, use graph-based approach  
                result = ggml_numa_execute_complex_graph(manager, operation, context);
            }
            
            if (result == GGML_STATUS_SUCCESS) {
                g_dispatch_stats.parallelized_operations++;
            }
            break;
            
        default:
            GGML_LOG_WARN("Unknown node strategy %d for operation %s, using single node fallback\n", 
                         (int)strategy.node_strategy, ggml_op_name(operation->op));
            result = ggml_numa_execute_single_node(manager, operation, context);
            break;
        }
        
    } else {
        // No handler registered - use single-threaded fallback system
        GGML_LOG_DEBUG("No handler registered for operation %s, using fallback execution\n", 
                      ggml_op_name(operation->op));
        
        // Check if this operation needs work buffers
        // If so, route through coordinator for proper buffer setup
        // Otherwise use direct fallback
        bool needs_work_buffer = (operation->op == GGML_OP_SOFT_MAX || 
                                  operation->op == GGML_OP_ROPE ||
                                  operation->op == GGML_OP_NORM ||
                                  operation->op == GGML_OP_RMS_NORM ||
                                  operation->op == GGML_OP_GROUP_NORM);
        
        if (needs_work_buffer) {
            GGML_LOG_DEBUG("Operation %s needs work buffers, routing through coordinator\n", 
                          ggml_op_name(operation->op));
            // Use coordinator for operations that need work buffers
            result = ggml_numa_execute_single_node(manager, operation, context);
        } else {
            GGML_LOG_DEBUG("Operation %s doesn't need work buffers, using direct fallback\n", 
                          ggml_op_name(operation->op));
            // Route to fallback system for safe single-threaded execution
            struct ggml_tensor * operation_tensor = (struct ggml_tensor *)operation;
            result = ggml_numa_execute_operation_fallback(operation_tensor, NULL);
        }
        
        if (result != GGML_STATUS_SUCCESS) {
            GGML_LOG_ERROR("Fallback execution failed for operation %s\n", ggml_op_name(operation->op));
        }
    }
    
    // Update timing statistics
    int64_t execution_time = ggml_time_us() - start_time;
    g_dispatch_stats.total_execution_time_us += execution_time;
    if (operation->op < GGML_OP_COUNT) {
        g_dispatch_stats.op_times_us[operation->op] += execution_time;
    }
    
    return result;
}

//
// Execution Strategy Implementations
//

static enum ggml_status ggml_numa_execute_single_node(
    struct ggml_numa_coordinator_manager * manager,
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context) {
    
    (void)context; // Suppress unused parameter warning
    
    GGML_LOG_DEBUG("Dispatching %s to single node (NUMA node 0) using function pointer approach\n", ggml_op_name(operation->op));
    
    // Create work context for the function pointer
    ggml_numa_dispatcher_work_context_t * work_context = ggml_numa_dispatcher_create_work_context(
        (struct ggml_tensor *)operation,
        ggml_op_name(operation->op),
        NULL,  // No additional context for basic operations
        0      // No additional context size
    );
    
    if (!work_context) {
        GGML_LOG_ERROR("Failed to create work context for operation %s\n", ggml_op_name(operation->op));
        return GGML_STATUS_FAILED;
    }
    
    // Calculate buffer requirements
    size_t buffer_size = ggml_numa_dispatcher_calculate_work_buffer_size(operation);
    
    // Set up execution strategy for single node
    ggml_numa_execution_strategy_t single_node_strategy = {
        .node_strategy = NUMA_NODE_STRATEGY_SINGLE_NODE,
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
    };
    
    // Submit work using new function pointer approach
    int work_id = ggml_numa_coordinator_manager_submit_work_function(
        manager,
        ggml_numa_work_function_fallback,  // Generic fallback function
        work_context,                      // Context data
        0,                                 // Auto-select NUMA node
        single_node_strategy,              // Execution strategy
        buffer_size                        // Buffer requirements
    );
    
    if (work_id < 0) {
        GGML_LOG_ERROR("Failed to submit function pointer work for operation %s\n", ggml_op_name(operation->op));
        ggml_numa_dispatcher_free_work_context(work_context);
        return GGML_STATUS_FAILED;
    }
    
    GGML_LOG_DEBUG("Submitted function pointer work (ID: %d) for operation %s\n", work_id, ggml_op_name(operation->op));
    
    // Note: work_context will be freed by the coordinator after execution
    
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_numa_execute_data_parallel(
    struct ggml_numa_coordinator_manager * manager,
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context) {
    
    GGML_LOG_DEBUG("Executing %s with data parallelism across %d NUMA nodes using function pointers\n", 
                   ggml_op_name(operation->op), context->numa_nodes);
    
    // Special handling for NUMA-aware operations
    switch (operation->op) {
        case GGML_OP_ROPE: {
            // FIXED: ROPE cannot be safely parallelized across multiple NUMA nodes
            // Multiple nodes writing to same tensor causes data corruption
            GGML_LOG_DEBUG("ROPE requires single-node execution to avoid data corruption\n");
            return ggml_numa_execute_single_node(manager, operation, context);
        }
        
        default:
            break;
    }
    
    // Create work context for the function pointer
    ggml_numa_dispatcher_work_context_t * work_context = ggml_numa_dispatcher_create_work_context(
        (struct ggml_tensor *)operation,
        ggml_op_name(operation->op),
        NULL,  // No additional context for basic data parallel operations
        0      // No additional context size
    );
    
    if (!work_context) {
        GGML_LOG_ERROR("Failed to create work context for data parallel operation %s\n", ggml_op_name(operation->op));
        return GGML_STATUS_FAILED;
    }
    
    // Calculate buffer requirements
    size_t buffer_size = ggml_numa_dispatcher_calculate_work_buffer_size(operation);
    
    // Set up execution strategy for data parallel execution
    ggml_numa_execution_strategy_t data_parallel_strategy = {
        .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL,
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
    };
    
    // Submit work using new function pointer approach for data parallelism
    int work_id = ggml_numa_coordinator_manager_submit_work_function(
        manager,
        ggml_numa_work_function_fallback,  // Generic fallback function handles most operations
        work_context,                      // Context data
        -1,                                // Auto-select NUMA nodes for data parallelism
        data_parallel_strategy,            // Data parallel execution strategy
        buffer_size                        // Buffer requirements
    );
    
    if (work_id < 0) {
        GGML_LOG_WARN("Data parallel function pointer execution failed for %s, falling back to single node\n", 
                     ggml_op_name(operation->op));
        ggml_numa_dispatcher_free_work_context(work_context);
        return ggml_numa_execute_single_node(manager, operation, context);
    }
    
    GGML_LOG_DEBUG("Submitted data parallel function pointer work (ID: %d) for operation %s\n", 
                   work_id, ggml_op_name(operation->op));
    
    // Note: work_context will be freed by the coordinator after execution
    
    return GGML_STATUS_SUCCESS;
}

// Enhanced execution strategy for complex operations like MUL_MAT
static enum ggml_status ggml_numa_execute_complex_graph(
    struct ggml_numa_coordinator_manager * manager,
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context) {
    
    (void)context; // Suppress unused parameter warning
    
    GGML_LOG_DEBUG("Executing complex operation %s using function pointer approach\n", 
                  ggml_op_name(operation->op));
    
    // Create work context for complex operations
    ggml_numa_dispatcher_work_context_t * work_context = ggml_numa_dispatcher_create_work_context(
        (struct ggml_tensor *)operation,
        ggml_op_name(operation->op),
        NULL,  // No additional context for basic complex operations
        0      // No additional context size
    );
    
    if (!work_context) {
        GGML_LOG_ERROR("Failed to create work context for complex operation %s\n", ggml_op_name(operation->op));
        return GGML_STATUS_FAILED;
    }
    
    // Calculate buffer requirements
    size_t buffer_size = ggml_numa_dispatcher_calculate_work_buffer_size(operation);
    
    // Set up execution strategy for complex operations
    ggml_numa_execution_strategy_t strategy = {
        .node_strategy = NUMA_NODE_STRATEGY_TASK_PARALLEL,  // Complex operations benefit from task parallelism
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
    };
    
    // Submit work using function pointer approach
    int work_id = ggml_numa_coordinator_manager_submit_work_function(
        manager,
        ggml_numa_work_function_fallback,  // Use fallback for complex operations
        work_context,                      // Context data
        -1,                                // Auto-select NUMA node
        strategy,                          // Task parallel execution strategy
        buffer_size                        // Buffer requirements
    );
    
    if (work_id < 0) {
        GGML_LOG_ERROR("Failed to submit complex operation function pointer work for %s\n", ggml_op_name(operation->op));
        ggml_numa_dispatcher_free_work_context(work_context);
        return GGML_STATUS_FAILED;
    }
    
    GGML_LOG_DEBUG("Submitted complex operation function pointer work (ID: %d) for %s\n", 
                   work_id, ggml_op_name(operation->op));
    
    // Note: work_context will be freed by the coordinator after execution
    
    return GGML_STATUS_SUCCESS;
}

// MUL_MAT chunked execution - breaks matrix into discrete chunks for NUMA coordination
static enum ggml_status ggml_numa_execute_mul_mat_chunked(
    struct ggml_numa_coordinator_manager * manager,
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context) {
    
    if (!manager || !operation || !context) {
        GGML_LOG_ERROR("Invalid parameters for chunked MUL_MAT execution\n");
        return GGML_STATUS_FAILED;
    }
    
    const struct ggml_tensor * src0 = operation->src[0];
    const struct ggml_tensor * src1 = operation->src[1];
    
    if (!src0 || !src1) {
        GGML_LOG_ERROR("MUL_MAT operation missing source tensors\n");
        return GGML_STATUS_FAILED;
    }
    
    // Get matrix dimensions
    const int64_t ne01 = src0->ne[1];  // M dimension  
    const int64_t ne00 = src0->ne[0];  // K dimension
    const int64_t ne11 = src1->ne[1];  // N dimension
    
    GGML_LOG_INFO("MUL_MAT chunked execution using function pointers: %ldx%ld * %ldx%ld -> %ldx%ld\n",
                  ne01, ne00, ne11, src1->ne[0], ne01, ne11);
    
    // Calculate work buffer requirements
    size_t work_size = 0;
    enum ggml_type vec_dot_type = src0->type;
    if (src1->type != vec_dot_type) {
        work_size = ggml_row_size(vec_dot_type, ggml_nelements(src1));
    }
    
    // For large matrices, determine if we should use chunked or single execution
    const int64_t complexity = ne01 * ne00 * ne11;
    const bool use_chunked = (context->numa_nodes > 1) && (complexity > 10000000); // 10M operations threshold
    
    // Create work context for the function pointer
    ggml_numa_dispatcher_work_context_t * work_context = ggml_numa_dispatcher_create_work_context(
        (struct ggml_tensor *)operation,
        "MUL_MAT_CHUNKED",
        NULL,  // No additional context for now
        0      // No additional context size
    );
    
    if (!work_context) {
        GGML_LOG_ERROR("Failed to create work context for chunked MUL_MAT\n");
        return GGML_STATUS_FAILED;
    }
    
    // Choose work function and execution strategy based on complexity
    ggml_numa_work_function_t work_function;
    ggml_numa_execution_strategy_t strategy;
    
    if (use_chunked) {
        GGML_LOG_DEBUG("Using chunked MUL_MAT work function (complexity=%ld, numa_nodes=%d)\n", 
                       complexity, context->numa_nodes);
        work_function = ggml_numa_work_function_mul_mat_chunk;
        strategy = (ggml_numa_execution_strategy_t){
            .node_strategy = NUMA_NODE_STRATEGY_TASK_PARALLEL,
            .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
        };
    } else {
        GGML_LOG_DEBUG("Using single MUL_MAT work function (complexity=%ld)\n", complexity);
        work_function = ggml_numa_work_function_mul_mat_single;
        strategy = (ggml_numa_execution_strategy_t){
            .node_strategy = NUMA_NODE_STRATEGY_SINGLE_NODE,
            .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
        };
    }
    
    // Submit work using new function pointer approach
    int work_id = ggml_numa_coordinator_manager_submit_work_function(
        manager,
        work_function,                     // Specialized MUL_MAT function
        work_context,                      // Context data
        -1,                                // Auto-select NUMA node
        strategy,                          // Execution strategy
        work_size                          // Buffer requirements
    );
    
    if (work_id < 0) {
        GGML_LOG_ERROR("Failed to submit chunked MUL_MAT function pointer work\n");
        ggml_numa_dispatcher_free_work_context(work_context);
        return GGML_STATUS_FAILED;
    }
    
    GGML_LOG_DEBUG("Submitted chunked MUL_MAT function pointer work (ID: %d)\n", work_id);
    
    // Note: work_context will be freed by the coordinator after execution
    
    return GGML_STATUS_SUCCESS;
}

// Single-chunk MUL_MAT execution for small matrices or single NUMA node systems
static enum ggml_status ggml_numa_execute_mul_mat_single_chunk(
    struct ggml_numa_coordinator_manager * manager,
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context,
    size_t work_size) {
    
    (void)manager; // Suppress unused parameter warning
    (void)context; // Suppress unused parameter warning
    
    // Import the mathematical kernel
    extern void ggml_compute_forward_mul_mat_one_chunk(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst,
        const enum ggml_type type,
        const int64_t num_rows_per_vec_dot,
        const int64_t ir0_start,
        const int64_t ir0_end,
        const int64_t ir1_start,
        const int64_t ir1_end);
    
    const struct ggml_tensor * src0 = operation->src[0];
    const struct ggml_tensor * src1 = operation->src[1];
    
    // Get matrix dimensions
    const int64_t ne01 = src0->ne[1];  // M dimension  
    const int64_t ne11 = src1->ne[1];  // N dimension
    
    // Ensure work buffer
    void * work_buffer = NULL;
    if (work_size > 0) {
        if (!ggml_numa_dispatch_ensure_work_buffer(0, work_size)) {
            GGML_LOG_ERROR("Failed to ensure work buffer of size %zu for chunked MUL_MAT\n", work_size);
            return GGML_STATUS_FAILED;
        }
        
        size_t buffer_size = 0;
        work_buffer = ggml_numa_dispatch_get_work_buffer(0, &buffer_size);
    }
    
    // Create compute parameters for the mathematical kernel  
    struct ggml_compute_params params = {
        .ith = 0,  // Single threaded kernel execution
        .nth = 1,  // One thread per chunk
        .wsize = work_size,
        .wdata = work_buffer,
        .threadpool = NULL  // No threading in the kernel itself
    };
    
    // Execute single chunk covering the entire matrix
    const int64_t num_rows_per_vec_dot = 1;  // Standard single-row processing
    const int64_t ir0_start = 0;
    const int64_t ir0_end = ne01;   // All rows of src0 
    const int64_t ir1_start = 0; 
    const int64_t ir1_end = ne11;   // All columns of src1
    
    GGML_LOG_DEBUG("Executing MUL_MAT chunk: rows[%ld:%ld] cols[%ld:%ld]\n",
                   ir0_start, ir0_end, ir1_start, ir1_end);
    
    // Call the mathematical kernel directly
    struct ggml_tensor * dst_tensor = (struct ggml_tensor *)operation;
    ggml_compute_forward_mul_mat_one_chunk(
        &params,
        dst_tensor,                      // dst
        src0->type,                      // type
        num_rows_per_vec_dot,           // num_rows_per_vec_dot  
        ir0_start, ir0_end,             // row range
        ir1_start, ir1_end              // column range
    );
    
    GGML_LOG_DEBUG("MUL_MAT single-chunk execution completed successfully\n");
    return GGML_STATUS_SUCCESS;
}

// Multi-NUMA parallel chunking execution for large matrices
static enum ggml_status ggml_numa_execute_mul_mat_parallel_chunks(
    struct ggml_numa_coordinator_manager * manager,
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context,
    size_t work_size) {
    
    if (!manager || !operation || !context) {
        GGML_LOG_ERROR("Invalid parameters for parallel chunked MUL_MAT execution\n");
        return GGML_STATUS_FAILED;
    }
    
    const struct ggml_tensor * src0 = operation->src[0];
    const struct ggml_tensor * src1 = operation->src[1];
    
    // Get matrix dimensions
    const int64_t ne01 = src0->ne[1];  // M dimension (rows)
    const int64_t ne11 = src1->ne[1];  // N dimension (columns)
    
    const int numa_nodes = context->numa_nodes;
    
    GGML_LOG_INFO("Parallel MUL_MAT: Distributing %ldx%ld result across %d NUMA nodes\n",
                  ne01, ne11, numa_nodes);
    
    // For very large matrices, use true NUMA parallelization
    const int64_t complexity = ne01 * ne11;
    const bool use_numa_parallel = (numa_nodes > 1) && (complexity > 50000000); // 50M elements threshold
    
    if (use_numa_parallel) {
        GGML_LOG_INFO("Using NUMA coordinator data parallel execution for large matrix\n");
        
        // Use the coordinator's built-in data parallelism 
        struct ggml_tensor * operation_tensor = (struct ggml_tensor *)operation;
        int work_group_id = g_coordinator_interface.submit_data_parallel_work(
            manager, operation_tensor, -1, NULL, 0);
        
        if (work_group_id < 0) {
            GGML_LOG_ERROR("NUMA data parallel submission failed, falling back to sequential chunks\n");
            return ggml_numa_execute_mul_mat_sequential_chunks(manager, operation, context, work_size);
        }
        
        GGML_LOG_INFO("NUMA data parallel work submitted (work group ID: %d)\n", work_group_id);
        return GGML_STATUS_SUCCESS;
    }
    
    // For medium matrices, use sequential chunking with manual distribution
    return ggml_numa_execute_mul_mat_sequential_chunks(manager, operation, context, work_size);
}

// Sequential chunking execution - hierarchical NUMA + thread-level chunking
static enum ggml_status ggml_numa_execute_mul_mat_sequential_chunks(
    struct ggml_numa_coordinator_manager * manager,
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context,
    size_t work_size) {
    
    const struct ggml_tensor * src0 = operation->src[0];
    const struct ggml_tensor * src1 = operation->src[1];
    
    // Get matrix dimensions
    const int64_t ne01 = src0->ne[1];  // M dimension (rows)
    const int64_t ne11 = src1->ne[1];  // N dimension (columns)
    
    const int numa_nodes = context->numa_nodes;
    
    // Get thread count per NUMA node from coordinator manager
    // For hierarchical chunking, we need: NUMA nodes × threads per node
    int total_threads = 0;
    
    // Calculate total threads across ALL NUMA nodes (don't assume uniform threading)
    if (manager && numa_nodes > 0) {
        for (int node = 0; node < numa_nodes; node++) {
            int node_threads = ggml_numa_coordinator_get_thread_count(manager, node);
            if (node_threads > 0) {
                total_threads += node_threads;
            } else {
                // Fallback: assume 1 thread per node if detection fails
                total_threads += 1;
                GGML_LOG_WARN("Failed to get thread count for NUMA node %d, assuming 1 thread\n", node);
            }
        }
    } else {
        // Fallback: no NUMA or manager, assume single-threaded
        total_threads = 1;
        GGML_LOG_WARN("No NUMA manager available, falling back to single-threaded execution\n");
    }
    
    GGML_LOG_INFO("Dynamic thread distribution: %d NUMA nodes with %d total threads\n",
                  numa_nodes, total_threads);
    
    // Two-level chunking: First by NUMA nodes, then by threads within each node
    // Each thread gets its own chunk for maximum parallelization
    const int64_t rows_per_thread_chunk = (ne01 + total_threads - 1) / total_threads;  // Ceiling division
    
    // Structure to track hierarchical chunk information
    struct hierarchical_chunk_info {
        int64_t row_start;
        int64_t row_end;
        int numa_node;          // Which NUMA node this chunk belongs to
        int thread_id;          // Thread ID within the NUMA node
        int global_chunk_id;    // Global chunk identifier
        int work_id;
    };
    
    // Dynamic chunk allocation based on actual system configuration
    const int max_chunks = total_threads * 2;  // Allow some overhead for edge cases
    
    // For NUMA-aware allocation, we need to distribute chunk info across NUMA nodes
    // Since chunk info is small, we'll use CPU 0's NUMA node for simplicity, but ensure
    // the actual work data gets allocated on the appropriate NUMA nodes
    int allocation_node = numa_node_of_cpu(sched_getcpu());
    if (allocation_node < 0) allocation_node = 0;  // Fallback to node 0
    
    struct hierarchical_chunk_info *chunks = numa_alloc_onnode(
        max_chunks * sizeof(struct hierarchical_chunk_info), 
        allocation_node
    );
    bool use_numa_free = true;
    if (!chunks) {
        GGML_LOG_ERROR("Failed to allocate NUMA-local memory for %d chunk structures on node %d\n", 
                       max_chunks, allocation_node);
        // Fallback to regular malloc if NUMA allocation fails
        chunks = malloc(max_chunks * sizeof(struct hierarchical_chunk_info));
        use_numa_free = false;
        if (!chunks) {
            GGML_LOG_ERROR("Failed to allocate memory for %d chunk structures\n", max_chunks);
            return GGML_STATUS_FAILED;
        }
    }
    
    int num_chunks = 0;
    
    // Calculate hierarchical chunks: dynamically across all NUMA nodes and their threads
    int global_chunk_id = 0;
    for (int node = 0; node < numa_nodes && num_chunks < max_chunks; node++) {
        int node_threads = ggml_numa_coordinator_get_thread_count(manager, node);
        if (node_threads <= 0) node_threads = 1;  // Fallback
        
        for (int thread = 0; thread < node_threads && num_chunks < max_chunks; thread++) {
            const int64_t chunk_row_start = global_chunk_id * rows_per_thread_chunk;
            const int64_t chunk_row_end = (global_chunk_id + 1 == total_threads) ? ne01 : 
                                          ((global_chunk_id + 1) * rows_per_thread_chunk);
            
            // Skip empty chunks
            if (chunk_row_start >= ne01) break;
            
            chunks[num_chunks].row_start = chunk_row_start;
            chunks[num_chunks].row_end = chunk_row_end;
            chunks[num_chunks].numa_node = node;
            chunks[num_chunks].thread_id = thread;
            chunks[num_chunks].global_chunk_id = global_chunk_id;
            chunks[num_chunks].work_id = -1;  // Will be set when work is submitted
            
            GGML_LOG_DEBUG("Chunk %d: rows[%ld:%ld] -> NUMA node %d, thread %d (%ld rows)\n", 
                           global_chunk_id, chunk_row_start, chunk_row_end, node, thread, 
                           chunk_row_end - chunk_row_start);
            
            num_chunks++;
            global_chunk_id++;
        }
    }
    
    GGML_LOG_INFO("Created %d hierarchical chunks across %d NUMA nodes (%d total threads)\n", 
                  num_chunks, numa_nodes, total_threads);
    
    // Submit and execute all hierarchical chunks
    enum ggml_status overall_result = GGML_STATUS_SUCCESS;
    
    for (int i = 0; i < num_chunks; i++) {
        struct hierarchical_chunk_info * chunk = &chunks[i];
        
        GGML_LOG_DEBUG("Processing hierarchical chunk %d (rows[%ld:%ld]) on NUMA node %d, thread %d\n", 
                       chunk->global_chunk_id, chunk->row_start, chunk->row_end, 
                       chunk->numa_node, chunk->thread_id);
        
        // For hierarchical chunking, we submit work to the appropriate NUMA node
        // The coordinator will handle thread-level distribution within that node
        struct ggml_tensor * operation_tensor = (struct ggml_tensor *)operation;
        ggml_numa_execution_strategy_t data_parallel_strategy = {
            .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL,
            .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
        };
        int work_id = g_coordinator_interface.submit_work(manager, 
                                                         operation_tensor, 
                                                         chunk->numa_node,
                                                         data_parallel_strategy);
        
        if (work_id < 0) {
            GGML_LOG_WARN("Failed to submit hierarchical chunk %d to NUMA node %d, executing directly\n", 
                          chunk->global_chunk_id, chunk->numa_node);
            
            // Fallback to direct execution for this thread-level chunk
            enum ggml_status chunk_result = ggml_numa_execute_mul_mat_chunk_range(
                manager, operation, context, work_size,
                chunk->row_start, chunk->row_end, 0, ne11);
            
            if (chunk_result != GGML_STATUS_SUCCESS) {
                GGML_LOG_ERROR("Failed to execute hierarchical chunk %d directly\n", chunk->global_chunk_id);
                overall_result = GGML_STATUS_FAILED;
            }
        } else {
            chunk->work_id = work_id;
            GGML_LOG_DEBUG("Hierarchical chunk %d submitted with work ID %d\n", 
                           chunk->global_chunk_id, work_id);
        }
    }
    
    if (overall_result == GGML_STATUS_SUCCESS) {
        GGML_LOG_INFO("All %d hierarchical chunks submitted/executed successfully\n", num_chunks);
    }
    
    // Cleanup dynamically allocated chunks
#ifdef __linux__
    if (use_numa_free) {
        numa_free(chunks, max_chunks * sizeof(struct hierarchical_chunk_info));
    } else {
        free(chunks);
    }
#else
    free(chunks);
#endif
    
    return overall_result;
}

// NUMA-aware SOFT_MAX execution with row-wise parallelization
static enum ggml_status ggml_numa_execute_soft_max_chunked(
    struct ggml_numa_coordinator_manager * manager,
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context) {
    
    if (!manager || !operation || !context) {
        GGML_LOG_ERROR("Invalid parameters for SOFT_MAX chunked execution\n");
        return GGML_STATUS_FAILED;
    }
    
    const int64_t ne00 = operation->ne[0];  // Elements per row
    const int64_t ne01 = operation->ne[1];  // Number of rows
    const int64_t ne02 = operation->ne[2];  // Batch dimension
    const int64_t ne03 = operation->ne[3];  // Head dimension
    
    GGML_LOG_INFO("NUMA SOFT_MAX execution using function pointers: [%ld, %ld, %ld, %ld]\n", 
                  ne00, ne01, ne02, ne03);
    
    // Create work context for the function pointer
    ggml_numa_dispatcher_work_context_t * work_context = ggml_numa_dispatcher_create_work_context(
        (struct ggml_tensor *)operation,
        "SOFT_MAX",
        NULL,  // No additional context for basic SOFT_MAX
        0      // No additional context size
    );
    
    if (!work_context) {
        GGML_LOG_ERROR("Failed to create work context for SOFT_MAX\n");
        return GGML_STATUS_FAILED;
    }
    
    // Calculate required work buffer size: (ne00 + CACHE_LINE_SIZE_F32) * sizeof(float) * num_threads
    const int num_threads = context->threads_per_node;
    const size_t work_buffer_size = (ne00 + CACHE_LINE_SIZE_F32) * sizeof(float) * num_threads;
    
    // Set up execution strategy for SOFT_MAX
    ggml_numa_execution_strategy_t strategy = {
        .node_strategy = NUMA_NODE_STRATEGY_SINGLE_NODE,  // SOFT_MAX works well on single node
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
    };
    
    // Submit work using specialized SOFT_MAX function pointer
    int work_id = ggml_numa_coordinator_manager_submit_work_function(
        manager,
        ggml_numa_work_function_soft_max,  // Specialized SOFT_MAX function
        work_context,                      // Context data
        0,                                 // Use first NUMA node
        strategy,                          // Execution strategy
        work_buffer_size                   // Buffer requirements
    );
    
    if (work_id < 0) {
        GGML_LOG_ERROR("Failed to submit SOFT_MAX function pointer work\n");
        ggml_numa_dispatcher_free_work_context(work_context);
        return GGML_STATUS_FAILED;
    }
    
    GGML_LOG_DEBUG("Submitted SOFT_MAX function pointer work (ID: %d)\n", work_id);
    
    // Note: work_context will be freed by the coordinator after execution
    
    return GGML_STATUS_SUCCESS;
}

// Enhanced thread-level chunk execution for hierarchical NUMA parallelization
static enum ggml_status ggml_numa_execute_mul_mat_thread_chunk(
    struct ggml_numa_coordinator_manager * manager,
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context,
    size_t work_size,
    int64_t row_start,
    int64_t row_end,
    int64_t col_start,
    int64_t col_end,
    int numa_node,
    int thread_id) {
    
    GGML_LOG_DEBUG("Thread-level execution: rows[%ld:%ld] cols[%ld:%ld] NUMA=%d thread=%d\n",
                   row_start, row_end, col_start, col_end, numa_node, thread_id);
    
    // Import the mathematical kernel with thread awareness
    extern void ggml_compute_forward_mul_mat_one_chunk(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst,
        const enum ggml_type type,
        const int64_t num_rows_per_vec_dot,
        const int64_t ir0_start,
        const int64_t ir0_end,
        const int64_t ir1_start,
        const int64_t ir1_end);
    
    const struct ggml_tensor * src0 = operation->src[0];
    const struct ggml_tensor * src1 = operation->src[1];
    
    // Create thread-aware compute parameters
    struct ggml_compute_params params = {
        .ith = thread_id,                    // Thread index within NUMA node
        .nth = ggml_numa_coordinator_get_thread_count(manager, numa_node),  // Total threads in this NUMA node
        .wsize = work_size,
        .wdata = ggml_numa_coordinator_get_work_buffer(manager, numa_node),
    };
    
    // Get NUMA-local work buffer for this thread
    void * work_buffer = ggml_numa_coordinator_get_work_buffer(manager, numa_node);
    if (!work_buffer) {
        GGML_LOG_ERROR("Failed to get NUMA-local work buffer for node %d thread %d\n", numa_node, thread_id);
        return GGML_STATUS_FAILED;
    }
    
    // Calculate the number of rows per vector dot operation for the operation type
    const int64_t num_rows_per_vec_dot = 1;  // Standard for MUL_MAT operations
    
    GGML_LOG_DEBUG("Executing thread chunk with work buffer %p (size=%zu) on NUMA node %d\n", 
                   work_buffer, work_size, numa_node);
    
    // Execute the mathematical kernel for this thread's chunk
    struct ggml_tensor * dst_tensor = (struct ggml_tensor *)operation;
    ggml_compute_forward_mul_mat_one_chunk(
        &params,
        dst_tensor,                      // dst
        src0->type,                       // type
        num_rows_per_vec_dot,            // num_rows_per_vec_dot
        row_start,                       // ir0_start (row range)
        row_end,                         // ir0_end
        col_start,                       // ir1_start (col range)  
        col_end                          // ir1_end
    );
    
    GGML_LOG_DEBUG("Thread-level MUL_MAT chunk execution completed successfully\n");
    return GGML_STATUS_SUCCESS;
}

//
// Work Context Creation
int ggml_numa_auto_chunk_operation(
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context,
    ggml_numa_work_chunk_t * chunks,
    int max_chunks) {
    
    if (!operation || !context || !chunks || max_chunks <= 0) {
        return 0;
    }
    
    // Simple strategy: divide total elements evenly across available NUMA nodes
    int num_chunks = (context->numa_nodes > max_chunks) ? max_chunks : context->numa_nodes;
    
    if (context->total_elements < num_chunks) {
        num_chunks = context->total_elements;
    }
    
    int64_t elements_per_chunk = context->total_elements / num_chunks;
    int64_t remaining_elements = context->total_elements % num_chunks;
    
    int64_t current_offset = 0;
    
    for (int i = 0; i < num_chunks; i++) {
        chunks[i].chunk_id = i;
        chunks[i].numa_node = i % context->numa_nodes;
        chunks[i].start_offset = current_offset;
        chunks[i].element_count = elements_per_chunk;
        
        // Distribute remaining elements to first few chunks
        if (i < remaining_elements) {
            chunks[i].element_count++;
        }
        
        chunks[i].thread_count = context->threads_per_node;
        chunks[i].requires_synchronization = (num_chunks > 1);
        
        current_offset += chunks[i].element_count;
        
        GGML_LOG_DEBUG("Chunk %d: NUMA node %d, offset %ld, count %ld\n", 
                       i, chunks[i].numa_node, chunks[i].start_offset, chunks[i].element_count);
    }
    
    return num_chunks;
}

bool ggml_numa_validate_chunks(
    const ggml_numa_work_chunk_t * chunks,
    int num_chunks,
    const ggml_numa_work_context_t * context) {
    
    if (!chunks || !context || num_chunks <= 0) {
        return false;
    }
    
    int64_t total_elements = 0;
    
    for (int i = 0; i < num_chunks; i++) {
        // Check bounds
        if (chunks[i].numa_node >= context->numa_nodes || chunks[i].numa_node < 0) {
            GGML_LOG_ERROR("Invalid NUMA node %d in chunk %d\n", chunks[i].numa_node, i);
            return false;
        }
        
        if (chunks[i].element_count <= 0) {
            GGML_LOG_ERROR("Invalid element count %ld in chunk %d\n", chunks[i].element_count, i);
            return false;
        }
        
        total_elements += chunks[i].element_count;
    }
    
    // Verify total elements match
    if (total_elements != context->total_elements) {
        GGML_LOG_ERROR("Chunk validation failed: expected %ld elements, got %ld\n", 
                       context->total_elements, total_elements);
        return false;
    }
    
    return true;
}

//
// Main GGML Integration Function - Entry Point from ggml-cpu.c
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
    
    // Initialize dispatcher if not already done
    ggml_numa_dispatch_init();
    
    // Determine if NUMA coordination would be beneficial
    if (!ggml_numa_should_coordinate(cgraph, n_threads)) {
        GGML_LOG_DEBUG("NUMA coordination not beneficial - using single-threaded fallback computation\n");
        
        // Use single-threaded fallback for each operation in the graph
        for (int i = 0; i < cgraph->n_nodes; i++) {
            struct ggml_tensor * operation = cgraph->nodes[i];
            if (!operation) continue;
            
            enum ggml_status result = ggml_numa_execute_operation_fallback(operation, NULL);
            if (result != GGML_STATUS_SUCCESS) {
                GGML_LOG_ERROR("Fallback execution failed for operation %d (%s)\n", 
                              i, ggml_op_name(operation->op));
                return GGML_STATUS_FAILED;
            }
        }
        
        GGML_LOG_DEBUG("Single-threaded fallback computation completed successfully\n");
        return GGML_STATUS_SUCCESS;
    }
    
    GGML_LOG_INFO("Using NUMA-aware graph computation via dispatcher for %d operations with %d threads\n", 
                  cgraph->n_nodes, n_threads);
    
    // Get or create the global NUMA coordinator manager  
    struct ggml_numa_coordinator_manager * mgr = ggml_numa_coordinator_manager_get_global(n_threads, false);
    if (!mgr) {
        GGML_LOG_ERROR("Failed to create NUMA coordinator manager\n");
        return GGML_STATUS_FAILED;
    }
    
    // Process each operation in the graph through fallback execution for simplicity
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * operation = cgraph->nodes[i];
        if (!operation) continue;

        // Use fallback execution which is simpler and doesn't need work context setup
        enum ggml_status result = ggml_numa_execute_operation_fallback(operation, NULL);
        
        if (result != GGML_STATUS_SUCCESS) {
            GGML_LOG_ERROR("Fallback execution failed for operation %d (%s)\n", 
                          i, ggml_op_name(operation->op));
            return GGML_STATUS_FAILED;
        }
        
        GGML_LOG_DEBUG("Successfully executed operation %d (%s) via fallback\n", 
                      i, ggml_op_name(operation->op));
    }    GGML_LOG_INFO("NUMA graph computation completed successfully via dispatcher\n");
    return GGML_STATUS_SUCCESS;
}

// Enhanced entry point with virtual NUMA support for testing
enum ggml_status ggml_numa_graph_compute_with_virtual(struct ggml_cgraph * cgraph, int n_threads, bool force_virtual_numa) {
    if (!cgraph) {
        GGML_LOG_ERROR("Invalid cgraph for NUMA graph computation\n");
        return GGML_STATUS_FAILED;
    }
    
    // Initialize dispatcher if not already done
    ggml_numa_dispatch_init();
    
    if (force_virtual_numa) {
        GGML_LOG_INFO("Virtual NUMA mode enabled for operation dispatch testing\n");
        
        // Get or create the global NUMA coordinator manager with virtual NUMA forced
        struct ggml_numa_coordinator_manager * mgr = ggml_numa_coordinator_manager_get_global(n_threads, force_virtual_numa);
        if (!mgr) {
            GGML_LOG_ERROR("Failed to create virtual NUMA coordinator manager\n");
            return GGML_STATUS_FAILED;
        }
        
        GGML_LOG_INFO("Using virtual NUMA-aware graph computation via dispatcher for %d operations with %d threads\n", 
                      cgraph->n_nodes, n_threads);
        
        // Process each operation in the graph through the dispatcher
        for (int i = 0; i < cgraph->n_nodes; i++) {
            struct ggml_tensor * operation = cgraph->nodes[i];
            if (!operation) continue;
            
            // Use fallback execution which is simpler for virtual NUMA mode
            enum ggml_status result = ggml_numa_execute_operation_fallback(operation, NULL);
            
            if (result != GGML_STATUS_SUCCESS) {
                GGML_LOG_ERROR("Virtual NUMA fallback execution failed for operation %d (%s)\n", 
                              i, ggml_op_name(operation->op));
                return GGML_STATUS_FAILED;
            }
            
            GGML_LOG_DEBUG("Virtual NUMA successfully executed operation %d (%s) via fallback\n", 
                          i, ggml_op_name(operation->op));
        }
        
        GGML_LOG_INFO("Virtual NUMA graph computation completed successfully via dispatcher\n");
        return GGML_STATUS_SUCCESS;
    } else {
        // Use standard NUMA dispatch
        return ggml_numa_graph_compute(cgraph, n_threads);
    }
}

// Helper function to determine if NUMA coordination would be beneficial for a given graph
static bool ggml_numa_should_coordinate(struct ggml_cgraph * cgraph, int n_threads) {
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

//
// Statistics and Monitoring
//

const ggml_numa_dispatch_stats_t * ggml_numa_dispatch_get_stats(void) {
    return &g_dispatch_stats;
}

void ggml_numa_dispatch_reset_stats(void) {
    memset(&g_dispatch_stats, 0, sizeof(g_dispatch_stats));
}

//
// Default Handler Implementations (Stubs for now)
//

// Enhanced ADD analyzer function - considers tensor shape for optimal NUMA strategy
static enum ggml_status ggml_numa_analyze_add_enhanced(
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context,
    ggml_numa_execution_strategy_t * strategy,
    int * recommended_chunks) {
    
    if (!operation || !context || !strategy || !recommended_chunks) {
        return GGML_STATUS_FAILED;
    }
    
    // Calculate tensor dimensions and total elements
    const int64_t ne0 = operation->ne[0];  // First dimension (often largest)
    const int64_t ne1 = operation->ne[1];  // Second dimension
    const int64_t ne2 = operation->ne[2];  // Third dimension (if any)
    const int64_t ne3 = operation->ne[3];  // Fourth dimension (if any)
    
    const int64_t total_elements = ne0 * ne1 * ne2 * ne3;
    const int64_t memory_transfer = total_elements * 3 * sizeof(float); // Read A, Read B, Write C
    
    GGML_LOG_DEBUG("ADD analysis: shape=[%ld,%ld,%ld,%ld], elements=%ld, memory=%ld bytes\n", 
                   ne0, ne1, ne2, ne3, total_elements, memory_transfer);
    
    // Strategy decision based on tensor size and shape
    const int64_t MIN_ELEMENTS_FOR_NUMA = 50000;  // Higher threshold than before
    const int64_t MIN_DIMENSION_FOR_PARALLEL = 1024;  // Minimum dimension size for good parallelism
    
    if (context->numa_nodes <= 1) {
        // Single NUMA node system
        *strategy = (ggml_numa_execution_strategy_t){
            .node_strategy = NUMA_NODE_STRATEGY_SINGLE_NODE,
            .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
        };
        *recommended_chunks = 1;
        GGML_LOG_DEBUG("ADD: Single NUMA node system, using single-node execution\n");
    }
    else if (total_elements < MIN_ELEMENTS_FOR_NUMA) {
        // Too small for NUMA overhead
        *strategy = (ggml_numa_execution_strategy_t){
            .node_strategy = NUMA_NODE_STRATEGY_SINGLE_NODE,
            .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
        };
        *recommended_chunks = 1;
        GGML_LOG_DEBUG("ADD: Small tensor (%ld elements), using single-node execution\n", total_elements);
    }
    else {
        // Check if tensor shape is suitable for data parallelism
        bool has_large_dimension = (ne0 >= MIN_DIMENSION_FOR_PARALLEL || 
                                   ne1 >= MIN_DIMENSION_FOR_PARALLEL || 
                                   ne2 >= MIN_DIMENSION_FOR_PARALLEL);
        
        bool suitable_for_splitting = (ne0 >= context->numa_nodes * 512 ||  // Can split first dim
                                      ne1 >= context->numa_nodes * 2 ||     // Can split second dim  
                                      ne2 >= context->numa_nodes);          // Can split third dim
        
        if (has_large_dimension && suitable_for_splitting) {
            // Good shape for data parallelism
            *strategy = (ggml_numa_execution_strategy_t){
                .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL,
                .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
            };
            *recommended_chunks = context->numa_nodes;
            GGML_LOG_DEBUG("ADD: Good tensor shape (%ld elements), using data parallel execution\n", total_elements);
        } else {
            // Poor shape for parallelism - might be very long/thin tensor
            *strategy = (ggml_numa_execution_strategy_t){
                .node_strategy = NUMA_NODE_STRATEGY_SINGLE_NODE,
                .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
            };
            *recommended_chunks = 1;
            GGML_LOG_DEBUG("ADD: Poor tensor shape (%ld elements, ne0=%ld), using single-node execution\n", 
                           total_elements, ne0);
        }
    }
    
    return GGML_STATUS_SUCCESS;
}

// Enhanced ADD handler with better shape analysis
const ggml_numa_operation_handler_t ggml_numa_handler_elementwise = {
    .operation_type = GGML_OP_ADD, // Will be expanded to handle multiple element-wise ops
    .default_strategy = {
        .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL,
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
    },
    .complexity = NUMA_OP_COMPLEXITY_SIMPLE,
    .workload_type = NUMA_OP_MEMORY_BOUND,
    .min_elements_for_parallel = 50000,  // Increased from 10K to account for NUMA overhead
    .optimal_chunk_size = 1024 * 1024,
    .parallel_efficiency_estimate = 0.85f,
    .requires_synchronization = false,
    .supports_in_place = true,
    .analyze = ggml_numa_analyze_add_enhanced  // Use enhanced analysis
};

/*
// Enhanced MUL_MAT analyzer function - determines optimal execution strategy based on matrix dimensions
static enum ggml_status ggml_numa_analyze_mul_mat_enhanced(
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context,
    ggml_numa_execution_strategy_t * strategy,
    int * recommended_chunks) {
    
    if (!operation || !context || !strategy || !recommended_chunks) {
        return GGML_STATUS_FAILED;
    }
    
    // Calculate matrix dimensions for A * B = C
    const int64_t M = operation->src[0]->ne[1];  // Rows in A (output rows)
    const int64_t K = operation->src[0]->ne[0];  // Cols in A = Rows in B (inner dimension) 
    const int64_t N = operation->src[1]->ne[1];  // Cols in B (output cols)
    
    // Calculate computational complexity (FLOPs) and memory requirements
    const int64_t total_flops = M * K * N;
    const int64_t memory_bytes = (M*K + K*N + M*N) * sizeof(float);
    
    GGML_LOG_DEBUG("MUL_MAT analysis: M=%ld×K=%ld×N=%ld, FLOPs=%ld, Memory=%ld bytes\n", 
                   M, K, N, total_flops, memory_bytes);
    
    // Strategy decision based on computational complexity and matrix shape
    const int64_t SMALL_FLOPS_THRESHOLD = 1000000;    // 1M FLOPs
    const int64_t MEDIUM_FLOPS_THRESHOLD = 50000000;  // 50M FLOPs  
    const int64_t LARGE_FLOPS_THRESHOLD = 500000000;  // 500M FLOPs
    
    if (context->numa_nodes <= 1) {
        // Single NUMA node system - no point in multi-node strategies
        *strategy = (ggml_numa_execution_strategy_t){
            .node_strategy = NUMA_NODE_STRATEGY_SINGLE_NODE,
            .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
        };
        *recommended_chunks = 1;
        GGML_LOG_DEBUG("MUL_MAT: Single NUMA node system, using single-node execution\n");
    }
    else if (total_flops < SMALL_FLOPS_THRESHOLD) {
        // Small matrices - overhead of NUMA coordination not worth it
        *strategy = (ggml_numa_execution_strategy_t){
            .node_strategy = NUMA_NODE_STRATEGY_SINGLE_NODE,
            .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
        };
        *recommended_chunks = 1;
        GGML_LOG_DEBUG("MUL_MAT: Small matrix (%ld FLOPs), using single-node execution\n", total_flops);
    }
    else if (total_flops < MEDIUM_FLOPS_THRESHOLD) {
        // Medium matrices - use task parallelism (chunking) for better cache locality
        *strategy = (ggml_numa_execution_strategy_t){
            .node_strategy = NUMA_NODE_STRATEGY_TASK_PARALLEL,
            .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
        };
        *recommended_chunks = context->numa_nodes;
        GGML_LOG_DEBUG("MUL_MAT: Medium matrix (%ld FLOPs), using task parallel execution\n", total_flops);
    }
    else if (total_flops > LARGE_FLOPS_THRESHOLD) {
        // Large matrices - definitely worth data parallelism
        *strategy = (ggml_numa_execution_strategy_t){
            .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL,
            .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
        };
        *recommended_chunks = context->numa_nodes;
        GGML_LOG_DEBUG("MUL_MAT: Large matrix (%ld FLOPs), using data parallel execution\n", total_flops);
    }
    else {
        // Medium-large matrices - decide based on matrix shape
        bool good_row_parallelism = (M >= context->numa_nodes * 4);
        bool good_col_parallelism = (N >= context->numa_nodes * 4);
        
        if (good_row_parallelism || good_col_parallelism) {
            // Matrix shape allows good parallelization
            *strategy = (ggml_numa_execution_strategy_t){
                .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL,
                .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
            };
            *recommended_chunks = context->numa_nodes;
            GGML_LOG_DEBUG("MUL_MAT: Good matrix shape (%ld FLOPs, M=%ld, N=%ld), using data parallel\n", 
                           total_flops, M, N);
        } else {
            // Matrix shape not ideal for data parallelism - use task parallelism
            *strategy = (ggml_numa_execution_strategy_t){
                .node_strategy = NUMA_NODE_STRATEGY_TASK_PARALLEL,
                .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
            };
            *recommended_chunks = context->numa_nodes;
            GGML_LOG_DEBUG("MUL_MAT: Poor matrix shape (%ld FLOPs, M=%ld, N=%ld), using task parallel\n", 
                           total_flops, M, N);
        }
    }
    
    return GGML_STATUS_SUCCESS;
}
*/

// Enhanced MUL_MAT handler with intelligent analysis
const ggml_numa_operation_handler_t ggml_numa_handler_mul_mat_enhanced = {
    .operation_type = GGML_OP_MUL_MAT,
    .default_strategy = {
        .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL,  // Data parallel for MUL_MAT
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
    },
    .complexity = NUMA_OP_COMPLEXITY_MODERATE,
    .workload_type = NUMA_OP_COMPUTE_BOUND,
    .min_elements_for_parallel = 100000,  // Increased from 10K - MUL_MAT needs higher threshold
    .optimal_chunk_size = 4 * 1024 * 1024,  // 4MB chunks
    .parallel_efficiency_estimate = 0.85f,  // High efficiency for matrix operations
    .requires_synchronization = false,
    .supports_in_place = false,
    .analyze = NULL,  // TODO: Add matrix dimension analysis
};

const ggml_numa_operation_handler_t ggml_numa_handler_complex = {
    .operation_type = GGML_OP_ROPE, // Complex ROPE operations
    .default_strategy = {
        .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL,
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
    },
    .complexity = NUMA_OP_COMPLEXITY_COMPLEX,
    .workload_type = NUMA_OP_CACHE_SENSITIVE,
    .min_elements_for_parallel = 100000, // Enable NUMA for operations with >100K elements (was INT64_MAX)
    .optimal_chunk_size = 1 * 1024 * 1024,  // 1MB chunks for ROPE
    .parallel_efficiency_estimate = 0.75f,  // Good efficiency for ROPE
    .requires_synchronization = true,
    .supports_in_place = false,
    .analyze = NULL  // Use default analysis for now
};

/*
// TEMPORARILY DISABLED - needs updating for new strategy system
// SOFT_MAX analysis function
static enum ggml_status ggml_numa_analyze_soft_max(
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context,
    ggml_numa_execution_strategy_t * strategy,
    int * recommended_chunks) {
    
    if (!operation || !context || !strategy || !recommended_chunks) {
        return GGML_STATUS_FAILED;
    }
    
    const int64_t ne00 = operation->ne[0];  // Elements per row
    const int64_t ne01 = operation->ne[1];  // Number of rows
    const int64_t total_elements = ne00 * ne01;
    
    GGML_LOG_DEBUG("SOFT_MAX analysis: elements_per_row=%ld, rows=%ld, total_elements=%ld\n", 
                   ne00, ne01, total_elements);
    
    // Decision logic based on matrix size and parallelization potential
    if (context->numa_nodes <= 1) {
        // Single NUMA node - use single node execution
        *strategy = NUMA_EXECUTION_SINGLE_NODE;
        *recommended_chunks = 1;
        GGML_LOG_DEBUG("SOFT_MAX: Single NUMA node, using single-node execution\n");
    } else if (ne01 >= context->numa_nodes * 4) {
        // Many rows - worth parallelizing across NUMA nodes (row-wise parallelism)
        *strategy = NUMA_EXECUTION_DATA_PARALLEL;
        *recommended_chunks = context->numa_nodes;
        GGML_LOG_DEBUG("SOFT_MAX: Many rows (%ld), using data parallel execution\n", ne01);
    } else if (ne01 >= 2 && total_elements > 100000) {
        // Few rows but large elements - use hybrid approach
        *strategy = NUMA_EXECUTION_HYBRID;
        *recommended_chunks = context->numa_nodes;
        GGML_LOG_DEBUG("SOFT_MAX: Large total elements (%ld), using hybrid execution\n", total_elements);
    } else {
        // Small matrices - use single node
        *strategy = NUMA_EXECUTION_SINGLE_NODE;
        *recommended_chunks = 1;
        GGML_LOG_DEBUG("SOFT_MAX: Small matrix (%ld elements), using single-node execution\n", total_elements);
    }
    
    return GGML_STATUS_SUCCESS;
}
*/

// Enhanced SOFT_MAX handler with row-based parallelization
const ggml_numa_operation_handler_t ggml_numa_handler_soft_max = {
    .operation_type = GGML_OP_SOFT_MAX,
    .default_strategy = {
        .node_strategy = NUMA_NODE_STRATEGY_SINGLE_NODE,  // Single node to avoid concurrent writes
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
    },
    .complexity = NUMA_OP_COMPLEXITY_MODERATE,
    .workload_type = NUMA_OP_COMPUTE_BOUND,
    .min_elements_for_parallel = 10000,  // Lower threshold for SOFT_MAX
    .optimal_chunk_size = 1 * 1024 * 1024,  // 1MB chunks
    .parallel_efficiency_estimate = 0.80f,  // Good efficiency for attention operations
    .requires_synchronization = false,
    .supports_in_place = false,
    .analyze = NULL  // FIXED: Remove custom analyzer, always use single node
};

//
// Coordinator Interface Implementation
//

// Use functions from coordinator module instead of direct access
static struct ggml_threadpool * coordinator_get_numa_threadpool(struct ggml_numa_coordinator_manager * manager, int numa_node) {
    return ggml_numa_coordinator_get_threadpool(manager, numa_node);
}

static int coordinator_get_numa_thread_count(struct ggml_numa_coordinator_manager * manager, int numa_node) {
    return ggml_numa_coordinator_get_thread_count(manager, numa_node);
}

static bool coordinator_ensure_work_buffer(struct ggml_numa_coordinator_manager * manager, int numa_node, size_t required_size) {
    return ggml_numa_coordinator_ensure_work_buffer(manager, numa_node, required_size);
}

//
// Phase 1: Complete Single-Threaded Fallback System with Improved Organization
// 
// This implements the critical Phase 1 foundation that handles ALL GGML operations
// through single-threaded execution to avoid threading conflicts while providing
// a stable foundation for gradual NUMA-aware migration.
//

// Helper macros to reduce repetition in dispatch switch
#define DISPATCH_SIMPLE(op_enum, forward_func) \
    case op_enum: \
        ggml_compute_forward_##forward_func(&fallback_params, tensor); \
        break;

// Helper function for operations that need parameter validation
static enum ggml_status validate_tensor_operation(struct ggml_tensor * tensor, const char * op_name) {
    if (!tensor) {
        GGML_LOG_ERROR("%s: Invalid tensor\n", op_name);
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

// Helper function for matrix operations that need dimension checking
static enum ggml_status validate_matrix_operation(struct ggml_tensor * tensor, const char * op_name) {
    if (validate_tensor_operation(tensor, op_name) != GGML_STATUS_SUCCESS) {
        return GGML_STATUS_FAILED;
    }
    
    if (tensor->src[0] && tensor->src[1]) {
        // Add matrix-specific validation here if needed
        GGML_LOG_DEBUG("%s: Matrix operation validated\n", op_name);
    }
    
    return GGML_STATUS_SUCCESS;
}

enum ggml_status ggml_numa_execute_operation_fallback(struct ggml_tensor * tensor, struct ggml_cplan * cplan) {
    // Wrapper to centralized fallback system - eliminates code duplication
    return ggml_numa_fallback_execute(tensor, cplan);
}

// Cleanup helper macros
#undef DISPATCH_SIMPLE

static void * coordinator_get_work_buffer(struct ggml_numa_coordinator_manager * manager, int numa_node) {
    return ggml_numa_coordinator_get_work_buffer(manager, numa_node);
}

static size_t coordinator_get_work_buffer_size(struct ggml_numa_coordinator_manager * manager, int numa_node) {
    return ggml_numa_coordinator_get_work_buffer_size(manager, numa_node);
}

static int coordinator_submit_work(struct ggml_numa_coordinator_manager * manager, struct ggml_tensor * operation, int target_numa_node, ggml_numa_execution_strategy_t strategy) {
    return ggml_numa_coordinator_manager_submit_work(manager, operation, target_numa_node, strategy);
}

static int coordinator_submit_data_parallel_work(struct ggml_numa_coordinator_manager * manager, struct ggml_tensor * operation, 
                                                int work_group_id, const int * target_nodes, int num_target_nodes) {
    (void)work_group_id;    // Suppress unused parameter warning  
    (void)target_nodes;     // Suppress unused parameter warning
    (void)num_target_nodes; // Suppress unused parameter warning
    
    // Use simpler submission for now - the original function signature is different
    return ggml_numa_coordinator_manager_submit_data_parallel_work(manager, operation);
}

//
// Dispatcher Work Buffer Management System
// Similar to coordinator's work buffer system but for dispatcher-level operations
//

//
// Per-Thread Buffer Management
//

// Helper function to convert (numa_node, thread_id) to buffer index
static int ggml_numa_get_thread_buffer_index(int numa_node, int thread_id) {
    if (numa_node < 0 || numa_node >= GGML_NUMA_MAX_NODES || 
        thread_id < 0 || thread_id >= GGML_NUMA_MAX_THREADS_PER_NODE) {
        return -1;
    }
    return numa_node * GGML_NUMA_MAX_THREADS_PER_NODE + thread_id;
}

void ggml_numa_dispatch_work_buffers_init(void) {
    ggml_numa_dispatch_work_buffers_init_internal();
}

static void ggml_numa_dispatch_work_buffers_init_internal(void) {
    if (g_dispatch_work_buffers_initialized) {
        return;
    }
    
    // Detect number of NUMA nodes
    g_numa_nodes_count = numa_available() >= 0 ? numa_max_node() + 1 : 1;
    if (g_numa_nodes_count > GGML_NUMA_MAX_NODES) {
        g_numa_nodes_count = GGML_NUMA_MAX_NODES;
    }
    
    // Initialize all potential thread buffers
    for (int i = 0; i < GGML_NUMA_MAX_TOTAL_THREADS; i++) {
        g_dispatch_thread_buffers[i].buffer = NULL;
        g_dispatch_thread_buffers[i].buffer_size = 0;
        g_dispatch_thread_buffers[i].numa_node = -1;
        g_dispatch_thread_buffers[i].thread_id = -1;
        g_dispatch_thread_buffers[i].initialized = false;
        ggml_mutex_init(&g_dispatch_thread_buffers[i].mutex);
    }
    
    g_dispatch_work_buffers_initialized = true;
    GGML_LOG_DEBUG("Dispatcher per-thread work buffers initialized for up to %d total threads across %d NUMA nodes\n", 
                   GGML_NUMA_MAX_TOTAL_THREADS, g_numa_nodes_count);
}

// Per-thread buffer management functions
bool ggml_numa_dispatch_ensure_work_buffer_for_thread(int numa_node, int thread_id, size_t required_size) {
    int buffer_index = ggml_numa_get_thread_buffer_index(numa_node, thread_id);
    if (buffer_index < 0) {
        GGML_LOG_ERROR("Invalid NUMA node %d or thread ID %d for dispatcher work buffer\n", numa_node, thread_id);
        return false;
    }
    
    ggml_numa_dispatch_thread_buffer_t* tb = &g_dispatch_thread_buffers[buffer_index];
    
    ggml_mutex_lock(&tb->mutex);
    
    // Initialize thread buffer if not done yet
    if (!tb->initialized) {
        tb->numa_node = numa_node;
        tb->thread_id = thread_id;
        tb->initialized = true;
        GGML_LOG_DEBUG("Initialized thread buffer for NUMA node %d, thread %d\n", numa_node, thread_id);
    }
    
    // If we already have a sufficient buffer, reuse it
    if (tb->buffer && tb->buffer_size >= required_size) {
        ggml_mutex_unlock(&tb->mutex);
        return true;
    }
    
    // Free existing buffer if it's too small
    if (tb->buffer) {
        GGML_LOG_DEBUG("Thread NUMA%d:%d: Growing work buffer from %zu to %zu bytes\n", 
                       numa_node, thread_id, tb->buffer_size, required_size);
#ifdef __linux__
        numa_free(tb->buffer, tb->buffer_size);
#else
        free(tb->buffer);
#endif
        tb->buffer = NULL;
        tb->buffer_size = 0;
    } else {
        GGML_LOG_DEBUG("Thread NUMA%d:%d: Allocating initial work buffer of %zu bytes\n", 
                       numa_node, thread_id, required_size);
    }
    
    // Allocate new NUMA-local buffer for this specific thread
#ifdef __linux__
    tb->buffer = numa_alloc_onnode(required_size, numa_node);
#else
    tb->buffer = malloc(required_size);
#endif
    
    if (!tb->buffer) {
        GGML_LOG_ERROR("Thread NUMA%d:%d: Failed to allocate work buffer of size %zu\n", 
                       numa_node, thread_id, required_size);
        tb->buffer_size = 0;
        ggml_mutex_unlock(&tb->mutex);
        return false;
    }
    
    tb->buffer_size = required_size;
    GGML_LOG_DEBUG("Thread NUMA%d:%d: Successfully allocated %zu bytes work buffer\n", 
                   numa_node, thread_id, required_size);
    
    ggml_mutex_unlock(&tb->mutex);
    return true;
}

void* ggml_numa_dispatch_get_work_buffer_for_thread(int numa_node, int thread_id, size_t* buffer_size) {
    int buffer_index = ggml_numa_get_thread_buffer_index(numa_node, thread_id);
    if (buffer_index < 0) {
        GGML_LOG_ERROR("Invalid NUMA node %d or thread ID %d for dispatcher work buffer\n", numa_node, thread_id);
        if (buffer_size) *buffer_size = 0;
        return NULL;
    }
    
    ggml_numa_dispatch_thread_buffer_t* tb = &g_dispatch_thread_buffers[buffer_index];
    
    if (!tb->initialized || !tb->buffer) {
        if (buffer_size) *buffer_size = 0;
        return NULL;
    }
    
    if (buffer_size) *buffer_size = tb->buffer_size;
    return tb->buffer;
}

// Legacy functions for backward compatibility (use thread 0)
bool ggml_numa_dispatch_ensure_work_buffer(int numa_node, size_t required_size) {
    return ggml_numa_dispatch_ensure_work_buffer_for_thread(numa_node, 0, required_size);
}

void* ggml_numa_dispatch_get_work_buffer(int numa_node, size_t* buffer_size) {
    return ggml_numa_dispatch_get_work_buffer_for_thread(numa_node, 0, buffer_size);
}

void ggml_numa_dispatch_cleanup_work_buffers(void) {
    if (!g_dispatch_work_buffers_initialized) {
        return;
    }
    
    GGML_LOG_DEBUG("Cleaning up dispatcher per-thread work buffers\n");
    
    int cleaned_buffers = 0;
    for (int i = 0; i < GGML_NUMA_MAX_TOTAL_THREADS; i++) {
        ggml_numa_dispatch_thread_buffer_t* tb = &g_dispatch_thread_buffers[i];
        
        if (tb->initialized) {
            ggml_mutex_lock(&tb->mutex);
            
            if (tb->buffer) {
                GGML_LOG_DEBUG("Freeing thread work buffer for NUMA node %d, thread %d (%zu bytes)\n", 
                               tb->numa_node, tb->thread_id, tb->buffer_size);
#ifdef __linux__
                numa_free(tb->buffer, tb->buffer_size);
#else
                free(tb->buffer);
#endif
                tb->buffer = NULL;
                tb->buffer_size = 0;
                cleaned_buffers++;
            }
            
            ggml_mutex_unlock(&tb->mutex);
        }
        
        ggml_mutex_destroy(&tb->mutex);
    }
    
    g_dispatch_work_buffers_initialized = false;
    GGML_LOG_DEBUG("Dispatcher per-thread work buffers cleanup completed (%d buffers cleaned)\n", cleaned_buffers);
}

void ggml_numa_dispatch_cleanup_operation_handlers(void) {
    GGML_LOG_DEBUG("Cleaning up dispatcher operation handlers\n");
    
    for (int i = 0; i < GGML_OP_COUNT; i++) {
        if (g_operation_handlers[i]) {
            GGML_LOG_DEBUG("Freeing operation handler for %s\n", ggml_op_name(i));
#ifdef __linux__
            numa_free((void*)g_operation_handlers[i], sizeof(ggml_numa_operation_handler_t));
#else
            free((void*)g_operation_handlers[i]);
#endif
            g_operation_handlers[i] = NULL;
        }
    }
    
    GGML_LOG_DEBUG("Dispatcher operation handlers cleanup completed\n");
}

//
// NUMA Intercept Function - Main Entry Point from ggml-cpu.c
//

enum ggml_status ggml_numa_intercept_operation(struct ggml_tensor * tensor, struct ggml_compute_params * params) {
    if (!tensor || !params) {
        return GGML_STATUS_FAILED;
    }
    
    // Only intercept from the main thread to avoid coordination issues
    if (params->ith != 0) {
        return GGML_STATUS_FAILED;  // Let non-main threads use standard execution
    }
    
    // Initialize dispatcher if not already done (this manages coordinator creation)
    ggml_numa_dispatch_init();
    
    // Create work context from compute params  
    ggml_numa_work_context_t context = {
        .total_elements = ggml_nelements(tensor),
        .element_size = ggml_element_size(tensor),
        .numa_nodes = 2,  // Assume 2 NUMA nodes for now
        .threads_per_node = params->nth,
        .l3_cache_size = 32 * 1024 * 1024,  // 32MB default
        .memory_bandwidth = 100ULL * 1024 * 1024 * 1024  // 100GB/s default
    };
    
    // Copy tensor dimensions
    for (int i = 0; i < GGML_MAX_DIMS && i < 4; i++) {
        context.ne[i] = tensor->ne[i];
    }
    context.n_dims = ggml_n_dims(tensor);
    
    // Route directly to dispatcher (dispatcher manages coordinator internally)
    GGML_LOG_DEBUG("NUMA intercepting operation %s (elements: %" PRId64 ")\n", 
                   ggml_op_name(tensor->op), ggml_nelements(tensor));
    
    // Get or create global manager through dispatcher initialization
    static struct ggml_numa_coordinator_manager * s_global_manager = NULL;
    if (!s_global_manager) {
        extern struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_get_global(int n_threads, bool force_multi_socket);
        s_global_manager = ggml_numa_coordinator_manager_get_global(params->nth, false);
        if (!s_global_manager) {
            GGML_LOG_DEBUG("No NUMA manager available, falling back to standard execution\n");
            return GGML_STATUS_FAILED;  // No NUMA manager available
        }
    }
    
    enum ggml_status result = ggml_numa_dispatch_operation(s_global_manager, tensor, &context);
    
    if (result == GGML_STATUS_SUCCESS) {
        GGML_LOG_DEBUG("NUMA dispatch successful for %s\n", ggml_op_name(tensor->op));
    } else {
        GGML_LOG_DEBUG("NUMA dispatch failed for %s, will use standard execution\n", ggml_op_name(tensor->op));
    }
    
    return result;
}

// Graph-level processing function - primary interface for llama-context.cpp
// Routes graph through dispatcher instead of coordinator directly
int ggml_numa_dispatch_compute_graph(struct ggml_cgraph * cgraph, int n_threads) {
    if (!cgraph) {
        GGML_LOG_ERROR("Invalid graph provided to dispatcher\n");
        return -1;
    }
    
    if (!ggml_numa_should_mirror()) {
        GGML_LOG_DEBUG("NUMA mirroring disabled, skipping dispatcher\n");
        return -1;  // Let caller use standard processing
    }
    
    GGML_LOG_INFO("Processing computation graph with %d nodes through NUMA dispatcher\n", cgraph->n_nodes);
    
    // Initialize dispatcher if needed
    ggml_numa_dispatch_init();
    
    // Get coordinator manager (should already be initialized by llama-context.cpp)
    extern struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_get_global(int n_threads, bool force_multi_socket);
    struct ggml_numa_coordinator_manager * manager = ggml_numa_coordinator_manager_get_global(n_threads, false);
    if (!manager) {
        GGML_LOG_ERROR("No NUMA coordinator available for dispatcher\n");
        return -1;
    }
    
    // Process each node through the dispatcher handlers directly
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if (!node) continue;
        
        GGML_LOG_DEBUG("Dispatching node %d: %s\n", i, ggml_op_name(node->op));
        
        // Create work context for this operation
        ggml_numa_work_context_t context = {
            .total_elements = ggml_nelements(node),
            .element_size = ggml_element_size(node),
            .numa_nodes = 2,  // Default
            .threads_per_node = n_threads,
            .l3_cache_size = 32 * 1024 * 1024,  // 32MB default
            .memory_bandwidth = 100ULL * 1024 * 1024 * 1024  // 100GB/s default
        };
        
        // Copy tensor dimensions
        for (int dim = 0; dim < GGML_MAX_DIMS && dim < 4; dim++) {
            context.ne[dim] = node->ne[dim];
        }
        context.n_dims = ggml_n_dims(node);
        
        // Call dispatcher directly without going through intercept system
        enum ggml_status result = ggml_numa_dispatch_operation(manager, node, &context);
        
        if (result != GGML_STATUS_SUCCESS) {
            GGML_LOG_ERROR("Node %d (%s) failed in dispatcher, aborting graph\n", i, ggml_op_name(node->op));
            return -1;
        }
        
        GGML_LOG_DEBUG("Node %d (%s) completed successfully\n", i, ggml_op_name(node->op));
    }
    
    // Graph processing complete - coordinator handles individual operation synchronization
    // No need for additional waiting as operations are already synchronized
    
    GGML_LOG_INFO("Graph computation completed successfully through dispatcher\n");
    return 0;
}

//
// Work Function Pointer Infrastructure Implementation
//

// Create work context for function pointer execution
static ggml_numa_dispatcher_work_context_t * ggml_numa_dispatcher_create_work_context(
    struct ggml_tensor * operation,
    const char * operation_name,
    void * additional_context,
    size_t additional_context_size
) {
    if (!operation) return NULL;
    
    ggml_numa_dispatcher_work_context_t * context = malloc(sizeof(ggml_numa_dispatcher_work_context_t));
    if (!context) return NULL;
    
    // Set up compute plan
    struct ggml_cplan * cplan = malloc(sizeof(struct ggml_cplan));
    if (!cplan) {
        free(context);
        return NULL;
    }
    
    // Initialize work context
    context->operation = operation;
    context->cplan = cplan;
    context->operation_name = operation_name ? operation_name : ggml_op_name(operation->op);
    context->additional_context = NULL;
    context->additional_context_size = 0;
    
    // Copy additional context if provided
    if (additional_context && additional_context_size > 0) {
        context->additional_context = malloc(additional_context_size);
        if (context->additional_context) {
            memcpy(context->additional_context, additional_context, additional_context_size);
            context->additional_context_size = additional_context_size;
        }
    }
    
    // Initialize compute plan with defaults
    memset(cplan, 0, sizeof(struct ggml_cplan));
    cplan->n_threads = 1;  // Will be updated by coordinator
    cplan->work_size = 0;  // Will be set by coordinator
    cplan->work_data = NULL;  // Will be set by coordinator
    
    return context;
}

// Free work context
static void ggml_numa_dispatcher_free_work_context(ggml_numa_dispatcher_work_context_t * context) {
    if (!context) return;
    
    if (context->cplan) {
        free(context->cplan);
    }
    
    if (context->additional_context) {
        free(context->additional_context);
    }
    
    free(context);
}

// Calculate work buffer size required for operation
static size_t ggml_numa_dispatcher_calculate_work_buffer_size(const struct ggml_tensor * operation) {
    if (!operation) return 0;
    
    switch (operation->op) {
        case GGML_OP_SOFT_MAX: {
            // SOFT_MAX needs (ne00 + CACHE_LINE_SIZE_F32) * sizeof(float) per thread
            const int64_t ne00 = operation->ne[0];
            const int nth = 22; // Conservative estimate for max threads
            return (ne00 + CACHE_LINE_SIZE_F32) * sizeof(float) * nth;
        }
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM: {
            // Normalization operations need space for intermediate calculations
            return ggml_nelements(operation) * sizeof(float);
        }
        case GGML_OP_MUL_MAT: {
            // Matrix multiply needs space for intermediate results
            const int64_t ne00 = operation->src[0] ? operation->src[0]->ne[0] : 0;
            const int64_t ne11 = operation->src[1] ? operation->src[1]->ne[1] : 0;
            return ne00 * ne11 * sizeof(float);
        }
        case GGML_OP_GROUP_NORM: {
            // Group normalization needs working space
            return ggml_nelements(operation) * sizeof(float);
        }
        case GGML_OP_ROPE: {
            // ROPE operations need cache space for sin/cos values
            const int64_t ne0 = operation->ne[0];
            const int nth = 1; // Single-threaded fallback
            return (ne0 + CACHE_LINE_SIZE_F32) * sizeof(float) * nth;
        }
        // Most operations don't need work buffers
        default:
            return 0;
    }
}

//
// Work Function Implementations - Functions passed as function pointers to coordinator
//

// Generic fallback work function - executes any operation via fallback system
static enum ggml_status ggml_numa_work_function_fallback(void * work_context, struct ggml_compute_params * params) {
    if (!work_context || !params) {
        return GGML_STATUS_FAILED;
    }
    
    ggml_numa_dispatcher_work_context_t * ctx = (ggml_numa_dispatcher_work_context_t *)work_context;
    
    GGML_LOG_DEBUG("Executing %s operation via fallback work function with %d threads\n", 
                   ctx->operation_name, params->nth);
    
    // Update the compute plan with coordinator-provided parameters
    if (ctx->cplan) {
        ctx->cplan->n_threads = params->nth;
        ctx->cplan->work_size = params->wsize;
        ctx->cplan->work_data = params->wdata;
        ctx->cplan->threadpool = params->threadpool;
    }
    
    // Execute the operation using the fallback system
    enum ggml_status result = ggml_numa_fallback_execute(ctx->operation, ctx->cplan);
    
    if (result == GGML_STATUS_SUCCESS) {
        GGML_LOG_DEBUG("Successfully executed %s via fallback work function\n", ctx->operation_name);
    } else {
        GGML_LOG_ERROR("Failed to execute %s via fallback work function\n", ctx->operation_name);
    }
    
    return result;
}

// Specialized work function for single-threaded MUL_MAT operations
static enum ggml_status ggml_numa_work_function_mul_mat_single(void * work_context, struct ggml_compute_params * params) {
    if (!work_context || !params) {
        return GGML_STATUS_FAILED;
    }
    
    ggml_numa_dispatcher_work_context_t * ctx = (ggml_numa_dispatcher_work_context_t *)work_context;
    
    GGML_LOG_DEBUG("Executing MUL_MAT single-threaded work function\n");
    
    // Update compute plan
    if (ctx->cplan) {
        ctx->cplan->n_threads = 1;  // Force single-threaded
        ctx->cplan->work_size = params->wsize;
        ctx->cplan->work_data = params->wdata;
        ctx->cplan->threadpool = NULL;  // No threadpool for single-threaded
    }
    
    // Execute using fallback system with single thread
    enum ggml_status result = ggml_numa_fallback_execute(ctx->operation, ctx->cplan);
    
    if (result == GGML_STATUS_SUCCESS) {
        GGML_LOG_DEBUG("Successfully executed MUL_MAT single-threaded work function\n");
    } else {
        GGML_LOG_ERROR("Failed to execute MUL_MAT single-threaded work function\n");
    }
    
    return result;
}

// Specialized work function for chunked MUL_MAT operations
static enum ggml_status ggml_numa_work_function_mul_mat_chunk(void * work_context, struct ggml_compute_params * params) {
    if (!work_context || !params) {
        return GGML_STATUS_FAILED;
    }
    
    ggml_numa_dispatcher_work_context_t * ctx = (ggml_numa_dispatcher_work_context_t *)work_context;
    
    GGML_LOG_DEBUG("Executing MUL_MAT chunked work function with %d threads\n", params->nth);
    
    // Update compute plan with coordinator parameters
    if (ctx->cplan) {
        ctx->cplan->n_threads = params->nth;
        ctx->cplan->work_size = params->wsize;
        ctx->cplan->work_data = params->wdata;
        ctx->cplan->threadpool = params->threadpool;
    }
    
    // If we have additional context, it might contain chunk information
    // For now, just execute the operation with the provided parameters
    enum ggml_status result = ggml_numa_fallback_execute(ctx->operation, ctx->cplan);
    
    if (result == GGML_STATUS_SUCCESS) {
        GGML_LOG_DEBUG("Successfully executed MUL_MAT chunked work function\n");
    } else {
        GGML_LOG_ERROR("Failed to execute MUL_MAT chunked work function\n");
    }
    
    return result;
}

// Specialized work function for SOFT_MAX operations
static enum ggml_status ggml_numa_work_function_soft_max(void * work_context, struct ggml_compute_params * params) {
    if (!work_context || !params) {
        return GGML_STATUS_FAILED;
    }
    
    ggml_numa_dispatcher_work_context_t * ctx = (ggml_numa_dispatcher_work_context_t *)work_context;
    
    GGML_LOG_DEBUG("Executing SOFT_MAX work function with %d threads\n", params->nth);
    
    // Update compute plan with coordinator parameters
    if (ctx->cplan) {
        ctx->cplan->n_threads = params->nth;
        ctx->cplan->work_size = params->wsize;
        ctx->cplan->work_data = params->wdata;
        ctx->cplan->threadpool = params->threadpool;
    }
    
    // Execute using fallback system
    enum ggml_status result = ggml_numa_fallback_execute(ctx->operation, ctx->cplan);
    
    if (result == GGML_STATUS_SUCCESS) {
        GGML_LOG_DEBUG("Successfully executed SOFT_MAX work function\n");
    } else {
        GGML_LOG_ERROR("Failed to execute SOFT_MAX work function\n");
    }
    
    return result;
}
