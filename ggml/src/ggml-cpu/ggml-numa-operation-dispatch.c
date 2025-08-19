/**
 * NUMA Operation Dispatch Infrastructure Implementation
 * 
 * This implements the intelligent dispatcher system that routes operations
 * to appropriate execution strategies while preserving existing thread synchronization.
 */

#include "ggml-numa-operation-dispatch.h"
#include "ggml-numa-coordinator.h"
#include "ggml-numa-fallback.h"
#include "ggml-numa-work-shared.h"
#include "ggml-impl.h"
#include "ggml-cpu-impl.h"  // For ggml_compute_params structure
#include "ggml.h"           // For ggml_cplan and graph functions

// Include all operation headers for fallback system
#include "ops.h"            // Main operations
#include "unary-ops.h"      // Unary operations (sin, cos, log, etc.)
#include "binary-ops.h"     // Binary operations (add, sub, mul, div)
#include "vec.h"            // Vector operations and SIMD optimizations

// Include NUMA work function headers for operation handlers
#include "numa-work/ggml-numa-mulmat.h"  // For ggml_numa_handler_mul_mat_enhanced

// Include NUMA work function headers
#include "numa-work/ggml-numa-add.h"
#include "numa-work/ggml-numa-mulmat.h"
#include "numa-work/ggml-numa-softmax.h"
#include "numa-work/ggml-numa-rope.h"
#include "numa-work/ggml-numa-glu.h"
#include "numa-work/ggml-numa-rms-norm.h"
#include "numa-work/ggml-numa-flash-attn-ext.h"

#include <unistd.h>         // For usleep

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
#include <unistd.h>         // For usleep

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

// Forward declaration for coordinator manager wait function
extern enum ggml_status ggml_numa_coordinator_manager_wait_for_completion(struct ggml_numa_coordinator_manager * mgr);
extern void ggml_numa_coordinator_manager_reset_status(struct ggml_numa_coordinator_manager * mgr);

// Work function prototypes - these are the actual functions passed to coordinator
// Note: These are now defined in their respective header files
static enum ggml_status ggml_numa_work_function_fallback(void * work_context, struct ggml_compute_params * params);

//
// Forward Declarations for Internal Functions
//

// Direct MUL_MAT execution function that bypasses coordinator
static enum ggml_status ggml_numa_mul_mat_execute_direct(
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context);

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

// MUL_MAT execution functions are now in numa-work/ggml-numa-mulmat.c

// SOFT_MAX chunked execution using row-wise parallelization
static enum ggml_status ggml_numa_execute_soft_max_chunked(
    struct ggml_numa_coordinator_manager * manager,
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context
);

// ROPE chunked execution using data parallelization across NUMA nodes
static enum ggml_status ggml_numa_execute_rope_data_parallel(
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

// MUL_MAT chunked execution functions are now in numa-work/ggml-numa-mulmat.c

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
// ggml_numa_handler_mul_mat_enhanced is now in numa-work/ggml-numa-mulmat.h
extern const ggml_numa_operation_handler_t ggml_numa_handler_soft_max;
extern const ggml_numa_operation_handler_t ggml_numa_handler_glu;
extern const ggml_numa_operation_handler_t ggml_numa_handler_rms_norm;
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
// Direct MUL_MAT Execution Implementation
//

static enum ggml_status ggml_numa_mul_mat_execute_direct(
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context) {
    
    GGML_LOG_DEBUG("Direct MUL_MAT execution: analyzing strategy\n");
    
    GGML_ASSERT(operation);
    GGML_ASSERT(context);
    GGML_ASSERT(operation->op == GGML_OP_MUL_MAT);
    
    // Analyze operation to determine optimal strategy
    ggml_numa_execution_strategy_t strategy;
    ggml_numa_work_function_t work_function;
    size_t work_buffer_size;
    
    enum ggml_status analysis_result = ggml_numa_mul_mat_analyze_strategy(
        operation, context, &strategy, &work_function, &work_buffer_size);
    
    if (analysis_result != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("Failed to analyze MUL_MAT strategy\n");
        return GGML_STATUS_FAILED;
    }
    
    GGML_LOG_DEBUG("Direct MUL_MAT: strategy=%s, buffer_size=%zu\n",
                   (strategy.node_strategy == NUMA_NODE_STRATEGY_SINGLE) ? "single" : "parallel",
                   work_buffer_size);
    
    // Create work context for the operation
    ggml_numa_dispatcher_work_context_t * work_context = ggml_numa_dispatcher_create_work_context(
        (struct ggml_tensor *)operation,
        "MUL_MAT_DIRECT",
        NULL,  // No additional context
        0      // No additional context size
    );
    
    if (!work_context) {
        GGML_LOG_ERROR("Failed to create work context for direct MUL_MAT\n");
        return GGML_STATUS_FAILED;
    }
    
    // Allocate work buffer if needed
    void * work_buffer = NULL;
    if (work_buffer_size > 0) {
        work_buffer = malloc(work_buffer_size);
        if (!work_buffer) {
            GGML_LOG_ERROR("Failed to allocate work buffer (%zu bytes) for direct MUL_MAT\n", work_buffer_size);
            ggml_numa_dispatcher_free_work_context(work_context);
            return GGML_STATUS_FAILED;
        }
        GGML_LOG_DEBUG("Allocated work buffer: %p (%zu bytes)\n", work_buffer, work_buffer_size);
    }
    
    // Set up compute params for the work function
    struct ggml_compute_params params = {
        .ith = 0,           // Thread 0 (single-threaded execution)
        .nth = 1,           // Total threads = 1
        .wsize = work_buffer_size,
        .wdata = work_buffer
    };
    
    // Execute the work function directly
    GGML_LOG_DEBUG("Executing MUL_MAT work function directly\n");
    enum ggml_status result = work_function(work_context, &params);
    
    // Cleanup
    if (work_buffer) {
        free(work_buffer);
        GGML_LOG_DEBUG("Freed work buffer: %p\n", work_buffer);
    }
    ggml_numa_dispatcher_free_work_context(work_context);
    
    if (result == GGML_STATUS_SUCCESS) {
        GGML_LOG_DEBUG("Direct MUL_MAT execution completed successfully\n");
    } else {
        GGML_LOG_ERROR("Direct MUL_MAT execution failed\n");
    }
    
    return result;
}

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
    ggml_numa_dispatch_register_handler(&ggml_numa_handler_glu);              // NUMA-aware GLU handler
    ggml_numa_dispatch_register_handler(&ggml_numa_handler_rms_norm);         // NUMA-aware RMS_NORM handler
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
    
    // CRITICAL DEBUGGING: Capture tensor types at entry point to catch corruption
    if (operation->op == GGML_OP_MUL_MAT) {
        const struct ggml_tensor * src0 = operation->src[0];
        const struct ggml_tensor * src1 = operation->src[1];
        
        fprintf(stderr, "🔍 DISPATCH ENTRY: MUL_MAT operation entry point\n");
        fprintf(stderr, "    📊 operation=%p, src[0]=%p, src[1]=%p\n", 
               (const void*)operation, (const void*)src0, (const void*)src1);
        
        if (src0) {
            fprintf(stderr, "    📊 src[0]: type=%d (%s), dims=(%ld,%ld,%ld,%ld)\n", 
                   src0->type, ggml_type_name(src0->type), src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
        }
        if (src1) {
            fprintf(stderr, "    📊 src[1]: type=%d (%s), dims=(%ld,%ld,%ld,%ld)\n", 
                   src1->type, ggml_type_name(src1->type), src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
        }
        fflush(stderr);
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
    
    // TEMPORARY FIX: Bypass NUMA optimization for F16 operations due to precision issues
    // But NOT for MUL_MAT which has its own handling
    bool has_f16_tensor = false;
    if (operation->src[0] && operation->src[0]->type == GGML_TYPE_F16) {
        has_f16_tensor = true;
    }
    if (operation->src[1] && operation->src[1]->type == GGML_TYPE_F16) {
        has_f16_tensor = true;
    }
    
    // Exclude MUL_MAT from F16 bypass since it has special handling
    bool should_bypass_for_f16 = has_f16_tensor && (operation->op != GGML_OP_MUL_MAT);
    
    if (should_bypass_for_f16) {
        GGML_LOG_DEBUG("F16 operation detected (%s), bypassing NUMA optimization and using fallback\n", 
                       ggml_op_name(operation->op));
        
        // Check if this operation needs work buffers (excluding MUL_MAT)
        bool needs_work_buffer = (operation->op == GGML_OP_SOFT_MAX || 
                                  operation->op == GGML_OP_ROPE ||
                                  operation->op == GGML_OP_NORM ||
                                  operation->op == GGML_OP_RMS_NORM ||
                                  operation->op == GGML_OP_GROUP_NORM ||
                                  operation->op == GGML_OP_FLASH_ATTN_EXT);
        
        if (needs_work_buffer) {
            GGML_LOG_DEBUG("Operation %s needs work buffers, setting up cplan for fallback\n", ggml_op_name(operation->op));
            
            // Calculate work buffer size for this operation
            size_t buffer_size = ggml_numa_dispatcher_calculate_work_buffer_size(operation);
            
            // Allocate actual work buffer if needed
            void * work_data = NULL;
            if (buffer_size > 0) {
                work_data = malloc(buffer_size);
                if (!work_data) {
                    GGML_LOG_ERROR("Failed to allocate %zu bytes for F16 work buffer\n", buffer_size);
                    return GGML_STATUS_FAILED;
                }
                GGML_LOG_DEBUG("F16 %s allocated work buffer: %p size=%zu\n", 
                              ggml_op_name(operation->op), work_data, buffer_size);
            }
            
            // Create a simple cplan with work buffer
            struct ggml_cplan cplan = {
                .n_threads = 1,
                .work_size = buffer_size,
                .work_data = work_data,
                .threadpool = NULL
            };
            
            GGML_LOG_DEBUG("F16 %s fallback with work_size=%zu work_data=%p\n", 
                          ggml_op_name(operation->op), buffer_size, work_data);
            result = ggml_numa_fallback_execute((struct ggml_tensor *)operation, &cplan);
            
            // Free the work buffer after execution
            if (work_data) {
                free(work_data);
                GGML_LOG_DEBUG("F16 %s freed work buffer %p\n", ggml_op_name(operation->op), work_data);
            }
        } else {
            GGML_LOG_DEBUG("Operation %s doesn't need work buffers, using direct fallback\n", ggml_op_name(operation->op));
            result = ggml_numa_fallback_execute((struct ggml_tensor *)operation, NULL);
        }
        
        // Update timing and return
        int64_t end_time = ggml_time_us();
        g_dispatch_stats.total_execution_time_us += (end_time - start_time);
        
        return result;
    }
    
    if (handler) {
        // Special case: MUL_MAT operations use direct execution with our mathematical kernels
        if (operation->op == GGML_OP_MUL_MAT) {
            GGML_LOG_DEBUG("Using direct MUL_MAT execution with NUMA mathematical kernels\n");
            result = ggml_numa_mul_mat_execute_direct(operation, context);
        } else {
            // Use registered handler to analyze and execute for other operations
            ggml_numa_execution_strategy_t strategy;
            int recommended_chunks;
            
            // Analyze operation to determine optimal strategy
            if (handler->analyze) {
                enum ggml_status analysis_result = handler->analyze(operation, context, &strategy, &recommended_chunks);
                if (analysis_result != GGML_STATUS_SUCCESS) {
                    GGML_LOG_WARN("Operation analysis failed for %s, falling back to single node\n", 
                                 ggml_op_name(operation->op));
                    strategy = (ggml_numa_execution_strategy_t){
                        .node_strategy = NUMA_NODE_STRATEGY_SINGLE,
                        .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
                    };
                }
            } else {
                // Use default strategy from handler
                strategy = handler->default_strategy;
                recommended_chunks = context->numa_nodes;
                
                // SIMPLIFIED: Only override data parallel if coordinator reports single NUMA node
                // The coordinator already handles force_multi_socket and real NUMA detection,
                // so we trust its NUMA node count directly
                
                if (context->numa_nodes <= 1 && strategy.node_strategy == NUMA_NODE_STRATEGY_DATA_PARALLEL) {
                    GGML_LOG_INFO("Overriding data parallel strategy to single node for %s (coordinator reports %d NUMA nodes)\n",
                                 ggml_op_name(operation->op), context->numa_nodes);
                    strategy.node_strategy = NUMA_NODE_STRATEGY_SINGLE;
                } else {
                    GGML_LOG_INFO("Using data parallel strategy for %s (coordinator reports %d NUMA nodes)\n",
                                 ggml_op_name(operation->op), context->numa_nodes);
                }
            }
            
            // Execute based on determined strategy
            switch (strategy.node_strategy) {
                case NUMA_NODE_STRATEGY_SINGLE:
                    result = ggml_numa_execute_single_node(manager, operation, context);
                    break;
                    
                case NUMA_NODE_STRATEGY_DATA_PARALLEL:
                    result = ggml_numa_execute_data_parallel(manager, operation, context);
                    g_dispatch_stats.parallelized_operations++;
                    break;
                
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
                                  operation->op == GGML_OP_GROUP_NORM ||
                                  operation->op == GGML_OP_GLU ||
                                  operation->op == GGML_OP_FLASH_ATTN_EXT);
        
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
    
    // Reset status for this new work submission
    ggml_numa_coordinator_manager_reset_status(manager);
    
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
    
    // Set up execution strategy for single node - FORCED TO SINGLE THREAD FOR DEBUGGING
    ggml_numa_execution_strategy_t single_node_strategy = {
        .node_strategy = NUMA_NODE_STRATEGY_SINGLE,
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD
    };
    
    // Choose appropriate work function based on operation type
    ggml_numa_work_function_t work_function;
    if (operation->op == GGML_OP_MUL_MAT) {
        work_function = ggml_numa_work_function_mul_mat_single;
        GGML_LOG_DEBUG("Using specialized MUL_MAT single work function\n");
    } else if (operation->op == GGML_OP_ADD) {
        work_function = ggml_numa_work_function_add_single;
        GGML_LOG_DEBUG("Using specialized ADD single work function for single node\n");
    } else if (operation->op == GGML_OP_GLU) {
        work_function = ggml_numa_work_function_glu_chunk;
        GGML_LOG_DEBUG("Using specialized GLU chunk work function for single node\n");
    } else if (operation->op == GGML_OP_RMS_NORM) {
        work_function = ggml_numa_work_function_rms_norm_chunk;
        GGML_LOG_DEBUG("Using specialized RMS_NORM chunk work function for single node\n");
    } else if (operation->op == GGML_OP_FLASH_ATTN_EXT) {
        work_function = ggml_numa_work_function_flash_attn_ext_chunk;
        GGML_LOG_DEBUG("Using specialized FLASH_ATTN_EXT chunk work function for single node\n");
    } else {
        work_function = ggml_numa_work_function_fallback;
        GGML_LOG_DEBUG("Using generic fallback work function for %s\n", ggml_op_name(operation->op));
    }
    
    // Submit work using new function pointer approach
    int work_id = ggml_numa_coordinator_manager_submit_work_function(
        manager,
        work_function,                     // Operation-specific function
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
    
    // Wait for work completion with enhanced debugging and synchronization
    GGML_LOG_ERROR("🔧 DISPATCH: About to wait for work completion (work_id=%d)\n", work_id);
    
    // Log the operation being dispatched for debugging
    GGML_LOG_ERROR("🔧 DISPATCH: About to wait for work completion (work_id=%d) for operation %s\n", 
                   work_id, ggml_op_name(operation->op));
    
    // Multiple wait attempts with debugging
    enum ggml_status wait_status = GGML_STATUS_FAILED;
    for (int attempt = 0; attempt < 10; attempt++) {
        wait_status = ggml_numa_coordinator_manager_wait_for_completion(manager);
        if (wait_status == GGML_STATUS_SUCCESS) {
            GGML_LOG_ERROR("🔧 DISPATCH: Wait completed successfully on attempt %d\n", attempt + 1);
            break;
        } else if (wait_status == GGML_STATUS_FAILED) {
            // CRITICAL: Work function failed (e.g., NUMA_ASSERT triggered)
            // This indicates data corruption or serious computational error
            // We must halt execution immediately
            GGML_LOG_ERROR("💥 CRITICAL ERROR: NUMA operation %s failed with GGML_STATUS_FAILED\n", ggml_op_name(operation->op));
            GGML_LOG_ERROR("💥 This indicates data corruption or computational failure - halting execution\n");
            GGML_LOG_ERROR("💥 Check NUMA_ASSERT errors above for details\n");
            
            // Note: work_context will be freed by the coordinator after execution
            return GGML_STATUS_FAILED;
        }
        GGML_LOG_ERROR("🔧 DISPATCH: Wait attempt %d returned status %d, retrying...\n", attempt + 1, (int)wait_status);
        usleep(1000); // 1ms delay between attempts
    }
    
    if (wait_status != GGML_STATUS_SUCCESS) {
        GGML_LOG_WARN("Failed to wait for work completion after 10 attempts (final status: %d)\n", (int)wait_status);
        // Return the actual failure status instead of continuing
        return wait_status;
    }
    
    // Additional memory barrier to ensure all coordinator work is visible
    __sync_synchronize();
    
    GGML_LOG_ERROR("🔧 DISPATCH: Work completion wait finished (final result: %d)\n", (int)wait_status);
    
    // Note: work_context will be freed by the coordinator after execution
    
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_numa_execute_data_parallel(
    struct ggml_numa_coordinator_manager * manager,
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context) {
    
    GGML_LOG_DEBUG("Executing %s with data parallelism across %d NUMA nodes using function pointers\n", 
                   ggml_op_name(operation->op), context->numa_nodes);
    
    // Special handling for operations that need custom data parallel execution
    switch (operation->op) {
        case GGML_OP_ROPE: {
            // ROPE uses specialized data parallel execution with NUMA-aware buffer management
            GGML_LOG_DEBUG("Routing ROPE to specialized data parallel execution\n");
            return ggml_numa_execute_rope_data_parallel(manager, operation, context);
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
    
    // Set up execution strategy for data parallel execution - FORCED TO SINGLE FOR DEBUGGING
    ggml_numa_execution_strategy_t data_parallel_strategy = {
        .node_strategy = NUMA_NODE_STRATEGY_SINGLE,
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD
    };
    
    // Choose appropriate work function based on operation type
    ggml_numa_work_function_t work_function;
    if (operation->op == GGML_OP_MUL_MAT) {
        work_function = ggml_numa_work_function_mul_mat_chunk;
        GGML_LOG_DEBUG("🔧 DEBUG: Selected MUL_MAT chunk work function %p\n", (void*)work_function);
        GGML_LOG_DEBUG("Using specialized MUL_MAT chunk work function for data parallelism\n");
    } else if (operation->op == GGML_OP_SOFT_MAX) {
        work_function = ggml_numa_work_function_soft_max_chunk;
        GGML_LOG_DEBUG("🔧 DEBUG: Selected SOFT_MAX chunk work function %p\n", (void*)work_function);
        GGML_LOG_DEBUG("Using specialized SOFT_MAX chunk work function for data parallelism\n");
    } else if (operation->op == GGML_OP_ADD) {
        work_function = ggml_numa_work_function_add_chunk;
        GGML_LOG_DEBUG("🔧 DEBUG: Selected ADD chunk work function %p\n", (void*)work_function);
        GGML_LOG_DEBUG("Using specialized ADD chunk work function for data parallelism\n");
    } else if (operation->op == GGML_OP_GLU) {
        work_function = ggml_numa_work_function_glu_chunk;
        GGML_LOG_DEBUG("🔧 DEBUG: Selected GLU chunk work function %p\n", (void*)work_function);
        GGML_LOG_DEBUG("Using specialized GLU chunk work function for data parallelism\n");
    } else if (operation->op == GGML_OP_RMS_NORM) {
        work_function = ggml_numa_work_function_rms_norm_chunk;
        GGML_LOG_DEBUG("🔧 DEBUG: Selected RMS_NORM chunk work function %p\n", (void*)work_function);
        GGML_LOG_DEBUG("Using specialized RMS_NORM chunk work function for data parallelism\n");
    } else if (operation->op == GGML_OP_FLASH_ATTN_EXT) {
        work_function = ggml_numa_work_function_flash_attn_ext_chunk;
        GGML_LOG_DEBUG("🔧 DEBUG: Selected FLASH_ATTN_EXT chunk work function %p\n", (void*)work_function);
        GGML_LOG_DEBUG("Using specialized FLASH_ATTN_EXT chunk work function for data parallelism\n");
    } else {
        work_function = ggml_numa_work_function_fallback;
        GGML_LOG_DEBUG("🔧 DEBUG: Selected fallback work function %p for %s\n", (void*)work_function, ggml_op_name(operation->op));
        GGML_LOG_DEBUG("Using generic fallback work function for %s data parallelism\n", ggml_op_name(operation->op));
    }
    
    // Submit work using new function pointer approach for data parallelism
    int work_id = ggml_numa_coordinator_manager_submit_work_function(
        manager,
        work_function,                     // Operation-specific function
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
    
    // Wait for work completion instead of using unreliable sleep
    int wait_result = ggml_numa_coordinator_manager_wait_for_completion(manager);
    if (wait_result != 0) {
        GGML_LOG_WARN("Failed to wait for work completion (result: %d), trying additional sync\n", wait_result);
        
        // Add explicit synchronization barrier to ensure all writes are complete
        __sync_synchronize();
        
        // Brief sleep to allow coordinator threads to complete any pending writes
        usleep(1000);  // 1ms sleep for thread synchronization
        
        // Try to wait again
        wait_result = ggml_numa_coordinator_manager_wait_for_completion(manager);
        if (wait_result != 0) {
            GGML_LOG_ERROR("Second wait also failed (result: %d) - may have race condition\n", wait_result);
            ggml_numa_dispatcher_free_work_context(work_context);
            return GGML_STATUS_FAILED;  // Return failure instead of continuing
        }
    }
    
    // Add final memory barrier to ensure all coordinator writes are visible
    __sync_synchronize();
    
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
    
    // Set up execution strategy for complex operations - FORCED TO SINGLE THREAD FOR DEBUGGING
    ggml_numa_execution_strategy_t strategy = {
        .node_strategy = NUMA_NODE_STRATEGY_SINGLE,  // Complex operations benefit from task parallelism
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD
    };
    
    // Choose appropriate work function based on operation type
    ggml_numa_work_function_t work_function;
    if (operation->op == GGML_OP_MUL_MAT) {
        work_function = ggml_numa_work_function_mul_mat_single;
        GGML_LOG_DEBUG("Using specialized MUL_MAT single work function for complex execution\n");
    } else if (operation->op == GGML_OP_ADD) {
        work_function = ggml_numa_work_function_add_chunk;
        GGML_LOG_DEBUG("Using specialized ADD chunk work function for complex execution\n");
    } else if (operation->op == GGML_OP_RMS_NORM) {
        work_function = ggml_numa_work_function_rms_norm_chunk;
        GGML_LOG_DEBUG("Using specialized RMS_NORM chunk work function for complex execution\n");
    } else {
        work_function = ggml_numa_work_function_fallback;
        GGML_LOG_DEBUG("Using generic fallback work function for complex %s\n", ggml_op_name(operation->op));
    }
    
    // Submit work using function pointer approach
    int work_id = ggml_numa_coordinator_manager_submit_work_function(
        manager,
        work_function,                     // Operation-specific function
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
    
    // Simple synchronization: Give work time to complete
    // TODO: Replace with proper work completion tracking
    usleep(1000); // 1ms should be enough for small operations
    
    // Note: work_context will be freed by the coordinator after execution
    
    return GGML_STATUS_SUCCESS;
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
    
    // Calculate required work buffer size: (ne00 + CACHE_LINE_SIZE_F32) * sizeof(float)
    // Since we use single-threaded params (ith=0, nth=1), only need space for one thread per node
    const size_t work_buffer_size = (ne00 + CACHE_LINE_SIZE_F32) * sizeof(float);
    
    // Set up execution strategy for SOFT_MAX
    ggml_numa_execution_strategy_t strategy = {
        .node_strategy = NUMA_NODE_STRATEGY_SINGLE,  // SOFT_MAX works well on single node
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

// NUMA-aware ROPE execution with data parallelization across NUMA nodes
static enum ggml_status ggml_numa_execute_rope_data_parallel(
    struct ggml_numa_coordinator_manager * manager,
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context) {
    
    if (!manager || !operation || !context) {
        GGML_LOG_ERROR("Invalid parameters for ROPE data parallel execution\n");
        return GGML_STATUS_FAILED;
    }
    
    const int64_t ne00 = operation->ne[0];  // Head dimension
    const int64_t ne01 = operation->ne[1];  // Sequence length
    const int64_t ne02 = operation->ne[2];  // Number of heads / batch
    const int64_t ne03 = operation->ne[3];  // Batch dimension
    
    GGML_LOG_INFO("NUMA ROPE data parallel execution: [%ld, %ld, %ld, %ld] across %d NUMA nodes\n", 
                  ne00, ne01, ne02, ne03, context->numa_nodes);
    
    // Calculate work buffer size for ROPE operations
    // ROPE needs cache space for sin/cos values: (ne00 + CACHE_LINE_SIZE_F32) * sizeof(float) per node
    // Since we use single-threaded params (ith=0, nth=1) on each node, each node needs this much
    const size_t rope_work_buffer_size = (ne00 + CACHE_LINE_SIZE_F32) * sizeof(float);
    
    // Create work context for each NUMA node
    ggml_numa_dispatcher_work_context_t * work_context = ggml_numa_dispatcher_create_work_context(
        (struct ggml_tensor *)operation,
        "ROPE_DATA_PARALLEL",
        NULL,  // No additional context needed - NUMA mirroring handles data distribution
        0      // No additional context size
    );
    
    if (!work_context) {
        GGML_LOG_ERROR("Failed to create work context for ROPE data parallel execution\n");
        return GGML_STATUS_FAILED;
    }
    
    // Set up execution strategy for ROPE data parallelism - FORCED SINGLE FOR DEBUGGING
    ggml_numa_execution_strategy_t strategy = {
        .node_strategy = NUMA_NODE_STRATEGY_SINGLE,  // FORCED SINGLE FOR DEBUGGING
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD
    };
    
    // Submit work using specialized ROPE function pointer for data parallel execution
    int work_id = ggml_numa_coordinator_manager_submit_work_function(
        manager,
        ggml_numa_work_function_rope_chunk,  // Specialized ROPE chunk function
        work_context,                        // Context data
        -1,                                  // Auto-select NUMA nodes for data parallelism
        strategy,                            // Data parallel execution strategy
        rope_work_buffer_size                // Buffer requirements
    );
    
    if (work_id < 0) {
        GGML_LOG_ERROR("Failed to submit ROPE data parallel function pointer work\n");
        ggml_numa_dispatcher_free_work_context(work_context);
        return GGML_STATUS_FAILED;
    }
    
    GGML_LOG_DEBUG("Submitted ROPE data parallel function pointer work (ID: %d) across %d NUMA nodes\n", 
                   work_id, context->numa_nodes);
    
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
    struct ggml_numa_coordinator_manager * mgr = ggml_numa_coordinator_manager_get_global(n_threads);
    if (!mgr) {
        GGML_LOG_ERROR("Failed to create NUMA coordinator manager\n");
        return GGML_STATUS_FAILED;
    }
    
    // Process each operation in the graph through proper dispatch (not fallback)
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * operation = cgraph->nodes[i];
        if (!operation) continue;

        // Use proper dispatch which includes direct MUL_MAT execution
        enum ggml_status result = ggml_numa_dispatch_operation(mgr, operation, NULL);
        
        if (result != GGML_STATUS_SUCCESS) {
            GGML_LOG_ERROR("NUMA dispatch failed for operation %d (%s)\n", 
                          i, ggml_op_name(operation->op));
            return GGML_STATUS_FAILED;
        }
        
        GGML_LOG_DEBUG("Successfully executed operation %d (%s) via NUMA dispatch\n", 
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
        struct ggml_numa_coordinator_manager * mgr = ggml_numa_coordinator_manager_get_global(n_threads);
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
    
    // SIMPLIFIED: Trust coordinator's availability - if it's active, NUMA coordination is worthwhile
    // The coordinator will only be initialized if NUMA is available and beneficial
    GGML_LOG_DEBUG("NUMA coordinator-based decision: coordination should proceed\n");

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

    // SIMPLIFIED: Trust coordinator's NUMA configuration - no independent detection
    GGML_LOG_DEBUG("Coordinator-centric: proceeding with NUMA coordination\n");    // Estimate if the computational load is large enough to justify NUMA coordination overhead
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
    
    GGML_LOG_INFO("NUMA coordination beneficial: %d operations, %d threads, %ld total elements\n",
                  cgraph->n_nodes, n_threads, total_elements);
    
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
    
    // SIMPLIFIED: Trust coordinator's NUMA configuration - no independent overrides
    if (total_elements < MIN_ELEMENTS_FOR_NUMA) {
        // Too small for NUMA overhead
        *strategy = (ggml_numa_execution_strategy_t){
            .node_strategy = NUMA_NODE_STRATEGY_SINGLE,
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
                .node_strategy = NUMA_NODE_STRATEGY_SINGLE,
                .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
            };
            *recommended_chunks = 1;
            GGML_LOG_DEBUG("ADD: Poor tensor shape (%ld elements, ne0=%ld), using single-node execution\n", 
                           total_elements, ne0);
        }
    }
    
    return GGML_STATUS_SUCCESS;
}

// Enhanced ADD handler with better shape analysis and multi-level parallelism
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
    .parallel_efficiency_estimate = 0.97f,  // Improved efficiency with multi-level parallelism
    .requires_synchronization = false,
    .supports_in_place = true,
    .analyze = ggml_numa_analyze_add_enhanced  // Use enhanced analysis
};

// MUL_MAT handler is now defined in numa-work/ggml-numa-mulmat.c

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

// Enhanced SOFT_MAX handler with row-based parallelization
const ggml_numa_operation_handler_t ggml_numa_handler_soft_max = {
    .operation_type = GGML_OP_SOFT_MAX,
    .default_strategy = {
        .node_strategy = NUMA_NODE_STRATEGY_SINGLE,  // FORCED SINGLE FOR DEBUGGING
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD
    },
    .complexity = NUMA_OP_COMPLEXITY_MODERATE,
    .workload_type = NUMA_OP_COMPUTE_BOUND,
    .min_elements_for_parallel = 10000,  // Lower threshold for SOFT_MAX
    .optimal_chunk_size = 1 * 1024 * 1024,  // 1MB chunks
    .parallel_efficiency_estimate = 0.80f,  // Good efficiency for attention operations
    .requires_synchronization = false,
    .supports_in_place = false,
    .analyze = NULL  // FIXED: Remove custom analyzer, now use data parallel
};

// GLU (SwiGLU) handler - element-wise operation with NUMA coordination
const ggml_numa_operation_handler_t ggml_numa_handler_glu = {
    .operation_type = GGML_OP_GLU,
    .default_strategy = {
        .node_strategy = NUMA_NODE_STRATEGY_SINGLE,  // FORCED SINGLE FOR DEBUGGING
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD
    },
    .complexity = NUMA_OP_COMPLEXITY_SIMPLE,
    .workload_type = NUMA_OP_COMPUTE_BOUND,
    .min_elements_for_parallel = 5000,   // Lower threshold due to activation computation
    .optimal_chunk_size = 512 * 1024,    // 512KB chunks for good cache performance
    .parallel_efficiency_estimate = 0.95f,  // Very high efficiency for element-wise operations
    .requires_synchronization = false,
    .supports_in_place = false,  // GLU typically writes to different tensor
    .analyze = NULL  // Use default strategy
};

// RMS_NORM handler - row-wise normalization with NUMA data parallelism and multi-level parallelism
const ggml_numa_operation_handler_t ggml_numa_handler_rms_norm = {
    .operation_type = GGML_OP_RMS_NORM,
    .default_strategy = {
        .node_strategy = NUMA_NODE_STRATEGY_SINGLE,  // FORCED SINGLE FOR DEBUGGING
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD
    },
    .complexity = NUMA_OP_COMPLEXITY_MODERATE,  // Has reduction phase per row but still parallelizable
    .workload_type = NUMA_OP_COMPUTE_BOUND,     // Involves square, sum, sqrt operations
    .min_elements_for_parallel = 10000,         // Higher threshold due to per-row reduction overhead
    .optimal_chunk_size = 256 * 1024,           // 256KB chunks for good cache performance
    .parallel_efficiency_estimate = 0.90f,      // Improved efficiency with multi-level parallelism
    .requires_synchronization = false,          // Each row is independent
    .supports_in_place = true,                  // RMS_NORM can write to same memory location
    .analyze = NULL  // Use default strategy
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
    // For legacy calls without buffer size, pass 0 and let fallback handle it
    return ggml_numa_coordinator_manager_submit_work(manager, operation, target_numa_node, strategy, 0);
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
    
    // SIMPLIFIED: Get NUMA node count from coordinator instead of independent detection
    g_numa_nodes_count = ggml_numa_coordinator_get_num_nodes();
    if (g_numa_nodes_count <= 0) {
        // Fallback if coordinator not initialized
        g_numa_nodes_count = 1;
    }
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
        extern struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_get_global(int n_threads);
        s_global_manager = ggml_numa_coordinator_manager_get_global(params->nth);
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
    
    if (!ggml_numa_should_dispatch()) {
        GGML_LOG_DEBUG("NUMA dispatch disabled, skipping dispatcher\n");
        return -1;  // Let caller use standard processing
    }
    
    GGML_LOG_INFO("Processing computation graph with %d nodes through NUMA dispatcher\n", cgraph->n_nodes);
    
    // Initialize dispatcher if needed
    ggml_numa_dispatch_init();
    
    // Get coordinator manager (should already be initialized by llama-context.cpp)
    extern struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_get_global(int n_threads);
    struct ggml_numa_coordinator_manager * manager = ggml_numa_coordinator_manager_get_global(n_threads);  // Use existing coordinator settings
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

// Utility function to extract operation name from tensor for logging
const char * ggml_numa_get_operation_name(const struct ggml_tensor * tensor) {
    if (!tensor) return "NULL_TENSOR";
    return ggml_op_name(tensor->op);
}

// Create work context for function pointer execution
ggml_numa_dispatcher_work_context_t * ggml_numa_dispatcher_create_work_context(
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
void ggml_numa_dispatcher_free_work_context(ggml_numa_dispatcher_work_context_t * context) {
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
size_t ggml_numa_dispatcher_calculate_work_buffer_size(const struct ggml_tensor * operation) {
    if (!operation) return 0;
    
    switch (operation->op) {
        case GGML_OP_SOFT_MAX: {
            // SOFT_MAX needs (ne00 + CACHE_LINE_SIZE_F32) * sizeof(float)
            // Since we use single-threaded params (ith=0, nth=1), only need space for one thread
            const int64_t ne00 = operation->ne[0];
            return (ne00 + CACHE_LINE_SIZE_F32) * sizeof(float);
        }
        case GGML_OP_ROPE: {
            // ROPE operations need cache buffer: (ne0 + CACHE_LINE_SIZE_F32) * sizeof(float)
            // Since we use single-threaded params (ith=0, nth=1), only need space for one thread
            const int64_t ne00 = operation->ne[0];
            return (ne00 + CACHE_LINE_SIZE_F32) * sizeof(float);
        }
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM: {
            // Normalization operations need space for intermediate calculations
            return ggml_nelements(operation) * sizeof(float);
        }
        case GGML_OP_MUL_MAT: {
            // Matrix multiply needs space for type conversion if required
            const struct ggml_tensor * src0 = operation->src[0];
            const struct ggml_tensor * src1 = operation->src[1];
            
            if (!src0 || !src1) {
                return 0;
            }
            
            // Check if type conversion is needed
            const struct ggml_type_traits_cpu * traits = ggml_get_type_traits_cpu(src0->type);
            enum ggml_type vec_dot_type = traits->vec_dot_type;
            
            if (src1->type != vec_dot_type) {
                // Calculate buffer size for converting ALL batches of src1 (required by ggml-cpu.c:1479)
                const int64_t ne10 = src1->ne[0];
                const int64_t ne11 = src1->ne[1]; 
                const int64_t ne12 = src1->ne[2];
                const int64_t ne13 = src1->ne[3]; // All batches
                
                const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
                const size_t nbw2 = nbw1 * ne11;
                const size_t nbw3 = nbw2 * ne12;
                const size_t total_size = ne13 * nbw3; // ALL batches (matches assertion requirement)
                
                GGML_LOG_DEBUG("🔧 MUL_MAT work buffer sized for full conversion: %zu bytes (ne13=%ld, nbw3=%zu)\n", 
                              total_size, ne13, nbw3);
                return total_size;
            }
            
            // No conversion needed, minimal buffer
            return 1024; // Small buffer for scratch space
        }
        case GGML_OP_GROUP_NORM: {
            // Group normalization needs working space
            return ggml_nelements(operation) * sizeof(float);
        }
        case GGML_OP_FLASH_ATTN_EXT: {
            // Flash attention needs working space for VKQ accumulator, V32 buffer, and Q quantized
            // Based on the implementation in ops.cpp, we need:
            // - DV * sizeof(float) for VKQ32 (FP32 VKQ accumulator)  
            // - DV * sizeof(float) for V32 (temporary FP32 V buffer)
            // - DK * sizeof(ggml_fp16_t) for Q_q (quantized Q buffer)
            // - CACHE_LINE_SIZE_F32 for alignment
            const struct ggml_tensor * q = operation->src[0];
            const struct ggml_tensor * v = operation->src[2];
            if (!q || !v) {
                return 0;
            }
            const int64_t DK = q->ne[0];  // Key/Query dimension
            const int64_t DV = v->ne[0];  // Value dimension
            
            size_t buffer_size = (1*DK + 2*DV + CACHE_LINE_SIZE_F32) * sizeof(float);
            GGML_LOG_DEBUG("🔧 FLASH_ATTN_EXT work buffer: DK=%ld, DV=%ld, size=%zu bytes\n", DK, DV, buffer_size);
            return buffer_size;
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
    GGML_LOG_DEBUG("🔥 ENTERING MUL_MAT_fallback function - FIRST LINE!\n");

    NUMA_ASSERT(work_context && params);
    
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
    NUMA_ASSERT(result == GGML_STATUS_SUCCESS);
    
    return result;
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
    
    // Get the vec_dot_type the same way as ggml_compute_forward_mul_mat
    const struct ggml_type_traits_cpu * traits = ggml_get_type_traits_cpu(src0->type);
    enum ggml_type vec_dot_type = traits->vec_dot_type;
    
    if (src1->type != vec_dot_type) {
        // Use the same calculation as ggml_compute_forward_mul_mat
        const size_t nbw1 = ggml_row_size(vec_dot_type, src1->ne[0]);  // ne10
        const size_t nbw2 = nbw1 * src1->ne[1];                       // nbw1*ne11
        const size_t nbw3 = nbw2 * src1->ne[2];                       // nbw2*ne12
        work_size = src1->ne[3] * nbw3;                                // ne13*nbw3
        
        GGML_LOG_DEBUG("MUL_MAT work buffer calculation: vec_dot_type=%d, src1_type=%d, nbw1=%zu, nbw2=%zu, nbw3=%zu, work_size=%zu\n",
                       vec_dot_type, src1->type, nbw1, nbw2, nbw3, work_size);
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
            .node_strategy = NUMA_NODE_STRATEGY_SINGLE,
            .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
        };
    } else {
        GGML_LOG_DEBUG("Using single MUL_MAT work function (complexity=%ld)\n", complexity);
        work_function = ggml_numa_work_function_mul_mat_single;
        strategy = (ggml_numa_execution_strategy_t){
            .node_strategy = NUMA_NODE_STRATEGY_SINGLE,
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
            GGML_LOG_ERROR("NUMA data parallel submission failed, falling back to fallback execution\n");
            return ggml_numa_fallback_execute((struct ggml_tensor *)operation, NULL);
        }
        
        GGML_LOG_INFO("NUMA data parallel work submitted (work group ID: %d)\n", work_group_id);
        return GGML_STATUS_SUCCESS;
    }
    
    // For medium matrices, use fallback execution
    return ggml_numa_fallback_execute((struct ggml_tensor *)operation, NULL);
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

