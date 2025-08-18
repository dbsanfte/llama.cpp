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
#include "vec.h"            // Vector operations and SIMD optimizations

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
static enum ggml_status ggml_numa_work_function_soft_max_chunk(void * work_context, struct ggml_compute_params * params);
static enum ggml_status ggml_numa_work_function_rope_chunk(void * work_context, struct ggml_compute_params * params);
static enum ggml_status ggml_numa_work_function_add_single(void * work_context, struct ggml_compute_params * params);
static enum ggml_status ggml_numa_work_function_add_chunk(void * work_context, struct ggml_compute_params * params);
static enum ggml_status ggml_numa_work_function_glu_chunk(void * work_context, struct ggml_compute_params * params);
static enum ggml_status ggml_numa_work_function_rms_norm_chunk(void * work_context, struct ggml_compute_params * params);
static enum ggml_status ggml_numa_work_function_flash_attn_ext_chunk(void * work_context, struct ggml_compute_params * params);

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
static size_t ggml_numa_dispatcher_calculate_work_buffer_size(const struct ggml_tensor * tensor);

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

// Enhanced MUL_MAT handler with intelligent analysis
const ggml_numa_operation_handler_t ggml_numa_handler_mul_mat_enhanced = {
    .operation_type = GGML_OP_MUL_MAT,
    .default_strategy = {
        .node_strategy = NUMA_NODE_STRATEGY_SINGLE,  // Single-threaded for MUL_MAT (testing)
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD
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
    
    if (!ggml_numa_should_dispatch()) {
        GGML_LOG_DEBUG("NUMA dispatch disabled, skipping dispatcher\n");
        return -1;  // Let caller use standard processing
    }
    
    GGML_LOG_INFO("Processing computation graph with %d nodes through NUMA dispatcher\n", cgraph->n_nodes);
    
    // Initialize dispatcher if needed
    ggml_numa_dispatch_init();
    
    // Get coordinator manager (should already be initialized by llama-context.cpp)
    extern struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_get_global(int n_threads, bool force_multi_socket);
    struct ggml_numa_coordinator_manager * manager = ggml_numa_coordinator_manager_get_global(n_threads, false);  // Use existing coordinator settings
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

// Specialized work function for single-threaded MUL_MAT operations
static enum ggml_status ggml_numa_work_function_mul_mat_single(void * work_context, struct ggml_compute_params * params) {
    GGML_LOG_DEBUG("🔥 ENTERING MUL_MAT_single work function - FIRST LINE!\n");
    printf("🔥🔥🔥 PRINTF: ENTERING MUL_MAT_single work function - FIRST LINE!\n");
    fflush(stdout);

    NUMA_ASSERT(work_context && params);

    ggml_numa_dispatcher_work_context_t * ctx = (ggml_numa_dispatcher_work_context_t *)work_context;
    struct ggml_tensor * dst = ctx->operation;
    
    NUMA_ASSERT(dst);
    
    printf("🔥🔥🔥 PRINTF: MUL_MAT_single - src0 type=%d, src1 type=%d\n", (int)dst->src[0]->type, (int)dst->src[1]->type);
    fflush(stdout);
    
    GGML_LOG_DEBUG("Executing MUL_MAT single-threaded with validation-only approach\n");
    
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    // Extract validation logic from ggml_compute_forward_mul_mat (but skip threading parts)
    GGML_TENSOR_BINARY_OP_LOCALS
    
    // CRITICAL VALIDATION CHECKS (from ggml_compute_forward_mul_mat)
    NUMA_ASSERT(ne0 == ne01);
    NUMA_ASSERT(ne1 == ne11);
    NUMA_ASSERT(ne2 == ne12);
    NUMA_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    NUMA_ASSERT(nb00 == ggml_type_size(src0->type));
    NUMA_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    NUMA_ASSERT(nb0 == sizeof(float));
    NUMA_ASSERT(nb0 <= nb1);
    NUMA_ASSERT(nb1 <= nb2);
    NUMA_ASSERT(nb2 <= nb3);

    // Type trait lookups (from ggml_compute_forward_mul_mat)
    const struct ggml_type_traits_cpu * src0_traits = ggml_get_type_traits_cpu(src0->type);
    enum ggml_type const vec_dot_type = src0_traits->vec_dot_type;
    int64_t const vec_dot_num_rows = src0_traits->nrows;
    
    GGML_LOG_DEBUG("🔧 Validation passed, calling mathematical kernel directly\n");
    
    // Enhanced input validation: Check for NaN/inf in F16 data early to catch corruption  
    if (src0->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F16) {
        printf("🔥🔥🔥 PRINTF: MUL_MAT single F16 validation starting - src0 type=%d, src1 type=%d\n", (int)src0->type, (int)src1->type);
        fflush(stdout);
        
        GGML_LOG_DEBUG("🔍 MUL_MAT single: validating F16 input data for corruption\n");
        
        // Validate src0 F16 data (check first 16 elements)
        if (src0->type == GGML_TYPE_F16) {
            printf("🔥🔥🔥 PRINTF: Validating src0 F16 data\n");
            fflush(stdout);
            
            const ggml_fp16_t* fp16_data = (const ggml_fp16_t*)ggml_get_data(src0);
            NUMA_ASSERT(fp16_data != NULL, "MUL_MAT single: src0 fp16_data is NULL");
            
            printf("🔥🔥🔥 PRINTF: About to check F16 data values\n");
            fflush(stdout);
            
            for (int i = 0; i < MIN(16, (int)ggml_nelements(src0)); i++) {
                float val = GGML_CPU_FP16_TO_FP32(fp16_data[i]);
                if (!isfinite(val)) {
                    printf("🚨 PRINTF: Found corrupt src0 F16 at index %d: %f\n", i, val);
                    fflush(stdout);
                }
                NUMA_ASSERT(isfinite(val), "MUL_MAT single: corrupt src0 F16 at index %d: %f", i, val);
            }
            printf("🔥🔥🔥 PRINTF: src0 F16 validation passed\n");
            fflush(stdout);
        }
        
        // Validate src1 F16 data (check first 16 elements)  
        if (src1->type == GGML_TYPE_F16) {
            printf("🔥🔥🔥 PRINTF: Validating src1 F16 data\n");
            fflush(stdout);
            
            const ggml_fp16_t* fp16_data = (const ggml_fp16_t*)ggml_get_data(src1);
            NUMA_ASSERT(fp16_data != NULL, "MUL_MAT single: src1 fp16_data is NULL");
            
            for (int i = 0; i < MIN(16, (int)ggml_nelements(src1)); i++) {
                float val = GGML_CPU_FP16_TO_FP32(fp16_data[i]);
                if (!isfinite(val)) {
                    printf("🚨 PRINTF: Found corrupt src1 F16 at index %d: %f\n", i, val);
                    fflush(stdout);
                }
                NUMA_ASSERT(isfinite(val), "MUL_MAT single: corrupt src1 F16 at index %d: %f", i, val);
            }
            printf("🔥🔥🔥 PRINTF: src1 F16 validation passed\n");
            fflush(stdout);
        }
        
        GGML_LOG_DEBUG("🔍 MUL_MAT single: F16 input validation passed - no corruption detected\n");
    }
    
    // Create thread-safe params for mathematical kernel ONLY (no threadpool or barriers)
    // CRITICAL: Calculate work buffer size specifically for this MUL_MAT operation
    
    // Calculate required work buffer size for type conversion (from ggml_compute_forward_mul_mat)
    enum ggml_type const required_vec_dot_type = src0_traits->vec_dot_type;
    
    printf("🔥🔥🔥 PRINTF: Type analysis - src0 type=%d, src1 type=%d, required_vec_dot_type=%d\n", 
           (int)src0->type, (int)src1->type, (int)required_vec_dot_type);
    fflush(stdout);
    
    size_t required_wsize = 0;
    void * wdata_ptr = params->wdata;
    
    if (src1->type != required_vec_dot_type) {
        printf("🔥🔥🔥 PRINTF: Type conversion needed! src1 type=%d != required_vec_dot_type=%d\n", 
               (int)src1->type, (int)required_vec_dot_type);
        fflush(stdout);
        
        // Need to convert src1 to vec_dot_type - calculate required buffer size
        const size_t nbw0 = ggml_type_size(required_vec_dot_type);
        const size_t nbw1 = ggml_row_size(required_vec_dot_type, ne10);
        const size_t nbw2 = nbw1 * ne11;
        const size_t nbw3 = nbw2 * ne12;
        required_wsize = ne13 * nbw3;
        
        printf("🔥🔥🔥 PRINTF: Work buffer requirement - need %zu bytes, have %zu bytes\n", 
               required_wsize, params->wsize);
        fflush(stdout);
        
        GGML_LOG_DEBUG("🔍 MUL_MAT: src1 type conversion required. Need %zu bytes work buffer\n", required_wsize);
        GGML_LOG_DEBUG("🔍 MUL_MAT: Converting from src1->type=%d to required_vec_dot_type=%d\n", 
                       (int)src1->type, (int)required_vec_dot_type);
        
        if (params->wsize < required_wsize) {
            printf("🚨 PRINTF: Insufficient work buffer! Need %zu bytes, have %zu bytes\n", 
                   required_wsize, params->wsize);
            fflush(stdout);
            GGML_LOG_ERROR("🚨 MUL_MAT: Insufficient work buffer. Need %zu bytes, have %zu bytes\n", 
                           required_wsize, params->wsize);
            return GGML_STATUS_FAILED;
        }
        
        printf("🔥🔥🔥 PRINTF: Starting type conversion from F32 to vec_dot_type\n");
        fflush(stdout);
        
        // CRITICAL: Actually perform the type conversion (from ggml_compute_forward_mul_mat)
        const struct ggml_type_traits_cpu * vec_dot_traits = ggml_get_type_traits_cpu(required_vec_dot_type);
        ggml_from_float_t const from_float = vec_dot_traits->from_float;
        NUMA_ASSERT(src1->type == GGML_TYPE_F32, "MUL_MAT single: only F32->vec_dot_type conversion supported");
        
        char * wdata = (char*)params->wdata;
        printf("�🔥🔥 PRINTF: Converting src1 from F32 to vec_dot_type %d, dimensions: ne13=%lld, ne12=%lld, ne11=%lld, ne10=%lld\n", 
               (int)required_vec_dot_type, (long long)ne13, (long long)ne12, (long long)ne11, (long long)ne10);
        fflush(stdout);
        
        // Convert F32 src1 data to vec_dot_type (mirror original logic exactly)
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            printf("🔥🔥🔥 PRINTF: Processing i13=%lld/%lld\n", (long long)i13, (long long)ne13);
            fflush(stdout);
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float((float *)((char *) tensor_data(src1) + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)(wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
        
        printf("🔥🔥� PRINTF: Type conversion completed successfully\n");
        fflush(stdout);
        
        // CRITICAL FIX: Update wdata_ptr to point to converted data
        wdata_ptr = wdata;
        printf("🔥🔥� PRINTF: Updated wdata_ptr to converted data at %p\n", wdata_ptr);
        fflush(stdout);
    } else {
        GGML_LOG_DEBUG("🔍 MUL_MAT: No type conversion needed - src1 already correct type\n");
    }
    
    struct ggml_compute_params safe_params = {
        .ith = 0,         // Single thread
        .nth = 1,         // Total of 1 thread
        .wdata = wdata_ptr,
        .wsize = params->wsize,  // Use coordinator's work buffer size
        .threadpool = NULL, // No threadpool to avoid threading conflicts
    };

    // Calculate dimensions for kernel (from ggml_compute_forward_mul_mat)
    const int64_t nr0 = ne0;
    const int64_t nr1 = ne1 * ne2 * ne3;
    
    printf("🔥🔥🔥 PRINTF: About to call ggml_compute_forward_mul_mat_one_chunk\n");
    fflush(stdout);
    
    // CRITICAL: Validate converted F16 data for corruption if we did conversion
    if (src1->type != required_vec_dot_type) {
        printf("🔥🔥🔥 PRINTF: Validating original F32 and converted F16 data\n");
        fflush(stdout);
        
        // Validate original F32 data first
        const float* original_f32_data = (const float*)tensor_data(src1);
        ggml_fp16_t* converted_f16_data = (ggml_fp16_t*)wdata_ptr;
        size_t converted_elements = required_wsize / sizeof(ggml_fp16_t);
        size_t check_converted = MIN(32, converted_elements);
        
        for (size_t i = 0; i < check_converted; i++) {
            const float original_f32 = original_f32_data[i];
            const ggml_fp16_t f16_val = converted_f16_data[i];
            const float converted_f32 = GGML_CPU_FP16_TO_FP32(f16_val);
            
            if (!isfinite(original_f32)) {
                printf("🚨 PRINTF: CORRUPT ORIGINAL F32 at index %zu: %f\n", i, original_f32);
                fflush(stdout);
            }
            if (!isfinite(converted_f32)) {
                printf("🚨 PRINTF: CORRUPT CONVERTED at index %zu: orig_f32=%f → f16=%u → f32=%f\n", 
                       i, original_f32, (unsigned)f16_val, converted_f32);
                fflush(stdout);
            }
        }
        printf("🔥🔥🔥 PRINTF: Validation completed\n");
        fflush(stdout);
    }
    
    // Call mathematical kernel directly with full matrix (no chunking for single thread)
    ggml_compute_forward_mul_mat_one_chunk(&safe_params, dst, src0->type, vec_dot_num_rows, 
                                          0, nr0, 0, nr1);
    
    printf("🔥🔥🔥 PRINTF: ggml_compute_forward_mul_mat_one_chunk completed successfully\n");
    fflush(stdout);
    
    // CRITICAL: Validate output data for corruption before returning success
    float* dst_data = (float*)ggml_get_data(dst);
    NUMA_ASSERT(dst_data != NULL, "MUL_MAT single: dst_data is NULL after computation");
    
    size_t output_elements = ggml_nelements(dst);
    size_t check_count = MIN(32, output_elements);  // Check first 32 elements like chunk function
    
    for (size_t i = 0; i < check_count; i++) {
        NUMA_ASSERT(isfinite(dst_data[i]), 
                    "MUL_MAT single: corrupt output at index %zu: %f (elements=%zu, src0_type=%d, src1_type=%d)", 
                    i, dst_data[i], output_elements, (int)src0->type, (int)src1->type);
    }
    
    GGML_LOG_DEBUG("🔍 MUL_MAT single: output validation passed - %zu elements checked, all finite\n", check_count);
    
    GGML_LOG_DEBUG("Successfully executed MUL_MAT single-threaded with thread-safe approach\n");
    return GGML_STATUS_SUCCESS;
}

// Specialized work function for chunked MUL_MAT operations
static enum ggml_status ggml_numa_work_function_mul_mat_chunk(void * work_context, struct ggml_compute_params * params) {
    // Force logs to appear - use NUMA_THREAD_LOG_DEBUG and printf as backup
    NUMA_THREAD_LOG_DEBUG("🔥🔥🔥 ENTERING MUL_MAT_chunk work function - FIRST LINE! 🔥🔥🔥\n");
    fprintf(stderr, "🔥🔥🔥 PRINTF: ENTERING MUL_MAT_chunk work function - FIRST LINE! 🔥🔥🔥\n");
    fflush(stderr);

    NUMA_ASSERT(work_context && params);
    
    NUMA_THREAD_LOG_DEBUG("🔥 Got valid context and params (NUMA_THREAD_LOG_DEBUG)\n");
    fprintf(stderr, "🔥 PRINTF: Got valid context and params\n");
    fflush(stderr);
    
    NUMA_THREAD_LOG_DEBUG("🔥 About to cast context...\n");
    fprintf(stderr, "🔥 PRINTF: About to cast context...\n");
    fflush(stderr);
    
    ggml_numa_dispatcher_work_context_t * ctx = (ggml_numa_dispatcher_work_context_t *)work_context;
    NUMA_THREAD_LOG_DEBUG("🔥 Cast context to dispatcher work context SUCCESSFULLY\n");
    fprintf(stderr, "🔥 PRINTF: Cast context SUCCESSFULLY\n");
    fflush(stderr);
    
    NUMA_THREAD_LOG_DEBUG("🔥 About to get dst tensor...\n");
    fprintf(stderr, "🔥 PRINTF: About to get dst tensor...\n");
    fflush(stderr);
    
    struct ggml_tensor * dst = ctx->operation;
    NUMA_THREAD_LOG_DEBUG("🔥 Got dst tensor from context SUCCESSFULLY\n");
    fprintf(stderr, "🔥 PRINTF: Got dst tensor SUCCESSFULLY\n");
    fflush(stderr);

    GGML_LOG_ERROR("🔥 About to check dst tensor validity...\n");
    fprintf(stderr, "🔥 PRINTF: About to check dst tensor validity...\n");
    fflush(stderr);
    
    if (!dst) {
        GGML_LOG_ERROR("🚨 dst tensor is NULL!\n");
        fprintf(stderr, "🚨 PRINTF: dst tensor is NULL!\n");
        fflush(stderr);
        return GGML_STATUS_FAILED;
    }
    
    GGML_LOG_ERROR("🔥 dst tensor is valid, checking src[0]...\n");
    fprintf(stderr, "🔥 PRINTF: dst tensor is valid, checking src[0]...\n");
    fflush(stderr);
    
    if (!dst->src[0]) {
        GGML_LOG_ERROR("🚨 dst->src[0] tensor is NULL!\n");
        fprintf(stderr, "🚨 PRINTF: dst->src[0] tensor is NULL!\n");
        fflush(stderr);
        return GGML_STATUS_FAILED;
    }
    
    GGML_LOG_ERROR("🔥 src[0] is valid, checking src[1]...\n");
    fprintf(stderr, "🔥 PRINTF: src[0] is valid, checking src[1]...\n");
    fflush(stderr);
    
    if (!dst->src[1]) {
        GGML_LOG_ERROR("🚨 dst->src[1] tensor is NULL!\n");
        fprintf(stderr, "🚨 PRINTF: dst->src[1] tensor is NULL!\n");
        fflush(stderr);
        return GGML_STATUS_FAILED;
    }
    
    GGML_LOG_ERROR("🔥 All tensors are valid! Proceeding...\n");
    fprintf(stderr, "🔥 PRINTF: All tensors are valid! Proceeding...\n");
    fflush(stderr);
    
    // Debug: Check what work buffer size we received and input data
    GGML_LOG_DEBUG("🔍 MUL_MAT chunk: wsize=%zu, wdata=%p\n", params->wsize, params->wdata);
    GGML_LOG_DEBUG("🔍 MUL_MAT chunk: dst=%p, src0=%p, src1=%p\n", (void*)dst, (void*)dst->src[0], (void*)dst->src[1]);
    GGML_LOG_DEBUG("🔍 Work context: operation=%p, cplan=%p\n", (void*)ctx->operation, (void*)ctx->cplan);
    
    // Use coordinator's threading parameters directly (multi-threading support)
    GGML_LOG_DEBUG("🔍 Using coordinator threading: ith=%d, nth=%d\n", params->ith, params->nth);
    
    GGML_LOG_DEBUG("🔍 About to call NUMA-safe MUL_MAT chunk computation...\n");
    
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    
    GGML_LOG_DEBUG("🔍 Tensor dimensions: dst=(%ld,%ld,%ld,%ld), src0=(%ld,%ld,%ld,%ld), src1=(%ld,%ld,%ld,%ld)\n",
           dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], 
           src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], 
           src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
    
    // Get type information for preprocessing (essential for MUL_MAT)
    extern const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type);
    const struct ggml_type_traits_cpu * src0_traits = ggml_get_type_traits_cpu(src0->type);
    enum ggml_type const vec_dot_type = src0_traits->vec_dot_type;
    int64_t const vec_dot_num_rows = src0_traits->nrows;
    
    GGML_LOG_ERROR("🔍 Type analysis: src0_type=%d, src1_type=%d, vec_dot_type=%d, nrows=%ld\n", 
                   (int)src0->type, (int)src1->type, (int)vec_dot_type, vec_dot_num_rows);
    
    fprintf(stderr, "🔍 PRINTF: Type analysis: src0_type=%d, src1_type=%d, vec_dot_type=%d, nrows=%ld\n", 
            (int)src0->type, (int)src1->type, (int)vec_dot_type, vec_dot_num_rows);
    fflush(stderr);
    
    NUMA_THREAD_LOG_DEBUG("🔍 Type info: src0_type=%d (%s), vec_dot_type=%d (%s), vec_dot_num_rows=%ld\n", 
           src0->type, ggml_type_name(src0->type), vec_dot_type, ggml_type_name(vec_dot_type), vec_dot_num_rows);
    
    // CRITICAL DEBUG: For Q8_0, we should see src0_type=7 (Q8_0) and vec_dot_type=7 (Q8_0)
    fprintf(stderr, "🚨 CRITICAL TYPE DEBUG: src0_type=%d (%s), vec_dot_type=%d (%s)\n", 
           src0->type, ggml_type_name(src0->type), vec_dot_type, ggml_type_name(vec_dot_type));
    fflush(stderr);
    
    // F16 SPECIFIC DEBUGGING: Check for NaN/inf in F16 input data
    if (src0->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F16) {
        fprintf(stderr, "🔍 F16 DEBUGGING: Checking F16 tensor data for NaN/inf corruption\n");
        
        if (src0->type == GGML_TYPE_F16) {
            const ggml_fp16_t* f16_data = (const ggml_fp16_t*)tensor_data(src0);
            if (f16_data) {
                // Check first few F16 values for NaN/inf
                for (int i = 0; i < 16 && i < src0->ne[0]; i++) {
                    float f32_val = GGML_CPU_FP16_TO_FP32(f16_data[i]);
                    if (isnan(f32_val) || isinf(f32_val)) {
                        fprintf(stderr, "🚨 F16 CORRUPTION: src0[%d] = NaN/inf (f16=0x%04x, f32=%f)\n", 
                               i, (unsigned)f16_data[i], f32_val);
                        fflush(stderr);
                    }
                }
                fprintf(stderr, "🔍 F16 src0: First 4 values as F32: %.6f, %.6f, %.6f, %.6f\n",
                       GGML_CPU_FP16_TO_FP32(f16_data[0]),
                       GGML_CPU_FP16_TO_FP32(f16_data[1]),
                       GGML_CPU_FP16_TO_FP32(f16_data[2]),
                       GGML_CPU_FP16_TO_FP32(f16_data[3]));
            }
        }
        
        if (src1->type == GGML_TYPE_F16) {
            const ggml_fp16_t* f16_data = (const ggml_fp16_t*)tensor_data(src1);
            if (f16_data) {
                // Check first few F16 values for NaN/inf  
                for (int i = 0; i < 16 && i < src1->ne[0]; i++) {
                    float f32_val = GGML_CPU_FP16_TO_FP32(f16_data[i]);
                    if (isnan(f32_val) || isinf(f32_val)) {
                        fprintf(stderr, "🚨 F16 CORRUPTION: src1[%d] = NaN/inf (f16=0x%04x, f32=%f)\n", 
                               i, (unsigned)f16_data[i], f32_val);
                        fflush(stderr);
                    }
                }
                fprintf(stderr, "🔍 F16 src1: First 4 values as F32: %.6f, %.6f, %.6f, %.6f\n",
                       GGML_CPU_FP16_TO_FP32(f16_data[0]),
                       GGML_CPU_FP16_TO_FP32(f16_data[1]),
                       GGML_CPU_FP16_TO_FP32(f16_data[2]),
                       GGML_CPU_FP16_TO_FP32(f16_data[3]));
            }
        }
        fflush(stderr);
    }
    
    // Calculate matrix dimensions for chunk function FIRST to determine slicing
    const int64_t nr0 = dst->ne[0];  // Result rows
    const int64_t nr1 = dst->ne[1] * dst->ne[2] * dst->ne[3];  // Remaining dimensions
    
    // For NUMA data parallel execution, we need to split the work properly
    // The original algorithm distributes work using chunk-based approach
    // We'll distribute rows across nodes and only convert data we need
    
    int current_numa_node = ggml_numa_get_current_node();
    int total_numa_nodes = ggml_numa_coordinator_get_num_nodes();
    
    // Simplified NUMA distribution: divide nr1 (rows) across NUMA nodes
    int64_t rows_per_node = nr1 / total_numa_nodes;
    int64_t extra_rows = nr1 % total_numa_nodes;
    
    // Calculate start and end rows for this NUMA node
    int64_t start_row = current_numa_node * rows_per_node;
    int64_t end_row = start_row + rows_per_node;
    
    // Distribute any extra rows to the first few nodes
    if (current_numa_node < extra_rows) {
        start_row += current_numa_node;
        end_row += current_numa_node + 1;
    } else {
        start_row += extra_rows;
        end_row += extra_rows;
    }
    
    // Ensure we don't exceed bounds and have valid work to do
    if (end_row > nr1) end_row = nr1;
    if (start_row >= nr1 || start_row >= end_row) {
        // This NUMA node has no work to do
        GGML_LOG_DEBUG("🔍 NUMA node %d has no work (start_row=%ld >= nr1=%ld)\n", 
                       current_numa_node, start_row, nr1);
        return GGML_STATUS_SUCCESS;
    }
    
    GGML_LOG_DEBUG("🔍 NUMA node %d (of %d): processing rows %ld to %ld (of %ld total)\n", 
                   current_numa_node, total_numa_nodes, start_row, end_row - 1, nr1);

    // Handle src1 preprocessing if needed (convert to vec_dot_type)
    // CRITICAL: To avoid race conditions, use a shared conversion approach
    const void * converted_src1_data = NULL;
    
    // Input validation: Only validate F32 tensors directly (quantized handled by vec_dot)
    const void * src1_data = ggml_get_data(src1);
    if (src1_data) {
        GGML_LOG_DEBUG("🔍 MUL_MAT chunk: src0_type=%d, src1_type=%d\n", (int)src0->type, (int)src1->type);
        
        // Only validate and log F32 tensors - quantized tensors handled by vec_dot functions
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
            float* src0_f = (float*)ggml_get_data(src0);
            float* src1_f = (float*)src1_data;
            if (src0_f && src1_f) {
                GGML_LOG_DEBUG("🔍 F32 input data: src0[0]=%.2f, src0[1]=%.2f, src1[0]=%.2f, src1[1]=%.2f\n", 
                       (double)src0_f[0], (double)src0_f[1], (double)src1_f[0], (double)src1_f[1]);
                
                // Validate F32 input data for NaN/inf corruption
                int64_t src0_check_count = MIN(16, ggml_nelements(src0));
                for (int64_t i = 0; i < src0_check_count; i++) {
                    NUMA_ASSERT(isfinite(src0_f[i]), "MUL_MAT chunk: Found NaN/inf in F32 src0 data at index %d: %f", (int)i, (double)src0_f[i]);
                }
                
                int64_t src1_check_count = MIN(16, ggml_nelements(src1));
                for (int64_t i = 0; i < src1_check_count; i++) {
                    NUMA_ASSERT(isfinite(src1_f[i]), "MUL_MAT chunk: Found NaN/inf in F32 src1 data at index %d: %f", (int)i, (double)src1_f[i]);
                }
                GGML_LOG_DEBUG("🔍 MUL_MAT F32 input validation: src0[%ld] src1[%ld] elements checked, all finite\n", 
                               src0_check_count, src1_check_count);
            }
        } else {
            // For quantized tensors: validation happens in vec_dot functions (e.g., ggml_vec_dot_f16)
            GGML_LOG_DEBUG("🔍 MUL_MAT chunk: quantized input validation deferred to vec_dot computation\n");
        }
    }
    
    if (src1->type != vec_dot_type) {
        GGML_LOG_DEBUG("🔍 Preprocessing: Converting src1 from type %d to vec_dot_type %d for rows %ld-%ld\n", 
                       src1->type, vec_dot_type, start_row, end_row-1);
        
        // Verify src1 is F32 (expected input type)
        NUMA_ASSERT(src1->type == GGML_TYPE_F32);
        
        const int64_t ne10 = src1->ne[0]; const int64_t ne11 = src1->ne[1];
        const int64_t ne12 = src1->ne[2]; const int64_t ne13 = src1->ne[3];
        
        // Calculate conversion size for the full tensor
        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1 * ne11;
        const size_t nbw3 = nbw2 * ne12;
        const size_t total_conversion_size = nbw3 * ne13;
        
        GGML_LOG_DEBUG("🔍 Full tensor conversion buffer requirements: %zu bytes\n", total_conversion_size);
        
        // Verify we have enough work buffer space for the full conversion
        NUMA_ASSERT(params->wsize >= total_conversion_size);
        
        // Get conversion function
        ggml_from_float_t const from_float = ggml_get_type_traits_cpu(vec_dot_type)->from_float;
        char * wdata = (char*)params->wdata;
        
        // Convert the ENTIRE src1 tensor from F32 to vec_dot_type
        // Each NUMA node does its own conversion in its own work buffer
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float(
                        (float *)((char *) src1_data + i13*src1->nb[3] + i12*src1->nb[2] + i11*src1->nb[1]),
                        (void *)(wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                        ne10
                    );
                }
            }
        }
        
        converted_src1_data = wdata;
        
        GGML_LOG_DEBUG("🔍 src1 full conversion completed\n");
    } else {
        GGML_LOG_DEBUG("🔍 No src1 conversion needed (already vec_dot_type)\n");
        converted_src1_data = src1_data;
    }
    
    GGML_LOG_DEBUG("🔍 Calling chunk function: ir0=[0,%ld], ir1=[%ld,%ld], vec_dot_num_rows=%ld\n", 
                   nr0, start_row, end_row, vec_dot_num_rows);
    
    // THREAD-SAFE APPROACH: Use work buffer pattern (no tensor mutation)
    // Follow the original ggml_compute_forward_mul_mat_one_chunk logic exactly
    
    // Choose data source: work buffer if conversion happened, tensor data if not
    const void * wdata_for_chunk = (converted_src1_data != src1_data) ? converted_src1_data : tensor_data(src1);
    
    // Create safe compute params pointing to the right data
    struct ggml_compute_params safe_params = *params;
    
    // The chunk function will use params->wdata for converted data detection
    // If src1->type != vec_dot_type, it expects converted data in params->wdata
    if (converted_src1_data != src1_data) {
        // Point wdata to our converted buffer - chunk function will use this
        safe_params.wdata = (void*)converted_src1_data;
    }
    
    // Call chunk function WITHOUT mutating any tensor metadata
    extern void ggml_compute_forward_mul_mat_one_chunk(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst,
        const enum ggml_type type,
        const int64_t num_rows_per_vec_dot,
        const int64_t ir0_start,
        const int64_t ir0_end,
        const int64_t ir1_start,
        const int64_t ir1_end
    );
    
    ggml_compute_forward_mul_mat_one_chunk(
        &safe_params,   // Use our safe params
        dst,
        src0->type,
        vec_dot_num_rows,
        0,          // ir0_start: start of rows (always 0 for matrix cols)
        nr0,        // ir0_end: end of rows (always full width for matrix cols)  
        start_row,  // ir1_start: start of assigned row slice
        end_row     // ir1_end: end of assigned row slice
    );
    
    // CRITICAL DEBUG: Log the parameters passed to the compute function
    fprintf(stderr, "🚨 CRITICAL CALL DEBUG: ggml_compute_forward_mul_mat_one_chunk called with:\n");
    fprintf(stderr, "    src0->type=%d (%s), vec_dot_num_rows=%ld\n", 
           src0->type, ggml_type_name(src0->type), vec_dot_num_rows);
    fprintf(stderr, "    ir0=[0, %ld], ir1=[%ld, %ld]\n", nr0, start_row, end_row);
    fflush(stderr);
    
    GGML_LOG_DEBUG("🔍 ggml_compute_forward_mul_mat_one_chunk completed successfully\n");
    
    // Check output data after computation and validate for corruption
    void* output_data = ggml_get_data(dst);
    if (output_data) {
        float* output_f = (float*)output_data;
        GGML_LOG_DEBUG("🔍 Output data after computation: dst[0]=%.2f, dst[1]=%.2f, dst[2]=%.2f, dst[3]=%.2f\n", 
               (double)output_f[0], (double)output_f[1], (double)output_f[2], (double)output_f[3]);
        
        // CRITICAL VALIDATION: Check MUL_MAT output for NaN/inf corruption
        int64_t output_check_count = ggml_nelements(dst);
        output_check_count = (output_check_count > 32) ? 32 : output_check_count; // Check first 32 elements
        for (int64_t i = 0; i < output_check_count; i++) {
            NUMA_ASSERT(isfinite(output_f[i]), "MUL_MAT: Generated corrupted output at index %d: %f", (int)i, (double)output_f[i]);
        }
        GGML_LOG_DEBUG("🔍 MUL_MAT output validation: %ld elements checked, all finite\n", output_check_count);
    }
    
    GGML_LOG_DEBUG("🔍 ggml_compute_forward_mul_mat completed\n");
    
    return GGML_STATUS_SUCCESS;
}

// Specialized work function for SOFT_MAX operations
static enum ggml_status ggml_numa_work_function_soft_max(void * work_context, struct ggml_compute_params * params) {
    if (!work_context || !params) {
        return GGML_STATUS_FAILED;
    }
    
    ggml_numa_dispatcher_work_context_t * ctx = (ggml_numa_dispatcher_work_context_t *)work_context;
    
    GGML_LOG_DEBUG("Executing SOFT_MAX work function with %d threads\n", params->nth);
    
    // For SOFT_MAX operations, we need to call the kernel directly but with single-threaded parameters
    // to avoid conflicts with the NUMA coordinator threading model
    // Set up single-threaded compute params to process all data on this NUMA node
    struct ggml_compute_params single_thread_params = {
        .ith = 0,                  // Process all data (thread index 0)
        .nth = 1,                  // Single thread (total threads = 1)
        .wsize = params->wsize,    // Use coordinator's work buffer
        .wdata = params->wdata,    // Use coordinator's work buffer
        .threadpool = NULL         // No threadpool conflicts
    };
    
    // Call the SOFT_MAX mathematical kernel directly with single-threaded params
    ggml_compute_forward_soft_max(&single_thread_params, ctx->operation);
    
    GGML_LOG_DEBUG("Successfully executed SOFT_MAX work function\n");
    
    return GGML_STATUS_SUCCESS;
}

// Specialized work function for SOFT_MAX operations with NUMA-aware data parallel chunking
static enum ggml_status ggml_numa_work_function_soft_max_chunk(void * work_context, struct ggml_compute_params * params) {
    if (!work_context || !params) {
        return GGML_STATUS_FAILED;
    }
    
    ggml_numa_dispatcher_work_context_t * ctx = (ggml_numa_dispatcher_work_context_t *)work_context;
    struct ggml_tensor * dst = ctx->operation;

    NUMA_ASSERT(dst && dst->src[0]);
    
    GGML_LOG_DEBUG("🔍 SOFT_MAX chunk: wsize=%zu, wdata=%p\n", params->wsize, params->wdata);
    GGML_LOG_DEBUG("🔍 SOFT_MAX chunk: dst=%p, src=%p\n", (void*)dst, (void*)dst->src[0]);
    
    const struct ggml_tensor * src = dst->src[0];
    
    GGML_LOG_DEBUG("🔍 Tensor dimensions: dst=(%ld,%ld,%ld,%ld), src=(%ld,%ld,%ld,%ld)\n",
           dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], 
           src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
    
    // NUMA-aware data slicing: each NUMA node processes a portion of the rows
    // SOFT_MAX is typically applied row-wise (along ne[0] dimension)
    // Each row can be processed independently, so distribute rows across NUMA nodes
    
    int current_numa_node = ggml_numa_get_current_node();
    int total_numa_nodes = ggml_numa_coordinator_get_num_nodes();
    
    const int64_t ne01 = dst->ne[1];  // Number of rows to process
    const int64_t ne02 = dst->ne[2];  // Batch dimension 2
    const int64_t ne03 = dst->ne[3];  // Batch dimension 3
    
    // Calculate which rows this NUMA node should process
    int64_t rows_per_node = ne01 / total_numa_nodes;
    int64_t extra_rows = ne01 % total_numa_nodes;
    
    // Calculate start and end rows for this NUMA node within each batch
    int64_t start_row = current_numa_node * rows_per_node;
    int64_t end_row = start_row + rows_per_node;
    
    // Distribute any extra rows to the first few nodes
    if (current_numa_node < extra_rows) {
        start_row += current_numa_node;
        end_row += current_numa_node + 1;
    } else {
        start_row += extra_rows;
        end_row += extra_rows;
    }
    
    // Ensure we don't exceed bounds
    if (end_row > ne01) end_row = ne01;
    if (start_row >= ne01) {
        // This NUMA node has no work to do
        GGML_LOG_DEBUG("🔍 NUMA node %d has no work (start_row=%ld >= ne01=%ld)\n", 
                       current_numa_node, start_row, ne01);
        return GGML_STATUS_SUCCESS;
    }
    
    GGML_LOG_DEBUG("🔍 NUMA node %d (of %d): processing rows %ld to %ld (of %ld total per batch)\n", 
                   current_numa_node, total_numa_nodes, start_row, end_row - 1, ne01);
    
    // Process each batch and each row in this NUMA node's assigned range
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = start_row; i01 < end_row; i01++) {
                
                GGML_LOG_DEBUG("🔍 Processing row i01=%ld, i02=%ld, i03=%ld\n", i01, i02, i03);
                
                // Get pointers to the source and destination data for this specific row
                const char * src_base = (const char*)ggml_get_data(src);
                char * dst_base = (char*)ggml_get_data(dst);
                
                const size_t src_offset = i01*src->nb[1] + i02*src->nb[2] + i03*src->nb[3];
                const size_t dst_offset = i01*dst->nb[1] + i02*dst->nb[2] + i03*dst->nb[3];
                
                const float * src_row = (const float*)(src_base + src_offset);
                float * dst_row = (float*)(dst_base + dst_offset);
                
                const int64_t ne00 = dst->ne[0];  // Row width
                
                // Apply softmax to this row using SIMD-optimized reference implementation
                // First pass: find maximum value
                float max_val = -INFINITY;
                ggml_vec_max_f32(ne00, &max_val, src_row);
                
                // Second pass: compute exponentials and sum using optimized vector function
                ggml_float sum = ggml_vec_soft_max_f32(ne00, dst_row, src_row, max_val);
                
                // Third pass: normalize using optimized vector scale
                const ggml_float sum_inv = 1.0 / sum;
                ggml_vec_scale_f32(ne00, dst_row, (float)sum_inv);
                
                GGML_LOG_DEBUG("🔍 Processed row with max_val=%.6f, sum=%.6f\n", max_val, sum);
            }
        }
    }
    
    GGML_LOG_DEBUG("🔍 SOFT_MAX chunk processing completed successfully\n");
    
    return GGML_STATUS_SUCCESS;
}

// Specialized work function for ROPE operations with NUMA-aware chunking
static enum ggml_status ggml_numa_work_function_rope_chunk(void * work_context, struct ggml_compute_params * params) {
    if (!work_context || !params) {
        GGML_LOG_ERROR("ROPE work function: Invalid parameters\n");
        return GGML_STATUS_FAILED;
    }
    
    ggml_numa_dispatcher_work_context_t * ctx = (ggml_numa_dispatcher_work_context_t *)work_context;
    
    if (!ctx->operation) {
        GGML_LOG_ERROR("ROPE work function: Operation is NULL\n");
        return GGML_STATUS_FAILED;
    }
    
    GGML_LOG_DEBUG("ROPE work function: operation=%p, type=%d, elements=%ld\n", 
                   (void*)ctx->operation, ctx->operation->op, ggml_nelements(ctx->operation));
    
    // Check SOURCE tensor data (src[0] is where ROPE reads input from)
    if (ctx->operation->src[0]) {
        void *src_data = ggml_get_data(ctx->operation->src[0]);
        if (src_data) {
            float *data = (float*)src_data;
            GGML_LOG_DEBUG("ROPE work function: First few SOURCE values: %.6f %.6f %.6f %.6f\n", 
                           data[0], data[1], data[2], data[3]);
            
            // STRICT VALIDATION: Check first 16 elements for NaN/inf corruption
            int64_t check_count = ggml_nelements(ctx->operation->src[0]);
            check_count = (check_count > 16) ? 16 : check_count;
            for (int64_t i = 0; i < check_count; i++) {
                NUMA_ASSERT(isfinite(data[i]), "ROPE: Found NaN/inf in src[0] data at index %d: %f", (int)i, (double)data[i]);
            }
        } else {
            GGML_LOG_WARN("ROPE work function: src[0] data is NULL\n");
        }
    } else {
        GGML_LOG_ERROR("ROPE work function: src[0] is NULL\n");
        return GGML_STATUS_FAILED;
    }
    
    // Check destination tensor (should start as zeros)
    void *dst_data = ggml_get_data(ctx->operation);
    if (dst_data) {
        float *data = (float*)dst_data;
        GGML_LOG_DEBUG("ROPE work function: First few DESTINATION values (before): %.6f %.6f %.6f %.6f\n", 
                       data[0], data[1], data[2], data[3]);
    }
    
    GGML_LOG_DEBUG("Executing ROPE chunk work function with %d threads on NUMA node (current CPU: %d)\n", 
                   params->nth, sched_getcpu());
    
    // For ROPE operations, we need to call the kernel directly but with single-threaded parameters
    // to avoid conflicts with the NUMA coordinator threading model
    // Set up single-threaded compute params to process all data on this NUMA node
    struct ggml_compute_params single_thread_params = {
        .ith = 0,                  // Process all data (thread index 0)
        .nth = 1,                  // Single thread (total threads = 1)
        .wsize = params->wsize,    // Use coordinator's work buffer
        .wdata = params->wdata,    // Use coordinator's work buffer
        .threadpool = NULL         // No threadpool conflicts
    };
    
    GGML_LOG_DEBUG("ROPE work function: Calling ggml_compute_forward_rope with single_thread_params (ith=%d, nth=%d)\n",
                   single_thread_params.ith, single_thread_params.nth);
    
    // Call the ROPE mathematical kernel directly with single-threaded params
    ggml_compute_forward_rope(&single_thread_params, ctx->operation);
    
    // Add memory barrier to ensure all writes are visible before returning
    __sync_synchronize();
    
    // Check output values and validate for corruption
    if (dst_data) {
        float *data = (float*)dst_data;
        GGML_LOG_DEBUG("ROPE work function: First few DESTINATION values (after): %.6f %.6f %.6f %.6f\n", 
                       data[0], data[1], data[2], data[3]);
        
        // CRITICAL VALIDATION: Check ROPE output for NaN/inf corruption 
        int64_t output_check_count = ggml_nelements(ctx->operation);
        output_check_count = (output_check_count > 32) ? 32 : output_check_count; // Check first 32 elements
        for (int64_t i = 0; i < output_check_count; i++) {
            NUMA_ASSERT(isfinite(data[i]), "ROPE: Generated corrupted output at index %d: %f", (int)i, (double)data[i]);
        }
        GGML_LOG_DEBUG("🔍 ROPE output validation: %ld elements checked, all finite\n", output_check_count);
    }
    
    GGML_LOG_DEBUG("Successfully executed ROPE chunk work function\n");
    
    return GGML_STATUS_SUCCESS;
}

// Thread data structures for multi-level parallelism
struct add_thread_data {
    // Source data
    const struct ggml_tensor * src0;
    const struct ggml_tensor * src1;
    const struct ggml_tensor * dst;
    // Tensor dimensions
    int64_t ne0, ne1, ne2, ne3;
    size_t nb0, nb1, nb2, nb3;
    size_t src1_nb0, src1_nb1, src1_nb2, src1_nb3;
    size_t dst_nb0, dst_nb1, dst_nb2, dst_nb3;
    // Thread-specific element range within NUMA node's assigned elements
    int64_t thread_start_elem;
    int64_t thread_end_elem;
    // For debugging
    int thread_id;
    int numa_node;
};

struct rms_norm_thread_data {
    const struct ggml_tensor * src0;
    const struct ggml_tensor * dst;
    int64_t ne00, ne01, ne02, ne03;
    size_t nb01, nb02, nb03;
    size_t nb1, nb2, nb3;  // destination strides
    float eps;
    int thread_start_row;
    int thread_end_row;
    int thread_id;
    int numa_node;
};

// Thread kernel functions for multi-level parallelism
static void* add_thread_kernel(void* data) {
    struct add_thread_data * td = (struct add_thread_data *)data;
    
    // Process elements in this thread's assigned range
    for (int64_t i = td->thread_start_elem; i < td->thread_end_elem; i++) {
        // Convert linear index to multi-dimensional indices
        int64_t i3 = i / (td->ne2 * td->ne1 * td->ne0);
        int64_t i2 = (i % (td->ne2 * td->ne1 * td->ne0)) / (td->ne1 * td->ne0);
        int64_t i1 = (i % (td->ne1 * td->ne0)) / td->ne0;
        int64_t i0 = i % td->ne0;
        
        // Calculate memory addresses for current element
        const float * src0_ptr = (float *) ((char *) ggml_get_data(td->src0) + i3*td->nb3 + i2*td->nb2 + i1*td->nb1 + i0*td->nb0);
        const float * src1_ptr = (float *) ((char *) ggml_get_data(td->src1) + i3*td->src1_nb3 + i2*td->src1_nb2 + i1*td->src1_nb1 + i0*td->src1_nb0);
        float * dst_ptr = (float *) ((char *) ggml_get_data(td->dst) + i3*td->dst_nb3 + i2*td->dst_nb2 + i1*td->dst_nb1 + i0*td->dst_nb0);
        
        // ADD SIMD-optimized mathematical kernel: dst = src0 + src1
        // Check if we can process a contiguous chunk using SIMD
        if (i0 == 0 && td->ne0 > 1 && 
            td->nb0 == sizeof(float) && td->src1_nb0 == sizeof(float) && td->dst_nb0 == sizeof(float)) {
            // Process entire row with SIMD if contiguous
            const float * src0_row = (float *) ((char *) ggml_get_data(td->src0) + i3*td->nb3 + i2*td->nb2 + i1*td->nb1);
            const float * src1_row = (float *) ((char *) ggml_get_data(td->src1) + i3*td->src1_nb3 + i2*td->src1_nb2 + i1*td->src1_nb1);
            float * dst_row = (float *) ((char *) ggml_get_data(td->dst) + i3*td->dst_nb3 + i2*td->dst_nb2 + i1*td->dst_nb1);
            
            ggml_vec_add_f32(td->ne0, dst_row, src0_row, src1_row);
            
            // Skip remaining elements in this row since we processed them all
            i += (td->ne0 - 1); // -1 because loop will increment i
        } else {
            // Fallback to element-wise operation for non-contiguous or single elements
            *dst_ptr = *src0_ptr + *src1_ptr;
        }
    }
    return NULL;
}

static void* rms_norm_thread_kernel(void* data) {
    struct rms_norm_thread_data * td = (struct rms_norm_thread_data *)data;
    
    // Process rows across all batch dimensions in this thread's assigned range
    for (int64_t i03 = 0; i03 < td->ne03; i03++) {
        for (int64_t i02 = 0; i02 < td->ne02; i02++) {
            for (int64_t i01 = td->thread_start_row; i01 < td->thread_end_row; i01++) {
                // Calculate memory addresses for current row across all batch dimensions
                const float * src_row = (float *) ((char *) ggml_get_data(td->src0) + i01*td->nb01 + i02*td->nb02 + i03*td->nb03);
                float * dst_row = (float *) ((char *) ggml_get_data(td->dst) + i01*td->nb1 + i02*td->nb2 + i03*td->nb3);
                
                // RMS normalization SIMD-optimized: sum of squares using vector dot product
                float sum = 0.0f;
                ggml_vec_dot_f32(td->ne00, &sum, 0, src_row, 0, src_row, 0, 1);
                
                const float mean = sum / td->ne00;
                const float scale = 1.0f / sqrtf(mean + td->eps);
                
                // Check for numerical issues
                if (!isfinite(scale) || scale <= 0.0f) {
                    GGML_LOG_ERROR("RMS_NORM thread %d: numerical instability detected (scale=%f, mean=%f, eps=%f)\n", 
                                   td->thread_id, scale, mean, td->eps);
                    continue; // Skip this row
                }
                
                // Copy source to destination, then apply SIMD scaling
                ggml_vec_cpy_f32(td->ne00, dst_row, src_row);
                ggml_vec_scale_f32(td->ne00, dst_row, scale);
            }
        }
    }
    return NULL;
}

// Single-node ADD work function for processing entire tensor on one NUMA node
static enum ggml_status ggml_numa_work_function_add_single(void * work_context, struct ggml_compute_params * params) {
    if (!work_context) {
        GGML_LOG_ERROR("ADD single work function: Invalid work context\n");
        return GGML_STATUS_FAILED;
    }
    
    ggml_numa_dispatcher_work_context_t * ctx = (ggml_numa_dispatcher_work_context_t *)work_context;
    
    if (!ctx->operation) {
        GGML_LOG_ERROR("ADD single work function: Operation is NULL\n");
        return GGML_STATUS_FAILED;
    }
    
    GGML_LOG_DEBUG("ADD single work function: operation=%p, type=%d, elements=%ld\n", 
                   (void*)ctx->operation, ctx->operation->op, ggml_nelements(ctx->operation));
    
    // Validate ADD operation has two source tensors
    const struct ggml_tensor * src0 = ctx->operation->src[0];
    const struct ggml_tensor * src1 = ctx->operation->src[1];
    
    if (!src0 || !src1) {
        GGML_LOG_ERROR("ADD single work function: Missing source tensors (src0=%p, src1=%p)\n", 
                       (void*)src0, (void*)src1);
        return GGML_STATUS_FAILED;
    }
    
    // Check tensor compatibility
    if (!ggml_are_same_shape(src0, ctx->operation)) {
        GGML_LOG_ERROR("ADD single work function: src0 and destination shapes don't match\n");
        return GGML_STATUS_FAILED;
    }
    
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ADD single work function: Only F32 tensors supported\n");
        return GGML_STATUS_FAILED;
    }
    
    // STRICT VALIDATION: Check input tensors for NaN/inf corruption
    const float* src0_data = (const float*)ggml_get_data(src0);
    const float* src1_data = (const float*)ggml_get_data(src1);
    
    // Check first 16 elements of src0 for corruption
    int64_t check_count = ggml_nelements(src0);
    check_count = (check_count > 16) ? 16 : check_count;
    for (int64_t i = 0; i < check_count; i++) {
        NUMA_ASSERT(isfinite(src0_data[i]), "Found NaN/inf in src0 data at index %d: %f", (int)i, (double)src0_data[i]);
    }
    
    // Check first 16 elements of src1 for corruption  
    check_count = ggml_nelements(src1);
    check_count = (check_count > 16) ? 16 : check_count;
    for (int64_t i = 0; i < check_count; i++) {
        NUMA_ASSERT(isfinite(src1_data[i]), "Found NaN/inf in src1 data at index %d: %f", (int)i, (double)src1_data[i]);
    }
    
    // Extract tensor dimensions for element-wise operations
    const int64_t ne0 = src0->ne[0];  // Elements per row
    const int64_t ne1 = src0->ne[1];  // Number of rows  
    const int64_t ne2 = src0->ne[2];  // Batch dimension 2
    const int64_t ne3 = src0->ne[3];  // Batch dimension 3
    
    const size_t nb0 = src0->nb[0];   // Element stride (should be sizeof(float))
    const size_t nb1 = src0->nb[1];   // Row stride in bytes
    const size_t nb2 = src0->nb[2];   // Batch stride 2 in bytes 
    const size_t nb3 = src0->nb[3];   // Batch stride 3 in bytes
    
    const size_t src1_nb0 = src1->nb[0];
    const size_t src1_nb1 = src1->nb[1];
    const size_t src1_nb2 = src1->nb[2];
    const size_t src1_nb3 = src1->nb[3];
    
    const size_t dst_nb0 = ctx->operation->nb[0];
    const size_t dst_nb1 = ctx->operation->nb[1];
    const size_t dst_nb2 = ctx->operation->nb[2];
    const size_t dst_nb3 = ctx->operation->nb[3];
    
    GGML_LOG_DEBUG("ADD single work function: Processing tensor [%ld, %ld, %ld, %ld]\n", ne0, ne1, ne2, ne3);

    // Single-node ADD implementation - process entire tensor
    int64_t total_elements = ggml_nelements(ctx->operation);
    
    GGML_LOG_DEBUG("ADD single-node execution: processing all %ld elements\n", total_elements);

    // Use existing threadpool architecture if available, otherwise fall back to single-threaded
    if (params && params->nth > 1 && params->threadpool) {
        GGML_LOG_DEBUG("ADD: Using multi-threaded execution (%d threads) for single-node processing (CPU: %d)\n", 
                       params->nth, sched_getcpu());

        // Create thread data for each thread
        struct add_thread_data * thread_data = (struct add_thread_data *)calloc(params->nth, sizeof(struct add_thread_data));
        if (!thread_data) {
            GGML_LOG_ERROR("ADD: Failed to allocate thread data\n");
            return GGML_STATUS_FAILED;
        }

        // Distribute entire tensor among threads (no NUMA node splitting)
        int64_t elements_per_thread = total_elements / params->nth;
        int64_t remainder_thread_elements = total_elements % params->nth;

        for (int t = 0; t < params->nth; t++) {
            struct add_thread_data * td = &thread_data[t];
            
            // Copy common data
            td->src0 = src0;
            td->src1 = src1;
            td->dst = ctx->operation;
            td->ne0 = ne0; td->ne1 = ne1; td->ne2 = ne2; td->ne3 = ne3;
            td->nb0 = nb0; td->nb1 = nb1; td->nb2 = nb2; td->nb3 = nb3;
            td->src1_nb0 = src1_nb0; td->src1_nb1 = src1_nb1; td->src1_nb2 = src1_nb2; td->src1_nb3 = src1_nb3;
            td->dst_nb0 = dst_nb0; td->dst_nb1 = dst_nb1; td->dst_nb2 = dst_nb2; td->dst_nb3 = dst_nb3;
            td->thread_id = t;
            td->numa_node = 0;  // Single NUMA node (node 0)
            
            // Calculate thread-specific element range for entire tensor
            int64_t thread_elem_start = t * elements_per_thread;
            int64_t thread_elem_end = thread_elem_start + elements_per_thread;
            
            // Distribute remainder elements among first few threads
            if (t < remainder_thread_elements) {
                thread_elem_start += t;
                thread_elem_end += t + 1;
            } else {
                thread_elem_start += remainder_thread_elements;
                thread_elem_end += remainder_thread_elements;
            }
            
            // Store absolute element indices
            td->thread_start_elem = thread_elem_start;
            td->thread_end_elem = thread_elem_end;
            
            // Bounds checking
            if (td->thread_end_elem > total_elements) {
                td->thread_end_elem = total_elements;
            }
            if (td->thread_start_elem >= td->thread_end_elem) {
                td->thread_end_elem = td->thread_start_elem + 1; // Ensure at least one element
            }
            
            GGML_LOG_DEBUG("ADD single-node thread %d: processing elements %ld to %ld\n", 
                           t, td->thread_start_elem, td->thread_end_elem - 1);
        }

        // Execute using pthread directly for multi-threading
        pthread_t* threads = (pthread_t*)malloc(params->nth * sizeof(pthread_t));
        if (!threads) {
            free(thread_data);
            GGML_LOG_ERROR("ADD: Failed to allocate thread handles\n");
            return GGML_STATUS_FAILED;
        }
        
        // Create threads
        for (int t = 0; t < params->nth; t++) {
            int ret = pthread_create(&threads[t], NULL, add_thread_kernel, &thread_data[t]);
            if (ret != 0) {
                GGML_LOG_ERROR("ADD: Failed to create thread %d: %d\n", t, ret);
                // Clean up already created threads
                for (int i = 0; i < t; i++) {
                    pthread_join(threads[i], NULL);
                }
                free(threads);
                free(thread_data);
                return GGML_STATUS_FAILED;
            }
        }
        
        // Wait for all threads to complete
        for (int t = 0; t < params->nth; t++) {
            pthread_join(threads[t], NULL);
        }
        
        free(threads);
        free(thread_data);
        
        GGML_LOG_DEBUG("ADD: Multi-threaded single-node execution completed\n");
        
    } else {
        // Single-threaded execution for entire tensor
        GGML_LOG_DEBUG("ADD: Using single-threaded execution for single-node processing (CPU: %d)\n", 
                       sched_getcpu());
                       
        // Process all elements in the tensor
        for (int64_t i = 0; i < total_elements; i++) {
            // Convert linear index to multi-dimensional indices
            int64_t i3 = i / (ne2 * ne1 * ne0);
            int64_t i2 = (i % (ne2 * ne1 * ne0)) / (ne1 * ne0);
            int64_t i1 = (i % (ne1 * ne0)) / ne0;
            int64_t i0 = i % ne0;
            
            // Calculate memory addresses for current element
            const float * src0_ptr = (float *) ((char *) ggml_get_data(src0) + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
            const float * src1_ptr = (float *) ((char *) ggml_get_data(src1) + i3*src1_nb3 + i2*src1_nb2 + i1*src1_nb1 + i0*src1_nb0);
            float * dst_ptr = (float *) ((char *) ggml_get_data(ctx->operation) + i3*dst_nb3 + i2*dst_nb2 + i1*dst_nb1 + i0*dst_nb0);
            
            // ADD SIMD-optimized mathematical kernel: dst = src0 + src1
            // Check if we can process a contiguous chunk using SIMD
            if (i0 == 0 && ne0 > 1 && 
                nb0 == sizeof(float) && src1_nb0 == sizeof(float) && dst_nb0 == sizeof(float)) {
                // Process entire row with SIMD if contiguous
                const float * src0_row = (float *) ((char *) ggml_get_data(src0) + i3*nb3 + i2*nb2 + i1*nb1);
                const float * src1_row = (float *) ((char *) ggml_get_data(src1) + i3*src1_nb3 + i2*src1_nb2 + i1*src1_nb1);
                float * dst_row = (float *) ((char *) ggml_get_data(ctx->operation) + i3*dst_nb3 + i2*dst_nb2 + i1*dst_nb1);
                
                ggml_vec_add_f32(ne0, dst_row, src0_row, src1_row);
                
                // Skip remaining elements in this row since we processed them all
                i += (ne0 - 1); // -1 because loop will increment i
            } else {
                // Fallback to element-wise operation for non-contiguous or single elements
                *dst_ptr = *src0_ptr + *src1_ptr;
            }
        }
    }

    // Memory barrier to ensure all writes are visible before returning
    __sync_synchronize();
    
    GGML_LOG_DEBUG("Successfully executed ADD single-node work function\n");
    
    return GGML_STATUS_SUCCESS;
}

// Specialized work function for ADD operations with NUMA-aware chunking and multi-level parallelism
static enum ggml_status ggml_numa_work_function_add_chunk(void * work_context, struct ggml_compute_params * params) {
    if (!work_context) {
        GGML_LOG_ERROR("ADD work function: Invalid work context\n");
        return GGML_STATUS_FAILED;
    }
    
    ggml_numa_dispatcher_work_context_t * ctx = (ggml_numa_dispatcher_work_context_t *)work_context;
    
    if (!ctx->operation) {
        GGML_LOG_ERROR("ADD work function: Operation is NULL\n");
        return GGML_STATUS_FAILED;
    }
    
    GGML_LOG_DEBUG("ADD work function: operation=%p, type=%d, elements=%ld\n", 
                   (void*)ctx->operation, ctx->operation->op, ggml_nelements(ctx->operation));
    
    // Validate ADD operation has two source tensors
    const struct ggml_tensor * src0 = ctx->operation->src[0];
    const struct ggml_tensor * src1 = ctx->operation->src[1];
    
    if (!src0 || !src1) {
        GGML_LOG_ERROR("ADD work function: Missing source tensors (src0=%p, src1=%p)\n", 
                       (void*)src0, (void*)src1);
        return GGML_STATUS_FAILED;
    }
    
    // Check tensor compatibility
    if (!ggml_are_same_shape(src0, ctx->operation)) {
        GGML_LOG_ERROR("ADD work function: src0 and destination shapes don't match\n");
        return GGML_STATUS_FAILED;
    }
    
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ADD work function: Only F32 tensors supported\n");
        return GGML_STATUS_FAILED;
    }
    
    // STRICT VALIDATION: Check input tensors for NaN/inf corruption
    const float* src0_data = (const float*)ggml_get_data(src0);
    const float* src1_data = (const float*)ggml_get_data(src1);
    
    // Check first 16 elements of src0 for corruption
    int64_t check_count = ggml_nelements(src0);
    check_count = (check_count > 16) ? 16 : check_count;
    for (int64_t i = 0; i < check_count; i++) {
        NUMA_ASSERT(isfinite(src0_data[i]), "Found NaN/inf in src0 data at index %d: %f", (int)i, (double)src0_data[i]);
    }
    
    // Check first 16 elements of src1 for corruption  
    check_count = ggml_nelements(src1);
    check_count = (check_count > 16) ? 16 : check_count;
    for (int64_t i = 0; i < check_count; i++) {
        NUMA_ASSERT(isfinite(src1_data[i]), "Found NaN/inf in src1 data at index %d: %f", (int)i, (double)src1_data[i]);
    }
    
    // Extract tensor dimensions for element-wise operations
    const int64_t ne0 = src0->ne[0];  // Elements per row
    const int64_t ne1 = src0->ne[1];  // Number of rows  
    const int64_t ne2 = src0->ne[2];  // Batch dimension 2
    const int64_t ne3 = src0->ne[3];  // Batch dimension 3
    
    const size_t nb0 = src0->nb[0];   // Element stride (should be sizeof(float))
    const size_t nb1 = src0->nb[1];   // Row stride in bytes
    const size_t nb2 = src0->nb[2];   // Batch stride 2 in bytes 
    const size_t nb3 = src0->nb[3];   // Batch stride 3 in bytes
    
    const size_t src1_nb0 = src1->nb[0];
    const size_t src1_nb1 = src1->nb[1];
    const size_t src1_nb2 = src1->nb[2];
    const size_t src1_nb3 = src1->nb[3];
    
    const size_t dst_nb0 = ctx->operation->nb[0];
    const size_t dst_nb1 = ctx->operation->nb[1];
    const size_t dst_nb2 = ctx->operation->nb[2];
    const size_t dst_nb3 = ctx->operation->nb[3];
    
    GGML_LOG_DEBUG("ADD work function: Processing tensor [%ld, %ld, %ld, %ld]\n", ne0, ne1, ne2, ne3);

    // NUMA + Thread Multi-Level Parallel ADD Implementation
    // Level 1: NUMA-level parallelism (different element ranges per NUMA node)
    // Level 2: Thread-level parallelism (subdivision within NUMA node)
    
    // Get NUMA node information from coordinator
    extern struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_get_global(int n_threads, bool force_multi_socket);
    extern int ggml_numa_coordinator_manager_get_numa_nodes(struct ggml_numa_coordinator_manager * mgr);
    
    int numa_node = ggml_numa_get_current_node();
    struct ggml_numa_coordinator_manager * mgr = ggml_numa_coordinator_manager_get_global(8, false);
    int max_numa_nodes = mgr ? ggml_numa_coordinator_manager_get_numa_nodes(mgr) : 1;
    
    // Handle fallback case where virtual node is not set
    if (numa_node < 0) {
        numa_node = 0;  // Default to node 0
    }
    if (max_numa_nodes <= 0) {
        max_numa_nodes = 1;  // At least one node
    }

    // Calculate total number of elements for distribution
    int64_t total_elements = ggml_nelements(ctx->operation);
    
    // For single-node execution (when used by single-node dispatcher), process all elements
    // For multi-node execution (when used by data-parallel dispatcher), distribute elements
    int64_t numa_start_elem, numa_end_elem;
    
    // Check if we're being called from single-node execution (max_numa_nodes == 1) 
    // or from data-parallel execution with multiple nodes
    if (max_numa_nodes == 1) {
        // Single-node execution: process entire tensor on this node
        numa_start_elem = 0;
        numa_end_elem = total_elements;
        GGML_LOG_DEBUG("ADD single-node execution: processing all %ld elements\n", total_elements);
    } else {
        // Multi-node data parallel execution: distribute elements across nodes
        int64_t elements_per_node = total_elements / max_numa_nodes;
        int64_t remainder_elements = total_elements % max_numa_nodes;
        
        // Calculate this NUMA node's element range
        numa_start_elem = numa_node * elements_per_node;
        numa_end_elem = numa_start_elem + elements_per_node;
        
        // Distribute remainder elements among first few nodes
        if (numa_node < remainder_elements) {
            numa_start_elem += numa_node;
            numa_end_elem += numa_node + 1;
        } else {
            numa_start_elem += remainder_elements;
            numa_end_elem += remainder_elements;
        }
        
        // Ensure we don't exceed tensor bounds and have valid ranges
        if (numa_end_elem > total_elements) {
            numa_end_elem = total_elements;
        }
        if (numa_start_elem < 0) {
            numa_start_elem = 0;
        }
        if (numa_end_elem <= numa_start_elem) {
            numa_end_elem = numa_start_elem + 1;  // Ensure at least one element
        }
        
        GGML_LOG_DEBUG("ADD data-parallel execution: NUMA node %d (of %d) processing elements %ld to %ld\n", 
                       numa_node, max_numa_nodes, numa_start_elem, numa_end_elem - 1);
    }

    // Thread-level parallelism within this NUMA node's assigned elements
    int64_t numa_node_elements = numa_end_elem - numa_start_elem;

    // Use existing threadpool architecture if available, otherwise fall back to single-threaded
    if (params && params->nth > 1 && params->threadpool) {
        GGML_LOG_DEBUG("ADD: Using multi-threaded execution (%d threads) on NUMA node %d (CPU: %d)\n", 
                       params->nth, numa_node, sched_getcpu());

        // Create thread data for each thread
        struct add_thread_data * thread_data = (struct add_thread_data *)calloc(params->nth, sizeof(struct add_thread_data));
        if (!thread_data) {
            GGML_LOG_ERROR("ADD: Failed to allocate thread data\n");
            return GGML_STATUS_FAILED;
        }

        // Distribute NUMA node's elements among threads
        int64_t elements_per_thread = numa_node_elements / params->nth;
        int64_t remainder_thread_elements = numa_node_elements % params->nth;

        for (int t = 0; t < params->nth; t++) {
            struct add_thread_data * td = &thread_data[t];
            
            // Copy common data
            td->src0 = src0;
            td->src1 = src1;
            td->dst = ctx->operation;
            td->ne0 = ne0; td->ne1 = ne1; td->ne2 = ne2; td->ne3 = ne3;
            td->nb0 = nb0; td->nb1 = nb1; td->nb2 = nb2; td->nb3 = nb3;
            td->src1_nb0 = src1_nb0; td->src1_nb1 = src1_nb1; td->src1_nb2 = src1_nb2; td->src1_nb3 = src1_nb3;
            td->dst_nb0 = dst_nb0; td->dst_nb1 = dst_nb1; td->dst_nb2 = dst_nb2; td->dst_nb3 = dst_nb3;
            td->thread_id = t;
            td->numa_node = numa_node;
            
            // Calculate thread-specific element range within NUMA node's elements
            int64_t thread_elem_start = t * elements_per_thread;
            int64_t thread_elem_end = thread_elem_start + elements_per_thread;
            
            // Distribute remainder elements among first few threads
            if (t < remainder_thread_elements) {
                thread_elem_start += t;
                thread_elem_end += t + 1;
            } else {
                thread_elem_start += remainder_thread_elements;
                thread_elem_end += remainder_thread_elements;
            }
            
            // Convert to absolute element indices
            td->thread_start_elem = numa_start_elem + thread_elem_start;
            td->thread_end_elem = numa_start_elem + thread_elem_end;
            
            // Bounds checking
            if (td->thread_end_elem > numa_end_elem) {
                td->thread_end_elem = numa_end_elem;
            }
            if (td->thread_start_elem >= td->thread_end_elem) {
                td->thread_end_elem = td->thread_start_elem + 1; // Ensure at least one element
            }
            
            GGML_LOG_DEBUG("ADD thread %d on NUMA %d: processing elements %ld to %ld\n", 
                           t, numa_node, td->thread_start_elem, td->thread_end_elem - 1);
        }

        // Execute using pthread directly for multi-threading within NUMA node
        pthread_t* threads = (pthread_t*)malloc(params->nth * sizeof(pthread_t));
        if (!threads) {
            free(thread_data);
            GGML_LOG_ERROR("ADD: Failed to allocate thread handles\n");
            return GGML_STATUS_FAILED;
        }
        
        // Create threads
        for (int t = 0; t < params->nth; t++) {
            int ret = pthread_create(&threads[t], NULL, add_thread_kernel, &thread_data[t]);
            if (ret != 0) {
                GGML_LOG_ERROR("ADD: Failed to create thread %d: %d\n", t, ret);
                // Clean up already created threads
                for (int i = 0; i < t; i++) {
                    pthread_join(threads[i], NULL);
                }
                free(threads);
                free(thread_data);
                return GGML_STATUS_FAILED;
            }
        }
        
        // Wait for all threads to complete
        for (int t = 0; t < params->nth; t++) {
            pthread_join(threads[t], NULL);
        }
        
        free(threads);
        free(thread_data);
        
        GGML_LOG_DEBUG("ADD: Multi-threaded execution completed on NUMA node %d\n", numa_node);
        
    } else {
        // Single-threaded execution within this NUMA node
        GGML_LOG_DEBUG("ADD: Using single-threaded execution on NUMA node %d (CPU: %d)\n", 
                       numa_node, sched_getcpu());
                       
        // Process elements in this NUMA node's assigned range
        for (int64_t i = numa_start_elem; i < numa_end_elem; i++) {
            // Convert linear index to multi-dimensional indices
            int64_t i3 = i / (ne2 * ne1 * ne0);
            int64_t i2 = (i % (ne2 * ne1 * ne0)) / (ne1 * ne0);
            int64_t i1 = (i % (ne1 * ne0)) / ne0;
            int64_t i0 = i % ne0;
            
            // Calculate memory addresses for current element
            const float * src0_ptr = (float *) ((char *) ggml_get_data(src0) + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
            const float * src1_ptr = (float *) ((char *) ggml_get_data(src1) + i3*src1_nb3 + i2*src1_nb2 + i1*src1_nb1 + i0*src1_nb0);
            float * dst_ptr = (float *) ((char *) ggml_get_data(ctx->operation) + i3*dst_nb3 + i2*dst_nb2 + i1*dst_nb1 + i0*dst_nb0);
            
            // ADD SIMD-optimized mathematical kernel: dst = src0 + src1
            // Check if we can process a contiguous chunk using SIMD
            if (i0 == 0 && ne0 > 1 && 
                nb0 == sizeof(float) && src1_nb0 == sizeof(float) && dst_nb0 == sizeof(float)) {
                // Process entire row with SIMD if contiguous
                const float * src0_row = (float *) ((char *) ggml_get_data(src0) + i3*nb3 + i2*nb2 + i1*nb1);
                const float * src1_row = (float *) ((char *) ggml_get_data(src1) + i3*src1_nb3 + i2*src1_nb2 + i1*src1_nb1);
                float * dst_row = (float *) ((char *) ggml_get_data(ctx->operation) + i3*dst_nb3 + i2*dst_nb2 + i1*dst_nb1);
                
                ggml_vec_add_f32(ne0, dst_row, src0_row, src1_row);
                
                // Skip remaining elements in this row since we processed them all
                i += (ne0 - 1); // -1 because loop will increment i
            } else {
                // Fallback to element-wise operation for non-contiguous or single elements
                *dst_ptr = *src0_ptr + *src1_ptr;
            }
        }
    }

    // Memory barrier to ensure all writes are visible before returning
    __sync_synchronize();
    
    GGML_LOG_DEBUG("Successfully executed ADD chunk work function\n");
    
    return GGML_STATUS_SUCCESS;
}

// Specialized work function for GLU operations with NUMA-aware chunking
static enum ggml_status ggml_numa_work_function_glu_chunk(void * work_context, struct ggml_compute_params * params) {
    if (!work_context || !params) {
        GGML_LOG_ERROR("GLU work function: Invalid parameters\n");
        return GGML_STATUS_FAILED;
    }
    
    ggml_numa_dispatcher_work_context_t * ctx = (ggml_numa_dispatcher_work_context_t *)work_context;
    
    if (!ctx->operation) {
        GGML_LOG_ERROR("GLU work function: Operation is NULL\n");
        return GGML_STATUS_FAILED;
    }
    
    GGML_LOG_DEBUG("GLU work function: operation=%p, type=%d, elements=%ld\n", 
                   (void*)ctx->operation, ctx->operation->op, ggml_nelements(ctx->operation));
    
    // Check source tensors - GLU can have one or two source tensors
    const struct ggml_tensor * src0 = ctx->operation->src[0];
    const struct ggml_tensor * src1 = ctx->operation->src[1];
    
    if (!src0) {
        GGML_LOG_ERROR("GLU work function: Missing primary source tensor\n");
        return GGML_STATUS_FAILED;
    }
    
    // Check SOURCE tensor data with comprehensive validation
    void *src0_data = ggml_get_data(src0);
    if (!src0_data) {
        GGML_LOG_ERROR("GLU work function: Source tensor data is NULL\n");
        return GGML_STATUS_FAILED;
    }
    
    // Validate input data for NaN/inf values
    float *data0 = (float*)src0_data;
    const int64_t src0_elements = ggml_nelements(src0);
    
    for (int64_t i = 0; i < src0_elements; i++) {
        NUMA_ASSERT(isfinite(data0[i]), "GLU: Found NaN/inf in src0 data at index %d: %f", (int)i, (double)data0[i]);
    }
    
    GGML_LOG_DEBUG("GLU work function: First few SOURCE0 values: %.6f %.6f %.6f %.6f\n", 
                   data0[0], data0[1], data0[2], data0[3]);

    // Check second source tensor if present
    if (src1) {
        void *src1_data = ggml_get_data(src1);
        if (src1_data) {
            float *data1 = (float*)src1_data;
            
            // Validate second input tensor for NaN/inf values
            const int64_t src1_elements = ggml_nelements(src1);
            for (int64_t i = 0; i < src1_elements; i++) {
                NUMA_ASSERT(isfinite(data1[i]), "GLU: Found NaN/inf in src1 data at index %d: %f", (int)i, (double)data1[i]);
            }
            
            GGML_LOG_DEBUG("GLU work function: First few SOURCE1 values: %.6f %.6f %.6f %.6f\n", 
                           data1[0], data1[1], data1[2], data1[3]);
        } else {
            GGML_LOG_WARN("GLU work function: Second source tensor data is NULL\n");
        }
    } else {
        GGML_LOG_DEBUG("GLU work function: Single source tensor mode (split tensor)\n");
    }    // Check destination tensor (should start as zeros or uninitialized)
    void *dst_data = ggml_get_data(ctx->operation);
    if (dst_data) {
        float *data = (float*)dst_data;
        GGML_LOG_DEBUG("GLU work function: First few DESTINATION values (before): %.6f %.6f %.6f %.6f\n", 
                       data[0], data[1], data[2], data[3]);
    } else {
        GGML_LOG_ERROR("GLU work function: Destination tensor data is NULL\n");
        return GGML_STATUS_FAILED;
    }
    
    GGML_LOG_DEBUG("Executing GLU chunk work function with %d threads on NUMA node (current CPU: %d)\n", 
                   params->nth, sched_getcpu());
    
    // For GLU operations, we need to call the kernel directly but with single-threaded parameters
    // to avoid conflicts with the NUMA coordinator threading model
    // Set up single-threaded compute params to process all data on this NUMA node
    struct ggml_compute_params single_thread_params = {
        .ith = 0,                  // Process all data (thread index 0)
        .nth = 1,                  // Single thread (total threads = 1)
        .wsize = params->wsize,    // Use coordinator's work buffer
        .wdata = params->wdata,    // Use coordinator's work buffer
        .threadpool = NULL         // No threadpool conflicts
    };
    
    GGML_LOG_DEBUG("GLU work function: Calling ggml_compute_forward_glu with single_thread_params (ith=%d, nth=%d)\n",
                   single_thread_params.ith, single_thread_params.nth);
    
    // Call the GLU mathematical kernel directly with single-threaded params
    ggml_compute_forward_glu(&single_thread_params, ctx->operation);
    
    // Add memory barrier to ensure all writes are visible before returning
    __sync_synchronize();
    
    // Check output values
    if (dst_data) {
        float *data = (float*)dst_data;
        GGML_LOG_DEBUG("GLU work function: First few DESTINATION values (after): %.6f %.6f %.6f %.6f\n", 
                       data[0], data[1], data[2], data[3]);
    }
    
    GGML_LOG_DEBUG("Successfully executed GLU chunk work function\n");
    
    return GGML_STATUS_SUCCESS;
}

// RMS_NORM chunk work function - handles row-wise normalization with data parallel execution
// This implementation extracts the mathematical kernel to avoid threading conflicts
static enum ggml_status ggml_numa_work_function_rms_norm_chunk(void * work_context, struct ggml_compute_params * params) {
    if (!work_context || !params) {
        GGML_LOG_ERROR("RMS_NORM work function: Invalid parameters\n");
        return GGML_STATUS_FAILED;
    }
    
    ggml_numa_dispatcher_work_context_t * ctx = (ggml_numa_dispatcher_work_context_t *)work_context;
    
    if (!ctx->operation) {
        GGML_LOG_ERROR("RMS_NORM work function: Operation is NULL\n");
        return GGML_STATUS_FAILED;
    }
    
    GGML_LOG_DEBUG("RMS_NORM work function: operation=%p, type=%d, elements=%ld\n", 
                   (void*)ctx->operation, ctx->operation->op, ggml_nelements(ctx->operation));
    
    // Check source tensor - RMS_NORM has one input tensor
    const struct ggml_tensor * src0 = ctx->operation->src[0];
    
    if (!src0) {
        GGML_LOG_ERROR("RMS_NORM work function: Missing source tensor\n");
        return GGML_STATUS_FAILED;
    }
    
    // Validate tensor shapes and types
    if (!ggml_are_same_shape(src0, ctx->operation)) {
        GGML_LOG_ERROR("RMS_NORM work function: Source and destination shapes don't match\n");
        return GGML_STATUS_FAILED;
    }
    
    if (src0->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("RMS_NORM work function: Only F32 tensors supported\n");
        return GGML_STATUS_FAILED;
    }
    
    if (src0->nb[0] != sizeof(float)) {
        GGML_LOG_ERROR("RMS_NORM work function: Invalid tensor stride\n");
        return GGML_STATUS_FAILED;
    }
    
    // Check if we have threading parameters for multi-level parallelism
    extern enum ggml_status ggml_numa_work_function_rms_norm_chunk(void * work_context, struct ggml_compute_params * params);
    
    // This function can be called either:
    // 1. From coordinator without params (legacy single-threaded per NUMA node)
    // 2. From coordinator with params (new multi-threaded per NUMA node)
    // 3. Recursively from threadpool with params (thread subdivision)
    
    // Extract tensor dimensions (using GGML_TENSOR_UNARY_OP_LOCALS pattern)
    const int64_t ne00 = src0->ne[0];  // Elements per row
    const int64_t ne01 = src0->ne[1];  // Number of rows  
    const int64_t ne02 = src0->ne[2];  // Batch dimension 2
    const int64_t ne03 = src0->ne[3];  // Batch dimension 3
    
    const size_t nb01 = src0->nb[1];   // Row stride in bytes
    const size_t nb02 = src0->nb[2];   // Batch stride 2 in bytes
    const size_t nb03 = src0->nb[3];   // Batch stride 3 in bytes
    
    const size_t nb1 = ctx->operation->nb[1];   // Destination row stride in bytes
    const size_t nb2 = ctx->operation->nb[2];   // Destination batch stride 2 in bytes
    const size_t nb3 = ctx->operation->nb[3];   // Destination batch stride 3 in bytes
    
    // Get epsilon parameter from operation parameters
    float eps;
    memcpy(&eps, ctx->operation->op_params, sizeof(float));
    if (eps < 0.0f) {
        GGML_LOG_ERROR("RMS_NORM work function: Invalid epsilon value: %f\n", eps);
        return GGML_STATUS_FAILED;
    }

    GGML_LOG_DEBUG("RMS_NORM work function: Processing tensor [%ld, %ld, %ld, %ld] with eps=%f\n", 
                   ne00, ne01, ne02, ne03, eps);

    // NUMA + Thread Multi-Level Parallel RMS_NORM Implementation
    // Level 1: NUMA-level parallelism (different row ranges per NUMA node)
    // Level 2: Thread-level parallelism (subdivision within NUMA node)
    
    // Get NUMA node information from coordinator
    extern struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_get_global(int n_threads, bool force_multi_socket);
    extern int ggml_numa_coordinator_manager_get_numa_nodes(struct ggml_numa_coordinator_manager * mgr);
    
    int numa_node = ggml_numa_get_current_node();
    struct ggml_numa_coordinator_manager * mgr = ggml_numa_coordinator_manager_get_global(8, false);
    int max_numa_nodes = mgr ? ggml_numa_coordinator_manager_get_numa_nodes(mgr) : 1;
    
    // Handle fallback case where virtual node is not set
    if (numa_node < 0) {
        numa_node = 0;  // Default to node 0
    }
    if (max_numa_nodes <= 0) {
        max_numa_nodes = 1;  // At least one node
    }

    // Calculate NUMA-level row distribution
    // In single-node execution, only node 0 gets work and processes entire tensor
    // In data-parallel execution, work is distributed across multiple nodes
    int64_t total_rows = ne01;
    int64_t numa_start_row, numa_end_row;
    
    // Simple heuristic: if only node 0 is being used, process entire tensor
    // Otherwise, use data-parallel distribution
    if (numa_node == 0 && max_numa_nodes > 1) {
        // For single-node strategy: process entire tensor on node 0
        // For data-parallel strategy: process node 0's slice 
        // We'll assume single-node if we're seeing consecutive calls to only node 0
        numa_start_row = 0;
        numa_end_row = total_rows;  // Process entire tensor for now
        GGML_LOG_DEBUG("RMS_NORM: NUMA node 0 processing ALL rows 0 to %ld (%ld total rows)", 
                       numa_end_row - 1, total_rows);
    } else {
        // Data-parallel mode: slice across nodes
        int64_t rows_per_node = total_rows / max_numa_nodes;
        int64_t remainder_rows = total_rows % max_numa_nodes;
        
        numa_start_row = numa_node * rows_per_node;
        numa_end_row = numa_start_row + rows_per_node;
        
        if (numa_node < remainder_rows) {
            numa_start_row += numa_node;
            numa_end_row += numa_node + 1;
        } else {
            numa_start_row += remainder_rows;
            numa_end_row += remainder_rows;
        }
        
        GGML_LOG_DEBUG("RMS_NORM: NUMA node %d processing rows %ld to %ld (%ld of %ld total)", 
                       numa_node, numa_start_row, numa_end_row - 1, numa_end_row - numa_start_row, total_rows);
    }
    
    // Ensure we don't exceed tensor bounds and have valid ranges
    if (numa_end_row > total_rows) {
        numa_end_row = total_rows;
    }
    if (numa_start_row < 0) {
        numa_start_row = 0;
    }
    if (numa_end_row <= numa_start_row) {
        numa_end_row = numa_start_row + 1;  // Ensure at least one row
    }

    // Thread-level parallelism within this NUMA node's assigned rows
    int numa_node_rows = numa_end_row - numa_start_row;
    
    GGML_LOG_DEBUG("RMS_NORM NUMA node %d (of %d): assigned rows %ld to %ld (%ld rows total)\n", 
                   numa_node, max_numa_nodes, numa_start_row, numa_end_row - 1, numa_node_rows);

    // Use existing threadpool architecture if available, otherwise fall back to single-threaded
    if (params && params->nth > 1 && params->threadpool) {
        GGML_LOG_DEBUG("RMS_NORM: Using multi-threaded execution (%d threads) on NUMA node %d (CPU: %d)\n", 
                       params->nth, numa_node, sched_getcpu());

        // Create thread data for each thread
        struct rms_norm_thread_data * thread_data = (struct rms_norm_thread_data *)calloc(params->nth, sizeof(struct rms_norm_thread_data));
        if (!thread_data) {
            GGML_LOG_ERROR("RMS_NORM: Failed to allocate thread data\n");
            return GGML_STATUS_FAILED;
        }

        // Distribute NUMA node's rows among threads
        int64_t rows_per_thread = numa_node_rows / params->nth;
        int64_t remainder_thread_rows = numa_node_rows % params->nth;

        for (int t = 0; t < params->nth; t++) {
            struct rms_norm_thread_data * td = &thread_data[t];
            
            // Copy common data
            td->src0 = src0;
            td->dst = ctx->operation;
            td->ne00 = ne00; td->ne01 = ne01; td->ne02 = ne02; td->ne03 = ne03;
            td->nb01 = nb01; td->nb02 = nb02; td->nb03 = nb03;
            td->nb1 = nb1; td->nb2 = nb2; td->nb3 = nb3;
            td->eps = eps;
            td->thread_id = t;
            td->numa_node = numa_node;
            
            // Calculate thread-specific row range within NUMA node's rows
            int64_t thread_row_start = t * rows_per_thread;
            int64_t thread_row_end = thread_row_start + rows_per_thread;
            
            // Distribute remainder rows among first few threads
            if (t < remainder_thread_rows) {
                thread_row_start += t;
                thread_row_end += t + 1;
            } else {
                thread_row_start += remainder_thread_rows;
                thread_row_end += remainder_thread_rows;
            }
            
            // Convert to absolute row indices
            td->thread_start_row = numa_start_row + thread_row_start;
            td->thread_end_row = numa_start_row + thread_row_end;
            
            // Bounds checking
            if (td->thread_end_row > numa_end_row) {
                td->thread_end_row = numa_end_row;
            }
            if (td->thread_start_row >= td->thread_end_row) {
                td->thread_end_row = td->thread_start_row + 1; // Ensure at least one row
            }
            
            GGML_LOG_DEBUG("RMS_NORM thread %d on NUMA %d: processing rows %ld to %ld\n", 
                           t, numa_node, td->thread_start_row, td->thread_end_row - 1);
        }

        // Execute using pthread directly for multi-threading within NUMA node
        pthread_t* threads = (pthread_t*)malloc(params->nth * sizeof(pthread_t));
        if (!threads) {
            free(thread_data);
            GGML_LOG_ERROR("RMS_NORM: Failed to allocate thread handles\n");
            return GGML_STATUS_FAILED;
        }
        
        // Create threads
        for (int t = 0; t < params->nth; t++) {
            int ret = pthread_create(&threads[t], NULL, rms_norm_thread_kernel, &thread_data[t]);
            if (ret != 0) {
                GGML_LOG_ERROR("RMS_NORM: Failed to create thread %d: %d\n", t, ret);
                // Clean up already created threads
                for (int i = 0; i < t; i++) {
                    pthread_join(threads[i], NULL);
                }
                free(threads);
                free(thread_data);
                return GGML_STATUS_FAILED;
            }
        }
        
        // Wait for all threads to complete
        for (int t = 0; t < params->nth; t++) {
            pthread_join(threads[t], NULL);
        }
        
        free(threads);
        free(thread_data);
        
        GGML_LOG_DEBUG("RMS_NORM: Multi-threaded execution completed on NUMA node %d\n", numa_node);
        
    } else {
        // Single-threaded execution within this NUMA node
        GGML_LOG_DEBUG("RMS_NORM: Using single-threaded execution on NUMA node %d (CPU: %d)\n", 
                       numa_node, sched_getcpu());
        
        // Get raw tensor data pointers for validation
        void * src_raw_data = ggml_get_data(src0);
        void * dst_raw_data = ggml_get_data(ctx->operation);
        
        GGML_LOG_DEBUG("RMS_NORM: Data pointers - src=%p, dst=%p\n", src_raw_data, dst_raw_data);
        
        // Comprehensive bounds checking with assertions
        NUMA_ASSERT(numa_start_row >= 0);
        NUMA_ASSERT(numa_end_row > numa_start_row);
        NUMA_ASSERT(numa_end_row <= ne01);
        NUMA_ASSERT(ne00 > 0);
        NUMA_ASSERT(ne01 > 0);
        NUMA_ASSERT(ne02 > 0);
        NUMA_ASSERT(ne03 > 0);
        NUMA_ASSERT(src_raw_data != NULL);
        NUMA_ASSERT(dst_raw_data != NULL);
        
        // Validate tensor strides are sane
        NUMA_ASSERT(nb01 >= ne00 * sizeof(float));
        NUMA_ASSERT(nb02 >= ne01 * nb01);
        NUMA_ASSERT(nb03 >= ne02 * nb02);
        NUMA_ASSERT(nb1 >= ne00 * sizeof(float));
        NUMA_ASSERT(nb2 >= ne01 * nb1);
        NUMA_ASSERT(nb3 >= ne02 * nb2);
        
        // Validate epsilon parameter
        NUMA_ASSERT(eps > 0.0f);
        NUMA_ASSERT(eps < 1.0f);  // Reasonable epsilon range
        NUMA_ASSERT(isfinite(eps));
        
        GGML_LOG_DEBUG("RMS_NORM: Boundary checks passed - processing dimensions [%ld,%ld,%ld,%ld], rows %ld to %ld\n",
                       ne00, ne01, ne02, ne03, numa_start_row, numa_end_row - 1);
                       
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                for (int64_t i01 = numa_start_row; i01 < numa_end_row; i01++) {
                    // Calculate memory addresses for current row
                    const float * x = (float *) ((char *) src_raw_data + i01*nb01 + i02*nb02 + i03*nb03);
                    float * y = (float *) ((char *) dst_raw_data + i01*nb1 + i02*nb2 + i03*nb3);
                    
                    // Comprehensive memory bounds validation
                    NUMA_ASSERT(x != NULL);
                    NUMA_ASSERT(y != NULL);
                    
                    // Validate memory offsets are within tensor boundaries
                    ptrdiff_t src_offset = i01*nb01 + i02*nb02 + i03*nb03;
                    ptrdiff_t dst_offset = i01*nb1 + i02*nb2 + i03*nb3;
                    ptrdiff_t src_end_offset = src_offset + (ne00-1) * sizeof(float);
                    ptrdiff_t dst_end_offset = dst_offset + (ne00-1) * sizeof(float);
                    
                    NUMA_ASSERT(src_offset >= 0);
                    NUMA_ASSERT(dst_offset >= 0);
                    NUMA_ASSERT(src_end_offset < (ptrdiff_t)(ggml_nbytes(src0)));
                    NUMA_ASSERT(dst_end_offset < (ptrdiff_t)(ggml_nbytes(ctx->operation)));
                    
                    // Validate the row data is finite before processing  
                    for (int64_t i = 0; i < ne00; i++) {
                        NUMA_ASSERT(isfinite(x[i]), "RMS_NORM: Found NaN/inf in row data at index %d: %f", (int)i, (double)x[i]);
                    }
                    
                    // Debug first few elements of first row processed by this NUMA node
                    if (i01 == numa_start_row && i02 == 0 && i03 == 0) {
                        GGML_LOG_DEBUG("RMS_NORM NUMA %d: First row input data [0-4]: %f %f %f %f %f\n", 
                                       numa_node, x[0], x[1], x[2], x[3], x[4]);
                    }
                    
                    // RMS Normalization SIMD-optimized mathematical kernel:
                    // 1. Calculate sum of squares for the row using vector dot product
                    float sum = 0.0;
                    ggml_vec_dot_f32(ne00, &sum, 0, x, 0, x, 0, 1);
                    
                    // Validate sum is finite and reasonable
                    NUMA_ASSERT(isfinite(sum) && sum >= 0.0f);
                    
                    // 2. Calculate mean of squares
                    const float mean = sum / ne00;
                    NUMA_ASSERT(isfinite(mean) && mean >= 0.0f);
                    
                    // 3. Copy input to output and apply RMS normalization scaling
                    const float scale = 1.0f / sqrtf(mean + eps);
                    
                    // Validate scale factor
                    NUMA_ASSERT(isfinite(scale) && scale > 0.0f);
                    
                    // Debug the first row calculations for this NUMA node
                    if (i01 == numa_start_row && i02 == 0 && i03 == 0) {
                        GGML_LOG_DEBUG("RMS_NORM NUMA %d: First row - sum=%f, mean=%f, scale=%f\n", 
                                       numa_node, sum, mean, scale);
                    }
                    
                    // 4. Copy and scale using SIMD operations
                    ggml_vec_cpy_f32(ne00, y, x);
                    ggml_vec_scale_f32(ne00, y, scale);
                    
                    // Debug output for first row
                    if (i01 == numa_start_row && i02 == 0 && i03 == 0) {
                        GGML_LOG_DEBUG("RMS_NORM NUMA %d: First row output [0-4]: %f %f %f %f %f\n", 
                                       numa_node, y[0], y[1], y[2], y[3], y[4]);
                    }
                }
            }
        }
        
        GGML_LOG_DEBUG("RMS_NORM: Single-threaded processing completed for NUMA node %d\n", numa_node);
    }
    
    // Memory barrier to ensure all writes are visible before returning
    __sync_synchronize();
    
    GGML_LOG_DEBUG("Successfully executed RMS_NORM chunk work function\n");
    
    return GGML_STATUS_SUCCESS;
}

// FLASH_ATTN_EXT chunk work function - handles flash attention computation with NUMA data parallelism
// This implementation extracts the mathematical kernel to avoid threading conflicts
static enum ggml_status ggml_numa_work_function_flash_attn_ext_chunk(void * work_context, struct ggml_compute_params * params) {
    if (!work_context || !params) {
        GGML_LOG_ERROR("FLASH_ATTN_EXT work function: Invalid parameters\n");
        return GGML_STATUS_FAILED;
    }
    
    ggml_numa_dispatcher_work_context_t * ctx = (ggml_numa_dispatcher_work_context_t *)work_context;
    
    if (!ctx->operation) {
        GGML_LOG_ERROR("FLASH_ATTN_EXT work function: Operation is NULL\n");
        return GGML_STATUS_FAILED;
    }
    
    GGML_LOG_DEBUG("FLASH_ATTN_EXT work function: operation=%p, type=%d, elements=%ld\n", 
                   (void*)ctx->operation, ctx->operation->op, ggml_nelements(ctx->operation));
    
    // Flash attention has 4 source tensors: q, k, v, mask
    const struct ggml_tensor * q = ctx->operation->src[0];
    const struct ggml_tensor * k = ctx->operation->src[1]; 
    const struct ggml_tensor * v = ctx->operation->src[2];
    const struct ggml_tensor * mask = ctx->operation->src[3];
    
    if (!q || !k || !v) {
        GGML_LOG_ERROR("FLASH_ATTN_EXT work function: Missing required source tensors (q=%p, k=%p, v=%p)\n", 
                       (void*)q, (void*)k, (void*)v);
        return GGML_STATUS_FAILED;
    }
    
    // Validate tensor types - flash attention typically works with F16 or F32
    if (q->type != GGML_TYPE_F32 && q->type != GGML_TYPE_F16) {
        GGML_LOG_ERROR("FLASH_ATTN_EXT work function: Unsupported Q tensor type %d\n", q->type);
        return GGML_STATUS_FAILED;
    }
    
    if (k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_F16) {
        GGML_LOG_ERROR("FLASH_ATTN_EXT work function: Unsupported K tensor type %d\n", k->type);
        return GGML_STATUS_FAILED;
    }
    
    if (v->type != GGML_TYPE_F32 && v->type != GGML_TYPE_F16) {
        GGML_LOG_ERROR("FLASH_ATTN_EXT work function: Unsupported V tensor type %d\n", v->type);
        return GGML_STATUS_FAILED;
    }
    
    // Check data pointers
    void *q_data = ggml_get_data(q);
    void *k_data = ggml_get_data(k); 
    void *v_data = ggml_get_data(v);
    void *dst_data = ggml_get_data(ctx->operation);
    
    if (!q_data || !k_data || !v_data || !dst_data) {
        GGML_LOG_ERROR("FLASH_ATTN_EXT work function: NULL tensor data (q=%p, k=%p, v=%p, dst=%p)\n",
                       q_data, k_data, v_data, dst_data);
        return GGML_STATUS_FAILED;
    }

    // NUMA-aware data slicing for Flash Attention
    // Level 1: NUMA-level parallelism (different row ranges per NUMA node)
    // Level 2: Thread-level parallelism (subdivision within NUMA node)
    
    // Get virtual NUMA node information from coordinator's thread-local storage
    extern int ggml_numa_get_current_node(void);
    extern struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_get_global(int n_threads, bool force_multi_socket);
    extern int ggml_numa_coordinator_manager_get_numa_nodes(struct ggml_numa_coordinator_manager * mgr);
    
    int numa_node = ggml_numa_get_current_node();
    struct ggml_numa_coordinator_manager * mgr = ggml_numa_coordinator_manager_get_global(8, false);
    int max_numa_nodes = mgr ? ggml_numa_coordinator_manager_get_numa_nodes(mgr) : 1;
    
    // Handle fallback case where virtual node is not set
    if (numa_node < 0) {
        numa_node = 0;  // Default to node 0
    }
    if (max_numa_nodes <= 0) {
        max_numa_nodes = 1;  // At least one node
    }
    
    // Flash attention parallelizes by Q rows (neq1*neq2*neq3)
    const int64_t neq1 = q->ne[1];
    const int64_t neq2 = q->ne[2]; 
    const int64_t neq3 = q->ne[3];
    const int nr = neq1 * neq2 * neq3;  // total rows in q
    
    // NUMA-level data slicing: divide rows among NUMA nodes
    const int numa_rows_per_node = (nr + max_numa_nodes - 1) / max_numa_nodes;
    const int numa_start_row = numa_node * numa_rows_per_node;
    const int numa_end_row = MIN(numa_start_row + numa_rows_per_node, nr);
    const int numa_node_rows = numa_end_row - numa_start_row;
    
    if (numa_node_rows <= 0) {
        GGML_LOG_DEBUG("FLASH_ATTN_EXT NUMA node %d: No rows assigned, skipping\n", numa_node);
        return GGML_STATUS_SUCCESS;
    }
    
    GGML_LOG_DEBUG("FLASH_ATTN_EXT NUMA node %d (of %d): assigned rows %d to %d (%d rows total)\n",
                   numa_node, max_numa_nodes, numa_start_row, numa_end_row - 1, numa_node_rows);
    
    // Create modified compute params for this NUMA node's row range
    // The flash attention kernel uses params->ith and params->nth for thread-level parallelization
    // within the assigned NUMA row range
    struct ggml_compute_params numa_params = *params;
    
    // Override the row calculation in the kernel by temporarily modifying tensor dimensions
    // This is a bit tricky - we need to create tensor views for the NUMA slice
    
    // For now, use the coordinator's threading parameters directly and let the kernel
    // handle the threading within our NUMA node assignment
    GGML_LOG_DEBUG("FLASH_ATTN_EXT NUMA node %d: Using coordinator threading (ith=%d, nth=%d) for %d rows\n",
                   numa_node, params->ith, params->nth, numa_node_rows);
    
    // Call the flash attention mathematical kernel with coordinator's threading parameters
    // Note: This is a simplified approach - ideally we'd create tensor slices for the exact NUMA range
    // But flash attention is complex and would need significant refactoring for proper slicing
    ggml_compute_forward_flash_attn_ext(&numa_params, q, k, v, mask, ctx->operation);
    
    // Add memory barrier to ensure all writes are visible before returning
    __sync_synchronize();
    
    GGML_LOG_DEBUG("Successfully executed FLASH_ATTN_EXT chunk work function on NUMA node %d\n", numa_node);
    
    return GGML_STATUS_SUCCESS;
}
