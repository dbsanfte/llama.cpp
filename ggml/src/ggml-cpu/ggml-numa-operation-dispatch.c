/**
 * NUMA Operation Dispatch Infrastructure Implementation
 * 
 * This implements the intelligent dispatcher system that routes operations
 * to appropriate execution strategies while preserving existing thread synchronization.
 */

#include "ggml-numa-operation-dispatch.h"
#include "ggml-numa-coordinator.h"
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
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdatomic.h>      // For atomic operations in statistics

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

static enum ggml_status ggml_numa_execute_fallback_direct(
    struct ggml_numa_coordinator_manager * manager,
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context
);

// MUL_MAT analyzer function
static enum ggml_status ggml_numa_analyze_mul_mat(
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context,
    ggml_numa_execution_strategy_t * strategy,
    int * recommended_chunks
);

//
// Coordinator Interface Implementation
//

// Forward declarations for coordinator interface functions
static struct ggml_threadpool * coordinator_get_numa_threadpool(struct ggml_numa_coordinator_manager * manager, int numa_node);
static int coordinator_get_numa_thread_count(struct ggml_numa_coordinator_manager * manager, int numa_node);
static bool coordinator_ensure_work_buffer(struct ggml_numa_coordinator_manager * manager, int numa_node, size_t required_size);
static void * coordinator_get_work_buffer(struct ggml_numa_coordinator_manager * manager, int numa_node);
static size_t coordinator_get_work_buffer_size(struct ggml_numa_coordinator_manager * manager, int numa_node);
static int coordinator_submit_work(struct ggml_numa_coordinator_manager * manager, struct ggml_tensor * operation, int target_numa_node);
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
extern const ggml_numa_operation_handler_t ggml_numa_handler_complex;

//
// Global Dispatch State
//

static ggml_numa_operation_handler_t * g_operation_handlers[GGML_OP_COUNT];
static ggml_numa_dispatch_stats_t g_dispatch_stats;
static bool g_dispatch_initialized = false;

//
// Forward Declarations
//

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
    
    // Register built-in handlers
    ggml_numa_dispatch_register_handler(&ggml_numa_handler_elementwise);
    ggml_numa_dispatch_register_handler(&ggml_numa_handler_mul_mat_enhanced);  // Use enhanced MUL_MAT handler
    ggml_numa_dispatch_register_handler(&ggml_numa_handler_complex);
    
    g_dispatch_initialized = true;
    GGML_LOG_INFO("NUMA operation dispatch system initialized with enhanced handlers\n");
}

void ggml_numa_dispatch_register_handler(const ggml_numa_operation_handler_t * handler) {
    if (!handler || handler->operation_type >= GGML_OP_COUNT) {
        GGML_LOG_ERROR("Invalid operation handler for registration\n");
        return;
    }
    
    // Allocate and copy handler
    ggml_numa_operation_handler_t * registered_handler = malloc(sizeof(ggml_numa_operation_handler_t));
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
                strategy = NUMA_EXECUTION_SINGLE_NODE;
            }
        } else {
            // Use default strategy from handler
            strategy = handler->default_strategy;
            recommended_chunks = context->numa_nodes;
        }
        
        // Execute based on determined strategy
        switch (strategy) {
            case NUMA_EXECUTION_SINGLE_NODE:
                result = ggml_numa_execute_single_node(manager, operation, context);
                break;
                
            case NUMA_EXECUTION_DATA_PARALLEL:
                result = ggml_numa_execute_data_parallel(manager, operation, context);
                g_dispatch_stats.parallelized_operations++;
                break;
                
            case NUMA_EXECUTION_TASK_PARALLEL:
            case NUMA_EXECUTION_HYBRID:
                // For complex operations like MUL_MAT, use graph-based execution
                GGML_LOG_DEBUG("Complex execution strategy for %s, using graph-based approach\n", 
                              ggml_op_name(operation->op));
                result = ggml_numa_execute_complex_graph(manager, operation, context);
                g_dispatch_stats.parallelized_operations++;
                break;
                
            case NUMA_EXECUTION_CUSTOM:
                // Fall back to single node for custom strategies not yet implemented
                GGML_LOG_DEBUG("Custom execution strategy for %s, falling back to single node\n", 
                              ggml_op_name(operation->op));
                result = ggml_numa_execute_single_node(manager, operation, context);
                break;
        }
        
    } else {
        // No handler registered - use single-threaded fallback system
        GGML_LOG_DEBUG("No handler registered for operation %s, using fallback execution\n", 
                      ggml_op_name(operation->op));
        
        // Route to fallback system for safe single-threaded execution
        result = ggml_numa_execute_operation_fallback((struct ggml_tensor *)operation, NULL);
        
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
    
    GGML_LOG_DEBUG("Executing %s on single node (NUMA node 0)\n", ggml_op_name(operation->op));
    
    // Route to primary coordinator (NUMA node 0)
    int work_id = g_coordinator_interface.submit_work(manager, (struct ggml_tensor *)operation, 0);
    if (work_id < 0) {
        GGML_LOG_ERROR("Failed to submit single-node work for operation %s\n", ggml_op_name(operation->op));
        return GGML_STATUS_FAILED;
    }
    
    GGML_LOG_DEBUG("Submitted single-node work (ID: %d) for operation %s\n", work_id, ggml_op_name(operation->op));
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_numa_execute_data_parallel(
    struct ggml_numa_coordinator_manager * manager,
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context) {
    
    GGML_LOG_DEBUG("Executing %s with data parallelism across %d NUMA nodes\n", 
                   ggml_op_name(operation->op), context->numa_nodes);
    
    // Use the coordinator interface for data parallel work submission
    int work_group_id = g_coordinator_interface.submit_data_parallel_work(
        manager, (struct ggml_tensor *)operation, -1, NULL, 0);
    
    if (work_group_id < 0) {
        GGML_LOG_WARN("Data parallel execution failed for %s, falling back to single node\n", 
                     ggml_op_name(operation->op));
        return ggml_numa_execute_single_node(manager, operation, context);
    }
    
    GGML_LOG_DEBUG("Submitted data parallel work group (ID: %d) for operation %s\n", 
                   work_group_id, ggml_op_name(operation->op));
    return GGML_STATUS_SUCCESS;
}

// Enhanced execution strategy for complex operations like MUL_MAT
static enum ggml_status ggml_numa_execute_complex_graph(
    struct ggml_numa_coordinator_manager * manager,
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context) {
    
    (void)context; // Suppress unused parameter warning
    
    GGML_LOG_DEBUG("Executing complex operation %s using graph-based approach\n", 
                  ggml_op_name(operation->op));    // Use primary coordinator for complex operations
    int primary_numa_node = 0;
    
    // Use the coordinator's graph execution capability
    return ggml_numa_coordinator_execute_graph_operation(manager, (struct ggml_tensor *)operation, primary_numa_node);
}

//
// Work Context Creation
//

ggml_numa_work_context_t ggml_numa_create_work_context(
    const struct ggml_tensor * operation,
    struct ggml_numa_coordinator_manager * manager) {
    
    ggml_numa_work_context_t context = {0};
    
    if (!operation || !manager) {
        return context;
    }
    
    // Extract tensor information
    context.total_elements = ggml_nelements(operation);
    context.element_size = ggml_element_size(operation);
    context.n_dims = ggml_n_dims(operation);
    
    // Copy tensor dimensions
    for (int i = 0; i < GGML_MAX_DIMS && i < context.n_dims; i++) {
        context.ne[i] = operation->ne[i];
    }
    
    // Get system information through coordinator interface
    context.numa_nodes = ggml_numa_coordinator_manager_get_numa_nodes(manager);
    context.threads_per_node = ggml_numa_coordinator_get_thread_count(manager, 0);  // Use first node as reference
    if (context.threads_per_node < 0) {
        context.threads_per_node = 1;
    }
    
    // Performance hints (these could be determined dynamically)
    context.l3_cache_size = 32 * 1024 * 1024;  // 32MB estimate
    context.memory_bandwidth = 100ULL * 1024ULL * 1024ULL * 1024ULL;  // 100GB/s estimate (fix overflow)
    
    return context;
}

//
// Auto-chunking Implementation (Basic Version)
//

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
    
    // Process each operation in the graph through the dispatcher
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * operation = cgraph->nodes[i];
        if (!operation) continue;
        
        // Create work context for this operation
        ggml_numa_work_context_t context = ggml_numa_create_work_context(operation, mgr);
        
        // Let dispatcher decide strategy and execute
        enum ggml_status result = ggml_numa_dispatch_operation(mgr, operation, &context);
        
        if (result != GGML_STATUS_SUCCESS) {
            GGML_LOG_ERROR("Dispatcher failed to execute operation %d (%s)\n", 
                          i, ggml_op_name(operation->op));
            return GGML_STATUS_FAILED;
        }
        
        GGML_LOG_DEBUG("Dispatcher successfully executed operation %d (%s)\n", 
                      i, ggml_op_name(operation->op));
    }
    
    GGML_LOG_INFO("NUMA graph computation completed successfully via dispatcher\n");
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
            
            // Create work context for this operation
            ggml_numa_work_context_t context = ggml_numa_create_work_context(operation, mgr);
            
            // Let dispatcher decide strategy and execute
            enum ggml_status result = ggml_numa_dispatch_operation(mgr, operation, &context);
            
            if (result != GGML_STATUS_SUCCESS) {
                GGML_LOG_ERROR("Virtual NUMA dispatcher failed to execute operation %d (%s)\n", 
                              i, ggml_op_name(operation->op));
                return GGML_STATUS_FAILED;
            }
            
            GGML_LOG_DEBUG("Virtual NUMA dispatcher successfully executed operation %d (%s)\n", 
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

// These will be implemented as separate, focused handlers
const ggml_numa_operation_handler_t ggml_numa_handler_elementwise = {
    .operation_type = GGML_OP_ADD, // Will be expanded to handle multiple element-wise ops
    .default_strategy = NUMA_EXECUTION_DATA_PARALLEL,
    .complexity = NUMA_OP_COMPLEXITY_SIMPLE,
    .workload_type = NUMA_OP_MEMORY_BOUND,
    .min_elements_for_parallel = 10000,
    .optimal_chunk_size = 1024 * 1024,
    .parallel_efficiency_estimate = 0.85f,
    .requires_synchronization = false,
    .supports_in_place = true
};

// MUL_MAT analyzer function - determines optimal execution strategy
static enum ggml_status ggml_numa_analyze_mul_mat(
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context,
    ggml_numa_execution_strategy_t * strategy,
    int * recommended_chunks) {
    
    if (!operation || !context || !strategy || !recommended_chunks) {
        return GGML_STATUS_FAILED;
    }
    
    // Calculate matrix dimensions
    const int64_t ne00 = operation->src[0]->ne[0]; // K dimension  
    const int64_t ne01 = operation->src[0]->ne[1]; // M dimension
    const int64_t ne10 = operation->src[1]->ne[0]; // K dimension
    const int64_t ne11 = operation->src[1]->ne[1]; // N dimension
    
    (void)ne10; // Suppress unused variable warning - used for validation
    
    // Calculate computational complexity (roughly proportional to M*K*N)
    const int64_t complexity = ne01 * ne00 * ne11;
    
    GGML_LOG_DEBUG("MUL_MAT analysis: M=%ld, K=%ld, N=%ld, complexity=%ld\n", 
                   ne01, ne00, ne11, complexity);
    
    // Decision logic based on matrix size and system resources
    if (context->numa_nodes <= 1) {
        // Single NUMA node - use graph-based execution for optimal threading
        *strategy = NUMA_EXECUTION_HYBRID;  // Will route to graph-based execution
        *recommended_chunks = 1;
        GGML_LOG_DEBUG("MUL_MAT: Single NUMA node, using graph-based execution\n");
    } else if (complexity > 100000000) {  // 100M operations
        // Large matrices - worth the overhead of multi-node execution
        *strategy = NUMA_EXECUTION_DATA_PARALLEL;
        *recommended_chunks = context->numa_nodes;
        GGML_LOG_DEBUG("MUL_MAT: Large matrix, using data parallel execution\n");
    } else {
        // Medium matrices - use single node with full threading
        *strategy = NUMA_EXECUTION_HYBRID;  // Will route to graph-based execution
        *recommended_chunks = 1;
        GGML_LOG_DEBUG("MUL_MAT: Medium matrix, using single-node graph execution\n");
    }
    
    return GGML_STATUS_SUCCESS;
}

// Enhanced MUL_MAT handler with intelligent analysis
const ggml_numa_operation_handler_t ggml_numa_handler_mul_mat_enhanced = {
    .operation_type = GGML_OP_MUL_MAT,
    .default_strategy = NUMA_EXECUTION_HYBRID,  // Use graph-based execution by default
    .complexity = NUMA_OP_COMPLEXITY_MODERATE,
    .workload_type = NUMA_OP_COMPUTE_BOUND,
    .min_elements_for_parallel = 10000,  // Lower threshold for MUL_MAT
    .optimal_chunk_size = 4 * 1024 * 1024,  // 4MB chunks
    .parallel_efficiency_estimate = 0.85f,  // High efficiency for matrix operations
    .requires_synchronization = false,
    .supports_in_place = false,
    .analyze = ggml_numa_analyze_mul_mat  // Custom analyzer
};

const ggml_numa_operation_handler_t ggml_numa_handler_complex = {
    .operation_type = GGML_OP_ROPE, // Complex ROPE operations
    .default_strategy = NUMA_EXECUTION_DATA_PARALLEL,  // Use data parallel for large ROPE
    .complexity = NUMA_OP_COMPLEXITY_COMPLEX,
    .workload_type = NUMA_OP_CACHE_SENSITIVE,
    .min_elements_for_parallel = 100000, // Enable NUMA for operations with >100K elements (was INT64_MAX)
    .optimal_chunk_size = 1 * 1024 * 1024,  // 1MB chunks for ROPE
    .parallel_efficiency_estimate = 0.75f,  // Good efficiency for ROPE
    .requires_synchronization = true,
    .supports_in_place = false,
    .analyze = NULL  // Use default analysis for now
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

#define DISPATCH_UNARY(op_enum) \
    case op_enum: \
        ggml_compute_forward_unary(&fallback_params, tensor); \
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
    if (!tensor) {
        GGML_LOG_ERROR("Invalid tensor for fallback execution\n");
        return GGML_STATUS_FAILED;
    }

    // Create single-threaded compute params to avoid threadpool conflicts
    struct ggml_compute_params fallback_params = {
        .ith = 0,                  // Single thread index
        .nth = 1,                  // Single thread total
        .wsize = cplan ? cplan->work_size : 0,
        .wdata = cplan ? cplan->work_data : NULL,
        .threadpool = NULL         // Critical: no threadpool conflicts
    };

    // Increment fallback usage statistics
    atomic_fetch_add_explicit(&g_dispatch_stats.fallback_operations, 1, memory_order_relaxed);

    GGML_LOG_DEBUG("Executing operation %s via single-threaded fallback\n", ggml_op_name(tensor->op));

    // Optimized switch statement with helper macros for reduced repetition
    switch (tensor->op) {
        // No operation - pass through
        case GGML_OP_NONE:
            break;
            
        // === BASIC MATH OPERATIONS (using macros for cleaner code) ===
        DISPATCH_SIMPLE(GGML_OP_DUP, dup)
        DISPATCH_SIMPLE(GGML_OP_ADD, add)
        DISPATCH_SIMPLE(GGML_OP_ADD1, add1)
        DISPATCH_SIMPLE(GGML_OP_ACC, acc)
        DISPATCH_SIMPLE(GGML_OP_SUB, sub)
        DISPATCH_SIMPLE(GGML_OP_MUL, mul)
        DISPATCH_SIMPLE(GGML_OP_DIV, div)
        
        // === UNARY MATH OPERATIONS (all use same handler) ===
        DISPATCH_UNARY(GGML_OP_SQR)
        DISPATCH_UNARY(GGML_OP_SQRT)
        DISPATCH_UNARY(GGML_OP_LOG)
        DISPATCH_UNARY(GGML_OP_SIN)
        DISPATCH_UNARY(GGML_OP_COS)
        
        // === REDUCTION OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_SUM, sum)
        DISPATCH_SIMPLE(GGML_OP_SUM_ROWS, sum_rows)
        DISPATCH_SIMPLE(GGML_OP_MEAN, mean)
        DISPATCH_SIMPLE(GGML_OP_ARGMAX, argmax)
        DISPATCH_SIMPLE(GGML_OP_COUNT_EQUAL, count_equal)
        
        // === TENSOR MANIPULATION ===
        DISPATCH_SIMPLE(GGML_OP_REPEAT, repeat)
        DISPATCH_SIMPLE(GGML_OP_REPEAT_BACK, repeat_back)
        DISPATCH_SIMPLE(GGML_OP_CONCAT, concat)
        DISPATCH_SIMPLE(GGML_OP_SILU_BACK, silu_back)
        DISPATCH_SIMPLE(GGML_OP_CPY, cpy)
        DISPATCH_SIMPLE(GGML_OP_CONT, cont)
        DISPATCH_SIMPLE(GGML_OP_RESHAPE, reshape)
        DISPATCH_SIMPLE(GGML_OP_VIEW, view)
        DISPATCH_SIMPLE(GGML_OP_PERMUTE, permute)
        DISPATCH_SIMPLE(GGML_OP_TRANSPOSE, transpose)
        DISPATCH_SIMPLE(GGML_OP_GET_ROWS, get_rows)
        DISPATCH_SIMPLE(GGML_OP_GET_ROWS_BACK, get_rows_back)
        DISPATCH_SIMPLE(GGML_OP_SET_ROWS, set_rows)
        
        // === NORMALIZATION OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_NORM, norm)
        DISPATCH_SIMPLE(GGML_OP_RMS_NORM, rms_norm)
        DISPATCH_SIMPLE(GGML_OP_RMS_NORM_BACK, rms_norm_back)
        DISPATCH_SIMPLE(GGML_OP_GROUP_NORM, group_norm)
        DISPATCH_SIMPLE(GGML_OP_L2_NORM, l2_norm)
        
        // === MATRIX OPERATIONS (with validation) ===
        case GGML_OP_MUL_MAT:
            if (validate_matrix_operation(tensor, "MUL_MAT") != GGML_STATUS_SUCCESS) {
                return GGML_STATUS_FAILED;
            }
            ggml_compute_forward_mul_mat(&fallback_params, tensor);
            break;
            
        case GGML_OP_MUL_MAT_ID:
            // MUL_MAT_ID uses same implementation as MUL_MAT for fallback
            if (validate_matrix_operation(tensor, "MUL_MAT_ID") != GGML_STATUS_SUCCESS) {
                return GGML_STATUS_FAILED;
            }
            ggml_compute_forward_mul_mat(&fallback_params, tensor);
            break;
            
        DISPATCH_SIMPLE(GGML_OP_OUT_PROD, out_prod)
        
        // === UTILITY OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_SCALE, scale)
        DISPATCH_SIMPLE(GGML_OP_SET, set)
        
        // === MASKING OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_DIAG, diag)
        DISPATCH_SIMPLE(GGML_OP_DIAG_MASK_INF, diag_mask_inf)
        DISPATCH_SIMPLE(GGML_OP_DIAG_MASK_ZERO, diag_mask_zero)
        
        // === ACTIVATION OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_SOFT_MAX, soft_max)
        case GGML_OP_SOFT_MAX_BACK:
            // Use soft_max_ext_back as fallback for soft_max_back
            ggml_compute_forward_soft_max_ext_back(&fallback_params, tensor);
            break;
            
        // === COMPLEX OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_ROPE, rope)
        DISPATCH_SIMPLE(GGML_OP_ROPE_BACK, rope_back)
        DISPATCH_SIMPLE(GGML_OP_CLAMP, clamp)
        
        // === CONVOLUTION OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_CONV_TRANSPOSE_1D, conv_transpose_1d)
        DISPATCH_SIMPLE(GGML_OP_IM2COL, im2col)
        case GGML_OP_IM2COL_BACK:
            ggml_compute_forward_im2col_back_f32(&fallback_params, tensor);
            break;
        DISPATCH_SIMPLE(GGML_OP_CONV_2D, conv_2d)
        DISPATCH_SIMPLE(GGML_OP_CONV_2D_DW, conv_2d_dw)
        DISPATCH_SIMPLE(GGML_OP_CONV_TRANSPOSE_2D, conv_transpose_2d)
        
        // === POOLING OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_POOL_1D, pool_1d)
        DISPATCH_SIMPLE(GGML_OP_POOL_2D, pool_2d)
        DISPATCH_SIMPLE(GGML_OP_POOL_2D_BACK, pool_2d_back)
        
        // === UTILITY OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_UPSCALE, upscale)
        DISPATCH_SIMPLE(GGML_OP_PAD, pad)
        DISPATCH_SIMPLE(GGML_OP_PAD_REFLECT_1D, pad_reflect_1d)
        DISPATCH_SIMPLE(GGML_OP_ROLL, roll)
        DISPATCH_SIMPLE(GGML_OP_ARANGE, arange)
        DISPATCH_SIMPLE(GGML_OP_TIMESTEP_EMBEDDING, timestep_embedding)
        DISPATCH_SIMPLE(GGML_OP_ARGSORT, argsort)
        DISPATCH_SIMPLE(GGML_OP_LEAKY_RELU, leaky_relu)
        
        // === ATTENTION OPERATIONS (special parameter handling) ===
        case GGML_OP_FLASH_ATTN_EXT:
            // Flash attention requires special parameter handling
            if (tensor->src[0] && tensor->src[1] && tensor->src[2]) {
                ggml_compute_forward_flash_attn_ext(&fallback_params,
                    tensor->src[0], tensor->src[1], tensor->src[2], 
                    tensor->src[3], tensor);
            } else {
                GGML_LOG_ERROR("FLASH_ATTN_EXT requires valid Q, K, V tensors\n");
                return GGML_STATUS_FAILED;
            }
            break;
            
        case GGML_OP_FLASH_ATTN_BACK:
            ggml_compute_forward_flash_attn_back(&fallback_params, false, tensor);
            break;
            
        // === STATE SPACE MODEL OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_SSM_CONV, ssm_conv)
        DISPATCH_SIMPLE(GGML_OP_SSM_SCAN, ssm_scan)
        
        // === WINDOW OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_WIN_PART, win_part)
        DISPATCH_SIMPLE(GGML_OP_WIN_UNPART, win_unpart)
        
        // === POSITIONAL OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_GET_REL_POS, get_rel_pos)
        DISPATCH_SIMPLE(GGML_OP_ADD_REL_POS, add_rel_pos)
        
        // === RWKV OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_RWKV_WKV6, rwkv_wkv6)
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_compute_forward_gla(&fallback_params, tensor);
            break;
        DISPATCH_SIMPLE(GGML_OP_RWKV_WKV7, rwkv_wkv7)
        
        // === GENERIC OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_UNARY, unary)
        
        // === CUSTOM MAP OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_MAP_CUSTOM1, map_custom1)
        DISPATCH_SIMPLE(GGML_OP_MAP_CUSTOM2, map_custom2)
        DISPATCH_SIMPLE(GGML_OP_MAP_CUSTOM3, map_custom3)
        DISPATCH_SIMPLE(GGML_OP_CUSTOM, custom)
        
        // === LOSS OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_CROSS_ENTROPY_LOSS, cross_entropy_loss)
        DISPATCH_SIMPLE(GGML_OP_CROSS_ENTROPY_LOSS_BACK, cross_entropy_loss_back)
        
        // === OPTIMIZATION OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_OPT_STEP_ADAMW, opt_step_adamw)
        
        // === ADDITIONAL OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_GLU, glu)
            
        default:
            GGML_LOG_ERROR("Unsupported operation %s (%d) in fallback system\n", 
                          ggml_op_name(tensor->op), tensor->op);
            return GGML_STATUS_FAILED;
    }

    GGML_LOG_DEBUG("Successfully executed operation %s via fallback\n", ggml_op_name(tensor->op));
    return GGML_STATUS_SUCCESS;
}

// Cleanup helper macros
#undef DISPATCH_SIMPLE
#undef DISPATCH_UNARY
            ggml_compute_forward_custom(&fallback_params, tensor);
            break;
            
        // Loss Operations
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_compute_forward_cross_entropy_loss(&fallback_params, tensor);
            break;
            
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_compute_forward_cross_entropy_loss_back(&fallback_params, tensor);
            break;
            
        // Optimization Operations
        case GGML_OP_OPT_STEP_ADAMW:
            ggml_compute_forward_opt_step_adamw(&fallback_params, tensor);
            break;
            
        // Additional Operations
        case GGML_OP_GLU:
            ggml_compute_forward_glu(&fallback_params, tensor);
            break;
            
        default:
            GGML_LOG_ERROR("Unsupported operation %s (%d) in fallback system\n", 
                          ggml_op_name(tensor->op), tensor->op);
            return GGML_STATUS_FAILED;
    }

    GGML_LOG_DEBUG("Successfully executed operation %s via fallback\n", ggml_op_name(tensor->op));
    return GGML_STATUS_SUCCESS;
}

static void * coordinator_get_work_buffer(struct ggml_numa_coordinator_manager * manager, int numa_node) {
    return ggml_numa_coordinator_get_work_buffer(manager, numa_node);
}

static size_t coordinator_get_work_buffer_size(struct ggml_numa_coordinator_manager * manager, int numa_node) {
    return ggml_numa_coordinator_get_work_buffer_size(manager, numa_node);
}

static int coordinator_submit_work(struct ggml_numa_coordinator_manager * manager, struct ggml_tensor * operation, int target_numa_node) {
    return ggml_numa_coordinator_manager_submit_work(manager, operation, target_numa_node);
}

static int coordinator_submit_data_parallel_work(struct ggml_numa_coordinator_manager * manager, struct ggml_tensor * operation, 
                                                int work_group_id, const int * target_nodes, int num_target_nodes) {
    (void)work_group_id;    // Suppress unused parameter warning  
    (void)target_nodes;     // Suppress unused parameter warning
    (void)num_target_nodes; // Suppress unused parameter warning
    
    // Use simpler submission for now - the original function signature is different
    return ggml_numa_coordinator_manager_submit_data_parallel_work(manager, operation);
}
