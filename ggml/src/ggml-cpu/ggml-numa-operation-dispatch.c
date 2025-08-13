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

#ifdef __linux__
#include <numa.h>
#include <numaif.h>
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

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
        // No handler registered - route to single-node coordinator execution
        GGML_LOG_DEBUG("No handler registered for operation %s, using single-node execution\n", 
                      ggml_op_name(operation->op));
        
        // Route to coordinator for execution - no fallback needed, coordinator handles it
        result = ggml_numa_execute_single_node(manager, operation, context);
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
    
    GGML_LOG_DEBUG("Executing complex operation %s using graph-based approach\n", 
                   ggml_op_name(operation->op));
    
    // Use primary coordinator for complex operations
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
        GGML_LOG_DEBUG("NUMA coordination not beneficial - falling back to standard computation\n");
        // TODO: For now, we'll just return failure when NUMA coordination isn't beneficial
        // In the future, this could fall back to standard GGML computation
        GGML_LOG_ERROR("Fallback to standard computation not implemented in dispatcher\n");
        return GGML_STATUS_FAILED;
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
    .operation_type = GGML_OP_ROPE, // Represents complex operations
    .default_strategy = NUMA_EXECUTION_SINGLE_NODE,
    .complexity = NUMA_OP_COMPLEXITY_COMPLEX,
    .workload_type = NUMA_OP_CACHE_SENSITIVE,
    .min_elements_for_parallel = INT64_MAX, // Never parallelize by default
    .optimal_chunk_size = 0,
    .parallel_efficiency_estimate = 0.0f,
    .requires_synchronization = true,
    .supports_in_place = false
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
    // Use simpler submission for now - the original function signature is different
    return ggml_numa_coordinator_manager_submit_data_parallel_work(manager, operation);
}
