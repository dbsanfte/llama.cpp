/*
 * NUMA Executor - Strategy Engine and Work Orchestration
 * 
 * This component absorbs the old dispatcher logic with a cleaner architecture:
 * - Analyzes operations and selects optimal execution strategies  
 * - Delegates to specialized kernels in numa-kernels/ directory
 * - Handles work submission to coordinator and completion synchronization
 */

#include "ggml-numa-executor.h"
#include "ggml-numa-simple-coordinator.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"  // For ggml_compute_forward_* function declarations
#include "ops.h"       // For actual ggml_compute_forward_* declarations
#include "ggml-numa-perf.h"  // Performance instrumentation

#ifdef __linux__
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#endif

#include <stdatomic.h>
#include <stdbool.h>

// Kernel headers - using the new query interface
#include "numa-kernels/numa-kernels.h"  // New centralized query interface

// Missing struct definition for MUL_MAT_ID work buffer calculation
struct mmid_row_mapping {
    int32_t i1; // i1
    int32_t i2; // i2  
};

// ============================================================================
// Core Executor Implementation  
// ============================================================================

// Compute graph execution - analyze nodes and dispatch to NUMA kernels or fallback
enum ggml_status ggml_numa_executor_execute_graph(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan) {
    if (!cgraph || !cplan) {
        return GGML_STATUS_FAILED;
    }
    
    GGML_LOG_DEBUG("NUMA Executor: Processing compute graph with %d nodes\n", cgraph->n_nodes);
    
    // Initialize kernel registry if not already done
    if (!ggml_numa_kernels_init()) {
        GGML_LOG_ERROR("NUMA Executor: Failed to initialize kernel registry\n");
        return GGML_STATUS_FAILED;
    }
    
    // Process each node in the graph
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        
        enum ggml_status result = ggml_numa_executor_execute_tensor(node, cplan);
        if (result != GGML_STATUS_SUCCESS) {
            GGML_LOG_ERROR("NUMA Executor: Failed to execute node %d (%s)\n", i, ggml_op_name(node->op));
            return result;
        }
    }
    
    GGML_LOG_DEBUG("NUMA Executor: Successfully completed graph execution\n");
    return GGML_STATUS_SUCCESS;
}

// ============================================================================
// Public API Implementation
// ============================================================================

enum ggml_status ggml_numa_executor_compute_graph(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan) {
    if (!cgraph || !cplan) {
        GGML_LOG_ERROR("NUMA Executor: Invalid graph or compute plan\n");
        return GGML_STATUS_FAILED;
    }
    
    // Check if NUMA dispatch is enabled
    extern bool ggml_numa_should_dispatch(void);
    if (!ggml_numa_should_dispatch()) {
        GGML_LOG_DEBUG("NUMA Executor: NUMA dispatch disabled, using standard ggml computation\n");
        // Call standard ggml computation instead of failing
        extern enum ggml_status ggml_graph_compute_impl(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);
        return ggml_graph_compute_impl(cgraph, cplan);
    }
    
    GGML_LOG_INFO("🎯 NUMA Executor: Processing graph with %d nodes (original plan: %d threads, fallback will use NUMA node 0 threads)\n", 
                  cgraph->n_nodes, cplan->n_threads);
    
    // Process each operation in the graph
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if (!node) continue;
        
        GGML_LOG_DEBUG("Executing node %d: %s\n", i, ggml_op_name(node->op));
        
        enum ggml_status result = ggml_numa_executor_execute_tensor(node, cplan);
        if (result != GGML_STATUS_SUCCESS) {
            GGML_LOG_ERROR("Operation %d (%s) failed with status %d\n", i, ggml_op_name(node->op), result);
            return result;
        }
    }
    
    GGML_LOG_INFO("✅ NUMA Executor: All %d operations completed successfully\n", cgraph->n_nodes);
    return GGML_STATUS_SUCCESS;
}

enum ggml_status ggml_numa_executor_execute_tensor(struct ggml_tensor * tensor, struct ggml_cplan * cplan) {
    if (!tensor || !cplan) {
        printf("DEBUG: NUMA Executor: NULL tensor=%p or cplan=%p, returning FAILED\n", tensor, cplan);
        return GGML_STATUS_FAILED;
    }
    
    const char* op_name = ggml_op_name(tensor->op);
    size_t tensor_size = ggml_nbytes(tensor);
    
    NUMA_PERF_START(NUMA_PERF_OPERATION_TOTAL, op_name, "numa_executor", -1, tensor_size, cplan->n_threads);
    
    printf("DEBUG: NUMA Executor: Starting execution for %s, threads=%d\n", op_name, cplan->n_threads);
    
    // Check NUMA environment
    #ifdef __linux__
    int current_cpu = sched_getcpu();
    int current_node = numa_node_of_cpu(current_cpu);
    int numa_nodes = numa_max_node() + 1;
    printf("DEBUG: NUMA Executor: Running on CPU %d (NUMA node %d), %d nodes available\n", 
           current_cpu, current_node, numa_nodes);
    #else
    printf("DEBUG: NUMA Executor: NUMA info not available (not Linux)\n");
    #endif
    
    // Query the kernel registry for execution information
    NUMA_PERF_START(NUMA_PERF_EXECUTOR_QUERY, op_name, "kernel_registry", -1, 0, 0);
    ggml_numa_kernel_query_result_t query_result = ggml_numa_kernels_query(tensor);
    NUMA_PERF_END();
    
    printf("DEBUG: NUMA Executor: Query result - supported=%s, kernel=%s\n", 
           query_result.supported ? "true" : "false", 
           query_result.supported ? query_result.kernel_name : "N/A");
    
    if (!query_result.supported) {
        GGML_LOG_DEBUG("NUMA Executor: Operation %s not supported by NUMA kernels, falling back to CPU\n", 
                      op_name);
        enum ggml_status result = ggml_numa_executor_fallback_to_cpu(tensor, cplan);
        NUMA_PERF_END();
        return result;
    }
    
    GGML_LOG_DEBUG("NUMA Executor: %s kernel selected for %s (efficiency=%.2f, strategy=%s, buffer=%zu bytes/thread)\n",
                   query_result.kernel_name,
                   op_name,
                   query_result.efficiency_score,
                   (query_result.strategy.node_strategy == NUMA_NODE_STRATEGY_DATA_PARALLEL) ? "data-parallel" : "single-node",
                   query_result.work_buffer_size_per_thread);
    
    // Initialize simple coordinator if needed
    NUMA_PERF_START(NUMA_PERF_COORDINATOR_INIT, op_name, query_result.kernel_name, -1, 0, cplan->n_threads);
    if (!ggml_numa_simple_coordinator_is_initialized()) {
        // Create threadpool parameters based on cplan
        struct ggml_threadpool_params tpp = {
            .n_threads = cplan->n_threads,
            .prio = GGML_SCHED_PRIO_NORMAL,
            .poll = 50,
            .strict_cpu = true,
            .paused = false,
            .numa_aware = false  // We ARE the NUMA coordinator
        };
        
        // Clear CPU mask - let the simple coordinator create optimal masks
        memset(tpp.cpumask, false, sizeof(tpp.cpumask));
        
        if (!ggml_numa_simple_coordinator_init(&tpp)) {
            NUMA_PERF_END();
            GGML_LOG_DEBUG("NUMA Executor: Failed to initialize simple coordinator, falling back to CPU for %s\n", 
                           op_name);
            enum ggml_status result = ggml_numa_executor_fallback_to_cpu(tensor, cplan);
            NUMA_PERF_END();
            return result;
        }
        NUMA_PERF_END();
    }
    
    // Execute using simple coordinator - no work groups, no complex synchronization
    enum ggml_status result = GGML_STATUS_SUCCESS;
    
    int num_numa_nodes = ggml_numa_simple_coordinator_get_num_nodes();
    printf("DEBUG: NUMA Executor: num_numa_nodes=%d, strategy=%s\n", 
           num_numa_nodes,
           (query_result.strategy.node_strategy == NUMA_NODE_STRATEGY_DATA_PARALLEL) ? "data-parallel" : "single-node");
           
    if (query_result.strategy.node_strategy == NUMA_NODE_STRATEGY_DATA_PARALLEL && num_numa_nodes > 1) {
        // Multi-node data-parallel execution
        NUMA_PERF_START(NUMA_PERF_EXECUTOR_KERNEL_EXEC, op_name, query_result.kernel_name, -1, tensor_size, num_numa_nodes);
        printf("DEBUG: NUMA Executor: Taking DATA_PARALLEL path with %d nodes\n", num_numa_nodes);
        GGML_LOG_DEBUG("NUMA Executor: Dispatching %s for data-parallel execution across %d nodes\n", 
                       op_name, num_numa_nodes);
        
        result = ggml_numa_simple_coordinator_execute_data_parallel(
            query_result.work_function, tensor, query_result.work_buffer_size_per_thread);
            
        printf("DEBUG: NUMA Executor: Data-parallel execution result=%d\n", result);
        NUMA_PERF_END();
        
    } else {
        // Single-node execution - choose optimal node (for now, use node 0)
        NUMA_PERF_START(NUMA_PERF_EXECUTOR_KERNEL_EXEC, op_name, query_result.kernel_name, 0, tensor_size, 1);
        int target_node = 0;
        
        printf("DEBUG: NUMA Executor: Taking SINGLE_NODE path, target_node=%d\n", target_node);
        GGML_LOG_DEBUG("NUMA Executor: Dispatching %s for single-node execution on node %d\n", 
                       ggml_op_name(tensor->op), target_node);
        
        result = ggml_numa_simple_coordinator_execute_single_node(
            query_result.work_function, tensor, target_node, query_result.work_buffer_size_per_thread);
        
        printf("DEBUG: NUMA Executor: Single-node execution result=%d\n", result);
        NUMA_PERF_END();
    }
    
    printf("DEBUG: NUMA Executor: Final result=%d for %s\n", result, op_name);
    if (result == GGML_STATUS_SUCCESS) {
        printf("DEBUG: NUMA Executor: SUCCESS - returning GGML_STATUS_SUCCESS\n");
        GGML_LOG_DEBUG("NUMA Executor: Successfully completed %s using %s\n", 
                       op_name, query_result.kernel_name);
    } else {
        printf("DEBUG: NUMA Executor: FAILURE - returning status %d\n", result);
        GGML_LOG_ERROR("NUMA Executor: Failed to execute %s using %s (status=%d)\n", 
                       op_name, query_result.kernel_name, (int)result);
    }
    
    NUMA_PERF_END();
    return result;
}

// ============================================================================
// Fallback to Standard CPU Implementation
// ============================================================================

// Forward declaration of function to control fallback flag in ggml-cpu.c
extern void ggml_numa_set_fallback_flag(bool value);
extern enum ggml_status ggml_graph_compute_impl(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);

// Calculate work buffer size needed for a specific operation (from ggml-cpu.c logic)
static size_t ggml_numa_calculate_work_size(struct ggml_tensor * tensor, int n_threads) {
    size_t work_size = 0;
    
    GGML_LOG_DEBUG("NUMA Work Buffer: Calculating for operation %s\n", ggml_op_name(tensor->op));
    
    switch (tensor->op) {
        case GGML_OP_CPY:
        case GGML_OP_DUP:
            {
                if (ggml_is_quantized(tensor->type) ||
                    // F16 -> BF16 and BF16 -> F16 copies go through intermediate F32
                    (tensor->src[0]->type == GGML_TYPE_F16  && tensor->src[1] && tensor->src[1]->type == GGML_TYPE_BF16) ||
                    (tensor->src[0]->type == GGML_TYPE_BF16 && tensor->src[1] && tensor->src[1]->type == GGML_TYPE_F16)) {
                    work_size = ggml_type_size(GGML_TYPE_F32) * tensor->ne[0] * n_threads;
                }
                GGML_LOG_DEBUG("NUMA Work Buffer: CPY/DUP calculated size: %zu bytes\n", work_size);
            } break;
        case GGML_OP_MUL_MAT:
            {
                if (tensor->src[0] && tensor->src[1]) {
                    const struct ggml_type_traits_cpu * traits = ggml_get_type_traits_cpu(tensor->src[0]->type);
                    const enum ggml_type vec_dot_type = traits->vec_dot_type;
                    
                    if (tensor->src[1]->type != vec_dot_type) {
                        work_size = ggml_row_size(vec_dot_type, ggml_nelements(tensor->src[1]));
                    }
                    GGML_LOG_DEBUG("NUMA Work Buffer: MUL_MAT src0_type=%d, src1_type=%d, vec_dot_type=%d, calculated size: %zu bytes\n", 
                                   (int)tensor->src[0]->type, (int)tensor->src[1]->type, (int)vec_dot_type, work_size);
                } else {
                    GGML_LOG_WARN("NUMA Work Buffer: MUL_MAT missing source tensors\n");
                }
            } break;
        case GGML_OP_MUL_MAT_ID:
            {
                if (tensor->src[0] && tensor->src[1] && tensor->src[2]) {
                    const struct ggml_tensor * src0 = tensor->src[0];
                    const struct ggml_tensor * src1 = tensor->src[1];
                    const struct ggml_tensor * ids = tensor->src[2];
                    const struct ggml_type_traits_cpu * traits = ggml_get_type_traits_cpu(src0->type);
                    const enum ggml_type vec_dot_type = traits->vec_dot_type;
                    const int n_as = src0->ne[2];
                    
                    // src1
                    if (src1->type != vec_dot_type) {
                        work_size += ggml_row_size(vec_dot_type, ggml_nelements(src1)) + sizeof(int64_t);
                    }
                    // matrix_row_counts + matrix_rows + atomic_current_chunk
                    work_size += n_as * sizeof(int64_t) + sizeof(int64_t);
                    work_size += n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping) + sizeof(int64_t);
                    work_size += 64*n_as + 64; // CACHE_LINE_SIZE approximation
                }
            } break;
        case GGML_OP_OUT_PROD:
            {
                if (tensor->src[0] && ggml_is_quantized(tensor->src[0]->type)) {
                    work_size = ggml_type_size(GGML_TYPE_F32) * tensor->src[0]->ne[0] * n_threads;
                }
            } break;
        default:
            // For unknown operations, use a reasonable default
            work_size = 1024 * 1024; // 1MB safety buffer
            break;
    }
    
    return work_size;
}

enum ggml_status ggml_numa_executor_fallback_to_cpu(struct ggml_tensor * tensor, struct ggml_cplan * cplan) {
    if (!tensor || !cplan) {
        return GGML_STATUS_FAILED;
    }
    
    const char* op_name = ggml_op_name(tensor->op);
    size_t tensor_size = ggml_nbytes(tensor);
    
    NUMA_PERF_START(NUMA_PERF_EXECUTOR_FALLBACK, op_name, "cpu_fallback", -1, tensor_size, cplan->n_threads);
    
    GGML_LOG_DEBUG("NUMA Fallback: Starting fallback for operation %s\n", op_name);
    
    // Set flag to disable NUMA dispatch during this call (prevents infinite recursion)
    ggml_numa_set_fallback_flag(true);
    
    // Calculate work buffer size needed for this specific operation
    size_t needed_work_size = ggml_numa_calculate_work_size(tensor, cplan->n_threads);
    
    void * work_data = NULL;
    bool allocated_work_buffer = false;
    
    // Check if existing work buffer is sufficient
    if (cplan->work_data && cplan->work_size >= needed_work_size) {
        work_data = cplan->work_data;
        GGML_LOG_DEBUG("NUMA Fallback: Using existing work buffer (%zu bytes >= %zu needed)\n", 
                       cplan->work_size, needed_work_size);
    } else if (needed_work_size > 0) {
        // Allocate NUMA-aware work buffer on node 0
        #ifdef __linux__
        if (numa_available() >= 0) {
            work_data = numa_alloc_onnode(needed_work_size, 0);
            if (work_data) {
                allocated_work_buffer = true;
                GGML_LOG_DEBUG("NUMA Fallback: Allocated %zu bytes work buffer on NUMA node 0\n", needed_work_size);
            }
        }
        #endif
        
        // Fall back to regular malloc if NUMA allocation failed
        if (!work_data) {
            work_data = malloc(needed_work_size);
            if (work_data) {
                allocated_work_buffer = true;
                GGML_LOG_DEBUG("NUMA Fallback: Allocated %zu bytes work buffer with malloc\n", needed_work_size);
            }
        }
        
        if (!work_data) {
            GGML_LOG_ERROR("NUMA Fallback: Failed to allocate %zu bytes work buffer\n", needed_work_size);
            ggml_numa_set_fallback_flag(false);
            return GGML_STATUS_FAILED;
        }
    }

    // Get fallback threadpool - use simple approach without complex coordinator
    struct ggml_threadpool * fallback_threadpool = NULL;
    int fallback_thread_count = cplan->n_threads; // Use original plan's thread count
    
    if (cplan->threadpool) {
        fallback_threadpool = cplan->threadpool;
        GGML_LOG_DEBUG("NUMA Fallback: Using existing threadpool with %d thread(s)\n", fallback_thread_count);
    } else {
        GGML_LOG_DEBUG("NUMA Fallback: No threadpool available, using single-threaded execution\n");
        fallback_thread_count = 1;
    }
    
    // Set up compute params for fallback execution  
    struct ggml_compute_params params = {
        .ith = 0,
        .nth = fallback_thread_count,
        .wsize = needed_work_size,
        .wdata = work_data,
        .threadpool = fallback_threadpool  // Use dedicated fallback threadpool
    };
    
    GGML_LOG_INFO("🔧 NUMA Fallback: Executing operation %s (work_size=%zu, fallback_threads=%d, params.nth=%d, threadpool=%p)\n", 
                   ggml_op_name(tensor->op), needed_work_size, fallback_thread_count, params.nth, (void*)fallback_threadpool);
    
    // CRITICAL FIX: Use ggml_graph_compute instead of ggml_compute_forward to properly activate all worker threads
    // Create a temporary single-node graph for this operation
    struct ggml_cgraph temp_graph = {0};
    temp_graph.nodes = &tensor;
    temp_graph.n_nodes = 1;
    temp_graph.size = 1;
    
    // Create a temporary compute plan for the fallback threadpool
    struct ggml_cplan temp_cplan = {
        .n_threads = fallback_thread_count,
        .threadpool = fallback_threadpool,
        .work_size = needed_work_size,
        .work_data = work_data,
        .abort_callback = NULL,
        .abort_callback_data = NULL
    };
    
    // Execute using the internal implementation that bypasses NUMA dispatcher
    enum ggml_status result = ggml_graph_compute_impl(&temp_graph, &temp_cplan);
    
    // Clear flag after computation
    ggml_numa_set_fallback_flag(false);
    
    // Check result
    if (result != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("NUMA Fallback: Graph computation failed with status %d\n", result);
        // Clean up allocated work buffer
        if (allocated_work_buffer && work_data) {
            #ifdef __linux__
            if (numa_available() >= 0) {
                numa_free(work_data, needed_work_size);
                GGML_LOG_DEBUG("NUMA Fallback: Freed NUMA work buffer\n");
            } else
            #endif
            {
                free(work_data);
                GGML_LOG_DEBUG("NUMA Fallback: Freed malloc work buffer\n");
            }
        }
        NUMA_PERF_END();
        return GGML_STATUS_FAILED;
    }
    
    // Clean up allocated work buffer
    if (allocated_work_buffer && work_data) {
        #ifdef __linux__
        if (numa_available() >= 0) {
            numa_free(work_data, needed_work_size);
            GGML_LOG_DEBUG("NUMA Fallback: Freed NUMA work buffer\n");
        } else
        #endif
        {
            free(work_data);
            GGML_LOG_DEBUG("NUMA Fallback: Freed malloc work buffer\n");
        }
    }
    
    GGML_LOG_DEBUG("NUMA Fallback: Operation %s completed successfully\n", op_name);
    NUMA_PERF_END();
    return GGML_STATUS_SUCCESS;
}
