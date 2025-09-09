/**
 * @file ggml-numa-executor.c
 * @brief NUMA Executor - Strategy Engine and Work Orchestration
 * 
 * This component serves as the central orchestration layer for NUMA-aware
 * computation, absorbing the old dispatcher logic with a cleaner, more
 * maintainable architecture:
 * 
 * Core Functions:
 * - Analyzes operations and selects optimal execution strategies  
 * - Delegates to specialized kernels in numa-kernels/ directory
 * - Handles work submission to coordinator and completion synchronization
 * - Provides fallback mechanisms for unsupported operations
 * - Implements performance monitoring and efficiency tracking
 * 
 * Architecture Benefits:
 * - O(1) kernel lookups via direct function pointer dispatch
 * - Eliminates switch statement maintenance overhead
 * - Supports dynamic strategy selection based on tensor complexity
 * - Provides seamless integration with NUMA coordinator
 * - Enables easy addition of new operations without central modifications
 * 
 * @author David Sanftenberg
 * @date 2025
 */

#include "ggml-numa-executor.h"
#include "ggml-numa-openmp-coordinator.h"
#include "numa-kernels/numa-kernels.h"  // For ggml_numa_is_kernel_noop
#include "ggml-cpu-impl.h"
#include "ops.h"
#include "binary-ops.h"  // For ggml_compute_forward_mul, sub, div
#include "unary-ops.h"   // For unary operations
#include "ggml-numa-perf.h"  // Performance instrumentation

#ifdef __linux__
#define _GNU_SOURCE  // For sched_getcpu
#include <sched.h>
#endif

// Forward declarations
static size_t ggml_numa_calculate_work_size(struct ggml_tensor * tensor, int n_threads);

/**
 * @brief Direct kernel dispatch implementation
 * 
 * High-performance execution path that calls compute functions directly
 * without the overhead of temporary graph creation. This function provides
 * zero-copy direct kernel invocation with minimal function call overhead.
 * 
 * @param tensor The operation tensor to execute
 * @param params The compute parameters with threading and buffer information
 * @return GGML_STATUS_SUCCESS on success, GGML_STATUS_FAILED on error
 */
enum ggml_status ggml_numa_executor_call_direct_kernel(struct ggml_tensor * tensor, struct ggml_compute_params * params) {
    if (!tensor || !params) {
        return GGML_STATUS_FAILED;
    }
    
    GGML_LOG_DEBUG("Direct Kernel: Dispatching operation %s directly\n", ggml_op_name(tensor->op));
    
    // Direct kernel dispatch based on operation type - no temporary graph overhead
    switch (tensor->op) {
        case GGML_OP_DUP:
            ggml_compute_forward_dup(params, tensor);
            break;
        case GGML_OP_ADD:
            ggml_compute_forward_add(params, tensor);
            break;
        case GGML_OP_ADD1:
            ggml_compute_forward_add1(params, tensor);
            break;
        case GGML_OP_ACC:
            ggml_compute_forward_acc(params, tensor);
            break;
        case GGML_OP_SUB:
            ggml_compute_forward_sub(params, tensor);
            break;
        case GGML_OP_MUL:
            ggml_compute_forward_mul(params, tensor);
            break;
        case GGML_OP_DIV:
            ggml_compute_forward_div(params, tensor);
            break;
        case GGML_OP_SQR:
            ggml_compute_forward_sqr(params, tensor);
            break;
        case GGML_OP_SQRT:
            ggml_compute_forward_sqrt(params, tensor);
            break;
        case GGML_OP_LOG:
            ggml_compute_forward_log(params, tensor);
            break;
        case GGML_OP_SIN:
            ggml_compute_forward_sin(params, tensor);
            break;
        case GGML_OP_COS:
            ggml_compute_forward_cos(params, tensor);
            break;
        case GGML_OP_SUM:
            ggml_compute_forward_sum(params, tensor);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_compute_forward_sum_rows(params, tensor);
            break;
        case GGML_OP_MEAN:
            ggml_compute_forward_mean(params, tensor);
            break;
        case GGML_OP_ARGMAX:
            ggml_compute_forward_argmax(params, tensor);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_compute_forward_count_equal(params, tensor);
            break;
        case GGML_OP_REPEAT:
            ggml_compute_forward_repeat(params, tensor);
            break;
        case GGML_OP_REPEAT_BACK:
            ggml_compute_forward_repeat_back(params, tensor);
            break;
        case GGML_OP_CONCAT:
            ggml_compute_forward_concat(params, tensor);
            break;
        case GGML_OP_SILU_BACK:
            ggml_compute_forward_silu_back(params, tensor);
            break;
        case GGML_OP_NORM:
            ggml_compute_forward_norm(params, tensor);
            break;
        case GGML_OP_RMS_NORM:
            ggml_compute_forward_rms_norm(params, tensor);
            break;
        case GGML_OP_RMS_NORM_BACK:
            ggml_compute_forward_rms_norm_back(params, tensor);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_compute_forward_group_norm(params, tensor);
            break;
        case GGML_OP_L2_NORM:
            ggml_compute_forward_l2_norm(params, tensor);
            break;
        case GGML_OP_OUT_PROD:
            ggml_compute_forward_out_prod(params, tensor);
            break;
        case GGML_OP_SCALE:
            ggml_compute_forward_scale(params, tensor);
            break;
        case GGML_OP_SET:
            ggml_compute_forward_set(params, tensor);
            break;
        case GGML_OP_CPY:
            ggml_compute_forward_cpy(params, tensor);
            break;
        case GGML_OP_CONT:
            ggml_compute_forward_cont(params, tensor);
            break;
        case GGML_OP_RESHAPE:
            ggml_compute_forward_reshape(params, tensor);
            break;
        case GGML_OP_VIEW:
            ggml_compute_forward_view(params, tensor);
            break;
        case GGML_OP_PERMUTE:
            ggml_compute_forward_permute(params, tensor);
            break;
        case GGML_OP_TRANSPOSE:
            ggml_compute_forward_transpose(params, tensor);
            break;
        case GGML_OP_GET_ROWS:
            ggml_compute_forward_get_rows(params, tensor);
            break;
        case GGML_OP_GET_ROWS_BACK:
            ggml_compute_forward_get_rows_back(params, tensor);
            break;
        case GGML_OP_SET_ROWS:
            ggml_compute_forward_set_rows(params, tensor);
            break;
        case GGML_OP_DIAG:
            ggml_compute_forward_diag(params, tensor);
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_compute_forward_diag_mask_inf(params, tensor);
            break;
        case GGML_OP_DIAG_MASK_ZERO:
            ggml_compute_forward_diag_mask_zero(params, tensor);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_compute_forward_soft_max(params, tensor);
            break;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_compute_forward_soft_max_ext_back(params, tensor);
            break;
        case GGML_OP_ROPE:
            ggml_compute_forward_rope(params, tensor);
            break;
        case GGML_OP_ROPE_BACK:
            ggml_compute_forward_rope_back(params, tensor);
            break;
        case GGML_OP_CLAMP:
            ggml_compute_forward_clamp(params, tensor);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_compute_forward_conv_transpose_1d(params, tensor);
            break;
        case GGML_OP_IM2COL:
            ggml_compute_forward_im2col(params, tensor);
            break;
        case GGML_OP_IM2COL_BACK:
            ggml_compute_forward_im2col_back_f32(params, tensor);
            break;
        case GGML_OP_CONV_2D:
            ggml_compute_forward_conv_2d(params, tensor);
            break;
        case GGML_OP_CONV_2D_DW:
            ggml_compute_forward_conv_2d_dw(params, tensor);
            break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            ggml_compute_forward_conv_transpose_2d(params, tensor);
            break;
        case GGML_OP_POOL_1D:
            ggml_compute_forward_pool_1d(params, tensor);
            break;
        case GGML_OP_POOL_2D:
            ggml_compute_forward_pool_2d(params, tensor);
            break;
        case GGML_OP_POOL_2D_BACK:
            ggml_compute_forward_pool_2d_back(params, tensor);
            break;
        case GGML_OP_UPSCALE:
            ggml_compute_forward_upscale(params, tensor);
            break;
        case GGML_OP_PAD:
            ggml_compute_forward_pad(params, tensor);
            break;
        case GGML_OP_PAD_REFLECT_1D:
            ggml_compute_forward_pad_reflect_1d(params, tensor);
            break;
        case GGML_OP_ROLL:
            ggml_compute_forward_roll(params, tensor);
            break;
        case GGML_OP_ARANGE:
            ggml_compute_forward_arange(params, tensor);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_compute_forward_timestep_embedding(params, tensor);
            break;
        case GGML_OP_ARGSORT:
            ggml_compute_forward_argsort(params, tensor);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_compute_forward_leaky_relu(params, tensor);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_compute_forward_flash_attn_ext(params, tensor->src[0], tensor->src[1], tensor->src[2], tensor->src[3], tensor);
            break;
        case GGML_OP_FLASH_ATTN_BACK:
            ggml_compute_forward_flash_attn_back(params, false, tensor);
            break;
        case GGML_OP_SSM_CONV:
            ggml_compute_forward_ssm_conv(params, tensor);
            break;
        case GGML_OP_SSM_SCAN:
            ggml_compute_forward_ssm_scan(params, tensor);
            break;
        case GGML_OP_WIN_PART:
            ggml_compute_forward_win_part(params, tensor);
            break;
        case GGML_OP_WIN_UNPART:
            ggml_compute_forward_win_unpart(params, tensor);
            break;
        case GGML_OP_UNARY:
            ggml_compute_forward_unary(params, tensor);
            break;
        case GGML_OP_GLU:
            ggml_compute_forward_glu(params, tensor);
            break;
        case GGML_OP_GET_REL_POS:
            ggml_compute_forward_get_rel_pos(params, tensor);
            break;
        case GGML_OP_ADD_REL_POS:
            ggml_compute_forward_add_rel_pos(params, tensor);
            break;
        case GGML_OP_RWKV_WKV6:
            ggml_compute_forward_rwkv_wkv6(params, tensor);
            break;
        case GGML_OP_RWKV_WKV7:
            ggml_compute_forward_rwkv_wkv7(params, tensor);
            break;
        case GGML_OP_MAP_CUSTOM1:
            ggml_compute_forward_map_custom1(params, tensor);
            break;
        case GGML_OP_MAP_CUSTOM2:
            ggml_compute_forward_map_custom2(params, tensor);
            break;
        case GGML_OP_MAP_CUSTOM3:
            ggml_compute_forward_map_custom3(params, tensor);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_compute_forward_cross_entropy_loss(params, tensor);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_compute_forward_cross_entropy_loss_back(params, tensor);
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            ggml_compute_forward_opt_step_adamw(params, tensor);
            break;
        case GGML_OP_MUL_MAT:
            ggml_compute_forward_mul_mat(params, tensor);
            break;
        case GGML_OP_MUL_MAT_ID:
            // For MUL_MAT_ID, fall back to legacy approach since it's complex
            GGML_LOG_DEBUG("Direct Kernel: MUL_MAT_ID not supported in direct dispatch, falling back to legacy");
            return GGML_STATUS_FAILED;
            break;
        case GGML_OP_NONE:
            // No operation
            break;
        default:
            GGML_LOG_ERROR("Direct Kernel: Unsupported operation %s (%d), falling back to legacy approach", 
                          ggml_op_name(tensor->op), tensor->op);
            return GGML_STATUS_FAILED;
    }
    
    GGML_LOG_DEBUG("Direct Kernel: Operation %s completed successfully", ggml_op_name(tensor->op));
    return GGML_STATUS_SUCCESS;
}

// Kernel headers - using the new query interface
#include "numa-kernels/numa-kernels.h"  // New centralized query interface

// External test tracking function (has weak default implementation in ggml-cpu.c)
extern void test_track_data_parallel(void);

// Missing struct definition for MUL_MAT_ID work buffer calculation
struct mmid_row_mapping {
    int32_t i1; // i1
    int32_t i2; // i2  
};

// ============================================================================
// Core Executor Implementation  
// ============================================================================

/**
 * @brief Compute graph execution with NUMA-aware optimization
 * 
 * Processes a complete compute graph by analyzing each node and dispatching
 * to appropriate NUMA kernels or fallback mechanisms. This function provides
 * the main execution loop for NUMA-optimized computation.
 * 
 * Execution Flow:
 * 1. Validates input parameters and initializes kernel registry
 * 2. Iterates through all graph nodes in dependency order
 * 3. For each node, selects optimal execution strategy
 * 4. Delegates to NUMA kernels or fallback as appropriate
 * 5. Collects performance statistics and handles errors
 * 
 * @param cgraph The compute graph to execute
 * @param cplan The compute plan with threading and buffer information
 * @return GGML_STATUS_SUCCESS on success, error code on failure
 */
enum ggml_status ggml_numa_executor_execute_graph(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan) {
    if (!cgraph || !cplan) {
        return GGML_STATUS_FAILED;
    }
    
    GGML_LOG_DEBUG("NUMA Executor: Processing compute graph with %d nodes\n", cgraph->n_nodes);
    
    // Initialize kernel registry if not already done
    if (ggml_numa_kernels_init() != GGML_STATUS_SUCCESS) {
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

/**
 * @brief Main NUMA-aware compute graph execution entry point
 * 
 * This is the primary entry point for executing compute graphs with NUMA
 * optimization. It provides intelligent fallback behavior and comprehensive
 * error handling while maximizing performance on multi-socket systems.
 * 
 * Key Features:
 * - Automatic NUMA dispatch detection and fallback
 * - Per-node execution with optimal strategy selection
 * - Integration with standard ggml computation when NUMA unavailable
 * - Comprehensive logging and performance monitoring
 * - Graceful degradation for unsupported operations
 * 
 * @param cgraph The compute graph to execute
 * @param cplan The compute plan with threading and buffer information
 * @return GGML_STATUS_SUCCESS on success, error code on failure
 */
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

/**
 * @brief Execute a single tensor operation with optimal NUMA strategy
 * 
 * Core execution function that analyzes a single tensor operation and
 * selects the most appropriate execution path. This function serves as
 * the decision hub for NUMA optimization, routing operations to:
 * - NUMA-aware kernels for supported operations
 * - Direct kernel dispatch for maximum performance
 * - Fallback mechanisms for compatibility
 * 
 * Execution Strategy Selection:
 * 1. Query kernel registry for NUMA support
 * 2. Analyze operation complexity and resource requirements
 * 3. Select optimal execution strategy (NUMA vs fallback)
 * 4. Route to appropriate execution path
 * 5. Monitor performance and collect statistics
 * 
 * @param tensor The operation tensor to execute
 * @param cplan The compute plan with threading and buffer information
 * @return GGML_STATUS_SUCCESS on success, error code on failure
 */
enum ggml_status ggml_numa_executor_execute_tensor(struct ggml_tensor * tensor, struct ggml_cplan * cplan) {
    if (!tensor || !cplan) {
        NUMA_LOG_DEBUG("DEBUG: NUMA Executor: NULL tensor=%p or cplan=%p, returning FAILED\n", tensor, cplan);
        return GGML_STATUS_FAILED;
    }
    
    const char* op_name = ggml_op_name(tensor->op);
    size_t tensor_size = ggml_nbytes(tensor);
    
    NUMA_PERF_START(NUMA_PERF_OPERATION_TOTAL, op_name, "numa_executor", -1, tensor_size, cplan->n_threads);
    
    NUMA_LOG_DEBUG("DEBUG: NUMA Executor: Starting execution for %s, threads=%d\n", op_name, cplan->n_threads);
    
    // TRACE: Log complete tensor information for debugging inference issues
    NUMA_LOG_TRACE("EXECUTOR_TENSOR_START: op=%s tensor=%p size=%zu bytes elements=%ld threads=%d", 
                   op_name, (void*)tensor, tensor_size, ggml_nelements(tensor), cplan->n_threads);
    NUMA_LOG_TRACE("EXECUTOR_TENSOR_SHAPE: %s shape=[%ld,%ld,%ld,%ld] strides=[%zu,%zu,%zu,%zu]",
                   op_name, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
                   tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]);
    if (tensor->src[0]) {
        NUMA_LOG_TRACE("EXECUTOR_SRC0_SHAPE: src0=[%ld,%ld,%ld,%ld] elements=%ld",
                       tensor->src[0]->ne[0], tensor->src[0]->ne[1], tensor->src[0]->ne[2], tensor->src[0]->ne[3],
                       ggml_nelements(tensor->src[0]));
    }
    if (tensor->src[1]) {
        NUMA_LOG_TRACE("EXECUTOR_SRC1_SHAPE: src1=[%ld,%ld,%ld,%ld] elements=%ld",
                       tensor->src[1]->ne[0], tensor->src[1]->ne[1], tensor->src[1]->ne[2], tensor->src[1]->ne[3],
                       ggml_nelements(tensor->src[1]));
    }
    
    // Check NUMA environment
    #ifdef __linux__
    int current_cpu = sched_getcpu();
    int current_node = numa_node_of_cpu(current_cpu);
    int numa_nodes = numa_max_node() + 1;
    NUMA_LOG_DEBUG("DEBUG: NUMA Executor: Running on CPU %d (NUMA node %d), %d nodes available\n", 
           current_cpu, current_node, numa_nodes);
    #else
    NUMA_LOG_DEBUG("DEBUG: NUMA Executor: NUMA info not available (not Linux)\n");
    #endif
    
    // Get cache entry and query for execution strategy (hot path - must be fast)
    NUMA_PERF_START(NUMA_PERF_EXECUTOR_QUERY, op_name, "kernel_registry", -1, 0, 0);
    const ggml_numa_kernel_cache_entry_t * cache_entry = ggml_numa_lookup_kernel_direct(tensor->op);
    ggml_numa_execution_strategy_t strategy = ggml_numa_kernels_query(tensor);
    NUMA_PERF_END();
    
    if (!cache_entry || !cache_entry->supported) {
        GGML_LOG_DEBUG("NUMA Executor: Operation %s not supported by NUMA kernels, using direct kernel dispatch\n", 
                      op_name);
        enum ggml_status result = ggml_numa_executor_direct_kernel_dispatch(tensor, cplan);
        NUMA_PERF_END();
        return result;
    }
    
    // Check cache entry exists (we already got it above)
    if (!cache_entry) {
        GGML_LOG_DEBUG("NUMA Executor: No cache entry for %s, using direct kernel dispatch\n", op_name);
        enum ggml_status result = ggml_numa_executor_direct_kernel_dispatch(tensor, cplan);
        NUMA_PERF_END();
        return result;
    }
    
    // Get work function and metadata from cache
    ggml_numa_work_function_t work_function = ggml_numa_get_work_function_from_cache(cache_entry, &strategy);
    const char * kernel_name = ggml_numa_get_kernel_name_from_cache(cache_entry);
    
    // Debug logging is now handled by kernel query functions with standardized format
    // This eliminates duplicate logging and ensures consistent operation analysis parsing
    
    // Check if this is a no-op kernel that doesn't require coordinator dispatch
    if (ggml_numa_is_kernel_noop(tensor->op)) {
        NUMA_LOG_DEBUG("DEBUG: NUMA Executor: Operation %s is a no-op kernel, skipping coordinator dispatch\n", op_name);
        NUMA_PERF_END();
        return GGML_STATUS_SUCCESS;
    }
    
    GGML_LOG_DEBUG("NUMA Executor: %s kernel selected for %s (strategy=%s)\n",
                   kernel_name,
                   op_name,
                   (strategy == NUMA_STRATEGY_DATA_PARALLEL) ? "data-parallel" : "single-node");
    
    // Initialize OpenMP coordinator if needed
    NUMA_PERF_START(NUMA_PERF_COORDINATOR_INIT, op_name, kernel_name, -1, 0, cplan->n_threads);
    if (!ggml_numa_openmp_coordinator_init()) {
        NUMA_PERF_END();
        GGML_LOG_DEBUG("NUMA Executor: Failed to initialize OpenMP coordinator, using direct kernel dispatch for %s\n", 
                       op_name);
        enum ggml_status result = ggml_numa_executor_direct_kernel_dispatch(tensor, cplan);
        NUMA_PERF_END();
        return result;
    }
    NUMA_PERF_END();
    
    // Get OpenMP coordinator configuration
    ggml_numa_openmp_config_t config = ggml_numa_openmp_coordinator_get_config();
    
    // Execute using OpenMP coordinator - clean three-strategy model
    enum ggml_status result = GGML_STATUS_SUCCESS;
    
    int num_numa_nodes = config.total_numa_nodes;
    NUMA_LOG_DEBUG("DEBUG: NUMA Executor: num_numa_nodes=%d, strategy=%s\n", 
           num_numa_nodes,
           (strategy == NUMA_STRATEGY_DATA_PARALLEL) ? "data-parallel" : "single-node");
           
    if (strategy == NUMA_STRATEGY_DATA_PARALLEL && num_numa_nodes > 1) {
        // Multi-node data-parallel execution using OpenMP
        NUMA_PERF_START(NUMA_PERF_EXECUTOR_KERNEL_EXEC, op_name, kernel_name, -1, tensor_size, num_numa_nodes);
        NUMA_LOG_DEBUG("DEBUG: NUMA Executor: Taking DATA_PARALLEL path with %d nodes\n", num_numa_nodes);
        GGML_LOG_DEBUG("NUMA Executor: Dispatching %s for data-parallel execution across %d nodes\n", 
                       op_name, num_numa_nodes);
        
        // Track data-parallel execution for debugging
        test_track_data_parallel();
        
        // Calculate work buffer size for data-parallel execution
        int total_threads = num_numa_nodes * config.threads_per_node;
        
        // Update cplan to reflect actual coordinator thread count
        // The coordinator makes threading decisions, not the caller
        cplan->n_threads = total_threads;
        
        // Get work buffer calculation function from kernel cache entry
        ggml_numa_kernel_work_buffer_calc_fn_t work_buffer_calc_fn = 
            (cache_entry && cache_entry->work_buffer_calc_fn) ? cache_entry->work_buffer_calc_fn : NULL;
        
        result = ggml_numa_openmp_execute_data_parallel(
            tensor, 
            work_function,
            work_buffer_calc_fn);
            
        NUMA_LOG_DEBUG("DEBUG: NUMA Executor: Data-parallel execution result=%d\n", result);
        
        // TRACE: Log execution completion details for debugging
        NUMA_LOG_TRACE("EXECUTOR_DATA_PARALLEL_COMPLETE: op=%s result=%s tensor=%p elements=%ld nodes=%d total_threads=%d",
                       op_name, (result == GGML_STATUS_SUCCESS) ? "SUCCESS" : "FAILED", 
                       (void*)tensor, ggml_nelements(tensor), num_numa_nodes, total_threads);
        
        NUMA_PERF_END();
        
    } else {
        // Single-node execution - choose target node based on NUMA strategy and data locality
        enum ggml_numa_strategy numa_strategy = ggml_numa_get_strategy();
        int target_node = 0;  // Default to node 0
        
        if (numa_strategy == GGML_NUMA_STRATEGY_ISOLATE) {
            // For isolate mode, use the specified isolation node
            int isolate_node = ggml_numa_get_isolate_node();
            if (isolate_node >= 0) {
                target_node = isolate_node;
                NUMA_LOG_DEBUG("DEBUG: NUMA Executor: ISOLATE mode - using isolation node %d\n", target_node);
            } else {
                NUMA_LOG_DEBUG("DEBUG: NUMA Executor: ISOLATE mode - no valid isolation node, using default node 0\n");
            }
        } else {
            // For other strategies, detect optimal node based on data locality
            // Check where the source data is actually located
            if (tensor->src[0]) {
                extern int get_memory_numa_node(void* ptr);
                void* src_data = ggml_get_data(tensor->src[0]);
                if (src_data) {
                    int data_node = get_memory_numa_node(src_data);
                    if (data_node >= 0) {
                        target_node = data_node;
                        NUMA_LOG_DEBUG("DEBUG: NUMA Executor: Data locality detection - src0 data on node %d, using that node\n", target_node);
                    } else {
                        NUMA_LOG_DEBUG("DEBUG: NUMA Executor: Data locality detection failed, using default node 0\n");
                    }
                } else {
                    NUMA_LOG_DEBUG("DEBUG: NUMA Executor: No source data available, using default node 0\n");
                }
            } else {
                NUMA_LOG_DEBUG("DEBUG: NUMA Executor: No source tensor available, using default node 0\n");
            }
        }
        
        NUMA_PERF_START(NUMA_PERF_EXECUTOR_KERNEL_EXEC, op_name, kernel_name, target_node, tensor_size, 1);
        NUMA_LOG_DEBUG("DEBUG: NUMA Executor: Taking SINGLE_NODE path, target_node=%d (numa_strategy=%d)\n", target_node, numa_strategy);
        GGML_LOG_DEBUG("NUMA Executor: Dispatching %s for single-node execution on node %d\n", 
                       ggml_op_name(tensor->op), target_node);
        
        // Choose between single-thread and multi-thread execution based on strategy
        if (strategy == NUMA_STRATEGY_SINGLE_THREAD) {
            NUMA_LOG_DEBUG("DEBUG: NUMA Executor: Using single-thread execution\n");
            
            // Get work buffer calculation function from kernel cache entry
            ggml_numa_kernel_work_buffer_calc_fn_t work_buffer_calc_fn = 
                (cache_entry && cache_entry->work_buffer_calc_fn) ? cache_entry->work_buffer_calc_fn : NULL;
            
            result = ggml_numa_openmp_execute_single_thread(
                tensor, work_function, target_node, work_buffer_calc_fn);
        } else {
            NUMA_LOG_DEBUG("DEBUG: NUMA Executor: Using multi-thread execution\n");
            
            // Get work buffer calculation function from kernel cache entry
            ggml_numa_kernel_work_buffer_calc_fn_t work_buffer_calc_fn = 
                (cache_entry && cache_entry->work_buffer_calc_fn) ? cache_entry->work_buffer_calc_fn : NULL;
            
            result = ggml_numa_openmp_execute_single_node(
                tensor, work_function, target_node, work_buffer_calc_fn);
        }
        
        NUMA_LOG_DEBUG("DEBUG: NUMA Executor: Single-node execution result=%d\n", result);
        
        // TRACE: Log single-node execution completion details for debugging
        NUMA_LOG_TRACE("EXECUTOR_SINGLE_NODE_COMPLETE: op=%s result=%s tensor=%p elements=%ld target_node=%d threads=%d",
                       op_name, (result == GGML_STATUS_SUCCESS) ? "SUCCESS" : "FAILED", 
                       (void*)tensor, ggml_nelements(tensor), target_node,
                       (strategy == NUMA_STRATEGY_SINGLE_THREAD) ? 1 : config.threads_per_node);
        
        NUMA_PERF_END();
    }
    
    NUMA_LOG_DEBUG("DEBUG: NUMA Executor: Final result=%d for %s\n", result, op_name);
    if (result == GGML_STATUS_SUCCESS) {
        NUMA_LOG_DEBUG("DEBUG: NUMA Executor: SUCCESS - returning GGML_STATUS_SUCCESS\n");
        GGML_LOG_DEBUG("NUMA Executor: Successfully completed %s using %s\n", 
                       op_name, kernel_name);
    } else {
        NUMA_LOG_DEBUG("DEBUG: NUMA Executor: FAILURE - returning status %d\n", result);
        GGML_LOG_ERROR("NUMA Executor: Failed to execute %s using %s (status=%d)\n", 
                       op_name, kernel_name, (int)result);
    }
    
    NUMA_PERF_END();
    return result;
}

/**
 * @brief Execute a single tensor operation with forced NUMA strategy
 * 
 * This function is identical to ggml_numa_executor_execute_tensor() except it 
 * overrides the automatic strategy selection with a forced strategy. This is 
 * primarily used for testing to validate specific execution paths.
 */
enum ggml_status ggml_numa_executor_execute_tensor_forced_strategy(
    struct ggml_tensor * tensor,
    struct ggml_cplan * cplan,
    ggml_numa_execution_strategy_t forced_strategy) {
    
    // Start performance tracking
    NUMA_PERF_START(NUMA_PERF_EXECUTOR_KERNEL_EXEC, ggml_op_name(tensor->op), "forced_strategy", -1, 0, 0);
    
    const char* op_name = ggml_op_name(tensor->op);
    const char* strategy_str = (forced_strategy == NUMA_STRATEGY_SINGLE_THREAD) ? "single-thread" :
                              (forced_strategy == NUMA_STRATEGY_SINGLE_NODE) ? "single-node" : "data-parallel";
    NUMA_LOG_DEBUG("DEBUG: NUMA Executor (FORCED): Starting execution for %s with forced strategy %s\n", 
           op_name, strategy_str);
    
    // Initialize kernel registry if not already done
    if (ggml_numa_kernels_init() != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("NUMA Executor (FORCED): Failed to initialize kernel registry\n");
        NUMA_PERF_END();
        return GGML_STATUS_FAILED;
    }
    
    // TRACE: Very explicit forced strategy execution tracking
    NUMA_LOG_DEBUG("🔥 FORCED STRATEGY EXECUTION PATH: op=%s strategy=%d (%s)", 
                   op_name, (int)forced_strategy, strategy_str);
    
    // Query the kernel registry for execution information (but override strategy)
    NUMA_PERF_START(NUMA_PERF_EXECUTOR_QUERY, op_name, "kernel_registry", -1, 0, 0);
    // Get cache entry for this operation
    const ggml_numa_kernel_cache_entry_t * cache_entry = ggml_numa_lookup_kernel_direct(tensor->op);
    if (!cache_entry || !cache_entry->supported) {
        GGML_LOG_DEBUG("NUMA Executor (FORCED): Operation %s not supported by NUMA kernels, using direct kernel dispatch\n", 
                      op_name);
        enum ggml_status result = ggml_numa_executor_direct_kernel_dispatch(tensor, cplan);
        NUMA_PERF_END();
        return result;
    }
    
    // Get metadata from cache
    const char * kernel_name = ggml_numa_get_kernel_name_from_cache(cache_entry);
    ggml_numa_work_function_t work_function = ggml_numa_get_work_function_from_cache(cache_entry, &forced_strategy);
    
    // Check if this is a no-op kernel
    if (ggml_numa_is_kernel_noop(tensor->op)) {
        NUMA_LOG_DEBUG("DEBUG: NUMA Executor (FORCED): Operation %s is a no-op kernel, skipping coordinator dispatch\n", op_name);
        NUMA_PERF_END();
        return GGML_STATUS_SUCCESS;
    }
    
    GGML_LOG_DEBUG("NUMA Executor (FORCED): %s kernel selected for %s with forced strategy=%s\n",
                   kernel_name,
                   op_name,
                   (forced_strategy == NUMA_STRATEGY_DATA_PARALLEL) ? "data-parallel" : "single-node");
    
    enum ggml_status result = GGML_STATUS_SUCCESS;
    
    // Get NUMA topology information
    int num_numa_nodes = numa_max_node() + 1;
    
    NUMA_LOG_DEBUG("DEBUG: NUMA Executor (FORCED): num_numa_nodes=%d, strategy=%s\n", 
           num_numa_nodes,
           (forced_strategy == NUMA_STRATEGY_DATA_PARALLEL) ? "data-parallel" : "single-node");
    
    if (forced_strategy == NUMA_STRATEGY_DATA_PARALLEL && num_numa_nodes > 1) {
        // Data-parallel execution across multiple NUMA nodes
        NUMA_LOG_DEBUG("DEBUG: NUMA Executor (FORCED): Using data-parallel execution across %d nodes\n", num_numa_nodes);
        
        // TRACE: Very explicit data-parallel path tracking
        NUMA_LOG_DEBUG("🚀 DATA_PARALLEL_PATH_TAKEN: About to call ggml_numa_openmp_execute_data_parallel");
        
        NUMA_PERF_START(NUMA_PERF_COORDINATOR_DISPATCH, op_name, "data_parallel", num_numa_nodes, 0, 0);
        
        // Get work buffer calculation function from kernel cache entry
        ggml_numa_kernel_work_buffer_calc_fn_t work_buffer_calc_fn = 
            (cache_entry && cache_entry->work_buffer_calc_fn) ? cache_entry->work_buffer_calc_fn : NULL;
        
        result = ggml_numa_openmp_execute_data_parallel(
            tensor, work_function, work_buffer_calc_fn);
            
        NUMA_LOG_DEBUG("DEBUG: NUMA Executor (FORCED): Data-parallel execution result=%d\n", result);
        NUMA_PERF_END();
    } else {
        // Single-node execution (either forced or fallback from data-parallel)
        NUMA_PERF_START(NUMA_PERF_COORDINATOR_DISPATCH, op_name, "single_node", 1, 0, 0);
        
        // Choose target NUMA node (simplified for forced strategy - use node 0)
        int target_node = 0;
        
        // Get work buffer calculation function from kernel cache entry
        ggml_numa_kernel_work_buffer_calc_fn_t work_buffer_calc_fn = 
            (cache_entry && cache_entry->work_buffer_calc_fn) ? cache_entry->work_buffer_calc_fn : NULL;
        
        if (forced_strategy == NUMA_STRATEGY_SINGLE_THREAD) {
            NUMA_LOG_DEBUG("DEBUG: NUMA Executor (FORCED): Using single-thread execution on node %d\n", target_node);
            
            result = ggml_numa_openmp_execute_single_thread(tensor, work_function, target_node, work_buffer_calc_fn);
        } else {
            NUMA_LOG_DEBUG("DEBUG: NUMA Executor (FORCED): Using multi-thread execution on node %d\n", target_node);
            
            result = ggml_numa_openmp_execute_single_node(
                tensor, work_function, target_node, work_buffer_calc_fn);
        }
        
        NUMA_LOG_DEBUG("DEBUG: NUMA Executor (FORCED): Single-node execution result=%d\n", result);
        NUMA_PERF_END();
    }
    
    NUMA_LOG_DEBUG("DEBUG: NUMA Executor (FORCED): Final result=%d for %s\n", result, op_name);
    if (result == GGML_STATUS_SUCCESS) {
        NUMA_LOG_DEBUG("DEBUG: NUMA Executor (FORCED): SUCCESS - returning GGML_STATUS_SUCCESS\n");
        GGML_LOG_DEBUG("NUMA Executor (FORCED): Successfully completed %s using forced strategy\n", op_name);
    } else {
        NUMA_LOG_DEBUG("DEBUG: NUMA Executor (FORCED): FAILURE - returning status %d\n", result);
        GGML_LOG_ERROR("NUMA Executor (FORCED): Failed to execute %s using forced strategy (status=%d)\n", 
                       op_name, (int)result);
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

// Calculate work buffer size needed for a specific operation using kernel function
static size_t ggml_numa_calculate_work_size(struct ggml_tensor * tensor, int n_threads) {
    if (!tensor) {
        return 0;
    }
    
    NUMA_LOG_DEBUG("NUMA Work Buffer: Calculating for operation %s", ggml_op_name(tensor->op));
    
    // Try to get work buffer calculation from kernel
    const ggml_numa_kernel_cache_entry_t * entry = ggml_numa_lookup_kernel_direct(tensor->op);
    if (entry && entry->work_buffer_calc_fn) {
        // Use kernel's work buffer calculation function
        ggml_numa_kernel_work_buffer_calc_fn_t calc_fn = entry->work_buffer_calc_fn;
        // Get NUMA node count for work size calculation
        ggml_numa_openmp_config_t config = ggml_numa_openmp_coordinator_get_config();
        int numa_nodes = config.total_numa_nodes;
        size_t work_size = calc_fn(tensor, numa_nodes, n_threads);
        
        NUMA_LOG_DEBUG("NUMA Work Buffer: Kernel function calculated %zu bytes for %s", 
                       work_size, ggml_op_name(tensor->op));
        return work_size;
    }
    
    // Fallback to legacy switch statement for kernels without work buffer calculation
    size_t work_size = 0;
    
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
                NUMA_LOG_DEBUG("NUMA Work Buffer: CPY/DUP calculated size: %zu bytes", work_size);
            } break;
        case GGML_OP_MUL_MAT:
            {
                if (tensor->src[0] && tensor->src[1]) {
                    const struct ggml_type_traits_cpu * traits = ggml_get_type_traits_cpu(tensor->src[0]->type);
                    const enum ggml_type vec_dot_type = traits->vec_dot_type;
                    
                    if (tensor->src[1]->type != vec_dot_type) {
                        work_size = ggml_row_size(vec_dot_type, ggml_nelements(tensor->src[1]));
                    }
                    NUMA_LOG_DEBUG("NUMA Work Buffer: MUL_MAT src0_type=%d, src1_type=%d, vec_dot_type=%d, calculated size: %zu bytes", 
                                   (int)tensor->src[0]->type, (int)tensor->src[1]->type, (int)vec_dot_type, work_size);
                } else {
                    NUMA_LOG_WARN("NUMA Work Buffer: MUL_MAT missing source tensors");
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
        case GGML_OP_ROPE:
            {
                // ROPE needs work buffer for pre-computed cosine/sine cache per thread
                // Each thread needs cache for ne0 elements + cache line alignment
                // Based on ggml_compute_forward_rope_f32 and ggml_compute_forward_rope_f16
                const size_t cache_line_size_f32 = 16;  // CACHE_LINE_SIZE_F32 approximation
                const int64_t ne0 = tensor->ne[0];
                size_t cache_size_per_thread = (ne0 + cache_line_size_f32) * sizeof(float);
                work_size = cache_size_per_thread * n_threads;
                NUMA_LOG_DEBUG("NUMA Work Buffer: ROPE calculated size: %zu bytes for %d threads (ne0=%lld)", 
                               work_size, n_threads, (long long)ne0);
            } break;
        case GGML_OP_SOFT_MAX:
            {
                // SOFT_MAX needs work buffer for temporary float calculations
                // Based on ggml_compute_forward_soft_max_f32: wp = (float *) params->wdata + (ne00 + CACHE_LINE_SIZE_F32) * ith
                const size_t cache_line_size_f32 = 16;  // CACHE_LINE_SIZE_F32 approximation
                work_size = (tensor->ne[0] + cache_line_size_f32) * n_threads * sizeof(float);
                NUMA_LOG_DEBUG("NUMA Work Buffer: SOFT_MAX calculated size: %zu bytes for %d threads", work_size, n_threads);
            } break;
        default:
            // For unknown operations, use zero work size
            work_size = 0;
            break;
    }
    
    NUMA_LOG_DEBUG("NUMA Work Buffer: Legacy calculation for %s: %zu bytes", 
                   ggml_op_name(tensor->op), work_size);
    return work_size;
}

/**
 * @brief Direct Fallback kernel dispatch function for maximum performance
 * 
 * High-performance execution path that eliminates temporary graph overhead
 * by calling compute functions directly. This function provides the fastest
 * possible execution for fallback operations in ggml-cpu.c
 * 
 * Performance Benefits:
 * - Zero-copy operation without graph allocation
 * - Direct function call without dispatch overhead
 * 
 * Use Cases:
 * - Fallback path when no NUMA kernel exists for an op
 * 
 * @param tensor The operation tensor to execute
 * @param cplan The compute plan with threading and buffer information
 * @return GGML_STATUS_SUCCESS on success, error code on failure
 */
enum ggml_status ggml_numa_executor_direct_kernel_dispatch(struct ggml_tensor * tensor, struct ggml_cplan * cplan) {
    if (!tensor || !cplan) {
        return GGML_STATUS_FAILED;
    }
    
    const char* op_name = ggml_op_name(tensor->op);
    size_t tensor_size = ggml_nbytes(tensor);
    
    NUMA_PERF_START(NUMA_PERF_EXECUTOR_FALLBACK, op_name, "direct_kernel_dispatch", -1, tensor_size, cplan->n_threads);
    
    GGML_LOG_DEBUG("NUMA Direct Fallback Kernel Dispatch: Starting for operation %s\n", op_name);
    
    // Set flag to disable NUMA dispatch during this call (prevents infinite recursion)
    ggml_numa_set_fallback_flag(true);
    
    // Calculate work buffer size needed for this specific operation
    size_t needed_work_size = ggml_numa_calculate_work_size(tensor, cplan->n_threads);
    
    void * work_data = NULL;
    
    // Check if existing work buffer is sufficient
    if (cplan->work_data && cplan->work_size >= needed_work_size) {
        work_data = cplan->work_data;
        GGML_LOG_DEBUG("NUMA Direct Kernel: Using existing work buffer (%zu bytes >= %zu needed)\n", 
                       cplan->work_size, needed_work_size);
    } else if (needed_work_size > 0) {
        // Allocate temporary work buffer for fallback operations
        work_data = malloc(needed_work_size);
        if (work_data) {
            GGML_LOG_DEBUG("NUMA Direct Kernel: Allocated temporary work buffer (%zu bytes)\n", needed_work_size);
        } else {
            GGML_LOG_DEBUG("NUMA Direct Kernel: Failed to allocate work buffer (%zu bytes) - continuing without\n", needed_work_size);
            // Continue without work buffer - some operations can work without it
        }
    } else {
        GGML_LOG_DEBUG("NUMA Direct Kernel: No work buffer needed (%zu bytes)\n", needed_work_size);
    }

    // TODO: Implement threadpool fallback for OpenMP coordinator
    struct ggml_threadpool * fallback_threadpool = NULL;
    int fallback_thread_count = cplan->n_threads; // Default to original plan's thread count
    
    // Try to get the dedicated fallback threadpool from coordinator
    fallback_threadpool = ggml_numa_openmp_get_fallback_threadpool();
    if (fallback_threadpool) {
        // Use the fallback threadpool's actual thread count
        fallback_thread_count = ggml_numa_openmp_get_fallback_thread_count();
        
        GGML_LOG_DEBUG("🚀 Using dedicated fallback threadpool: %p (bound to NUMA node 0)\n", (void*)fallback_threadpool);
        GGML_LOG_DEBUG("📊 Direct Kernel Execution: threads=%d (fallback capacity), threadpool=%p\n", 
                       fallback_thread_count, (void*)fallback_threadpool);
    } else if (cplan->threadpool) {
        fallback_threadpool = cplan->threadpool;
        GGML_LOG_DEBUG("NUMA Direct Kernel: Using existing threadpool %p with %d thread(s)\n", (void*)fallback_threadpool, fallback_thread_count);
    } else {
        GGML_LOG_ERROR("NUMA Direct Kernel: No threadpool available - this should not happen with OpenMP coordinator\n");
        return GGML_STATUS_FAILED;
    }
    
    // Set up compute params for direct kernel execution - use single thread with fallback threadpool
    struct ggml_compute_params params = {
        .ith = 0,
        .nth = 1, // Single-threaded fallback execution 
        .wsize = needed_work_size,
        .wdata = work_data,
        .threadpool = fallback_threadpool  // Must provide valid threadpool for barrier operations
    };
    
    GGML_LOG_INFO("🚀 NUMA Direct Fallback Kernel Dispatch: Executing operation %s (work_size=%zu, threads=%d, threadpool=%p)\n", 
                   ggml_op_name(tensor->op), needed_work_size, params.nth, (void*)params.threadpool);
    
    // OPTIMIZATION: Direct kernel dispatch - call the operation's compute function directly
    // This eliminates temporary graph creation, temporary compute plan creation, and graph computation pipeline overhead
    enum ggml_status result = ggml_numa_executor_call_direct_kernel(tensor, &params);
    
    // Clean up temporary work buffer if we allocated one
    if (work_data && work_data != cplan->work_data) {
        free(work_data);
        GGML_LOG_DEBUG("NUMA Direct Kernel: Freed temporary work buffer\n");
    }
    
    // Clear flag after computation
    ggml_numa_set_fallback_flag(false);
    
    // Check result
    if (result != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("NUMA Direct Fallback Kernel Dispatch: Kernel execution failed with status %d\n", result);
        NUMA_PERF_END();
        return GGML_STATUS_FAILED;
    }

    GGML_LOG_DEBUG("NUMA Direct Fallback Kernel Dispatch: Operation %s completed successfully\n", op_name);
    NUMA_PERF_END();
    return GGML_STATUS_SUCCESS;
}
