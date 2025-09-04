/**
 * @file get_rows.c
 * @brief NUMA-aware GET_ROWS kernel implementation for tensor row extraction operations
 * 
 * GET_ROWS Operation Implementation:
 * - Extracts specified rows from source tensor based on index tensor
 * - Supports all quantization types (Q4_0, Q5_0, Q8_0, etc.) and F32/F16
 * - Implements efficient row-wise parallelization with NUMA awareness
 * - Uses SIMD-optimized memory copying for maximum performance
 * 
 * NUMA Strategy Implementation:
 * - Small operations: Single-thread execution for minimal overhead
 * - Medium operations: Multi-thread single-node for balanced performance
 * - Large operations: Data-parallel across NUMA nodes for maximum throughput
 * 
 * Memory Access Patterns:
 * - Sequential access to index tensor (src1) for cache efficiency
 * - Random access to source tensor (src0) based on row indices
 * - Sequential writes to destination tensor with proper NUMA placement
 * - Efficient bounds checking and error handling for index validation
 * 
 * Performance Characteristics:
 * - Memory bandwidth limited operation
 * - Benefits significantly from NUMA-local memory access
 * - Scales well with multi-threading for large tensor operations
 * - SIMD acceleration for memory copying operations
 * 
 * @author David Sanftenberg
 * @date 2025
 */

#include "get_rows.h"
#include "numa-kernels.h"
#include "ggml-numa-shared.h"
#include "ggml-cpu.h"
#include "ops.h"
#include "vec.h"

#include <assert.h>
#include <string.h>

/**
 * @brief Execute NUMA-aware GET_ROWS operation with optimal performance
 * 
 * This function implements the GET_ROWS operation with NUMA optimizations for
 * memory bandwidth and thread distribution. It delegates to the optimized
 * ggml_compute_forward_get_rows implementation while providing NUMA-aware
 * execution context and memory access patterns.
 * 
 * Operation Details:
 * - Extracts rows from src0 tensor based on indices in src1 tensor
 * - Maintains exact element values during row extraction
 * - Supports all quantization types and data formats
 * - Implements efficient row-wise work distribution for parallel execution
 * 
 * NUMA Optimizations:
 * - Thread-local memory access patterns for cache efficiency
 * - Optimal work distribution across NUMA nodes for large tensors
 * - Memory bandwidth optimization through NUMA-local data placement
 * - SIMD acceleration for memory copying operations
 * 
 * @param work_context Pointer to destination tensor for GET_ROWS operation
 * @param params Compute parameters including thread configuration and NUMA context
 * @return GGML_STATUS_SUCCESS on successful execution, error status otherwise
 * 
 * @note The function validates input tensors and delegates to the appropriate
 *       type-specific implementation in ggml_compute_forward_get_rows.
 */
enum ggml_status ggml_numa_kernel_get_rows_execute(void * work_context, struct ggml_compute_params * params) {
    // Validate input parameters
    NUMA_ASSERT(work_context != NULL, "Work context cannot be null");
    NUMA_ASSERT(params != NULL, "Compute params cannot be null");
    
    struct ggml_tensor * tensor = (struct ggml_tensor *)work_context;
    NUMA_ASSERT(tensor != NULL, "Tensor cannot be null");
    NUMA_ASSERT(tensor->src[0] != NULL, "Source tensor (src0) cannot be null");
    NUMA_ASSERT(tensor->src[1] != NULL, "Index tensor (src1) cannot be null");
    
    // Get NUMA execution context from thread-local variables
    extern __thread int ggml_current_numa_node;
    extern __thread bool ggml_numa_is_data_parallel_execution;
    
    // Log execution details for debugging
    const struct ggml_tensor * src0 = tensor->src[0]; // Data tensor
    const struct ggml_tensor * src1 = tensor->src[1]; // Index tensor
    
    const int64_t ne00 = src0->ne[0]; // Source tensor row size
    const int64_t nr = ggml_nelements(src1); // Number of rows to extract
    
    // Log execution strategy in standardized format for integration test parsing
    if (ggml_numa_is_data_parallel_execution) {
        NUMA_LOG_STRATEGY_DATA_PARALLEL("GET_ROWS");
    } else if (params->nth > 1) {
        NUMA_LOG_STRATEGY_SINGLE_MULTI("GET_ROWS");
    } else {
        NUMA_LOG_STRATEGY_SINGLE_SINGLE("GET_ROWS");
    }
    
    NUMA_LOG_DEBUG("GET_ROWS execution: src0=[%ld,%ld,%ld,%ld], src1=[%ld,%ld,%ld,%ld], "
                   "nr=%ld, row_size=%ld, numa_node=%d, data_parallel=%s",
                   (long)src0->ne[0], (long)src0->ne[1], (long)src0->ne[2], (long)src0->ne[3],
                   (long)src1->ne[0], (long)src1->ne[1], (long)src1->ne[2], (long)src1->ne[3],
                   (long)nr, (long)ne00, ggml_current_numa_node, 
                   ggml_numa_is_data_parallel_execution ? "true" : "false");
    
    // Validate tensor dimensions for GET_ROWS operation
    NUMA_ASSERT(tensor->ne[0] == ne00, "Output tensor row size must match source tensor row size");
    NUMA_ASSERT(ggml_nrows(tensor) == nr, "Output tensor rows must match number of indices");
    
    // GET_ROWS is implemented as ggml_compute_forward_get_rows in the original code
    // This function handles all quantization types and optimizations including:
    // - Quantized types: Q4_0, Q5_0, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, etc.
    // - Float types: F32, F16, BF16
    // - Integer types: I32
    // - SIMD-optimized memory copying with ggml_vec_cpy_* functions
    // - Efficient row indexing and bounds checking
    
    // The NUMA optimization comes from:
    // 1. Thread distribution managed by NUMA coordinator
    // 2. Memory access patterns optimized for NUMA topology
    // 3. Work distribution across NUMA nodes for large tensors
    
    ggml_compute_forward_get_rows(params, tensor);
    
    NUMA_LOG_TRACE("GET_ROWS completed successfully on NUMA node %d", ggml_current_numa_node);
    
    return GGML_STATUS_SUCCESS;
}

/**
 * @brief Query optimal NUMA strategy for GET_ROWS operation
 * 
 * Analyzes the GET_ROWS operation characteristics to determine the most efficient
 * execution strategy based on tensor dimensions, data types, and system resources.
 * 
 * Strategy Selection Logic:
 * - Small operations (< 4K rows): Single-thread for minimal overhead
 * - Medium operations (4K-128K rows): Multi-thread single-node for balanced performance
 * - Large operations (> 128K rows): Data-parallel across NUMA nodes for maximum throughput
 * 
 * The analysis considers:
 * - Number of rows to extract (primary factor)
 * - Row size and memory bandwidth requirements
 * - Data type and quantization overhead
 * - NUMA topology and thread availability
 * - Memory access patterns (sequential index reads, random data reads)
 * 
 * @param tensor Destination tensor for strategy analysis
 * @return Query result containing optimal strategy, efficiency score, and resource requirements
 */
ggml_numa_kernel_query_result_t ggml_numa_kernel_get_rows_query(const struct ggml_tensor * tensor) {
    ggml_numa_kernel_query_result_t result = {0};
    
    // Validate tensor for GET_ROWS operation
    if (!tensor || tensor->op != GGML_OP_GET_ROWS) {
        NUMA_LOG_DEBUG("GET_ROWS query: REJECTING - invalid tensor or operation");
        result.supported = false;
        return result;
    }
    
    const struct ggml_tensor * src0 = tensor->src[0]; // Data tensor
    const struct ggml_tensor * src1 = tensor->src[1]; // Index tensor
    
    if (!src0 || !src1) {
        NUMA_LOG_DEBUG("GET_ROWS query: REJECTING - missing source or index tensor");
        result.supported = false;
        return result;
    }
    
    // Get tensor characteristics for GET_ROWS operation
    const int64_t ne00 = src0->ne[0]; // Source tensor row size
    const int64_t nr = ggml_nelements(src1); // Number of rows to extract
    
    // Calculate memory requirements for the operation
    const size_t element_size = ggml_type_size(src0->type);
    const size_t row_size = ne00 * element_size;
    
    // Define strategy thresholds optimized for row extraction operations
    const size_t SINGLE_THREAD_THRESHOLD = 4096;      // 4K rows
    const size_t MULTI_THREAD_THRESHOLD = 131072;     // 128K rows
    
    // Determine optimal strategy based on number of rows to extract
    ggml_numa_execution_strategy_t strategy;
    float efficiency_score;
    
    if (nr <= SINGLE_THREAD_THRESHOLD) {
        // Small operations: Single-thread execution for minimal overhead
        strategy = (ggml_numa_execution_strategy_t){
            .node_strategy = NUMA_NODE_STRATEGY_SINGLE,
            .on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD
        };
        efficiency_score = 0.85f; // Good efficiency, minimal overhead
    } else if (nr <= MULTI_THREAD_THRESHOLD) {
        // Medium operations: Multi-thread single-node for balanced performance
        strategy = (ggml_numa_execution_strategy_t){
            .node_strategy = NUMA_NODE_STRATEGY_SINGLE,
            .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
        };
        efficiency_score = 0.90f; // High efficiency with good parallelization
    } else {
        // Large operations: Data-parallel across NUMA nodes for maximum throughput
        strategy = (ggml_numa_execution_strategy_t){
            .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL,
            .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
        };
        efficiency_score = 0.95f; // Highest efficiency for large-scale operations
    }
    
    // Support all data types that ggml_compute_forward_get_rows supports
    enum ggml_type src_type = src0->type;
    bool type_supported = (
        // Quantized types
        src_type == GGML_TYPE_Q4_0 || src_type == GGML_TYPE_Q4_1 ||
        src_type == GGML_TYPE_Q5_0 || src_type == GGML_TYPE_Q5_1 ||
        src_type == GGML_TYPE_Q8_0 || src_type == GGML_TYPE_Q8_1 ||
        src_type == GGML_TYPE_Q2_K || src_type == GGML_TYPE_Q3_K ||
        src_type == GGML_TYPE_Q4_K || src_type == GGML_TYPE_Q5_K ||
        src_type == GGML_TYPE_Q6_K || src_type == GGML_TYPE_TQ1_0 ||
        src_type == GGML_TYPE_TQ2_0 || src_type == GGML_TYPE_IQ2_XXS ||
        src_type == GGML_TYPE_IQ2_XS || src_type == GGML_TYPE_IQ3_XXS ||
        src_type == GGML_TYPE_IQ1_S || src_type == GGML_TYPE_IQ1_M ||
        src_type == GGML_TYPE_IQ4_NL || src_type == GGML_TYPE_IQ4_XS ||
        src_type == GGML_TYPE_IQ3_S || src_type == GGML_TYPE_IQ2_S ||
        // Float types
        src_type == GGML_TYPE_F16 || src_type == GGML_TYPE_BF16 ||
        src_type == GGML_TYPE_F32 || src_type == GGML_TYPE_I32
    );
    
    if (!type_supported) {
        NUMA_LOG_DEBUG("GET_ROWS query: REJECTING - unsupported data type %d", (int)src_type);
        result.supported = false;
        return result;
    }
    
    // Accept all supported data types
    NUMA_LOG_DEBUG("GET_ROWS query: ACCEPTING - src0_type=%s", ggml_type_name(src0->type));
    
    // Build successful query result
    result.supported = true;
    result.strategy = strategy;
    result.work_buffer_size_per_thread = 64;  // Minimal buffer for coordination
    result.work_function = ggml_numa_kernel_get_rows_execute;
    result.efficiency_score = efficiency_score;
    result.kernel_name = "NUMA GET_ROWS Kernel";
    
    // No aggregation needed for GET_ROWS (direct row copying)
    result.aggregation_policy = GGML_NUMA_AGGREGATION_NONE;
    result.aggregation_function = NULL;
    
    NUMA_LOG_DEBUG("GET_ROWS query: nr=%ld, row_size=%zu, node_strategy=%d, efficiency=%.2f",
                   (long)nr, row_size, (int)strategy.node_strategy, (double)efficiency_score);
    
    return result;
}

/**
 * @brief Register GET_ROWS kernel with NUMA system
 * 
 * Provides comprehensive registration information for the GET_ROWS kernel including
 * operation type, strategy thresholds, function pointers, and kernel metadata.
 * 
 * Registration Configuration:
 * - Operation type: GGML_OP_GET_ROWS
 * - Strategy thresholds: 4K/128K rows for optimal strategy selection
 * - Work functions: Execute function for all three strategies
 * - Aggregation functions: Not required (no result aggregation needed)
 * - Query function: Direct dispatch query function pointer
 * 
 * The registration enables:
 * - O(1) kernel lookup and dispatch
 * - Automatic strategy selection based on tensor characteristics
 * - Direct function pointer dispatch without switch statements
 * - Seamless integration with NUMA executor and coordinator
 * 
 * @return Registration information structure for GET_ROWS kernel
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_get_rows_register(void) {
    ggml_numa_kernel_registration_info_t info = {0};
    
    // Set operation type and basic information
    info.op_type = GGML_OP_GET_ROWS;
    info.supported = true;
    info.kernel_name = "NUMA GET_ROWS Kernel";
    
    // Configure strategy thresholds optimized for row extraction operations
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_SINGLE] = 9999999;      // Single-thread strategy
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_MULTI] = 9999999; //TODO: remove     // Multi-thread strategy
    // Above this: data-parallel strategy
    info.strategy_array.valid = true;
    
    // Set work function pointers (same function handles all strategies)
    info.work_funcs.single_single_fn = ggml_numa_kernel_get_rows_execute;
    info.work_funcs.single_multi_fn = ggml_numa_kernel_get_rows_execute;
    info.work_funcs.data_parallel_fn = ggml_numa_kernel_get_rows_execute;
    info.work_funcs.valid = true;
    
    // GET_ROWS doesn't require aggregation functions (direct row extraction)
    info.agg_funcs.single_single_fn = NULL;
    info.agg_funcs.single_multi_fn = NULL;
    info.agg_funcs.data_parallel_fn = NULL;
    info.agg_funcs.valid = false;
    
    // Set query function pointer for direct dispatch (O(1) lookup)
    info.query_fn = (void*)ggml_numa_kernel_get_rows_query;
    
    return info;
}
