/**
 * @file cpy.c
 * @brief NUMA-aware CPY (copy/duplicate) operation kernel implementation
 * 
 * This kernel implements efficient tensor copying with NUMA-aware memory access patterns.
 * The CPY operation copies data from source to destination tensor, potentially with
 * type conversion and different memory layouts.
 * 
 * Key Implementation Details:
 * - Delegates to optimized ggml_compute_forward_dup for mathematical correctness
 * - Uses NUMA-aware memory allocation and thread distribution
 * - Supports all tensor types and layouts supported by reference implementation
 * - Strategy selection based on tensor size and memory characteristics
 * - Data-parallel execution for large tensors across NUMA nodes
 * 
 * Performance Characteristics:
 * - Memory bandwidth limited, benefits from NUMA locality
 * - Multi-threading effective for large tensors
 * - Type conversion overhead handled by optimized reference kernels
 */

#include "cpy.h"
#include "../ggml-impl.h"
#include "../ggml-numa-shared.h"
#include "../ops.h"
#include "../vec.h"
#include <assert.h>
#include <string.h>

/**
 * @brief Execute CPY operation using NUMA-aware patterns
 * 
 * This function performs tensor copying by delegating to the optimized
 * ggml_compute_forward_dup implementation while ensuring NUMA-aware
 * memory access patterns and optimal thread distribution.
 * 
 * The CPY operation is mathematically straightforward but performance-critical:
 * - For contiguous same-type tensors: optimized memcpy
 * - For type conversion: element-wise conversion with SIMD
 * - For strided tensors: stride-aware copying with proper indexing
 * 
 * NUMA Strategy:
 * - Small tensors: single-thread execution for minimal overhead
 * - Medium tensors: multi-thread single-node for cache efficiency  
 * - Large tensors: data-parallel across NUMA nodes for memory bandwidth
 */
enum ggml_status ggml_numa_kernel_cpy_execute(void * work_context, struct ggml_compute_params * params) {
    struct ggml_tensor * tensor = (struct ggml_tensor *)work_context;
    
    // Validate inputs
    NUMA_ASSERT(tensor != NULL, "Tensor cannot be null");
    NUMA_ASSERT(params != NULL, "Compute params cannot be null");
    NUMA_ASSERT(tensor->src[0] != NULL, "Source tensor cannot be null");
    
    // Get source tensor characteristics
    const struct ggml_tensor * src = tensor->src[0];
    
    // Validate tensor compatibility
    NUMA_ASSERT(ggml_nelements(tensor) == ggml_nelements(src), 
                "Destination and source must have same element count");
    
    // Log execution details for debugging
    NUMA_LOG_DEBUG("CPY execute: src_type=%s, dst_type=%s, elements=%zu", 
                   ggml_type_name(src->type), ggml_type_name(tensor->type), 
                   ggml_nelements(tensor));
    
    // Get NUMA execution context from thread-local variables
    extern __thread int ggml_current_numa_node;
    extern __thread int ggml_numa_total_nodes_for_data_parallel; 
    extern __thread bool ggml_numa_is_data_parallel_execution;
    
    // Log execution strategy for integration test parsing
    if (ggml_numa_is_data_parallel_execution) {
        NUMA_LOG_STRATEGY_DATA_PARALLEL("CPY");
    } else if (params->nth > 1) {
        NUMA_LOG_STRATEGY_SINGLE_MULTI("CPY");
    } else {
        NUMA_LOG_STRATEGY_SINGLE_SINGLE("CPY");
    }
    
    NUMA_LOG_TRACE("CPY NUMA context: node=%d, total_nodes=%d, data_parallel=%s",
                    ggml_current_numa_node, ggml_numa_total_nodes_for_data_parallel,
                    ggml_numa_is_data_parallel_execution ? "true" : "false");
    
    // CPY operation is implemented as ggml_compute_forward_dup in the reference
    // implementation. This provides mathematically correct results for all
    // type combinations and tensor layouts.
    //
    // The reference implementation includes:
    // - Optimized same-type contiguous copying (memcpy)
    // - Type conversion with SIMD optimization where possible
    // - Proper stride handling for non-contiguous tensors
    // - Multi-threading support for large tensors
    //
    // By delegating to the reference implementation, we ensure mathematical
    // correctness while the NUMA executor provides optimal memory allocation
    // and thread distribution across NUMA nodes.
    ggml_compute_forward_dup(params, tensor);
    
    NUMA_LOG_TRACE("CPY execute completed successfully");
    
    return GGML_STATUS_SUCCESS;
}

/**
 * @brief Query CPY operation for optimal execution strategy
 * 
 * Analyzes tensor characteristics to determine the most efficient execution strategy.
 * CPY operations have different performance characteristics based on:
 * - Tensor size (memory bandwidth vs. setup overhead)
 * - Type conversion requirements (same-type vs. different types)
 * - Memory layout (contiguous vs. strided access patterns)
 * - Available NUMA nodes and memory bandwidth
 */
ggml_numa_kernel_query_result_t ggml_numa_kernel_cpy_query(const struct ggml_tensor * tensor) {
    ggml_numa_kernel_query_result_t result = {0};
    
    // Validate tensor for CPY operation
    if (!tensor || tensor->op != GGML_OP_CPY) {
        NUMA_LOG_DEBUG("CPY query: REJECTING - invalid tensor or operation");
        result.supported = false;
        return result;
    }
    
    const struct ggml_tensor * src = tensor->src[0];
    if (!src) {
        NUMA_LOG_DEBUG("CPY query: REJECTING - missing source tensor");
        result.supported = false;
        return result;
    }
    
    // Get tensor characteristics
    const size_t total_elements = ggml_nelements(tensor);
    const size_t src_element_size = ggml_type_size(src->type);
    const size_t dst_element_size = ggml_type_size(tensor->type);
    const size_t total_bytes = total_elements * MAX(src_element_size, dst_element_size);
    
    // Check if tensors are contiguous (affects performance characteristics)
    const bool src_contiguous = ggml_is_contiguous(src);
    const bool dst_contiguous = ggml_is_contiguous(tensor);
    const bool same_type = (src->type == tensor->type);
    
    // Define strategy thresholds optimized for memory copying operations
    // These thresholds are tuned for memory bandwidth vs. thread overhead
    const size_t SINGLE_THREAD_THRESHOLD = 8192;     // 8K elements - minimal overhead
    const size_t MULTI_THREAD_THRESHOLD = 262144;    // 256K elements - single-node multi-thread
    // Above 256K elements: data-parallel across NUMA nodes for memory bandwidth
    
    // Determine optimal strategy based on tensor characteristics
    ggml_numa_execution_strategy_t strategy;
    float efficiency_score;
    
    if (total_elements <= SINGLE_THREAD_THRESHOLD) {
        // Small tensors: single-thread execution for minimal overhead
        strategy = (ggml_numa_execution_strategy_t){
            .node_strategy = NUMA_NODE_STRATEGY_SINGLE,
            .on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD
        };
        efficiency_score = 0.80f; // Good for small tensors, minimal overhead
    } else if (total_elements <= MULTI_THREAD_THRESHOLD) {
        // Medium tensors: multi-thread single-node for cache efficiency
        strategy = (ggml_numa_execution_strategy_t){
            .node_strategy = NUMA_NODE_STRATEGY_SINGLE,
            .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
        };
        efficiency_score = 0.90f; // High efficiency with good parallelization
    } else {
        // Large tensors: data-parallel across NUMA nodes for memory bandwidth
        strategy = (ggml_numa_execution_strategy_t){
            .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL,
            .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
        };
        efficiency_score = 0.95f; // Highest efficiency for memory-bound operations
    }
    
    // Adjust efficiency based on operation characteristics
    if (!same_type) {
        // Type conversion adds computational overhead
        efficiency_score *= 0.95f;
    }
    if (!src_contiguous || !dst_contiguous) {
        // Non-contiguous access reduces memory bandwidth efficiency
        efficiency_score *= 0.90f;
    }
    
    // CPY operation supports all tensor types that ggml_compute_forward_dup supports
    bool type_supported = true; // ggml_compute_forward_dup handles all types
    
    if (!type_supported) {
        NUMA_LOG_DEBUG("CPY query: REJECTING - unsupported tensor types");
        result.supported = false;
        return result;
    }
    
    // Accept the operation
    NUMA_LOG_DEBUG("CPY query: ACCEPTING - src_type=%s, dst_type=%s, elements=%zu", 
                   ggml_type_name(src->type), ggml_type_name(tensor->type), total_elements);
    
    // Build successful query result
    result.supported = true;
    result.strategy = strategy;
    result.work_buffer_size_per_thread = 64; // Minimal buffer for coordination
    result.work_function = ggml_numa_kernel_cpy_execute;
    result.efficiency_score = efficiency_score;
    result.kernel_name = "NUMA CPY Kernel";
    
    // CPY operations don't need aggregation (direct copying)
    result.aggregation_policy = GGML_NUMA_AGGREGATION_NONE;
    result.aggregation_function = NULL;
    
    NUMA_LOG_DEBUG("CPY query: elements=%zu, bytes=%zu, same_type=%s, contiguous=%s/%s, strategy=%d, efficiency=%.2f",
                   total_elements, total_bytes, same_type ? "yes" : "no",
                   src_contiguous ? "yes" : "no", dst_contiguous ? "yes" : "no",
                   (int)strategy.node_strategy, (double)efficiency_score);
    
    return result;
}

/**
 * @brief Register CPY kernel with NUMA system
 * 
 * Provides registration information for the CPY kernel including strategy
 * thresholds, function pointers, and operation characteristics.
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_cpy_register(void) {
    ggml_numa_kernel_registration_info_t info = {0};
    
    info.op_type = GGML_OP_CPY;
    info.supported = true;
    info.kernel_name = "NUMA CPY Kernel";
    
    // Strategy thresholds optimized for memory copying operations
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_SINGLE] = 128;     // Single-thread strategy
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_MULTI] = 1024;    // Multi-thread strategy
    // Above this: data-parallel strategy
    info.strategy_array.valid = true;
    
    // Function pointers for different execution strategies
    info.work_funcs.single_single_fn = ggml_numa_kernel_cpy_execute;
    info.work_funcs.single_multi_fn = ggml_numa_kernel_cpy_execute;
    info.work_funcs.data_parallel_fn = ggml_numa_kernel_cpy_execute;
    info.work_funcs.valid = true;
    
    // Query function pointer for direct dispatch
    info.query_fn = (void*)ggml_numa_kernel_cpy_query;
    
    // CPY operations don't need aggregation functions
    info.agg_funcs.single_single_fn = NULL;
    info.agg_funcs.single_multi_fn = NULL;
    info.agg_funcs.data_parallel_fn = NULL;
    info.agg_funcs.valid = false;
    
    return info;
}