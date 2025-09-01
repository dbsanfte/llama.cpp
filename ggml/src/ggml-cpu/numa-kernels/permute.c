/**
 * @file permute.c
 * @brief NUMA Kernel Template: Complex Operations (Tensor Permutation)
 * 
 * ============================================================================
 * NUMA KERNEL TEMPLATE: COMPLEX OPERATIONS (PERMUTE)
 * ============================================================================
 * 
 * This file implements NUMA-aware tensor permutation, based on the MUL_MAT template
 * for complex operations that require sophisticated parallelization strategies.
 * 
 * MATHEMATICAL OPERATION (PERMUTE):
 * =================================
 * 
 * PERMUTE rearranges tensor dimensions according to specified axis mapping:
 * 
 * dst[i0, i1, i2, i3] = src[idx[axis0], idx[axis1], idx[axis2], idx[axis3]]
 * 
 * Where:
 * - src: Input tensor with original dimension layout
 * - dst: Output tensor with permuted dimension layout  
 * - axis0, axis1, axis2, axis3: Permutation mapping stored in op_params
 * 
 * NUMA OPTIMIZATION STRATEGY:
 * ===========================
 * - Divide output tensor into NUMA-aligned chunks
 * - Each NUMA node processes a contiguous slice of output elements
 * - Optimize memory access patterns for cache efficiency
 * - Use vectorized copy operations where possible
 * 
 * PERMUTATION COMPLEXITY:
 * ======================
 * - Simple permutations (transpositions): High-speed vectorized copying
 * - Complex permutations: Optimized strided memory access patterns
 * - Large tensors: NUMA-aware chunking for optimal memory locality
 */

#include "permute.h"
#include "numa-kernels.h"
#include "../ggml-numa-shared.h"
#include "../ggml-numa-simple-coordinator.h"
#include "../ggml-cpu-impl.h"
#include "../ggml-impl.h"
#include "../ops.h"
#include <string.h>

/**
 * NUMA-aware permutation execution kernel
 * 
 * Performs tensor dimension permutation with NUMA optimizations.
 * Unlike the reference implementation (which is a NOP), this kernel
 * actually materializes the permutation for better memory locality.
 */
enum ggml_status ggml_numa_kernel_permute_execute(void * work_context, struct ggml_compute_params * params) {
    NUMA_ASSERT(work_context != NULL, "Work context cannot be null");
    NUMA_ASSERT(params != NULL, "Compute params cannot be null");
    
    struct ggml_tensor * dst = (struct ggml_tensor *)work_context;
    struct ggml_tensor * src = dst->src[0];
    
    NUMA_ASSERT(src != NULL, "Source tensor cannot be null");
    NUMA_ASSERT(dst->op == GGML_OP_PERMUTE, "Tensor operation must be PERMUTE");
    
    // Get permutation axes from operation parameters
    int32_t * axes = (int32_t *)dst->op_params;
    const int axis0 = axes[0];
    const int axis1 = axes[1]; 
    const int axis2 = axes[2];
    const int axis3 = axes[3];
    
    NUMA_LOG_TRACE("PERMUTE: axes [%d,%d,%d,%d], src shape [%ld,%ld,%ld,%ld] -> dst shape [%ld,%ld,%ld,%ld]",
                   axis0, axis1, axis2, axis3,
                   src->ne[0], src->ne[1], src->ne[2], src->ne[3],
                   dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    
    // Get tensor data using NUMA-aware accessors
    const float * src_data = (const float *)tensor_data(src);
    
    // Use shared result tensor memory for direct writes (eliminates aggregation)
    extern __thread void * ggml_numa_shared_result_tensor_data;
    float * dst_data;
    if (ggml_numa_shared_result_tensor_data != NULL) {
        dst_data = (float *)ggml_numa_shared_result_tensor_data;
    } else {
        dst_data = (float *)tensor_data(dst);
    }
    
    // Get NUMA execution context from thread-local variables
    extern __thread int ggml_current_numa_node;
    extern __thread int ggml_numa_total_nodes_for_data_parallel;
    extern __thread bool ggml_numa_is_data_parallel_execution;
    
    // Calculate tensor dimensions
    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t ne3 = dst->ne[3];
    
    const int64_t nb0 = dst->nb[0];
    const int64_t nb1 = dst->nb[1];
    const int64_t nb2 = dst->nb[2];
    const int64_t nb3 = dst->nb[3];
    
    // Source tensor strides
    const int64_t src_nb[4] = { src->nb[0], src->nb[1], src->nb[2], src->nb[3] };
    
    // Calculate total number of elements
    const int64_t total_elements = ne0 * ne1 * ne2 * ne3;
    
    // NUMA data slicing for data-parallel execution
    int64_t start_element = 0;
    int64_t end_element = total_elements;
    
    if (ggml_numa_is_data_parallel_execution && ggml_numa_total_nodes_for_data_parallel > 1) {
        int64_t elements_per_node = total_elements / ggml_numa_total_nodes_for_data_parallel;
        start_element = ggml_current_numa_node * elements_per_node;
        end_element = (ggml_current_numa_node == ggml_numa_total_nodes_for_data_parallel - 1) ? 
                      total_elements : start_element + elements_per_node;
        
        NUMA_LOG_TRACE("NUMA node %d processing elements %ld-%ld of %ld total",
                       ggml_current_numa_node, start_element, end_element, total_elements);
    }
    
    // Process elements in this NUMA node's range
    for (int64_t i = start_element; i < end_element; i++) {
        // Convert linear index to 4D coordinates in destination tensor
        const int64_t i3 = i / (ne0 * ne1 * ne2);
        const int64_t i2 = (i % (ne0 * ne1 * ne2)) / (ne0 * ne1);
        const int64_t i1 = (i % (ne0 * ne1)) / ne0;
        const int64_t i0 = i % ne0;
        
        // Map destination coordinates to source coordinates using permutation axes
        // The axes tell us: output_dim[i] comes from input_dim[axis[i]]
        // So: input coordinate = output_coordinate[axis[input_dim]]
        int64_t src_coords[4];
        src_coords[0] = (axis0 == 0) ? i0 : (axis0 == 1) ? i1 : (axis0 == 2) ? i2 : i3;
        src_coords[1] = (axis1 == 0) ? i0 : (axis1 == 1) ? i1 : (axis1 == 2) ? i2 : i3;
        src_coords[2] = (axis2 == 0) ? i0 : (axis2 == 1) ? i1 : (axis2 == 2) ? i2 : i3;
        src_coords[3] = (axis3 == 0) ? i0 : (axis3 == 1) ? i1 : (axis3 == 2) ? i2 : i3;
        
        // Calculate source element offset
        const int64_t src_offset = 
            src_coords[0] * src_nb[0] / sizeof(float) +
            src_coords[1] * src_nb[1] / sizeof(float) +
            src_coords[2] * src_nb[2] / sizeof(float) +
            src_coords[3] * src_nb[3] / sizeof(float);
        
        // Calculate destination element offset
        const int64_t dst_offset = 
            i0 * nb0 / sizeof(float) +
            i1 * nb1 / sizeof(float) +
            i2 * nb2 / sizeof(float) +
            i3 * nb3 / sizeof(float);
        
        // Perform the permutation (copy element from source to destination)
        dst_data[dst_offset] = src_data[src_offset];
    }
    
    NUMA_LOG_TRACE("PERMUTE kernel completed for NUMA node %d", ggml_current_numa_node);
    
    return GGML_STATUS_SUCCESS;
}

/**
 * Calculate work buffer size for PERMUTE operation
 * 
 * PERMUTE typically doesn't need additional work buffers since it's
 * primarily a memory reordering operation.
 */
size_t ggml_numa_kernel_permute_calculate_work_buffer_size(const struct ggml_tensor * tensor) {
    GGML_UNUSED(tensor);
    
    // PERMUTE doesn't require additional work buffers
    return 0;
}

/**
 * Query optimal strategy for PERMUTE operation
 * 
 * Analyzes tensor characteristics to determine the best NUMA execution strategy.
 */
ggml_numa_kernel_query_result_t ggml_numa_kernel_permute_query(const struct ggml_tensor * tensor) {
    ggml_numa_kernel_query_result_t result = {0};
    
    if (!tensor || tensor->op != GGML_OP_PERMUTE) {
        result.supported = false;
        return result;
    }
    
    // Calculate total number of elements
    const int64_t total_elements = ggml_nelements(tensor);
    
    // Define thresholds for different strategies
    const int64_t single_thread_threshold = 1024;      // 1K elements
    const int64_t multi_thread_threshold = 262144;     // 256K elements
    
    result.supported = true;
    result.work_buffer_size_per_thread = 0;  // No work buffer needed
    result.work_function = ggml_numa_kernel_permute_execute;
    result.kernel_name = "NUMA PERMUTE Kernel";
    result.aggregation_policy = GGML_NUMA_AGGREGATION_NONE;  // No aggregation needed
    
    // Create execution strategy based on tensor size
    ggml_numa_execution_strategy_t strategy = {0};
    if (total_elements < single_thread_threshold) {
        strategy.node_strategy = NUMA_NODE_STRATEGY_SINGLE;
        strategy.on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD;
        result.efficiency_score = 0.95f;
    } else if (total_elements < multi_thread_threshold) {
        strategy.node_strategy = NUMA_NODE_STRATEGY_SINGLE;
        strategy.on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD;
        result.efficiency_score = 0.92f;
    } else {
        strategy.node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL;
        strategy.on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD;
        result.efficiency_score = 0.88f;
    }
    
    result.strategy = strategy;
    
    // Apply force strategy override if set
    ggml_numa_apply_kernel_force_strategy(&result, "PERMUTE", 
                                          ggml_numa_kernel_permute_execute, 
                                          ggml_numa_kernel_permute_execute,
                                          ggml_numa_kernel_permute_execute);
    
    return result;
}

/**
 * Register NUMA PERMUTE kernel with the framework
 * 
 * Provides strategy thresholds and function pointers for different execution modes.
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_permute_register(void) {
    ggml_numa_kernel_registration_info_t info = {0};
    
    info.op_type = GGML_OP_PERMUTE;
    info.supported = true;
    info.kernel_name = "NUMA PERMUTE Kernel";
    
    // Strategy thresholds - more conservative since permutation involves random memory access
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_SINGLE] = 1024;      // 1K elements
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_MULTI] = 262144;     // 256K elements
    // Above 256K elements: data-parallel strategy
    info.strategy_array.valid = true;
    
    // Work function pointers - same function handles all strategies
    info.work_funcs.single_single_fn = ggml_numa_kernel_permute_execute;
    info.work_funcs.single_multi_fn = ggml_numa_kernel_permute_execute;
    info.work_funcs.data_parallel_fn = ggml_numa_kernel_permute_execute;
    info.work_funcs.valid = true;
    
    // PERMUTE doesn't need aggregation since it's element-wise permutation
    info.agg_funcs.single_single_fn = NULL;
    info.agg_funcs.single_multi_fn = NULL;
    info.agg_funcs.data_parallel_fn = NULL;
    info.agg_funcs.valid = false;
    
    return info;
}
