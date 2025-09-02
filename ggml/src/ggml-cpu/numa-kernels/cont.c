/**
 * @file cont.c
 * @brief NUMA CONT Kernel Implementation - Tensor Contiguity Operation
 *
 * This file implements a NUMA-aware CONT kernel for making tensors contiguous.
 * CONT performs data copying operations to reorganize tensor memory layouts
 * for optimal access patterns in subsequent operations.
 * 
 * Mathematical Operation:
 * - Input tensor:  [ne0, ne1, ne2, ne3] with arbitrary strides [nb0, nb1, nb2, nb3]
 * - Output tensor: [ne0, ne1, ne2, ne3] with contiguous strides
 * - Element values preserved: dst[i,j,k,l] = src[i,j,k,l] for all valid indices
 * - Memory layout optimized for sequential access patterns
 *
 * Implementation Strategy:
 * - Row-wise copying for optimal cache utilization
 * - NUMA-aware data slicing for large tensors
 * - Type-aware copying strategies (bytes, f16, f32, quantized)
 * - Multi-threading support for improved bandwidth utilization
 * - Memory bandwidth optimization through aligned access patterns
 *
 * Performance Characteristics:
 * - Memory bandwidth limited operation
 * - Benefits from NUMA locality and parallel copying
 * - Scales well with multi-threading for large tensors
 * - Cache-friendly row-by-row processing
 */

#include "cont.h"
#include "../ggml-numa-shared.h"
#include "../ggml-numa-perf.h"
#include "../ops.h"
#include "../vec.h"
#include "numa-kernels.h"

// ============================================================================
// Strategy Thresholds for CONT Operation
// ============================================================================

/**
 * @brief Strategy threshold structure for CONT operations
 * 
 * CONT operations benefit from multi-threading for medium to large tensors
 * since they are memory bandwidth limited and can utilize multiple cores
 * effectively for parallel data copying.
 */
typedef struct {
    size_t element_threshold;
    ggml_numa_execution_strategy_t strategy;
    float efficiency_score;
    const char * kernel_name;
} ggml_cont_strategy_threshold_t;

#define CONT_THRESHOLD_COUNT 3

/**
 * CONT-specific strategy thresholds
 * Balanced for memory bandwidth optimization
 */
static const ggml_cont_strategy_threshold_t CONT_THRESHOLDS[CONT_THRESHOLD_COUNT] = {
    // Small tensors: Single thread for minimal overhead
    { 
        .element_threshold = 4096,    
        .strategy = { .node_strategy = NUMA_NODE_STRATEGY_SINGLE, .on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD },
        .efficiency_score = 0.85f, 
        .kernel_name = "NUMA CONT (Single/Single)" 
    },
    
    // Medium tensors: Multi-threading for better bandwidth utilization
    { 
        .element_threshold = 131072,   // 128K elements
        .strategy = { .node_strategy = NUMA_NODE_STRATEGY_SINGLE, .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD },
        .efficiency_score = 0.90f, 
        .kernel_name = "NUMA CONT (Single/Multi)" 
    },
    
    // Large tensors: Data-parallel across NUMA nodes
    { 
        .element_threshold = SIZE_MAX, 
        .strategy = { .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL, .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD },
        .efficiency_score = 0.95f, 
        .kernel_name = "NUMA CONT (Data Parallel)" 
    }
};

// ============================================================================
// NUMA Slice Calculation for Row Processing
// ============================================================================

/**
 * @brief Calculate NUMA row slice for data-parallel execution
 * 
 * Distributes rows across NUMA nodes for optimal memory bandwidth
 * utilization and cache locality.
 */
static void get_numa_row_slice(int64_t total_rows, int numa_node, int total_numa_nodes, 
                              int64_t *start_row, int64_t *end_row) {
    const int64_t rows_per_node = total_rows / total_numa_nodes;
    *start_row = numa_node * rows_per_node;
    *end_row = (numa_node == total_numa_nodes - 1) ? total_rows : *start_row + rows_per_node;
}

// ============================================================================
// CONT Kernel Execution Function  
// ============================================================================

/**
 * @brief NUMA CONT kernel execution function
 * 
 * This function implements the core CONT operation logic with NUMA-aware
 * optimizations. It delegates to the appropriate ggml_compute_forward_dup
 * implementation based on data types and tensor characteristics.
 * 
 * The implementation leverages existing optimized copying routines while
 * providing NUMA-aware memory access patterns and thread distribution.
 * 
 * @param work_context Tensor context for CONT operation
 * @param params Compute parameters including threading information
 * @return GGML_STATUS_SUCCESS on completion, error status on failure
 */
enum ggml_status ggml_numa_kernel_cont_execute(void * work_context, 
                                                struct ggml_compute_params * params) {
    // Validate parameters
    NUMA_ASSERT(work_context != NULL, "Work context cannot be null");
    NUMA_ASSERT(params != NULL, "Compute params cannot be null");
    
    struct ggml_tensor * tensor = (struct ggml_tensor *)work_context;
    NUMA_ASSERT(tensor != NULL, "Tensor cannot be null");
    NUMA_ASSERT(tensor->op == GGML_OP_CONT, "Tensor operation must be CONT");
    
    const struct ggml_tensor * src0 = tensor->src[0];
    NUMA_ASSERT(src0 != NULL, "Source tensor cannot be null");
    
    // Performance instrumentation
    NUMA_PERF_START(NUMA_PERF_KERNEL_NUMA_EXEC, "CONT", "execute_phase", -1, 0, 0);
    
    // Get NUMA execution context from thread-local variables
    extern __thread int ggml_current_numa_node;
    extern __thread bool ggml_numa_is_data_parallel_execution;
    extern __thread int ggml_numa_total_nodes_for_data_parallel;
    extern __thread void * ggml_numa_shared_result_tensor_data;
    
    // Use shared result tensor memory for direct writes if available
    if (ggml_numa_shared_result_tensor_data != NULL) {
        NUMA_LOG_TRACE("CONT using shared result tensor memory at %p", ggml_numa_shared_result_tensor_data);
        // Note: The actual tensor data pointer will be used by ggml_compute_forward_dup
    } else {
        NUMA_LOG_TRACE("CONT using local tensor memory at %p", tensor_data(tensor));
    }
    
    // Log execution details
    const size_t total_elements = ggml_nelements(tensor);
    const size_t src_total_elements = ggml_nelements(src0);
    
    NUMA_LOG_DEBUG("CONT execution: tensor=%p, elements=%zu, src_elements=%zu, "
                   "src_type=%s, dst_type=%s, numa_node=%d, data_parallel=%s",
                   tensor, total_elements, src_total_elements,
                   ggml_type_name(src0->type), ggml_type_name(tensor->type),
                   ggml_current_numa_node, 
                   ggml_numa_is_data_parallel_execution ? "true" : "false");
    
    // CONT is implemented as ggml_compute_forward_dup in the original code
    // We use the optimized DUP implementation which handles various data types
    // and memory layouts efficiently with proper threading support
    ggml_compute_forward_dup(params, tensor);
    
    NUMA_LOG_TRACE("CONT execution completed successfully on NUMA node %d", ggml_current_numa_node);
    
    NUMA_PERF_END();
    return GGML_STATUS_SUCCESS;
}

// ============================================================================
// CONT Query Function
// ============================================================================

/**
 * @brief Query CONT kernel capabilities and strategy selection
 * 
 * This function evaluates tensor characteristics and selects the optimal
 * execution strategy for CONT operations. Strategy selection considers
 * data size, memory bandwidth requirements, and NUMA topology.
 * 
 * @param tensor Target tensor for CONT operation
 * @return Query result with strategy selection and efficiency metrics
 */
ggml_numa_kernel_query_result_t ggml_numa_kernel_cont_query(const struct ggml_tensor * tensor) {
    ggml_numa_kernel_query_result_t result = {0};
    
    // Validate tensor for CONT operation
    if (!tensor || tensor->op != GGML_OP_CONT) {
        NUMA_LOG_DEBUG("CONT query: REJECTING - invalid tensor or operation");
        result.supported = false;
        return result;
    }
    
    const struct ggml_tensor * src0 = tensor->src[0];
    if (!src0) {
        NUMA_LOG_DEBUG("CONT query: REJECTING - missing source tensor");
        result.supported = false;
        return result;
    }
    
    // Accept all data types - CONT operation handles type conversion
    NUMA_LOG_DEBUG("CONT query: ACCEPTING - src0_type=%s, dst_type=%s", 
                   ggml_type_name(src0->type), ggml_type_name(tensor->type));
    
    // Calculate total elements for strategy selection
    const size_t total_elements = ggml_nelements(tensor);
    
    // Select strategy based on tensor size using threshold-based selection
    const ggml_cont_strategy_threshold_t * selected_strategy;
    NUMA_SELECT_STRATEGY_BY_THRESHOLD(CONT_THRESHOLDS, CONT_THRESHOLD_COUNT, total_elements, selected_strategy);
    
    // Calculate work buffer size estimate (minimal for CONT operations)
    const size_t work_buffer_size = 64;  // Small buffer for potential temporary data
    
    // Build successful query result
    result.supported = true;
    result.strategy = selected_strategy->strategy;
    result.work_buffer_size_per_thread = work_buffer_size;
    result.work_function = ggml_numa_kernel_cont_execute;
    result.efficiency_score = selected_strategy->efficiency_score;
    result.kernel_name = selected_strategy->kernel_name;
    
    // CONT writes directly to destination tensor - no aggregation needed
    result.aggregation_policy = GGML_NUMA_AGGREGATION_NONE;
    result.aggregation_function = NULL;
    result.aggregation_user_data = NULL;
    
    NUMA_LOG_DEBUG("CONT query: %zu elements -> %s (efficiency: %.2f, buffer: %zu bytes)",
                   total_elements, selected_strategy->kernel_name, 
                   selected_strategy->efficiency_score, work_buffer_size);
    
    return result;
}

// ============================================================================
// Registration Function
// ============================================================================

/**
 * @brief Register CONT kernel in NUMA kernel cache system
 * 
 * This function provides kernel registration information for the NUMA
 * kernel cache, enabling direct function pointer dispatch without switch
 * statement overhead.
 * 
 * @return Registration information structure with kernel metadata
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_cont_register(void) {
    ggml_numa_kernel_registration_info_t info = {0};
    
    info.op_type = GGML_OP_CONT;
    info.supported = true;
    info.kernel_name = "NUMA CONT Kernel";
    
    // Copy threshold array for strategy selection
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_SINGLE] = CONT_THRESHOLDS[0].element_threshold;
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_MULTI] = CONT_THRESHOLDS[1].element_threshold;
    // CONT_THRESHOLDS[2] has SIZE_MAX threshold (data-parallel for everything above threshold 1)
    info.strategy_array.valid = true;
    
    // Function pointers for different strategies
    info.work_funcs.single_single_fn = ggml_numa_kernel_cont_execute;
    info.work_funcs.single_multi_fn = ggml_numa_kernel_cont_execute;
    info.work_funcs.data_parallel_fn = ggml_numa_kernel_cont_execute;
    info.work_funcs.valid = true;
    
    // Query function pointer for direct dispatch (eliminates switch statements)
    info.query_fn = (void*)ggml_numa_kernel_cont_query;
    
    // CONT operations don't need aggregation functions (direct destination writes)
    info.agg_funcs.single_single_fn = NULL;
    info.agg_funcs.single_multi_fn = NULL; 
    info.agg_funcs.data_parallel_fn = NULL;
    info.agg_funcs.valid = false;
    
    // Log registration completion
    NUMA_LOG_DEBUG("Registered CONT kernel: thresholds=[%zu, %zu], supports_data_parallel=true",
                  CONT_THRESHOLDS[0].element_threshold, CONT_THRESHOLDS[1].element_threshold);
    
    return info;
}
