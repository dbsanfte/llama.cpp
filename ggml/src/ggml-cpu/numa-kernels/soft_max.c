/**
 * @file soft_max.c
 * @brief NUMA Kernel Implementation: Softmax Activation (SOFT_MAX)
 * 
 * ============================================================================
 * NUMA KERNEL IMPLEMENTATION: SOFTMA        get_numa_row_slice(ne01, ggml_current_numa_node, ggml_numa_total_nodes_for_data_parallel, 
                           &numa_start_row, &numa_end_row);ACTIVATION (SOFT_MAX)
 * ============================================================================
 * 
 * This implementation provides NUMA-aware softmax activation computation
 * following the complex operations pattern (MUL_MAT template).
 * 
 * MATHEMATICAL OPERATION:
 * ======================
 * 
 * Softmax computes: output[i] = exp(input[i] - max) / sum(exp(input[j] - max))
 * 
 * Where:
 * - input: Input tensor (src0) with shape [ne00, ne01, ne02, ne03]
 * - output: Result tensor (dst) with same shape as input
 * - Processing is done row-wise (along ne00 dimension)
 * 
 * NUMA PARALLELIZATION STRATEGY:
 * ==============================
 * - Single-node strategies: Process full rows on one NUMA node
 * - Data-parallel strategies: Distribute rows across NUMA nodes
 * - Each row is processed atomically for numerical stability
 * 
 * OPTIMIZATIONS:
 * =============
 * - Uses SIMD-optimized ggml_vec_soft_max_f32 for core computation
 * - ALiBi attention bias support (scale, max_bias parameters)
 * - Numerical stability through max subtraction
 * - Row-wise processing for optimal cache utilization
 */

#include "soft_max.h"
#include "ggml-numa-shared.h"
#include "ggml-numa-perf.h"
#include "ggml-cpu.h"
#include "../vec.h"
#include "../ops.h"

// ============================================================================
// Strategy Thresholds for SOFT_MAX Operation
// ============================================================================

/**
 * Strategy threshold configuration for SOFT_MAX kernel selection
 * Balanced thresholds considering the row-wise processing pattern
 */
typedef struct {
    size_t element_threshold;
    ggml_numa_execution_strategy_t strategy;
    float efficiency_score;
    const char * kernel_name;
} ggml_soft_max_strategy_threshold_t;

#define SOFT_MAX_THRESHOLD_COUNT 3

static const ggml_soft_max_strategy_threshold_t SOFT_MAX_THRESHOLDS[SOFT_MAX_THRESHOLD_COUNT] = {
    // Small tensors: Single thread for minimal overhead
    { 
        .element_threshold = 8192,    
        .strategy = { .node_strategy = NUMA_NODE_STRATEGY_SINGLE, .on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD },
        .efficiency_score = 0.95f, 
        .kernel_name = "NUMA SOFT_MAX (Single/Single)" 
    },
    
    // Medium tensors: Multi-threading on single node for good cache locality  
    { 
        .element_threshold = 65536,   
        .strategy = { .node_strategy = NUMA_NODE_STRATEGY_SINGLE, .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD },
        .efficiency_score = 0.90f, 
        .kernel_name = "NUMA SOFT_MAX (Single/Multi)" 
    },
    
    // Large tensors: Data-parallel across NUMA nodes
    { 
        .element_threshold = SIZE_MAX, 
        .strategy = { .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL, .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD },
        .efficiency_score = 0.85f, 
        .kernel_name = "NUMA SOFT_MAX (Data Parallel)" 
    }
};

// ============================================================================
// NUMA Slice Calculation for Row Processing
// ============================================================================

/**
 * Calculate NUMA node's row slice for data-parallel execution
 * 
 * @param total_rows   Total number of rows to process (ne01)
 * @param current_node NUMA node executing this slice (0-based)
 * @param total_nodes  Total number of NUMA nodes participating
 * @param start        [OUT] Starting row index for this node's slice
 * @param end          [OUT] Ending row index (exclusive) for this node's slice
 */
static inline void get_numa_row_slice(int64_t total_rows, 
                                     int current_node, 
                                     int total_nodes,
                                     int64_t * start, 
                                     int64_t * end) {
    if (total_nodes <= 1) {
        *start = 0;
        *end = total_rows;
        return;
    }
    
    const int64_t rows_per_node = total_rows / total_nodes;
    *start = current_node * rows_per_node;
    *end = (current_node == total_nodes - 1) ? total_rows : *start + rows_per_node;
}

/**
 * Calculate thread's row slice within a NUMA node
 * 
 * @param total_rows  Total number of rows to process
 * @param thread_id   Thread ID within the node (0-based)
 * @param num_threads Number of threads in this node
 * @param start       [OUT] Starting row index for this thread
 * @param end         [OUT] Ending row index (exclusive) for this thread
 */
static inline void get_thread_row_slice(int64_t total_rows,
                                       int thread_id,
                                       int num_threads,
                                       int64_t * start,
                                       int64_t * end) {
    if (num_threads <= 1) {
        *start = 0;
        *end = total_rows;
        return;
    }
    
    const int64_t rows_per_thread = total_rows / num_threads;
    *start = thread_id * rows_per_thread;
    *end = (thread_id == num_threads - 1) ? total_rows : *start + rows_per_thread;
}

// ============================================================================
// Core SOFT_MAX NUMA Kernel Implementation
// ============================================================================

enum ggml_status ggml_numa_kernel_soft_max_execute(void * work_context, struct ggml_compute_params * params) {
    NUMA_ASSERT(work_context != NULL, "Work context cannot be null");
    NUMA_ASSERT(params != NULL, "Compute params cannot be null");
    
    // PERFORMANCE: Start timing the kernel execution
    NUMA_PERF_START(NUMA_PERF_KERNEL_NUMA_EXEC, "SOFT_MAX", "execute_phase", -1, 0, 0);
    
    struct ggml_tensor * tensor = (struct ggml_tensor *)work_context;
    
    // Validate tensor operation type
    if (tensor->op != GGML_OP_SOFT_MAX) {
        NUMA_LOG_DEBUG("SOFT_MAX kernel called with wrong operation type: %d", tensor->op);
        NUMA_PERF_END();
        return GGML_STATUS_FAILED;
    }
    
    // Extract source tensors
    const struct ggml_tensor * src0 = tensor->src[0];  // Input tensor
    const struct ggml_tensor * src1 = tensor->src[1];  // Optional mask tensor
    
    if (!src0) {
        NUMA_LOG_DEBUG("SOFT_MAX kernel: Missing input tensor");
        NUMA_PERF_END();
        return GGML_STATUS_FAILED;
    }
    
    // Extract tensor dimensions (row-wise processing)
    const int64_t ne00 = src0->ne[0];  // Row size (elements per row)
    const int64_t ne01 = src0->ne[1];  // Number of rows
    const int64_t ne02 = src0->ne[2];  // Batch dimension 2
    const int64_t ne03 = src0->ne[3];  // Batch dimension 3
    
    // Extract strides
    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];
    const int64_t nb03 = src0->nb[3];
    
    const int64_t nb1 = tensor->nb[1];
    const int64_t nb2 = tensor->nb[2];
    const int64_t nb3 = tensor->nb[3];
    
    // Extract operation parameters (scale and max_bias for ALiBi attention)
    float scale = 1.0f;
    float max_bias = 0.0f;
    if (tensor->op_params) {
        memcpy(&scale,    (float *) tensor->op_params + 0, sizeof(float));
        memcpy(&max_bias, (float *) tensor->op_params + 1, sizeof(float));
    }
    
    // Get NUMA execution context from thread-local variables
    extern __thread int ggml_current_numa_node;
    extern __thread int ggml_numa_total_nodes_for_data_parallel;
    extern __thread bool ggml_numa_is_data_parallel_execution;
    extern __thread void * ggml_numa_shared_result_tensor_data;
    
    // Determine memory access strategy (shared vs local)
    float * dst_data;
    if (ggml_numa_shared_result_tensor_data != NULL) {
        // Use shared result tensor memory - eliminates aggregation overhead
        dst_data = (float *)ggml_numa_shared_result_tensor_data;
        NUMA_LOG_TRACE("SOFT_MAX using shared result tensor memory at %p", dst_data);
    } else {
        // Fallback to local tensor data for compatibility
        dst_data = (float *)tensor_data(tensor);
        NUMA_LOG_TRACE("SOFT_MAX using local tensor memory at %p", dst_data);
    }
    
    // Calculate row slice for this NUMA node
    int64_t numa_start_row = 0, numa_end_row = ne01;
    if (ggml_numa_is_data_parallel_execution) {
        get_numa_row_slice(ne01, ggml_current_numa_node, ggml_numa_total_nodes_for_data_parallel, 
                          &numa_start_row, &numa_end_row);
        NUMA_LOG_TRACE("SOFT_MAX NUMA node %d processing rows %ld-%ld", 
                      ggml_current_numa_node, numa_start_row, numa_end_row);
    }
    
    // Calculate thread slice within NUMA node
    const int ith = params->ith;
    const int nth = params->nth;
    int64_t thread_start_row, thread_end_row;
    get_thread_row_slice(numa_end_row - numa_start_row, ith, nth, 
                        &thread_start_row, &thread_end_row);
    
    // Adjust thread slice to global row indices
    thread_start_row += numa_start_row;
    thread_end_row += numa_start_row;
    
    NUMA_LOG_TRACE("SOFT_MAX thread %d/%d processing rows %ld-%ld", 
                  ith, nth, thread_start_row, thread_end_row);
    
    // Allocate working memory for row processing
    float * wp = (float *) params->wdata + (ne00 + CACHE_LINE_SIZE_F32) * ith;
    
    // Pre-calculate ALiBi parameters for attention bias
    const uint32_t n_head = ne02;
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));
    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    
    // Process assigned batch and row slice
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            // ALiBi slope calculation for attention bias
            const uint32_t h = i02; // head index
            const float slope = (max_bias > 0.0f) ? 
                h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;
            
            for (int64_t i01 = thread_start_row; i01 < thread_end_row; i01++) {
                // Calculate pointers for current row
                float * sp = (float *)((char *) tensor_data(src0) + i01*nb01 + i02*nb02 + i03*nb03);
                float * dp = (float *)((char *) dst_data + i01*nb1 + i02*nb2 + i03*nb3);
                
                // Handle optional mask tensor (for attention masking)
                float * mp_f32 = NULL;
                ggml_fp16_t * mp_f16 = NULL;
                bool use_f16 = false;
                
                if (src1) {
                    const int64_t nb11 = src1->nb[1];
                    const int64_t nb12 = src1->nb[2];
                    const int64_t nb13 = src1->nb[3];
                    const int64_t ne12 = src1->ne[2];
                    const int64_t ne13 = src1->ne[3];
                    
                    const int64_t i11 = i01;
                    const int64_t i12 = i02 % ne12;
                    const int64_t i13 = i03 % ne13;
                    
                    use_f16 = (src1->type == GGML_TYPE_F16);
                    if (use_f16) {
                        mp_f16 = (ggml_fp16_t *)((char *) tensor_data(src1) + i11*nb11 + i12*nb12 + i13*nb13);
                    } else {
                        mp_f32 = (float *)((char *) tensor_data(src1) + i11*nb11 + i12*nb12 + i13*nb13);
                    }
                }
                
                // Step 1: Copy input row and apply scale
                ggml_vec_cpy_f32(ne00, wp, sp);
                ggml_vec_scale_f32(ne00, wp, scale);
                
                // Step 2: Apply mask with ALiBi bias if present
                if (mp_f32 || mp_f16) {
                    if (use_f16) {
                        for (int i = 0; i < ne00; ++i) {
                            wp[i] += slope * GGML_CPU_FP16_TO_FP32(mp_f16[i]);
                        }
                    } else {
                        for (int i = 0; i < ne00; ++i) {
                            wp[i] += slope * mp_f32[i];
                        }
                    }
                }
                
                // Step 3: Find maximum for numerical stability
                float max = -INFINITY;
                ggml_vec_max_f32(ne00, &max, wp);
                
                // Step 4: Compute softmax using SIMD-optimized function
                ggml_float sum = ggml_vec_soft_max_f32(ne00, dp, wp, max);
                NUMA_ASSERT(sum > 0.0f, "Softmax sum must be positive");
                
                // Step 5: Normalize by dividing by sum
                sum = 1.0f / sum;
                ggml_vec_scale_f32(ne00, dp, sum);
            }
        }
    }
    
    NUMA_LOG_TRACE("SOFT_MAX kernel completed successfully on NUMA node %d", ggml_current_numa_node);
    
    NUMA_PERF_END();
    return GGML_STATUS_SUCCESS;
}

// ============================================================================
// Query Function for Strategy Selection
// ============================================================================

ggml_numa_kernel_query_result_t ggml_numa_kernel_soft_max_query(const struct ggml_tensor * tensor) {
    // PERFORMANCE: Start timing the query operation
    NUMA_PERF_START(NUMA_PERF_EXECUTOR_QUERY, "SOFT_MAX", "query_phase", -1, 0, 0);
    
    ggml_numa_kernel_query_result_t result = { .supported = false };
    
    // Validate this is a SOFT_MAX operation
    if (!tensor || tensor->op != GGML_OP_SOFT_MAX) {
        NUMA_PERF_END();
        return result;
    }
    
    // Validate tensor structure
    if (!tensor->src[0]) {
        NUMA_LOG_DEBUG("SOFT_MAX query: Missing source tensor");
        NUMA_PERF_END();
        return result;
    }
    
    const struct ggml_tensor * src0 = tensor->src[0];
    
    // Check if tensor is compatible (must be F32 for current implementation)
    if (src0->type != GGML_TYPE_F32) {
        NUMA_LOG_DEBUG("SOFT_MAX query: REJECTING - src0_type=%d (only F32 supported)", src0->type);
        NUMA_PERF_END();
        return result;
    }
    
    // Check if this kernel is actually registered and supported
    if (!ggml_numa_is_kernel_supported(GGML_OP_SOFT_MAX)) {
        NUMA_LOG_DEBUG("SOFT_MAX kernel not supported - registration disabled");
        result.supported = false;
        NUMA_PERF_END();
        return result;
    }
    
    NUMA_LOG_DEBUG("SOFT_MAX query: ACCEPTING - src0_type=%d", src0->type);
    
    // Calculate total elements in tensor for strategy selection
    const size_t total_elements = ggml_nelements(tensor);
    
    // Find optimal strategy using threshold search
    const ggml_soft_max_strategy_threshold_t * selected_strategy;
    NUMA_SELECT_STRATEGY_BY_THRESHOLD(SOFT_MAX_THRESHOLDS, SOFT_MAX_THRESHOLD_COUNT, total_elements, selected_strategy);
    
    // Calculate work buffer size (need space for one row + cache line padding)
    const size_t work_buffer_size = (tensor->src[0]->ne[0] + CACHE_LINE_SIZE_F32) * sizeof(float);
    
    // Build successful query result
    result.supported = true;
    result.strategy = selected_strategy->strategy;
    result.work_buffer_size_per_thread = work_buffer_size;
    result.work_function = ggml_numa_kernel_soft_max_execute;
    result.efficiency_score = selected_strategy->efficiency_score;
    result.kernel_name = selected_strategy->kernel_name;
    
    // SOFT_MAX writes directly to shared memory - no aggregation needed
    result.aggregation_policy = GGML_NUMA_AGGREGATION_NONE;
    result.aggregation_function = NULL;
    result.aggregation_user_data = NULL;
    
    NUMA_LOG_DEBUG("SOFT_MAX query: %zu elements -> %s (efficiency: %.2f, buffer: %zu bytes)",
                  total_elements, selected_strategy->kernel_name, 
                  selected_strategy->efficiency_score, work_buffer_size);
    
    NUMA_PERF_END();
    return result;
}

// ============================================================================
// Registration Function
// ============================================================================

ggml_numa_kernel_registration_info_t ggml_numa_kernel_soft_max_register(void) {
    ggml_numa_kernel_registration_info_t info = {0};
    
    info.op_type = GGML_OP_SOFT_MAX;
    info.supported = true;
    info.kernel_name = "NUMA SOFT_MAX Kernel";
    
    // Copy threshold array for strategy selection
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_SINGLE] = SOFT_MAX_THRESHOLDS[0].element_threshold;
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_MULTI] = SOFT_MAX_THRESHOLDS[1].element_threshold;
    // SOFT_MAX_THRESHOLDS[2] has SIZE_MAX threshold (data-parallel for everything above threshold 1)
    info.strategy_array.valid = true;
    
    // Function pointers for different strategies
    info.work_funcs.single_single_fn = ggml_numa_kernel_soft_max_execute;
    info.work_funcs.single_multi_fn = ggml_numa_kernel_soft_max_execute;
    info.work_funcs.data_parallel_fn = ggml_numa_kernel_soft_max_execute;
    info.work_funcs.valid = true;
    
    // Query function pointer for direct dispatch
    info.query_fn = (void*)ggml_numa_kernel_soft_max_query;
    
    // SOFT_MAX doesn't need aggregation functions (writes directly to shared memory)
    info.agg_funcs.single_single_fn = NULL;
    info.agg_funcs.single_multi_fn = NULL;
    info.agg_funcs.data_parallel_fn = NULL;
    info.agg_funcs.valid = false;
    
    NUMA_LOG_DEBUG("Registered SOFT_MAX kernel: thresholds=[%zu, %zu], supports_data_parallel=true",
                  SOFT_MAX_THRESHOLDS[0].element_threshold, SOFT_MAX_THRESHOLDS[1].element_threshold);
    
    return info;
}
