/**
 * @file rms_norm.c
 * @brief NUMA Kernel: Root Mean Square Normalization (RMS_NORM)
 * 
 * ============================================================================
 * NUMA KERNEL: ROOT MEAN SQUARE NORMALIZATION (RMS_NORM)
 * ============================================================================
 * 
 * This kernel implements NUMA-aware RMS normalization using optimized
 * row-wise processing with proper data slicing across NUMA nodes.
 * 
 * MATHEMATICAL OPERATION:
 * =====================
 * 
 * RMS_NORM performs row-wise normalization: y = x / sqrt(mean(x²) + eps)
 * 
 * For each row in the input tensor:
 * 1. Calculate sum of squares: sum = Σ(x[i]²) for i in [0, ne00)
 * 2. Calculate mean: mean = sum / ne00
 * 3. Calculate scale: scale = 1.0 / sqrt(mean + eps)
 * 4. Apply scaling: y[i] = x[i] * scale for i in [0, ne00)
 * 
 * Where:
 * - x: Input tensor of shape [ne00, ne01, ne02, ne03]
 * - y: Output tensor (same shape as input)
 * - eps: Small constant for numerical stability (from op_params)
 * 
 * PARALLELIZATION STRATEGY:
 * ========================
 * 
 * The RMS_NORM operation is parallelized along the row dimension:
 * 1. Each row (ne00 elements) is processed independently
 * 2. Rows (ne01) are distributed across threads and NUMA nodes
 * 3. For data-parallel mode, rows are sliced across NUMA nodes
 * 4. Within each node, threads process different subsets of rows
 * 
 * DATA SLICING PATTERN:
 * ====================
 * 
 * For data-parallel execution:
 * - Each NUMA node processes a slice of rows 
 * - Slice along row dimension (ne01) for optimal cache locality
 * - Each row's computation is independent, no inter-row dependencies
 * 
 * PERFORMANCE CONSIDERATIONS:
 * ==========================
 * 
 * - Uses ggml_vec_scale_f32 for optimized SIMD scaling
 * - Processes one row at a time for optimal cache utilization
 * - No reduction across NUMA nodes needed (independent rows)
 * - Memory access pattern is highly cache-friendly (sequential within row)
 * 
 * ============================================================================
 */

#include "rms_norm.h"
#include "numa-kernels.h"
#include "../ggml-numa-shared.h"
#include "../ggml-numa-simple-coordinator.h"
#include "../ggml-numa-perf.h"  // Performance instrumentation
#include "../ggml-cpu-impl.h"
#include "../ggml-impl.h"
#include "../vec.h"

#ifdef GGML_USE_NUMA
#include <numa.h>
#endif

#include <math.h>
#include <string.h>
#include <assert.h>

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

// ============================================================================
// NUMA RMS_NORM Kernel Implementation
// ============================================================================

/**
 * Execute RMS_NORM kernel with NUMA-aware row distribution
 * 
 * This function implements the core RMS normalization logic with proper
 * NUMA data slicing and thread coordination.
 */
enum ggml_status ggml_numa_kernel_rms_norm_execute(void * work_context, struct ggml_compute_params * params) {
    if (!work_context || !params) {
        NUMA_LOG_DEBUG("RMS_NORM kernel: Invalid work_context or params");
        return GGML_STATUS_FAILED;
    }
    
    struct ggml_tensor * dst = (struct ggml_tensor *)work_context;
    
    // =============================================================================
    // Input Validation & Setup
    // =============================================================================
    
    if (!dst || !dst->src[0]) {
        NUMA_LOG_DEBUG("RMS_NORM kernel: Invalid tensor pointers");
        return GGML_STATUS_FAILED;
    }
    
    const struct ggml_tensor * src0 = dst->src[0];  // Input tensor
    
    // Validate that input and output tensors have same shape
    if (!ggml_are_same_shape(src0, dst)) {
        NUMA_LOG_DEBUG("RMS_NORM kernel: Input and output shapes don't match");
        return GGML_STATUS_FAILED;
    }
    
    // RMS_NORM requires F32 input
    if (src0->type != GGML_TYPE_F32) {
        NUMA_LOG_DEBUG("RMS_NORM kernel: Only F32 input supported, got type %d", src0->type);
        return GGML_STATUS_FAILED;
    }
    
    // Validate stride requirement for F32
    if (src0->nb[0] != sizeof(float)) {
        NUMA_LOG_DEBUG("RMS_NORM kernel: Invalid stride for F32 data");
        return GGML_STATUS_FAILED;
    }
    
    // Extract tensor dimensions
    const int64_t ne00 = src0->ne[0];  // Row size (elements to normalize)
    const int64_t ne01 = src0->ne[1];  // Number of rows
    const int64_t ne02 = src0->ne[2];  // Batch dimension 2
    const int64_t ne03 = src0->ne[3];  // Batch dimension 3
    
    // Extract memory strides
    const size_t nb01 = src0->nb[1];   // Stride between rows in src0
    const size_t nb02 = src0->nb[2];   // Stride between batch dim 2 in src0
    const size_t nb03 = src0->nb[3];   // Stride between batch dim 3 in src0
    
    const size_t nb1 = dst->nb[1];     // Stride between rows in dst
    const size_t nb2 = dst->nb[2];     // Stride between batch dim 2 in dst
    const size_t nb3 = dst->nb[3];     // Stride between batch dim 3 in dst
    
    // Extract epsilon from operation parameters
    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    if (eps < 0.0f) {
        NUMA_LOG_DEBUG("RMS_NORM kernel: Invalid epsilon value %f", eps);
        return GGML_STATUS_FAILED;
    }
    
    // =============================================================================
    // NUMA-Aware Data Access 
    // =============================================================================
    
    // Get NUMA-local data pointers
    const float * src_data = (const float *)tensor_data(src0);
    
    // Use shared result tensor memory for direct writes (eliminates aggregation)
    extern __thread void * ggml_numa_shared_result_tensor_data;
    float * dst_data;
    if (ggml_numa_shared_result_tensor_data != NULL) {
        dst_data = (float *)ggml_numa_shared_result_tensor_data;
        NUMA_LOG_DEBUG("RMS_NORM Node %d kernel using shared result tensor memory at %p", ggml_current_numa_node, dst_data);
    } else {
        dst_data = (float *)tensor_data(dst);
        NUMA_LOG_DEBUG("RMS_NORM Node %d kernel using local tensor memory at %p", ggml_current_numa_node, dst_data);
    }
    
    // =============================================================================
    // NUMA Execution Context
    // =============================================================================
    
    // Read NUMA execution context from thread-local variables
    extern __thread bool ggml_numa_is_data_parallel_execution;
    extern __thread int ggml_numa_total_nodes_for_data_parallel;
    extern __thread int ggml_current_numa_node;
    
    const int current_node = ggml_current_numa_node;
    const int total_nodes = ggml_numa_is_data_parallel_execution ? 
                           ggml_numa_total_nodes_for_data_parallel : 1;
    const bool is_data_parallel = ggml_numa_is_data_parallel_execution;
    
    const int thread_id = params->ith;      // Thread ID within this execution
    const int num_threads = params->nth;    // Total threads for this execution
    
    NUMA_LOG_DEBUG("RMS_NORM Node %d: Processing tensor [%ld,%ld,%ld,%ld], eps=%f, threads=%d, data_parallel=%d", 
                   current_node, ne00, ne01, ne02, ne03, eps, num_threads, is_data_parallel);
    
    // =============================================================================
    // Work Distribution Calculation
    // =============================================================================
    
    // Calculate total number of rows to process
    const int64_t total_rows = ne01 * ne02 * ne03;
    
    // For data-parallel execution, distribute rows across NUMA nodes
    int64_t node_row_start = 0;
    int64_t node_row_end = total_rows;
    
    if (is_data_parallel) {
        // Each NUMA node processes a slice of rows
        const int64_t rows_per_node = total_rows / total_nodes;
        node_row_start = current_node * rows_per_node;
        node_row_end = (current_node == total_nodes - 1) ? total_rows : node_row_start + rows_per_node;
        
        NUMA_LOG_DEBUG("RMS_NORM Node %d: Processing rows %ld-%ld of %ld total", 
                       current_node, node_row_start, node_row_end, total_rows);
    }
    
    // Within this NUMA node, distribute rows across threads
    const int64_t node_total_rows = node_row_end - node_row_start;
    
    // CRITICAL FIX: In direct kernel execution mode, each node runs with only one thread (ith=0)
    // So this single thread must process ALL rows for this node, not just a fraction
    int64_t thread_row_start, thread_row_end;
    
    if (num_threads == 1 || thread_id == 0) {
        // Direct execution mode: single thread handles all node rows
        thread_row_start = node_row_start;
        thread_row_end = node_row_end;
        NUMA_LOG_DEBUG("RMS_NORM Node %d Thread %d: Direct execution mode - processing ALL node rows %ld-%ld", 
                       current_node, thread_id, thread_row_start, thread_row_end);
    } else {
        // Multi-threaded mode: distribute rows across threads
        const int64_t rows_per_thread = (node_total_rows + num_threads - 1) / num_threads;  // Round up
        thread_row_start = node_row_start + thread_id * rows_per_thread;
        thread_row_end = MIN(thread_row_start + rows_per_thread, node_row_end);
        NUMA_LOG_DEBUG("RMS_NORM Node %d Thread %d: Multi-threaded mode - processing rows %ld-%ld", 
                       current_node, thread_id, thread_row_start, thread_row_end);
    }
    
    // =============================================================================
    // Core RMS Normalization Computation
    // =============================================================================
    
    const size_t tensor_size = ggml_nelements(dst) * sizeof(float);
    NUMA_PERF_START(NUMA_PERF_KERNEL_NUMA_EXEC, "RMS_NORM", "kernel_execute", current_node, tensor_size, num_threads);
    
    int rows_processed = 0;
    
    // Process each row assigned to this thread
    for (int64_t row_idx = thread_row_start; row_idx < thread_row_end; row_idx++) {
        
        // Convert linear row index back to 3D coordinates
        const int64_t i03 = row_idx / (ne02 * ne01);                    // Batch dimension 3
        const int64_t i02 = (row_idx - i03 * ne02 * ne01) / ne01;      // Batch dimension 2  
        const int64_t i01 = row_idx - i03 * ne02 * ne01 - i02 * ne01;  // Row index
        
        // CRITICAL FIX: Use correct strides for source and destination tensors
        const float * src_row = (const float *)((const char *)src_data + i01*nb01 + i02*nb02 + i03*nb03);
        
        // Use destination tensor strides for destination memory access
        const size_t dst_nb01 = dst->nb[1];   // Destination tensor strides  
        const size_t dst_nb02 = dst->nb[2];
        const size_t dst_nb03 = dst->nb[3];
        float * dst_row = (float *)((char *)dst_data + i01*dst_nb01 + i02*dst_nb02 + i03*dst_nb03);
        
        // Step 1: Calculate sum of squares for this row
        ggml_float sum_squares = 0.0f;
        for (int64_t i00 = 0; i00 < ne00; i00++) {
            const float x = src_row[i00];
            sum_squares += (ggml_float)(x * x);
        }
        
        // Step 2: Calculate mean and scale factor
        const float mean = sum_squares / ne00;
        const float scale = 1.0f / sqrtf(mean + eps);
        
        // Validate scale factor (catch numerical issues early)
        if (scale <= 0.0f || !isfinite(scale)) {
            NUMA_LOG_DEBUG("RMS_NORM Node %d: Invalid scale factor %f for row %ld (mean=%f, eps=%f)", 
                           current_node, scale, row_idx, mean, eps);
            return GGML_STATUS_FAILED;
        }
        
        // Step 3: Copy input to output and apply scaling
        // First copy the data (handles potential memory overlap)
        memcpy(dst_row, src_row, ne00 * sizeof(float));
        
        // Then apply SIMD-optimized scaling
        ggml_vec_scale_f32(ne00, dst_row, scale);
        
        rows_processed++;
        
        // Debug logging for first few rows with memory addresses, and specifically row 19
        if (row_idx < 3 || row_idx == 19) {
            NUMA_LOG_VERBOSE("RMS_NORM Node %d Thread %d: Row %ld - mean=%.6f, scale=%.6f, first_out=%.6f", 
                             current_node, thread_id, row_idx, mean, scale, dst_row[0]);
            NUMA_LOG_VERBOSE("RMS_NORM Node %d Thread %d: Row %ld - dst_row=%p, src_row=%p, dst_data=%p", 
                             current_node, thread_id, row_idx, (void*)dst_row, (void*)src_row, (void*)dst_data);
        }
    }
    
    NUMA_PERF_END();
    
    NUMA_LOG_DEBUG("RMS_NORM Node %d Thread %d: Completed %d rows", 
                   current_node, thread_id, rows_processed);
    
    // Verify writes persisted for debugging data-parallel issues
    if (is_data_parallel && rows_processed > 0) {
        // Check first row we processed
        int64_t first_row_idx = thread_row_start;
        const int64_t i03 = first_row_idx / (ne02 * ne01);
        const int64_t i02 = (first_row_idx - i03 * ne02 * ne01) / ne01;
        const int64_t i01 = first_row_idx - i03 * ne02 * ne01 - i02 * ne01;
        
        // CRITICAL FIX: Use dest tensor strides when accessing shared memory
        const size_t dst_nb01 = dst->nb[1];   // Destination tensor strides
        const size_t dst_nb02 = dst->nb[2];
        const size_t dst_nb03 = dst->nb[3];
        float * verify_row = (float *)((char *)dst_data + i01*dst_nb01 + i02*dst_nb02 + i03*dst_nb03);
        
        NUMA_LOG_VERBOSE("RMS_NORM Node %d Thread %d: Verification - first_row=%ld, value=%.6f, addr=%p", 
                         current_node, thread_id, first_row_idx, verify_row[0], (void*)verify_row);
                         
        // Add memory barrier to ensure NUMA coherency
        __sync_synchronize();
        NUMA_LOG_VERBOSE("RMS_NORM Node %d Thread %d: Memory barrier completed", current_node, thread_id);
    }
    
    return GGML_STATUS_SUCCESS;
}

// ============================================================================
// Strategy Selection and Work Buffer Size Calculation
// ============================================================================

/**
 * Calculate work buffer size needed for RMS_NORM operation
 * 
 * RMS_NORM doesn't need work buffers for type conversion (only F32 supported),
 * so this always returns 0.
 */
static size_t ggml_numa_kernel_rms_norm_calculate_work_buffer_size(const struct ggml_tensor * tensor) {
    (void)tensor;  // Unused parameter
    
    // RMS_NORM operates in-place on F32 data, no work buffer needed
    return 0;
}

/**
 * Query function for RMS_NORM kernel selection and strategy
 * 
 * Returns optimal execution strategy based on tensor size and complexity.
 */
ggml_numa_kernel_query_result_t ggml_numa_kernel_rms_norm_query(const struct ggml_tensor * tensor) {
    ggml_numa_kernel_query_result_t result = {0};
    
    // Input validation
    if (!tensor || !tensor->src[0]) {
        NUMA_LOG_DEBUG("RMS_NORM query: Invalid tensor");
        return result;
    }
    
    const struct ggml_tensor * src0 = tensor->src[0];
    
    // Only support F32 input
    if (src0->type != GGML_TYPE_F32) {
        NUMA_LOG_DEBUG("RMS_NORM query: Only F32 supported, got type %d", src0->type);
        return result;
    }
    
    // Calculate tensor characteristics
    const int64_t ne00 = src0->ne[0];  // Row size
    const int64_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const size_t total_elements = ggml_nelements(src0);
    
    // Strategy selection based on complexity and row count
    ggml_numa_execution_strategy_t strategy = {0};
    
    if (total_rows < 16) {
        // Very few rows: single-threaded execution
        strategy.node_strategy = NUMA_NODE_STRATEGY_SINGLE;
        strategy.on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD;
        result.efficiency_score = 0.6f;
        result.kernel_name = "RMS_NORM Single Thread";
        
    } else if (total_rows < 256 || total_elements < 65536) {
        // Moderate number of rows: single-node multi-threaded
        strategy.node_strategy = NUMA_NODE_STRATEGY_SINGLE;
        strategy.on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD;
        result.efficiency_score = 0.8f;
        result.kernel_name = "RMS_NORM Single Node";
        
    } else {
        // Many rows: data-parallel across NUMA nodes
        strategy.node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL;
        strategy.on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD;
        result.efficiency_score = 0.95f;
        result.kernel_name = "RMS_NORM Data Parallel";
    }
    
    // Fill in the result
    result.supported = true;
    result.strategy = strategy;
    result.work_function = ggml_numa_kernel_rms_norm_execute;
    result.work_buffer_size_per_thread = ggml_numa_kernel_rms_norm_calculate_work_buffer_size(tensor);
    result.aggregation_policy = GGML_NUMA_AGGREGATION_NONE;  // No aggregation needed
    result.aggregation_function = NULL;
    result.aggregation_user_data = NULL;
    
    NUMA_LOG_DEBUG("RMS_NORM query: Selected strategy %s for tensor with %zu elements, %ld rows, efficiency=%.2f",
                   result.kernel_name, total_elements, total_rows, result.efficiency_score);
    
    return result;
}

// ============================================================================
// Kernel Registration
// ============================================================================

/**
 * Register RMS_NORM kernels with the NUMA kernel registry
 */
void ggml_numa_register_rms_norm_kernels(void) {
    ggml_numa_kernel_registration_info_t info = {
        .op_type = GGML_OP_RMS_NORM,
        .strategy_array = {
            .thresholds = {256, 65536},  // Row count and element count thresholds
            .valid = true
        },
        .work_funcs = {
            .single_single_fn = ggml_numa_kernel_rms_norm_execute,
            .single_multi_fn = ggml_numa_kernel_rms_norm_execute,
            .data_parallel_fn = ggml_numa_kernel_rms_norm_execute,
            .valid = true
        },
        .agg_funcs = {
            .valid = false  // No aggregation needed for RMS_NORM
        },
        .kernel_name = "NUMA RMS_NORM Kernel",
        .supported = true
    };
    
    // Register with the NUMA kernel registry
    ggml_numa_register_kernel_strategy(info.op_type, &info.strategy_array, 
                                       &info.work_funcs, &info.agg_funcs);
    
    NUMA_LOG_DEBUG("Registered RMS_NORM NUMA kernel with thresholds [%zu, %zu]", 
                   info.strategy_array.thresholds[0], info.strategy_array.thresholds[1]);
}
