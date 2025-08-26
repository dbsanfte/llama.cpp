/**
 * @file mul_mat.c
 * @brief NUMA Kernel: Matrix Multiplication (MUL_MAT)
 * 
 * ============================================================================
 * NUMA KERNEL: MATRIX MULTIPLICATION (MUL_MAT)
 * ============================================================================
 * 
 * This kernel implements NUMA-aware matrix multiplication using optimized
 * chunk-based processing with proper data slicing across NUMA nodes.
 * 
 * MATHEMATICAL OPERATION:
 * =====================
 * 
 * MUL_MAT performs matrix multiplication: C = A × B
 * 
 * Where:
 * - A (src0): Matrix of shape [ne00, ne01, ne02, ne03] 
 * - B (src1): Matrix of shape [ne10, ne11, ne12, ne13]
 * - C (dst):  Result matrix of shape [ne0, ne1, ne2, ne3]
 * 
 * Constraints:
 * - ne0 == ne01 (A's columns match result rows)
 * - ne1 == ne11 (B's columns match result columns) 
 * - ne2 == ne12 && ne3 == ne13 (batch dimensions match)
 * 
 * PARALLELIZATION STRATEGY:
 * ========================
 * 
 * The MUL_MAT operation is parallelized along the output matrix dimensions:
 * 1. Divide output rows (ne0) across threads within each NUMA node
 * 2. For data-parallel mode, divide batch dimensions (ne1*ne2*ne3) across NUMA nodes
 * 3. Use optimized vec_dot operations for inner products
 * 
 * DATA SLICING PATTERN:
 * ====================
 * 
 * For data-parallel execution:
 * - Each NUMA node processes a slice of the output matrix
 * - Slice along batch dimensions (ne1, ne2, ne3) for memory locality
 * - Within each node, threads divide the row dimension (ne0)
 * 
 * PERFORMANCE CONSIDERATIONS:
 * ==========================
 * 
 * - Uses type-specific vec_dot functions for optimal SIMD utilization
 * - Block-tiling with 16x16 blocks to improve cache locality
 * - Handles different data types (F32, F16, quantized) via type traits
 * - Minimizes memory bandwidth through NUMA-local access patterns
 * 
 * ============================================================================
 */

#include "../binary-ops.h"
#include "mul_mat.h"
#include "numa-kernels.h"
#include "../ggml-numa-shared.h"
#include "../ggml-numa-simple-coordinator.h"
#include "../ggml-cpu-impl.h"
#include "../ggml-impl.h"
#include "../vec.h"
#include "ggml-cpu.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

// ============================================================================
// Custom F16 Dot Product Implementation  
// ============================================================================

/**
 * Custom F16 dot product implementation optimized for NUMA execution
 * 
 * This function provides a specialized F16 dot product that can be tested
 * against the reference implementation for mathematical correctness.
 * 
 * @param n      Vector length
 * @param s      Output scalar result
 * @param s_off  Output offset (should be 0)
 * @param x      First vector (F16 data)  
 * @param x_off  First vector offset
 * @param y      Second vector (F16 data)
 * @param y_off  Second vector offset
 * @param nrc    Number of rows per call (should be 1)
 */
void ggml_numa_vec_dot_f16_custom(int n, float * restrict s, size_t s_off, 
                                 const void * restrict x, size_t x_off,
                                 const void * restrict y, size_t y_off, int nrc) {
    // Ensure single row operation
    assert(nrc == 1);
    assert(s_off == 0);
    
    const ggml_fp16_t * restrict x_f16 = (const ggml_fp16_t *)x + x_off;
    const ggml_fp16_t * restrict y_f16 = (const ggml_fp16_t *)y + y_off;
    
    // Custom F16 dot product implementation
    float sum = 0.0f;
    
    // Process in chunks of 4 for better cache utilization and potential SIMD
    int i = 0;
    for (; i + 3 < n; i += 4) {
        // Convert F16 to F32 and accumulate
        float x0 = ggml_fp16_to_fp32(x_f16[i + 0]);
        float y0 = ggml_fp16_to_fp32(y_f16[i + 0]);
        float x1 = ggml_fp16_to_fp32(x_f16[i + 1]);
        float y1 = ggml_fp16_to_fp32(y_f16[i + 1]);
        float x2 = ggml_fp16_to_fp32(x_f16[i + 2]);
        float y2 = ggml_fp16_to_fp32(y_f16[i + 2]);
        float x3 = ggml_fp16_to_fp32(x_f16[i + 3]);
        float y3 = ggml_fp16_to_fp32(y_f16[i + 3]);
        
        // Accumulate products (unrolled for better performance)
        sum += x0 * y0 + x1 * y1 + x2 * y2 + x3 * y3;
    }
    
    // Handle remaining elements
    for (; i < n; i++) {
        float x_val = ggml_fp16_to_fp32(x_f16[i]);
        float y_val = ggml_fp16_to_fp32(y_f16[i]);
        sum += x_val * y_val;
    }
    
    *s = sum;
}

// ============================================================================
// Type Traits Access for vec_dot Operations
// ============================================================================

// External type traits from ggml-cpu.c
extern const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type);

// ============================================================================
// Fast NUMA Slice Calculation for Matrix Operations
// ============================================================================

/**
 * Calculate NUMA node's slice for batch dimensions (optimized for cache locality)
 * 
 * For MUL_MAT, we slice along the batch dimensions (ne1*ne2*ne3) to distribute
 * different output matrix slices across NUMA nodes.
 * 
 * @param total_batch_size Total number of batch elements (ne1*ne2*ne3)
 * @param current_node     NUMA node executing this slice (0-based)
 * @param total_nodes      Total number of NUMA nodes participating
 * @param start            [OUT] Starting batch index for this node's slice
 * @param end              [OUT] Ending batch index (exclusive) for this node's slice
 */
static inline void get_numa_batch_slice(int64_t total_batch_size, 
                                       int current_node, 
                                       int total_nodes,
                                       int64_t * start, 
                                       int64_t * end) {
    if (total_nodes <= 1) {
        *start = 0;
        *end = total_batch_size;
        return;
    }
    
    const int64_t batch_per_node = total_batch_size / total_nodes;
    *start = current_node * batch_per_node;
    *end = (current_node == total_nodes - 1) ? total_batch_size : *start + batch_per_node;
}

/**
 * Calculate thread's slice within a NUMA node for row processing
 * 
 * @param total_rows  Total number of rows to process (ne0)
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
    const int64_t rows_per_thread = total_rows / num_threads;
    *start = thread_id * rows_per_thread;
    *end = (thread_id == num_threads - 1) ? total_rows : *start + rows_per_thread;
}

// ============================================================================
// NUMA Matrix Multiplication Kernel
// ============================================================================

/**
 * High-performance MUL_MAT kernel with NUMA-aware data distribution
 * 
 * This kernel implements the mathematical core of matrix multiplication
 * with optimal NUMA memory access patterns and SIMD utilization.
 * 
 * @param work_context  Tensor to process (cast from void*)
 * @param params        Threadpool parameters (thread ID, thread count)
 * @return              GGML_STATUS_SUCCESS on success, GGML_STATUS_FAILED on error
 */
enum ggml_status ggml_numa_kernel_mul_mat_execute(void * work_context, 
                                                  struct ggml_compute_params * params) {
    struct ggml_tensor * dst = (struct ggml_tensor *)work_context;
    
    // Fast validation - minimal checks
    if (!dst || !dst->src[0] || !dst->src[1]) {
        NUMA_LOG_DEBUG("MUL_MAT kernel: Invalid tensor pointers");
        return GGML_STATUS_FAILED;
    }
    
    const struct ggml_tensor * src0 = dst->src[0];  // Matrix A
    const struct ggml_tensor * src1 = dst->src[1];  // Matrix B
    
    // Extract tensor dimensions using GGML macros
    const int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    const size_t  nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];
    const size_t  nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    const int64_t ne0  = dst->ne[0],  ne1  = dst->ne[1],  ne2  = dst->ne[2],  ne3  = dst->ne[3];
    const size_t  nb0  = dst->nb[0],  nb1  = dst->nb[1],  nb2  = dst->nb[2],  nb3  = dst->nb[3];
    
    // Validate matrix multiplication constraints: C = A × B
    // A's columns (ne00) must match B's rows (ne10)
    // Result dimensions: ne0=A_rows(ne01), ne1=B_cols(ne11) 
    if (ne00 != ne10 || ne0 != ne01 || ne1 != ne11 || ne2 != ne12 || ne3 != ne13) {
        NUMA_LOG_DEBUG("MUL_MAT kernel: Dimension mismatch - A:[%ld,%ld] × B:[%ld,%ld] -> C:[%ld,%ld], batch:[%ld,%ld]", 
                       ne01, ne00, ne10, ne11, ne0, ne1, ne2, ne3);
        return GGML_STATUS_FAILED;
    }
    
    // Get type information for vec_dot operations
    const struct ggml_type_traits_cpu * type_traits = ggml_get_type_traits_cpu(src0->type);
    ggml_vec_dot_t const vec_dot = type_traits->vec_dot;
    enum ggml_type const vec_dot_type = type_traits->vec_dot_type;
    const int64_t vec_dot_num_rows = type_traits->nrows;
    
    // Check if we should use custom F16 dot product for testing
    bool use_custom_f16_dot = (src0->type == GGML_TYPE_F16 && vec_dot_type == GGML_TYPE_F16);
    
    // Get NUMA-local data pointers
    const void * src0_data = tensor_data(src0);
    const void * src1_data = tensor_data(src1);
    float * dst_data = (float *)tensor_data(dst);
    
    // Handle type conversion for src1 if needed (use work buffer from params)
    const void * wdata = (src1->type == vec_dot_type) ? src1_data : params->wdata;
    const size_t row_size = ggml_row_size(vec_dot_type, ne10);
    const bool src1_cont = ggml_is_contiguous(src1);
    
    // Calculate broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;
    
    // Validate broadcast constraints
    if (ne12 % ne02 != 0 || ne13 % ne03 != 0) {
        NUMA_LOG_DEBUG("MUL_MAT kernel: Invalid broadcast constraints");
        return GGML_STATUS_FAILED;
    }
    
    // Read NUMA context
    extern __thread bool ggml_numa_is_data_parallel_execution;
    extern __thread int ggml_numa_total_nodes_for_data_parallel;
    extern __thread int ggml_current_numa_node;
    
    const int current_node = ggml_current_numa_node;
    const int total_nodes = ggml_numa_is_data_parallel_execution ? 
                           ggml_numa_total_nodes_for_data_parallel : 1;
    const bool is_data_parallel = ggml_numa_is_data_parallel_execution;
    
    const int thread_id = params->ith;
    const int num_threads = params->nth;
    
    NUMA_LOG_DEBUG("MUL_MAT Node %d: src0_type=%d, src1_type=%d, vec_dot_type=%d, use_custom_f16=%s", 
                   current_node, src0->type, src1->type, vec_dot_type, use_custom_f16_dot ? "true" : "false");
    
    NUMA_LOG_DEBUG("MUL_MAT Node %d, Thread %d/%d: dims=[%ld,%ld,%ld,%ld] x [%ld,%ld,%ld,%ld] -> [%ld,%ld,%ld,%ld]", 
                   current_node, thread_id, num_threads,
                   ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3);
    
    // Use the original MUL_MAT work distribution pattern
    // The original algorithm divides work into chunks across nr0 (result rows) and nr1 (result columns * batch)
    const int64_t nr0 = ne0;  // Result rows
    const int64_t nr1 = ne1 * ne2 * ne3;  // Result columns * batch dimensions
    
    // Calculate this thread's work range using the original chunking algorithm
    int chunk_size = 16;
    if (nr0 == 1 || nr1 == 1) {
        chunk_size = 64;
    }
    
    int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;
    
    // For NUMA systems, distribute by thread (original behavior)
    if (nchunk0 * nchunk1 < num_threads * 4) {
        nchunk0 = nr0 > nr1 ? num_threads : 1;
        nchunk1 = nr0 > nr1 ? 1 : num_threads;
    }
    
    // CRITICAL FIX: If we have only 1 thread, it must process all chunks
    if (num_threads == 1) {
        nchunk0 = 1;
        nchunk1 = 1;
    }
    
    const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;
    
    // Calculate this thread's chunk assignment
    int64_t ith0, ith1;
    if (is_data_parallel && total_nodes > 1) {
        // For data-parallel mode, distribute chunks across NUMA nodes first
        int total_chunks = nchunk0 * nchunk1;
        int chunks_per_node = (total_chunks + total_nodes - 1) / total_nodes;
        int node_chunk_start = current_node * chunks_per_node;
        int node_chunk_end = MIN(node_chunk_start + chunks_per_node, total_chunks);
        
        // Then assign thread within node's chunk range
        int node_chunks = node_chunk_end - node_chunk_start;
        if (thread_id >= node_chunks) {
            // This thread has no work in this node
            return GGML_STATUS_SUCCESS;
        }
        
        int current_chunk = node_chunk_start + thread_id;
        ith0 = current_chunk % nchunk0;
        ith1 = current_chunk / nchunk0;
    } else {
        // Single-node mode: direct thread assignment
        if (thread_id >= nchunk0 * nchunk1) {
            return GGML_STATUS_SUCCESS;
        }
        ith0 = thread_id % nchunk0;
        ith1 = thread_id / nchunk0;
    }
    
    const int64_t ir0_start = dr0 * ith0;
    const int64_t ir0_end = MIN(ir0_start + dr0, nr0);
    const int64_t ir1_start = dr1 * ith1;
    const int64_t ir1_end = MIN(ir1_start + dr1, nr1);
    
    NUMA_LOG_DEBUG("MUL_MAT Node %d, Thread %d: ir0_range=[%ld,%ld), ir1_range=[%ld,%ld)", 
                   current_node, thread_id, ir0_start, ir0_end, ir1_start, ir1_end);
    
    // Early exit if no work assigned
    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        NUMA_LOG_DEBUG("MUL_MAT Node %d, Thread %d: NO WORK ASSIGNED - early exit", current_node, thread_id);
        return GGML_STATUS_SUCCESS;
    }
    
    NUMA_LOG_DEBUG("MUL_MAT Node %d, Thread %d: STARTING COMPUTATION LOOPS", current_node, thread_id);
    
    // Calculate src1 column stride for contiguous vs non-contiguous data
    const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;
    
    // Block tiling parameters for cache optimization (matching original implementation)
    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;
    
    // Main computation loop following original ggml_compute_forward_mul_mat_one_chunk pattern
    int total_operations = 0;
    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += vec_dot_num_rows) {
                // Convert linear batch index to 3D coordinates (matching original)
                const int64_t i13 = (ir1 / (ne12 * ne1));
                const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);
                
                // Broadcast src0 coordinates for this batch
                const int64_t i03 = i13 / r3;
                const int64_t i02 = i12 / r2;
                
                const int64_t i1 = i11;
                const int64_t i2 = i12;
                const int64_t i3 = i13;
                
                // Get pointers for this batch slice (matching original implementation)
                const char * src0_row = (const char*)src0_data + (0 + i02 * nb02 + i03 * nb03);
                
                // Calculate src1 column pointer based on data layout
                const char * src1_col = (const char*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                
                float * dst_col = (float*)((char*)dst_data + (i1 * nb1 + i2 * nb2 + i3 * nb3));
                
                // Process rows in the current block
                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += vec_dot_num_rows) {
                    if (vec_dot_num_rows == 1) {
                        // Single row case - use custom F16 dot product if applicable
                        if (use_custom_f16_dot) {
                            ggml_numa_vec_dot_f16_custom(ne00, &dst_col[ir0], 0, src0_row + ir0*nb01, 0, src1_col, 0, 1);
                        } else {
                            vec_dot(ne00, &dst_col[ir0], 0, src0_row + ir0*nb01, 0, src1_col, 0, 1);
                        }
                        total_operations++;
                    } else {
                        // Multi-row case for SIMD optimizations (matching original)
                        for (int cn = 0; cn < vec_dot_num_rows; ++cn) {
                            float * dst_ptr = &dst_col[ir0 + cn * nb1 / nb0];
                            const char * src0_ptr = src0_row + (ir0 + cn) * nb01;
                            const char * src1_ptr = src1_col + cn * src1_col_stride;
                            
                            if (use_custom_f16_dot) {
                                ggml_numa_vec_dot_f16_custom(ne00, dst_ptr, 0, src0_ptr, 0, src1_ptr, 0, 1);
                            } else {
                                vec_dot(ne00, dst_ptr, 0, src0_ptr, 0, src1_ptr, 0, 1);
                            }
                            total_operations++;
                        }
                    }
                }
            }
        }
    }
    
    NUMA_LOG_DEBUG("MUL_MAT Node %d, Thread %d: COMPLETED %d operations", current_node, thread_id, total_operations);
    
    return GGML_STATUS_SUCCESS;
}

// ============================================================================
// MUL_MAT Threshold-Based Strategy Selection
// ============================================================================

/**
 * MUL_MAT operation-specific thresholds for optimal strategy selection
 * These thresholds are tuned specifically for matrix multiplication characteristics
 */
typedef struct {
    size_t element_threshold;                      // Threshold in number of elements
    ggml_numa_execution_strategy_t strategy;      // Strategy to use
    size_t work_buffer_size_per_thread;          // Buffer size needed
    float efficiency_score;                       // Expected efficiency
    const char * kernel_name;                     // Strategy description
} ggml_mul_mat_strategy_threshold_t;

/**
 * MUL_MAT-specific strategy thresholds
 * Optimized for matrix multiplication workload characteristics
 */
static const ggml_mul_mat_strategy_threshold_t MUL_MAT_THRESHOLDS[] = {
    // Very small matrices: single-threaded is fastest due to low overhead
    {
        .element_threshold = 1024,  // < 1K elements
        .strategy = { .node_strategy = NUMA_NODE_STRATEGY_SINGLE, .on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD },
        .work_buffer_size_per_thread = 0,
        .efficiency_score = 0.90f,
        .kernel_name = "NUMA MUL_MAT (Single/Single)"
    },
    
    // Small matrices: multi-threaded on single node for cache efficiency
    {
        .element_threshold = 16384,  // 1K - 16K elements
        .strategy = { .node_strategy = NUMA_NODE_STRATEGY_SINGLE, .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD },
        .work_buffer_size_per_thread = 0,
        .efficiency_score = 0.92f,
        .kernel_name = "NUMA MUL_MAT (Single/Multi)"
    },
    
    // Medium matrices: start using data-parallel for memory bandwidth
    {
        .element_threshold = 262144,  // 16K - 256K elements
        .strategy = { .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL, .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD },
        .work_buffer_size_per_thread = 1024,
        .efficiency_score = 0.93f,
        .kernel_name = "NUMA MUL_MAT (Data-Parallel/Multi)"
    },
    
    // Large matrices: data-parallel with larger buffers
    {
        .element_threshold = 4194304,  // 256K - 4M elements
        .strategy = { .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL, .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD },
        .work_buffer_size_per_thread = 4096,
        .efficiency_score = 0.95f,
        .kernel_name = "NUMA MUL_MAT (Data-Parallel/Large)"
    },
    
    // Huge matrices: optimized for memory bandwidth utilization
    {
        .element_threshold = 67108864,  // 4M - 64M elements
        .strategy = { .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL, .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD },
        .work_buffer_size_per_thread = 8192,
        .efficiency_score = 0.96f,
        .kernel_name = "NUMA MUL_MAT (Data-Parallel/Huge)"
    },
    
    // GB-scale: 1GB+ matrices with maximum buffer efficiency
    {
        .element_threshold = 268435456,  // 64M - 256M elements (~1GB)
        .strategy = { .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL, .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD },
        .work_buffer_size_per_thread = 16384,
        .efficiency_score = 0.97f,
        .kernel_name = "NUMA MUL_MAT (Data-Parallel/1GB)"
    },
    
    // Ultra-large: 2GB+ matrices with optimized chunk distribution
    {
        .element_threshold = SIZE_MAX,  // 256M+ elements (2GB+)
        .strategy = { .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL, .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD },
        .work_buffer_size_per_thread = 32768,
        .efficiency_score = 0.98f,
        .kernel_name = "NUMA MUL_MAT (Data-Parallel/Ultra)"
    }
};

#define MUL_MAT_THRESHOLD_COUNT (sizeof(MUL_MAT_THRESHOLDS) / sizeof(MUL_MAT_THRESHOLDS[0]))

/**
 * Query MUL_MAT kernel for optimal strategy based on tensor characteristics
 * 
 * This function analyzes the tensor and returns the optimal execution strategy
 * without requiring exact complexity class matching.
 * 
 * @param tensor The tensor to analyze
 * @return Query result with optimal strategy, or unsupported result if not applicable
 */
ggml_numa_kernel_query_result_t ggml_numa_kernel_mul_mat_query(const struct ggml_tensor * tensor) {
    ggml_numa_kernel_query_result_t result = { .supported = false };
    
    // Validate this is a MUL_MAT operation
    if (!tensor || tensor->op != GGML_OP_MUL_MAT) {
        return result;
    }
    
    // Validate tensor structure
    if (!tensor->src[0] || !tensor->src[1]) {
        NUMA_LOG_DEBUG("MUL_MAT query: Missing source tensors");
        return result;
    }
    
    const struct ggml_tensor * src0 = tensor->src[0];
    const struct ggml_tensor * src1 = tensor->src[1];
    
    // CRITICAL: Only support F32 operations for now
    // Our kernel was developed and tested with F32 data only.
    // Quantized types (Q8_0, etc.) require additional validation and testing.
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) {
        NUMA_LOG_DEBUG("MUL_MAT query: REJECTING - src0_type=%d, src1_type=%d (only F32 supported)", 
                       src0->type, src1->type);
        result.supported = false;
        return result;
    }
    
    // Calculate total elements in result tensor
    const size_t total_elements = ggml_nelements(tensor);
    
    // Find optimal strategy using threshold search
    const ggml_mul_mat_strategy_threshold_t * selected_strategy = &MUL_MAT_THRESHOLDS[MUL_MAT_THRESHOLD_COUNT - 1];
    
    for (size_t i = 0; i < MUL_MAT_THRESHOLD_COUNT; i++) {
        if (total_elements < MUL_MAT_THRESHOLDS[i].element_threshold) {
            selected_strategy = &MUL_MAT_THRESHOLDS[i];
            break;
        }
    }
    
    // Build successful query result
    result.supported = true;
    result.strategy = selected_strategy->strategy;
    result.work_buffer_size_per_thread = selected_strategy->work_buffer_size_per_thread;
    result.work_function = ggml_numa_kernel_mul_mat_execute;
    result.efficiency_score = selected_strategy->efficiency_score;
    result.kernel_name = selected_strategy->kernel_name;
    
    NUMA_LOG_DEBUG("MUL_MAT query: ACCEPTING F32 - %zu elements -> %s (efficiency: %.2f)", 
                   total_elements, result.kernel_name, result.efficiency_score);
    
    return result;
}

// ============================================================================
// Legacy Cache Population for Backward Compatibility
// ============================================================================

void ggml_numa_kernel_mul_mat_populate_cache(ggml_numa_cache_entry_t cache_entries[COMPLEXITY_COUNT]) {
    NUMA_LOG_DEBUG("Populating MUL_MAT NUMA cache entries (legacy compatibility)");
    
    // CRITICAL: Disable legacy cache to force use of query-based validation
    // The legacy cache bypasses our type validation in the query function.
    // All cache entries are set to invalid to prevent fallback execution.
    
    for (int i = 0; i < COMPLEXITY_COUNT; i++) {
        cache_entries[i] = (ggml_numa_cache_entry_t){
            .valid = false,
            .strategy = { .node_strategy = NUMA_NODE_STRATEGY_SINGLE, .on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD },
            .work_buffer_size_per_thread = 0,
            .work_function = NULL,
            .efficiency_score = 0.0f,
            .kernel_name = "MUL_MAT Cache Disabled"
        };
    }
}
