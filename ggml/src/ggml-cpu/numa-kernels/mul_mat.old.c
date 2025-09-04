/**
 * @file mul_mat.c
 * @brief NUMA Kernel Template: Complex Operations (Matrix Multiplication)
 * 
 * ============================================================================
 * NUMA KERNEL TEMPLATE: COMPLEX OPERATIONS (MUL_MAT)
 * ============================================================================
 * 
 * This file serves as the CANONICAL TEMPLATE for implementing NUMA kernels 
 * for COMPLEX OPERATIONS that require specialized parallelization strategies.
 * 
 * USE THIS TEMPLATE FOR:
 * =====================
 * ✅ Matrix operations (MUL_MAT, CONV_1D, CONV_2D)
 * ✅ Complex transformations requiring multidimensional slicing
 * ✅ Operations with intricate data dependencies
 * ✅ Operations requiring specialized patterns and strategies
 * ✅ Operations with custom aggregation requirements
 * ✅ Operations needing chunk-based or block-based processing
 * 
 * DO NOT USE THIS TEMPLATE FOR:
 * ============================
 * ❌ Simple element-wise operations (ADD, MUL, SUB, DIV) → Use add.c template
 * ❌ Simple reductions (SUM, MEAN) → Use rms_norm.c template  
 * ❌ Single-pass operations → Use add.c template
 * ❌ Operations with uniform data access patterns → Use add.c template
 * 
 * TEMPLATE RATIONALE:
 * ==================
 * Complex operations like matrix multiplication require sophisticated NUMA 
 * parallelization strategies due to:
 * - Non-uniform memory access patterns
 * - Multidimensional data slicing requirements  
 * - Cache optimization considerations
 * - Specialized SIMD vectorization patterns
 * - Custom work distribution algorithms
 * 
 * MATHEMATICAL OPERATION (MUL_MAT EXAMPLE):
 * ========================================
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
 * COMPLEX PARALLELIZATION STRATEGY:
 * =================================
 * 
 * The MUL_MAT operation demonstrates complex parallelization along multiple
 * dimensions with sophisticated chunk-based distribution:
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
#include "../ggml-numa-perf.h"  // Performance instrumentation
#include "../ggml-cpu-impl.h"
#include "../ggml-impl.h"

#ifdef GGML_USE_NUMA
#include <numa.h>
#endif
#include "../vec.h"
#include "ggml-cpu.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

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
    
    // PERFORMANCE: Start timing the kernel execution
    extern __thread int ggml_current_numa_node;
    extern __thread bool ggml_numa_is_data_parallel_execution;
    const size_t tensor_size = ggml_nelements(dst) * sizeof(float);
    NUMA_PERF_START(NUMA_PERF_KERNEL_NUMA_EXEC, "MUL_MAT", "kernel_execute", ggml_current_numa_node, tensor_size, params->nth);
    
    // FORCED DEBUG: Always log MUL_MAT kernel execution for verification
    NUMA_LOG_DEBUG("MUL_MAT kernel called: data_parallel=%s, nth=%d, numa_node=%d", 
                   ggml_numa_is_data_parallel_execution ? "YES" : "NO", params->nth, ggml_current_numa_node);
    
    // Log execution strategy in standardized format for integration test parsing
    // These strategy logs use debug level 1 and should be visible with GGML_NUMA_DEBUG=1
    if (ggml_numa_is_data_parallel_execution) {
        NUMA_LOG_STRATEGY_DATA_PARALLEL("MUL_MAT");
    } else if (params->nth > 1) {
        NUMA_LOG_STRATEGY_SINGLE_MULTI("MUL_MAT");
    } else {
        NUMA_LOG_STRATEGY_SINGLE_SINGLE("MUL_MAT");
    }
    
    // =============================================================================
    // Input Validation & Setup
    // =============================================================================
    
    if (!dst || !dst->src[0] || !dst->src[1]) {
        NUMA_LOG_DEBUG("MUL_MAT kernel: Invalid tensor pointers");
        return GGML_STATUS_FAILED;
    }
    
    const struct ggml_tensor * src0 = dst->src[0];  // Matrix A (left operand)
    const struct ggml_tensor * src1 = dst->src[1];  // Matrix B (right operand)
    
    // Extract tensor dimensions using GGML macros
    // src0 dimensions: ne00=K, ne01=M (A is M×K)
    // src1 dimensions: ne10=K, ne11=N (B is K×N) 
    // dst dimensions:  ne0=M,  ne1=N  (C is M×N)
    const int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    const size_t  nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];
    const size_t  nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    const int64_t ne0  = dst->ne[0],  ne1  = dst->ne[1],  ne2  = dst->ne[2],  ne3  = dst->ne[3];
    const size_t  nb0  = dst->nb[0],  nb1  = dst->nb[1],  nb2  = dst->nb[2],  nb3  = dst->nb[3];
    
    // Validate matrix multiplication constraints: C = A × B
    // - A's columns (ne00) must match B's rows (ne10) for valid multiplication
    // - Result dimensions must match expected output: M×N where M=ne01, N=ne11
    if (ne00 != ne10 || ne0 != ne01 || ne1 != ne11 || ne2 != ne12 || ne3 != ne13) {
        NUMA_LOG_DEBUG("MUL_MAT kernel: Dimension mismatch - A:[%ld,%ld] × B:[%ld,%ld] -> C:[%ld,%ld], batch:[%ld,%ld]", 
                       ne01, ne00, ne10, ne11, ne0, ne1, ne2, ne3);
        return GGML_STATUS_FAILED;
    }
    
    // =============================================================================
    // Type System & Vector Dot Product Setup
    // =============================================================================
    
    // Get type information for optimized vector operations
    // Each quantization type (F32, Q8_0, etc.) has specific traits:
    // - vec_dot: optimized function for computing dot products
    // - vec_dot_type: the type both operands must be converted to
    // - nrows: number of rows processed per vec_dot call (SIMD optimization)
    const struct ggml_type_traits_cpu * type_traits = ggml_get_type_traits_cpu(src0->type);
    ggml_vec_dot_t const vec_dot = type_traits->vec_dot;
    enum ggml_type const vec_dot_type = type_traits->vec_dot_type;
    const int64_t vec_dot_num_rows = type_traits->nrows;
    
    // =============================================================================
    // NUMA-Aware Data Access 
    // =============================================================================

    // Get NUMA-local data pointers - this ensures we access data from the 
    // appropriate NUMA node to minimize memory latency
    const void * src0_data = tensor_data(src0);
    const void * src1_data = tensor_data(src1);
    
    // For kernels with GGML_NUMA_AGGREGATION_NEVER policy, write directly to shared result tensor
    // This eliminates the need for data aggregation across NUMA nodes
    extern __thread void * ggml_numa_shared_result_tensor_data;
    float * dst_data;
    if (ggml_numa_shared_result_tensor_data != NULL) {
        // Use shared result tensor memory - eliminates aggregation overhead
        dst_data = (float *)ggml_numa_shared_result_tensor_data;
        NUMA_LOG_VERBOSE("MUL_MAT kernel using shared result tensor memory");
    } else {
        // Fallback to local tensor data for compatibility
        dst_data = (float *)tensor_data(dst);
        NUMA_LOG_VERBOSE("MUL_MAT kernel using local tensor memory");
    }    // =============================================================================
    // Memory Layout & Broadcasting Setup
    // =============================================================================
    
    // Calculate memory layout parameters
    const size_t row_size = ggml_row_size(vec_dot_type, ne10);  // Size of one converted row
    const bool src1_cont = ggml_is_contiguous(src1);            // Is src1 contiguous in memory?
    
    // Broadcasting factors for handling different batch dimensions
    // This allows operations like [A,B,1,1] × [A,B,C,D] where C,D are broadcast
    const int64_t r2 = ne12 / ne02;  // Broadcast factor for dimension 2
    const int64_t r3 = ne13 / ne03;  // Broadcast factor for dimension 3
    
    // Validate that broadcasting is mathematically valid
    if (ne12 % ne02 != 0 || ne13 % ne03 != 0) {
        NUMA_LOG_DEBUG("MUL_MAT kernel: Invalid broadcast constraints");
        return GGML_STATUS_FAILED;
    }
    
    // =============================================================================
    // NUMA Execution Context
    // =============================================================================
    
    // Read NUMA execution context from thread-local variables
    // These are set by the NUMA coordinator to control data distribution
    extern __thread bool ggml_numa_is_data_parallel_execution;
    extern __thread int ggml_numa_total_nodes_for_data_parallel;
    extern __thread int ggml_current_numa_node;
    
    const int current_node = ggml_current_numa_node;
    
    // Debug: Log which vec_dot function and type traits we're using
    NUMA_LOG_VERBOSE("MUL_MAT Node %d: Type traits - src0_type=%d, vec_dot_type=%d, nrows=%ld, vec_dot_func=%p", 
                     current_node, src0->type, vec_dot_type, vec_dot_num_rows, (void*)vec_dot);
    const int total_nodes = ggml_numa_is_data_parallel_execution ? 
                           ggml_numa_total_nodes_for_data_parallel : 1;
    const bool is_data_parallel = ggml_numa_is_data_parallel_execution;

    // NUMA OPTIMIZATION: In data-parallel mode, write results directly to shared tensor data
    // This eliminates the need for complex aggregation logic at the end
    if (is_data_parallel && ggml_numa_shared_result_tensor_data) {
        dst_data = (float *)ggml_numa_shared_result_tensor_data;
        NUMA_LOG_VERBOSE("MUL_MAT Node %d: Using shared tensor data at %p (no aggregation needed)", 
                         current_node, dst_data);
    } else {
        NUMA_LOG_VERBOSE("MUL_MAT Node %d: Using local tensor data at %p", current_node, dst_data);
    }    const int thread_id = params->ith;      // Thread ID within this execution
    const int num_threads = params->nth;    // Total threads for this execution
    
    // =============================================================================
    // Type Conversion Setup (Critical for Quantized Operations)
    // =============================================================================
    
    // For quantized operations like Q8_0, both operands must be the same type
    // Example: Q8_0 × F32 -> convert F32 to Q8_0, then Q8_0 × Q8_0
    // This follows the exact same pattern as the reference implementation
    const void * wdata;  // Pointer to data for src1 (either original or converted)
    
    if (src1->type != vec_dot_type) {
        // Case: Type conversion needed (e.g., F32 -> Q8_0)
        
        if (src1->type != GGML_TYPE_F32) {
            NUMA_LOG_DEBUG("MUL_MAT Node %d: ERROR - src1 must be F32 when conversion is needed, got type %d", 
                           current_node, src1->type);
            return GGML_STATUS_FAILED;
        }
        
        // Get the conversion function (F32 -> target type)
        ggml_from_float_t const from_float = ggml_get_type_traits_cpu(vec_dot_type)->from_float;
        if (!from_float) {
            NUMA_LOG_DEBUG("MUL_MAT Node %d: ERROR - no from_float function for vec_dot_type %d", 
                           current_node, vec_dot_type);
            return GGML_STATUS_FAILED;
        }
        
        // Use pre-allocated work buffer (managed by NUMA coordinator)
        // This is much more efficient than malloc/free for each operation
        char * work_buffer = params->wdata;
        if (!work_buffer) {
            NUMA_LOG_DEBUG("MUL_MAT Node %d: ERROR - work buffer is null but conversion needed", current_node);
            return GGML_STATUS_FAILED;
        }
        
        // Calculate buffer layout following reference implementation exactly
        // This ensures proper memory alignment and efficient access patterns
        const size_t nbw0 = ggml_type_size(vec_dot_type);      // Size per element
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10); // Size per row  
        const size_t nbw2 = nbw1 * ne11;                       // Size per matrix
        const size_t nbw3 = nbw2 * ne12;                       // Size per batch
        const size_t total_size = ne13 * nbw3;                 // Total required size
        
        // Verify we have sufficient buffer space
        if (params->wsize < total_size) {
            NUMA_LOG_DEBUG("MUL_MAT Node %d: ERROR - work buffer too small: have %zu, need %zu", 
                           current_node, params->wsize, total_size);
            return GGML_STATUS_FAILED;
        }
        
        // Convert src1 data from F32 to vec_dot_type 
        // We use the reference implementation's pattern for thread-safe conversion
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    // Source: F32 data at the current batch/matrix/row position
                    const float * src1_row = (const float *)((const char *)src1_data + 
                                             i13*nb13 + i12*nb12 + i11*nb11);
                    
                    // Destination: converted data in work buffer with proper layout
                    void * dst_row = work_buffer + i13*nbw3 + i12*nbw2 + i11*nbw1;
                    
                    // DEBUG: Log conversion details for first few rows (TRACE level)
                    if (i13 == 0 && i12 == 0 && i11 < 3) {
                        NUMA_LOG_TRACE("MUL_MAT Node %d: Converting F32->Q8_0 row[%ld]: src_ptr=%p, dst_ptr=%p", 
                                       current_node, i11, src1_row, dst_row);
                        NUMA_LOG_TRACE("MUL_MAT Node %d: F32 row[%ld] samples: {%.3f,%.3f,%.3f,%.3f,%.3f}", 
                                       current_node, i11, src1_row[0], src1_row[1], src1_row[2], src1_row[3], src1_row[4]);
                    }
                    
                    // Perform the actual conversion (e.g., F32 -> Q8_0)
                    from_float(src1_row, dst_row, ne10);
                    
                    // DEBUG: Log converted values for verification (TRACE level)
                    if (i13 == 0 && i12 == 0 && i11 < 3) {
                        const uint8_t* q8_data = (const uint8_t*)dst_row;
                        NUMA_LOG_TRACE("MUL_MAT Node %d: Q8_0 row[%ld] first bytes: {%02x,%02x,%02x,%02x,%02x}", 
                                       current_node, i11, q8_data[0], q8_data[1], q8_data[2], q8_data[3], q8_data[4]);
                    }
                }
            }
        }
        
        wdata = work_buffer;  // Use converted data
        
        NUMA_LOG_VERBOSE("MUL_MAT Node %d: Converted src1 from F32 to vec_dot_type %d using work buffer (size=%zu bytes)", 
                         current_node, vec_dot_type, total_size);
    } else {
        // Case: No conversion needed - both operands already have compatible types
        wdata = src1_data;
        NUMA_LOG_VERBOSE("MUL_MAT Node %d: Using src1 data directly (no conversion needed)", current_node);
    }
    
    // Debug logging for execution context
    NUMA_LOG_VERBOSE("MUL_MAT Node %d: src0_type=%d, src1_type=%d, vec_dot_type=%d", 
                     current_node, src0->type, src1->type, vec_dot_type);
    
    NUMA_LOG_VERBOSE("MUL_MAT Node %d: Conversion check - src1->type=%d, vec_dot_type=%d, needs_conversion=%s", 
                     current_node, src1->type, vec_dot_type, (src1->type != vec_dot_type) ? "YES" : "NO");
    
    NUMA_LOG_VERBOSE("MUL_MAT Node %d, Thread %d/%d: dims=[%ld,%ld,%ld,%ld] x [%ld,%ld,%ld,%ld] -> [%ld,%ld,%ld,%ld]", 
                     current_node, thread_id + 1, num_threads,
                     ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3);
    
    // =============================================================================
    // Work Distribution Algorithm (Following Reference Implementation)
    // =============================================================================
    
    // The reference implementation uses a sophisticated work distribution algorithm
    // that balances cache efficiency with parallelization overhead
    
    // Step 1: Calculate total work units
    const int64_t nr0 = ne0;                    // Result rows (M dimension)
    const int64_t nr1 = ne1 * ne2 * ne3;       // Result columns × batch (N × batch)
    
    // Step 2: Determine chunk size based on problem characteristics
    int chunk_size = 16;  // Default block size for cache efficiency
    if (nr0 == 1 || nr1 == 1) {
        chunk_size = 64;  // Larger chunks for skinny matrices to reduce overhead
    }
    
    // Step 3: Calculate number of chunks in each dimension
    int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;  // Row chunks
    int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;  // Column chunks
    
    // Step 4: Apply NUMA-specific optimizations
    // For NUMA systems, thread-based distribution often performs better than
    // fine-grained chunking due to memory locality considerations
    if (nchunk0 * nchunk1 < num_threads * 4) {
        // If we don't have enough chunks to keep all threads busy,
        // switch to simpler thread-based distribution
        nchunk0 = nr0 > nr1 ? num_threads : 1;  // Parallelize along the larger dimension
        nchunk1 = nr0 > nr1 ? 1 : num_threads;
    }
    
    // Step 5: Handle single-threaded case
    if (num_threads == 1) {
        // Single thread processes everything - no need for complex chunking
        nchunk0 = 1;
        nchunk1 = 1;
    }
    
    // Step 6: Calculate chunk sizes
    const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;  // Rows per chunk
    const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;  // Columns per chunk
    
    // =============================================================================
    // Thread Work Assignment
    // =============================================================================
    
    // Assign work to this specific thread, considering NUMA data distribution
    int64_t ir0_start, ir0_end, ir1_start, ir1_end;  // Work ranges for this thread
    
    if (is_data_parallel && total_nodes > 1) {
        // Data-parallel mode: use clean row-based splitting for memory isolation
        // Each NUMA node processes a contiguous range of output rows
        
        const int64_t rows_per_node = (nr0 + total_nodes - 1) / total_nodes;
        const int64_t node_row_start = current_node * rows_per_node;
        const int64_t node_row_end = MIN(node_row_start + rows_per_node, nr0);
        
        // Within the node's row range, distribute work among threads
        const int64_t node_rows = node_row_end - node_row_start;
        if (node_rows <= 0) {
            // This NUMA node has no rows assigned
            goto cleanup;
        }
        
        const int64_t rows_per_thread = (node_rows + num_threads - 1) / num_threads;
        const int64_t thread_row_start = thread_id * rows_per_thread;
        const int64_t thread_row_end = MIN(thread_row_start + rows_per_thread, node_rows);
        
        if (thread_row_start >= thread_row_end) {
            // This thread has no work assigned within this NUMA node
            goto cleanup;
        }
        
        // Final row range for this thread (absolute coordinates)
        ir0_start = node_row_start + thread_row_start;
        ir0_end = node_row_start + thread_row_end;
        
        // All threads process all columns (no column-wise splitting in data-parallel mode)
        ir1_start = 0;
        ir1_end = nr1;
        
        NUMA_LOG_VERBOSE("MUL_MAT Node %d, Thread %d: Data-parallel row split - node_rows=[%ld,%ld), thread_rows=[%ld,%ld), final=[%ld,%ld)", 
                       current_node, thread_id, node_row_start, node_row_end, thread_row_start, thread_row_end, ir0_start, ir0_end);
        
    } else {
        // Single-node mode: use original chunk-based assignment
        int64_t ith0, ith1;  // This thread's chunk coordinates
        
        if (thread_id >= nchunk0 * nchunk1) {
            // This thread has no work assigned
            goto cleanup;
        }
        ith0 = thread_id % nchunk0;
        ith1 = thread_id / nchunk0;
        
        // Calculate this thread's actual work range
        ir0_start = dr0 * ith0;                    // Starting row
        ir0_end = MIN(ir0_start + dr0, nr0);       // Ending row (exclusive)
        ir1_start = dr1 * ith1;                    // Starting column
        ir1_end = MIN(ir1_start + dr1, nr1);       // Ending column (exclusive)
        
        NUMA_LOG_VERBOSE("MUL_MAT Node %d, Thread %d: Single-node chunk - ir0_range=[%ld,%ld), ir1_range=[%ld,%ld)", 
                        current_node, thread_id, ir0_start, ir0_end, ir1_start, ir1_end);
    }
    
    // Early exit if no work assigned to this thread
    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        NUMA_LOG_VERBOSE("MUL_MAT Node %d, Thread %d: NO WORK ASSIGNED - early exit", current_node, thread_id);
        goto cleanup;
    }
    
    NUMA_LOG_VERBOSE("MUL_MAT Node %d, Thread %d: STARTING COMPUTATION LOOPS", current_node, thread_id);
    
    // =============================================================================
    // Matrix Multiplication Core Algorithm
    // =============================================================================
    
    // Calculate stride for accessing converted src1 data
    // This handles both contiguous and non-contiguous memory layouts
    const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;
    
    // Block tiling parameters - these values are tuned for optimal cache usage
    const int64_t blck_0 = 16;  // Row block size
    const int64_t blck_1 = 16;  // Column block size
    
    // Main computation loop using cache-friendly block tiling
    // This follows the exact pattern from the reference implementation
    
    // PERFORMANCE: Start timing the core computation phase
    NUMA_PERF_START(NUMA_PERF_KERNEL_NUMA_EXEC, "MUL_MAT", "computation_loop", current_node, tensor_size, num_threads);
    
    int total_operations = 0;
    
    // Outer loop: process column blocks
    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        // Middle loop: process row blocks  
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            // Inner loop: process individual columns within block
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += vec_dot_num_rows) {
                
                // Convert linear column index back to 3D batch coordinates
                // This handles multi-dimensional tensors and broadcasting
                const int64_t i13 = (ir1 / (ne12 * ne1));                    // Batch dimension 3
                const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;         // Batch dimension 2  
                const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);   // Column index
                
                // Apply broadcasting for src0 coordinates
                const int64_t i03 = i13 / r3;  // Broadcast along dimension 3
                const int64_t i02 = i12 / r2;  // Broadcast along dimension 2
                
                // Set up batch coordinates for result tensor
                const int64_t i1 = i11;  // Column in result
                const int64_t i2 = i12;  // Batch dimension 2 in result
                const int64_t i3 = i13;  // Batch dimension 3 in result
                
                // Calculate memory pointers for this batch slice
                
                // src0: Get pointer to the start of relevant rows for this batch
                const char * src0_row = (const char*)src0_data + (0 + i02 * nb02 + i03 * nb03);
                
                // src1: Get pointer to the column data (either original or converted)
                // The complex addressing handles both contiguous and strided layouts
                const char * src1_col = (const char*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size  // Contiguous layout
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));           // Strided layout
                
                // dst: Get pointer to output column for this batch
                float * dst_col = (float*)((char*)dst_data + (i1 * nb1 + i2 * nb2 + i3 * nb3));
                
                // Inner-most loop: compute dot products for rows within the current block
                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += vec_dot_num_rows) {
                    
                    if (vec_dot_num_rows == 1) {
                        // Single row case: compute one dot product
                        
                        // DETAILED VEC_DOT DEBUGGING: Create wrapper to log internal computation
                        // This logs what happens inside the vec_dot function call
                        float * dst_ptr = &dst_col[ir0];
                        const void * src0_ptr = src0_row + ir0*nb01;
                        const void * src1_ptr = src1_col;
                        
                        // Pre-computation logging for extreme tracing
                        if (ir0 < 3 && i11 < 3) {
                            NUMA_LOG_TRACE("MUL_MAT Node %d: PRE vec_dot[%ld,%ld]: dst_ptr=%p, dst_value=%.6f", 
                                          current_node, ir0, i11, dst_ptr, *dst_ptr);
                            
                            // Log first few bytes of source data for comparison
                            const uint8_t* src0_bytes = (const uint8_t*)src0_ptr;
                            const uint8_t* src1_bytes = (const uint8_t*)src1_ptr;
                            NUMA_LOG_TRACE("MUL_MAT Node %d: PRE src0[0-7]={%02x,%02x,%02x,%02x,%02x,%02x,%02x,%02x}", 
                                          current_node, src0_bytes[0], src0_bytes[1], src0_bytes[2], src0_bytes[3],
                                          src0_bytes[4], src0_bytes[5], src0_bytes[6], src0_bytes[7]);
                            NUMA_LOG_TRACE("MUL_MAT Node %d: PRE src1[0-7]={%02x,%02x,%02x,%02x,%02x,%02x,%02x,%02x}", 
                                          current_node, src1_bytes[0], src1_bytes[1], src1_bytes[2], src1_bytes[3],
                                          src1_bytes[4], src1_bytes[5], src1_bytes[6], src1_bytes[7]);
                        }
                        
                        // Use optimized type-specific vec_dot function
                        // This is where the actual mathematical computation happens
                        vec_dot(ne00, dst_ptr, 0, src0_ptr, 0, src1_ptr, 0, 1);
                        
                        // Post-computation logging for extreme tracing
                        if (ir0 < 3 && i11 < 3) {
                            NUMA_LOG_TRACE("MUL_MAT Node %d: POST vec_dot[%ld,%ld]: dst_value=%.8f", 
                                          current_node, ir0, i11, *dst_ptr);
                        }
                        
                        total_operations++;
                        
                    } else {
                        // Multi-row case: SIMD optimization processes multiple rows at once
                        // This is used on architectures that support wider SIMD operations
                        
                        for (int cn = 0; cn < vec_dot_num_rows; ++cn) {
                            float * dst_ptr = &dst_col[ir0 + cn * nb1 / nb0];
                            const char * src0_ptr = src0_row + (ir0 + cn) * nb01;
                            const char * src1_ptr = src1_col + cn * src1_col_stride;
                            
                            vec_dot(ne00, dst_ptr, 0, src0_ptr, 0, src1_ptr, 0, 1);
                            total_operations++;
                        }
                    }
                }
            }
        }
    }
    
    NUMA_LOG_DEBUG("MUL_MAT Node %d, Thread %d: COMPLETED %d operations", current_node, thread_id, total_operations);
    
    // PERFORMANCE: End timing the core computation phase
    NUMA_PERF_END();
    
cleanup:
    // PERFORMANCE: End timing the overall kernel execution
    NUMA_PERF_END();
    
    // No explicit cleanup needed - work buffer is managed by NUMA coordinator
    // Memory allocations are handled at a higher level for efficiency
    return GGML_STATUS_SUCCESS;
}

// ============================================================================
// Dynamic Work Buffer Size Calculation
// ============================================================================

/**
 * Calculate the exact work buffer size needed for MUL_MAT operation
 * 
 * This function analyzes the tensor types and dimensions to determine
 * the precise buffer size needed for type conversion operations.
 * 
 * @param tensor The MUL_MAT tensor to analyze
 * @return Size in bytes needed for work buffer, or 0 if no buffer needed
 */
size_t ggml_numa_kernel_mul_mat_calculate_work_buffer_size(const struct ggml_tensor * tensor) {
    if (!tensor || !tensor->src[0] || !tensor->src[1]) {
        return 0;
    }
    
    const struct ggml_tensor * src0 = tensor->src[0];
    const struct ggml_tensor * src1 = tensor->src[1];
    
    // Get type information for optimized vector operations
    const struct ggml_type_traits_cpu * type_traits = ggml_get_type_traits_cpu(src0->type);
    if (!type_traits) {
        return 0;
    }
    
    const enum ggml_type vec_dot_type = type_traits->vec_dot_type;
    
    // Check if src1 needs type conversion
    if (src1->type == vec_dot_type) {
        // No conversion needed
        return 0;
    }
    
    // src1 needs conversion from F32 to vec_dot_type
    if (src1->type != GGML_TYPE_F32) {
        // We only support F32 -> vec_dot_type conversion
        return 0;
    }
    
    // Calculate buffer size for converting entire src1 tensor
    const int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];
    
    // Calculate buffer layout following reference implementation exactly
    const size_t nbw0 = ggml_type_size(vec_dot_type);        // Size per element
    const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);   // Size per row  
    const size_t nbw2 = nbw1 * ne11;                         // Size per matrix
    const size_t nbw3 = nbw2 * ne12;                         // Size per batch
    const size_t total_size = ne13 * nbw3;                   // Total required size
    
    NUMA_LOG_DEBUG("MUL_MAT buffer calc: src1_type=%d, vec_dot_type=%d, dims=[%ld,%ld,%ld,%ld], buffer_size=%zu", 
                   src1->type, vec_dot_type, ne10, ne11, ne12, ne13, total_size);
    
    return total_size;
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
    float efficiency_score;                       // Expected efficiency
    const char * kernel_name;                     // Strategy description
} ggml_mul_mat_strategy_threshold_t;

/**
 * @brief MUL_MAT kernel strategy thresholds now use cache-based configuration
 * 
 * Thresholds are stored in the kernel registry cache as single source of truth.
 * Cache registration defines: 128 (single-single), 512 (single-multi) element thresholds.
 * Above 512 elements: data-parallel across NUMA nodes.
 * 
 * Previous strategy threshold structure removed - cache serves as single source of truth.
 */

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
    // FORCED DEBUG: Always log MUL_MAT query calls for verification
    NUMA_LOG_DEBUG("MUL_MAT query called for tensor with op=%d", tensor ? tensor->op : -1);
    
    // PERFORMANCE: Start timing the query operation
    NUMA_PERF_START(NUMA_PERF_EXECUTOR_QUERY, "MUL_MAT", "query_phase", -1, 0, 0);
    
    ggml_numa_kernel_query_result_t result = { .supported = false };
    
    // Validate this is a MUL_MAT operation
    if (!tensor || tensor->op != GGML_OP_MUL_MAT) {
        NUMA_PERF_END();
        return result;
    }
    
    // Validate tensor structure
    if (!tensor->src[0] || !tensor->src[1]) {
        NUMA_LOG_DEBUG("MUL_MAT query: Missing source tensors");
        return result;
    }
    
    const struct ggml_tensor * src0 = tensor->src[0];
    const struct ggml_tensor * src1 = tensor->src[1];
    
    // Verify that we have type traits for src0 (needed for vec_dot operations)
    const struct ggml_type_traits_cpu * type_traits = ggml_get_type_traits_cpu(src0->type);
    if (!type_traits || !type_traits->vec_dot) {
        NUMA_LOG_DEBUG("MUL_MAT query: REJECTING - src0_type=%d has no vec_dot support", src0->type);
        result.supported = false;
        return result;
    }
    
    // Quantized matrix operations are fully supported
    // The NUMA kernel handles Q8_0 × F32, Q4_0 × F32, etc. operations efficiently
    NUMA_LOG_DEBUG("MUL_MAT query: ACCEPTING - src0_type=%d, src1_type=%d", src0->type, src1->type);
    
    // Check if this kernel is actually registered and supported
    if (!ggml_numa_is_kernel_supported(GGML_OP_MUL_MAT)) {
        NUMA_LOG_DEBUG("MUL_MAT kernel not supported - registration disabled");
        result.supported = false;
        return result;
    }
    
    // Calculate total elements in result tensor
    // Get cache entry for this operation
    const ggml_numa_kernel_cache_entry_t * cache_entry = ggml_numa_lookup_kernel_direct(tensor->op);
    if (!cache_entry) {
        NUMA_LOG_DEBUG("MUL_MAT query: Cache entry not found for operation");
        result.supported = false;
        NUMA_PERF_END();
        return result;
    }
    
    const size_t total_elements = ggml_nelements(tensor);
    
    // Use unified cache-based strategy selection
    ggml_numa_execution_strategy_t selected_strategy;
    NUMA_SELECT_STRATEGY_FROM_CACHE(cache_entry, total_elements, selected_strategy);
    
    // Calculate work buffer size dynamically based on tensor requirements
    const size_t work_buffer_size = ggml_numa_kernel_mul_mat_calculate_work_buffer_size(tensor);
    
    // Build successful query result
    result.supported = true;
    result.strategy = selected_strategy;
    result.work_buffer_size_per_thread = work_buffer_size;  // Dynamic calculation!
    result.work_function = ggml_numa_kernel_mul_mat_execute;
    result.efficiency_score = 0.95f; // Static efficiency score for cache-based approach
    result.kernel_name = "NUMA MUL_MAT Kernel";
    
    // MUL_MAT writes directly to shared memory - no aggregation needed
    result.aggregation_policy = GGML_NUMA_AGGREGATION_NONE;
    result.aggregation_function = NULL;
    result.aggregation_user_data = NULL;
    
    // Apply force strategy override if set
    ggml_numa_apply_kernel_force_strategy(&result, "MUL_MAT", 
                                          ggml_numa_kernel_mul_mat_execute, 
                                          ggml_numa_kernel_mul_mat_execute,
                                          ggml_numa_kernel_mul_mat_execute);
    
    NUMA_LOG_DEBUG("MUL_MAT query: ACCEPTING - %zu elements -> %s (efficiency: %.2f, buffer: %zu bytes)", 
                   total_elements, result.kernel_name, result.efficiency_score, work_buffer_size);
    
    // PERFORMANCE: End timing the query operation
    NUMA_PERF_END();
    
    return result;
}

// ============================================================================
// Kernel Registration Function
// ============================================================================

/**
 * Register MUL_MAT kernel with strategy arrays and function pointers
 * This function provides the strategy thresholds and function pointers
 * that the registry will use for O(1) lookups.
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_mul_mat_register(void) {
    ggml_numa_kernel_registration_info_t info = {0};
    
    info.op_type = GGML_OP_MUL_MAT;
    info.supported = true;
    info.kernel_name = "NUMA MUL_MAT Kernel";
    
    // Strategy thresholds for MUL_MAT kernel 
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_SINGLE] = 9999999;   // Single-thread threshold
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_MULTI] = 9999999; //TODO: remove    // Multi-thread threshold
    // Above this: data-parallel strategy
    info.strategy_array.valid = true;
    
    // Function pointers for different strategies - using work_funcs not agg_funcs
    // MUL_MAT kernel handles all strategies within the same function
    info.work_funcs.single_single_fn = ggml_numa_kernel_mul_mat_execute;
    info.work_funcs.single_multi_fn = ggml_numa_kernel_mul_mat_execute;
    info.work_funcs.data_parallel_fn = ggml_numa_kernel_mul_mat_execute;
    info.work_funcs.valid = true;
    
    // Query function pointer for direct dispatch
    info.query_fn = (void*)ggml_numa_kernel_mul_mat_query;
    
    // MUL_MAT doesn't need aggregation functions (no result aggregation needed)
    info.agg_funcs.single_single_fn = NULL;
    info.agg_funcs.single_multi_fn = NULL;
    info.agg_funcs.data_parallel_fn = NULL;
    info.agg_funcs.valid = false;
    
    // MUL_MAT is a computational operation, not a no-op
    info.is_noop = false;
    
    return info;
}
