/**
 * @file mul_mat.c
 * @brief NUMA Matrix Multiplication (MUL_MAT) Kernel Implementation
 * @author David Sanftenberg
 * 
 * ============================================================================
 * NUMA KERNEL: MUL_MAT (Matrix Multiplication) - Complete Implementation
 * ============================================================================
 * 
 * This implementation provides comprehensive MUL_MAT kernel functionality with:
 * - All quantization types supported by reference (F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S, IQ1_S, IQ1_M, IQ4_NL, IQ4_XS, Q8_K, BF16, TQ1_0, TQ2_0)
 * - NUMA-aware chunk-based work distribution
 * - Type-specific SIMD operations using vec_dot dispatch
 * - Work buffer management for type conversions
 * - Optimal cache utilization through block-tiling
 * 
 * OPERATION CHARACTERISTICS:
 * ========================
 * - Complex 4D tensor operations: dst = src0 @ src1
 * - Type dispatch based on src0 quantization type
 * - Chunk-based parallelization for optimal NUMA performance
 * - Work buffer allocation for type conversions
 * - Block-tiling for cache efficiency
 * 
 * IMPLEMENTATION STRATEGY:
 * =======================
 * 1. Mirror reference implementation logic exactly
 * 2. NUMA-aware chunk distribution across nodes
 * 3. Type-specific vec_dot function dispatch
 * 4. Work buffer management for non-native types
 * 5. Comprehensive error handling and validation
 * 
 * MATRIX MULTIPLICATION VARIANTS:
 * ==============================
 * - Standard matrix multiplication with quantization support
 * - Type conversion handling for src1 when needed
 * - Block-tiling optimization for cache efficiency
 * - Broadcasting support for batch operations
 * 
 * ============================================================================
 */

#include "mul_mat.h"
#include <stdatomic.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include "numa-kernels.h"
#include "../ggml-numa-shared.h"
#include "../ggml-cpu-impl.h"
#include "../ggml-impl.h"
#include "../vec.h"

// Forward declarations
// External functions for vec_dot operations - from ggml-cpu.c

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

// Cache line padding for performance
#define CACHE_LINE_SIZE_F32 (64/sizeof(float))

// ============================================================================
// MUL_MAT Kernel Registration
// ============================================================================

ggml_numa_kernel_registration_info_t ggml_numa_kernel_mul_mat_register(void) {
    printf("🐛 DEBUG: MUL_MAT register function called!\n");
    fflush(stdout);
    
    ggml_numa_kernel_registration_info_t info = {0};
    
    info.op_type = GGML_OP_MUL_MAT;
    info.supported = true;
    info.kernel_name = "NUMA MUL_MAT Kernel";
    
    // Strategy thresholds for MUL_MAT operation
    // Use larger thresholds due to computational intensity
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_SINGLE] = 4096;      // Single thread below 4K elements
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_MULTI] = 1048576;    // Multi-thread below 1M elements
    // Above 1M elements: data-parallel strategy
    info.strategy_array.valid = true;
    
    // Function pointers for different strategies
    info.work_funcs.single_single_fn = ggml_numa_kernel_mul_mat_execute;
    info.work_funcs.single_multi_fn = ggml_numa_kernel_mul_mat_execute;
    info.work_funcs.data_parallel_fn = ggml_numa_kernel_mul_mat_execute;
    info.work_funcs.valid = true;
    
    // Query function pointer - enables direct dispatch
    info.query_fn = (void*)ggml_numa_kernel_mul_mat_query;
    
    // Work buffer calculation function pointer
    info.work_buffer_calc_fn = (void*)ggml_numa_kernel_mul_mat_work_buffer_calc;
    
    // MUL_MAT doesn't need aggregation functions
    info.agg_funcs.single_single_fn = NULL;
    info.agg_funcs.single_multi_fn = NULL; 
    info.agg_funcs.data_parallel_fn = NULL;
    info.agg_funcs.valid = false;
    
    return info;
}

ggml_numa_kernel_query_result_t ggml_numa_kernel_mul_mat_query(const struct ggml_tensor * tensor) {
    ggml_numa_kernel_query_result_t result = { .supported = false };
    
    // Validate this is a MUL_MAT operation
    if (!tensor || tensor->op != GGML_OP_MUL_MAT) {
        return result;
    }
    
    // Validate tensor structure for MUL_MAT
    if (!tensor->src[0] || !tensor->src[1]) {
        NUMA_LOG_DEBUG("MUL_MAT query: Missing source tensors");
        return result;
    }
    
    // Check if this kernel is actually registered and supported
    if (!ggml_numa_is_kernel_supported(GGML_OP_MUL_MAT)) {
        NUMA_LOG_DEBUG("MUL_MAT kernel not supported - registration disabled");
        result.supported = false;
        return result;
    }
    
    // Get our own cache entry with the registered thresholds
    const ggml_numa_kernel_cache_entry_t * cache_entry = ggml_numa_lookup_kernel_direct(GGML_OP_MUL_MAT);
    if (!cache_entry || !cache_entry->strategy_array.valid) {
        NUMA_LOG_DEBUG("MUL_MAT query: No valid strategy array in cache");
        return result;
    }

    // Calculate total operations for strategy selection (matrix dimensions)
    const struct ggml_tensor * src0 = tensor->src[0];
    const struct ggml_tensor * src1 = tensor->src[1];
    
    const int64_t ne00 = src0->ne[0]; // cols of src0
    const int64_t ne01 = src0->ne[1]; // rows of src0  
    const int64_t ne11 = src1->ne[1]; // cols of src1
    
    // For MUL_MAT, use total operations (FLOPS) as complexity metric
    const size_t total_ops = ne00 * ne01 * ne11;
    
    // Use the registered thresholds for strategy selection
    ggml_numa_execution_strategy_t selected_strategy;
    NUMA_SELECT_STRATEGY_FROM_CACHE(cache_entry, total_ops, selected_strategy);
    
    // Build successful query result
    result.supported = true;
    result.strategy = selected_strategy;
    result.work_buffer_size_per_thread = ggml_numa_kernel_mul_mat_work_buffer_calc(tensor, 1, 1);  // Calculate actual buffer size needed
    result.aggregation_policy = GGML_NUMA_AGGREGATION_NONE;  // Direct writes, no aggregation needed
    result.aggregation_function = NULL;
    result.aggregation_user_data = NULL;
    
    // Select work function and metadata based on strategy
    if (selected_strategy.node_strategy == NUMA_NODE_STRATEGY_SINGLE) {
        if (selected_strategy.on_node_strategy == NUMA_ON_NODE_STRATEGY_SINGLE_THREAD) {
            result.work_function = ggml_numa_kernel_mul_mat_execute;
            result.efficiency_score = 0.85f;  // Lower due to no parallelism
            result.kernel_name = "NUMA MUL_MAT (Single/Single)";
        } else {
            result.work_function = ggml_numa_kernel_mul_mat_execute;
            result.efficiency_score = 0.92f;  // Good thread parallelism
            result.kernel_name = "NUMA MUL_MAT (Single/Multi)";
        }
    } else {
        // Data-parallel strategy
        result.work_function = ggml_numa_kernel_mul_mat_execute;
        result.efficiency_score = 0.98f;  // Excellent NUMA + thread parallelism
        result.kernel_name = "NUMA MUL_MAT (Data-Parallel)";
    }
    
    // Apply force strategy override if environment variable is set
    ggml_numa_apply_kernel_force_strategy(&result, "MUL_MAT", 
                                          ggml_numa_kernel_mul_mat_execute,
                                          ggml_numa_kernel_mul_mat_execute,
                                          ggml_numa_kernel_mul_mat_execute);
    
    NUMA_LOG_DEBUG("MUL_MAT query: ops=%ld, strategy=%s/%s, efficiency=%.2f", 
                   total_ops,
                   selected_strategy.node_strategy == NUMA_NODE_STRATEGY_SINGLE ? "single" : "data-parallel",
                   selected_strategy.on_node_strategy == NUMA_ON_NODE_STRATEGY_SINGLE_THREAD ? "single-thread" : "multi-thread",
                   result.efficiency_score);
    
    return result;
}

size_t ggml_numa_kernel_mul_mat_work_buffer_calc(const struct ggml_tensor * tensor, int total_numa_nodes, int total_threads) {
    if (!tensor || !tensor->src[0] || !tensor->src[1]) {
        return 0;
    }
    
    const struct ggml_tensor * src0 = tensor->src[0];
    const struct ggml_tensor * src1 = tensor->src[1];
    
    // Get type traits to determine if type conversion is needed
    const struct ggml_type_traits_cpu * type_traits = ggml_get_type_traits_cpu(src0->type);
    if (!type_traits) {
        return 0;
    }
    
    const enum ggml_type vec_dot_type = type_traits->vec_dot_type;
    
    // Work buffer is only needed if src1 type != vec_dot_type
    if (src1->type == vec_dot_type) {
        return 0;  // No conversion needed
    }
    
    // Calculate work buffer size for type conversion
    // Based on reference implementation logic
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    
    const size_t nbw0 = ggml_type_size(vec_dot_type);
    const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
    const size_t nbw2 = nbw1 * ne11;
    const size_t nbw3 = nbw2 * ne12;
    
    // Total work buffer size for all batches
    const size_t total_work_buffer = ne13 * nbw3;
    
    NUMA_LOG_DEBUG("MUL_MAT work buffer: src1_type=%s, vec_dot_type=%s, size=%zu bytes", 
                   ggml_type_name(src1->type), ggml_type_name(vec_dot_type), total_work_buffer);
    
    return total_work_buffer;
}

// ============================================================================
// NUMA MUL_MAT Kernel Implementation
// ============================================================================

/**
 * Execute one chunk of MUL_MAT operation - mirrors reference implementation
 */
static enum ggml_status ggml_numa_kernel_mul_mat_one_chunk(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst,
    const enum ggml_type src0_type,
    const int64_t num_rows_per_vec_dot,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    // Tensor dimension extraction macro - exact mirror of reference
    GGML_TENSOR_BINARY_OP_LOCALS

    const bool src1_cont = ggml_is_contiguous(src1);

    // Get type traits for vec_dot dispatch
    const struct ggml_type_traits_cpu * type_traits = ggml_get_type_traits_cpu(src0_type);
    NUMA_ASSERT(type_traits != NULL, "Invalid type traits for src0_type");

    ggml_vec_dot_t const vec_dot = type_traits->vec_dot;
    enum ggml_type const vec_dot_type = type_traits->vec_dot_type;

    NUMA_ASSERT(vec_dot != NULL, "No vec_dot function available for type");

    // Broadcast factors - exact mirror of reference
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    // Early exit for threads with no work
    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return GGML_STATUS_SUCCESS;
    }

    // Get work data (for type conversion) or use direct src1 data
    const void * wdata = (src1->type == vec_dot_type) ? tensor_data(src1) : params->wdata;
    const size_t row_size = ggml_row_size(vec_dot_type, ne10);

    NUMA_ASSERT(ne12 % ne02 == 0, "Dimension mismatch in broadcast factor r2");
    NUMA_ASSERT(ne13 % ne03 == 0, "Dimension mismatch in broadcast factor r3");

    // Block-tiling parameters - exact mirror of reference
    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;

    // Get NUMA execution context from thread-local variables (set by coordinator)
    extern __thread int ggml_current_numa_node;
    extern __thread bool ggml_numa_is_data_parallel_execution;
    extern __thread void * ggml_numa_shared_result_tensor_data;

    // Use shared result tensor memory for direct writes (eliminates aggregation)
    float * dst_data;
    if (ggml_numa_shared_result_tensor_data != NULL) {
        dst_data = (float *)ggml_numa_shared_result_tensor_data;
    } else {
        dst_data = (float *)tensor_data(dst);
    }

    NUMA_LOG_TRACE("MUL_MAT chunk execution: ir0=[%ld,%ld), ir1=[%ld,%ld), NUMA_node=%d, shared_mem=%s",
                   ir0_start, ir0_end, ir1_start, ir1_end, ggml_current_numa_node,
                   ggml_numa_shared_result_tensor_data ? "yes" : "no");

    // Mirror exact iteration pattern from reference implementation
    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
                const int64_t i13 = (ir1 / (ne12 * ne1));
                const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // Broadcast src0 into src1 - exact mirror of reference
                const int64_t i03 = i13 / r3;
                const int64_t i02 = i12 / r2;

                const int64_t i1 = i11;
                const int64_t i2 = i12;
                const int64_t i3 = i13;

                const char * src0_row = (const char*)tensor_data(src0) + (0 + i02 * nb02 + i03 * nb03);

                // Calculate src1 column pointer - handles both contiguous and non-contiguous cases
                const char * src1_col = (const char*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));

                float * dst_col = (float*)((char*)dst_data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                // Direct assignment - mirror reference implementation pattern
                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                    if (num_rows_per_vec_dot == 1) {
                        // Single row case - direct vec_dot call
                        vec_dot(ne00, &dst_col[ir0], 0, src0_row + ir0*nb01, 0, src1_col, 0, 1);
                    } else {
                        // Multi-row case - handle multiple rows per vec_dot call
                        for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
                            float * dst_ptr = &dst_col[ir0 + cn * nb1 / nb0];
                            const char * src0_ptr = src0_row + (ir0 + cn) * nb01;
                            const char * src1_ptr = src1_col + cn * src1_col_stride;
                            
                            vec_dot(ne00, dst_ptr, 0, src0_ptr, 0, src1_ptr, 0, 1);
                        }
                    }
                }
            }
        }
    }

    return GGML_STATUS_SUCCESS;
}

enum ggml_status ggml_numa_kernel_mul_mat_execute(void * work_context, struct ggml_compute_params * params) {
    struct ggml_tensor * dst = (struct ggml_tensor *)work_context;
    
    // Validate inputs
    NUMA_ASSERT(dst != NULL, "Destination tensor cannot be null");
    NUMA_ASSERT(dst->src[0] != NULL, "Source tensor 0 cannot be null");
    NUMA_ASSERT(dst->src[1] != NULL, "Source tensor 1 cannot be null");
    NUMA_ASSERT(params != NULL, "Compute params cannot be null");
    
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    
    // Get type traits and verify support
    const struct ggml_type_traits_cpu * type_traits = ggml_get_type_traits_cpu(src0->type);
    NUMA_ASSERT(type_traits != NULL, "Type traits not found for src0 type");
    
    const enum ggml_type vec_dot_type = type_traits->vec_dot_type;
    const ggml_from_float_t from_float = type_traits->from_float;
    const int64_t vec_dot_num_rows = type_traits->nrows;
    
    NUMA_ASSERT(type_traits->vec_dot != NULL, "No vec_dot function for src0 type");
    
    // Tensor dimension extraction - exact mirror of reference
    GGML_TENSOR_BINARY_OP_LOCALS
    
    // Dimension validation - exact mirror of reference
    NUMA_ASSERT(ne0 == ne01, "Matrix dimension mismatch: ne0 != ne01");
    NUMA_ASSERT(ne1 == ne11, "Matrix dimension mismatch: ne1 != ne11");
    NUMA_ASSERT(ne2 == ne12, "Matrix dimension mismatch: ne2 != ne12");
    NUMA_ASSERT(ne3 == ne13, "Matrix dimension mismatch: ne3 != ne13");
    
    // Stride validation - exact mirror of reference
    NUMA_ASSERT(nb00 == ggml_type_size(src0->type), "src0 not contiguous");
    NUMA_ASSERT(nb10 == ggml_type_size(src1->type), "src1 not contiguous");
    NUMA_ASSERT(nb0 == sizeof(float), "dst not contiguous");
    NUMA_ASSERT(nb0 <= nb1, "dst stride validation failed");
    NUMA_ASSERT(nb1 <= nb2, "dst stride validation failed");
    NUMA_ASSERT(nb2 <= nb3, "dst stride validation failed");
    
    // Get NUMA execution context from thread-local variables (set by coordinator)
    extern __thread int ggml_current_numa_node;
    extern __thread bool ggml_numa_is_data_parallel_execution;
    extern __thread int ggml_numa_total_nodes_for_data_parallel;
    
    const int ith = params->ith;
    const int nth = params->nth;
    
    // Log execution strategy in standardized format for integration test parsing
    // Only log once per operation (thread 0 of NUMA node 0) to avoid inflated counts
    if (ith == 0 && ggml_current_numa_node == 0) {
        if (ggml_numa_is_data_parallel_execution) {
            NUMA_LOG_STRATEGY_DATA_PARALLEL("MUL_MAT");
        } else if (params->nth > 1) {
            NUMA_LOG_STRATEGY_SINGLE_MULTI("MUL_MAT");
        } else {
            NUMA_LOG_STRATEGY_SINGLE_SINGLE("MUL_MAT");
        }
    }
    
    NUMA_LOG_DEBUG("MUL_MAT execution: src0_type=%s, src1_type=%s, dst_type=%s, dims=[%ld,%ld,%ld,%ld]",
                   ggml_type_name(src0->type), ggml_type_name(src1->type), ggml_type_name(dst->type),
                   ne0, ne1, ne2, ne3);
    
    // Handle type conversion if needed - exact mirror of reference logic
    if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;
        NUMA_ASSERT(wdata != NULL, "Work buffer required for type conversion but not provided");
        
        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;
        
        NUMA_ASSERT(params->wsize >= ne13*nbw3, "Work buffer too small for type conversion");
        NUMA_ASSERT(src1->type == GGML_TYPE_F32, "Type conversion only supports F32 source");
        NUMA_ASSERT(from_float != NULL, "No conversion function available");
        
        // Convert src1 tensor to vec_dot_type using thread coordination (exact mirror of reference)
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    // Thread coordination: each thread processes different blocks within each row
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    
                    // Skip if this thread has no work for this block
                    if (ne10_block_start >= ne10_block_end) {
                        continue;
                    }
                    
                    // Source: F32 data at the current batch/matrix/row position + block offset
                    const float * src1_row = (const float *)((const char *)tensor_data(src1) + 
                                             i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10);
                    
                    // Destination: converted data in work buffer with proper layout + block offset
                    void * dst_row = wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0;
                    
                    // Perform the actual conversion (e.g., F32 -> Q8_0) for this thread's block
                    from_float(src1_row, dst_row, (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
        
        NUMA_LOG_VERBOSE("MUL_MAT: Thread %d/%d converted src1 blocks [%ld-%ld) from F32 to vec_dot_type %d", 
                         ith, nth, (ith * ne10/ggml_blck_size(vec_dot_type)) / nth, 
                         ((ith + 1) * ne10/ggml_blck_size(vec_dot_type)) / nth, vec_dot_type);
    }
    
    // Thread synchronization for type conversion - use OpenMP barrier
    // Only synchronize if multiple threads are participating
    if (nth > 1) {
        #pragma omp barrier
        NUMA_LOG_DEBUG("MUL_MAT: Thread %d/%d completed type conversion using OpenMP barrier", ith, nth);
    } else {
        NUMA_LOG_DEBUG("MUL_MAT: Single thread execution, skipping barrier synchronization");
    }
    
    // Coordinate work distribution - exact mirror of reference logic
    const int64_t nr0 = ne0;  // Rows in result
    const int64_t nr1 = ne1 * ne2 * ne3;  // Columns in result
    
    // Dynamic chunk sizing - exact mirror of reference
    int chunk_size = 16;
    if (nr0 == 1 || nr1 == 1) {
        chunk_size = 64;
    }
    
    int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;
    
    // NUMA-aware chunking adjustment
    if (nchunk0 * nchunk1 < nth * 4 || ggml_numa_is_data_parallel_execution) {
        nchunk0 = nr0 > nr1 ? nth : 1;
        nchunk1 = nr0 > nr1 ? 1 : nth;
    }
    
    const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;
    
    // Calculate chunk assignment for this thread
    const int64_t ith0 = ith % nchunk0;
    const int64_t ith1 = ith / nchunk0;
    
    const int64_t ir0_start = dr0 * ith0;
    const int64_t ir0_end = MIN(ir0_start + dr0, nr0);
    
    const int64_t ir1_start = dr1 * ith1;
    const int64_t ir1_end = MIN(ir1_start + dr1, nr1);
    
    // Determine rows per vec_dot operation - exact mirror of reference
    int64_t num_rows_per_vec_dot = vec_dot_num_rows;
    
    // Optimization constraints - exact mirror of reference
    if ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
        num_rows_per_vec_dot = 1;
    }
    
    NUMA_LOG_TRACE("MUL_MAT thread work: thread=%d/%d, chunks=[%ld,%ld), ir0=[%ld,%ld), ir1=[%ld,%ld), rows_per_vec_dot=%ld",
                   ith, nth, ith0, ith1, ir0_start, ir0_end, ir1_start, ir1_end, num_rows_per_vec_dot);
    
    // Execute the chunk
    return ggml_numa_kernel_mul_mat_one_chunk(params, dst, src0->type, num_rows_per_vec_dot, 
                                             ir0_start, ir0_end, ir1_start, ir1_end);
}
