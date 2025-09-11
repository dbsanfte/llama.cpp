/**
 * @file mul_mat.c
 * @brief NUMA-aware matrix multiplication kernel implementation
 * @author David Sanftenberg
 * 
 * This implementation provides NUMA-optimized matrix multiplication using
 * sophisticated 2D chunking, block tiling, and the new macro system.
 */

#include "mul_mat.h"
#include "numa-kernels.h"
#include "ggml-numa-shared.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

/**
 * @brief Calculate work buffer size for matrix multiplication
 * 
 * Matrix multiplication requires work buffers for type conversion when
 * src1 type doesn't match the optimal vec_dot_type.
 */
size_t ggml_numa_kernel_mul_mat_work_buffer_calc(const struct ggml_tensor * tensor, int total_numa_nodes, int total_threads) {
    NUMA_ASSERT(tensor != NULL, "Tensor cannot be null");
    NUMA_ASSERT(tensor->src[0] != NULL, "src0 cannot be null");  
    NUMA_ASSERT(tensor->src[1] != NULL, "src1 cannot be null");

    // Unused parameters (required by function signature)
    (void)total_numa_nodes;
    (void)total_threads;

    const struct ggml_tensor * src0 = tensor->src[0];
    const struct ggml_tensor * src1 = tensor->src[1];
    
    // Get type traits for src0 to determine required vec_dot_type
    const struct ggml_type_traits_cpu * type_traits = ggml_get_type_traits_cpu(src0->type);
    const enum ggml_type vec_dot_type = type_traits->vec_dot_type;
    
    // Only need work buffer if src1 needs type conversion
    if (src1->type == vec_dot_type) {
        return 0;  // No conversion needed
    }
    
    // Calculate total size needed for converted src1 data
    const size_t src1_converted_size = ggml_row_size(vec_dot_type, ggml_nelements(src1));
    
    NUMA_LOG_DEBUG("MUL_MAT work buffer: src1_type=%d, vec_dot_type=%d, size=%zu bytes", 
                   src1->type, vec_dot_type, src1_converted_size);
    
    return src1_converted_size;
}

/**
 * @brief Query execution strategy for matrix multiplication using custom logic
 * 
 * MUL_MAT needs custom query logic to check vec_dot function availability
 */
ggml_numa_execution_strategy_t ggml_numa_kernel_mul_mat_query(const struct ggml_tensor * tensor) {
    NUMA_ASSERT(tensor != NULL, "Tensor cannot be null");
    NUMA_ASSERT(tensor->src[0] != NULL, "src0 tensor cannot be null");
    
    // Check if vec_dot function is available for src0 type
    const enum ggml_type src0_type = tensor->src[0]->type;
    const struct ggml_type_traits_cpu * traits = ggml_get_type_traits_cpu(src0_type);
    if (traits->vec_dot == NULL) {
        NUMA_LOG_DEBUG("MUL_MAT: vec_dot function not available for type %s, using reference implementation", 
                       ggml_type_name(src0_type));
        return NUMA_STRATEGY_RESERVED;  // Signal unsupported - executor should fall back
    }
    
    // Use standard thresholds for strategy selection
    const size_t total_elements = ggml_nelements(tensor);
    
    if (total_elements < 128) {
        return NUMA_STRATEGY_SINGLE_THREAD;
    } else if (total_elements < 1024) {
        return NUMA_STRATEGY_SINGLE_NODE;
    } else {
        return NUMA_STRATEGY_DATA_PARALLEL;
    }
}


/**
 * @brief Execute matrix multiplication kernel using proper NUMA pattern
 * 
 * Implements the correct NUMA pattern:
 * 1. Phase 1: Multithreaded type conversion per NUMA node
 * 2. Barrier synchronization
 * 3. Phase 2: Multithreaded matrix computation with shared result writes
 */
enum ggml_status ggml_numa_kernel_mul_mat_execute(void * work_context, struct ggml_compute_params * params) {
    struct ggml_tensor * tensor = (struct ggml_tensor *)work_context;
    
    // Validation
    NUMA_ASSERT(tensor != NULL, "Tensor cannot be null");
    NUMA_ASSERT(tensor->src[0] != NULL, "Source tensor 0 cannot be null");
    NUMA_ASSERT(tensor->src[1] != NULL, "Source tensor 1 cannot be null");
    NUMA_ASSERT(params != NULL, "Compute params cannot be null");
    
    const struct ggml_tensor * src0 = tensor->src[0];
    const struct ggml_tensor * src1 = tensor->src[1];
    
    // Extract tensor dimensions (mirrors GGML_TENSOR_BINARY_OP_LOCALS)
    const int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    const int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];
    const int64_t ne0 = tensor->ne[0], ne1 = tensor->ne[1], ne2 = tensor->ne[2], ne3 = tensor->ne[3];
    
    const size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    const size_t nb0 = tensor->nb[0], nb1 = tensor->nb[1], nb2 = tensor->nb[2], nb3 = tensor->nb[3];
    
    // Suppress unused parameter warnings for variables used only in debug/conditional code
    (void)ne01; (void)ne0; (void)ne2; (void)ne3; (void)nb00; (void)nb10;
    
    // Get type traits for vec_dot operation
    const enum ggml_type src0_type = src0->type;
    const struct ggml_type_traits_cpu * traits = ggml_get_type_traits_cpu(src0_type);

    ggml_vec_dot_t const vec_dot = traits->vec_dot;
    enum ggml_type const vec_dot_type = traits->vec_dot_type;
    ggml_from_float_t const from_float = ggml_get_type_traits_cpu(vec_dot_type)->from_float;
    int64_t const num_rows_per_vec_dot = traits->nrows;
    
    // Check if vec_dot function is available for this type
    if (vec_dot == NULL) {
        NUMA_LOG_DEBUG("MUL_MAT: vec_dot function not available for type %s, falling back to reference implementation", 
                       ggml_type_name(src0_type));
        return GGML_STATUS_FAILED;  // Let reference implementation handle this
    }
    
    // Broadcast factors (from reference implementation)
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;
    
    // Work data and memory layout setup
    const bool src1_cont = ggml_is_contiguous(src1);
    const size_t row_size = ggml_row_size(vec_dot_type, ne10);
    
    // Thread and NUMA context setup
    const int ith = params->ith;
    const int nth = params->nth;
    
    // Get shared result tensor data for direct writes
    float * dst_data;
    NUMA_GET_SHARED_DATA(tensor, dst_data, float);
    
    // PHASE 1: MULTITHREADED TYPE CONVERSION (PER NUMA NODE)
    // Each NUMA node converts full src1 tensor using multiple threads for speed
    char * wdata = NULL;
    if (src1->type != vec_dot_type) {
        wdata = (char *)params->wdata;
        NUMA_ASSERT(wdata != NULL, "Work buffer required for type conversion");
        
        // Calculate work buffer strides (mirrors reference implementation)
        const size_t nbw1 = row_size;
        const size_t nbw2 = nbw1 * ne11;
        const size_t nbw3 = nbw2 * ne12;
        
        NUMA_LOG_DEBUG("MUL_MAT: Converting src1 from %s to %s (thread %d/%d)", 
                       ggml_type_name(src1->type), ggml_type_name(vec_dot_type), ith, nth);
        
        // MULTITHREADED conversion using the 3D threaded loop macro
        // Pattern: each thread handles its portion of the innermost dimension (i11)
        NUMA_3D_THREADED_LOOP(src1, ith, nth, {
            const float * src1_row = (const float *)((char *)tensor_data(src1) + 
                                        i13*nb13 + i12*nb12 + i11*nb11);
            void * wdata_row = wdata + i13*nbw3 + i12*nbw2 + i11*nbw1;
            
            from_float(src1_row, wdata_row, ne10);
        });
        
        // BARRIER: All threads on this NUMA node must complete conversion before proceeding
        NUMA_OPENMP_BARRIER();
        
        NUMA_LOG_DEBUG("MUL_MAT: Type conversion completed, proceeding to computation (thread %d/%d)", ith, nth);
    } else {
        // No conversion needed - use original data
        wdata = (char *)tensor_data(src1);
    }
    
    // PHASE 2: MULTITHREADED MATRIX COMPUTATION
    NUMA_LOG_DEBUG("MUL_MAT: Type conversion completed, proceeding to computation (thread %d/%d)\n", ith, nth);
    
    // CRITICAL FIX: Each thread must use their NUMA node's converted data
    // In data-parallel mode, each NUMA node converted the full tensor in their local work buffer
    // Threads should access their local NUMA node's converted data, not shared wdata
    const char * numa_converted_data = (const char*)wdata;
    
    // Use chunk-based iteration matching reference implementation
    const int64_t nr0 = ne0;  // Result first dimension 
    const int64_t nr1 = ne1 * ne2 * ne3;  // Result remaining dimensions
    
    // Chunk size setup (matches reference)
    int chunk_size = 16;
    if (nr0 == 1 || nr1 == 1) {
        chunk_size = 64;
    }
    
    // Calculate chunks for distribution
    int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;
    
    // NUMA-aware chunking: distribute by thread like reference
    if (nchunk0 * nchunk1 < nth * 4 || ggml_current_numa_node >= 0) {
        nchunk0 = nr0 > nr1 ? nth : 1;
        nchunk1 = nr0 > nr1 ? 1 : nth;
    }
    
    const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;
    
    // Each thread processes multiple chunks
    for (int64_t chunk_id = ith; chunk_id < nchunk0 * nchunk1; chunk_id += nth) {
        const int64_t ith0 = chunk_id % nchunk0;
        const int64_t ith1 = chunk_id / nchunk0;
        
        const int64_t ir0_start = dr0 * ith0;
        const int64_t ir0_end = MIN(ir0_start + dr0, nr0);
        const int64_t ir1_start = dr1 * ith1;
        const int64_t ir1_end = MIN(ir1_start + dr1, nr1);
        
        // Skip empty chunks
        if (ir0_start >= ir0_end || ir1_start >= ir1_end) continue;
        
        // Block tiling (matches reference)
        const int64_t blck_0 = 16;
        const int64_t blck_1 = 16;
        const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;
        
        // Process this chunk with matrix chunked loop macro
        NUMA_MATRIX_CHUNKED_LOOP(ir0_start, ir0_end, ir1_start, ir1_end, 
                                 blck_0, blck_1, num_rows_per_vec_dot, {
            // Coordinate calculation (exact reference pattern)
            const int64_t i13 = (ir1 / (ne12 * ne1));
            const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
            const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);
            
            // Broadcast src0 into src1 (from reference)
            const int64_t i03 = i13 / r3;
            const int64_t i02 = i12 / r2;
            
            const int64_t i1 = i11;
            const int64_t i2 = i12;
            const int64_t i3 = i13;
            
            // Memory access pointers (exact reference pattern)
            const char * src0_row = (const char*)tensor_data(src0) + (0 + i02 * nb02 + i03 * nb03);
            // CRITICAL FIX: Use numa_converted_data instead of wdata for thread safety
            const char * src1_col = numa_converted_data +
                (src1_cont || src1->type != vec_dot_type
                    ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                    : (i11 * nb11 + i12 * nb12 + i13 * nb13));
            float * dst_col = (float*)((char*)dst_data + (i1 * nb1 + i2 * nb2 + i3 * nb3));
            
            // Vec_dot computation (exact reference pattern)
            for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                if (num_rows_per_vec_dot == 1) {
                    vec_dot(ne00, &dst_col[ir0], 0, src0_row + ir0*nb01, 0, src1_col, 0, 1);
                } else {
                    // Multi-row case
                    for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
                        float * dst_ptr = &dst_col[ir0 + cn * nb1 / nb0];
                        const char * src0_ptr = src0_row + (ir0 + cn) * nb01;
                        const char * src1_ptr = src1_col + cn * src1_col_stride;
                        vec_dot(ne00, dst_ptr, 0, src0_ptr, 0, src1_ptr, 0, 1);
                    }
                }
            }
        });
    }
    
    return GGML_STATUS_SUCCESS;
}

// Generate registration function using standard macro (no aggregation needed)
NUMA_KERNEL_REGISTRATION_FUNCTION_NO_AGG(
    mul_mat,                                    // op_name
    GGML_OP_MUL_MAT,                           // ggml_op_type
    "NUMA Matrix Multiplication Kernel",       // kernel_display_name
    1024,                                       // threshold_single_single (very small matrices)
    8192,                                      // threshold_single_multi (small-medium matrices)  
    ggml_numa_kernel_mul_mat_execute           // execute_function
)
