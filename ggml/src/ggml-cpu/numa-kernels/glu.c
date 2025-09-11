/**
 * @file glu.c
 * @brief NUMA-aware GLU (Gated Linear Unit) operation kernel implementation
 * @author David Sanftenberg
 * 
 * This file implements NUMA-optimized GLU operations including REGLU, SWIGLU,
 * GEGLU, GEGLU_ERF, and GEGLU_QUICK variants with full composable macro support.
 * 
 * GLU operations are element-wise binary operations that combine two tensors
 * using gated activation functions: y[i] = glu_variant(x[i], g[i])
 * 
 * Architecture:
 * - Uses full composable macro approach (NUMA_ROWWISE_KERNEL_SETUP)
 * - Supports F32 and F16 data types
 * - Leverages SIMD optimization via ggml_vec_* functions
 * - Three execution strategies for optimal performance
 * - Zero work buffer requirements (element-wise operation)
 */

#include "glu.h"
#include "numa-kernels.h"
#include "../ggml-numa-shared.h"
#include "../ggml-numa-openmp-coordinator.h"
#include "../ggml-cpu-impl.h"
#include "../ggml-impl.h"
#include "../ops.h"
#include "../vec.h"
#include <assert.h>

/**
 * @brief Execute GLU operation using NUMA kernel with row-based processing
 * 
 * Implements all GLU variants (REGLU, SWIGLU, GEGLU, etc.) using row-based
 * processing that matches the reference implementation behavior.
 * 
 * The implementation handles both single-tensor (split) and dual-tensor modes:
 * - Single tensor: Input tensor is split in half along first dimension (nc = ne[0]/2)
 * - Dual tensor: Two separate input tensors are provided (nc = ne[0])
 * 
 * GLU operations process data row by row, with each row containing nc elements.
 * Thread distribution is done across rows, not individual elements.
 */
enum ggml_status ggml_numa_kernel_glu_execute(void * work_context, struct ggml_compute_params * params) {
    NUMA_ASSERT(work_context != NULL, "Work context cannot be null");
    NUMA_ASSERT(params != NULL, "Compute params cannot be null");
    
    struct ggml_tensor * dst = (struct ggml_tensor *)work_context;
    
    // GLU operation requires source tensors
    NUMA_ASSERT(dst->src[0] != NULL, "GLU source tensor 0 cannot be null");
    NUMA_ASSERT(dst->op == GGML_OP_GLU, "Expected GLU operation");
    
    // Extract GLU operation type for SIMD function selection
    const enum ggml_glu_op glu_op = ggml_get_glu_op(dst);
    
    // GLU supports both dual-tensor and single-tensor (split) modes
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const bool has_separate_src1 = (src1 != NULL);
    
    // Calculate dimensions - GLU processes rows, not individual elements
    const int nc = has_separate_src1 ? src0->ne[0] : src0->ne[0] / 2;  // elements per row
    const int nr = ggml_nrows(src0);  // total number of rows
    
    NUMA_ASSERT(dst->ne[0] == nc, "Destination width mismatch");
    NUMA_ASSERT(ggml_nrows(dst) == nr, "Destination row count mismatch");
    
    // Get swap parameter for single-tensor mode
    const int32_t swapped = ggml_get_op_params_i32(dst, 1);
    
    // === COMPOSABLE KERNEL SETUP USING BUILDING BLOCKS ===
    NUMA_INIT_CONTEXT(ctx, dst, params);
    NUMA_VALIDATE_INPUTS(dst, params);
    
    // Custom row-based thread distribution (not using the standard row slice macros)
    const int dr = (nr + ctx.total_threads - 1) / ctx.total_threads;  // rows per thread
    const int ir0 = dr * ctx.thread_id;                              // start row
    const int ir1 = (ir0 + dr < nr) ? ir0 + dr : nr;                 // end row
    
    if (ir0 >= ir1) {
        NUMA_BARRIER_AUTO(ctx);  // Still participate in barriers
        return GGML_STATUS_SUCCESS;
    }
    
// Macro to generate GLU processing code for a specific data type
#define GLU_PROCESS_ROWS(type_suffix, data_type) do { \
    char * src0_d = (char *) tensor_data(src0); \
    char * src1_d = (char *) (has_separate_src1 ? tensor_data(src1) : tensor_data(src0)); \
    const size_t src0_o = src0->nb[1];  /* stride between rows */ \
    const size_t src1_o = has_separate_src1 ? src1->nb[1] : src0->nb[1]; \
    \
    /* Process each row assigned to this thread */ \
    for (int i1 = ir0; i1 < ir1; i1++) { \
        data_type * src0_p = (data_type *) (src0_d + i1 * src0_o); \
        data_type * src1_p = (data_type *) (src1_d + i1 * src1_o); \
        \
        if (!has_separate_src1) { \
            /* Single tensor mode: adjust pointers based on swap setting */ \
            src0_p += swapped ? nc : 0; \
            src1_p += swapped ? 0 : nc; \
        } \
        \
        data_type * dst_p = (data_type *) ((char *) tensor_data(dst) + i1 * dst->nb[1]); \
        \
        /* Execute appropriate GLU variant using SIMD functions */ \
        switch (glu_op) { \
            case GGML_GLU_OP_REGLU: \
                ggml_vec_reglu_##type_suffix(nc, dst_p, src0_p, src1_p); \
                break; \
            case GGML_GLU_OP_SWIGLU: \
                ggml_vec_swiglu_##type_suffix(nc, dst_p, src0_p, src1_p); \
                break; \
            case GGML_GLU_OP_GEGLU: \
                ggml_vec_geglu_##type_suffix(nc, dst_p, src0_p, src1_p); \
                break; \
            case GGML_GLU_OP_GEGLU_ERF: \
                ggml_vec_geglu_erf_##type_suffix(nc, dst_p, src0_p, src1_p); \
                break; \
            case GGML_GLU_OP_GEGLU_QUICK: \
                ggml_vec_geglu_quick_##type_suffix(nc, dst_p, src0_p, src1_p); \
                break; \
            default: \
                NUMA_ASSERT(false, "Unsupported GLU operation type"); \
                return GGML_STATUS_FAILED; \
        } \
    } \
} while (0)

    // Handle different data types with row-based processing
    switch (src0->type) {
        case GGML_TYPE_F32:
            GLU_PROCESS_ROWS(f32, float);
            break;
        
        case GGML_TYPE_F16:
            GLU_PROCESS_ROWS(f16, ggml_fp16_t);
            break;
        
        default:
            NUMA_ASSERT(false, "Unsupported data type for GLU operation");
            return GGML_STATUS_FAILED;
    }
    
    NUMA_BARRIER_AUTO(ctx);  // Synchronize all threads
    
    NUMA_LOG_TRACE("GLU processed rows %d-%d with thread %d/%d on NUMA node %d", 
                   ir0, ir1, ctx.thread_id, ctx.total_threads, ctx.numa_node);
    
    return GGML_STATUS_SUCCESS;
}

// ============================================================================ 
// Complete kernel implementation using shared macros
// ============================================================================

NUMA_KERNEL_REGISTER_METADATA(
    glu,                                   // kernel name
    GGML_OP_GLU,                           // operation type
    "NUMA GLU Kernel",                     // kernel description
    4096,                                  // single_single threshold
    8192,                                  // single_multi threshold
    ggml_numa_kernel_glu_execute           // execution function
)
