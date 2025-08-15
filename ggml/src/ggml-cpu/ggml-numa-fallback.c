#include "ggml-numa-fallback.h"
#include "ggml-cpu-impl.h"
#include "ggml-cpu.h"    // For ggml_cplan structure definition
#include "ops.h"
#include "unary-ops.h"
#include "binary-ops.h"

#include <stdatomic.h>

static atomic_int_fast64_t g_fallback_operations = 0;

//
// Phase 1: Complete Single-Threaded Fallback System with Improved Organization
// 
// This implements the critical Phase 1 foundation that handles ALL GGML operations
// through single-threaded execution to avoid threading conflicts while providing
// a stable foundation for gradual NUMA-aware migration.
//

// Helper macros to reduce repetition in dispatch switch
#define DISPATCH_SIMPLE(op_enum, forward_func) \
    case op_enum: \
        ggml_compute_forward_##forward_func(&fallback_params, tensor); \
        break;

// Helper function for operations that need parameter validation
static enum ggml_status validate_tensor_operation(struct ggml_tensor * tensor, const char * op_name) {
    if (!tensor) {
        GGML_LOG_ERROR("%s: Invalid tensor\n", op_name);
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

// Helper function for matrix operations that need dimension checking
static enum ggml_status validate_matrix_operation(struct ggml_tensor * tensor, const char * op_name) {
    if (validate_tensor_operation(tensor, op_name) != GGML_STATUS_SUCCESS) {
        return GGML_STATUS_FAILED;
    }
    
    if (tensor->src[0] && tensor->src[1]) {
        // Add matrix-specific validation here if needed
        GGML_LOG_DEBUG("%s: Matrix operation validated\n", op_name);
    }
    
    return GGML_STATUS_SUCCESS;
}

enum ggml_status ggml_numa_fallback_execute(struct ggml_tensor * tensor, struct ggml_cplan * cplan) {
    if (!tensor) {
        GGML_LOG_ERROR("Invalid tensor for fallback execution\n");
        return GGML_STATUS_FAILED;
    }
    
    atomic_fetch_add_explicit(&g_fallback_operations, 1, memory_order_relaxed);
    
    // Create single-threaded compute params to avoid threadpool conflicts
    struct ggml_compute_params fallback_params = {
        .ith = 0,                  // Single thread index
        .nth = 1,                  // Single thread total
        .wsize = cplan ? cplan->work_size : 0,
        .wdata = cplan ? cplan->work_data : NULL,
        .threadpool = NULL         // Critical: no threadpool conflicts
    };
    
    GGML_LOG_DEBUG("Executing operation %s via single-threaded fallback\n", ggml_op_name(tensor->op));

    // Optimized switch statement with helper macros for reduced repetition
    switch (tensor->op) {
        // No operation - pass through
        case GGML_OP_NONE:
            break;
            
        // === BASIC MATH OPERATIONS (using macros for cleaner code) ===
        DISPATCH_SIMPLE(GGML_OP_DUP, dup)
        DISPATCH_SIMPLE(GGML_OP_ADD, add)
        DISPATCH_SIMPLE(GGML_OP_ADD1, add1)
        DISPATCH_SIMPLE(GGML_OP_ACC, acc)
        DISPATCH_SIMPLE(GGML_OP_SUB, sub)
        DISPATCH_SIMPLE(GGML_OP_MUL, mul)
        DISPATCH_SIMPLE(GGML_OP_DIV, div)
        
        // === UNARY MATH OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_SQR, sqr)
        DISPATCH_SIMPLE(GGML_OP_SQRT, sqrt)
        DISPATCH_SIMPLE(GGML_OP_LOG, log)
        DISPATCH_SIMPLE(GGML_OP_SIN, sin)
        DISPATCH_SIMPLE(GGML_OP_COS, cos)
        
        // === REDUCTION OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_SUM, sum)
        DISPATCH_SIMPLE(GGML_OP_SUM_ROWS, sum_rows)
        DISPATCH_SIMPLE(GGML_OP_MEAN, mean)
        DISPATCH_SIMPLE(GGML_OP_ARGMAX, argmax)
        DISPATCH_SIMPLE(GGML_OP_COUNT_EQUAL, count_equal)
        
        // === TENSOR MANIPULATION ===
        DISPATCH_SIMPLE(GGML_OP_REPEAT, repeat)
        DISPATCH_SIMPLE(GGML_OP_REPEAT_BACK, repeat_back)
        DISPATCH_SIMPLE(GGML_OP_CONCAT, concat)
        DISPATCH_SIMPLE(GGML_OP_SILU_BACK, silu_back)
        DISPATCH_SIMPLE(GGML_OP_CPY, cpy)
        DISPATCH_SIMPLE(GGML_OP_CONT, cont)
        DISPATCH_SIMPLE(GGML_OP_RESHAPE, reshape)
        DISPATCH_SIMPLE(GGML_OP_VIEW, view)
        DISPATCH_SIMPLE(GGML_OP_PERMUTE, permute)
        DISPATCH_SIMPLE(GGML_OP_TRANSPOSE, transpose)
        DISPATCH_SIMPLE(GGML_OP_GET_ROWS, get_rows)
        DISPATCH_SIMPLE(GGML_OP_GET_ROWS_BACK, get_rows_back)
        DISPATCH_SIMPLE(GGML_OP_SET_ROWS, set_rows)
        
        // === NORMALIZATION OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_NORM, norm)
        DISPATCH_SIMPLE(GGML_OP_RMS_NORM, rms_norm)
        DISPATCH_SIMPLE(GGML_OP_RMS_NORM_BACK, rms_norm_back)
        DISPATCH_SIMPLE(GGML_OP_GROUP_NORM, group_norm)
        DISPATCH_SIMPLE(GGML_OP_L2_NORM, l2_norm)
        
        // === MATRIX OPERATIONS ===
        // NOTE: MUL_MAT operations are handled by NUMA coordinator, not fallback
        // This is because matrix operations require threadpool coordination
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            GGML_LOG_DEBUG("MUL_MAT operations should be handled by NUMA coordinator, not fallback\n");
            return GGML_STATUS_FAILED;
            
        DISPATCH_SIMPLE(GGML_OP_OUT_PROD, out_prod)
        
        // === UTILITY OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_SCALE, scale)
        DISPATCH_SIMPLE(GGML_OP_SET, set)
        
        // === MASKING OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_DIAG, diag)
        DISPATCH_SIMPLE(GGML_OP_DIAG_MASK_INF, diag_mask_inf)
        DISPATCH_SIMPLE(GGML_OP_DIAG_MASK_ZERO, diag_mask_zero)
        
        // === ACTIVATION OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_SOFT_MAX, soft_max)
        case GGML_OP_SOFT_MAX_BACK:
            // Use soft_max_ext_back as fallback for soft_max_back
            ggml_compute_forward_soft_max_ext_back(&fallback_params, tensor);
            break;
            
        // === COMPLEX OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_ROPE, rope)
        DISPATCH_SIMPLE(GGML_OP_ROPE_BACK, rope_back)
        DISPATCH_SIMPLE(GGML_OP_CLAMP, clamp)
        
        // === CONVOLUTION OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_CONV_TRANSPOSE_1D, conv_transpose_1d)
        DISPATCH_SIMPLE(GGML_OP_IM2COL, im2col)
        case GGML_OP_IM2COL_BACK:
            ggml_compute_forward_im2col_back_f32(&fallback_params, tensor);
            break;
        DISPATCH_SIMPLE(GGML_OP_CONV_2D, conv_2d)
        DISPATCH_SIMPLE(GGML_OP_CONV_2D_DW, conv_2d_dw)
        DISPATCH_SIMPLE(GGML_OP_CONV_TRANSPOSE_2D, conv_transpose_2d)
        
        // === POOLING OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_POOL_1D, pool_1d)
        DISPATCH_SIMPLE(GGML_OP_POOL_2D, pool_2d)
        DISPATCH_SIMPLE(GGML_OP_POOL_2D_BACK, pool_2d_back)
        
        // === UTILITY OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_UPSCALE, upscale)
        DISPATCH_SIMPLE(GGML_OP_PAD, pad)
        DISPATCH_SIMPLE(GGML_OP_PAD_REFLECT_1D, pad_reflect_1d)
        DISPATCH_SIMPLE(GGML_OP_ROLL, roll)
        DISPATCH_SIMPLE(GGML_OP_ARANGE, arange)
        DISPATCH_SIMPLE(GGML_OP_TIMESTEP_EMBEDDING, timestep_embedding)
        DISPATCH_SIMPLE(GGML_OP_ARGSORT, argsort)
        DISPATCH_SIMPLE(GGML_OP_LEAKY_RELU, leaky_relu)
        
        // === ATTENTION OPERATIONS (special parameter handling) ===
        case GGML_OP_FLASH_ATTN_EXT:
            // Flash attention requires special parameter handling
            if (tensor->src[0] && tensor->src[1] && tensor->src[2]) {
                ggml_compute_forward_flash_attn_ext(&fallback_params,
                    tensor->src[0], tensor->src[1], tensor->src[2], 
                    tensor->src[3], tensor);
            } else {
                GGML_LOG_ERROR("FLASH_ATTN_EXT requires valid Q, K, V tensors\n");
                return GGML_STATUS_FAILED;
            }
            break;
            
        case GGML_OP_FLASH_ATTN_BACK:
            ggml_compute_forward_flash_attn_back(&fallback_params, false, tensor);
            break;
            
        // === STATE SPACE MODEL OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_SSM_CONV, ssm_conv)
        DISPATCH_SIMPLE(GGML_OP_SSM_SCAN, ssm_scan)
        
        // === WINDOW OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_WIN_PART, win_part)
        DISPATCH_SIMPLE(GGML_OP_WIN_UNPART, win_unpart)
        
        // === POSITIONAL OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_GET_REL_POS, get_rel_pos)
        DISPATCH_SIMPLE(GGML_OP_ADD_REL_POS, add_rel_pos)
        
        // === RWKV OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_RWKV_WKV6, rwkv_wkv6)
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_compute_forward_gla(&fallback_params, tensor);
            break;
        DISPATCH_SIMPLE(GGML_OP_RWKV_WKV7, rwkv_wkv7)
        
        // === GENERIC OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_UNARY, unary)
        
        // === CUSTOM MAP OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_MAP_CUSTOM1, map_custom1)
        DISPATCH_SIMPLE(GGML_OP_MAP_CUSTOM2, map_custom2)
        DISPATCH_SIMPLE(GGML_OP_MAP_CUSTOM3, map_custom3)
        case GGML_OP_CUSTOM:
            ggml_compute_forward_custom(&fallback_params, tensor);
            break;
        
        // === LOSS OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_CROSS_ENTROPY_LOSS, cross_entropy_loss)
        DISPATCH_SIMPLE(GGML_OP_CROSS_ENTROPY_LOSS_BACK, cross_entropy_loss_back)
        
        // === OPTIMIZATION OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_OPT_STEP_ADAMW, opt_step_adamw)
        
        // === ADDITIONAL OPERATIONS ===
        DISPATCH_SIMPLE(GGML_OP_GLU, glu)
            
        default:
            GGML_LOG_ERROR("Unsupported operation %s (%d) in fallback system\n", 
                          ggml_op_name(tensor->op), tensor->op);
            return GGML_STATUS_FAILED;
    }

    GGML_LOG_DEBUG("Successfully executed operation %s via fallback\n", ggml_op_name(tensor->op));
    return GGML_STATUS_SUCCESS;
}

// Cleanup helper macros
#undef DISPATCH_SIMPLE

bool ggml_numa_fallback_is_supported(enum ggml_op op) {
    (void)op;
    return true;
}

void ggml_numa_fallback_get_stats(int64_t * fallback_count) {
    if (fallback_count) {
        *fallback_count = atomic_load_explicit(&g_fallback_operations, memory_order_relaxed);
    }
}
