// Example: Macro-Based Switch to Reduce Repetition
// Uses macros to make the giant switch more maintainable

#include "ggml.h"
#include "ops.h"

// Macro for simple direct dispatch operations
#define DISPATCH_SIMPLE(op_enum, forward_func) \
    case op_enum: \
        return forward_func(&fallback_params, tensor);

// Macro for operations needing special parameter handling  
#define DISPATCH_SPECIAL(op_enum, custom_code) \
    case op_enum: \
        custom_code \
        break;

// Macro for unary operations that all use the same handler
#define DISPATCH_UNARY(op_enum) \
    case op_enum: \
        return ggml_compute_forward_unary(&fallback_params, tensor);

// Much cleaner switch statement using macros
enum ggml_status ggml_numa_dispatch_via_macros(
    struct ggml_tensor * tensor, struct ggml_compute_params * fallback_params) {
    
    switch (tensor->op) {
        // No operation
        case GGML_OP_NONE:
            return GGML_STATUS_SUCCESS;
            
        // Basic math operations - all follow same pattern
        DISPATCH_SIMPLE(GGML_OP_ADD, ggml_compute_forward_add)
        DISPATCH_SIMPLE(GGML_OP_SUB, ggml_compute_forward_sub)
        DISPATCH_SIMPLE(GGML_OP_MUL, ggml_compute_forward_mul)
        DISPATCH_SIMPLE(GGML_OP_DIV, ggml_compute_forward_div)
        
        // Unary math operations - all use unary handler
        DISPATCH_UNARY(GGML_OP_SQR)
        DISPATCH_UNARY(GGML_OP_SQRT)
        DISPATCH_UNARY(GGML_OP_LOG)
        DISPATCH_UNARY(GGML_OP_SIN)
        DISPATCH_UNARY(GGML_OP_COS)
        
        // Matrix operations
        DISPATCH_SIMPLE(GGML_OP_MUL_MAT, ggml_compute_forward_mul_mat)
        DISPATCH_SIMPLE(GGML_OP_OUT_PROD, ggml_compute_forward_out_prod)
        
        // Tensor operations
        DISPATCH_SIMPLE(GGML_OP_DUP, ggml_compute_forward_dup)
        DISPATCH_SIMPLE(GGML_OP_CPY, ggml_compute_forward_cpy)
        DISPATCH_SIMPLE(GGML_OP_RESHAPE, ggml_compute_forward_reshape)
        DISPATCH_SIMPLE(GGML_OP_VIEW, ggml_compute_forward_view)
        DISPATCH_SIMPLE(GGML_OP_PERMUTE, ggml_compute_forward_permute)
        
        // Normalization operations
        DISPATCH_SIMPLE(GGML_OP_NORM, ggml_compute_forward_norm)
        DISPATCH_SIMPLE(GGML_OP_RMS_NORM, ggml_compute_forward_rms_norm)
        DISPATCH_SIMPLE(GGML_OP_GROUP_NORM, ggml_compute_forward_group_norm)
        
        // Special operations that need custom handling
        DISPATCH_SPECIAL(GGML_OP_FLASH_ATTN_EXT, {
            if (tensor->src[0] && tensor->src[1] && tensor->src[2]) {
                return ggml_compute_forward_flash_attn_ext(fallback_params,
                    tensor->src[0], tensor->src[1], tensor->src[2], 
                    tensor->src[3], tensor);
            } else {
                GGML_LOG_ERROR("FLASH_ATTN_EXT requires valid Q, K, V tensors\n");
                return GGML_STATUS_FAILED;
            }
        })
        
        DISPATCH_SPECIAL(GGML_OP_MUL_MAT_ID, {
            // MUL_MAT_ID uses same implementation as MUL_MAT
            return ggml_compute_forward_mul_mat(fallback_params, tensor);
        })
        
        default:
            GGML_LOG_ERROR("Unsupported operation %s\n", ggml_op_name(tensor->op));
            return GGML_STATUS_FAILED;
    }
    
    return GGML_STATUS_SUCCESS;
}

// Alternative: X-Macro approach for even more automation
#define OPERATION_LIST \
    X(GGML_OP_ADD, ggml_compute_forward_add) \
    X(GGML_OP_SUB, ggml_compute_forward_sub) \
    X(GGML_OP_MUL, ggml_compute_forward_mul) \
    X(GGML_OP_DIV, ggml_compute_forward_div) \
    X(GGML_OP_MUL_MAT, ggml_compute_forward_mul_mat) \
    X(GGML_OP_DUP, ggml_compute_forward_dup) \
    /* ... continue for all operations */

enum ggml_status ggml_numa_dispatch_via_x_macros(
    struct ggml_tensor * tensor, struct ggml_compute_params * params) {
    
    switch (tensor->op) {
        case GGML_OP_NONE:
            return GGML_STATUS_SUCCESS;
            
#define X(op_enum, forward_func) \
        case op_enum: \
            return forward_func(params, tensor);
        
        OPERATION_LIST
#undef X
        
        default:
            GGML_LOG_ERROR("Unsupported operation %s\n", ggml_op_name(tensor->op));
            return GGML_STATUS_FAILED;
    }
}

// Pros:
// - Maintains switch statement speed
// - Much less repetitive code
// - Easy to add new operations
// - Compiler still optimizes to jump table
// - Can handle special cases cleanly

// Cons:
// - Macros can be harder to debug
// - Some developers dislike heavy macro use
// - Need to be careful with macro hygiene
