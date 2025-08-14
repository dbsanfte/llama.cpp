// Example: Categorized Dispatch Approach
// Groups operations by type for cleaner organization

#include "ggml.h"
#include "ops.h"

typedef enum {
    OP_CATEGORY_MATH,      // ADD, SUB, MUL, DIV, etc.
    OP_CATEGORY_MATRIX,    // MUL_MAT, OUT_PROD, etc.
    OP_CATEGORY_TENSOR,    // RESHAPE, VIEW, PERMUTE, etc.
    OP_CATEGORY_NORM,      // NORM, RMS_NORM, GROUP_NORM, etc.
    OP_CATEGORY_ACTIVATION,// SOFT_MAX, ROPE, RELU, etc.
    OP_CATEGORY_CONV,      // CONV_2D, POOL_2D, etc.
    OP_CATEGORY_ATTENTION, // FLASH_ATTN_EXT, etc.
    OP_CATEGORY_SPECIAL,   // Custom, complex operations
} operation_category_t;

// Fast category lookup table
static const operation_category_t g_op_categories[GGML_OP_COUNT] = {
    [GGML_OP_ADD] = OP_CATEGORY_MATH,
    [GGML_OP_SUB] = OP_CATEGORY_MATH,
    [GGML_OP_MUL] = OP_CATEGORY_MATH,
    [GGML_OP_DIV] = OP_CATEGORY_MATH,
    
    [GGML_OP_MUL_MAT] = OP_CATEGORY_MATRIX,
    [GGML_OP_OUT_PROD] = OP_CATEGORY_MATRIX,
    
    [GGML_OP_RESHAPE] = OP_CATEGORY_TENSOR,
    [GGML_OP_VIEW] = OP_CATEGORY_TENSOR,
    [GGML_OP_PERMUTE] = OP_CATEGORY_TENSOR,
    
    [GGML_OP_NORM] = OP_CATEGORY_NORM,
    [GGML_OP_RMS_NORM] = OP_CATEGORY_NORM,
    
    // ... etc
};

// Category-specific dispatchers with shared logic
static enum ggml_status dispatch_math_operations(
    enum ggml_op op, struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    
    // Shared pre-processing for math operations
    if (tensor->src[0] && tensor->src[1] && !ggml_are_same_shape(tensor->src[0], tensor->src[1])) {
        GGML_LOG_DEBUG("Math op %s with broadcasting\n", ggml_op_name(op));
    }
    
    switch (op) {
        case GGML_OP_ADD: return ggml_compute_forward_add(params, tensor);
        case GGML_OP_SUB: return ggml_compute_forward_sub(params, tensor);
        case GGML_OP_MUL: return ggml_compute_forward_mul(params, tensor);
        case GGML_OP_DIV: return ggml_compute_forward_div(params, tensor);
        case GGML_OP_SQR: return ggml_compute_forward_sqr(params, tensor);
        case GGML_OP_SQRT: return ggml_compute_forward_sqrt(params, tensor);
        default:
            GGML_LOG_ERROR("Unknown math operation: %s\n", ggml_op_name(op));
            return GGML_STATUS_FAILED;
    }
}

static enum ggml_status dispatch_matrix_operations(
    enum ggml_op op, struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    
    // Shared pre-processing for matrix operations
    if (tensor->src[0] && tensor->src[1]) {
        int64_t k_dim_a = tensor->src[0]->ne[0];
        int64_t k_dim_b = tensor->src[1]->ne[0]; 
        if (op == GGML_OP_MUL_MAT && k_dim_a != k_dim_b) {
            GGML_LOG_ERROR("Matrix multiplication dimension mismatch: %ld != %ld\n", k_dim_a, k_dim_b);
            return GGML_STATUS_FAILED;
        }
    }
    
    switch (op) {
        case GGML_OP_MUL_MAT: return ggml_compute_forward_mul_mat(params, tensor);
        case GGML_OP_MUL_MAT_ID: return ggml_compute_forward_mul_mat(params, tensor);
        case GGML_OP_OUT_PROD: return ggml_compute_forward_out_prod(params, tensor);
        default:
            GGML_LOG_ERROR("Unknown matrix operation: %s\n", ggml_op_name(op));
            return GGML_STATUS_FAILED;
    }
}

// Main categorized dispatcher
enum ggml_status ggml_numa_dispatch_via_categories(
    struct ggml_tensor * tensor, struct ggml_compute_params * params) {
    
    if (tensor->op >= GGML_OP_COUNT || tensor->op < 0) {
        GGML_LOG_ERROR("Invalid operation %d\n", tensor->op);
        return GGML_STATUS_FAILED;
    }
    
    operation_category_t category = g_op_categories[tensor->op];
    
    switch (category) {
        case OP_CATEGORY_MATH:
            return dispatch_math_operations(tensor->op, params, tensor);
            
        case OP_CATEGORY_MATRIX:
            return dispatch_matrix_operations(tensor->op, params, tensor);
            
        case OP_CATEGORY_TENSOR:
            return dispatch_tensor_operations(tensor->op, params, tensor);
            
        // ... other categories
            
        default:
            GGML_LOG_ERROR("Unknown category for operation %s\n", ggml_op_name(tensor->op));
            return GGML_STATUS_FAILED;
    }
}

// Pros:
// - Logical grouping makes maintenance easier
// - Can share category-specific logic (validation, preprocessing)
// - Smaller switch statements
// - Still relatively fast (two lookups)

// Cons:
// - Two-level dispatch (slightly slower than direct)
// - Need to categorize all operations correctly
// - More complex than direct dispatch
