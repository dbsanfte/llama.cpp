// Example: Function Pointer Table Approach
// This would replace the giant switch statement while maintaining speed

#include "ggml.h"
#include "ops.h"

// Function pointer type for operation handlers
typedef enum ggml_status (*ggml_operation_handler_func_t)(
    const struct ggml_compute_params * params, 
    struct ggml_tensor * tensor
);

// Global dispatch table - initialized once at startup
static ggml_operation_handler_func_t g_operation_handlers[GGML_OP_COUNT] = {0};
static bool g_dispatch_table_initialized = false;

// Initialize the dispatch table once
static void init_operation_dispatch_table(void) {
    if (g_dispatch_table_initialized) return;
    
    // Basic operations
    g_operation_handlers[GGML_OP_ADD] = ggml_compute_forward_add;
    g_operation_handlers[GGML_OP_MUL] = ggml_compute_forward_mul;
    g_operation_handlers[GGML_OP_SUB] = ggml_compute_forward_sub;
    g_operation_handlers[GGML_OP_DIV] = ggml_compute_forward_div;
    
    // Matrix operations
    g_operation_handlers[GGML_OP_MUL_MAT] = ggml_compute_forward_mul_mat;
    g_operation_handlers[GGML_OP_OUT_PROD] = ggml_compute_forward_out_prod;
    
    // Tensor manipulation
    g_operation_handlers[GGML_OP_DUP] = ggml_compute_forward_dup;
    g_operation_handlers[GGML_OP_CPY] = ggml_compute_forward_cpy;
    g_operation_handlers[GGML_OP_CONT] = ggml_compute_forward_cont;
    g_operation_handlers[GGML_OP_RESHAPE] = ggml_compute_forward_reshape;
    g_operation_handlers[GGML_OP_VIEW] = ggml_compute_forward_view;
    g_operation_handlers[GGML_OP_PERMUTE] = ggml_compute_forward_permute;
    g_operation_handlers[GGML_OP_TRANSPOSE] = ggml_compute_forward_transpose;
    
    // Normalization
    g_operation_handlers[GGML_OP_NORM] = ggml_compute_forward_norm;
    g_operation_handlers[GGML_OP_RMS_NORM] = ggml_compute_forward_rms_norm;
    g_operation_handlers[GGML_OP_GROUP_NORM] = ggml_compute_forward_group_norm;
    
    // Activations
    g_operation_handlers[GGML_OP_SOFT_MAX] = ggml_compute_forward_soft_max;
    g_operation_handlers[GGML_OP_ROPE] = ggml_compute_forward_rope;
    
    // ... continue for all operations
    
    g_dispatch_table_initialized = true;
}

// Fast dispatch using function pointer table
enum ggml_status ggml_numa_dispatch_via_function_table(
    struct ggml_tensor * tensor, 
    struct ggml_compute_params * params) {
    
    // Initialize table if needed (once per program)
    init_operation_dispatch_table();
    
    // Validate operation
    if (tensor->op >= GGML_OP_COUNT || tensor->op < 0) {
        GGML_LOG_ERROR("Invalid operation %d\n", tensor->op);
        return GGML_STATUS_FAILED;
    }
    
    // Look up handler
    ggml_operation_handler_func_t handler = g_operation_handlers[tensor->op];
    
    if (!handler) {
        GGML_LOG_ERROR("No handler registered for operation %s\n", ggml_op_name(tensor->op));
        return GGML_STATUS_FAILED;
    }
    
    // Execute via function pointer - still very fast!
    return handler(params, tensor);
}

// Pros:
// - Still O(1) fast (array lookup + function call)
// - Much more compact code
// - Easy to extend dynamically
// - Can share handlers between similar operations
// - Less repetitive than giant switch

// Cons:
// - Slightly less type safe (function pointers)
// - Indirect calls (minor debugging complexity)
// - Need to ensure table is fully initialized
