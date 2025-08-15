#include "ggml-numa-fallback.h"
#include "ggml-cpu-impl.h"
#include "ggml-cpu.h"    // For ggml_cplan structure definition
#include "ops.h"         // For operation functions
#include "unary-ops.h"
#include "binary-ops.h"

#include <stdatomic.h>

static atomic_int_fast64_t g_fallback_operations = 0;

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

enum ggml_status ggml_numa_fallback_execute(struct ggml_tensor * tensor, struct ggml_cplan * cplan) {
    if (!tensor) {
        GGML_LOG_ERROR("Invalid tensor for fallback execution\n");
        return GGML_STATUS_FAILED;
    }
    
    atomic_fetch_add_explicit(&g_fallback_operations, 1, memory_order_relaxed);
    
    // Use work buffer from cplan if available (coordinator should have set this up)
    void * work_data = cplan ? cplan->work_data : NULL;
    size_t work_size = cplan ? cplan->work_size : 0;
    
    // Create single-threaded compute params to avoid threadpool conflicts
    struct ggml_compute_params fallback_params = {
        .ith = 0,                  // Single thread index
        .nth = 1,                  // Single thread total
        .wsize = work_size,
        .wdata = work_data,
        .threadpool = NULL         // Critical: no threadpool conflicts
    };
    
    // Execute the operation directly using the operation-specific compute functions
    GGML_LOG_DEBUG("Executing operation %s via single-threaded fallback\n", 
                  ggml_op_name(tensor->op));

    // Get the operation executor based on tensor operation type
    switch (tensor->op) {
        // Binary operations (most common)
        DISPATCH_SIMPLE(GGML_OP_ADD, add)
        DISPATCH_SIMPLE(GGML_OP_SUB, sub)
        DISPATCH_SIMPLE(GGML_OP_MUL, mul)
        DISPATCH_SIMPLE(GGML_OP_DIV, div)
        
        // Unary math operations
        DISPATCH_SIMPLE(GGML_OP_SQR, sqr)
        DISPATCH_SIMPLE(GGML_OP_SQRT, sqrt)
        DISPATCH_SIMPLE(GGML_OP_LOG, log)
        DISPATCH_SIMPLE(GGML_OP_SIN, sin)
        DISPATCH_SIMPLE(GGML_OP_COS, cos)
        
        // Normalization operations
        DISPATCH_SIMPLE(GGML_OP_NORM, norm)
        DISPATCH_SIMPLE(GGML_OP_RMS_NORM, rms_norm)
        DISPATCH_SIMPLE(GGML_OP_GROUP_NORM, group_norm)
        
        // Attention operations
        DISPATCH_SIMPLE(GGML_OP_SOFT_MAX, soft_max)
        DISPATCH_SIMPLE(GGML_OP_ROPE, rope)
        
        // Activation operations
        DISPATCH_SIMPLE(GGML_OP_GLU, glu)
        
        // Matrix operations
        case GGML_OP_MUL_MAT:
            // NOTE: MUL_MAT operations should be handled by NUMA coordinator, not fallback
            // Reject these operations so they get routed to the proper dispatcher
            GGML_LOG_DEBUG("MUL_MAT operations should be handled by NUMA coordinator, not fallback\n");
            return GGML_STATUS_FAILED;
        
        // View operations (no computation)
        DISPATCH_SIMPLE(GGML_OP_VIEW, view)
        DISPATCH_SIMPLE(GGML_OP_RESHAPE, reshape)
        DISPATCH_SIMPLE(GGML_OP_PERMUTE, permute)
        DISPATCH_SIMPLE(GGML_OP_TRANSPOSE, transpose)
        
        // Data movement operations
        DISPATCH_SIMPLE(GGML_OP_CPY, cpy)
        DISPATCH_SIMPLE(GGML_OP_CONT, cont)
        DISPATCH_SIMPLE(GGML_OP_DUP, dup)
        
        // Slice operations
        DISPATCH_SIMPLE(GGML_OP_GET_ROWS, get_rows)
        DISPATCH_SIMPLE(GGML_OP_GET_ROWS_BACK, get_rows_back)
        
        // Special operations
        DISPATCH_SIMPLE(GGML_OP_CLAMP, clamp)
        DISPATCH_SIMPLE(GGML_OP_SCALE, scale)
        DISPATCH_SIMPLE(GGML_OP_SET, set)
        
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
