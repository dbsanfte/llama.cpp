/**
 * @file div.c
 * @brief NUMA-aware element-wise division kernel implementation using shared broadcasting macros
 * 
 * This kernel demonstrates 99% code reuse from ADD kernel using the shared macro framework.
 * Only the arithmetic operation changes: val0 / val1 instead of val0 + val1.
 * 
 * @author David Sanftenberg
 */

#include "numa-kernels.h"
#include "div.h"
#include "ggml-cpu-impl.h"
#include "ggml-numa-shared.h"
#include "ggml-numa-openmp-coordinator.h"
#include "../quants.h"
#include "../vec.h"
#include <stdlib.h>
#include <string.h>

/**
 * @brief Execute F32 element-wise division using NUMA coordinator and macro-based broadcasting
 * 
 * Uses NUMA_COMPLEX_BROADCAST_LOOP macro for all coordinate calculation and broadcasting logic.
 * This provides 99% code reuse with ADD kernel - only the arithmetic operation differs.
 * 
 * @param work_context Tensor context (struct ggml_tensor*)
 * @param params Compute parameters with NUMA threading info
 * @return GGML_STATUS_SUCCESS on completion, GGML_STATUS_FAILED on error
 */
enum ggml_status ggml_numa_kernel_div_execute(void * work_context, struct ggml_compute_params * params) {
    struct ggml_tensor * tensor = (struct ggml_tensor *)work_context;
    
    // Input validation
    if (!tensor || !tensor->src[0] || !tensor->src[1]) {
        NUMA_LOG_DEBUG("DIV kernel: NULL tensor or source inputs\n");
        return GGML_STATUS_FAILED;
    }
    
    // For now, only support F32 types for initial testing
    if (tensor->type != GGML_TYPE_F32 || tensor->src[0]->type != GGML_TYPE_F32 || 
        tensor->src[1]->type != GGML_TYPE_F32) {
        return GGML_STATUS_FAILED;
    }

    // Use the new unified binary broadcasting setup and operation
    NUMA_KERNEL_SETUP_BINARY_BROADCAST(ctx, tensor, params, float) {
        // Simple operations based on broadcasting pattern
        if (__is_scalar) {
            const float __scalar = __src1_data[0];
            for (size_t __i = ctx.thread_start; __i < ctx.thread_end; __i++) {
                __dst_data[__i] = __src0_data[__i] / __scalar;  // DIV operation
            }
        } else if (__is_same_shape) {
            // Use element-wise division for F32 same-shape operations
            for (size_t __i = 0; __i < ctx.thread_elements; __i++) {
                __dst_data[ctx.thread_start + __i] = __src0_data[ctx.thread_start + __i] / __src1_data[ctx.thread_start + __i];
            }
        } else {
            // Complex broadcasting case - use reusable macro
            NUMA_COMPLEX_BROADCAST_LOOP(ctx, tensor, float, val0 / val1);
        }
    }
    
    return GGML_STATUS_SUCCESS;
}

// ============================================================================ 
// Complete kernel implementation using shared macros
// ============================================================================

NUMA_KERNEL_REGISTER_METADATA(
    div,                                    // kernel name
    GGML_OP_DIV,                           // operation type
    "NUMA DIV Kernel",                     // kernel description
    1024,                                  // single_single threshold
    262144,                                // single_multi threshold
    ggml_numa_kernel_div_execute           // execution function
)


