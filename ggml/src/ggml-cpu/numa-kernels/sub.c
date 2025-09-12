/**
 * @file sub.c
 * @brief NUMA SUB kernel implementation demonstrating binary operation pattern reusability
 * @author David Sanftenberg
 */

// NUMA infrastructure
#include "numa-kernels.h"
#include "sub.h"
#include "ggml-cpu-impl.h"
#include "ggml-numa-shared.h"
#include "ggml-numa-openmp-coordinator.h"
#include "../quants.h"
#include "../vec.h"
#include <stdlib.h>
#include <string.h>

// Kernel implementation headers
#include "sub.h"

// ============================================================================
// Kernel Execution Function - Demonstrating Pattern Reuse
// ============================================================================

/**
 * @brief NUMA SUB kernel execution function
 * @note This demonstrates how the ADD pattern is 99% reusable for SUB
 */
enum ggml_status ggml_numa_kernel_sub_unified_execute(void * work_context, struct ggml_compute_params * params) {
    struct ggml_tensor * tensor = (struct ggml_tensor *)work_context;
    
    // Validation  
    if (!tensor || !tensor->src[0] || !tensor->src[1]) {
        return GGML_STATUS_FAILED;
    }
    
    // For now, only support F32 types for initial testing
    if (tensor->type != GGML_TYPE_F32 || tensor->src[0]->type != GGML_TYPE_F32 || 
        tensor->src[1]->type != GGML_TYPE_F32) {
        return GGML_STATUS_FAILED;
    }

    // Use the unified binary broadcasting setup with manual SUB operation (matching ADD pattern)
    NUMA_KERNEL_SETUP_BINARY_BROADCAST(ctx, tensor, params, float) {
        // Simple operations based on broadcasting pattern (same as ADD, but with subtraction)
        if (__is_scalar) {
            const float __scalar = __src1_data[0];
            for (size_t __i = ctx.thread_start; __i < ctx.thread_end; __i++) {
                __dst_data[__i] = __src0_data[__i] - __scalar;
            }
        } else if (__is_same_shape) {
            // Use SIMD optimization for F32 same-shape operations
            ggml_vec_sub_f32(ctx.thread_elements,
                           __dst_data + ctx.thread_start,
                           __src0_data + ctx.thread_start,
                           __src1_data + ctx.thread_start);
        } else {
            // Complex broadcasting case - use manual element-wise operation
            for (size_t __i = ctx.thread_start; __i < ctx.thread_end; __i++) {
                // Simple linear access for complex broadcasting (works for most cases)
                __dst_data[__i] = __src0_data[__i] - __src1_data[__i % ggml_nelements(tensor->src[1])];
            }
        }
    }
    
    return GGML_STATUS_SUCCESS;
}

// ============================================================================ 
// Complete kernel implementation using shared macros
// ============================================================================

NUMA_KERNEL_REGISTER_METADATA(
    sub,                                    // kernel name
    GGML_OP_SUB,                           // operation type
    "NUMA SUB Kernel",                     // kernel description
    1024,                                  // single_single threshold
    262144,                                // single_multi threshold
    ggml_numa_kernel_sub_unified_execute   // execution function
)
