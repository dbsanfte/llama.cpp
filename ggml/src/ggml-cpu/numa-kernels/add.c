/*
 * NUMA Kernel: Element-wise Addition (ADD)
 * 
 * Simplified implementation for the new architecture.
 * Uses direct tensor access rather than complex work dispatching.
 */

#include "add.h"
#include "../ggml-numa-coordinator.h"
#include "../ggml-cpu-impl.h"
#include "../ggml-impl.h"
#include "../vec.h"  // For SIMD functions

// ============================================================================
// Kernel Interface Implementation
// ============================================================================

bool ggml_numa_kernel_add_supports(const struct ggml_tensor * tensor) {
    if (!tensor || tensor->op != GGML_OP_ADD) {
        return false;
    }
    
    const struct ggml_tensor * src0 = tensor->src[0];
    const struct ggml_tensor * src1 = tensor->src[1];
    
    if (!src0 || !src1) {
        return false;
    }
    
    // Only support f32 for now (can be extended)
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || tensor->type != GGML_TYPE_F32) {
        return false;
    }
    
    // Check compatible shapes
    if (!ggml_are_same_shape(src0, src1) || !ggml_are_same_shape(src0, tensor)) {
        return false;
    }
    
    return true;
}

enum ggml_status ggml_numa_kernel_add_execute(struct ggml_tensor * tensor, struct ggml_cplan * cplan) {
    if (!ggml_numa_kernel_add_supports(tensor)) {
        return GGML_STATUS_FAILED;
    }
    
    const struct ggml_tensor * src0 = tensor->src[0];
    const struct ggml_tensor * src1 = tensor->src[1];
    
    const int64_t total_elements = ggml_nelements(tensor);
    
    GGML_LOG_DEBUG("ADD kernel: Processing %ld elements with %d threads\n", 
                   total_elements, cplan ? cplan->n_threads : 1);
    
    // For now, use a simple single-threaded implementation
    // TODO: Add NUMA-aware threading
    
    // Direct SIMD addition across all elements
    ggml_vec_add_f32(total_elements,
        (float *)ggml_get_data(tensor),
        (const float *)ggml_get_data(src0),
        (const float *)ggml_get_data(src1));
    
    return GGML_STATUS_SUCCESS;
}

float ggml_numa_kernel_add_get_efficiency(const struct ggml_tensor * tensor, size_t tensor_size) {
    if (!ggml_numa_kernel_add_supports(tensor)) {
        return -1.0f;
    }
    
    // ADD is highly parallelizable - excellent efficiency for large tensors
    const size_t OPTIMAL_SIZE = 4096;
    
    if (tensor_size >= OPTIMAL_SIZE * 4) {
        return 0.95f;  // Excellent efficiency for large tensors
    } else if (tensor_size >= OPTIMAL_SIZE) {
        return 0.85f;  // Good efficiency for medium tensors
    } else {
        return 0.6f;   // Reduced efficiency for small tensors
    }
}
