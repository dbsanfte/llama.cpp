/*
 * NUMA Kernel: Matrix Multiplication (MUL_MAT)
 * 
 * Implements NUMA-aware matrix multiplication with multiple execution strategies.
 * Based on the successful pattern from ggml-numa-mulmat.c with clean integration.
 */

#include "mul_mat.h"
#include "../ggml-numa-coordinator.h"
#include "../ggml-numa-work-shared.h"
#include "../ggml-cpu-impl.h"
#include "../ggml-impl.h"

// Re-use the proven implementation from ggml-numa-mulmat.c
// We'll include the headers and re-use the functions directly
extern enum ggml_status ggml_numa_mul_mat_dispatch(
    struct ggml_numa_coordinator_manager * manager,
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context);

extern size_t ggml_numa_mul_mat_calculate_work_buffer_size(const struct ggml_tensor * operation);

// ============================================================================
// Kernel Interface Implementation
// ============================================================================

bool ggml_numa_kernel_mul_mat_supports(const struct ggml_tensor * tensor) {
    if (!tensor || tensor->op != GGML_OP_MUL_MAT) {
        return false;
    }
    
    const struct ggml_tensor * src0 = tensor->src[0];
    const struct ggml_tensor * src1 = tensor->src[1];
    
    if (!src0 || !src1) {
        return false;
    }
    
    // Basic compatibility checks
    if (src1->type != GGML_TYPE_F32 && 
        src1->type != ggml_get_type_traits_cpu(src0->type)->vec_dot_type) {
        return false;
    }
    
    // Check tensor dimensions are compatible
    if (src0->ne[0] != src1->ne[0]) {
        return false;
    }
    
    return true;
}

enum ggml_status ggml_numa_kernel_mul_mat_execute(struct ggml_tensor * tensor, struct ggml_cplan * cplan) {
    if (!ggml_numa_kernel_mul_mat_supports(tensor)) {
        return GGML_STATUS_FAILED;
    }
    
    // Get NUMA coordinator
    struct ggml_numa_coordinator_manager * coordinator = 
        ggml_numa_coordinator_manager_get_global(cplan->n_threads);
    
    if (!coordinator) {
        GGML_LOG_ERROR("MUL_MAT kernel: NUMA coordinator not available\n");
        return GGML_STATUS_FAILED;
    }
    
    // Create work context for the operation
    ggml_numa_work_context_t context = {
        .numa_nodes = ggml_numa_coordinator_manager_get_numa_nodes(coordinator),
        .threads_per_node = cplan->n_threads / ggml_numa_coordinator_manager_get_numa_nodes(coordinator)
    };
    
    if (context.numa_nodes <= 0) context.numa_nodes = 1;
    if (context.threads_per_node <= 0) context.threads_per_node = cplan->n_threads;
    
    GGML_LOG_DEBUG("MUL_MAT kernel: Executing with %d NUMA nodes, %d threads per node\n",
                   context.numa_nodes, context.threads_per_node);
    
    // Delegate to the proven dispatch implementation
    return ggml_numa_mul_mat_dispatch(coordinator, tensor, &context);
}

float ggml_numa_kernel_mul_mat_get_efficiency(const struct ggml_tensor * tensor, size_t tensor_size) {
    (void)tensor_size;  // Unused for now
    
    if (!ggml_numa_kernel_mul_mat_supports(tensor)) {
        return -1.0f;
    }
    
    const struct ggml_tensor * src0 = tensor->src[0];
    const struct ggml_tensor * src1 = tensor->src[1];
    
    // Calculate computational complexity
    const int64_t ne01 = src0->ne[1];  // M dimension  
    const int64_t ne00 = src0->ne[0];  // K dimension
    const int64_t ne11 = src1->ne[1];  // N dimension
    
    const int64_t complexity = ne01 * ne00 * ne11;
    const int64_t COMPLEXITY_THRESHOLD = 10000000; // 10M operations
    
    // Base efficiency depends on matrix size and type
    float base_efficiency = 0.85f;
    
    // Quantized matrices can achieve higher efficiency
    if (ggml_is_quantized(src0->type)) {
        base_efficiency = 0.90f;
    }
    
    // Large matrices achieve better efficiency
    if (complexity > COMPLEXITY_THRESHOLD * 10) {
        base_efficiency *= 1.1f;
    } else if (complexity < COMPLEXITY_THRESHOLD / 10) {
        base_efficiency *= 0.7f;
    }
    
    return base_efficiency < 1.0f ? base_efficiency : 1.0f;
}
