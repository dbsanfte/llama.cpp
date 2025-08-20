/*
 * NUMA Kernel Stubs - Placeholder implementations
 * These provide basic fallback implementations until full kernels are developed
 */

#include "rms_norm.h"
#include "soft_max.h"
#include "rope.h" 
#include "cpy.h"
#include "get_rows.h"

// ============================================================================
// RMS_NORM Kernel Stub
// ============================================================================

bool ggml_numa_kernel_rms_norm_supports(const struct ggml_tensor * tensor) {
    return tensor && tensor->op == GGML_OP_RMS_NORM;
}

enum ggml_status ggml_numa_kernel_rms_norm_execute(struct ggml_tensor * tensor, struct ggml_cplan * cplan) {
    GGML_LOG_WARN("RMS_NORM kernel: Using stub implementation\n");
    // TODO: Implement full NUMA-aware RMS normalization
    return GGML_STATUS_SUCCESS;
}

float ggml_numa_kernel_rms_norm_get_efficiency(const struct ggml_tensor * tensor, size_t tensor_size) {
    return 0.8f; // Placeholder efficiency
}

// ============================================================================
// SOFT_MAX Kernel Stub
// ============================================================================

bool ggml_numa_kernel_soft_max_supports(const struct ggml_tensor * tensor) {
    return tensor && tensor->op == GGML_OP_SOFT_MAX;
}

enum ggml_status ggml_numa_kernel_soft_max_execute(struct ggml_tensor * tensor, struct ggml_cplan * cplan) {
    GGML_LOG_WARN("SOFT_MAX kernel: Using stub implementation\n");
    // TODO: Implement full NUMA-aware softmax
    return GGML_STATUS_SUCCESS;
}

float ggml_numa_kernel_soft_max_get_efficiency(const struct ggml_tensor * tensor, size_t tensor_size) {
    return 0.7f; // Placeholder efficiency
}

// ============================================================================
// ROPE Kernel Stub
// ============================================================================

bool ggml_numa_kernel_rope_supports(const struct ggml_tensor * tensor) {
    return tensor && tensor->op == GGML_OP_ROPE;
}

enum ggml_status ggml_numa_kernel_rope_execute(struct ggml_tensor * tensor, struct ggml_cplan * cplan) {
    GGML_LOG_WARN("ROPE kernel: Using stub implementation\n");
    // TODO: Implement full NUMA-aware RoPE
    return GGML_STATUS_SUCCESS;
}

float ggml_numa_kernel_rope_get_efficiency(const struct ggml_tensor * tensor, size_t tensor_size) {
    return 0.9f; // Placeholder efficiency
}

// ============================================================================
// CPY Kernel Stub
// ============================================================================

bool ggml_numa_kernel_cpy_supports(const struct ggml_tensor * tensor) {
    return tensor && tensor->op == GGML_OP_CPY;
}

enum ggml_status ggml_numa_kernel_cpy_execute(struct ggml_tensor * tensor, struct ggml_cplan * cplan) {
    GGML_LOG_WARN("CPY kernel: Using stub implementation\n");
    // TODO: Implement full NUMA-aware copy
    return GGML_STATUS_SUCCESS;
}

float ggml_numa_kernel_cpy_get_efficiency(const struct ggml_tensor * tensor, size_t tensor_size) {
    return 0.95f; // Placeholder efficiency
}

// ============================================================================
// GET_ROWS Kernel Stub
// ============================================================================

bool ggml_numa_kernel_get_rows_supports(const struct ggml_tensor * tensor) {
    return tensor && tensor->op == GGML_OP_GET_ROWS;
}

enum ggml_status ggml_numa_kernel_get_rows_execute(struct ggml_tensor * tensor, struct ggml_cplan * cplan) {
    GGML_LOG_WARN("GET_ROWS kernel: Using stub implementation\n");
    // TODO: Implement full NUMA-aware get_rows
    return GGML_STATUS_SUCCESS;
}

float ggml_numa_kernel_get_rows_get_efficiency(const struct ggml_tensor * tensor, size_t tensor_size) {
    return 0.9f; // Placeholder efficiency
}
