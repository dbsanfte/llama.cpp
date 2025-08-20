/*
 * NUMA Kernel: Matrix Multiplication (MUL_MAT)
 * 
 * This kernel implements NUMA-aware matrix multiplication following the
 * established pattern from ggml-numa-mulmat.c with proper ggml integration.
 */

#pragma once

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"  // For complete ggml_cplan definition

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Check if this kernel supports the given tensor configuration
 * 
 * @param tensor The MUL_MAT operation tensor to check
 * @return true if supported, false otherwise
 */
bool ggml_numa_kernel_mul_mat_supports(const struct ggml_tensor * tensor);

/**
 * Execute MUL_MAT operation with optimal NUMA strategy
 * 
 * @param tensor The operation tensor
 * @param cplan The compute plan with threading and buffer info
 * @return GGML_STATUS_SUCCESS on success, error code on failure
 */
enum ggml_status ggml_numa_kernel_mul_mat_execute(struct ggml_tensor * tensor, struct ggml_cplan * cplan);

/**
 * Get efficiency estimate for this tensor configuration
 * 
 * @param tensor The operation tensor
 * @param tensor_size Size in elements
 * @return Efficiency estimate (0.0-1.0)
 */
float ggml_numa_kernel_mul_mat_get_efficiency(const struct ggml_tensor * tensor, size_t tensor_size);

#ifdef __cplusplus
}
#endif
