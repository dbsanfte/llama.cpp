/**
 * @file mul_mat.h
 * @brief NUMA Matrix Multiplication (MUL_MAT) Kernel Interface
 * 
 * NUMA-aware implementation of matrix multiplication operations with support for:
 * - All quantization types supported by reference implementation
 * - Optimized data-parallel execution across NUMA nodes
 * - Chunk-based work distribution for optimal cache utilization
 * - Type-specific SIMD operations with vec_dot dispatch
 * - Work buffer management for type conversions
 */

#pragma once

#include "numa-kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// NUMA MUL_MAT Kernel Registration
// ============================================================================

/**
 * Register MUL_MAT kernel with NUMA strategy array and work functions
 * Returns registration info for the NUMA kernel registry system
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_mul_mat_register(void);

/**
 * Query MUL_MAT kernel for optimal execution strategy
 * Returns strategy recommendation and kernel info for given tensor
 */
ggml_numa_execution_strategy_t ggml_numa_kernel_mul_mat_query(const struct ggml_tensor * tensor);

/**
 * Calculate work buffer size for MUL_MAT operation
 * Returns total work buffer size in bytes for type conversion operations
 */
size_t ggml_numa_kernel_mul_mat_work_buffer_calc(const struct ggml_tensor * tensor, int total_numa_nodes, int total_threads);

// ============================================================================
// NUMA MUL_MAT Kernel Work Functions
// ============================================================================

/**
 * NUMA MUL_MAT kernel execution function
 * Supports all quantization types with NUMA-aware chunk-based parallelization
 * 
 * @param work_context   Tensor containing MUL_MAT operation parameters
 * @param params         Compute parameters (threading, NUMA context, work buffer)
 * @return               GGML_STATUS_SUCCESS on success, error code on failure
 */
enum ggml_status ggml_numa_kernel_mul_mat_execute(void * work_context, struct ggml_compute_params * params);

#ifdef __cplusplus
}
#endif
