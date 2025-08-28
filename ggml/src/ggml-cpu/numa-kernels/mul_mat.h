/**
 * @file mul_mat.h
 * @brief NUMA Kernel Header: Matrix Multiplication (MUL_MAT)
 * 
 * Public interface for NUMA-aware matrix multiplication kernel.
 */

#ifndef GGML_NUMA_KERNEL_MUL_MAT_H
#define GGML_NUMA_KERNEL_MUL_MAT_H

#include "ggml-cpu.h"
#include "numa-kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Execute NUMA-aware matrix multiplication kernel
 * 
 * @param work_context The tensor to process (cast from void*)
 * @param params       Thread parameters with thread ID and count
 * @return             GGML_STATUS_SUCCESS on success
 */
enum ggml_status ggml_numa_kernel_mul_mat_execute(void * work_context, struct ggml_compute_params * params);

/**
 * Calculate the exact work buffer size needed for MUL_MAT operation
 * 
 * This function analyzes the tensor types and dimensions to determine
 * the precise buffer size needed for type conversion operations.
 * 
 * @param tensor The MUL_MAT tensor to analyze
 * @return Size in bytes needed for work buffer, or 0 if no buffer needed
 */
size_t ggml_numa_kernel_mul_mat_calculate_work_buffer_size(const struct ggml_tensor * tensor);

/**
 * Query MUL_MAT kernel for optimal strategy based on tensor characteristics
 * 
 * This function analyzes the tensor and returns the optimal execution strategy
 * using operation-specific thresholds rather than rigid complexity classes.
 * 
 * @param tensor The tensor to analyze
 * @return Query result with optimal strategy, or unsupported result if not applicable
 */
ggml_numa_kernel_query_result_t ggml_numa_kernel_mul_mat_query(const struct ggml_tensor * tensor);

#ifdef __cplusplus
}
#endif

#endif // GGML_NUMA_KERNEL_MUL_MAT_H
