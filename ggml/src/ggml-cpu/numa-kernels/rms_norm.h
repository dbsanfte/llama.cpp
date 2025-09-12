/**
 * @file rms_norm.h
 * @brief Header for NUMA RMS_NORM kernel (Root Mean Square normalization)
 * 
 * @author David Sanftenberg
 * @date 2025-09-09
 * 
 * Declares the interface for the NUMA-aware RMS_NORM kernel implementation.
 * RMS_NORM performs Root Mean Square normalization with row-wise reduction.
 */

#pragma once

#include "ggml.h"
#include "ggml-numa-shared.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Execute RMS_NORM operation with NUMA row-wise optimization
 * 
 * Performs Root Mean Square normalization on input tensor. Each row is
 * processed independently: computes mean of squares, then scales by
 * 1/sqrt(mean + eps) using SIMD-optimized operations.
 * 
 * @param work_context Pointer to destination tensor (cast from ggml_tensor*)
 * @param params Compute parameters containing thread information
 * @return GGML_STATUS_SUCCESS on success, GGML_STATUS_FAILED on error
 */
enum ggml_status ggml_numa_kernel_rms_norm_execute(void * work_context, struct ggml_compute_params * params);

/**
 * @brief Query optimal execution strategy for RMS_NORM operation
 * 
 * @param tensor Target tensor for RMS normalization
 * @return Optimal execution strategy based on tensor size and reduction characteristics
 */
ggml_numa_execution_strategy_t ggml_numa_kernel_rms_norm_query(const struct ggml_tensor * tensor);

/**
 * @brief Calculate work buffer size for RMS_NORM operation
 * 
 * @param tensor Target tensor (unused - no work buffer needed)
 * @param total_numa_nodes Total NUMA nodes (unused)
 * @param total_threads Total threads (unused)
 * @return 0 (no work buffer required for in-place row processing)
 */
size_t ggml_numa_kernel_rms_norm_work_buffer_calc(const struct ggml_tensor * tensor, int total_numa_nodes, int total_threads);

/**
 * @brief Register RMS_NORM kernel with NUMA system
 * 
 * @return Populated registration info structure for RMS_NORM kernel
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_rms_norm_register(void);

#ifdef __cplusplus
}
#endif
