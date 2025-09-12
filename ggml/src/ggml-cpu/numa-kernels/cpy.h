/**
 * @file cpy.h
 * @brief NUMA-aware CPY/DUP kernel header with type conversion support
 * @author David Sanftenberg
 */

#pragma once

#include "ggml.h"
#include "ggml-numa-shared.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Execute CPY operation using NUMA kernels with type conversion support
 * 
 * Handles tensor copying with optional type conversion between:
 * - Same types (optimized memcpy path)
 * - F32 ↔ F16 conversion
 * - F32 ↔ BF16 conversion  
 * - Quantized → F32 dequantization
 * 
 * @param work_context Tensor context (struct ggml_tensor*)
 * @param params Compute parameters with threading info
 * @return GGML_STATUS_SUCCESS on completion, GGML_STATUS_FAILED on error
 */
enum ggml_status ggml_numa_kernel_cpy_execute(void * work_context, struct ggml_compute_params * params);

/**
 * @brief Query execution strategy for CPY operations
 * @param tensor Target tensor
 * @return Recommended NUMA execution strategy
 */
ggml_numa_execution_strategy_t ggml_numa_kernel_cpy_query(const struct ggml_tensor * tensor);

/**
 * @brief Register CPY kernel with metadata
 * @return Kernel registration information
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_cpy_register(void);

/**
 * @brief Calculate work buffer size for CPY operations (unused - CPY doesn't need work buffers)
 * @param tensor Target tensor
 * @param total_numa_nodes Number of NUMA nodes
 * @param total_threads Total number of threads
 * @return Work buffer size (always 0 for CPY)
 */
size_t ggml_numa_kernel_cpy_work_buffer_calc(const struct ggml_tensor * tensor, int total_numa_nodes, int total_threads);

#ifdef __cplusplus
}
#endif
