/**
 * @file mul_mat.h
 * @brief NUMA-aware matrix multiplication kernel
 * @author David Sanftenberg
 * 
 * Matrix multiplication kernel implementing the sophisticated 2D chunking
 * and block tiling strategy from the reference implementation with NUMA
 * optimizations.
 */

#pragma once

#include "numa-kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Register the NUMA matrix multiplication kernel
 * @return Registration information for the MUL_MAT operation
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_mul_mat_register(void);

/**
 * @brief Query execution strategy for matrix multiplication
 * @param tensor The tensor to query strategy for
 * @return Recommended execution strategy based on tensor characteristics
 */
ggml_numa_execution_strategy_t ggml_numa_kernel_mul_mat_query(const struct ggml_tensor * tensor);

/**
 * @brief Calculate work buffer size for matrix multiplication
 * @param tensor The tensor being processed
 * @param total_numa_nodes Total NUMA nodes participating
 * @param total_threads Total threads participating across all nodes
 * @return Total work buffer size needed for all threads
 */
size_t ggml_numa_kernel_mul_mat_work_buffer_calc(const struct ggml_tensor * tensor, int total_numa_nodes, int total_threads);

/**
 * @brief Execute matrix multiplication kernel
 * @param work_context The tensor to process
 * @param params Compute parameters
 * @return Status of the operation
 */
enum ggml_status ggml_numa_kernel_mul_mat_execute(void * work_context, struct ggml_compute_params * params);

#ifdef __cplusplus
}
#endif
