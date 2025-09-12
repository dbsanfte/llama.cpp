/**
 * @file sub.h
 * @brief NUMA SUB kernel declarations demonstrating binary operation pattern reusability
 * @author David Sanftenberg
 */

#pragma once

#include "ggml.h"
#include "numa-kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Kernel Function Declarations - IDENTICAL TO ADD
// ============================================================================

/**
 * @brief NUMA SUB kernel execution function
 */
enum ggml_status ggml_numa_kernel_sub_unified_execute(void * work_context, struct ggml_compute_params * params);

/**
 * @brief Query SUB kernel capabilities and optimal strategy
 */
ggml_numa_execution_strategy_t ggml_numa_kernel_sub_query(const struct ggml_tensor * tensor);

/**
 * @brief Calculate work buffer requirements for SUB kernel
 */
size_t ggml_numa_kernel_sub_work_buffer_calc(const struct ggml_tensor * tensor, int total_numa_nodes, int total_threads);

/**
 * @brief Register SUB kernel with NUMA system
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_sub_register(void);

#ifdef __cplusplus
}
#endif
