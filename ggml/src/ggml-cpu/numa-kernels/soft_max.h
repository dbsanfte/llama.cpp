/**
 * @file soft_max.h
 * @brief NUMA SOFT_MAX (Softmax Activation) Kernel Header
 * @author David Sanftenberg
 * 
 * ============================================================================
 * NUMA KERNEL HEADER: SOFT_MAX (Softmax Activation)
 * ============================================================================
 */

#ifndef GGML_NUMA_KERNEL_SOFT_MAX_H
#define GGML_NUMA_KERNEL_SOFT_MAX_H

#include "numa-kernels.h"
#include "../ggml-numa-shared.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// SOFT_MAX Kernel Function Declarations
// ============================================================================

/**
 * @brief Execute SOFT_MAX kernel computation
 * @param work_context Pointer to tensor being processed
 * @param params Compute parameters from coordinator
 * @return Status of the computation
 */
enum ggml_status ggml_numa_kernel_soft_max_execute(void * work_context, struct ggml_compute_params * params);

/**
 * @brief Query optimal execution strategy for SOFT_MAX operation
 * @param tensor Tensor to be processed
 * @return Recommended execution strategy
 */
ggml_numa_execution_strategy_t ggml_numa_kernel_soft_max_query(const struct ggml_tensor * tensor);

/**
 * @brief Calculate work buffer size needed for SOFT_MAX operation
 * @param tensor Tensor to be processed
 * @param total_numa_nodes Total NUMA nodes participating
 * @param total_threads Total threads participating across all nodes  
 * @return Work buffer size in bytes
 */
size_t ggml_numa_kernel_soft_max_work_buffer_calc(const struct ggml_tensor * tensor, int total_numa_nodes, int total_threads);

/**
 * @brief Register SOFT_MAX kernel with NUMA system
 * @return Registration information structure
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_soft_max_register(void);

#ifdef __cplusplus
}
#endif

#endif // GGML_NUMA_KERNEL_SOFT_MAX_H
