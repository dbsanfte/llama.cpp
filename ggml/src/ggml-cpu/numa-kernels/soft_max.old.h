/**
 * @file soft_max.h
 * @brief NUMA Kernel Header: Softmax Activation (SOFT_MAX)
 * 
 * Public interface for NUMA-aware softmax activation kernel.
 */

#ifndef GGML_NUMA_KERNEL_SOFT_MAX_H
#define GGML_NUMA_KERNEL_SOFT_MAX_H

#include "ggml-cpu.h"
#include "numa-kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Execute NUMA-aware softmax activation kernel
 * 
 * @param work_context The tensor to process (cast from void*)
 * @param params       Thread parameters with thread ID and count
 * @return             GGML_STATUS_SUCCESS on success
 */
enum ggml_status ggml_numa_kernel_soft_max_execute(void * work_context, struct ggml_compute_params * params);

/**
 * Kernel registration function - provides strategy arrays and function pointers
 * 
 * @return Registration info containing strategies and function pointers for SOFT_MAX
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_soft_max_register(void);

/**
 * Query function for NUMA kernel strategy selection
 * 
 * @param tensor The tensor to query strategy for
 * @return Query result with selected strategy and efficiency
 */
ggml_numa_kernel_query_result_t ggml_numa_kernel_soft_max_query(const struct ggml_tensor * tensor);

#ifdef __cplusplus
}
#endif

#endif // GGML_NUMA_KERNEL_SOFT_MAX_H
