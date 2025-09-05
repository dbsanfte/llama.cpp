/**
 * @file permute.h
 * @brief NUMA Kernel Header: Tensor Permutation (PERMUTE)
 * 
 * Public interface for NUMA-aware tensor permutation kernel.
 */

#ifndef GGML_NUMA_KERNEL_PERMUTE_H
#define GGML_NUMA_KERNEL_PERMUTE_H

#include "ggml-cpu.h"
#include "numa-kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Execute NUMA-aware tensor permutation kernel
 * 
 * @param work_context The tensor to process (cast from void*)
 * @param params       Thread parameters with thread ID and count
 * @return             GGML_STATUS_SUCCESS on success
 */
enum ggml_status ggml_numa_kernel_permute_execute(void * work_context, struct ggml_compute_params * params);

/**
 * Kernel registration function - provides strategy arrays and function pointers
 * 
 * @return Registration info containing strategies and function pointers for PERMUTE
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_permute_register(void);

/**
 * Calculate the exact work buffer size needed for PERMUTE operation
 * 
 * @param tensor The PERMUTE tensor to analyze
 * @return Size in bytes needed for work buffer, or 0 if no buffer needed
 */
size_t ggml_numa_kernel_permute_calculate_work_buffer_size(const struct ggml_tensor * tensor);

/**
 * Query PERMUTE kernel for optimal strategy based on tensor characteristics
 * 
 * @param tensor The tensor to analyze
 * @return Query result with optimal strategy, or unsupported result if not applicable
 */
ggml_numa_kernel_query_result_t ggml_numa_kernel_permute_query(const struct ggml_tensor * tensor);

#ifdef __cplusplus
}
#endif

#endif // GGML_NUMA_KERNEL_PERMUTE_H
