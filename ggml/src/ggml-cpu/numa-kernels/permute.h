/**
 * @file permute.h
 * @brief Header for NUMA PERMUTE kernel (metadata-only tensor dimension permutation)
 * 
 * @author David Sanftenberg
 * @date 2025-09-09
 * 
 * Declares the interface for the NUMA-aware PERMUTE kernel implementation.
 * PERMUTE is a metadata-only operation that reorders tensor dimensions.
 */

#pragma once

#include "ggml.h"
#include "ggml-numa-shared.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Execute PERMUTE operation (metadata-only, no-op implementation)
 * 
 * Performs tensor dimension permutation. Since PERMUTE is metadata-only,
 * this function performs no actual computation - the permutation is handled
 * by the ggml tensor system at the metadata level.
 * 
 * @param work_context Pointer to destination tensor (cast from ggml_tensor*)
 * @param params Compute parameters (unused for no-op)
 * @return GGML_STATUS_SUCCESS always
 */
enum ggml_status ggml_numa_kernel_permute_execute(void * work_context, struct ggml_compute_params * params);

/**
 * @brief Query optimal execution strategy for PERMUTE operation
 * 
 * @param tensor Target tensor for permutation
 * @return NUMA_STRATEGY_SINGLE_THREAD (metadata-only operations use minimal overhead)
 */
ggml_numa_execution_strategy_t ggml_numa_kernel_permute_query(const struct ggml_tensor * tensor);

/**
 * @brief Calculate work buffer size for PERMUTE operation
 * 
 * @param tensor Target tensor (unused - no work buffer needed)
 * @param total_numa_nodes Total NUMA nodes (unused)
 * @param total_threads Total threads (unused)
 * @return 0 (no work buffer required for metadata-only operations)
 */
size_t ggml_numa_kernel_permute_work_buffer_calc(const struct ggml_tensor * tensor, int total_numa_nodes, int total_threads);

/**
 * @brief Register PERMUTE kernel with NUMA system
 * 
 * @return Populated registration info structure for PERMUTE kernel
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_permute_register(void);

#ifdef __cplusplus
}
#endif
