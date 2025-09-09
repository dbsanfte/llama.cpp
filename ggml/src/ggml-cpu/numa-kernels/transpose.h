/**
 * @file transpose.h
 * @brief NUMA TRANSPOSE Kernel Header - Tensor Dimension Swapping Operation
 *
 * This header defines the interface for the NUMA-aware TRANSPOSE kernel.
 * TRANSPOSE is a view operation that swaps the first two dimensions of a tensor
 * without performing any actual data movement.
 * 
 * @author David Sanftenberg
 */

#ifndef NUMA_KERNEL_TRANSPOSE_H
#define NUMA_KERNEL_TRANSPOSE_H

#include "../ggml-cpu-impl.h"
#include "../ggml-impl.h" 
#include "numa-kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief NUMA TRANSPOSE kernel execution function
 * 
 * Executes a TRANSPOSE operation (metadata-only, returns immediately).
 * 
 * @param work_context Pointer to tensor being transposed
 * @param params Compute parameters
 * @return GGML_STATUS_SUCCESS always
 */
enum ggml_status ggml_numa_kernel_transpose_execute(void * work_context, 
                                                     struct ggml_compute_params * params);

/**
 * @brief Query function for NUMA TRANSPOSE kernel strategy selection
 * 
 * @param tensor Target tensor for strategy selection
 * @return Single-thread strategy recommendation
 */
ggml_numa_execution_strategy_t ggml_numa_kernel_transpose_query(const struct ggml_tensor * tensor);

/**
 * @brief Work buffer calculation function for TRANSPOSE operations
 * 
 * @param tensor Target tensor
 * @param total_numa_nodes Number of NUMA nodes  
 * @param total_threads Total number of threads
 * @return 0 (no work buffer needed)
 */
size_t ggml_numa_kernel_transpose_work_buffer_calc(const struct ggml_tensor * tensor, 
                                                    int total_numa_nodes, 
                                                    int total_threads);

/**
 * @brief Register the TRANSPOSE kernel with the NUMA system
 * 
 * @return Registration information for the TRANSPOSE kernel
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_transpose_register(void);

#ifdef __cplusplus
}
#endif

#endif // NUMA_KERNEL_TRANSPOSE_H
