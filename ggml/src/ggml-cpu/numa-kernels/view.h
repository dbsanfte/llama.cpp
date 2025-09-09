/**
 * @file view.h
 * @brief NUMA VIEW Kernel Header - Tensor View Metadata Operation
 *
 * This header defines the interface for the NUMA-aware VIEW kernel.
 * VIEW is a metadata-only operation that creates a view of an existing tensor
 * with potentially different shape, offset, or strides without performing any
 * data movement or computation.
 * 
 * @author David Sanftenberg
 */

#ifndef NUMA_KERNEL_VIEW_H
#define NUMA_KERNEL_VIEW_H

#include "../ggml-cpu-impl.h"
#include "../ggml-impl.h" 
#include "numa-kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief NUMA VIEW kernel execution function
 * 
 * Executes a VIEW operation (metadata-only, returns immediately).
 * 
 * @param work_context Pointer to tensor being viewed
 * @param params Compute parameters
 * @return GGML_STATUS_SUCCESS always
 */
enum ggml_status ggml_numa_kernel_view_execute(void * work_context, 
                                               struct ggml_compute_params * params);

/**
 * @brief Query function for NUMA VIEW kernel strategy selection
 * 
 * @param tensor Target tensor for strategy selection
 * @return Single-thread strategy recommendation
 */
ggml_numa_execution_strategy_t ggml_numa_kernel_view_query(const struct ggml_tensor * tensor);

/**
 * @brief Work buffer calculation function for VIEW operations
 * 
 * @param tensor Target tensor
 * @param total_numa_nodes Number of NUMA nodes  
 * @param total_threads Total number of threads
 * @return 0 (no work buffer needed)
 */
size_t ggml_numa_kernel_view_work_buffer_calc(const struct ggml_tensor * tensor, 
                                               int total_numa_nodes, 
                                               int total_threads);

/**
 * @brief Register the VIEW kernel with the NUMA system
 * 
 * @return Registration information for the VIEW kernel
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_view_register(void);

#ifdef __cplusplus
}
#endif

#endif // NUMA_KERNEL_VIEW_H
