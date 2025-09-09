/**
 * @file reshape.h
 * @brief NUMA RESHAPE Kernel Header - Tensor Shape Metadata Operation
 *
 * This header defines the interface for the NUMA-aware RESHAPE kernel.
 * RESHAPE is a metadata-only operation that changes tensor dimensions
 * without performing any data movement or computation.
 * 
 * @author David Sanftenberg
 */

#ifndef NUMA_KERNEL_RESHAPE_H
#define NUMA_KERNEL_RESHAPE_H

#include "../ggml-cpu-impl.h"
#include "../ggml-impl.h" 
#include "numa-kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief NUMA RESHAPE kernel execution function
 * 
 * Executes a RESHAPE operation (metadata-only, returns immediately).
 * 
 * @param work_context Pointer to tensor being reshaped
 * @param params Compute parameters
 * @return GGML_STATUS_SUCCESS always
 */
enum ggml_status ggml_numa_kernel_reshape_execute(void * work_context, 
                                                   struct ggml_compute_params * params);

/**
 * @brief Query function for NUMA RESHAPE kernel strategy selection
 * 
 * @param tensor Target tensor for strategy selection
 * @return Single-thread strategy recommendation
 */
ggml_numa_execution_strategy_t ggml_numa_kernel_reshape_query(const struct ggml_tensor * tensor);

/**
 * @brief Work buffer calculation function for RESHAPE operations
 * 
 * @param tensor Target tensor
 * @param total_numa_nodes Number of NUMA nodes  
 * @param total_threads Total number of threads
 * @return 0 (no work buffer needed)
 */
size_t ggml_numa_kernel_reshape_work_buffer_calc(const struct ggml_tensor * tensor, 
                                                  int total_numa_nodes, 
                                                  int total_threads);

/**
 * @brief Register the RESHAPE kernel with the NUMA system
 * 
 * @return Registration information for the RESHAPE kernel
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_reshape_register(void);

#ifdef __cplusplus
}
#endif

#endif // NUMA_KERNEL_RESHAPE_H
