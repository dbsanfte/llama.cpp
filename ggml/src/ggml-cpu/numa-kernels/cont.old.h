/**
 * @file cont.h
 * @brief NUMA CONT Kernel Header - Tensor Contiguity Operation
 *
 * This header provides declarations for the NUMA-aware CONT kernel
 * implementation. CONT makes tensors contiguous by copying data into
 * a new memory layout with optimal strides.
 * 
 * CONT Operation:
 * - Copies source tensor data to create a contiguous layout
 * - Preserves tensor shape and element values
 * - Optimizes memory access patterns for subsequent operations
 * - Can involve type conversion and data reorganization
 * 
 * The implementation provides sophisticated NUMA-aware data copying
 * with multi-threading support and optimal memory bandwidth utilization.
 */

#ifndef GGML_NUMA_CONT_H
#define GGML_NUMA_CONT_H

#include "ggml-cpu.h"
#include "numa-kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Execute CONT operation with NUMA-aware optimization
 * 
 * This function implements a NUMA-aware CONT kernel that efficiently
 * copies tensor data to create contiguous memory layouts. The implementation
 * supports various data types and uses optimal copying strategies based
 * on tensor characteristics and NUMA topology.
 * 
 * @param work_context Tensor context pointer for CONT operation
 * @param params Compute parameters including threading information
 * @return GGML_STATUS_SUCCESS on completion, error status on failure
 */
enum ggml_status ggml_numa_kernel_cont_execute(void * work_context, 
                                                struct ggml_compute_params * params);

/**
 * @brief Query CONT kernel capabilities and strategy selection
 * 
 * This function evaluates tensor characteristics and selects the optimal
 * execution strategy for CONT operations. Strategy selection considers
 * data size, type compatibility, and memory layout requirements.
 * 
 * @param tensor Target tensor for CONT operation
 * @return Query result with strategy selection and efficiency metrics
 */
ggml_numa_kernel_query_result_t ggml_numa_kernel_cont_query(const struct ggml_tensor * tensor);

/**
 * @brief Register CONT kernel in NUMA kernel cache system
 * 
 * This function provides kernel registration information for the NUMA
 * kernel cache, enabling direct function pointer dispatch without switch
 * statement overhead.
 * 
 * @return Registration information structure with kernel metadata
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_cont_register(void);

#ifdef __cplusplus
}
#endif

#endif // GGML_NUMA_CONT_H
