/**
 * @file transpose.h
 * @brief NUMA TRANSPOSE Kernel Header - Tensor Dimension Swapping Operation
 *
 * This header provides declarations for the NUMA-aware TRANSPOSE kernel
 * implementation. TRANSPOSE is a view operation that swaps tensor dimensions
 * without performing actual data movement.
 * 
 * TRANSPOSE swaps the first two dimensions of a tensor:
 * - Input:  [ne0, ne1, ne2, ne3]
 * - Output: [ne1, ne0, ne2, ne3]
 * - Strides are also swapped: nb0 ↔ nb1
 * 
 * Like other view operations (RESHAPE, PERMUTE), TRANSPOSE is implemented
 * as a minimal no-op kernel that provides NUMA infrastructure compliance
 * while maintaining optimal performance characteristics.
 */

#ifndef GGML_NUMA_TRANSPOSE_H
#define GGML_NUMA_TRANSPOSE_H

#include "ggml-cpu.h"
#include "numa-kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Execute TRANSPOSE operation with NUMA-aware optimization
 * 
 * This function implements a no-operation TRANSPOSE kernel that matches the
 * behavior of ggml_compute_forward_transpose(). Since TRANSPOSE is a view
 * operation that only modifies tensor metadata, no actual computation is
 * performed during execution.
 * 
 * @param work_context Tensor context pointer (validated but unused)
 * @param params Compute parameters (validated but unused)
 * @return GGML_STATUS_SUCCESS always
 */
enum ggml_status ggml_numa_kernel_transpose_execute(void * work_context, 
                                                     struct ggml_compute_params * params);

/**
 * @brief Query TRANSPOSE kernel capabilities and strategy selection
 * 
 * This function evaluates tensor characteristics and selects the optimal
 * execution strategy for TRANSPOSE operations. Since TRANSPOSE is a view
 * operation, it always uses single-node single-thread execution.
 * 
 * @param tensor Target tensor for TRANSPOSE operation
 * @return Query result with strategy selection and efficiency metrics
 */
ggml_numa_kernel_query_result_t ggml_numa_kernel_transpose_query(const struct ggml_tensor * tensor);

/**
 * @brief Register TRANSPOSE kernel in NUMA kernel cache system
 * 
 * This function provides kernel registration information for the NUMA
 * kernel cache, enabling direct function pointer dispatch without switch
 * statement overhead.
 * 
 * @return Registration information structure with kernel metadata
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_transpose_register(void);

#ifdef __cplusplus
}
#endif

#endif // GGML_NUMA_TRANSPOSE_H
