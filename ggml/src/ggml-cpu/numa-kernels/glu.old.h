#ifndef GGML_NUMA_GLU_H
#define GGML_NUMA_GLU_H

#include "numa-kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Register GLU NUMA kernel with the kernel registry
 * 
 * Returns kernel registration information for GLU operations including:
 * - Strategy thresholds for single-node vs data-parallel execution
 * - Work function pointers for different execution strategies
 * - Support status and kernel identification
 * 
 * GLU operations (REGLU, SWIGLU, GEGLU, etc.) are binary element-wise operations
 * that combine two tensors using gated linear unit transformations.
 * 
 * @return ggml_numa_kernel_registration_info_t Registration information
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_glu_register(void);

/**
 * @brief Execute GLU operation using NUMA kernel
 * 
 * Performs GLU operation (REGLU, SWIGLU, GEGLU, etc.) on tensor data using
 * NUMA-aware execution. Supports multiple GLU variants through dispatch to
 * appropriate SIMD functions.
 * 
 * Operation: dst[i] = glu_variant(src0[i], src1[i])
 * Where glu_variant depends on the specific GLU operation type.
 * 
 * This kernel supports data-parallel execution across NUMA nodes
 * for optimal memory locality and cache performance.
 * 
 * @param work_context Pointer to tensor being processed
 * @param params Compute parameters including thread information
 * @return GGML_STATUS_SUCCESS on successful completion
 */
enum ggml_status ggml_numa_kernel_glu_execute(void * work_context, struct ggml_compute_params * params);

/**
 * @brief Query GLU kernel efficiency and characteristics
 * 
 * Provides kernel selection information for the NUMA executor including:
 * - Efficiency score (0.0-1.0) for different execution strategies
 * - Recommended strategy (single-node vs data-parallel)
 * - Buffer requirements per thread
 * 
 * @param tensor Tensor to be processed
 * @return ggml_numa_kernel_query_result_t Query results for strategy selection
 */
ggml_numa_kernel_query_result_t ggml_numa_kernel_glu_query(const struct ggml_tensor * tensor);

#ifdef __cplusplus
}
#endif

#endif // GGML_NUMA_GLU_H
