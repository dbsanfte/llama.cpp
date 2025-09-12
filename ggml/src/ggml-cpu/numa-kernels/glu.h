/**
 * @file glu.h
 * @brief NUMA-aware GLU (Gated Linear Unit) operation kernel header
 * @author David Sanftenberg
 * 
 * This header defines the interface for NUMA-optimized GLU operations,
 * including REGLU, SWIGLU, GEGLU, GEGLU_ERF, and GEGLU_QUICK variants.
 * 
 * GLU operations are element-wise binary operations that take two tensors
 * and apply gated activation functions for enhanced neural network performance.
 */

#ifndef GGML_NUMA_KERNEL_GLU_H
#define GGML_NUMA_KERNEL_GLU_H

#include "../ggml-numa-executor.h"
#include "numa-kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Execute GLU operation using NUMA kernel
 * 
 * This kernel implements all GLU variants (REGLU, SWIGLU, GEGLU, etc.) using
 * NUMA-aware execution strategies for optimal performance on multi-socket systems.
 * 
 * The kernel supports:
 * - All GLU operation types: REGLU, SWIGLU, GEGLU, GEGLU_ERF, GEGLU_QUICK
 * - F32 and F16 data types
 * - Three execution strategies: single-thread, multi-thread, data-parallel
 * - Automatic strategy selection based on tensor size
 * - SIMD optimization using ggml_vec_* functions
 * 
 * @param work_context Tensor to process (struct ggml_tensor*)
 * @param params Compute parameters including thread info and work buffer
 * @return GGML_STATUS_SUCCESS on success, error status on failure
 */
enum ggml_status ggml_numa_kernel_glu_execute(void * work_context, struct ggml_compute_params * params);

/**
 * @brief Query optimal execution strategy for GLU operation
 * 
 * Determines the best NUMA execution strategy based on tensor size and
 * system characteristics. Uses threshold-based selection for optimal performance.
 * 
 * @param tensor Input tensor to analyze
 * @return Recommended execution strategy
 */
ggml_numa_execution_strategy_t ggml_numa_kernel_glu_query(const struct ggml_tensor * tensor);

/**
 * @brief Register GLU kernel with NUMA system
 * @brief Register GLU kernel with NUMA system
 * 
 * Registers the GLU kernel with appropriate strategy thresholds and function pointers.
 * Called during NUMA system initialization.
 * 
 * @return Registration information structure
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_glu_register(void);

#ifdef __cplusplus
}
#endif

#endif // GGML_NUMA_KERNEL_GLU_H
