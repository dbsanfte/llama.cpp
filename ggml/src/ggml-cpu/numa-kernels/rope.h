/**
 * @file rope.h
 * @brief NUMA ROPE (Rotary Position Embedding) Kernel Interface
 * 
 * NUMA-aware implementation of ROPE operations with support for:
 * - Standard ROPE (original and NEOX variants)
 * - Multi-modal ROPE (mrope)
 * - Vision ROPE 
 * - Forward and backward passes
 * - NUMA-optimized data-parallel execution
 */

#pragma once

#include "numa-kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// NUMA ROPE Kernel Registration
// ============================================================================

/**
 * Register ROPE kernel with NUMA strategy array and work functions
 * Returns registration info for the NUMA kernel registry system
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_rope_register(void);

/**
 * Query ROPE kernel for optimal execution strategy
 * Returns strategy recommendation and kernel info for given tensor
 */
ggml_numa_kernel_query_result_t ggml_numa_kernel_rope_query(const struct ggml_tensor * tensor);

// ============================================================================
// NUMA ROPE Kernel Work Functions
// ============================================================================

/**
 * NUMA ROPE kernel execution function
 * Supports all ROPE variants with NUMA-aware parallelization
 * 
 * @param work_context   Tensor containing ROPE operation parameters
 * @param params         Compute parameters (threading, NUMA context)
 * @return               GGML_STATUS_SUCCESS on success, error code on failure
 */
enum ggml_status ggml_numa_kernel_rope_execute(void * work_context, struct ggml_compute_params * params);

#ifdef __cplusplus
}
#endif
