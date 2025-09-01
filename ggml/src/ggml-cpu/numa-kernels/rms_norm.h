/**
 * NUMA Kernel: Root Mean Square Normalization (RMS_NORM)
 * 
 * Row-wise normalization with NUMA-aware distribution across nodes.
 */

#pragma once

#include "../ggml-impl.h"
#include "numa-kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

// Core kernel execution function
enum ggml_status ggml_numa_kernel_rms_norm_execute(void * work_context, struct ggml_compute_params * params);

// Query function for strategy selection
ggml_numa_kernel_query_result_t ggml_numa_kernel_rms_norm_query(const struct ggml_tensor * tensor);

// Kernel registration function
void ggml_numa_register_rms_norm_kernels(void);

// Register function for NUMA_REGISTER_KERNEL macro compatibility
ggml_numa_kernel_registration_info_t ggml_numa_kernel_rms_norm_register(void);

#ifdef __cplusplus
}
#endif
