/**
 * NUMA Kernel: Element-wise Addition (ADD) - Direct Mirrored Data Version
 * 
 * Uses pre-allocated NUMA-local mirrored tensor data directly without migration.
 */

#pragma once

#include "../ggml-impl.h"
#include "numa-kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

// Kernel interface functions
bool ggml_numa_kernel_add_direct_supports(const struct ggml_tensor * tensor);
ggml_numa_execution_strategy_t ggml_numa_kernel_add_direct_get_strategy(const struct ggml_tensor * tensor);
size_t ggml_numa_kernel_add_direct_get_work_buffer_size(const struct ggml_tensor * tensor);
float ggml_numa_kernel_add_direct_get_efficiency_score(const struct ggml_tensor * tensor);
const char * ggml_numa_kernel_add_direct_get_name(const struct ggml_tensor * tensor);

// Main execution function
enum ggml_status ggml_numa_kernel_add_direct_execute(void * work_context, 
                                                     struct ggml_compute_params * params);

// Debug and performance functions  
enum ggml_status ggml_numa_kernel_add_direct_debug_data_locality(const struct ggml_tensor * tensor);

// Cache population function for registry
void ggml_numa_kernel_add_direct_populate_cache(void * cache_array);

#ifdef __cplusplus
}
#endif
