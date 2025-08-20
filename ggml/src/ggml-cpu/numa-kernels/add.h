/*
 * NUMA Kernel: Element-wise Addition (ADD)
 * 
 * This kernel implements NUMA-aware element-wise tensor addition.
 * Simple operation with high parallelization potential.
 */

#pragma once

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"  // For complete ggml_cplan definition
#include "../ggml-numa-coordinator.h"  // For execution strategy and work function types

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration
struct ggml_cplan;

// New architecture function signatures for kernel registry
bool ggml_numa_kernel_add_supports(const struct ggml_tensor * tensor);
ggml_numa_execution_strategy_t ggml_numa_kernel_add_get_strategy(const struct ggml_tensor * tensor);
size_t ggml_numa_kernel_add_get_buffer_size(const struct ggml_tensor * tensor);
ggml_numa_work_function_t ggml_numa_kernel_add_get_work_function(const struct ggml_tensor * tensor);
float ggml_numa_kernel_add_get_efficiency(const struct ggml_tensor * tensor);

// Work function that coordinator will execute
enum ggml_status ggml_numa_kernel_add_work_function(void * work_context, struct ggml_compute_params * params);

// Legacy signatures (kept for compatibility during transition)
enum ggml_status ggml_numa_kernel_add_execute(struct ggml_tensor * tensor, struct ggml_cplan * cplan);
float ggml_numa_kernel_add_get_efficiency_legacy(const struct ggml_tensor * tensor, size_t tensor_size);

#ifdef __cplusplus
}
#endif
