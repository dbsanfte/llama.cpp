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

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration
struct ggml_cplan;

bool ggml_numa_kernel_add_supports(const struct ggml_tensor * tensor);
enum ggml_status ggml_numa_kernel_add_execute(struct ggml_tensor * tensor, struct ggml_cplan * cplan);
float ggml_numa_kernel_add_get_efficiency(const struct ggml_tensor * tensor, size_t tensor_size);

#ifdef __cplusplus
}
#endif
