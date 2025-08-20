/*
 * NUMA Kernel: Copy Operation (CPY)
 */

#pragma once

#include "ggml.h"
#include "ggml-impl.h"

// Forward declarations
struct ggml_cplan;

#ifdef __cplusplus
extern "C" {
#endif

bool ggml_numa_kernel_cpy_supports(const struct ggml_tensor * tensor);
enum ggml_status ggml_numa_kernel_cpy_execute(struct ggml_tensor * tensor, struct ggml_cplan * cplan);
float ggml_numa_kernel_cpy_get_efficiency(const struct ggml_tensor * tensor, size_t tensor_size);

#ifdef __cplusplus
}
#endif
