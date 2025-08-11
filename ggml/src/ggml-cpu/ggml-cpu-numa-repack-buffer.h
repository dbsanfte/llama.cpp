#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Hybrid CPU buffer type that combines NUMA-aware allocation with repack optimizations.
 * 
 * This buffer type provides:
 * - NUMA-aware memory allocation on appropriate nodes
 * - Automatic repacking of quantized weights for optimal compute performance
 * - Optional replication of repacked data across NUMA nodes for locality
 * 
 * For tensors that support repacking (Q4_0, Q4_K, IQ4_NL):
 * - Data is automatically repacked to optimized layouts during set_tensor
 * - Replicated copies store the repacked data for consistent compute benefits
 * 
 * For tensors that don't support repacking:
 * - Falls back to standard NUMA allocation and replication
 * 
 * @return Buffer type for hybrid NUMA+repack allocation, or NULL if not available
 */
GGML_API ggml_backend_buffer_type_t ggml_backend_cpu_numa_repack_buffer_type(void);

#ifdef __cplusplus
}
#endif
