/**
 * @file noop.h
 * @brief NUMA NOOP Kernel - Performance Testing Operation
 *
 * This file implements a NUMA-aware NOOP (No Operation) kernel specifically designed
 * for performance testing and benchmarking the NUMA kernel dispatch system.
 * 
 * The NOOP kernel performs no actual computation but follows the full NUMA kernel
 * execution path, allowing measurement of the overhead introduced by the NUMA
 * coordinator, executor, and kernel dispatch system.
 *
 * Key Features:
 * - Zero computational work (immediate return)
 * - Full NUMA kernel registration and lookup
 * - Standard work function interface
 * - Minimal resource usage
 * - Performance measurement friendly
 *
 * Usage:
 * This kernel is intended for benchmarking purposes to compare the dispatch
 * overhead between the NUMA kernel system and the standard ggml-cpu fallback
 * system when paired with GGML_OP_NUMA_FALLBACK_NOOP.
 *
 * Performance Characteristics:
 * - Operation type: GGML_OP_NUMA_NOOP
 * - Work buffer size: 0 bytes
 * - Execution time: ~1-2 nanoseconds (function call overhead only)
 * - Memory access: Minimal (only function parameters)
 * - Thread safety: Fully thread-safe (no shared state)
 *
 * @see ggml_compute_forward_numa_fallback_noop() in ggml-cpu.c for comparison
 */

#pragma once

#include "numa-kernels.h"
#include "../ggml-numa-shared.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief NUMA NOOP kernel work function
 * 
 * This function performs no operation and returns immediately. It serves as a
 * minimal baseline for measuring the overhead of the NUMA kernel dispatch system.
 * 
 * @param work_context Tensor context (unused in NOOP)
 * @param params Compute parameters (unused in NOOP)
 * @return GGML_STATUS_SUCCESS always
 */
enum ggml_status ggml_numa_kernel_noop_execute(void * work_context, 
                                                struct ggml_compute_params * params);

/**
 * @brief Query function for NUMA NOOP kernel strategy selection
 * 
 * Returns optimal strategy for NOOP operations. Since NOOP requires no computation,
 * all strategies are equally efficient.
 * 
 * @param tensor Target tensor (used for strategy consistency)
 * @param total_elements Element count (for strategy selection patterns)
 * @return Kernel query result with strategy recommendation
 */
ggml_numa_kernel_query_result_t ggml_numa_kernel_noop_query(
    const struct ggml_tensor * tensor,
    size_t total_elements
);

/**
 * @brief Calculate work buffer size for NOOP operations
 * 
 * NOOP operations require no work buffer.
 * 
 * @param tensor Target tensor (unused)
 * @return 0 (no work buffer needed)
 */
size_t ggml_numa_kernel_noop_calculate_work_buffer_size(const struct ggml_tensor * tensor);

/**
 * @brief Register NUMA NOOP kernels with the kernel cache system
 * 
 * Registers the NOOP kernel for GGML_OP_NUMA_NOOP operations with appropriate
 * strategy thresholds and work function mappings.
 */
void ggml_numa_register_noop_kernels(void);

#ifdef __cplusplus
}
#endif
