/**
 * @file reshape.h
 * @brief NUMA RESHAPE Kernel - Tensor Shape Metadata Operation
 *
 * This file implements a NUMA-aware RESHAPE kernel for tensor shape transformation
 * operations. RESHAPE is a metadata-only operation that changes tensor dimensions
 * without moving or modifying the underlying data.
 * 
 * The RESHAPE kernel performs no actual computation but follows the full NUMA kernel
 * execution path, similar to other view operations like PERMUTE and TRANSPOSE.
 * It's essentially a no-operation in terms of data processing.
 *
 * Key Features:
 * - Zero computational work (metadata-only operation)
 * - Full NUMA kernel registration and lookup
 * - Standard work function interface
 * - Minimal resource usage
 * - Supports all tensor types and dimensions
 *
 * Usage:
 * This kernel handles GGML_OP_RESHAPE operations which change tensor shape
 * while preserving the total number of elements and underlying memory layout.
 *
 * Performance Characteristics:
 * - Operation type: GGML_OP_RESHAPE
 * - Work buffer size: 0 bytes (no computation required)
 * - Execution time: ~1-2 nanoseconds (function call overhead only)
 * - Memory access: Minimal (only function parameters)
 * - Thread safety: Fully thread-safe (no shared state)
 *
 * Mathematical Properties:
 * - Input/Output element count must be identical
 * - Input tensor must be contiguous
 * - Only tensor shape metadata is modified
 * - No data movement or computation occurs
 *
 * @see ggml_compute_forward_reshape() in ops.cpp for standard implementation
 */

#pragma once

#include "numa-kernels.h"
#include "../ggml-numa-shared.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief NUMA RESHAPE kernel work function
 * 
 * This function performs no operation and returns immediately, similar to the
 * standard ggml_compute_forward_reshape implementation. RESHAPE operations are
 * metadata-only and require no data processing.
 * 
 * @param work_context Tensor context (validated but unused in RESHAPE)
 * @param params Compute parameters (validated but unused in RESHAPE)
 * @return GGML_STATUS_SUCCESS always
 */
enum ggml_status ggml_numa_kernel_reshape_execute(void * work_context, 
                                                   struct ggml_compute_params * params);

/**
 * @brief Query function for NUMA RESHAPE kernel strategy selection
 * 
 * Analyzes the tensor and returns strategy recommendations for RESHAPE operations.
 * Since RESHAPE requires no computation, all strategies have equal efficiency
 * and the selection is based on consistency with other kernel patterns.
 * 
 * Strategy Selection Logic:
 * - ≤1024 elements: Single-node, single-thread (minimal overhead)
 * - 1025-262144 elements: Single-node, multi-thread
 * - >262144 elements: Multi-node, data-parallel (large tensors)
 * 
 * @param tensor Target tensor for strategy selection
 * @return ggml_numa_kernel_query_result_t Query results for strategy selection
 */
ggml_numa_kernel_query_result_t ggml_numa_kernel_reshape_query(const struct ggml_tensor * tensor);

/**
 * @brief Register NUMA RESHAPE kernel in the kernel cache system
 * 
 * Returns registration information for the RESHAPE kernel including strategy
 * thresholds, work functions, and aggregation policies.
 * 
 * Registration Details:
 * - Operation: GGML_OP_RESHAPE
 * - Strategies: Single-single, single-multi, data-parallel
 * - Work function: ggml_numa_kernel_reshape_execute
 * - Aggregation: None required (no-op)
 * - Buffer size: 0 bytes
 * 
 * @return ggml_numa_kernel_registration_info_t Registration information
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_reshape_register(void);

#ifdef __cplusplus
}
#endif
