/**
 * @file view.h
 * @brief NUMA VIEW Kernel - Tensor View Metadata Operation
 *
 * This file implements a NUMA-aware VIEW kernel for tensor view operations.
 * VIEW is a metadata-only operation that creates a new view of an existing tensor
 * with potentially different shape or offset without moving or modifying the underlying data.
 * 
 * The VIEW kernel performs no actual computation but follows the full NUMA kernel
 * execution path, similar to other view operations like RESHAPE and PERMUTE.
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
 * This kernel handles GGML_OP_VIEW operations which create tensor views with
 * potentially different offsets, shapes, or strides while sharing the same underlying memory.
 *
 * Performance Characteristics:
 * - Operation type: GGML_OP_VIEW
 * - Work buffer size: 0 bytes (no computation required)
 * - Execution time: ~1-2 nanoseconds (function call overhead only)
 * - Memory access: Minimal (only function parameters)
 * - Thread safety: Fully thread-safe (no shared state)
 *
 * Mathematical Properties:
 * - Creates a view into existing tensor data
 * - May change shape, offset, or strides
 * - No data movement or computation occurs
 * - Underlying tensor data is shared between views
 *
 * @see ggml_compute_forward_view() in ops.cpp for standard implementation
 */

#pragma once

#include "numa-kernels.h"
#include "../ggml-numa-shared.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief NUMA VIEW kernel work function
 * 
 * Implements a no-operation VIEW kernel that matches the behavior of
 * ggml_compute_forward_view(). VIEW operations only modify tensor
 * metadata during graph construction.
 * 
 * @param work_context Tensor context (validated but unused)
 * @param params Compute parameters (validated but unused)
 * @return GGML_STATUS_SUCCESS always
 */
enum ggml_status ggml_numa_kernel_view_execute(void * work_context, 
                                               struct ggml_compute_params * params);

/**
 * @brief Query optimal execution strategy for VIEW operations
 * 
 * Returns strategy recommendations for VIEW operations. Since VIEW
 * requires no computation, all strategies have equal efficiency.
 * 
 * @param tensor Target tensor for strategy selection
 * @return Kernel query result with strategy and efficiency metrics
 */
ggml_numa_kernel_query_result_t ggml_numa_kernel_view_query(const struct ggml_tensor * tensor);

/**
 * @brief Register VIEW kernel in the NUMA kernel cache system
 * 
 * Returns registration information for the VIEW kernel including strategy
 * thresholds, work functions, and aggregation policies suitable for view operations.
 * 
 * @return Registration information for the VIEW kernel
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_view_register(void);

#ifdef __cplusplus
}
#endif
