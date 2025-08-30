/**
 * @file mul.h
 * @brief NUMA Kernel: Element-wise Multiplication (MUL)
 * 
 * NUMA-aware element-wise multiplication operations with SIMD optimization
 * and data-parallel execution support.
 * 
 * Template Pattern: Based on proven ADD kernel implementation
 * 
 * OPERATION CHARACTERISTICS:
 * - Element-wise multiplication: dst[i] = src0[i] * src1[i]
 * - Perfect data-parallel scalability
 * - High SIMD optimization potential
 * - Broadcasting support for scalar operations
 * 
 * PERFORMANCE EXPECTATIONS:
 * - Similar to ADD kernel performance
 * - 810 operations per 32t/32t benchmark run (7.6% frequency)
 * - High impact target for NUMA acceleration
 */

#pragma once

#include "../ggml-cpu-impl.h"
#include "../ggml-impl.h"
#include "numa-kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Ultra-fast MUL kernel - minimal validation, maximum performance
 * 
 * Template Pattern: High-performance NUMA kernel implementation based on ADD
 * 
 * EXECUTION FLOW:
 * 1. Fast validation (assume coordinator pre-validated)
 * 2. Extract tensor data using NUMA-local tensor_data()
 * 3. Read thread-local NUMA context from coordinator
 * 4. Calculate data slice for this thread/node combination
 * 5. Execute SIMD operations on assigned slice
 * 6. Handle broadcasting and edge cases efficiently
 * 
 * THREAD SAFETY: Thread-safe via data slicing (no shared state)
 * NUMA AWARENESS: Accesses only NUMA-local memory via tensor_data()
 * PERFORMANCE: Optimized for minimal overhead, maximum SIMD utilization
 * 
 * @param work_context  Tensor to process (cast from void*)
 * @param params        Threadpool parameters (thread ID, thread count)
 * @return              GGML_STATUS_SUCCESS on success, GGML_STATUS_FAILED on error
 */
enum ggml_status ggml_numa_kernel_mul_execute_optimized(void * work_context, 
                                                       struct ggml_compute_params * params);

/**
 * Low-overhead MUL kernel for smaller tensors
 * 
 * Template Pattern: Lightweight execution with type-aware optimizations
 * 
 * Designed for single-node execution with minimal coordination overhead.
 * Supports multiple data types and broadcasting patterns.
 * 
 * @param work_context  Tensor to process (cast from void*)
 * @param params        Threadpool parameters (thread ID, thread count)
 * @return              GGML_STATUS_SUCCESS on success, GGML_STATUS_FAILED on error
 */
enum ggml_status ggml_numa_kernel_mul_execute_low_overhead(void * work_context,
                                                          struct ggml_compute_params * params);

/**
 * No-aggregation MUL kernel for data-parallel execution
 * 
 * Template Pattern: Shared memory approach for large tensors
 * 
 * Uses shared result tensor memory to eliminate aggregation overhead.
 * Optimized for large tensors with data-parallel NUMA execution.
 * 
 * @param work_context  Tensor to process (cast from void*)
 * @param params        Threadpool parameters (thread ID, thread count)
 * @return              GGML_STATUS_SUCCESS on success, GGML_STATUS_FAILED on error
 */
enum ggml_status ggml_numa_kernel_mul_execute_no_aggregation(void * work_context,
                                                            struct ggml_compute_params * params);

/**
 * Register MUL kernel with the NUMA kernel registry
 * 
 * Template Pattern: Registry integration following ADD kernel patterns
 * 
 * Provides strategy thresholds and function pointers for O(1) lookups.
 * Includes complexity-based strategy selection for optimal performance.
 * 
 * @return Registration information structure for the registry
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_mul_register(void);

/**
 * Query function for MUL operations
 * 
 * Template Pattern: Strategy selection based on ADD kernel patterns
 * 
 * Provides strategy selection based on tensor size and characteristics.
 * Returns optimal execution strategy and function pointer for the operation.
 * 
 * @param tensor  The tensor to query for MUL operation support
 * @return        Query result with strategy and function pointer
 */
ggml_numa_kernel_query_result_t ggml_numa_kernel_mul_query(const struct ggml_tensor * tensor);

#ifdef __cplusplus
}
#endif
