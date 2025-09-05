#pragma once

#include "../ggml-impl.h"
#include "numa-kernels.h"

/**
 * @file cpy.h 
 * @brief NUMA-aware CPY (copy/duplicate) operation kernel
 * 
 * This kernel implements NUMA-optimized tensor copying with proper type conversion
 * and stride handling. The CPY operation copies data from source tensor to destination
 * tensor, potentially with type conversion and different memory layouts.
 * 
 * Key Features:
 * - Type conversion support (F32, F16, quantized types)
 * - Stride-aware copying for non-contiguous tensors
 * - NUMA-aware memory access patterns
 * - Multi-threading with data-parallel execution
 * 
 * Performance Characteristics:
 * - Memory bandwidth limited operation
 * - Benefits from NUMA locality for large tensors
 * - Efficient SIMD utilization where applicable
 * 
 * @note This function delegates to optimized ggml_compute_forward_dup implementation
 *       while providing NUMA-aware memory allocation and thread distribution.
 */

/**
 * @brief Execute CPY operation with NUMA-aware optimizations
 * 
 * Performs tensor copying operation with proper type conversion and stride handling.
 * Uses NUMA-aware memory patterns and multi-threading for optimal performance.
 * 
 * @param work_context Pointer to tensor to be processed
 * @param params Compute parameters with thread information
 * @return GGML_STATUS_SUCCESS on success, error code otherwise
 * 
 * @note This kernel handles all type combinations supported by ggml_compute_forward_dup:
 *       - Same-type copying (F32→F32, F16→F16, Q8_0→Q8_0, etc.)
 *       - Type conversion (F16→F32, Q8_0→F32, etc.)
 *       - Contiguous and non-contiguous tensor layouts
 *       - Complex stride patterns and dimension permutations
 */
enum ggml_status ggml_numa_kernel_cpy_execute(void * work_context, struct ggml_compute_params * params);

/**
 * @brief Query CPY operation characteristics for strategy selection
 * 
 * Analyzes tensor properties to determine optimal execution strategy based on:
 * - Tensor size and element count
 * - Source and destination types
 * - Memory layout (contiguous vs. strided)
 * - NUMA node availability
 * 
 * @param tensor Tensor to analyze
 * @return Query result with strategy recommendation and efficiency score
 */
ggml_numa_kernel_query_result_t ggml_numa_kernel_cpy_query(const struct ggml_tensor * tensor);

/**
 * @brief Register CPY kernel with NUMA system
 * 
 * Provides kernel registration information including:
 * - Operation type (GGML_OP_CPY)
 * - Strategy thresholds for different execution modes
 * - Function pointers for work execution
 * - Support for all relevant data types
 * 
 * @return Registration info structure
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_cpy_register(void);