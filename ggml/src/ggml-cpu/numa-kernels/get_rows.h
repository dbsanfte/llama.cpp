/**
 * @file get_rows.h
 * @brief NUMA-aware GET_ROWS kernel for tensor row extraction operations
 * 
 * GET_ROWS Operation: Extracts specified rows from a source tensor based on index tensor
 * - Input: src0 (data tensor), src1 (index tensor containing row indices)
 * - Output: dst tensor with extracted rows in order specified by indices
 * - Characteristics: Memory bandwidth limited, row-wise parallelizable, indexing operation
 * 
 * NUMA Optimizations:
 * - Row-wise work distribution across NUMA nodes for data-parallel execution
 * - SIMD-optimized memory copying using ggml_vec_cpy_* functions
 * - Efficient indexing with bounds checking and memory layout optimization
 * - Multi-threading support for medium to large tensor operations
 * 
 * Strategy Selection:
 * - Small tensors (< 4K rows): Single-thread execution for minimal overhead
 * - Medium tensors (4K-128K rows): Multi-thread single-node for balanced performance  
 * - Large tensors (> 128K rows): Data-parallel across NUMA nodes for maximum throughput
 * 
 * Mathematical Properties:
 * - Exact element preservation during row extraction
 * - Index bounds validation for memory safety
 * - Support for all quantization types and F32/F16 data types
 * 
 * @author David Sanftenberg
 * @date 2025
 */

#pragma once

#include "numa-kernels.h"
#include "ggml-cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Execute NUMA-aware GET_ROWS operation
 * 
 * Performs row extraction from source tensor based on index tensor with NUMA optimizations.
 * Handles all quantization types and implements efficient row-wise work distribution.
 * 
 * @param work_context Pointer to the destination tensor for GET_ROWS operation
 * @param params Compute parameters including thread configuration
 * @return GGML_STATUS_SUCCESS on successful execution, error status otherwise
 * 
 * @note This function delegates to ggml_compute_forward_get_rows with NUMA-aware
 *       memory access patterns and optimal thread distribution.
 */
enum ggml_status ggml_numa_kernel_get_rows_execute(void * work_context, struct ggml_compute_params * params);

/**
 * @brief Query optimal NUMA strategy for GET_ROWS operation
 * 
 * Analyzes tensor dimensions, data types, and system characteristics to determine
 * the most efficient execution strategy for the GET_ROWS operation.
 * 
 * @param tensor Destination tensor for strategy analysis
 * @return Query result containing optimal strategy, efficiency score, and resource requirements
 * 
 * @note Strategy selection considers:
 *       - Number of rows to extract (from src1 tensor)
 *       - Row size and data type of source tensor
 *       - Memory bandwidth requirements for indexing operation
 *       - NUMA topology and available thread resources
 */
ggml_numa_kernel_query_result_t ggml_numa_kernel_get_rows_query(const struct ggml_tensor * tensor);

/**
 * @brief Register GET_ROWS kernel with NUMA system
 * 
 * Provides kernel registration information including operation type, strategy thresholds,
 * function pointers, and kernel metadata for the NUMA kernel registry.
 * 
 * @return Registration information structure for GET_ROWS kernel
 * 
 * @note Configures thresholds optimized for row extraction operations:
 *       - Single-thread threshold: 4K rows (minimal overhead for small operations)
 *       - Multi-thread threshold: 128K rows (balanced performance for medium operations)
 *       - Data-parallel: > 128K rows (maximum throughput for large operations)
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_get_rows_register(void);

#ifdef __cplusplus
}
#endif
