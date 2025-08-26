/**
 * @file mul_mat.h
 * @brief NUMA Kernel Header: Matrix Multiplication (MUL_MAT)
 * 
 * Public interface for NUMA-aware matrix multiplication kernel.
 */

#ifndef GGML_NUMA_KERNEL_MUL_MAT_H
#define GGML_NUMA_KERNEL_MUL_MAT_H

#include "ggml-cpu.h"
#include "numa-kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Execute NUMA-aware matrix multiplication kernel
 * 
 * @param work_context The tensor to process (cast from void*)
 * @param params       Thread parameters with thread ID and count
 * @return             GGML_STATUS_SUCCESS on success
 */
enum ggml_status ggml_numa_kernel_mul_mat_execute(void * work_context, struct ggml_compute_params * params);

/**
 * Custom F16 dot product implementation for testing and optimization
 * 
 * @param n      Vector length
 * @param s      Output scalar result
 * @param s_off  Output offset (should be 0)
 * @param x      First vector (F16 data)  
 * @param x_off  First vector offset
 * @param y      Second vector (F16 data)
 * @param y_off  Second vector offset
 * @param nrc    Number of rows per call (should be 1)
 */
void ggml_numa_vec_dot_f16_custom(int n, float * restrict s, size_t s_off, 
                                 const void * restrict x, size_t x_off,
                                 const void * restrict y, size_t y_off, int nrc);

/**
 * Query MUL_MAT kernel for optimal strategy based on tensor characteristics
 * 
 * This function analyzes the tensor and returns the optimal execution strategy
 * using operation-specific thresholds rather than rigid complexity classes.
 * 
 * @param tensor The tensor to analyze
 * @return Query result with optimal strategy, or unsupported result if not applicable
 */
ggml_numa_kernel_query_result_t ggml_numa_kernel_mul_mat_query(const struct ggml_tensor * tensor);

/**
 * Populate cache entries for MUL_MAT operation across all complexity levels
 * (Legacy compatibility function for backward compatibility during transition)
 * 
 * @param cache_entries Array of cache entries to populate (COMPLEXITY_COUNT elements)
 */
void ggml_numa_kernel_mul_mat_populate_cache(ggml_numa_cache_entry_t cache_entries[COMPLEXITY_COUNT]);

#ifdef __cplusplus
}
#endif

#endif // GGML_NUMA_KERNEL_MUL_MAT_H
