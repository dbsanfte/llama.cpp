/**
 * NUMA Kernel: Element-wise Addition (ADD)
 * 
 * Uses pre-allocated NUMA-local mirrored tensor data directly without migration.
 */

#pragma once

#include "../ggml-impl.h"
#include "numa-kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

// Kernel interface functions
bool ggml_numa_kernel_add_supports(const struct ggml_tensor * tensor);
ggml_numa_execution_strategy_t ggml_numa_kernel_add_get_strategy(const struct ggml_tensor * tensor);
size_t ggml_numa_kernel_add_get_work_buffer_size(const struct ggml_tensor * tensor);
float ggml_numa_kernel_add_get_efficiency_score(const struct ggml_tensor * tensor);
const char * ggml_numa_kernel_add_get_name(const struct ggml_tensor * tensor);

/**
 * Query ADD kernel for optimal strategy based on tensor characteristics
 * 
 * This function analyzes the tensor and returns the optimal execution strategy
 * using operation-specific thresholds rather than rigid complexity classes.
 * 
 * @param tensor The tensor to analyze
 * @return Query result with optimal strategy, or unsupported result if not applicable
 */
ggml_numa_kernel_query_result_t ggml_numa_kernel_add_query(const struct ggml_tensor * tensor);

// Main execution function
enum ggml_status ggml_numa_kernel_add_execute(void * work_context, 
                                              struct ggml_compute_params * params);

// Optimized execution functions
enum ggml_status ggml_numa_kernel_add_execute_low_overhead(void * work_context, 
                                                          struct ggml_compute_params * params);
enum ggml_status ggml_numa_kernel_add_execute_no_aggregation(void * work_context, 
                                                            struct ggml_compute_params * params);

// Debug and performance functions  
enum ggml_status ggml_numa_kernel_add_debug_data_locality(const struct ggml_tensor * tensor);

// Cache population function for registry (legacy compatibility)
void ggml_numa_kernel_add_populate_cache(void * cache_array);

#ifdef __cplusplus
}
#endif
