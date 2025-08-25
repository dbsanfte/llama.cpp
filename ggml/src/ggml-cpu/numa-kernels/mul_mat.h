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
 * @param work_context  Tensor to process (cast from void*)
 * @param params        Threadpool parameters (thread ID, thread count)
 * @return              GGML_STATUS_SUCCESS on success, GGML_STATUS_FAILED on error
 */
enum ggml_status ggml_numa_kernel_mul_mat_execute(void * work_context, 
                                                  struct ggml_compute_params * params);

/**
 * Populate cache entries for MUL_MAT operation across all complexity levels
 * 
 * @param cache_entries Array of cache entries to populate (COMPLEXITY_COUNT elements)
 */
void ggml_numa_kernel_mul_mat_populate_cache(ggml_numa_cache_entry_t cache_entries[COMPLEXITY_COUNT]);

#ifdef __cplusplus
}
#endif

#endif // GGML_NUMA_KERNEL_MUL_MAT_H
