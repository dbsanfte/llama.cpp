/**
 * @file div.h
 * @brief NUMA-aware element-wise division kernel interface
 * 
 * Header file for F32 element-wise division kernel using shared broadcasting macros.
 * Demonstrates 99% code reuse pattern with ADD kernel through macro framework.
 * 
 * @author David Sanftenberg
 */

#ifndef GGML_NUMA_KERNEL_DIV_H
#define GGML_NUMA_KERNEL_DIV_H

#include "ggml.h"
#include "ggml-numa-shared.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Execute F32 element-wise division using NUMA coordinator
 * 
 * Handles all broadcasting patterns using shared macro framework:
 * - Scalar ÷ Tensor broadcasting  
 * - Tensor ÷ Scalar broadcasting
 * - Same-shape element-wise division
 * - Complex multi-dimensional broadcasting
 * 
 * @param work_context Tensor context (struct ggml_tensor*)
 * @param params Compute parameters with NUMA threading info
 * @return GGML_STATUS_SUCCESS on completion, GGML_STATUS_FAILED on error
 */
enum ggml_status ggml_numa_kernel_div_execute(void * work_context, struct ggml_compute_params * params);

/**
 * @brief Query optimal execution strategy for DIV operation
 * 
 * Uses cache-based threshold lookup for strategy selection:
 * - < 1K elements: Single thread, single node
 * - < 256K elements: Multi-thread, single node  
 * - >= 256K elements: Data-parallel across nodes
 * 
 * @param tensor Input tensor for strategy analysis
 * @return Execution strategy recommendation
 */
ggml_numa_execution_strategy_t ggml_numa_kernel_div_query(const struct ggml_tensor * tensor);

/**
 * @brief Calculate work buffer requirements for DIV kernel
 * 
 * Element-wise division requires no additional work buffers.
 * 
 * @param tensor Input tensor for analysis
 * @param total_numa_nodes Number of NUMA nodes participating
 * @param total_threads Total thread count across all nodes
 * @return Work buffer size in bytes (always 0 for DIV)
 */
size_t ggml_numa_kernel_div_work_buffer_calc(const struct ggml_tensor * tensor, int total_numa_nodes, int total_threads);

/**
 * @brief Register DIV kernel with NUMA system
 * 
 * Provides registration information using shared patterns identical to ADD kernel.
 * Strategy thresholds and function pointers use same framework.
 * 
 * @return Complete kernel registration information
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_div_register(void);

#ifdef __cplusplus
}
#endif

#endif // GGML_NUMA_KERNEL_DIV_H
