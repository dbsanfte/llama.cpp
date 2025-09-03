/**
 * @file add.h
 * @brief NUMA Kernel: Element-wise Addition (ADD)
 * @author David Sanftenberg
 * 
 * NUMA-aware element-wise addition operations with comprehensive type support,
 * broadcasting, and SIMD optimization.
 * 
 * OPERATION CHARACTERISTICS:
 * - Element-wise addition: dst[i] = src0[i] + src1[i]
 * - Full broadcasting support matching reference implementation
 * - Comprehensive quantization type coverage (F32, F16, BF16, Q4_0, Q5_0, Q8_0, etc.)
 * - Perfect data-parallel scalability for same-shape operations
 * - Complex broadcasting logic for mismatched tensor shapes
 * - High SIMD optimization potential with ggml_vec_add_f32()
 * 
 * IMPLEMENTATION STRATEGY:
 * - Based on proven NUMA kernel patterns from MUL kernel
 * - Ultra-fast optimized path for large same-shape tensors
 * - Low-overhead path for smaller tensors
 * - Comprehensive broadcasting support following reference binary-ops.cpp
 * - Type-aware dispatch for F32/F16/BF16 and quantized types
 * - Shared memory optimization for no-aggregation execution
 * 
 * BROADCASTING SUPPORT:
 * - Full compatibility with reference implementation
 * - Contiguous and non-contiguous src1 broadcasting
 * - Complex indexing for multi-dimensional broadcast scenarios
 * - Regression testing for previously broken broadcasting cases
 */

#pragma once

#include "../ggml-cpu-impl.h"
#include "../ggml-impl.h" 
#include "numa-kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Main ADD kernel execution function
 * 
 * Handles element-wise addition with full broadcasting support and quantization
 * type coverage. Uses NUMA-aware data slicing for optimal performance.
 * 
 * EXECUTION FLOW:
 * 1. Validate tensor inputs and check for broadcasting requirements
 * 2. Extract tensor data using NUMA-local tensor_data()
 * 3. Read thread-local NUMA context from coordinator 
 * 4. Dispatch to appropriate type-specific implementation
 * 5. Handle broadcasting logic following reference implementation
 * 6. Execute SIMD operations on assigned NUMA slice
 * 
 * THREAD SAFETY: Thread-safe via data slicing (no shared state)
 * NUMA AWARENESS: Accesses only NUMA-local memory via tensor_data()
 * PERFORMANCE: Optimized for minimal overhead, maximum SIMD utilization
 * 
 * @param work_context  Tensor to process (cast from void*)
 * @param params        Threadpool parameters (thread ID, thread count)
 * @return              GGML_STATUS_SUCCESS on success, GGML_STATUS_FAILED on error
 */
enum ggml_status ggml_numa_kernel_add_execute(void * work_context, struct ggml_compute_params * params);

/**
 * Strategy query function for ADD operations
 * 
 * Determines optimal execution strategy based on tensor characteristics:
 * - Single-node single-thread for tiny tensors (< 128 elements)
 * - Single-node multi-thread for small tensors (< 1024 elements)  
 * - Data-parallel across NUMA nodes for larger tensors
 * 
 * @param tensor  The destination tensor to analyze
 * @return        Query result with selected strategy and efficiency estimate
 */
ggml_numa_kernel_query_result_t ggml_numa_kernel_add_query(const struct ggml_tensor * tensor);

/**
 * Kernel registration function 
 * 
 * Returns registration info for the ADD kernel including strategy thresholds,
 * function pointers, and metadata for the NUMA registry system.
 * 
 * @return  Registration info structure for ADD kernel
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_add_register(void);

#ifdef __cplusplus
}
#endif
