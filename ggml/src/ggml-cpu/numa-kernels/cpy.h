#pragma once

#include "../ggml-impl.h"
#include "numa-kernels.h"

/**
 * @file cpy.h 
 * @brief NUMA-aware CPY (copy/duplicate) operation kernels
 * 
 * The CPY operation performs memory copying from source to destination tensor,
 * with optional type conversion and reshape handling. This is a fundamental
 * memory bandwidth operation that benefits significantly from NUMA locality.
 * 
 * Operation Coverage:
 * - 792 operations per 32t/32t benchmark (7.4% frequency)
 * - High-priority target for NUMA acceleration
 * - Memory bandwidth intensive operation
 * 
 * NUMA Optimization Strategy:
 * - Data-parallel execution across NUMA nodes for large tensors
 * - Memory bandwidth optimization through NUMA-local copying  
 * - Shared memory approach to eliminate aggregation overhead
 * - SIMD acceleration where applicable (contiguous cases)
 * 
 * Mathematical Operation:
 * - dst[i] = src[i] (direct copy)
 * - Handles type conversion: dst[i] = convert(src[i])
 * - Supports reshape: dst[new_layout] = src[old_layout]
 * 
 * Template Pattern: Based on proven ADD/MUL kernel designs
 * - Ultra-fast optimized kernel for data-parallel execution
 * - Low-overhead kernel for smaller tensors
 * - No-aggregation shared memory optimization
 */

/**
 * @brief Ultra-fast optimized CPY kernel for data-parallel execution
 * 
 * Designed for large tensors (>256K elements) using data-parallel strategy
 * across multiple NUMA nodes with multi-threading on each node.
 * 
 * Features:
 * - NUMA-aware data slicing for optimal memory bandwidth
 * - Bulk memory copy operations (memcpy) for contiguous tensors
 * - Element-wise copy with SIMD where applicable
 * - Shared memory optimization to eliminate aggregation
 * 
 * @param work_context Tensor context for the CPY operation
 * @param params Compute parameters including thread information
 * @return GGML_STATUS_SUCCESS on successful execution
 */
enum ggml_status ggml_numa_kernel_cpy_optimized_execute(void * work_context, struct ggml_compute_params * params);

/**
 * @brief Low-overhead CPY kernel for smaller tensors
 * 
 * Optimized for tensors with 1K-256K elements using single-node multi-threading
 * or single-node single-threading strategies.
 * 
 * Features:
 * - Minimized NUMA coordination overhead
 * - Direct memory operations for optimal cache utilization
 * - Efficient handling of non-contiguous tensors
 * - Thread-safe execution without shared state
 * 
 * @param work_context Tensor context for the CPY operation  
 * @param params Compute parameters including thread information
 * @return GGML_STATUS_SUCCESS on successful execution
 */
enum ggml_status ggml_numa_kernel_cpy_low_overhead_execute(void * work_context, struct ggml_compute_params * params);

/**
 * @brief No-aggregation shared memory CPY kernel
 * 
 * Specialized kernel that writes directly to shared result tensor memory,
 * eliminating aggregation overhead for data-parallel operations.
 * 
 * Features:
 * - Direct writes to final tensor memory locations
 * - Zero-copy architecture with NUMA memory locality
 * - Optimal for GB-scale tensors (1GB-16GB)
 * - 62% performance improvement over aggregated approaches
 * 
 * @param work_context Tensor context for the CPY operation
 * @param params Compute parameters including thread information  
 * @return GGML_STATUS_SUCCESS on successful execution
 */
enum ggml_status ggml_numa_kernel_cpy_no_aggregation_execute(void * work_context, struct ggml_compute_params * params);

/**
 * @brief Initialize CPY kernel registry entries
 * 
 * Populates the NUMA kernel cache with pre-computed CPY strategies
 * for all complexity classes. Called during NUMA system initialization.
 * 
 * Cache Entries:
 * - Efficiency scores based on memory bandwidth characteristics
 * - Thread count recommendations for optimal performance
 * - Aggregation policies (NONE for element-wise operations)
 * - Work buffer requirements (0 for shared memory approach)
 */
void ggml_numa_kernel_cpy_init_cache_entries(void);

/**
 * @brief Query CPY kernel availability and strategy
 * 
 * Checks if NUMA CPY acceleration is available for given tensor properties
 * and returns optimal execution strategy.
 * 
 * Query Process:
 * 1. Calculate tensor complexity class
 * 2. Lookup pre-computed strategy in O(1) cache
 * 3. Validate tensor properties (contiguity, type support)
 * 4. Return strategy or fallback recommendation
 * 
 * @param tensor Target tensor for CPY operation
 * @return Query result with strategy and kernel information
 */
ggml_numa_kernel_query_result_t ggml_numa_kernel_cpy_query(const struct ggml_tensor * tensor);

/**
 * @brief Register CPY kernel with NUMA system
 * 
 * Provides registration information for integration into the O(1) hash table system.
 * Called during NUMA system initialization to register CPY kernel strategies,
 * thresholds, and function pointers.
 * 
 * Registration Details:
 * - Operation type: GGML_OP_CPY
 * - Thresholds: 1K/256K elements for strategy selection
 * - Function pointers: Low-overhead and no-aggregation execution paths
 * - Aggregation: None required (direct memory operations)
 * 
 * @return Registration information structure with strategies and function pointers
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_cpy_register(void);
