/**
 * @file ggml-numa-executor.h
 * @brief NUMA Executor - Strategy Engine and Work Orchestration
 * 
 * This component handles execution strategy selection and work submission,
 * serving as the central orchestration layer for NUMA-aware computation.
 * It replaces the old dispatcher logic with a cleaner, more maintainable
 * architecture that supports dynamic strategy selection and optimal
 * resource utilization.
 * 
 * Key responsibilities:
 * - Tensor operation analysis and strategy selection
 * - Kernel registry integration for O(1) lookups
 * - Work orchestration and coordinator management
 * - Performance monitoring and efficiency tracking
 * - Fallback execution for unsupported operations
 * 
 * @author David Sanftenberg
 * @date 2025
 */

#pragma once

#include "ggml.h"
#include "../ggml-impl.h"
#include "ggml-cpu.h"  // For complete ggml_cplan definition
#include "ggml-cpu-impl.h"  // For ggml_compute_params definition
#include "ggml-numa-shared.h"  // For ggml_numa_execution_strategy_t

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Forward declaration for NUMA coordinator manager */
struct ggml_numa_coordinator_manager;
/** @brief Forward declaration for compute plan */
struct ggml_cplan;

/**
 * @brief Initialize the NUMA executor and kernel registry
 * 
 * Sets up the entire NUMA execution infrastructure including:
 * - Kernel registry initialization with all available operations
 * - Strategy selection algorithms and efficiency scoring
 * - Coordinator integration and resource management
 * - Performance monitoring initialization
 * 
 * This must be called before using any executor functions.
 * 
 * @return GGML_STATUS_SUCCESS on success, error code on failure
 */
enum ggml_status ggml_numa_executor_init(void);

/**
 * @brief Cleanup the NUMA executor and kernel registry
 * 
 * Performs comprehensive cleanup of all executor resources:
 * - Releases kernel registry allocations
 * - Shuts down coordinator connections
 * - Frees performance monitoring data
 * - Resets all internal state
 */
void ggml_numa_executor_cleanup(void);

/**
 * @brief Execute a compute graph using NUMA-aware strategies
 * 
 * Main entry point for NUMA-aware graph execution. Analyzes each tensor
 * operation in the graph and applies optimal NUMA strategies for maximum
 * performance on multi-socket systems.
 * 
 * Features:
 * - Per-operation strategy optimization
 * - Automatic fallback for unsupported operations
 * - Resource-aware scheduling and load balancing
 * - Performance monitoring and statistics collection
 * 
 * This is called by the CPU backend after cplan preparation.
 * 
 * @param cgraph The compute graph to execute
 * @param cplan The compute plan with threading and buffer info
 * @return GGML_STATUS_SUCCESS on success, error code on failure
 */
enum ggml_status ggml_numa_executor_compute_graph(
    struct ggml_cgraph * cgraph, 
    struct ggml_cplan * cplan);

/**
 * @brief Execute a single tensor operation using optimal NUMA strategy
 * 
 * Core execution function that analyzes a single tensor operation and
 * delegates to the most appropriate NUMA kernel. Performs:
 * - Operation complexity analysis and strategy selection
 * - Kernel registry lookup with O(1) direct dispatch
 * - Resource allocation and NUMA topology optimization
 * - Work distribution across coordinator threadpools
 * 
 * Analyzes the operation and delegates to appropriate kernel
 * 
 * @param tensor The operation tensor to execute
 * @param cplan The compute plan with resources
 * @return GGML_STATUS_SUCCESS on success, error code on failure
 */
enum ggml_status ggml_numa_executor_execute_tensor(
    struct ggml_tensor * tensor,
    struct ggml_cplan * cplan);

/**
 * @brief Execute a single tensor operation with forced NUMA strategy
 * 
 * Similar to ggml_numa_executor_execute_tensor() but forces a specific
 * execution strategy instead of using automatic strategy selection.
 * This is primarily used for testing to validate specific execution paths.
 * 
 * @param tensor The operation tensor to execute
 * @param cplan The compute plan with resources
 * @param forced_strategy The strategy to force (overrides automatic selection)
 * @return GGML_STATUS_SUCCESS on success, error code on failure
 */
enum ggml_status ggml_numa_executor_execute_tensor_forced_strategy(
    struct ggml_tensor * tensor,
    struct ggml_cplan * cplan,
    ggml_numa_execution_strategy_t forced_strategy);

/**
 * @brief Direct Fallback kernel dispatch function for maximum performance
 * 
 * High-performance execution path that eliminates temporary graph overhead
 * by calling compute functions directly. This function provides the fastest
 * possible execution for fallback operations in ggml-cpu.c
 * 
 * Performance Benefits:
 * - Zero-copy operation without graph allocation
 * - Direct function call without dispatch overhead
 * 
 * Use Cases:
 * - Fallback path when no NUMA kernel exists for an op
 * 
 * @param tensor The operation tensor to execute
 * @param cplan The compute plan with threading and buffer information
 * @return GGML_STATUS_SUCCESS on success, error code on failure
 */
enum ggml_status ggml_numa_executor_direct_kernel_dispatch(struct ggml_tensor * tensor, struct ggml_cplan * cplan);

/**
 * @brief Call the direct kernel compute function for a given tensor operation
 * 
 * Internal helper function for direct kernel dispatch that handles the
 * low-level invocation of compute kernels with proper parameter setup
 * and error handling.
 * 
 * @param tensor The operation tensor
 * @param params The compute parameters
 * @return GGML_STATUS_SUCCESS on success, error code on failure
 */
enum ggml_status ggml_numa_executor_call_direct_kernel(struct ggml_tensor * tensor, struct ggml_compute_params * params);

#ifdef __cplusplus
}
#endif
