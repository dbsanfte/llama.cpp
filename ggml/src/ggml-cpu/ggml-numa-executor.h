/*
 * NUMA Executor - Strategy Engine and Work Orchestration
 * 
 * This component handles execution strategy selection and work submission.
 * It replaces the old dispatcher logic with a cleaner architecture.
 */

#pragma once

#include "ggml.h"
#include "../ggml-impl.h"
#include "ggml-cpu.h"  // For complete ggml_cplan definition
#include "ggml-cpu-impl.h"  // For ggml_compute_params definition

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct ggml_numa_coordinator_manager;
struct ggml_cplan;

/**
 * Initialize the NUMA executor and kernel registry
 * This must be called before using any executor functions
 * 
 * @return GGML_STATUS_SUCCESS on success, error code on failure
 */
enum ggml_status ggml_numa_executor_init(void);

/**
 * Cleanup the NUMA executor and kernel registry
 */
void ggml_numa_executor_cleanup(void);

/**
 * Execute a compute graph using NUMA-aware strategies
 * This is called by the CPU backend after cplan preparation
 * 
 * @param cgraph The compute graph to execute
 * @param cplan The compute plan with threading and buffer info
 * @return GGML_STATUS_SUCCESS on success, error code on failure
 */
enum ggml_status ggml_numa_executor_compute_graph(
    struct ggml_cgraph * cgraph, 
    struct ggml_cplan * cplan);

/**
 * Execute a single tensor operation using optimal NUMA strategy
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
 * Execute compute graph using NUMA-aware executor
 * Main entry point for compute graph execution with NUMA optimization
 * 
 * @param cgraph The compute graph to execute
 * @param cplan The compute plan
 * @return GGML_STATUS_SUCCESS on success, error code on failure
 */
enum ggml_status ggml_numa_executor_execute_graph(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);

/**
 * Direct kernel dispatch for maximum performance - OPTIMIZED VERSION
 * Calls compute functions directly without temporary graph creation overhead
 * This eliminates the performance bottlenecks in the fallback system
 * 
 * @param tensor The operation tensor
 * @param cplan The compute plan
 * @return GGML_STATUS_SUCCESS on success, error code on failure
 */
enum ggml_status ggml_numa_executor_direct_kernel_dispatch(struct ggml_tensor * tensor, struct ggml_cplan * cplan);

/**
 * Call the direct kernel compute function for a given tensor operation
 * Internal helper function for direct kernel dispatch
 * 
 * @param tensor The operation tensor
 * @param params The compute parameters
 * @return GGML_STATUS_SUCCESS on success, error code on failure
 */
enum ggml_status ggml_numa_executor_call_direct_kernel(struct ggml_tensor * tensor, struct ggml_compute_params * params);

/**
 * Fallback to standard CPU implementation for unsupported operations - LEGACY VERSION
 * This calls ggml-cpu.c functions via temporary graph (high overhead)
 * Should be replaced by direct kernel dispatch for better performance
 * 
 * @param tensor The operation tensor
 * @param cplan The compute plan
 * @return GGML_STATUS_SUCCESS on success, error code on failure
 */
enum ggml_status ggml_numa_executor_fallback_to_cpu(struct ggml_tensor * tensor, struct ggml_cplan * cplan);

#ifdef __cplusplus
}
#endif
