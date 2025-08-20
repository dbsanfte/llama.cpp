/*
 * NUMA Executor - Strategy Engine and Work Orchestration
 * 
 * This component handles execution strategy selection and work submission.
 * It replaces the old dispatcher logic with a cleaner architecture.
 */

#pragma once

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"  // For complete ggml_cplan definition

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct ggml_numa_coordinator_manager;
struct ggml_cplan;

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
 * Check if executor supports a specific operation
 * 
 * @param op The operation type to check
 * @return true if supported, false otherwise
 */
bool ggml_numa_executor_supports_op(enum ggml_op op);

/**
 * Get estimated efficiency for an operation
 * 
 * @param op The operation type
 * @param tensor_size Size in elements
 * @return Efficiency estimate (0.0-1.0) or -1.0 if unsupported
 */
float ggml_numa_executor_get_efficiency(enum ggml_op op, size_t tensor_size);

/**
 * Fallback to standard CPU implementation for unsupported operations
 * This calls ggml-cpu.c functions directly to avoid infinite recursion
 * 
 * @param tensor The operation tensor
 * @param cplan The compute plan
 * @return GGML_STATUS_SUCCESS on success, error code on failure
 */
enum ggml_status ggml_numa_executor_fallback_to_cpu(struct ggml_tensor * tensor, struct ggml_cplan * cplan);

#ifdef __cplusplus
}
#endif
