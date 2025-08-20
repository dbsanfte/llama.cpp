/*
 * NUMA-Aware CPU Backend for GGML - Header
 * 
 * This header defines the interface for the complete NUMA-aware CPU backend
 * that replaces ggml-cpu.c when NUMA is enabled.
 */

#pragma once

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"  // For complete ggml_cplan definition

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration
struct ggml_cplan;

/**
 * Main entry point for NUMA-aware graph computation
 * This function completely replaces ggml_graph_compute when NUMA is enabled
 * 
 * @param cgraph The compute graph to execute
 * @param cplan The compute plan with thread configuration
 * @return GGML_STATUS_SUCCESS on success, error code on failure
 */
enum ggml_status ggml_numa_graph_compute_impl(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);

/**
 * Initialize the NUMA CPU backend
 * Sets up NUMA-aware computation resources
 * 
 * @param n_threads Number of threads to use
 * @return true on success, false on failure
 */
bool ggml_numa_cpu_backend_init(int n_threads);

/**
 * Clean up the NUMA CPU backend
 * Releases NUMA-aware computation resources
 */
void ggml_numa_cpu_backend_cleanup(void);

/**
 * Check if a specific operation is supported by the NUMA backend
 * 
 * @param op The operation to check
 * @return true if supported, false otherwise
 */
bool ggml_numa_cpu_backend_supports_op(enum ggml_op op);

/**
 * Get performance estimate for an operation on the NUMA backend
 * 
 * @param op The operation
 * @param tensor_size Size of the tensor in elements
 * @return Efficiency estimate (0.0-1.0) or -1.0 if not supported
 */
float ggml_numa_cpu_backend_get_efficiency(enum ggml_op op, size_t tensor_size);

#ifdef __cplusplus
}
#endif
