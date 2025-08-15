#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations to avoid include dependencies
struct ggml_cplan;

/**
 * @file ggml-numa-fallback.h
 * @brief Centralized NUMA-aware fallback system for unsupported operations
 * 
 * This module provides a unified fallback mechanism for operations that don't have
 * NUMA-aware implementations. It's designed to be used by both the dispatcher
 * and coordinator without creating circular dependencies.
 */

/**
 * Execute an operation using single-threaded fallback
 * 
 * This is the fast fallback path that directly calls operation-specific functions
 * without creating temporary contexts or graphs. It should be used for all 
 * performance-critical fallback scenarios.
 *
 * @param tensor The operation tensor to execute
 * @param cplan Optional computation plan with work buffer (can be NULL)
 * @return GGML_STATUS_SUCCESS on success, GGML_STATUS_FAILED on error
 */
enum ggml_status ggml_numa_fallback_execute(struct ggml_tensor * tensor, struct ggml_cplan * cplan);

/**
 * Check if an operation is supported by the fallback system
 *
 * @param op The operation type to check
 * @return true if the operation has a fallback implementation, false otherwise
 */
bool ggml_numa_fallback_is_supported(enum ggml_op op);

/**
 * Get statistics about fallback usage
 * 
 * @param fallback_count Pointer to store the number of fallback executions
 */
void ggml_numa_fallback_get_stats(int64_t * fallback_count);

#ifdef __cplusplus
}
#endif
