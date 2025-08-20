/*
 * NUMA Kernel Registry - Centralized Kernel Management
 * 
 * This module provides a centralized registry for all NUMA kernels,
 * acting as a database that the executor queries to determine
 * execution strategies, compute buffer requirements, and work functions.
 */

#pragma once

#include "ggml.h"
#include "../ggml-numa-coordinator.h"  // For execution strategy types

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct ggml_cplan;

/**
 * Kernel execution information returned by registry queries
 * Contains all information needed for the executor to dispatch work to coordinator
 */
typedef struct {
    bool supported;                                    // Whether this operation is supported
    ggml_numa_execution_strategy_t strategy;          // Recommended execution strategy
    size_t work_buffer_size_per_thread;              // Required compute buffer size per thread
    ggml_numa_work_function_t work_function;         // Function pointer for coordinator execution
    float efficiency_score;                           // Efficiency estimate (0.0-1.0)
    const char * kernel_name;                         // Human-readable kernel name
} ggml_numa_kernel_query_result_t;

/**
 * Initialize the NUMA kernel registry
 * Registers all available NUMA kernels
 * 
 * @return true on success, false on failure
 */
bool ggml_numa_kernels_init(void);

/**
 * Cleanup the NUMA kernel registry
 */
void ggml_numa_kernels_cleanup(void);

/**
 * Query the kernel registry for execution information
 * 
 * This is the main interface used by the executor to determine:
 * - Whether an operation is supported by NUMA kernels
 * - What execution strategy should be used
 * - How much compute buffer each thread needs
 * - Which work function the coordinator should execute
 * 
 * @param tensor The tensor operation to query about
 * @return Query result with all execution information
 */
ggml_numa_kernel_query_result_t ggml_numa_kernels_query(const struct ggml_tensor * tensor);

/**
 * Legacy compatibility functions (for backward compatibility during transition)
 * These may be removed once executor is fully updated to use query interface
 */
bool ggml_numa_kernels_supports(enum ggml_op op, const struct ggml_tensor * tensor);

/**
 * Execute a tensor operation using the appropriate NUMA kernel
 * 
 * @param tensor The tensor operation to execute
 * @param cplan The compute plan with threading and buffer information
 * @return GGML_STATUS_SUCCESS on success, error code on failure
 */
enum ggml_status ggml_numa_kernels_execute(struct ggml_tensor * tensor, struct ggml_cplan * cplan);

/**
 * Get the efficiency score for a tensor operation
 * Higher scores indicate better NUMA performance characteristics
 * 
 * @param tensor The tensor to evaluate
 * @param tensor_size The size of the tensor in bytes
 * @return Efficiency score (0.0 to 1.0), or 0.0 if no kernel available
 */
// Get efficiency score for a kernel with the given tensor
float ggml_numa_kernels_get_efficiency(enum ggml_op op, const struct ggml_tensor * tensor, size_t tensor_size);

/**
 * Get the number of registered kernels
 * 
 * @return Number of registered NUMA kernels
 */
// Get total number of registered kernels
size_t ggml_numa_kernels_get_count(void);

#ifdef __cplusplus
}
#endif
