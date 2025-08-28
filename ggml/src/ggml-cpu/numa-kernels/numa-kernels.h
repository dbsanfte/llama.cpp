/*
 * NUMA Kernel Registry - Flexible Threshold-Based Strategy Selection
 * 
 * This module provides a centralized registry for all NUMA kernels with
 * improved threshold-based strategy selection. Operations define their
 * own optimal thresholds rather than fitting into rigid complexity classes.
 */

#pragma once

#include "ggml.h"
#include "ggml-numa-shared.h"  // For execution strategy types

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct ggml_cplan;

/**
 * Legacy complexity classes for backward compatibility during transition
 * New operations should use threshold-based queries instead
 */
typedef enum {
    COMPLEXITY_TINY = 0,    // < 1K elements
    COMPLEXITY_SMALL,       // 1K - 16K elements  
    COMPLEXITY_MEDIUM,      // 16K - 256K elements
    COMPLEXITY_LARGE,       // 256K - 4M elements
    COMPLEXITY_HUGE,        // 4M - 64M elements
    COMPLEXITY_GIGANTIC_1GB,  // 64M - 512M elements (~1GB)
    COMPLEXITY_GIGANTIC_2GB,  // 512M - 1024M elements (~2GB)
    COMPLEXITY_GIGANTIC_4GB,  // 1024M - 2048M elements (~4GB)
    COMPLEXITY_GIGANTIC_8GB,  // 2048M - 4096M elements (~8GB)
    COMPLEXITY_GIGANTIC_16GB, // > 4096M elements (~16GB+)
    COMPLEXITY_COUNT
} ggml_numa_complexity_class_t;

/**
 * Legacy cache entry for backward compatibility during transition
 * New operations should use kernel-specific query functions instead
 */
typedef struct {
    bool valid;                                    // Cache entry is valid
    ggml_numa_execution_strategy_t strategy;      // Pre-computed strategy
    size_t work_buffer_size_per_thread;          // Pre-computed buffer size
    ggml_numa_work_function_t work_function;     // Pre-selected work function
    float efficiency_score;                       // Pre-computed efficiency
    const char * kernel_name;                     // Kernel identifier
} ggml_numa_cache_entry_t;

/**
 * Aggregation policy for NUMA kernels
 * Defines how results should be combined across NUMA nodes
 */
typedef enum {
    GGML_NUMA_AGGREGATION_NONE = 0,      // No aggregation needed, kernel writes directly to final location
    GGML_NUMA_AGGREGATION_CUSTOM         // Use kernel-provided custom aggregation function
} ggml_numa_aggregation_policy_t;

/**
 * Custom aggregation function provided by kernels
 * Called by coordinator to aggregate results from multiple NUMA nodes
 * 
 * @param tensor       The tensor to aggregate
 * @param num_nodes    Number of NUMA nodes that participated in computation
 * @param user_data    Optional user data pointer provided by kernel
 * @return GGML_STATUS_SUCCESS on success, error code on failure
 */
typedef enum ggml_status (*ggml_numa_aggregation_function_t)(
    struct ggml_tensor * tensor, 
    int num_nodes, 
    void * user_data
);

/**
 * Kernel execution information returned by registry queries
 * Contains all information needed for the executor to dispatch work to coordinator
 * 
 * This is the primary interface between the registry and executor.
 * Both legacy cache lookups and new threshold-based queries return this structure.
 */
typedef struct {
    bool supported;                                    // Whether this operation is supported
    ggml_numa_execution_strategy_t strategy;          // Recommended execution strategy
    size_t work_buffer_size_per_thread;              // Required compute buffer size per thread
    ggml_numa_work_function_t work_function;         // Function pointer for coordinator execution
    float efficiency_score;                           // Efficiency estimate (0.0-1.0)
    const char * kernel_name;                         // Human-readable kernel name
    ggml_numa_aggregation_policy_t aggregation_policy; // How to handle result aggregation
    ggml_numa_aggregation_function_t aggregation_function; // Custom aggregation function (if policy is CUSTOM)
    void * aggregation_user_data;                     // User data passed to custom aggregation function
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
 * IMPROVED: Now calls kernel-specific threshold queries first,
 * falling back to legacy cache for operations without threshold support.
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
float ggml_numa_kernels_get_efficiency(enum ggml_op op, const struct ggml_tensor * tensor, size_t tensor_size);

/**
 * Get the number of registered kernels
 * 
 * @return Number of registered NUMA kernels
 */
size_t ggml_numa_kernels_get_count(void);

#ifdef __cplusplus
}
#endif
