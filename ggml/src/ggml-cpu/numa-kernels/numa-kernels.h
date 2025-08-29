/*
 * NUMA Kernel Registry - O(1) Hash Table Strategy Selection
 * 
 * This module provides a centralized registry for all NUMA kernels with
 * ultra-fast O(1) hash table strategy selection. Kernels provide simple
 * threshold arrays at registration time for optimal performance.
 */

#pragma once

#include "ggml.h"
#include "ggml-numa-shared.h"  // For execution strategy types and hash table utilities

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct ggml_cplan;

/**
 * NUMA Strategy Cache Manager - O(1) Hash Table System
 * 
 * Each kernel registers simple threshold arrays and function pointers.
 * The cache manager builds hash tables at startup for O(1) lookups.
 */

/**
 * Hash table entry for strategy cache
 * Maps operation type to strategy threshold array
 */
typedef struct {
    enum ggml_op op_type;                                // Operation type (hash key)
    ggml_numa_kernel_strategy_array_t strategy_array;   // Threshold array
    ggml_numa_kernel_aggregation_funcs_t agg_funcs;     // Function pointers
    bool initialized;                                    // True if entry is valid
} ggml_numa_strategy_cache_entry_t;

/**
 * Global strategy cache with O(1) hash table access
 * Built at startup from kernel registrations
 */
typedef struct {
    ggml_numa_strategy_cache_entry_t entries[NUMA_OP_HASH_TABLE_SIZE];  // Hash table
    bool cache_initialized;                                              // Initialization flag
    size_t num_registered_ops;                                          // Count of registered operations
} ggml_numa_strategy_cache_t;

/**
 * Kernel registration interface - called by each kernel at startup
 * Kernels provide simple threshold arrays and function pointers
 */
enum ggml_status ggml_numa_register_kernel_strategy(
    enum ggml_op op_type,
    const ggml_numa_kernel_strategy_array_t * strategy_array,
    const ggml_numa_kernel_aggregation_funcs_t * agg_funcs
);

/**
 * Initialize the global strategy cache (called once at startup)
 */
enum ggml_status ggml_numa_init_strategy_cache(void);

/**
 * O(1) strategy lookup - ultra-fast hash table access
 * Returns execution strategy based on operation type and element count
 */
const ggml_numa_execution_strategy_t * ggml_numa_lookup_strategy_fast(
    enum ggml_op op_type,
    size_t element_count
);

/**
 * O(1) aggregation function lookup - ultra-fast hash table access
 * Returns function pointer for aggregation based on operation and strategy
 */
enum ggml_status (*ggml_numa_lookup_aggregation_fast(
    enum ggml_op op_type,
    const ggml_numa_execution_strategy_t * strategy
))(void *, int, struct ggml_tensor *, struct ggml_cplan *);

/**
 * NUMA aggregation policy for kernels
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
 * Both cache and threshold-based queries return this structure.
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
 * @return ggml_status on success/failure
 */
enum ggml_status ggml_numa_kernels_init(void);

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
 * IMPROVED: Now uses kernel-specific O(1) hash table lookups for optimal performance.
 * 
 * @param tensor The tensor operation to query about
 * @return Query result with all execution information
 */
ggml_numa_kernel_query_result_t ggml_numa_kernels_query(const struct ggml_tensor * tensor);

/**
 * Fast strategy lookup for a tensor operation (simplified interface)
 * 
 * @param tensor The tensor operation to query about
 * @return Execution strategy result with O(1) threshold lookup
 */
ggml_numa_kernel_query_result_t ggml_numa_kernels_strategy_lookup(const struct ggml_tensor * tensor);

/**
 * Get strategy cache statistics for performance monitoring
 */
typedef struct {
    uint64_t total_lookups;     // Total strategy lookups
    uint64_t cache_hits;        // Cache hits
    uint64_t cache_misses;      // Cache misses requiring computation
    double hit_rate;            // Hit rate percentage
} numa_strategy_cache_stats_t;

numa_strategy_cache_stats_t ggml_numa_kernels_get_strategy_stats(void);

/**
 * Clear all strategy caches (for testing)
 */
void ggml_numa_kernels_clear_strategy_cache(void);

#ifdef __cplusplus
}
#endif
