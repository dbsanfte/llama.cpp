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
 * Kernel cache entry - stores complete kernel information
 * Maps operation type to pre-computed query results for ultra-fast lookup
 */
typedef struct {
    enum ggml_op op_type;                                        // Operation type
    ggml_numa_kernel_strategy_array_t strategy_array;           // Strategy thresholds for fast lookup
    ggml_numa_kernel_work_funcs_t work_funcs;                   // Work function pointers
    ggml_numa_kernel_aggregation_funcs_t agg_funcs;             // Aggregation function pointers (optional)
    bool supported;                                              // Whether kernel is enabled/supported
} ggml_numa_kernel_cache_entry_t;

/**
 * Global kernel cache with direct array access - NO HASH TABLE!
 * 
 * Two-array system for maximum performance:
 * 1. g_kernel_cache[GGML_OP_COUNT]: Main storage (sparse array, most entries NULL)
 * 2. g_kernel_lookup[GGML_OP_COUNT]: Fast lookup pointers (what we query in hot path)
 * 
 * Usage: result = g_kernel_lookup[op_type]; if (!result) fallback();
 * Performance: Single memory access + NULL check = ~2-3 CPU cycles
 */
typedef struct {
    ggml_numa_kernel_cache_entry_t cache_storage[GGML_OP_COUNT];        // Array 1: Main cache storage
    ggml_numa_kernel_cache_entry_t* lookup_table[GGML_OP_COUNT];        // Array 2: Fast lookup pointers
    bool cache_initialized;                                             // Initialization flag  
    size_t num_registered_ops;                                         // Count of registered operations
} ggml_numa_kernel_array_cache_t;

/**
 * Kernel registration interface - called by each kernel at startup
 * Kernels provide simple threshold arrays and function pointers
 */
enum ggml_status ggml_numa_register_kernel_strategy(
    enum ggml_op op_type, 
    const ggml_numa_kernel_strategy_array_t * strategy_array,
    const ggml_numa_kernel_work_funcs_t * work_funcs,
    const ggml_numa_kernel_aggregation_funcs_t * agg_funcs,
    bool supported);

/**
 * Initialize the global kernel array cache (called once at startup)
 */
enum ggml_status ggml_numa_init_kernel_array_cache(void);

/**
 * Direct array lookup - ultra-fast single memory access
 * Returns complete kernel information or NULL if unsupported
 */
const ggml_numa_kernel_cache_entry_t * ggml_numa_lookup_kernel_direct(enum ggml_op op_type);

/**
 * Check if a kernel is registered and supported
 * Returns true if the kernel is available for use, false otherwise
 */
bool ggml_numa_is_kernel_supported(enum ggml_op op_type);

/**
 * Array-based strategy lookup - direct access using operation type as index
 * Returns execution strategy based on operation type and element count
 */
const ggml_numa_execution_strategy_t * ggml_numa_lookup_strategy_direct(
    enum ggml_op op_type,
    size_t element_count
);

/**
 * Array-based work function lookup - direct access using operation type as index
 * Returns function pointer for execution based on operation and strategy
 */
ggml_numa_work_function_t ggml_numa_lookup_work_function_direct(
    enum ggml_op op_type,
    const ggml_numa_execution_strategy_t * strategy
);

/**
 * Array-based aggregation function lookup - direct access using operation type as index
 * Returns aggregation function pointer based on operation and strategy
 */
enum ggml_status (*ggml_numa_lookup_aggregation_direct(
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

/**
 * Apply force strategy override to query result
 * 
 * This helper function applies NUMA_FORCE_STRATEGY environment variable
 * overrides to kernel query results. It should be called by all kernel
 * query functions after normal strategy selection.
 * 
 * @param result Pointer to query result to potentially override
 * @param op_name Operation name for logging purposes
 * @param single_single_fn Function pointer for single-node, single-thread execution
 * @param single_multi_fn Function pointer for single-node, multi-thread execution  
 * @param data_parallel_fn Function pointer for data-parallel execution
 * @return true if strategy was overridden, false if unchanged
 */
bool ggml_numa_apply_kernel_force_strategy(ggml_numa_kernel_query_result_t * result,
                                           const char * op_name,
                                           ggml_numa_work_function_t single_single_fn,
                                           ggml_numa_work_function_t single_multi_fn,
                                           ggml_numa_work_function_t data_parallel_fn);

/**
 * NUMA_REGISTER_KERNEL - Macro to simplify kernel registration
 * 
 * This macro eliminates code duplication when registering NUMA kernels.
 * It handles the registration pattern, error checking, and debug logging.
 * 
 * @param kname - The kernel name (e.g., add, mul, cpy)
 * 
 * Usage example:
 *   NUMA_REGISTER_KERNEL(add);
 *   NUMA_REGISTER_KERNEL(mul);
 * 
 * The macro expects:
 * - Function: ggml_numa_kernel_{kname}_register()
 * - Returns: ggml_numa_kernel_registration_info_t with supported flag
 */
#define NUMA_REGISTER_KERNEL(kname) do { \
    ggml_numa_kernel_registration_info_t kname##_info = ggml_numa_kernel_##kname##_register(); \
    enum ggml_status kname##_result = ggml_numa_register_kernel_strategy( \
        kname##_info.op_type, &kname##_info.strategy_array, \
        &kname##_info.work_funcs, &kname##_info.agg_funcs, kname##_info.supported); \
    if (kname##_result != GGML_STATUS_SUCCESS) { \
        NUMA_LOG_ERROR("Failed to register " #kname " kernel strategy"); \
        return kname##_result; \
    } \
    if (kname##_info.supported) { \
        NUMA_LOG_DEBUG("✅ Registered %s (thresholds: %zu/%zu)", kname##_info.kernel_name, \
                      kname##_info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_SINGLE], \
                      kname##_info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_MULTI]); \
    } else { \
        NUMA_LOG_DEBUG("🚫 Disabled %s (marked as unsupported)", kname##_info.kernel_name); \
    } \
} while(0)

#ifdef __cplusplus
}
#endif
