/*
 * NUMA Kernel Registry - O(1) Array-Based Strategy Selection
 * 
 * This module provides a centralized registry for all NUMA kernels with
 * ultra-fast O(1) strategy selection using direct array access.
 * Kernels provide simple threshold arrays at registration time for optimal performance.
 */

#pragma once

#include "ggml.h"
#include "ggml-numa-shared.h"  // For execution strategy types and utilities

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct ggml_cplan;

/**
 * NUMA Strategy Array Manager - O(1) Array-Based Selection
 *
 * Each kernel registers simple threshold arrays and function pointers.
 * The array manager builds sparse arrays at startup for O(1) direct access lookups.
 */

// Forward declaration for query result type  
struct ggml_tensor;  // Ensure struct ggml_tensor is declared
typedef struct ggml_numa_kernel_query_result_s ggml_numa_kernel_query_result_t;

/**
 * Query function pointer type for kernels
 * Each kernel provides a query function of this type
 */
typedef ggml_numa_kernel_query_result_t (*ggml_numa_kernel_query_fn_t)(const struct ggml_tensor * tensor);

/**
 * Work buffer size calculation function pointer type
 * Kernels can provide this function to calculate required work buffer sizes
 * @param tensor - The tensor being processed
 * @param total_numa_nodes - Total NUMA nodes participating
 * @param total_threads - Total threads participating across all nodes
 * @return Per-thread work buffer size in bytes
 */
typedef size_t (*ggml_numa_kernel_work_buffer_calc_fn_t)(const struct ggml_tensor * tensor, int total_numa_nodes, int total_threads);

/**
 * Kernel array entry for O(1) kernel lookup and dispatch
 * Contains direct function pointers for maximum performance
 */
typedef struct ggml_numa_kernel_cache_entry {
    enum ggml_op op_type;                                ///< Operation type for this cache entry
    bool supported;                                      ///< True if kernel is supported
    bool is_noop;                                        ///< Skip coordinator dispatch for no-op kernels
    char kernel_name[64];                               ///< Human-readable kernel name for debugging
    
    // Direct function pointers for O(1) dispatch
    ggml_numa_kernel_query_fn_t query_fn;              ///< Query function for strategy selection
    ggml_numa_kernel_work_buffer_calc_fn_t work_buffer_calc_fn; ///< Work buffer size calculation (optional)
    
    // Strategy thresholds for automatic selection
    ggml_numa_kernel_strategy_array_t strategy_array;   ///< Strategy thresholds for element count based selection
    
    // Work function pointers for different strategies
    ggml_numa_kernel_work_funcs_t work_funcs;          ///< Work functions for computation
    ggml_numa_kernel_aggregation_funcs_t agg_funcs;    ///< Aggregation functions (optional)
} ggml_numa_kernel_cache_entry_t;

/**
 * Global kernel array with direct access - NO HASH COMPUTATION!
 * 
 * Two-array system for maximum performance:
 * 1. cache_storage[GGML_OP_COUNT]: Main storage (sparse array, most entries NULL)
 * 2. lookup_table[GGML_OP_COUNT]: Fast lookup pointers (what we query in hot path)
 * 
 * Usage: result = lookup_table[op_type]; if (!result) fallback();
 * Performance: Single memory access + NULL check = ~2-3 CPU cycles
 */
typedef struct {
    ggml_numa_kernel_cache_entry_t cache_storage[GGML_OP_COUNT];        // Array 1: Main array storage
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
    ggml_numa_kernel_query_fn_t query_fn,
    ggml_numa_kernel_work_buffer_calc_fn_t work_buffer_calc_fn,
    bool supported,
    bool is_noop);

/**
 * Initialize the global kernel array system (called once at startup)
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
 * Check if a kernel is a no-op kernel that skips coordinator dispatch
 * Returns true if the kernel is flagged as no-op, false otherwise
 */
bool ggml_numa_is_kernel_noop(enum ggml_op op_type);

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
typedef struct ggml_numa_kernel_query_result_s {
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
 * IMPROVED: Now uses direct array access for optimal performance.
 * 
 * @param tensor The tensor operation to query about
 * @return Query result with all execution information
 */
ggml_numa_kernel_query_result_t ggml_numa_kernels_query(const struct ggml_tensor * tensor);

/**
 * Fast strategy lookup for a tensor operation (simplified interface)
 * 
 * @param tensor The tensor operation to query about
 * @return Execution strategy result with O(1) array lookup
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
 * NUMA_SELECT_STRATEGY_BY_THRESHOLD - Macro for optimal threshold-based strategy selection
 * 
 * This macro implements the efficient threshold search pattern used by multiple kernels.
 * It starts from the simplest strategy (fastest for tiny operations) and works up.
 * 
 * @param threshold_array - Array of threshold structures (e.g., ADD_THRESHOLDS)
 * @param threshold_count - Number of elements in the array (e.g., ADD_THRESHOLD_COUNT)
 * @param total_elements - Number of elements in the tensor
 * @param selected_strategy - [OUT] Pointer to the selected strategy structure
 * 
 * Usage example:
 *   const ggml_add_strategy_threshold_t * selected_strategy;
 *   NUMA_SELECT_STRATEGY_BY_THRESHOLD(ADD_THRESHOLDS, ADD_THRESHOLD_COUNT, total_elements, selected_strategy);
 * 
 * Performance: O(1) for tiny operations (most common case), O(log n) worst case
 */
#define NUMA_SELECT_STRATEGY_BY_THRESHOLD(threshold_array, threshold_count, total_elements, selected_strategy) do { \
    /* Find the first threshold where total_elements < element_threshold */ \
    selected_strategy = &threshold_array[threshold_count - 1]; /* Default to last (largest) strategy */ \
    for (size_t i = 0; i < threshold_count; i++) { \
        if (total_elements < threshold_array[i].element_threshold) { \
            selected_strategy = &threshold_array[i]; \
            break; \
        } \
    } \
} while(0)

/**
 * NUMA_SELECT_STRATEGY_FROM_CACHE - Macro for unified array-based strategy selection
 * 
 * This macro implements the standard three-tier strategy selection pattern used by kernels.
 * It encapsulates the threshold comparison logic using registered array entry thresholds.
 * Eliminates code duplication across kernel query functions.
 * 
 * @param cache_entry - Pointer to the array-stored kernel registration info
 * @param total_elements - Number of elements in the tensor
 * @param selected_strategy - [OUT] Variable to store the selected strategy
 * 
 * Usage example in kernel query functions:
 *   ggml_numa_execution_strategy_t selected_strategy;
 *   NUMA_SELECT_STRATEGY_FROM_CACHE(cache_entry, total_elements, selected_strategy);
 * 
 * Three-tier strategy:
 * - Below threshold[0]: Single node, single thread (fastest for tiny tensors)
 * - Below threshold[1]: Single node, multi-thread (good for small-medium tensors)  
 * - Above threshold[1]: Data-parallel across NUMA nodes (optimal for large tensors)
 */
#define NUMA_SELECT_STRATEGY_FROM_CACHE(cache_entry, total_elements, selected_strategy) do { \
    if (total_elements < cache_entry->strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_SINGLE]) { \
        /* Very small tensors: single-threaded */ \
        selected_strategy.node_strategy = NUMA_NODE_STRATEGY_SINGLE; \
        selected_strategy.on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD; \
    } else if (total_elements < cache_entry->strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_MULTI]) { \
        /* Small to medium tensors: multi-threaded on single node */ \
        selected_strategy.node_strategy = NUMA_NODE_STRATEGY_SINGLE; \
        selected_strategy.on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD; \
    } else { \
        /* Large tensors: data-parallel across NUMA nodes */ \
        selected_strategy.node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL; \
        selected_strategy.on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD; \
    } \
} while(0)

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
 * - Function: ggml_numa_kernel_{kname}_query()
 * - Returns: ggml_numa_kernel_registration_info_t with supported flag
 */
#define NUMA_REGISTER_KERNEL(kname) do { \
    ggml_numa_kernel_registration_info_t kname##_info = ggml_numa_kernel_##kname##_register(); \
    ggml_numa_kernel_query_fn_t kname##_query_fn = ggml_numa_kernel_##kname##_query; \
    ggml_numa_kernel_work_buffer_calc_fn_t kname##_work_buffer_fn = NULL; \
    /* Check if kernel provides work buffer calculation function */ \
    if (kname##_info.work_buffer_calc_fn != NULL) { \
        kname##_work_buffer_fn = kname##_info.work_buffer_calc_fn; \
    } \
    enum ggml_status kname##_result = ggml_numa_register_kernel_strategy( \
        kname##_info.op_type, &kname##_info.strategy_array, \
        &kname##_info.work_funcs, &kname##_info.agg_funcs, kname##_query_fn, kname##_work_buffer_fn, kname##_info.supported, kname##_info.is_noop); \
    if (kname##_result != GGML_STATUS_SUCCESS) { \
        NUMA_LOG_ERROR("Failed to register " #kname " kernel strategy"); \
        return kname##_result; \
    } \
    if (kname##_info.supported) { \
        NUMA_LOG_DEBUG("✅ Registered %s (thresholds: %zu/%zu%s%s)", kname##_info.kernel_name, \
                      kname##_info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_SINGLE], \
                      kname##_info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_MULTI], \
                      kname##_info.is_noop ? ", no-op" : "", \
                      kname##_work_buffer_fn ? ", work-buffer" : ""); \
    } else { \
        NUMA_LOG_DEBUG("🚫 Disabled %s (marked as unsupported)", kname##_info.kernel_name); \
    } \
} while(0)

#ifdef __cplusplus
}
#endif
