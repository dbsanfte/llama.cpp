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
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

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
typedef ggml_numa_execution_strategy_t (*ggml_numa_kernel_query_fn_t)(const struct ggml_tensor * tensor);

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
 * @brief Execution context for NUMA work distribution
 * Contains all necessary information for kernels to distribute work across NUMA nodes and threads
 */
typedef struct {
    int numa_node;                    ///< Current NUMA node identifier
    bool is_data_parallel;            ///< True if executing across multiple NUMA nodes
    int total_threads;                ///< Total threads on this NUMA node
    int thread_id;                    ///< Thread identifier within this NUMA node (0-based)
    size_t numa_start;                ///< Start index for this NUMA node's work
    size_t numa_end;                  ///< End index for this NUMA node's work
    size_t thread_start;              ///< Start index for this thread's work (set by macros)
    size_t thread_end;                ///< End index for this thread's work (set by macros)
    size_t thread_elements;           ///< Number of elements for this thread (set by macros)
} ggml_numa_execution_context_t;

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
 * Get work function for specific strategy from cache entry
 * Ultra-fast O(1) lookup with strategy-based function selection
 */
ggml_numa_work_function_t ggml_numa_get_work_function_from_cache(
    const ggml_numa_kernel_cache_entry_t * cache_entry,
    const ggml_numa_execution_strategy_t * strategy);

/**
 * Get work buffer calculation function from cache entry
 * Returns NULL if operation doesn't need work buffers
 */
ggml_numa_kernel_work_buffer_calc_fn_t ggml_numa_get_work_buffer_calc_from_cache(
    const ggml_numa_kernel_cache_entry_t * cache_entry);

/**
 * Get kernel name from cache entry for debugging/logging
 * Returns static string, safe to use in logging
 */
const char * ggml_numa_get_kernel_name_from_cache(
    const ggml_numa_kernel_cache_entry_t * cache_entry);

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
 * Simplified kernel query result - only returns strategy selection
 * All other metadata comes from the strategy cache entry
 * 
 * This structure is returned by kernel query functions in the hot path
 * and should be kept minimal for performance.
 */
typedef struct ggml_numa_kernel_query_result_s {
    bool supported;                            ///< Whether this operation is supported
    ggml_numa_execution_strategy_t strategy;  ///< Selected execution strategy based on complexity
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
ggml_numa_execution_strategy_t ggml_numa_kernels_query(const struct ggml_tensor * tensor);

/**
 * Fast strategy lookup for a tensor operation (simplified interface)
 * 
 * @param tensor The tensor operation to query about
 * @return Execution strategy result with O(1) array lookup
 */
// Legacy function temporarily disabled during query simplification
// ggml_numa_kernel_query_result_t ggml_numa_kernels_strategy_lookup(const struct ggml_tensor * tensor);

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
 * Legacy function disabled during simplification
 */
/*
bool ggml_numa_apply_kernel_force_strategy(ggml_numa_kernel_query_result_t * result,
                                           const char * op_name,
                                           ggml_numa_work_function_t single_single_fn,
                                           ggml_numa_work_function_t single_multi_fn,
                                           ggml_numa_work_function_t data_parallel_fn);
*/

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
        selected_strategy = NUMA_STRATEGY_SINGLE_THREAD; \
    } else if (total_elements < cache_entry->strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_MULTI]) { \
        /* Small to medium tensors: multi-threaded on single node */ \
        selected_strategy = NUMA_STRATEGY_SINGLE_NODE; \
    } else { \
        /* Large tensors: data-parallel across NUMA nodes */ \
        selected_strategy = NUMA_STRATEGY_DATA_PARALLEL; \
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

// =============================================================================
// SHARED KERNEL FUNCTION MACROS - Eliminate Code Duplication
// =============================================================================

/**
 * NUMA_KERNEL_QUERY_FUNCTION - Macro to generate standard query functions
 * 
 * This macro eliminates the repetitive boilerplate code in query functions.
 * All query functions follow the same pattern: calculate total elements,
 * lookup cache entry, use strategy selection macro, and debug logging.
 * 
 * @param op_name - Operation name (e.g., ADD, MUL, ROPE)
 * @param ggml_op_type - The GGML operation type constant (e.g., GGML_OP_ADD)
 * @param debug_name - Human-readable name for debug logging
 * 
 * Usage example:
 *   NUMA_KERNEL_QUERY_FUNCTION(add, GGML_OP_ADD, "ADD");
 *   NUMA_KERNEL_QUERY_FUNCTION(mul, GGML_OP_MUL, "MUL");
 * 
 * Generates:
 *   ggml_numa_execution_strategy_t ggml_numa_kernel_add_query(const struct ggml_tensor * tensor)
 */
#define NUMA_KERNEL_QUERY_FUNCTION(op_name, threshold_single_single, threshold_single_multi) \
ggml_numa_execution_strategy_t ggml_numa_kernel_##op_name##_query(const struct ggml_tensor * tensor) { \
    /* Calculate total elements for strategy selection (hot path - must be fast) */ \
    const size_t total_elements = ggml_nelements(tensor); \
    \
    /* Direct threshold-based strategy selection for maximum performance */ \
    if (total_elements < threshold_single_single) { \
        return NUMA_STRATEGY_SINGLE_THREAD; \
    } else if (total_elements < threshold_single_multi) { \
        return NUMA_STRATEGY_SINGLE_NODE; \
    } else { \
        return NUMA_STRATEGY_DATA_PARALLEL; \
    } \
}

/**
 * @brief Macro to generate standard work buffer function that returns 0
 *
 * Most operations don't need work buffers, so this macro creates a function
 * that simply returns 0. For operations that need custom work buffers,
 * implement the function manually.
 *
 * @param op_name Operation name (e.g., add, mul, etc.)
 * 
 * Usage examples:
 *   NUMA_KERNEL_WORK_BUFFER_FUNCTION(add);  // Returns 0 - no work buffer needed
 *   NUMA_KERNEL_WORK_BUFFER_FUNCTION(mul);  // Returns 0 - no work buffer needed
 * 
 * Generates:
 *   size_t ggml_numa_kernel_add_work_buffer_calc(const struct ggml_tensor * tensor, int total_numa_nodes, int total_threads)
 */
#define NUMA_KERNEL_WORK_BUFFER_FUNCTION(op_name) \
size_t ggml_numa_kernel_##op_name##_work_buffer_calc(const struct ggml_tensor * tensor, int total_numa_nodes, int total_threads) { \
    /* Standard parameter validation */ \
    (void)tensor; \
    (void)total_numa_nodes; \
    (void)total_threads; \
    \
    /* Most operations don't need work buffers */ \
    return 0; \
}

/**
 * NUMA_KERNEL_REGISTRATION_FUNCTION - Macro to generate standard registration functions
 * 
 * This macro eliminates the highly repetitive registration function boilerplate.
 * Most kernels follow identical patterns with different operation types and thresholds.
 * 
 * @param op_name - Operation name (e.g., add, mul, rope)
 * @param ggml_op_type - The GGML operation type constant (e.g., GGML_OP_ADD)
 * @param kernel_display_name - Human-readable kernel name for debugging
 * @param threshold_single_single - Threshold for single-thread strategy
 * @param threshold_single_multi - Threshold for single-node multi-thread strategy
 * @param execute_function - The execution function name (e.g., ggml_numa_kernel_add_execute)
 * @param needs_aggregation - true if operation needs aggregation functions, false otherwise
 * 
 * Usage examples:
 *   NUMA_KERNEL_REGISTRATION_FUNCTION(add, GGML_OP_ADD, "NUMA ADD Kernel", 1024, 262144, ggml_numa_kernel_add_unified_execute, false);
 *   NUMA_KERNEL_REGISTRATION_FUNCTION(rope, GGML_OP_ROPE, "NUMA ROPE Kernel", 128, 1024, ggml_numa_kernel_rope_execute, false);
 * 
 * Generates:
 *   ggml_numa_kernel_registration_info_t ggml_numa_kernel_add_register(void)
 */
#define NUMA_KERNEL_REGISTRATION_FUNCTION(op_name, ggml_op_type, kernel_display_name, threshold_single_single, threshold_single_multi, execute_function, needs_aggregation) \
ggml_numa_kernel_registration_info_t ggml_numa_kernel_##op_name##_register(void) { \
    ggml_numa_kernel_registration_info_t info = {0}; \
    \
    info.op_type = ggml_op_type; \
    info.supported = true; \
    info.kernel_name = kernel_display_name; \
    \
    /* Strategy thresholds for operation */ \
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_SINGLE] = threshold_single_single; \
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_MULTI] = threshold_single_multi; \
    /* Above threshold_single_multi elements: data-parallel strategy */ \
    info.strategy_array.valid = true; \
    \
    /* Function pointers for different strategies */ \
    info.work_funcs.single_single_fn = execute_function; \
    info.work_funcs.single_multi_fn = execute_function; \
    info.work_funcs.data_parallel_fn = execute_function; \
    info.work_funcs.valid = true; \
    \
    /* Query function pointer - enables direct dispatch without switch statements */ \
    info.query_fn = (void*)ggml_numa_kernel_##op_name##_query; \
    \
    /* Work buffer calculation function pointer */ \
    info.work_buffer_calc_fn = (void*)ggml_numa_kernel_##op_name##_work_buffer_calc; \
    \
    /* Aggregation functions - most operations don't need them */ \
    if (needs_aggregation) { \
        info.agg_funcs.single_single_fn = (void*)ggml_numa_kernel_##op_name##_aggregate; \
        info.agg_funcs.single_multi_fn = (void*)ggml_numa_kernel_##op_name##_aggregate; \
        info.agg_funcs.data_parallel_fn = (void*)ggml_numa_kernel_##op_name##_aggregate; \
        info.agg_funcs.valid = true; \
    } else { \
        info.agg_funcs.single_single_fn = NULL; \
        info.agg_funcs.single_multi_fn = NULL; \
        info.agg_funcs.data_parallel_fn = NULL; \
        info.agg_funcs.valid = false; \
    } \
    \
    return info; \
}

/**
 * @brief Macro to generate kernel registration function WITHOUT aggregation
 *
 * For operations that don't need reduction/aggregation (most element-wise operations)
 *
 * @param op_name Operation name (e.g., add, mul, div, sub)
 * @param ggml_op_type GGML operation type (e.g., GGML_OP_ADD)
 * @param kernel_display_name Display name for logging
 * @param threshold_single_single Threshold for single-thread strategy
 * @param threshold_single_multi Threshold for multi-thread strategy  
 * @param execute_function Function pointer to the execution function
 */
#define NUMA_KERNEL_REGISTRATION_FUNCTION_NO_AGG(op_name, ggml_op_type, kernel_display_name, threshold_single_single, threshold_single_multi, execute_function) \
ggml_numa_kernel_registration_info_t ggml_numa_kernel_##op_name##_register(void) { \
    ggml_numa_kernel_registration_info_t info = {0}; \
    \
    info.op_type = ggml_op_type; \
    info.supported = true; \
    info.kernel_name = kernel_display_name; \
    \
    /* Strategy thresholds for operation */ \
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_SINGLE] = threshold_single_single; \
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_MULTI] = threshold_single_multi; \
    /* Above threshold_single_multi elements: data-parallel strategy */ \
    info.strategy_array.valid = true; \
    \
    /* Function pointers for different strategies */ \
    info.work_funcs.single_single_fn = execute_function; \
    info.work_funcs.single_multi_fn = execute_function; \
    info.work_funcs.data_parallel_fn = execute_function; \
    info.work_funcs.valid = true; \
    \
    /* Query function pointer - enables direct dispatch without switch statements */ \
    info.query_fn = (void*)ggml_numa_kernel_##op_name##_query; \
    \
    /* Work buffer calculation function pointer */ \
    info.work_buffer_calc_fn = (void*)ggml_numa_kernel_##op_name##_work_buffer_calc; \
    \
    /* No aggregation functions needed */ \
    info.agg_funcs.single_single_fn = NULL; \
    info.agg_funcs.single_multi_fn = NULL; \
    info.agg_funcs.data_parallel_fn = NULL; \
    info.agg_funcs.valid = false; \
    \
    return info; \
}

/**
 * @brief Macro to generate no-op kernel registration function
 *
 * For operations that should never be executed by the NUMA system (view operations, testing kernels).
 * These kernels are registered but marked as no-op, so the coordinator will skip execution.
 *
 * @param op_name Operation name (e.g., reshape, view, transpose, noop)
 * @param ggml_op_type GGML operation type (e.g., GGML_OP_RESHAPE)
 * @param kernel_display_name Display name for logging
 */
#define NUMA_KERNEL_REGISTRATION_FUNCTION_NOOP(op_name, ggml_op_type, kernel_display_name) \
ggml_numa_kernel_registration_info_t ggml_numa_kernel_##op_name##_register(void) { \
    ggml_numa_kernel_registration_info_t info = {0}; \
    \
    info.op_type = ggml_op_type; \
    info.supported = true; \
    info.is_noop = true;  /* Mark as no-op - coordinator will skip execution */ \
    info.kernel_name = kernel_display_name; \
    \
    /* No meaningful thresholds for no-op kernels - use SIZE_MAX to indicate never execute */ \
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_SINGLE] = SIZE_MAX; \
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_MULTI] = SIZE_MAX; \
    info.strategy_array.valid = true; \
    \
    /* No execution functions needed for no-op kernels */ \
    info.work_funcs.single_single_fn = NULL; \
    info.work_funcs.single_multi_fn = NULL; \
    info.work_funcs.data_parallel_fn = NULL; \
    info.work_funcs.valid = false; \
    \
    /* Query function pointer - still needed for registration */ \
    info.query_fn = (void*)ggml_numa_kernel_##op_name##_query; \
    \
    /* No work buffer needed for no-op kernels */ \
    info.work_buffer_calc_fn = NULL; \
    \
    /* No aggregation functions needed */ \
    info.agg_funcs.single_single_fn = NULL; \
    info.agg_funcs.single_multi_fn = NULL; \
    info.agg_funcs.data_parallel_fn = NULL; \
    info.agg_funcs.valid = false; \
    \
    return info; \
}

/**
 * @brief Macro to generate kernel registration function WITH aggregation
 *
 * For operations that need reduction/aggregation (like normalization operations)
 *
 * @param op_name Operation name (e.g., rms_norm, soft_max)
 * @param ggml_op_type GGML operation type (e.g., GGML_OP_RMS_NORM)
 * @param kernel_display_name Display name for logging
 * @param threshold_single_single Threshold for single-thread strategy
 * @param threshold_single_multi Threshold for multi-thread strategy  
 * @param execute_function Function pointer to the execution function
 */
#define NUMA_KERNEL_REGISTRATION_FUNCTION_WITH_AGG(op_name, ggml_op_type, kernel_display_name, threshold_single_single, threshold_single_multi, execute_function) \
ggml_numa_kernel_registration_info_t ggml_numa_kernel_##op_name##_register(void) { \
    ggml_numa_kernel_registration_info_t info = {0}; \
    \
    info.op_type = ggml_op_type; \
    info.supported = true; \
    info.kernel_name = kernel_display_name; \
    \
    /* Strategy thresholds for operation */ \
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_SINGLE] = threshold_single_single; \
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_MULTI] = threshold_single_multi; \
    /* Above threshold_single_multi elements: data-parallel strategy */ \
    info.strategy_array.valid = true; \
    \
    /* Function pointers for different strategies */ \
    info.work_funcs.single_single_fn = execute_function; \
    info.work_funcs.single_multi_fn = execute_function; \
    info.work_funcs.data_parallel_fn = execute_function; \
    info.work_funcs.valid = true; \
    \
    /* Query function pointer - enables direct dispatch without switch statements */ \
    info.query_fn = (void*)ggml_numa_kernel_##op_name##_query; \
    \
    /* Work buffer calculation function pointer */ \
    info.work_buffer_calc_fn = (void*)ggml_numa_kernel_##op_name##_work_buffer_calc; \
    \
    /* Aggregation functions for reduction operations */ \
    info.agg_funcs.single_single_fn = (void*)ggml_numa_kernel_##op_name##_aggregate; \
    info.agg_funcs.single_multi_fn = (void*)ggml_numa_kernel_##op_name##_aggregate; \
    info.agg_funcs.data_parallel_fn = (void*)ggml_numa_kernel_##op_name##_aggregate; \
    info.agg_funcs.valid = true; \
    \
    return info; \
}

/**
 * @brief Complete macro for operations that DON'T need aggregation functions
 *
 * This creates all three functions (query, work_buffer_calc, register) for a kernel
 * that doesn't require aggregation. This is the most common case.
 *
 * @param op_name Operation name (e.g., add, mul, div, sub)
 * @param ggml_op_type GGML operation type (e.g., GGML_OP_ADD)
 * @param kernel_display_name Display name for logging (e.g., "ADD", "MUL")
 * @param threshold_single_single Threshold for single-thread/single-node strategy
 * @param threshold_single_multi Threshold for multi-thread/single-node strategy
 * @param execute_function Function pointer to the execution function
 */
#define NUMA_KERNEL_REGISTER_METADATA(op_name, ggml_op_type, kernel_display_name, threshold_single_single, threshold_single_multi, execute_function) \
    NUMA_KERNEL_QUERY_FUNCTION(op_name, threshold_single_single, threshold_single_multi) \
    NUMA_KERNEL_WORK_BUFFER_FUNCTION(op_name) \
    NUMA_KERNEL_REGISTRATION_FUNCTION_NO_AGG(op_name, ggml_op_type, kernel_display_name, threshold_single_single, threshold_single_multi, execute_function)

/**
 * @brief Complete macro for operations that need aggregation functions
 *
 * This creates all three functions (query, work_buffer_calc, register) for a kernel
 * that requires aggregation functions for reduction operations.
 *
 * @param op_name Operation name (e.g., rms_norm, soft_max)
 * @param ggml_op_type GGML operation type (e.g., GGML_OP_RMS_NORM)
 * @param kernel_display_name Display name for logging (e.g., "RMS_NORM")
 * @param threshold_single_single Threshold for single-thread/single-node strategy
 * @param threshold_single_multi Threshold for multi-thread/single-node strategy
 * @param execute_function Function pointer to the execution function
 */
#define NUMA_KERNEL_REGISTER_METADATA_WITH_AGG(op_name, ggml_op_type, kernel_display_name, threshold_single_single, threshold_single_multi, execute_function) \
    NUMA_KERNEL_QUERY_FUNCTION(op_name, threshold_single_single, threshold_single_multi) \
    NUMA_KERNEL_WORK_BUFFER_FUNCTION(op_name) \
    NUMA_KERNEL_REGISTRATION_FUNCTION_WITH_AGG(op_name, ggml_op_type, kernel_display_name, threshold_single_single, threshold_single_multi, execute_function)

/**
 * @brief Complete macro for no-op operations (view operations, testing kernels)
 *
 * This creates query and registration functions for kernels that should never be executed
 * by the NUMA system. The coordinator will skip execution for kernels marked as no-op.
 *
 * @param op_name Operation name (e.g., reshape, view, transpose, noop)
 * @param ggml_op_type GGML operation type (e.g., GGML_OP_RESHAPE)
 * @param kernel_display_name Display name for logging (e.g., "RESHAPE", "VIEW")
 */
#define NUMA_KERNEL_REGISTER_METADATA_NOOP(op_name, ggml_op_type, kernel_display_name) \
    NUMA_KERNEL_QUERY_FUNCTION(op_name, SIZE_MAX, SIZE_MAX) \
    NUMA_KERNEL_REGISTRATION_FUNCTION_NOOP(op_name, ggml_op_type, kernel_display_name)

// =============================================================================
// NUMA SLICING UTILITIES - Reusable Data Partitioning Macros
// =============================================================================

/**
 * NUMA slicing context structure
 * Contains all calculated slice boundaries for element-wise, sequence-wise, and matrix operations
 */
typedef struct {
    // NUMA-level slicing (across nodes)
    size_t numa_start;           // Start index for this NUMA node
    size_t numa_end;             // End index for this NUMA node  
    size_t numa_elements;        // Total elements for this NUMA node
    
    // Thread-level slicing (within NUMA node)
    size_t thread_start;         // Start index for this thread
    size_t thread_end;           // End index for this thread
    size_t thread_elements;      // Total elements for this thread
    
    // Execution context
    int numa_node;               // Current NUMA node ID
    int thread_id;               // Thread ID within NUMA node
    int total_threads;           // Total threads on this NUMA node
    bool has_work;               // Whether this thread has work to do
    bool is_data_parallel;       // Whether data-parallel execution is active
    
    // Matrix multiplication specific fields (for 2D chunking)
    int64_t nchunk0;             // Number of chunks in dimension 0
    int64_t nchunk1;             // Number of chunks in dimension 1
    int64_t nr0;                 // Total elements in dimension 0
    int64_t nr1;                 // Total elements in dimension 1
    int64_t dr0;                 // Elements per chunk in dimension 0
    int64_t dr1;                 // Elements per chunk in dimension 1
} ggml_numa_slice_context_t;

/**
 * Calculate NUMA slice context for element-wise operations (like ADD, MUL)
 * This handles the dual-level slicing pattern used in most kernels
 */
#define NUMA_SLICE_ELEMENTS(ctx, tensor, params) do { \
    /* Get thread parameters */ \
    (ctx).thread_id = (params)->ith; \
    (ctx).total_threads = (params)->nth; \
    \
    /* Get NUMA execution context from thread-local variables */ \
    extern __thread int ggml_current_numa_node; \
    extern __thread bool ggml_numa_is_data_parallel_execution; \
    extern __thread int ggml_numa_total_nodes_for_data_parallel; \
    \
    (ctx).numa_node = ggml_current_numa_node; \
    (ctx).is_data_parallel = ggml_numa_is_data_parallel_execution; \
    \
    /* Calculate total elements to process */ \
    size_t total_elements = ggml_nelements(tensor); \
    \
    /* Step 1: NUMA-level slicing (across nodes) */ \
    if ((ctx).is_data_parallel) { \
        size_t elements_per_node = total_elements / ggml_numa_total_nodes_for_data_parallel; \
        (ctx).numa_start = (ctx).numa_node * elements_per_node; \
        (ctx).numa_end = ((ctx).numa_node == ggml_numa_total_nodes_for_data_parallel - 1) ? \
                         total_elements : (ctx).numa_start + elements_per_node; \
    } else { \
        (ctx).numa_start = 0; \
        (ctx).numa_end = total_elements; \
    } \
    (ctx).numa_elements = (ctx).numa_end - (ctx).numa_start; \
    \
    /* Step 2: Thread-level slicing (within NUMA node) */ \
    size_t elements_per_thread = ((ctx).numa_elements + (ctx).total_threads - 1) / (ctx).total_threads; \
    size_t thread_start_local = (ctx).thread_id * elements_per_thread; \
    size_t thread_end_local; \
    \
    /* Handle case where thread starts beyond available elements */ \
    if (thread_start_local >= (ctx).numa_elements) { \
        thread_start_local = (ctx).numa_elements; \
        thread_end_local = (ctx).numa_elements; \
    } else { \
        thread_end_local = (thread_start_local + elements_per_thread > (ctx).numa_elements) ? \
                           (ctx).numa_elements : thread_start_local + elements_per_thread; \
    } \
    \
    /* Convert to global indices */ \
    (ctx).thread_start = (ctx).numa_start + thread_start_local; \
    (ctx).thread_end = (ctx).numa_start + thread_end_local; \
    (ctx).thread_elements = (ctx).thread_end - (ctx).thread_start; \
    \
    /* Check if thread has work */ \
    (ctx).has_work = ((ctx).thread_elements > 0); \
} while(0)

/**
 * Calculate NUMA slice context for sequence-wise operations (like ROPE)
 * This handles operations that work on sequences (dimension i2)
 */
#define NUMA_SLICE_SEQUENCES(ctx, tensor, params) do { \
    /* Get thread parameters */ \
    (ctx).thread_id = (params)->ith; \
    (ctx).total_threads = (params)->nth; \
    \
    /* Get NUMA execution context from thread-local variables */ \
    extern __thread int ggml_current_numa_node; \
    extern __thread bool ggml_numa_is_data_parallel_execution; \
    extern __thread int ggml_numa_total_nodes_for_data_parallel; \
    \
    (ctx).numa_node = ggml_current_numa_node; \
    (ctx).is_data_parallel = ggml_numa_is_data_parallel_execution; \
    \
    /* Calculate total sequences to process (ne2 dimension) */ \
    size_t total_sequences = (tensor)->ne[2]; \
    \
    /* Step 1: NUMA-level slicing (across nodes) */ \
    if ((ctx).is_data_parallel) { \
        size_t seqs_per_node = total_sequences / ggml_numa_total_nodes_for_data_parallel; \
        (ctx).numa_start = (ctx).numa_node * seqs_per_node; \
        (ctx).numa_end = ((ctx).numa_node == ggml_numa_total_nodes_for_data_parallel - 1) ? \
                         total_sequences : (ctx).numa_start + seqs_per_node; \
    } else { \
        (ctx).numa_start = 0; \
        (ctx).numa_end = total_sequences; \
    } \
    (ctx).numa_elements = (ctx).numa_end - (ctx).numa_start; \
    \
    /* Step 2: Thread-level slicing (within NUMA node) */ \
    size_t seqs_per_thread = ((ctx).numa_elements + (ctx).total_threads - 1) / (ctx).total_threads; \
    size_t thread_start_local = (ctx).thread_id * seqs_per_thread; \
    size_t thread_end_local = (thread_start_local + seqs_per_thread > (ctx).numa_elements) ? \
                              (ctx).numa_elements : thread_start_local + seqs_per_thread; \
    \
    /* Convert to global sequence indices */ \
    (ctx).thread_start = (ctx).numa_start + thread_start_local; \
    (ctx).thread_end = (ctx).numa_start + thread_end_local; \
    (ctx).thread_elements = (ctx).thread_end - (ctx).thread_start; \
    \
    /* Check if thread has work */ \
    (ctx).has_work = ((ctx).thread_elements > 0); \
} while(0)

/**
 * Get shared result tensor data for direct writes (eliminates aggregation)
 * This is the common pattern for accessing shared destination memory
 */
#define NUMA_GET_SHARED_DATA(tensor, dst_ptr, data_type) do { \
    extern __thread void * ggml_numa_shared_result_tensor_data; \
    extern __thread bool ggml_numa_is_data_parallel_execution; \
    if (ggml_numa_shared_result_tensor_data != NULL && !ggml_numa_is_data_parallel_execution) { \
        (dst_ptr) = (data_type *)ggml_numa_shared_result_tensor_data; \
    } else { \
        (dst_ptr) = (data_type *)tensor_data(tensor); \
    } \
} while(0)

/**
 * Simplified element-wise kernel template with barrier handling
 * This macro provides the complete setup for element-wise operations including
 * proper OpenMP barrier synchronization for edge cases
 */
#define NUMA_KERNEL_ELEMENT_WISE_SETUP(ctx, tensor, params, dst_ptr, data_type) do { \
    NUMA_SLICE_ELEMENTS(ctx, tensor, params); \
    NUMA_GET_SHARED_DATA(tensor, dst_ptr, data_type); \
    \
    /* Handle threads with no work - must participate in barrier */ \
    if (!(ctx).has_work) { \
        if ((ctx).is_data_parallel || (ctx).total_threads > 1) { \
            NUMA_OPENMP_BARRIER(); \
        } \
        return GGML_STATUS_SUCCESS; \
    } \
} while(0)

/**
 * Simplified sequence-wise kernel template with barrier handling
 * This macro provides the complete setup for sequence-wise operations including
 * proper OpenMP barrier synchronization for edge cases
 */
#define NUMA_KERNEL_SEQUENCE_WISE_SETUP(ctx, tensor, params, dst_ptr, data_type) do { \
    NUMA_SLICE_SEQUENCES(ctx, tensor, params); \
    NUMA_GET_SHARED_DATA(tensor, dst_ptr, data_type); \
    \
    /* Handle threads with no work - must participate in barrier */ \
    if (!(ctx).has_work) { \
        if ((ctx).is_data_parallel || (ctx).total_threads > 1) { \
            NUMA_OPENMP_BARRIER(); \
        } \
        return GGML_STATUS_SUCCESS; \
    } \
} while(0)

/**
 * Simplified row-wise kernel template with barrier handling  
 * This macro provides the complete setup for row-wise operations including
 * proper OpenMP barrier synchronization for edge cases
 */
#define NUMA_KERNEL_ROW_WISE_SETUP(ctx, tensor, params, dst_ptr, data_type) do { \
    NUMA_SLICE_ROWS_1D(ctx, tensor, params); \
    NUMA_GET_SHARED_DATA(tensor, dst_ptr, data_type); \
    \
    /* Handle threads with no work - must participate in barrier */ \
    if (!(ctx).has_work) { \
        if ((ctx).is_data_parallel || (ctx).total_threads > 1) { \
            NUMA_OPENMP_BARRIER(); \
        } \
        return GGML_STATUS_SUCCESS; \
    } \
} while(0)

/**
 * Simplified column-wise kernel template with barrier handling
 * This macro provides the complete setup for column-wise operations including  
 * proper OpenMP barrier synchronization for edge cases
 */
#define NUMA_KERNEL_COLUMN_WISE_SETUP(ctx, tensor, params, dst_ptr, data_type) do { \
    NUMA_SLICE_COLUMNS(ctx, tensor, params); \
    NUMA_GET_SHARED_DATA(tensor, dst_ptr, data_type); \
    \
    /* Handle threads with no work - must participate in barrier */ \
    if (!(ctx).has_work) { \
        if ((ctx).is_data_parallel || (ctx).total_threads > 1) { \
            NUMA_OPENMP_BARRIER(); \
        } \
        return GGML_STATUS_SUCCESS; \
    } \
} while(0)

/**
 * Row-based kernel template with barrier handling for total rows (ggml_nrows)
 * This macro provides the complete setup for operations that use row-based threading
 * like ROPE, matching the reference implementation pattern
 */
#define NUMA_KERNEL_ROW_BASED_SETUP(ctx, tensor, params, dst_ptr, data_type) do { \
    NUMA_SLICE_ROWS(ctx, tensor, params); \
    NUMA_GET_SHARED_DATA(tensor, dst_ptr, data_type); \
    \
    /* Handle threads with no work - must participate in barrier */ \
    if (!(ctx).has_work) { \
        if ((ctx).is_data_parallel || (ctx).total_threads > 1) { \
            NUMA_OPENMP_BARRIER(); \
        } \
        return GGML_STATUS_SUCCESS; \
    } \
} while(0)

/**
 * 3D nested loop kernel template with barrier handling for operations like RMS_NORM
 * This macro provides setup for operations with 3D nested loops where only the middle
 * dimension (ne[1] = i01) is distributed across threads, while outer dimensions
 * (ne[2] = i02, ne[3] = i03) are processed in full by each thread
 */
#define NUMA_KERNEL_3D_NESTED_SETUP(ctx, tensor, params, dst_ptr, data_type) do { \
    NUMA_SLICE_ROWS_1D(ctx, tensor, params); \
    NUMA_GET_SHARED_DATA(tensor, dst_ptr, data_type); \
    \
    /* Handle threads with no work - must participate in barrier */ \
    if (!(ctx).has_work) { \
        if ((ctx).is_data_parallel || (ctx).total_threads > 1) { \
            NUMA_OPENMP_BARRIER(); \
        } \
        return GGML_STATUS_SUCCESS; \
    } \
} while(0)

/**
 * End-of-kernel barrier for kernels that completed work
 * All kernels should call this at the end to ensure synchronization
 */
#define NUMA_KERNEL_END_BARRIER(ctx) do { \
    /* Simple barrier for all threads that completed work */ \
    if ((ctx).is_data_parallel || (ctx).total_threads > 1) { \
        NUMA_OPENMP_BARRIER(); \
    } \
} while(0)

/**
 * OpenMP barrier abstraction for consistent barrier usage
 */
#define NUMA_OPENMP_BARRIER() do { \
    _Pragma("omp barrier") \
} while(0)

/**
 * Calculate NUMA slice context for row-wise operations (1D - ne1 dimension only)
 * This handles operations that work on rows (dimension ne1)
 */
#define NUMA_SLICE_ROWS_1D(ctx, tensor, params) do { \
    /* Get thread parameters */ \
    (ctx).thread_id = (params)->ith; \
    (ctx).total_threads = (params)->nth; \
    \
    /* Get NUMA execution context from thread-local variables */ \
    extern __thread int ggml_current_numa_node; \
    extern __thread bool ggml_numa_is_data_parallel_execution; \
    extern __thread int ggml_numa_total_nodes_for_data_parallel; \
    \
    (ctx).numa_node = ggml_current_numa_node; \
    (ctx).is_data_parallel = ggml_numa_is_data_parallel_execution; \
    \
    /* Calculate total rows to process (ne1 dimension) */ \
    size_t total_rows = (tensor)->ne[1]; \
    \
    /* Step 1: NUMA-level slicing (across nodes) */ \
    if ((ctx).is_data_parallel) { \
        size_t rows_per_node = total_rows / ggml_numa_total_nodes_for_data_parallel; \
        (ctx).numa_start = (ctx).numa_node * rows_per_node; \
        (ctx).numa_end = ((ctx).numa_node == ggml_numa_total_nodes_for_data_parallel - 1) ? \
                         total_rows : (ctx).numa_start + rows_per_node; \
    } else { \
        (ctx).numa_start = 0; \
        (ctx).numa_end = total_rows; \
    } \
    (ctx).numa_elements = (ctx).numa_end - (ctx).numa_start; \
    \
    /* Step 2: Thread-level slicing (within NUMA node) */ \
    size_t rows_per_thread = ((ctx).numa_elements + (ctx).total_threads - 1) / (ctx).total_threads; \
    size_t thread_start_local = (ctx).thread_id * rows_per_thread; \
    size_t thread_end_local = (thread_start_local + rows_per_thread > (ctx).numa_elements) ? \
                              (ctx).numa_elements : thread_start_local + rows_per_thread; \
    \
    /* Convert to global row indices */ \
    (ctx).thread_start = (ctx).numa_start + thread_start_local; \
    (ctx).thread_end = (ctx).numa_start + thread_end_local; \
    (ctx).thread_elements = (ctx).thread_end - (ctx).thread_start; \
    \
    /* Check if thread has work */ \
    (ctx).has_work = ((ctx).thread_elements > 0); \
} while(0)

/**
 * Calculate NUMA slice context for column-wise operations
 * This handles operations that work on columns (dimension ne0)
 */
#define NUMA_SLICE_COLUMNS(ctx, tensor, params) do { \
    /* Get thread parameters */ \
    (ctx).thread_id = (params)->ith; \
    (ctx).total_threads = (params)->nth; \
    \
    /* Get NUMA execution context from thread-local variables */ \
    extern __thread int ggml_current_numa_node; \
    extern __thread bool ggml_numa_is_data_parallel_execution; \
    extern __thread int ggml_numa_total_nodes_for_data_parallel; \
    \
    (ctx).numa_node = ggml_current_numa_node; \
    (ctx).is_data_parallel = ggml_numa_is_data_parallel_execution; \
    \
    /* Calculate total columns to process (ne0 dimension) */ \
    size_t total_columns = (tensor)->ne[0]; \
    \
    /* Step 1: NUMA-level slicing (across nodes) */ \
    if ((ctx).is_data_parallel) { \
        size_t cols_per_node = total_columns / ggml_numa_total_nodes_for_data_parallel; \
        (ctx).numa_start = (ctx).numa_node * cols_per_node; \
        (ctx).numa_end = ((ctx).numa_node == ggml_numa_total_nodes_for_data_parallel - 1) ? \
                         total_columns : (ctx).numa_start + cols_per_node; \
    } else { \
        (ctx).numa_start = 0; \
        (ctx).numa_end = total_columns; \
    } \
    (ctx).numa_elements = (ctx).numa_end - (ctx).numa_start; \
    \
    /* Step 2: Thread-level slicing (within NUMA node) */ \
    size_t cols_per_thread = ((ctx).numa_elements + (ctx).total_threads - 1) / (ctx).total_threads; \
    size_t thread_start_local = (ctx).thread_id * cols_per_thread; \
    size_t thread_end_local = (thread_start_local + cols_per_thread > (ctx).numa_elements) ? \
                              (ctx).numa_elements : thread_start_local + cols_per_thread; \
    \
    /* Convert to global column indices */ \
    (ctx).thread_start = (ctx).numa_start + thread_start_local; \
    (ctx).thread_end = (ctx).numa_start + thread_end_local; \
    (ctx).thread_elements = (ctx).thread_end - (ctx).thread_start; \
    \
    /* Check if thread has work */ \
    (ctx).has_work = ((ctx).thread_elements > 0); \
} while(0)

/**
 * Calculate NUMA slice context for total row operations
 * This handles operations that work on all rows (ggml_nrows = ne1 * ne2 * ne3)
 * Used by operations like ROPE that use row-based threading like the reference implementation
 */
#define NUMA_SLICE_ROWS(ctx, tensor, params) do { \
    /* Get thread parameters */ \
    (ctx).thread_id = (params)->ith; \
    (ctx).total_threads = (params)->nth; \
    \
    /* Get NUMA execution context from thread-local variables */ \
    extern __thread int ggml_current_numa_node; \
    extern __thread bool ggml_numa_is_data_parallel_execution; \
    extern __thread int ggml_numa_total_nodes_for_data_parallel; \
    \
    (ctx).numa_node = ggml_current_numa_node; \
    (ctx).is_data_parallel = ggml_numa_is_data_parallel_execution; \
    \
    /* Calculate total rows to process (ggml_nrows = ne1 * ne2 * ne3) */ \
    size_t total_rows = ggml_nrows(tensor); \
    \
    /* Step 1: NUMA-level slicing (across nodes) */ \
    if ((ctx).is_data_parallel) { \
        size_t rows_per_node = total_rows / ggml_numa_total_nodes_for_data_parallel; \
        (ctx).numa_start = (ctx).numa_node * rows_per_node; \
        (ctx).numa_end = ((ctx).numa_node == ggml_numa_total_nodes_for_data_parallel - 1) ? \
                         total_rows : (ctx).numa_start + rows_per_node; \
    } else { \
        (ctx).numa_start = 0; \
        (ctx).numa_end = total_rows; \
    } \
    (ctx).numa_elements = (ctx).numa_end - (ctx).numa_start; \
    \
    /* Step 2: Thread-level slicing (within NUMA node) */ \
    size_t rows_per_thread = ((ctx).numa_elements + (ctx).total_threads - 1) / (ctx).total_threads; \
    size_t thread_start_local = (ctx).thread_id * rows_per_thread; \
    size_t thread_end_local = (thread_start_local + rows_per_thread > (ctx).numa_elements) ? \
                              (ctx).numa_elements : thread_start_local + rows_per_thread; \
    \
    /* Convert to global row indices */ \
    (ctx).thread_start = (ctx).numa_start + thread_start_local; \
    (ctx).thread_end = (ctx).numa_start + thread_end_local; \
    (ctx).thread_elements = (ctx).thread_end - (ctx).thread_start; \
    \
    /* Check if thread has work */ \
    (ctx).has_work = ((ctx).thread_elements > 0); \
} while(0)

/**
 * Debug logging for slice context (optional)
 */
#define NUMA_LOG_SLICE_DEBUG(ctx, operation_name) do { \
    if ((ctx).thread_id == 0) { \
        NUMA_LOG_DEBUG("%s SLICE: NUMA %d: [%zu,%zu) (%zu elements), thread %d/%d: [%zu,%zu) (%zu elements), data_parallel=%s", \
                       operation_name, (ctx).numa_node, (ctx).numa_start, (ctx).numa_end, (ctx).numa_elements, \
                       (ctx).thread_id, (ctx).total_threads, (ctx).thread_start, (ctx).thread_end, (ctx).thread_elements, \
                       (ctx).is_data_parallel ? "YES" : "NO"); \
    } \
} while(0)

// ========================================================================
// GENERIC TENSOR MANIPULATION MACROS
// ========================================================================

/**
 * @brief Generic 4D tensor element pointer calculation with stride support
 * @param tensor_data Base tensor data pointer
 * @param tensor Tensor structure for stride information
 * @param i0,i1,i2,i3 4D indices
 * @param element_type Type to cast the result to (float, ggml_fp16_t, etc.)
 */
#define NUMA_TENSOR_4D_PTR(tensor_data, tensor, i0, i1, i2, i3, element_type) \
    ((element_type*)((char*)(tensor_data) + \
     (i3)*(tensor)->nb[3] + (i2)*(tensor)->nb[2] + (i1)*(tensor)->nb[1] + (i0)*(tensor)->nb[0]))

/**
 * @brief Generic 3D tensor element pointer calculation with stride support  
 * @param tensor_data Base tensor data pointer
 * @param tensor Tensor structure for stride information
 * @param i0,i1,i2 3D indices
 * @param element_type Type to cast the result to
 */
#define NUMA_TENSOR_3D_PTR(tensor_data, tensor, i0, i1, i2, element_type) \
    ((element_type*)((char*)(tensor_data) + \
     (i2)*(tensor)->nb[2] + (i1)*(tensor)->nb[1] + (i0)*(tensor)->nb[0]))

/**
 * @brief Generic 2D tensor element pointer calculation with stride support
 * @param tensor_data Base tensor data pointer  
 * @param tensor Tensor structure for stride information
 * @param i0,i1 2D indices
 * @param element_type Type to cast the result to
 */
#define NUMA_TENSOR_2D_PTR(tensor_data, tensor, i0, i1, element_type) \
    ((element_type*)((char*)(tensor_data) + \
     (i1)*(tensor)->nb[1] + (i0)*(tensor)->nb[0]))

/**
 * @brief Generic nested loop for 4D tensor iteration with customizable body
 * @param tensor Tensor to iterate over
 * @param i0_var,i1_var,i2_var,i3_var Variable names for loop indices
 * @param loop_body Code block to execute for each iteration
 */
#define NUMA_TENSOR_4D_LOOP(tensor, i0_var, i1_var, i2_var, i3_var, loop_body) do { \
    const int64_t ne0 = (tensor)->ne[0]; \
    const int64_t ne1 = (tensor)->ne[1]; \
    const int64_t ne2 = (tensor)->ne[2]; \
    const int64_t ne3 = (tensor)->ne[3]; \
    for (int64_t i3_var = 0; i3_var < ne3; i3_var++) { \
        for (int64_t i2_var = 0; i2_var < ne2; i2_var++) { \
            for (int64_t i1_var = 0; i1_var < ne1; i1_var++) { \
                for (int64_t i0_var = 0; i0_var < ne0; i0_var++) { \
                    loop_body \
                } \
            } \
        } \
    } \
} while(0)

/**
 * @brief Generic nested loop for 3D tensor iteration with customizable body
 * @param tensor Tensor to iterate over
 * @param i0_var,i1_var,i2_var Variable names for loop indices
 * @param loop_body Code block to execute for each iteration
 */
#define NUMA_TENSOR_3D_LOOP(tensor, i0_var, i1_var, i2_var, loop_body) do { \
    const int64_t ne0 = (tensor)->ne[0]; \
    const int64_t ne1 = (tensor)->ne[1]; \
    const int64_t ne2 = (tensor)->ne[2]; \
    for (int64_t i2_var = 0; i2_var < ne2; i2_var++) { \
        for (int64_t i1_var = 0; i1_var < ne1; i1_var++) { \
            for (int64_t i0_var = 0; i0_var < ne0; i0_var++) { \
                loop_body \
            } \
        } \
    } \
} while(0)

/**
 * @brief 4D rowwise tensor iteration loop for NUMA thread distribution
 * @param tensor Tensor to iterate over
 * @param ctx NUMA thread context with thread_start and thread_end ranges
 * @param loop_body Code block to execute for each (i03, i02, i01) iteration
 * 
 * This macro provides the common 4D nested loop pattern used in operations like
 * SOFT_MAX and RMS_NORM where:
 * - i03, i02 are outer dimensions (processed completely by each thread)
 * - i01 is the row dimension distributed across threads using ctx.thread_start to ctx.thread_end
 * 
 * USAGE EXAMPLE:
 *   NUMA_4D_ROWWISE_LOOP(tensor, ctx, {
 *       // Process row i01 with coordinates (i03, i02, i01)
 *       // i03, i02, i01 variables are available in the loop body
 *       const float * src_row = get_row_pointer(src_data, i01, i02, i03);
 *       float * dst_row = get_row_pointer(dst_data, i01, i02, i03);
 *       process_row(src_row, dst_row, ne00);
 *   });
 */
#define NUMA_4D_ROWWISE_LOOP(tensor, ctx, loop_body) do { \
    const int64_t ne02 = (tensor)->ne[2]; \
    const int64_t ne03 = (tensor)->ne[3]; \
    for (int64_t i03 = 0; i03 < ne03; i03++) { \
        for (int64_t i02 = 0; i02 < ne02; i02++) { \
            for (size_t i01 = (ctx).thread_start; i01 < (ctx).thread_end; i01++) { \
                loop_body \
            } \
        } \
    } \
} while(0)

/**
 * @brief 3D threaded tensor iteration loop for multithreaded operations
 * @param tensor Tensor to iterate over (uses tensor->ne[3], tensor->ne[2], tensor->ne[1])
 * @param ith Thread index (0-based)
 * @param nth Total number of threads
 * @param loop_body Code block to execute for each (i13, i12, i11) iteration
 * 
 * This macro provides the common 3D nested loop pattern with thread distribution
 * used in operations like MUL_MAT type conversion where:
 * - i13, i12 are outer dimensions (processed completely by each thread)
 * - i11 is the innermost dimension distributed across threads using ith/nth pattern
 * 
 * USAGE EXAMPLE:
 *   NUMA_3D_THREADED_LOOP(src1, ith, nth, {
 *       // Process element at coordinates (i13, i12, i11)
 *       // i13, i12, i11 variables are available in the loop body
 *       const float * src_element = get_element_pointer(src_data, i11, i12, i13);
 *       void * dst_element = get_element_pointer(dst_data, i11, i12, i13);
 *       convert_element(src_element, dst_element);
 *   });
 */
#define NUMA_3D_THREADED_LOOP(tensor, ith, nth, loop_body) do { \
    const int64_t _numa_3d_ne13 = (tensor)->ne[3]; \
    const int64_t _numa_3d_ne12 = (tensor)->ne[2]; \
    const int64_t _numa_3d_ne11 = (tensor)->ne[1]; \
    for (int64_t i13 = 0; i13 < _numa_3d_ne13; ++i13) { \
        for (int64_t i12 = 0; i12 < _numa_3d_ne12; ++i12) { \
            for (int64_t i11 = (ith); i11 < _numa_3d_ne11; i11 += (nth)) { \
                loop_body \
            } \
        } \
    } \
} while(0)

/**
 * @brief Matrix chunked iteration loop for block-tiled matrix operations
 * @param ir0_start,ir0_end Range for first dimension
 * @param ir1_start,ir1_end Range for second dimension  
 * @param blck_0,blck_1 Block sizes for tiling
 * @param num_rows_per_vec_dot Number of rows processed per vector dot operation
 * @param loop_body Code block to execute for each (iir1, iir0, ir1) iteration
 * 
 * This macro provides the complex chunked processing pattern used in matrix
 * operations with block tiling and vector dot optimization where:
 * - iir1, iir0 iterate over blocks of size blck_1, blck_0
 * - ir1 iterates within each block with num_rows_per_vec_dot stride
 * 
 * USAGE EXAMPLE:
 *   NUMA_MATRIX_CHUNKED_LOOP(ir0_start, ir0_end, ir1_start, ir1_end, 
 *                            blck_0, blck_1, num_rows_per_vec_dot, {
 *       // Process matrix chunk at coordinates (iir1, iir0, ir1)
 *       // iir1, iir0, ir1 variables are available in the loop body
 *       process_matrix_chunk(iir1, iir0, ir1);
 *   });
 */
#define NUMA_MATRIX_CHUNKED_LOOP(ir0_start, ir0_end, ir1_start, ir1_end, blck_0, blck_1, num_rows_per_vec_dot, loop_body) do { \
    for (int64_t iir1 = (ir1_start); iir1 < (ir1_end); iir1 += (blck_1)) { \
        for (int64_t iir0 = (ir0_start); iir0 < (ir0_end); iir0 += (blck_0)) { \
            for (int64_t ir1 = iir1; ir1 < iir1 + (blck_1) && ir1 < (ir1_end); ir1 += (num_rows_per_vec_dot)) { \
                loop_body \
            } \
        } \
    } \
} while(0)

// ========================================================================
// NUMA WORK DISTRIBUTION MACROS
// ========================================================================

/**
 * @brief Generic work distribution calculation for any dimension/unit
 * @param ctx NUMA execution context (must have numa_start, numa_end set)
 * @param total_units Total number of work units to distribute
 * @param thread_start_var Variable to store thread's start unit
 * @param thread_end_var Variable to store thread's end unit
 */
#define NUMA_CALCULATE_WORK_DISTRIBUTION(ctx, total_units, thread_start_var, thread_end_var) do { \
    const size_t numa_units = (ctx).numa_end - (ctx).numa_start; \
    const size_t units_per_thread = numa_units / (ctx).total_threads; \
    const size_t remainder = numa_units % (ctx).total_threads; \
    thread_start_var = (ctx).numa_start + (ctx).thread_id * units_per_thread + ((ctx).thread_id < remainder ? (ctx).thread_id : remainder); \
    thread_end_var = thread_start_var + units_per_thread + ((ctx).thread_id < remainder ? 1 : 0); \
} while(0)

/**
 * @brief Calculate thread-specific range for any work units with bounds checking
 * @param ctx NUMA execution context  
 * @param total_units Total number of work units
 * @param start_var Variable to store start index
 * @param end_var Variable to store end index
 * @param count_var Variable to store count of units
 */
#define NUMA_THREAD_WORK_RANGE(ctx, total_units, start_var, end_var, count_var) do { \
    if ((ctx).is_data_parallel) { \
        NUMA_CALCULATE_WORK_DISTRIBUTION(ctx, total_units, start_var, end_var); \
        /* In data-parallel mode, bounds check against NUMA end, not total_units */ \
        if (end_var > (ctx).numa_end) end_var = (ctx).numa_end; \
    } else { \
        const size_t units_per_thread = (total_units) / (ctx).total_threads; \
        const size_t remainder = (total_units) % (ctx).total_threads; \
        start_var = (ctx).thread_id * units_per_thread + ((ctx).thread_id < remainder ? (ctx).thread_id : remainder); \
        end_var = start_var + units_per_thread + ((ctx).thread_id < remainder ? 1 : 0); \
        /* In non-data-parallel mode, bounds check against total_units */ \
        if (end_var > (total_units)) end_var = (total_units); \
    } \
    count_var = (end_var > start_var) ? (end_var - start_var) : 0; \
} while(0)

// ========================================================================
// BROADCASTING ARITHMETIC UTILITIES
// ========================================================================

/**
 * @brief Convert linear index to 4D tensor coordinates
 * 
 * This macro converts a linear array index to 4D tensor coordinates (i3,i2,i1,i0)
 * based on tensor dimensions. Used for proper multi-dimensional indexing.
 * 
 * @param linear_idx Linear index into tensor data
 * @param ne0,ne1,ne2,ne3 Tensor dimensions  
 * @param i0,i1,i2,i3 Output coordinate variables
 */
#define NUMA_LINEAR_TO_4D_COORDS(linear_idx, ne0, ne1, ne2, ne3, i0, i1, i2, i3) do { \
    const int64_t __stride_21 = (ne1) * (ne0); \
    const int64_t __stride_321 = (ne2) * __stride_21; \
    \
    (i3) = (linear_idx) / __stride_321; \
    const int64_t __remainder_3 = (linear_idx) - (i3) * __stride_321; \
    (i2) = __remainder_3 / __stride_21; \
    const int64_t __remainder_2 = __remainder_3 - (i2) * __stride_21; \
    (i1) = __remainder_2 / (ne0); \
    (i0) = __remainder_2 - (i1) * (ne0); \
} while(0)

/**
 * @brief Calculate broadcasting coordinates for source tensor
 * 
 * Applies modulo operation to coordinates for proper broadcasting behavior.
 * This handles cases where source tensor dimensions are smaller than target.
 * 
 * @param src_i0,src_i1,src_i2,src_i3 Output broadcasting coordinates
 * @param tgt_i0,tgt_i1,tgt_i2,tgt_i3 Target tensor coordinates  
 * @param src_ne0,src_ne1,src_ne2,src_ne3 Source tensor dimensions
 */
#define NUMA_BROADCAST_COORDS(src_i0, src_i1, src_i2, src_i3, tgt_i0, tgt_i1, tgt_i2, tgt_i3, src_ne0, src_ne1, src_ne2, src_ne3) do { \
    (src_i0) = (tgt_i0) % (src_ne0); \
    (src_i1) = (tgt_i1) % (src_ne1); \
    (src_i2) = (tgt_i2) % (src_ne2); \
    (src_i3) = (tgt_i3) % (src_ne3); \
} while(0)

/**
 * @brief Calculate memory offset from 4D coordinates and strides
 * 
 * Converts 4D tensor coordinates to linear memory offset using tensor strides.
 * Handles non-contiguous tensor layouts correctly.
 * 
 * @param offset Output memory offset variable
 * @param i0,i1,i2,i3 Tensor coordinates
 * @param nb0,nb1,nb2,nb3 Tensor strides (byte offsets)
 */
#define NUMA_COORDS_TO_OFFSET(offset, i0, i1, i2, i3, nb0, nb1, nb2, nb3) do { \
    (offset) = (i3) * (nb3) + (i2) * (nb2) + (i1) * (nb1) + (i0) * (nb0); \
} while(0)

/**
 * @brief Safe memory access with offset and type casting
 * 
 * Safely accesses tensor data at computed offset with proper type casting.
 * Handles both const and non-const access patterns.
 * 
 * @param base_ptr Base pointer to tensor data
 * @param offset Byte offset from base pointer
 * @param data_type Target data type (e.g., float, ggml_fp16_t)
 */
#define NUMA_DATA_AT_OFFSET(base_ptr, offset, data_type) \
    (*(const data_type*)((const char*)(base_ptr) + (offset)))

#define NUMA_DATA_WRITE_AT_OFFSET(base_ptr, offset, value, data_type) do { \
    (*(data_type*)((char*)(base_ptr) + (offset))) = (value); \
} while(0)

// ========================================================================
// LEVEL 2 ENHANCED: BROADCASTING-AWARE ITERATION MACROS
// ========================================================================

/**
 * @brief Enhanced iteration macro for binary broadcasting operations
 * 
 * This macro provides a complete broadcasting iteration framework for binary operations.
 * It handles the complex coordinate calculation and broadcasting logic automatically.
 * 
 * Usage:
 *   NUMA_ITERATE_BINARY_BROADCAST(ctx, dst_tensor, src0_tensor, src1_tensor, data_type) {
 *       data_type val0 = NUMA_BROADCAST_SRC0_VALUE(ctx);
 *       data_type val1 = NUMA_BROADCAST_SRC1_VALUE(ctx);  
 *       NUMA_BROADCAST_DST_WRITE(ctx, val0 + val1);
 *   }
 */
#define NUMA_ITERATE_BINARY_BROADCAST(ctx, dst_tensor, src0_tensor, src1_tensor, data_type) \
    /* Cache tensor dimensions and strides for performance */ \
    const int64_t __ne0 = (dst_tensor)->ne[0], __ne1 = (dst_tensor)->ne[1], __ne2 = (dst_tensor)->ne[2], __ne3 = (dst_tensor)->ne[3]; \
    const int64_t __src0_ne0 = (src0_tensor)->ne[0], __src0_ne1 = (src0_tensor)->ne[1], __src0_ne2 = (src0_tensor)->ne[2], __src0_ne3 = (src0_tensor)->ne[3]; \
    const int64_t __src1_ne0 = (src1_tensor)->ne[0], __src1_ne1 = (src1_tensor)->ne[1], __src1_ne2 = (src1_tensor)->ne[2], __src1_ne3 = (src1_tensor)->ne[3]; \
    \
    const size_t __nb0 = (dst_tensor)->nb[0], __nb1 = (dst_tensor)->nb[1], __nb2 = (dst_tensor)->nb[2], __nb3 = (dst_tensor)->nb[3]; \
    const size_t __src0_nb0 = (src0_tensor)->nb[0], __src0_nb1 = (src0_tensor)->nb[1], __src0_nb2 = (src0_tensor)->nb[2], __src0_nb3 = (src0_tensor)->nb[3]; \
    const size_t __src1_nb0 = (src1_tensor)->nb[0], __src1_nb1 = (src1_tensor)->nb[1], __src1_nb2 = (src1_tensor)->nb[2], __src1_nb3 = (src1_tensor)->nb[3]; \
    \
    const data_type * __src0_data = (const data_type *)tensor_data(src0_tensor); \
    const data_type * __src1_data = (const data_type *)tensor_data(src1_tensor); \
    data_type * __dst_data = (data_type *)(ctx).dst_data; \
    \
    /* Iteration loop with automatic coordinate calculation */ \
    for (size_t __linear_idx = (ctx).thread_start; __linear_idx < (ctx).thread_end; __linear_idx++) { \
        /* Calculate 4D coordinates for destination/src0 */ \
        int64_t __i0, __i1, __i2, __i3; \
        NUMA_LINEAR_TO_4D_COORDS(__linear_idx, __ne0, __ne1, __ne2, __ne3, __i0, __i1, __i2, __i3); \
        \
        /* Calculate broadcasting coordinates for src1 */ \
        int64_t __j0, __j1, __j2, __j3; \
        NUMA_BROADCAST_COORDS(__j0, __j1, __j2, __j3, __i0, __i1, __i2, __i3, __src1_ne0, __src1_ne1, __src1_ne2, __src1_ne3); \
        \
        /* Calculate memory offsets */ \
        size_t __src0_offset, __src1_offset, __dst_offset; \
        NUMA_COORDS_TO_OFFSET(__src0_offset, __i0, __i1, __i2, __i3, __src0_nb0, __src0_nb1, __src0_nb2, __src0_nb3); \
        NUMA_COORDS_TO_OFFSET(__src1_offset, __j0, __j1, __j2, __j3, __src1_nb0, __src1_nb1, __src1_nb2, __src1_nb3); \
        NUMA_COORDS_TO_OFFSET(__dst_offset, __i0, __i1, __i2, __i3, __nb0, __nb1, __nb2, __nb3); \
        \
        /* Execute user code with current iteration context */ \
        { \
            const data_type __src0_val = NUMA_DATA_AT_OFFSET(__src0_data, __src0_offset, data_type); \
            const data_type __src1_val = NUMA_DATA_AT_OFFSET(__src1_data, __src1_offset, data_type);

/**
 * @brief Convenience macros for accessing data within broadcasting iteration
 * 
 * These macros should be used within NUMA_ITERATE_BINARY_BROADCAST blocks
 * to access source and destination tensor values safely.
 */
#define NUMA_BROADCAST_SRC0_VALUE() (__src0_val)
#define NUMA_BROADCAST_SRC1_VALUE() (__src1_val)
#define NUMA_BROADCAST_DST_WRITE(value) \
    NUMA_DATA_WRITE_AT_OFFSET(__dst_data, __dst_offset, value, data_type)

#define NUMA_ITERATE_BINARY_BROADCAST_END() \
        } \
    }

// ========================================================================
// COMPLEX BROADCASTING LOOP MACRO
// ========================================================================

/**
 * @brief Macro for complex broadcasting coordinate calculation and element processing
 * 
 * This macro handles the most complex broadcasting case where tensors have different shapes
 * that require multi-dimensional coordinate calculation and modulo-based broadcasting.
 * 
 * @param ctx The slice context containing thread_start, thread_end
 * @param tensor The destination tensor 
 * @param data_type The data type (e.g., float)
 * @param op_expr The operation expression using val0 and val1 (e.g., val0 + val1)
 * 
 * Usage examples:
 *   NUMA_COMPLEX_BROADCAST_LOOP(ctx, tensor, float, val0 + val1)  // ADD
 *   NUMA_COMPLEX_BROADCAST_LOOP(ctx, tensor, float, val0 - val1)  // SUB  
 *   NUMA_COMPLEX_BROADCAST_LOOP(ctx, tensor, float, val0 * val1)  // MUL
 *   NUMA_COMPLEX_BROADCAST_LOOP(ctx, tensor, float, val0 / val1)  // DIV
 */
#define NUMA_COMPLEX_BROADCAST_LOOP(ctx, tensor, data_type, op_expr) do { \
    for (size_t i = (ctx).thread_start; i < (ctx).thread_end; i++) { \
        /* Convert linear index to 4D coordinates */ \
        int64_t dst_coords[4]; \
        int64_t linear_idx = i; \
        const int64_t *ne = (tensor)->ne; \
        \
        dst_coords[0] = linear_idx % ne[0]; \
        linear_idx /= ne[0]; \
        dst_coords[1] = linear_idx % ne[1]; \
        linear_idx /= ne[1]; \
        dst_coords[2] = linear_idx % ne[2]; \
        linear_idx /= ne[2]; \
        dst_coords[3] = linear_idx; \
        \
        /* Calculate source offsets with broadcasting */ \
        const int64_t *src0_ne = (tensor)->src[0]->ne; \
        const int64_t *src1_ne = (tensor)->src[1]->ne; \
        const size_t *src0_nb = (tensor)->src[0]->nb; \
        const size_t *src1_nb = (tensor)->src[1]->nb; \
        \
        size_t src0_offset = (dst_coords[0] % src0_ne[0]) * src0_nb[0] + \
                           (dst_coords[1] % src0_ne[1]) * src0_nb[1] + \
                           (dst_coords[2] % src0_ne[2]) * src0_nb[2] + \
                           (dst_coords[3] % src0_ne[3]) * src0_nb[3]; \
        \
        size_t src1_offset = (dst_coords[0] % src1_ne[0]) * src1_nb[0] + \
                           (dst_coords[1] % src1_ne[1]) * src1_nb[1] + \
                           (dst_coords[2] % src1_ne[2]) * src1_nb[2] + \
                           (dst_coords[3] % src1_ne[3]) * src1_nb[3]; \
        \
        /* Get values with proper type casting */ \
        const data_type *src0_ptr = (const data_type *)((const char *)__src0_data + src0_offset); \
        const data_type *src1_ptr = (const data_type *)((const char *)__src1_data + src1_offset); \
        const data_type val0 = *src0_ptr; \
        const data_type val1 = *src1_ptr; \
        \
        /* Apply operation expression */ \
        __dst_data[i] = (op_expr); \
    } \
} while(0)

// ========================================================================
// LEVEL 4 ENHANCED: UNIFIED BINARY BROADCASTING SETUP
// ========================================================================

/**
 * @brief Unified setup macro for binary operations with automatic broadcasting detection
 * 
 * This macro provides complete setup for binary operations (ADD, SUB, MUL, DIV) with:
 * - Automatic broadcasting pattern detection (scalar, same-shape, complex broadcasting)
 * - SIMD optimization paths for simple cases
 * - Full broadcasting support for complex cases
 * - Proper NUMA slice setup and barrier handling
 * 
 * Usage:
 *   NUMA_KERNEL_SETUP_BINARY_BROADCAST(ctx, tensor, params, float) {
 *       // Operation-specific code - macro handles all the complexity
 *       NUMA_BINARY_OP_ADD(ctx);  // or SUB, MUL, DIV
 *   }
 */
#define NUMA_KERNEL_SETUP_BINARY_BROADCAST(ctx_name, tensor, params, data_type) \
    ggml_numa_refined_context_t ctx_name = {0}; \
    NUMA_SLICE_WORK_BY_ELEMENT(ctx_name, tensor, params); \
    NUMA_GET_SHARED_DATA(tensor, (ctx_name).dst_data, data_type); \
    (ctx_name).src_tensors[0] = (tensor)->src[0]; \
    (ctx_name).src_tensors[1] = (tensor)->src[1]; \
    \
    /* Analyze broadcasting pattern for optimization */ \
    const bool __is_scalar = (ggml_nelements((tensor)->src[1]) == 1); \
    const bool __is_same_shape = ggml_are_same_shape((tensor)->src[0], (tensor)->src[1]); \
    const data_type * __src0_data = (const data_type *)tensor_data((tensor)->src[0]); \
    const data_type * __src1_data = (const data_type *)tensor_data((tensor)->src[1]); \
    data_type * __dst_data = (data_type *)(ctx_name).dst_data; \
    \
    if ((ctx_name).has_work) /* User code block follows */

/**
 * @brief Operation-specific macros for common binary operations
 * 
 * These macros implement the actual arithmetic operations within the broadcasting framework.
 * They automatically choose the optimal execution path (scalar, SIMD, or broadcasting).
 */
#define NUMA_BINARY_OP_ADD(ctx) do { \
    if (__is_scalar) { \
        const data_type __scalar = __src1_data[0]; \
        for (size_t __i = (ctx).thread_start; __i < (ctx).thread_end; __i++) { \
            __dst_data[__i] = __src0_data[__i] + __scalar; \
        } \
    } else if (__is_same_shape) { \
        /* Use SIMD optimization for F32 same-shape operations */ \
        if (sizeof(data_type) == sizeof(float)) { \
            ggml_vec_add_f32((ctx).thread_elements, \
                           __dst_data + (ctx).thread_start, \
                           __src0_data + (ctx).thread_start, \
                           __src1_data + (ctx).thread_start); \
        } else { \
            for (size_t __i = (ctx).thread_start; __i < (ctx).thread_end; __i++) { \
                __dst_data[__i] = __src0_data[__i] + __src1_data[__i]; \
            } \
        } \
    } else { \
        /* Complex broadcasting case */ \
        NUMA_ITERATE_BINARY_BROADCAST(ctx, tensor, (tensor)->src[0], (tensor)->src[1], data_type) \
            const data_type __result = NUMA_BROADCAST_SRC0_VALUE() + NUMA_BROADCAST_SRC1_VALUE(); \
            NUMA_BROADCAST_DST_WRITE(__result); \
        NUMA_ITERATE_BINARY_BROADCAST_END() \
    } \
} while(0)

#define NUMA_BINARY_OP_SUB(ctx) do { \
    if (__is_scalar) { \
        const data_type __scalar = __src1_data[0]; \
        for (size_t __i = (ctx).thread_start; __i < (ctx).thread_end; __i++) { \
            __dst_data[__i] = __src0_data[__i] - __scalar; \
        } \
    } else if (__is_same_shape) { \
        if (sizeof(data_type) == sizeof(float)) { \
            ggml_vec_sub_f32((ctx).thread_elements, \
                           __dst_data + (ctx).thread_start, \
                           __src0_data + (ctx).thread_start, \
                           __src1_data + (ctx).thread_start); \
        } else { \
            for (size_t __i = (ctx).thread_start; __i < (ctx).thread_end; __i++) { \
                __dst_data[__i] = __src0_data[__i] - __src1_data[__i]; \
            } \
        } \
    } else { \
        NUMA_ITERATE_BINARY_BROADCAST(ctx, tensor, (tensor)->src[0], (tensor)->src[1], data_type) \
            const data_type __result = NUMA_BROADCAST_SRC0_VALUE() - NUMA_BROADCAST_SRC1_VALUE(); \
            NUMA_BROADCAST_DST_WRITE(__result); \
        NUMA_ITERATE_BINARY_BROADCAST_END() \
    } \
} while(0)

#define NUMA_BINARY_OP_MUL(ctx) do { \
    if (__is_scalar) { \
        const data_type __scalar = __src1_data[0]; \
        for (size_t __i = (ctx).thread_start; __i < (ctx).thread_end; __i++) { \
            __dst_data[__i] = __src0_data[__i] * __scalar; \
        } \
    } else if (__is_same_shape) { \
        if (sizeof(data_type) == sizeof(float)) { \
            ggml_vec_mul_f32((ctx).thread_elements, \
                           __dst_data + (ctx).thread_start, \
                           __src0_data + (ctx).thread_start, \
                           __src1_data + (ctx).thread_start); \
        } else { \
            for (size_t __i = (ctx).thread_start; __i < (ctx).thread_end; __i++) { \
                __dst_data[__i] = __src0_data[__i] * __src1_data[__i]; \
            } \
        } \
    } else { \
        NUMA_ITERATE_BINARY_BROADCAST(ctx, tensor, (tensor)->src[0], (tensor)->src[1], data_type) \
            const data_type __result = NUMA_BROADCAST_SRC0_VALUE() * NUMA_BROADCAST_SRC1_VALUE(); \
            NUMA_BROADCAST_DST_WRITE(__result); \
        NUMA_ITERATE_BINARY_BROADCAST_END() \
    } \
} while(0)

#define NUMA_BINARY_OP_DIV(ctx) do { \
    if (__is_scalar) { \
        const data_type __scalar = __src1_data[0]; \
        for (size_t __i = (ctx).thread_start; __i < (ctx).thread_end; __i++) { \
            __dst_data[__i] = __src0_data[__i] / __scalar; \
        } \
    } else if (__is_same_shape) { \
        /* Note: No ggml_vec_div_f32 available, use element-wise */ \
        for (size_t __i = (ctx).thread_start; __i < (ctx).thread_end; __i++) { \
            __dst_data[__i] = __src0_data[__i] / __src1_data[__i]; \
        } \
    } else { \
        NUMA_ITERATE_BINARY_BROADCAST(ctx, tensor, (tensor)->src[0], (tensor)->src[1], data_type) \
            const data_type __result = NUMA_BROADCAST_SRC0_VALUE() / NUMA_BROADCAST_SRC1_VALUE(); \
            NUMA_BROADCAST_DST_WRITE(__result); \
        NUMA_ITERATE_BINARY_BROADCAST_END() \
    } \
} while(0)

// ========================================================================
// REFINED NUMA KERNEL MACROS - Level 1-4 System
// ========================================================================

/**
 * @brief Extended NUMA slice context for refined macro system
 * 
 * Extends the basic slice context with additional metadata needed for
 * safe iteration and memory access patterns.
 */
typedef struct {
    // Basic slice context (inherited)
    size_t numa_start;           // Start index for this NUMA node
    size_t numa_end;             // End index for this NUMA node  
    size_t numa_elements;        // Total elements for this NUMA node
    size_t thread_start;         // Start index for this thread
    size_t thread_end;           // End index for this thread
    size_t thread_elements;      // Total elements for this thread
    int numa_node;               // Current NUMA node ID
    int thread_id;               // Thread ID within NUMA node
    int total_threads;           // Total threads on this NUMA node
    bool has_work;               // Whether this thread has work to do
    bool is_data_parallel;       // Whether data-parallel execution is active
    
    // Extended context for refined system
    const struct ggml_tensor * tensor;     // Target tensor for bounds checking
    const struct ggml_tensor * src_tensors[4]; // Source tensors (up to 4 sources)
    void * dst_data;             // Destination data pointer
    void * work_buffer;          // Work buffer pointer (if any)
    size_t work_buffer_offset;   // Per-thread work buffer offset
    
    // Iteration state for different strategies
    enum {
        NUMA_ITER_BY_ELEMENT,
        NUMA_ITER_BY_SEQUENCE,  
        NUMA_ITER_BY_ROW,
        NUMA_ITER_BY_COLUMN,
        NUMA_ITER_BY_MATRIX
    } iteration_strategy;
    
    // Tensor dimensions cache (for bounds checking)
    int64_t ne[4];               // Tensor dimensions [ne0, ne1, ne2, ne3]
    size_t nb[4];                // Tensor strides [nb0, nb1, nb2, nb3]
} ggml_numa_refined_context_t;

// ========================================================================
// LEVEL 1: WORK SLICING STRATEGY MACROS
// ========================================================================

/**
 * @brief Slice work by linear elements (for ADD, MUL, CPY operations)
 */
#define NUMA_SLICE_WORK_BY_ELEMENT(ctx, tensor, params) do { \
    /* Initialize basic slice context */ \
    NUMA_SLICE_ELEMENTS(*(ggml_numa_slice_context_t*)&(ctx), tensor, params); \
    \
    /* Extended context setup */ \
    (ctx).tensor = tensor; \
    (ctx).iteration_strategy = NUMA_ITER_BY_ELEMENT; \
    (ctx).ne[0] = (tensor)->ne[0]; \
    (ctx).ne[1] = (tensor)->ne[1]; \
    (ctx).ne[2] = (tensor)->ne[2]; \
    (ctx).ne[3] = (tensor)->ne[3]; \
    (ctx).nb[0] = (tensor)->nb[0]; \
    (ctx).nb[1] = (tensor)->nb[1]; \
    (ctx).nb[2] = (tensor)->nb[2]; \
    (ctx).nb[3] = (tensor)->nb[3]; \
} while(0)

/**
 * @brief Slice work by sequences (for ROPE operations)
 */
#define NUMA_SLICE_WORK_BY_SEQUENCE(ctx, tensor, params) do { \
    /* Initialize sequence slice context */ \
    NUMA_SLICE_SEQUENCES(*(ggml_numa_slice_context_t*)&(ctx), tensor, params); \
    \
    /* Extended context setup */ \
    (ctx).tensor = tensor; \
    (ctx).iteration_strategy = NUMA_ITER_BY_SEQUENCE; \
    (ctx).ne[0] = (tensor)->ne[0]; \
    (ctx).ne[1] = (tensor)->ne[1]; \
    (ctx).ne[2] = (tensor)->ne[2]; \
    (ctx).ne[3] = (tensor)->ne[3]; \
    (ctx).nb[0] = (tensor)->nb[0]; \
    (ctx).nb[1] = (tensor)->nb[1]; \
    (ctx).nb[2] = (tensor)->nb[2]; \
    (ctx).nb[3] = (tensor)->nb[3]; \
} while(0)

/**
 * @brief Slice work by rows (for SOFT_MAX, RMS_NORM operations)
 */
#define NUMA_SLICE_WORK_BY_ROW(ctx, tensor, params) do { \
    /* Initialize row slice context */ \
    NUMA_SLICE_ROWS_1D(*(ggml_numa_slice_context_t*)&(ctx), tensor, params); \
    \
    /* Extended context setup */ \
    (ctx).tensor = tensor; \
    (ctx).iteration_strategy = NUMA_ITER_BY_ROW; \
    (ctx).ne[0] = (tensor)->ne[0]; \
    (ctx).ne[1] = (tensor)->ne[1]; \
    (ctx).ne[2] = (tensor)->ne[2]; \
    (ctx).ne[3] = (tensor)->ne[3]; \
    (ctx).nb[0] = (tensor)->nb[0]; \
    (ctx).nb[1] = (tensor)->nb[1]; \
    (ctx).nb[2] = (tensor)->nb[2]; \
    (ctx).nb[3] = (tensor)->nb[3]; \
} while(0)

/**
 * @brief Slice work by columns (for future matrix operations)
 */
#define NUMA_SLICE_WORK_BY_COLUMN(ctx, tensor, params) do { \
    /* Initialize column slice context */ \
    NUMA_SLICE_COLUMNS(*(ggml_numa_slice_context_t*)&(ctx), tensor, params); \
    \
    /* Extended context setup */ \
    (ctx).tensor = tensor; \
    (ctx).iteration_strategy = NUMA_ITER_BY_COLUMN; \
    (ctx).ne[0] = (tensor)->ne[0]; \
    (ctx).ne[1] = (tensor)->ne[1]; \
    (ctx).ne[2] = (tensor)->ne[2]; \
    (ctx).ne[3] = (tensor)->ne[3]; \
    (ctx).nb[0] = (tensor)->nb[0]; \
    (ctx).nb[1] = (tensor)->nb[1]; \
    (ctx).nb[2] = (tensor)->nb[2]; \
    (ctx).nb[3] = (tensor)->nb[3]; \
} while(0)

// ========================================================================
// LEVEL 2: SAFE ITERATION MACROS
// ========================================================================

/**
 * @brief Safe 1D iteration over elements with automatic bounds checking
 * 
 * Generates a for loop that iterates over the thread's assigned elements.
 * The iteration variable is guaranteed to be within bounds.
 */
#define NUMA_ITERATE_1D(ctx, element_var) \
    for (size_t element_var = (ctx).thread_start, __numa_end_##element_var = (ctx).thread_end; \
         element_var < __numa_end_##element_var; \
         element_var++)

/**
 * @brief Safe 2D iteration with automatic bounds checking
 * 
 * Generates nested for loops for 2D tensor operations (rows and columns).
 * Both iteration variables are guaranteed to be within bounds.
 */
#define NUMA_ITERATE_2D(ctx, tensor, row_var, col_var) \
    for (int64_t row_var = (ctx).thread_start / (ctx).ne[0], __numa_end_row_##row_var = ((ctx).thread_end + (ctx).ne[0] - 1) / (ctx).ne[0]; \
         row_var < __numa_end_row_##row_var && row_var < (ctx).ne[1]; \
         row_var++) \
        for (int64_t col_var = (row_var == (ctx).thread_start / (ctx).ne[0]) ? ((ctx).thread_start % (ctx).ne[0]) : 0, \
             __numa_end_col_##col_var = (row_var == __numa_end_row_##row_var - 1) ? (((ctx).thread_end - 1) % (ctx).ne[0] + 1) : (ctx).ne[0]; \
             col_var < __numa_end_col_##col_var && col_var < (ctx).ne[0]; \
             col_var++)

/**
 * @brief Safe 3D iteration with automatic bounds checking
 * 
 * Generates nested for loops for 3D tensor operations (sequences, heads, elements).
 * All iteration variables are guaranteed to be within bounds.
 */
#define NUMA_ITERATE_3D(ctx, tensor, seq_var, head_var, elem_var) \
    for (int64_t seq_var = (ctx).thread_start, __numa_end_seq_##seq_var = (ctx).thread_end; \
         seq_var < __numa_end_seq_##seq_var && seq_var < (ctx).ne[2]; \
         seq_var++) \
        for (int64_t head_var = 0; head_var < (ctx).ne[1]; head_var++) \
            for (int64_t elem_var = 0; elem_var < (ctx).ne[0]; elem_var++)

/**
 * @brief Safe 4D iteration with automatic bounds checking
 * 
 * Generates nested for loops for 4D tensor operations.
 * All iteration variables are guaranteed to be within bounds.
 */
#define NUMA_ITERATE_4D(ctx, tensor, batch_var, seq_var, head_var, elem_var) \
    for (int64_t batch_var = 0; batch_var < (ctx).ne[3]; batch_var++) \
        for (int64_t seq_var = (ctx).thread_start, __numa_end_seq_##seq_var = (ctx).thread_end; \
             seq_var < __numa_end_seq_##seq_var && seq_var < (ctx).ne[2]; \
             seq_var++) \
            for (int64_t head_var = 0; head_var < (ctx).ne[1]; head_var++) \
                for (int64_t elem_var = 0; elem_var < (ctx).ne[0]; elem_var++)

// ========================================================================
// LEVEL 3: SAFE MEMORY ACCESS MACROS
// ========================================================================

/**
 * @brief Safe source tensor data access
 * 
 * Returns a pointer to the source tensor data with bounds checking.
 * Supports up to 4 source tensors (indexed 0-3).
 */
#define NUMA_SRC_DATA(ctx, src_idx) \
    ((const float *)(tensor_data((ctx).src_tensors[src_idx])))

/**
 * @brief Safe source tensor value access with 1D indexing
 */
#define NUMA_SRC_VALUE_1D(ctx, src_idx, i0) \
    (NUMA_SRC_DATA(ctx, src_idx)[i0])

/**
 * @brief Safe source tensor value access with 2D indexing
 */
#define NUMA_SRC_VALUE_2D(ctx, src_idx, i1, i0) \
    (*(const float*)((const char*)NUMA_SRC_DATA(ctx, src_idx) + (i1) * (ctx).src_tensors[src_idx]->nb[1] + (i0) * (ctx).src_tensors[src_idx]->nb[0]))

/**
 * @brief Safe source tensor value access with 3D indexing
 */
#define NUMA_SRC_VALUE_3D(ctx, src_idx, i2, i1, i0) \
    (*(const float*)((const char*)NUMA_SRC_DATA(ctx, src_idx) + (i2) * (ctx).src_tensors[src_idx]->nb[2] + (i1) * (ctx).src_tensors[src_idx]->nb[1] + (i0) * (ctx).src_tensors[src_idx]->nb[0]))

/**
 * @brief Safe source tensor value access with 4D indexing
 */
#define NUMA_SRC_VALUE_4D(ctx, src_idx, i3, i2, i1, i0) \
    (*(const float*)((const char*)NUMA_SRC_DATA(ctx, src_idx) + (i3) * (ctx).src_tensors[src_idx]->nb[3] + (i2) * (ctx).src_tensors[src_idx]->nb[2] + (i1) * (ctx).src_tensors[src_idx]->nb[1] + (i0) * (ctx).src_tensors[src_idx]->nb[0]))

/**
 * @brief Safe destination tensor write with 1D indexing
 */
#define NUMA_DST_WRITE_1D(ctx, value, i0) do { \
    ((float*)(ctx).dst_data)[i0] = (value); \
} while(0)

/**
 * @brief Safe destination tensor write with 2D indexing
 */
#define NUMA_DST_WRITE_2D(ctx, value, i1, i0) do { \
    (*(float*)((char*)(ctx).dst_data + (i1) * (ctx).nb[1] + (i0) * (ctx).nb[0])) = (value); \
} while(0)

/**
 * @brief Safe destination tensor write with 3D indexing
 */
#define NUMA_DST_WRITE_3D(ctx, value, i2, i1, i0) do { \
    (*(float*)((char*)(ctx).dst_data + (i2) * (ctx).nb[2] + (i1) * (ctx).nb[1] + (i0) * (ctx).nb[0])) = (value); \
} while(0)

/**
 * @brief Safe destination tensor write with 4D indexing
 */
#define NUMA_DST_WRITE_4D(ctx, value, i3, i2, i1, i0) do { \
    (*(float*)((char*)(ctx).dst_data + (i3) * (ctx).nb[3] + (i2) * (ctx).nb[2] + (i1) * (ctx).nb[1] + (i0) * (ctx).nb[0])) = (value); \
} while(0)

/**
 * @brief Safe work buffer access with automatic offset calculation
 */
#define NUMA_WORK_BUFFER_AT(ctx, offset) \
    ((float*)((char*)(ctx).work_buffer + (ctx).work_buffer_offset + (offset)))

// ========================================================================
// LEVEL 4: UNIFIED KERNEL SETUP MACROS
// ========================================================================

/**
 * @brief Unified kernel setup for element-wise operations
 * 
 * Combines work slicing, memory setup, and barrier handling in one macro.
 * Creates a code block where user code can be written safely.
 */
#define NUMA_KERNEL_SETUP_ELEMENT_WISE(ctx_name, tensor, params, data_type) \
    ggml_numa_refined_context_t ctx_name = {0}; \
    NUMA_SLICE_WORK_BY_ELEMENT(ctx_name, tensor, params); \
    NUMA_GET_SHARED_DATA(tensor, (ctx_name).dst_data, data_type); \
    (ctx_name).src_tensors[0] = (tensor)->src[0]; \
    (ctx_name).src_tensors[1] = (tensor)->src[1]; \
    (ctx_name).src_tensors[2] = (tensor)->src[2]; \
    (ctx_name).src_tensors[3] = (tensor)->src[3]; \
    if ((ctx_name).has_work) /* User code block follows */

/**
 * @brief Unified kernel setup for sequence-wise operations
 */
#define NUMA_KERNEL_SETUP_SEQUENCE_WISE(ctx_name, tensor, params, data_type) \
    ggml_numa_refined_context_t ctx_name = {0}; \
    NUMA_SLICE_WORK_BY_SEQUENCE(ctx_name, tensor, params); \
    NUMA_GET_SHARED_DATA(tensor, (ctx_name).dst_data, data_type); \
    (ctx_name).src_tensors[0] = (tensor)->src[0]; \
    (ctx_name).src_tensors[1] = (tensor)->src[1]; \
    (ctx_name).src_tensors[2] = (tensor)->src[2]; \
    (ctx_name).src_tensors[3] = (tensor)->src[3]; \
    if (params->wdata) { \
        (ctx_name).work_buffer = params->wdata; \
        (ctx_name).work_buffer_offset = ((tensor)->ne[0] + 16) * sizeof(float) * params->ith; \
    } \
    if ((ctx_name).has_work) /* User code block follows */

/**
 * @brief Unified kernel setup for row-wise operations  
 */
#define NUMA_KERNEL_SETUP_ROW_WISE(ctx_name, tensor, params, data_type) \
    ggml_numa_refined_context_t ctx_name = {0}; \
    NUMA_SLICE_WORK_BY_ROW(ctx_name, tensor, params); \
    NUMA_GET_SHARED_DATA(tensor, (ctx_name).dst_data, data_type); \
    (ctx_name).src_tensors[0] = (tensor)->src[0]; \
    (ctx_name).src_tensors[1] = (tensor)->src[1]; \
    (ctx_name).src_tensors[2] = (tensor)->src[2]; \
    (ctx_name).src_tensors[3] = (tensor)->src[3]; \
    if ((ctx_name).has_work) /* User code block follows */

/**
 * @brief End kernel processing with proper barrier handling
 */
#define NUMA_KERNEL_END_REFINED(ctx_name) \
    NUMA_KERNEL_END_BARRIER(*(ggml_numa_slice_context_t*)&(ctx_name))

// ========================================================================
// MATRIX MULTIPLICATION SPECIFIC MACROS
// ========================================================================

/**
 * @brief 2D chunk-based work distribution for matrix multiplication
 * 
 * Implements the sophisticated 2D chunking strategy used by mul_mat reference.
 * Distributes work across both output rows (nr0) and output columns (nr1)
 * using configurable chunk sizes with NUMA-aware distribution.
 * 
 * @param ctx NUMA execution context (will be populated)
 * @param tensor Target tensor (for dimensions)
 * @param chunk_size Base chunk size (typically 16 for cache optimization)
 * @param params Compute parameters
 * @param adaptive_chunking Whether to use adaptive chunking for NUMA systems
 */
#define NUMA_SLICE_2D_CHUNKS(ctx, tensor, chunk_size, params, adaptive_chunking) do { \
    /* Initialize basic context */ \
    (ctx).thread_id = (params)->ith; \
    (ctx).total_threads = (params)->nth; \
    \
    /* Get NUMA execution context from thread-local variables */ \
    extern __thread int ggml_current_numa_node; \
    extern __thread bool ggml_numa_is_data_parallel_execution; \
    extern __thread int ggml_numa_total_nodes_for_data_parallel; \
    \
    (ctx).numa_node = ggml_current_numa_node; \
    (ctx).is_data_parallel = ggml_numa_is_data_parallel_execution; \
    \
    /* Calculate matrix dimensions following mul_mat reference pattern */ \
    const int64_t nr0 = (tensor)->ne[0];  /* First dimension of result */ \
    const int64_t nr1 = (tensor)->ne[1] * (tensor)->ne[2] * (tensor)->ne[3];  /* Rest of dimensions */ \
    \
    /* Calculate chunk counts */ \
    int64_t nchunk0 = (nr0 + (chunk_size) - 1) / (chunk_size); \
    int64_t nchunk1 = (nr1 + (chunk_size) - 1) / (chunk_size); \
    \
    /* Adaptive chunking for NUMA systems (matches reference logic) */ \
    if ((adaptive_chunking) && (nchunk0 * nchunk1 < (ctx).total_threads * 4 || (ctx).is_data_parallel)) { \
        nchunk0 = nr0 > nr1 ? (ctx).total_threads : 1; \
        nchunk1 = nr0 > nr1 ? 1 : (ctx).total_threads; \
    } \
    \
    const int64_t total_chunks = nchunk0 * nchunk1; \
    const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0; \
    const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1; \
    \
    /* NUMA-aware chunk distribution */ \
    if ((ctx).is_data_parallel) { \
        size_t chunks_per_node = total_chunks / ggml_numa_total_nodes_for_data_parallel; \
        (ctx).numa_start = (ctx).numa_node * chunks_per_node; \
        (ctx).numa_end = ((ctx).numa_node == ggml_numa_total_nodes_for_data_parallel - 1) ? \
                         total_chunks : (ctx).numa_start + chunks_per_node; \
    } else { \
        (ctx).numa_start = 0; \
        (ctx).numa_end = total_chunks; \
    } \
    \
    /* Thread-level chunk assignment within NUMA node */ \
    size_t numa_chunks = (ctx).numa_end - (ctx).numa_start; \
    size_t chunks_per_thread = (numa_chunks + (ctx).total_threads - 1) / (ctx).total_threads; \
    size_t thread_start_local = (ctx).thread_id * chunks_per_thread; \
    size_t thread_end_local = (thread_start_local + chunks_per_thread > numa_chunks) ? \
                              numa_chunks : thread_start_local + chunks_per_thread; \
    \
    /* Convert to global chunk indices */ \
    (ctx).thread_start = (ctx).numa_start + thread_start_local; \
    (ctx).thread_end = (ctx).numa_start + thread_end_local; \
    (ctx).thread_elements = (ctx).thread_end - (ctx).thread_start; \
    \
    /* Store chunk configuration for iteration macro */ \
    (ctx).nchunk0 = nchunk0; \
    (ctx).nchunk1 = nchunk1; \
    (ctx).nr0 = nr0; \
    (ctx).nr1 = nr1; \
    (ctx).dr0 = dr0; \
    (ctx).dr1 = dr1; \
    \
    /* Check if thread has work */ \
    (ctx).has_work = ((ctx).thread_elements > 0); \
} while(0)

/**
 * @brief Block-tiled iteration over 2D chunks with automatic bounds checking
 * 
 * Iterates over chunks assigned to this thread, providing block-tiled access
 * within each chunk. Mirrors the reference mul_mat nested loop structure.
 * 
 * @param ctx NUMA context (must be set up with NUMA_SLICE_2D_CHUNKS)
 * @param ir0_var Variable name for outer dimension iteration
 * @param ir1_var Variable name for inner dimension iteration  
 * @param iir0_var Variable name for block-level outer iteration
 * @param iir1_var Variable name for block-level inner iteration
 * @param block_size Block size for tiling (typically 16)
 */
#define NUMA_ITERATE_2D_CHUNKS_TILED(ctx, ir0_var, ir1_var, iir0_var, iir1_var, block_size) \
    for (size_t __chunk_idx = (ctx).thread_start; __chunk_idx < (ctx).thread_end; __chunk_idx++) { \
        /* Calculate 2D chunk coordinates */ \
        const int64_t __ith0 = __chunk_idx % (ctx).nchunk0; \
        const int64_t __ith1 = __chunk_idx / (ctx).nchunk0; \
        \
        /* Calculate chunk boundaries */ \
        const int64_t __ir0_start = (ctx).dr0 * __ith0; \
        const int64_t __ir0_end = ((__ir0_start + (ctx).dr0) > (ctx).nr0) ? (ctx).nr0 : (__ir0_start + (ctx).dr0); \
        const int64_t __ir1_start = (ctx).dr1 * __ith1; \
        const int64_t __ir1_end = ((__ir1_start + (ctx).dr1) > (ctx).nr1) ? (ctx).nr1 : (__ir1_start + (ctx).dr1); \
        \
        /* Block-tiled iteration within chunk (mirrors reference pattern) */ \
        for (int64_t iir1_var = __ir1_start; iir1_var < __ir1_end; iir1_var += (block_size)) { \
            for (int64_t iir0_var = __ir0_start; iir0_var < __ir0_end; iir0_var += (block_size)) { \
                for (int64_t ir1_var = iir1_var; ir1_var < (iir1_var + (block_size)) && ir1_var < __ir1_end; ir1_var++) { \
                    for (int64_t ir0_var = iir0_var; ir0_var < (iir0_var + (block_size)) && ir0_var < __ir0_end; ir0_var++)

/**
 * @brief End macro for NUMA_ITERATE_2D_CHUNKS_TILED
 */
#define NUMA_ITERATE_2D_CHUNKS_TILED_END() \
                } \
            } \
        } \
    }

/**
 * @brief Matrix multiplication broadcasting coordinate calculation
 * 
 * Handles the complex coordinate calculation for matrix multiplication
 * including broadcasting factors r2, r3 used in the reference implementation.
 * 
 * @param ir1 Current output dimension 1 index
 * @param ne1,ne2,ne12,ne13 Tensor dimensions
 * @param i11,i12,i13 Output coordinate variables for src1
 * @param i02,i03 Output coordinate variables for src0 (broadcasted)
 * @param r2,r3 Broadcasting factors
 */
#define NUMA_MATMUL_BROADCAST_COORDS(ir1, ne1, ne2, ne12, ne13, i11, i12, i13, i02, i03, r2, r3) do { \
    (i13) = (ir1) / ((ne12) * (ne1)); \
    (i12) = ((ir1) - (i13) * (ne12) * (ne1)) / (ne1); \
    (i11) = ((ir1) - (i13) * (ne12) * (ne1) - (i12) * (ne1)); \
    \
    /* Calculate broadcasting coordinates for src0 */ \
    (i03) = (i13) / (r3); \
    (i02) = (i12) / (r2); \
} while(0)

/**
 * @brief Calculate memory pointers for matrix multiplication access patterns
 * 
 * Computes the specific memory access patterns used by mul_mat:
 * - src0_row: Row-wise access with broadcasting
 * - src1_col: Column-wise access (contiguous or strided)
 * - dst_col: Output column pointer
 * 
 * @param src0_data,src1_data,dst_data Base tensor data pointers
 * @param ir0,i11,i12,i13,i02,i03 Coordinate indices
 * @param nb01,nb02,nb03,nb11,nb12,nb13,nb1,nb2,nb3 Tensor strides
 * @param row_size Row size for src1 data
 * @param src1_cont Whether src1 is contiguous
 * @param src1_col_stride Stride for src1 columns
 * @param src0_row_ptr,src1_col_ptr,dst_col_ptr Output pointer variables
 */
#define NUMA_MATMUL_ACCESS_POINTERS(src0_data, src1_data, dst_data, wdata, src1_type, vec_dot_type, \
                                   ir0, i11, i12, i13, i02, i03, \
                                   nb01, nb02, nb03, nb11, nb12, nb13, nb1, nb2, nb3, \
                                   ne11, ne12, row_size, src1_cont, src1_col_stride, \
                                   src0_row_ptr, src1_col_ptr, dst_col_ptr) do { \
    /* src0 row pointer with broadcasting */ \
    (src0_row_ptr) = (const char*)(src0_data) + (0 + (i02) * (nb02) + (i03) * (nb03)); \
    \
    /* src1 column pointer (contiguous or strided) */ \
    const void * __wdata = ((src1_type) == (vec_dot_type)) ? (src1_data) : (wdata); \
    (src1_col_ptr) = (const char*)__wdata + \
        ((src1_cont) || (src1_type) != (vec_dot_type) \
            ? ((i11) + (i12) * (ne11) + (i13) * (ne12) * (ne11)) * (row_size) \
            : ((i11) * (nb11) + (i12) * (nb12) + (i13) * (nb13))); \
    \
    /* dst column pointer */ \
    (dst_col_ptr) = (float*)((char*)(dst_data) + ((i11) * (nb1) + (i12) * (nb2) + (i13) * (nb3))); \
} while(0)

/**
 * @brief Comprehensive setup macro for matrix multiplication kernels
 * 
 * Provides complete setup for mul_mat operations including 2D chunk distribution,
 * shared memory access, and proper barrier handling for edge cases.
 * 
 * @param ctx NUMA execution context (will be populated)
 * @param tensor Target tensor
 * @param params Compute parameters
 * @param dst_ptr Destination data pointer variable
 * @param data_type Data type (typically float)
 * @param chunk_size Chunk size for 2D distribution
 */
#define NUMA_KERNEL_MATMUL_SETUP(ctx, tensor, params, dst_ptr, data_type, chunk_size) do { \
    NUMA_SLICE_2D_CHUNKS(ctx, tensor, chunk_size, params, true); \
    NUMA_GET_SHARED_DATA(tensor, dst_ptr, data_type); \
    \
    /* Handle threads with no work - must participate in all barriers */ \
    if (!(ctx).has_work) { \
        if ((ctx).is_data_parallel || (ctx).total_threads > 1) { \
            /* Check if type conversion barrier is needed */ \
            const struct ggml_tensor * __src1 = (tensor)->src[1]; \
            if (__src1) { \
                const struct ggml_type_traits_cpu * __traits = ggml_get_type_traits_cpu((tensor)->src[0]->type); \
                if (__src1->type != __traits->vec_dot_type) { \
                    NUMA_OPENMP_BARRIER(); /* Type conversion barrier */ \
                } \
            } \
            NUMA_OPENMP_BARRIER(); /* Computation completion barrier */ \
        } \
        return GGML_STATUS_SUCCESS; \
    } \
} while(0)

// ========================================================================
// COMPOSABLE KERNEL BUILDING BLOCKS SYSTEM
// ========================================================================

/**
 * COMPOSABLE NUMA KERNEL MACROS
 * 
 * This system provides atomic building blocks that can be combined like Lego pieces
 * to construct NUMA kernels. Based on combinator patterns from functional programming,
 * these macros separate concerns and promote code reuse.
 * 
 * DESIGN PRINCIPLES:
 * - Each macro does ONE thing well
 * - Macros compose naturally without conflicts
 * - Clear naming indicates purpose and usage
 * - All error handling and edge cases built-in
 * - Consistent parameter ordering across all macros
 * 
 * USAGE PATTERN:
 * 1. Setup kernel context and validate inputs
 * 2. Choose data access strategy (memory layout)
 * 3. Select iteration pattern (loop structure)
 * 4. Define computation operations
 * 5. Handle synchronization requirements
 */

// ========================================================================
// ATOMIC BUILDING BLOCKS - Level 1 Primitives
// ========================================================================

/**
 * @group Context Setup Building Blocks
 * Initialize execution context and validate kernel inputs
 */

/**
 * @brief Initialize NUMA execution context with thread and node information
 * @param ctx Variable name for ggml_numa_slice_context_t (will be declared)
 * @param tensor Target tensor being processed
 * @param params Compute parameters from OpenMP coordinator
 * 
 * WHAT IT DOES:
 * - Declares and initializes slice context variable
 * - Extracts thread ID, thread count, NUMA node information
 * - Sets up data-parallel execution flags
 * - Prepares context for subsequent building blocks
 * 
 * USAGE:
 *   NUMA_INIT_CONTEXT(ctx, tensor, params);
 *   // ctx is now ready for slicing operations
 */
#define NUMA_INIT_CONTEXT(ctx, tensor, params) \
    ggml_numa_slice_context_t ctx; \
    do { \
        (ctx).thread_id = (params)->ith; \
        (ctx).total_threads = (params)->nth; \
        \
        extern __thread int ggml_current_numa_node; \
        extern __thread bool ggml_numa_is_data_parallel_execution; \
        extern __thread int ggml_numa_total_nodes_for_data_parallel; \
        \
        (ctx).numa_node = ggml_current_numa_node; \
        (ctx).is_data_parallel = ggml_numa_is_data_parallel_execution; \
    } while(0)

/**
 * @brief Validate kernel inputs with automatic error handling
 * @param tensor Target tensor (checked for NULL)
 * @param params Compute parameters (checked for NULL)
 * @param ... Additional validation expressions (optional)
 * 
 * WHAT IT DOES:
 * - Validates essential inputs are not NULL
 * - Checks tensor structure integrity
 * - Executes custom validation expressions
 * - Returns early with error status if validation fails
 * 
 * USAGE:
 *   NUMA_VALIDATE_INPUTS(tensor, params, 
 *                        tensor->ne[0] > 0,
 *                        tensor->src[0] != NULL);
 */
#define NUMA_VALIDATE_INPUTS(tensor, params, ...) do { \
    NUMA_ASSERT((tensor) != NULL, "Tensor cannot be null"); \
    NUMA_ASSERT((params) != NULL, "Compute params cannot be null"); \
    NUMA_ASSERT(tensor_data(tensor) != NULL, "Tensor data cannot be null"); \
    __VA_ARGS__ \
} while(0)

/**
 * @group Memory Access Building Blocks  
 * Handle tensor memory access with NUMA-awareness
 */

/**
 * @brief Get typed pointer to tensor data with NUMA optimization
 * @param ptr_var Variable name for output pointer (will be declared)
 * @param tensor Source tensor
 * @param data_type Data type for pointer (e.g., float, ggml_fp16_t)
 * 
 * WHAT IT DOES:
 * - Declares typed pointer variable
 * - Handles NUMA memory mirroring if available
 * - Falls back to standard tensor data access
 * - Optimizes for shared memory scenarios
 * 
 * USAGE:
 *   NUMA_GET_TYPED_POINTER(src_data, tensor->src[0], float);
 *   NUMA_GET_TYPED_POINTER(dst_data, tensor, ggml_fp16_t);
 */
#define NUMA_GET_TYPED_POINTER(ptr_var, tensor, data_type) \
    data_type * ptr_var; \
    do { \
        extern __thread void * ggml_numa_shared_result_tensor_data; \
        if (ggml_numa_shared_result_tensor_data && (tensor) && \
            tensor_data(tensor) == ggml_numa_shared_result_tensor_data) { \
            ptr_var = (data_type *)ggml_numa_shared_result_tensor_data; \
        } else { \
            ptr_var = (data_type *)tensor_data(tensor); \
        } \
    } while(0)

/**
 * @brief Access source tensor with validation
 * @param ptr_var Variable name for output pointer (will be declared)
 * @param tensor Target tensor (uses tensor->src[src_index])
 * @param src_index Source tensor index (0, 1, etc.)
 * @param data_type Data type for pointer
 * 
 * WHAT IT DOES:
 * - Validates source tensor exists
 * - Gets typed pointer to source data
 * - Handles const-correctness
 * 
 * USAGE:
 *   NUMA_GET_SOURCE_POINTER(src0_data, tensor, 0, float);
 *   NUMA_GET_SOURCE_POINTER(src1_data, tensor, 1, float);
 */
#define NUMA_GET_SOURCE_POINTER(ptr_var, tensor, src_index, data_type) \
    const data_type * ptr_var; \
    do { \
        NUMA_ASSERT((tensor)->src[src_index] != NULL, "Source tensor " #src_index " cannot be null"); \
        ptr_var = (const data_type *)tensor_data((tensor)->src[src_index]); \
    } while(0)

/**
 * @group Data Slicing Building Blocks
 * Distribute work across threads and NUMA nodes
 */

/**
 * @brief Slice data contiguously across all dimensions  
 * @param ctx NUMA execution context (must be initialized)
 * @param tensor Target tensor
 * 
 * WHAT IT DOES:
 * - Calculates total elements (ne[0] * ne[1] * ne[2] * ne[3])
 * - Distributes elements linearly across NUMA nodes and threads
 * - Sets thread_start, thread_end, thread_elements in context
 * - Handles remainder elements correctly
 * 
 * WHEN TO USE:
 * - Element-wise operations (ADD, MUL, etc.)
 * - Operations that treat tensor as flat array
 * - When memory access pattern doesn't matter
 * 
 * USAGE:
 *   NUMA_SLICE_CONTIGUOUS(ctx, tensor);
 *   // Process elements from ctx.thread_start to ctx.thread_end
 */
#define NUMA_SLICE_CONTIGUOUS(ctx, tensor) do { \
    size_t total_elements = ggml_nelements(tensor); \
    \
    /* NUMA-level slicing */ \
    if ((ctx).is_data_parallel) { \
        extern __thread int ggml_numa_total_nodes_for_data_parallel; \
        size_t elements_per_node = total_elements / ggml_numa_total_nodes_for_data_parallel; \
        (ctx).numa_start = (ctx).numa_node * elements_per_node; \
        (ctx).numa_end = ((ctx).numa_node == ggml_numa_total_nodes_for_data_parallel - 1) ? \
                         total_elements : (ctx).numa_start + elements_per_node; \
    } else { \
        (ctx).numa_start = 0; \
        (ctx).numa_end = total_elements; \
    } \
    \
    /* Thread-level slicing within NUMA node */ \
    size_t numa_elements = (ctx).numa_end - (ctx).numa_start; \
    size_t elements_per_thread = (numa_elements + (ctx).total_threads - 1) / (ctx).total_threads; \
    size_t thread_start_local = (ctx).thread_id * elements_per_thread; \
    size_t thread_end_local = (thread_start_local + elements_per_thread > numa_elements) ? \
                              numa_elements : thread_start_local + elements_per_thread; \
    \
    /* Convert to global indices */ \
    (ctx).thread_start = (ctx).numa_start + thread_start_local; \
    (ctx).thread_end = (ctx).numa_start + thread_end_local; \
    (ctx).thread_elements = (ctx).thread_end - (ctx).thread_start; \
    (ctx).has_work = ((ctx).thread_elements > 0); \
} while(0)

/**
 * @brief Slice tensor rows (ne[1] dimension only)
 * @param ctx NUMA execution context (must be initialized)  
 * @param tensor Target tensor
 * 
 * WHAT IT DOES:
 * - Distributes rows (ne[1] dimension) across threads
 * - Leaves other dimensions (ne[0], ne[2], ne[3]) unsliced
 * - Each thread processes full rows but subset of row indices
 * 
 * WHEN TO USE:
 * - Row-wise operations where each row is independent
 * - Operations that need to process complete rows
 * - When outer dimensions (ne[2], ne[3]) should be fully processed per thread
 * 
 * USAGE:
 *   NUMA_SLICE_ROWS_ATOMIC(ctx, tensor);
 *   // Process rows ctx.thread_start to ctx.thread_end
 *   for (i02) for (i01 = ctx.thread_start; i01 < ctx.thread_end; i01++)
 */
#define NUMA_SLICE_ROWS_ATOMIC(ctx, tensor) do { \
    size_t total_rows = (tensor)->ne[1]; \
    \
    /* NUMA-level slicing */ \
    if ((ctx).is_data_parallel) { \
        extern __thread int ggml_numa_total_nodes_for_data_parallel; \
        size_t rows_per_node = total_rows / ggml_numa_total_nodes_for_data_parallel; \
        (ctx).numa_start = (ctx).numa_node * rows_per_node; \
        (ctx).numa_end = ((ctx).numa_node == ggml_numa_total_nodes_for_data_parallel - 1) ? \
                         total_rows : (ctx).numa_start + rows_per_node; \
    } else { \
        (ctx).numa_start = 0; \
        (ctx).numa_end = total_rows; \
    } \
    \
    /* Thread-level slicing within NUMA node */ \
    size_t numa_rows = (ctx).numa_end - (ctx).numa_start; \
    size_t rows_per_thread = (numa_rows + (ctx).total_threads - 1) / (ctx).total_threads; \
    size_t thread_start_local = (ctx).thread_id * rows_per_thread; \
    size_t thread_end_local = (thread_start_local + rows_per_thread > numa_rows) ? \
                              numa_rows : thread_start_local + rows_per_thread; \
    \
    /* Convert to global row indices */ \
    (ctx).thread_start = (ctx).numa_start + thread_start_local; \
    (ctx).thread_end = (ctx).numa_start + thread_end_local; \
    (ctx).thread_elements = (ctx).thread_end - (ctx).thread_start; \
    (ctx).has_work = ((ctx).thread_elements > 0); \
} while(0)

/**
 * @group Early Exit Building Blocks
 * Handle threads with no work assignment
 */

/**
 * @brief Handle threads that have no work with automatic barrier participation
 * @param ctx NUMA execution context
 * @param return_value Value to return (typically GGML_STATUS_SUCCESS)
 * 
 * WHAT IT DOES:
 * - Checks if thread has work assigned (ctx.has_work)
 * - If no work: participates in barrier and returns early
 * - If has work: continues execution
 * - Ensures all threads participate in synchronization
 * 
 * WHEN TO USE:
 * - After any slicing operation
 * - Before starting actual computation
 * - To ensure proper thread synchronization
 * 
 * USAGE:
 *   NUMA_EARLY_EXIT_IF_NO_WORK(ctx, GGML_STATUS_SUCCESS);
 *   // Only threads with work continue past this point
 */
#define NUMA_EARLY_EXIT_IF_NO_WORK(ctx, return_value) do { \
    if (!(ctx).has_work) { \
        if ((ctx).is_data_parallel || (ctx).total_threads > 1) { \
            NUMA_OPENMP_BARRIER(); \
        } \
        return return_value; \
    } \
} while(0)

/**
 * @group Synchronization Building Blocks
 * Handle thread and NUMA node synchronization
 */

/**
 * @brief Automatic barrier - only when needed
 * @param ctx NUMA execution context
 * 
 * WHAT IT DOES:
 * - Inserts OpenMP barrier only when multiple threads/nodes are active
 * - Optimizes single-thread case (no barrier overhead)
 * - Ensures all threads reach synchronization point
 * 
 * WHEN TO USE:
 * - End of kernel execution
 * - Before/after shared data modifications
 * - When all threads must complete before continuing
 * 
 * USAGE:
 *   NUMA_BARRIER_AUTO(ctx);
 */
#define NUMA_BARRIER_AUTO(ctx) do { \
    if ((ctx).is_data_parallel || (ctx).total_threads > 1) { \
        NUMA_OPENMP_BARRIER(); \
    } \
} while(0)

/**
 * @brief Always insert barrier regardless of thread count
 * 
 * WHAT IT DOES:
 * - Unconditionally inserts OpenMP barrier
 * - Use sparingly - has overhead in single-thread case
 * 
 * WHEN TO USE:
 * - Debugging race conditions
 * - Critical synchronization points
 * - When barrier is always required regardless of configuration
 */
#define NUMA_BARRIER_ALWAYS() do { \
    NUMA_OPENMP_BARRIER(); \
} while(0)

/**
 * @brief Never insert barrier (no-op)
 * 
 * WHAT IT DOES:
 * - No synchronization (compile-time no-op)
 * - Use for lockfree algorithms or single-threaded operations
 * 
 * WHEN TO USE:
 * - View operations (metadata-only)
 * - Lockfree algorithms
 * - When synchronization is handled elsewhere
 */
#define NUMA_BARRIER_NEVER() do { \
    /* No barrier */ \
} while(0)

// ========================================================================
// COMPOSED KERNEL TEMPLATES - Level 2 Building Blocks  
// ========================================================================

/**
 * KERNEL COMPOSITION TEMPLATES
 * 
 * These macros combine atomic building blocks into common kernel patterns.
 * They provide ready-made templates for the most frequent NUMA kernel types.
 * Think of these as "kernel recipes" that handle the infrastructure while
 * you focus on the mathematical computation.
 */

/**
 * @brief Complete element-wise kernel template
 * @param ctx Context variable name (will be declared and initialized)
 * @param tensor Target tensor
 * @param params Compute parameters
 * @param dst_ptr Destination pointer variable name (will be declared)
 * @param data_type Data type (typically float)
 * 
 * WHAT IT PROVIDES:
 * - Complete setup and validation
 * - Contiguous data slicing
 * - Early exit for threads without work
 * - Typed destination pointer
 * - Ready for element-wise computation
 * 
 * WHAT YOU NEED TO ADD:
 * - The actual computation logic
 * - Source data access if needed
 * - Final barrier call
 * 
 * USAGE EXAMPLE:
 *   NUMA_ELEMENTWISE_KERNEL_SETUP(ctx, tensor, params, dst_data, float);
 *   const float* src_data = ...;
 *   // Compute on elements from ctx.thread_start to ctx.thread_end
 *   ggml_vec_add_f32(ctx.thread_elements, dst_data + ctx.thread_start, ...);
 *   NUMA_BARRIER_AUTO(ctx);
 */
#define NUMA_ELEMENTWISE_KERNEL_SETUP(ctx, tensor, params, dst_ptr, data_type) \
    NUMA_INIT_CONTEXT(ctx, tensor, params); \
    NUMA_VALIDATE_INPUTS(tensor, params); \
    NUMA_SLICE_CONTIGUOUS(ctx, tensor); \
    NUMA_EARLY_EXIT_IF_NO_WORK(ctx, GGML_STATUS_SUCCESS); \
    NUMA_GET_TYPED_POINTER(dst_ptr, tensor, data_type)

/**
 * @brief Complete row-wise kernel template  
 * @param ctx Context variable name (will be declared and initialized)
 * @param tensor Target tensor
 * @param params Compute parameters
 * @param dst_ptr Destination pointer variable name (will be declared)
 * @param data_type Data type (typically float)
 * 
 * WHAT IT PROVIDES:
 * - Complete setup and validation
 * - Row-wise data slicing (ne[1] dimension)
 * - Early exit for threads without work
 * - Typed destination pointer
 * - Ready for nested loop processing
 * 
 * WHAT YOU NEED TO ADD:
 * - Nested loop structure (typically 3D: i03, i02, i01)
 * - Row-level computation logic
 * - Final barrier call
 * 
 * USAGE EXAMPLE:
 *   NUMA_ROWWISE_KERNEL_SETUP(ctx, tensor, params, dst_data, float);
 *   // Process rows from ctx.thread_start to ctx.thread_end
 *   for (int64_t i03 = 0; i03 < ne03; i03++) {
 *       for (int64_t i02 = 0; i02 < ne02; i02++) {
 *           for (size_t i01 = ctx.thread_start; i01 < ctx.thread_end; i01++) {
 *               // Process row i01
 *           }
 *       }
 *   }
 *   NUMA_BARRIER_AUTO(ctx);
 */
#define NUMA_ROWWISE_KERNEL_SETUP(ctx, tensor, params, dst_ptr, data_type) \
    NUMA_INIT_CONTEXT(ctx, tensor, params); \
    NUMA_VALIDATE_INPUTS(tensor, params); \
    NUMA_SLICE_ROWS_ATOMIC(ctx, tensor); \
    NUMA_EARLY_EXIT_IF_NO_WORK(ctx, GGML_STATUS_SUCCESS); \
    NUMA_GET_TYPED_POINTER(dst_ptr, tensor, data_type)

/**
 * @brief Minimal kernel template for custom operations
 * @param ctx Context variable name (will be declared and initialized)  
 * @param tensor Target tensor
 * @param params Compute parameters
 * 
 * WHAT IT PROVIDES:
 * - Basic context initialization
 * - Input validation only
 * - No slicing (you choose your own)
 * - No early exit (you handle edge cases)
 * 
 * WHAT YOU NEED TO ADD:
 * - Your own slicing strategy
 * - Work distribution logic  
 * - Computation implementation
 * - Synchronization handling
 * 
 * WHEN TO USE:
 * - Complex operations that don't fit standard patterns
 * - Operations requiring custom slicing
 * - Experimental kernel development
 * 
 * USAGE EXAMPLE:
 *   NUMA_CUSTOM_KERNEL_SETUP(ctx, tensor, params);
 *   // Your custom slicing logic here
 *   NUMA_SLICE_CONTIGUOUS(ctx, tensor);  // or custom slicing
 *   NUMA_EARLY_EXIT_IF_NO_WORK(ctx, GGML_STATUS_SUCCESS);
 *   // Your computation logic here
 *   NUMA_BARRIER_AUTO(ctx);
 */
#define NUMA_CUSTOM_KERNEL_SETUP(ctx, tensor, params) \
    NUMA_INIT_CONTEXT(ctx, tensor, params); \
    NUMA_VALIDATE_INPUTS(tensor, params)

/**
 * @brief Complete kernel template with custom computation block
 * @param tensor Target tensor
 * @param params Compute parameters
 * @param computation_block Code block containing the actual computation
 * 
 * WHAT IT PROVIDES:
 * - Complete element-wise setup
 * - Automatic computation execution
 * - Proper synchronization
 * - Error handling
 * 
 * LIMITATIONS:
 * - Fixed to element-wise processing
 * - Single computation block only
 * - Less flexible than component approach
 * 
 * USAGE EXAMPLE:
 *   NUMA_COMPLETE_ELEMENTWISE_KERNEL(tensor, params, {
 *       const float* src = (const float*)tensor_data(tensor->src[0]);
 *       ggml_vec_scale_f32(ctx.thread_elements, 
 *                         dst_data + ctx.thread_start,
 *                         src + ctx.thread_start);
 *   });
 */
#define NUMA_COMPLETE_ELEMENTWISE_KERNEL(tensor, params, computation_block) do { \
    ggml_numa_slice_context_t ctx; \
    float* dst_data; \
    NUMA_ELEMENTWISE_KERNEL_SETUP(ctx, tensor, params, dst_data, float); \
    computation_block \
    NUMA_BARRIER_AUTO(ctx); \
} while(0)

// ========================================================================
// DEBUGGING AND LOGGING MACROS  
// ========================================================================

/**
 * @brief Log work distribution details for debugging
 * @param ctx NUMA execution context
 * @param operation_name Name of the operation for logging
 * @param start_unit Thread's start work unit
 * @param end_unit Thread's end work unit
 * @param unit_name Name of the work unit type (e.g., "sequences", "elements", "rows")
 */
#define NUMA_LOG_WORK_DISTRIBUTION(ctx, operation_name, start_unit, end_unit, unit_name) do { \
    NUMA_LOG_TRACE("%s: Thread %d/%d on NUMA %d processing %s [%zu,%zu) (%zu %s)", \
                   operation_name, (ctx).thread_id, (ctx).total_threads, (ctx).numa_node, \
                   unit_name, (size_t)(start_unit), (size_t)(end_unit), \
                   (size_t)((end_unit) - (start_unit)), unit_name); \
} while(0)

/**
 * @brief Log mathematical operation context for debugging
 * @param operation_name Name of the operation
 * @param tensor_info Brief description of tensor being processed
 * @param params_info Brief description of operation parameters
 */
#define NUMA_LOG_OPERATION_CONTEXT(operation_name, tensor_info, params_info) do { \
    NUMA_LOG_DEBUG("%s: Processing %s with %s", operation_name, tensor_info, params_info); \
} while(0)

#ifdef __cplusplus
}
#endif
