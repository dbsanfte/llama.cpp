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

// =============================================================================
// NUMA SLICING UTILITIES - Reusable Data Partitioning Macros
// =============================================================================

/**
 * NUMA slicing context structure
 * Contains all calculated slice boundaries for element-wise and sequence-wise operations
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
