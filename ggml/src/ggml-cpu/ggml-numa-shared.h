/*
 * NUMA Shared Utilities and Macros
 * 
 * Performance-optimized NUMA utilities for llama.cpp with zero-overhead design:
 * - Debug logging controlled by GGML_NUMA_DEBUG (zero overhead when disabled)
 * - Performance timing controlled by GGML_NUMA_PERF (zero overhead when disabled)
 * - Debug assertions compile to no-op in release builds (zero overhead with NDEBUG)
 * - Legacy assertions remain always-on for critical safety checks
 * 
 * This header provides optimal performance in production while preserving
 * comprehensive debugging capabilities during development.
 */

#pragma once

#include "ggml.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

// Forward declarations to avoid circular dependencies
struct ggml_tensor;
struct ggml_cplan;

// ============================================================================
// NUMA Type Definitions
// ============================================================================

/**
 * Forward declarations for NUMA types
 */
struct ggml_compute_params;

/**
 * NUMA work function type - function pointer for kernel execution
 * Used by coordinator to execute work on specific NUMA nodes
 */
typedef enum ggml_status (*ggml_numa_work_function_t)(
    void * work_context,                    // Function-specific context data
    struct ggml_compute_params * params     // Compute parameters (threads, buffer, etc.)
);

/**
 * Node distribution strategies - how work is distributed across NUMA nodes
 */
typedef enum {
    NUMA_NODE_STRATEGY_SINGLE,            // Execute on a single node
    NUMA_NODE_STRATEGY_DATA_PARALLEL      // Distribute data across multiple nodes
} ggml_numa_node_strategy_t;

/**
 * On-node execution strategies - how work is executed within each NUMA node
 */
typedef enum {
    NUMA_ON_NODE_STRATEGY_SINGLE_THREAD,  // Single thread execution
    NUMA_ON_NODE_STRATEGY_MULTI_THREAD    // Multi-threaded execution
} ggml_numa_on_node_strategy_t;

/**
 * Combined execution strategy - combines node distribution and on/node execution
 */
typedef struct {
    ggml_numa_node_strategy_t node_strategy;      // How to distribute across nodes
    ggml_numa_on_node_strategy_t on_node_strategy; // How to execute within each node
} ggml_numa_execution_strategy_t;

// ============================================================================
// NUMA Strategy Cache System (O(1) Hash Table Performance)
// ============================================================================

/**
 * Strategy threshold array indices (by convention)
 * Each kernel provides a simple array with element count thresholds
 * Extended for testing with custom threshold points
 */
typedef enum {
    NUMA_STRATEGY_IDX_SINGLE_SINGLE = 0,   // Single node, single thread threshold
    NUMA_STRATEGY_IDX_SINGLE_MULTI = 1,    // Single node, multi-thread threshold
    NUMA_STRATEGY_IDX_DATA_PARALLEL = 2,   // Data-parallel threshold (for testing)
    NUMA_STRATEGY_IDX_COUNT = 3             // Total number of threshold indices
} ggml_numa_strategy_idx_t;

/**
 * Kernel strategy array - provided by each kernel at registration
 * Simple threshold array for O(1) strategy selection
 */
typedef struct {
    size_t thresholds[NUMA_STRATEGY_IDX_COUNT];  // Element count thresholds
    bool valid;                                   // True if thresholds are provided
} ggml_numa_kernel_strategy_array_t;

/**
 * Work function pointers for each strategy
 * Kernels provide these at registration time
 * These match the ggml_numa_work_function_t signature expected by the coordinator
 */
typedef struct {
    // Function pointer for single-node, single-thread execution
    ggml_numa_work_function_t single_single_fn;
    
    // Function pointer for single-node, multi-thread execution  
    ggml_numa_work_function_t single_multi_fn;
    
    // Function pointer for data-parallel execution
    ggml_numa_work_function_t data_parallel_fn;
                                        
    bool valid;  // True if function pointers are provided
} ggml_numa_kernel_work_funcs_t;

/**
 * Aggregation function pointers for each strategy
 * Kernels provide these at registration time for operations that need result aggregation
 */
typedef struct {
    // Function pointer for single-node, single-thread aggregation
    enum ggml_status (*single_single_fn)(void * work_context, int numa_node, 
                                        struct ggml_tensor * tensor, struct ggml_cplan * cplan);
    
    // Function pointer for single-node, multi-thread aggregation  
    enum ggml_status (*single_multi_fn)(void * work_context, int numa_node,
                                      struct ggml_tensor * tensor, struct ggml_cplan * cplan);
    
    // Function pointer for data-parallel aggregation
    enum ggml_status (*data_parallel_fn)(void * work_context, int numa_node,
                                        struct ggml_tensor * tensor, struct ggml_cplan * cplan);
                                        
    bool valid;  // True if function pointers are provided
} ggml_numa_kernel_aggregation_funcs_t;

/**
 * Kernel registration info returned by each kernel's registration function
 * This allows each kernel to define its own strategies and function pointers
 */
typedef struct {
    enum ggml_op op_type;                                     // Operation type this kernel handles
    ggml_numa_kernel_strategy_array_t strategy_array;        // Strategy thresholds 
    ggml_numa_kernel_work_funcs_t work_funcs;                // Work function pointers
    ggml_numa_kernel_aggregation_funcs_t agg_funcs;          // Aggregation function pointers (optional)
    const char * kernel_name;                                // Human-readable name
    bool supported;                                           // Whether kernel is available
} ggml_numa_kernel_registration_info_t;

/**
 * Function pointer type for kernel registration functions
 * Each kernel provides a function of this type to register itself
 */
typedef ggml_numa_kernel_registration_info_t (*ggml_numa_kernel_register_fn_t)(void);

// ============================================================================
// NUMA Logging Macros (Simplified versions without ggml-impl dependency)
// ============================================================================

/**
 * Environment variable-controlled debug logging system
 * Set GGML_NUMA_DEBUG=1 to enable debug output
 * Set GGML_NUMA_DEBUG=2 for verbose debug output
 * Set GGML_NUMA_DEBUG=3 for trace debug output (very detailed, includes individual operations)
 * 
 * Separate from performance measurements (use GGML_NUMA_PERF for that)
 */
static inline int ggml_numa_debug_enabled(void) {
    static int debug_level = -1;
    if (debug_level == -1) {
        const char *env = getenv("GGML_NUMA_DEBUG");
        debug_level = env ? atoi(env) : 0;
    }
    return debug_level;
}

/**
 * Environment variable-controlled performance measurement system
 * Set GGML_NUMA_PERF=1 to enable performance measurements
 * Set GGML_NUMA_PERF=2 for detailed performance logging
 */
static inline int ggml_numa_perf_enabled(void) {
    static int perf_level = -1;
    if (perf_level == -1) {
        const char *env = getenv("GGML_NUMA_PERF");
        perf_level = env ? atoi(env) : 0;
    }
    return perf_level;
}

/**
 * Debug logging with NUMA node context
 * Usage: NUMA_COORD_LOG_DEBUG(numa_node, "Message %d", value);
 */
#define NUMA_COORD_LOG_DEBUG(numa_node, ...) \
    do { if (ggml_numa_debug_enabled() >= 1) { \
        fprintf(stderr, "NUMA[%d] DEBUG: " __VA_ARGS__, (int)(numa_node)); \
        fprintf(stderr, "\n"); \
    } } while(0)

/**
 * Verbose debug logging (only with GGML_NUMA_DEBUG=2 or higher)
 */
#define NUMA_COORD_LOG_VERBOSE(numa_node, ...) \
    do { if (ggml_numa_debug_enabled() >= 2) { \
        fprintf(stderr, "NUMA[%d] VERBOSE: " __VA_ARGS__, (int)(numa_node)); \
        fprintf(stderr, "\n"); \
    } } while(0)

/**
 * Trace debug logging (only with GGML_NUMA_DEBUG=3 or higher)
 * Used for very detailed internal operations like individual tensor rows
 */
#define NUMA_COORD_LOG_TRACE(numa_node, ...) \
    do { if (ggml_numa_debug_enabled() >= 3) { \
        fprintf(stderr, "NUMA[%d] TRACE: " __VA_ARGS__, (int)(numa_node)); \
        fprintf(stderr, "\n"); \
    } } while(0)

/**
 * Info logging with NUMA node context
 */
#define NUMA_COORD_LOG_INFO(numa_node, ...) \
    fprintf(stderr, "NUMA[%d] INFO: " __VA_ARGS__, (int)(numa_node)); \
    fprintf(stderr, "\n")

/**
 * Warning logging with NUMA node context
 */
#define NUMA_COORD_LOG_WARN(numa_node, ...) \
    fprintf(stderr, "NUMA[%d] WARNING: " __VA_ARGS__, (int)(numa_node)); \
    fprintf(stderr, "\n")

/**
 * Error logging with NUMA node context
 */
#define NUMA_COORD_LOG_ERROR(numa_node, ...) \
    fprintf(stderr, "NUMA[%d] ERROR: " __VA_ARGS__, (int)(numa_node)); \
    fprintf(stderr, "\n")

/**
 * General NUMA logging without node-specific context
 */
#define NUMA_LOG_DEBUG(...) \
    do { if (ggml_numa_debug_enabled() >= 1) { \
        fprintf(stderr, "NUMA DEBUG: " __VA_ARGS__); \
        fprintf(stderr, "\n"); \
    } } while(0)

#define NUMA_LOG_VERBOSE(...) \
    do { if (ggml_numa_debug_enabled() >= 2) { \
        fprintf(stderr, "NUMA VERBOSE: " __VA_ARGS__); \
        fprintf(stderr, "\n"); \
    } } while(0)

/**
 * Trace logging (only with GGML_NUMA_DEBUG=3 or higher)
 * Used for very detailed internal operations like individual tensor rows
 */
#define NUMA_LOG_TRACE(...) \
    do { if (ggml_numa_debug_enabled() >= 3) { \
        fprintf(stderr, "NUMA TRACE: " __VA_ARGS__); \
        fprintf(stderr, "\n"); \
    } } while(0)

#define NUMA_LOG_INFO(...) \
    fprintf(stderr, "NUMA INFO: " __VA_ARGS__); \
    fprintf(stderr, "\n")
#define NUMA_LOG_WARN(...) \
    fprintf(stderr, "NUMA WARNING: " __VA_ARGS__); \
    fprintf(stderr, "\n")
#define NUMA_LOG_ERROR(...) \
    fprintf(stderr, "NUMA ERROR: " __VA_ARGS__); \
    fprintf(stderr, "\n")

// ============================================================================
// NUMA Assertions
// ============================================================================

/**
 * NUMA-specific assertion macro with descriptive error message
 * Note: NUMA_ASSERT may already be defined in coordinator header with different signature
 * Usage: NUMA_ASSERT_MSG(condition, "Description of what failed");
 */
#define NUMA_ASSERT_MSG(condition, message) \
    do { \
        if (!(condition)) { \
            NUMA_LOG_ERROR("ASSERTION FAILED: %s (at %s:%d)", \
                          (message), __FILE__, __LINE__); \
            assert(condition); \
        } \
    } while(0)

/**
 * NUMA assertion with formatted message
 * Usage: NUMA_ASSERT_FMT(ptr != NULL, "Pointer %p should not be NULL", ptr);
 */
#define NUMA_ASSERT_FMT(condition, fmt, ...) \
    do { \
        if (!(condition)) { \
            NUMA_LOG_ERROR("ASSERTION FAILED: " fmt " (at %s:%d)", \
                          __VA_ARGS__, __FILE__, __LINE__); \
            assert(condition); \
        } \
    } while(0)

/**
 * Simple NUMA assertion macro (for backward compatibility)
 * Usage: NUMA_ASSERT(condition, "Description of what failed");
 */
#define NUMA_ASSERT(condition, message) NUMA_ASSERT_MSG(condition, message)

// ============================================================================
// NUMA Debug-Only Assertion Macros (Performance Optimized)
// ============================================================================

/**
 * Debug-only assertion macros that compile to no-op in release builds
 * These provide zero performance overhead in production while preserving 
 * full debugging capabilities during development
 */

// Debug build detection - assertions enabled when DEBUG is defined or NDEBUG is not defined
#if defined(DEBUG) || !defined(NDEBUG)
    #define NUMA_DEBUG_ASSERTIONS_ENABLED 1
#else
    #define NUMA_DEBUG_ASSERTIONS_ENABLED 0
#endif

#if NUMA_DEBUG_ASSERTIONS_ENABLED

/**
 * Debug-only assertion with message (enabled in debug builds only)
 * Usage: NUMA_DEBUG_ASSERT_MSG(ptr != NULL, "Pointer should not be NULL");
 */
#define NUMA_DEBUG_ASSERT_MSG(condition, message) \
    do { \
        if (!(condition)) { \
            NUMA_LOG_ERROR("DEBUG ASSERTION FAILED: %s (at %s:%d)", \
                          (message), __FILE__, __LINE__); \
            assert(condition); \
        } \
    } while(0)

/**
 * Debug-only assertion with formatted message (enabled in debug builds only)
 * Usage: NUMA_DEBUG_ASSERT_FMT(ptr != NULL, "Pointer %p should not be NULL", ptr);
 */
#define NUMA_DEBUG_ASSERT_FMT(condition, fmt, ...) \
    do { \
        if (!(condition)) { \
            NUMA_LOG_ERROR("DEBUG ASSERTION FAILED: " fmt " (at %s:%d)", \
                          __VA_ARGS__, __FILE__, __LINE__); \
            assert(condition); \
        } \
    } while(0)

/**
 * Simple debug-only assertion (enabled in debug builds only)
 * Usage: NUMA_DEBUG_ASSERT(condition, "Description of what failed");
 */
#define NUMA_DEBUG_ASSERT(condition, message) NUMA_DEBUG_ASSERT_MSG(condition, message)

#else

// Release build - all debug assertions compile to no-op for zero overhead
#define NUMA_DEBUG_ASSERT_MSG(condition, message) ((void)0)
#define NUMA_DEBUG_ASSERT_FMT(condition, fmt, ...) ((void)0)
#define NUMA_DEBUG_ASSERT(condition, message) ((void)0)

#endif

// ============================================================================
// NUMA Utility Macros
// ============================================================================

/**
 * Fast strategy selection from threshold array (O(1) performance)
 * Given element count, returns the appropriate execution strategy
 */
static inline ggml_numa_execution_strategy_t numa_select_strategy_fast(
    const ggml_numa_kernel_strategy_array_t * strategy_array,
    size_t element_count) {
    
    ggml_numa_execution_strategy_t result;
    
    if (!strategy_array || !strategy_array->valid) {
        // Default fallback: single node, multi-thread for safety
        result.node_strategy = NUMA_NODE_STRATEGY_SINGLE;
        result.on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD;
        return result;
    }
    
    // O(1) threshold comparison for strategy selection with 3-level support
    if (element_count <= strategy_array->thresholds[NUMA_STRATEGY_IDX_SINGLE_SINGLE]) {
        result.node_strategy = NUMA_NODE_STRATEGY_SINGLE;
        result.on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD;
    } else if (element_count <= strategy_array->thresholds[NUMA_STRATEGY_IDX_SINGLE_MULTI]) {
        result.node_strategy = NUMA_NODE_STRATEGY_SINGLE;
        result.on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD;
    } else if (element_count <= strategy_array->thresholds[NUMA_STRATEGY_IDX_DATA_PARALLEL]) {
        // Third threshold: data-parallel with controlled size
        result.node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL;
        result.on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD;
    } else {
        // Above all thresholds: use data-parallel strategy (large)
        result.node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL;
        result.on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD;
    }
    
    return result;
}

/**
 * Get work function pointer based on selected strategy (O(1) performance)
 */
static inline ggml_numa_work_function_t numa_get_work_func_fast(
    const ggml_numa_kernel_work_funcs_t * work_funcs,
    const ggml_numa_execution_strategy_t * strategy) {
    
    if (!work_funcs || !work_funcs->valid || !strategy) {
        return NULL;
    }
    
    // O(1) function pointer selection based on strategy
    if (strategy->node_strategy == NUMA_NODE_STRATEGY_SINGLE) {
        if (strategy->on_node_strategy == NUMA_ON_NODE_STRATEGY_SINGLE_THREAD) {
            return work_funcs->single_single_fn;
        } else {
            return work_funcs->single_multi_fn;
        }
    } else {
        return work_funcs->data_parallel_fn;
    }
}

/**
 * Get aggregation function pointer based on selected strategy (O(1) performance)
 */
static inline enum ggml_status (*numa_get_aggregation_func_fast(
    const ggml_numa_kernel_aggregation_funcs_t * agg_funcs,
    const ggml_numa_execution_strategy_t * strategy))(void *, int, struct ggml_tensor *, struct ggml_cplan *) {
    
    if (!agg_funcs || !agg_funcs->valid || !strategy) {
        return NULL;
    }
    
    // O(1) function pointer selection based on strategy
    if (strategy->node_strategy == NUMA_NODE_STRATEGY_SINGLE) {
        if (strategy->on_node_strategy == NUMA_ON_NODE_STRATEGY_SINGLE_THREAD) {
            return agg_funcs->single_single_fn;
        } else {
            return agg_funcs->single_multi_fn;
        }
    } else {
        return agg_funcs->data_parallel_fn;
    }
}

/**
 * Simple hash function for ggml_op to array index conversion
 * Uses operation type directly as hash (enum values are sequential)
 */
static inline size_t numa_op_hash(enum ggml_op op) {
    return (size_t)op;
}

/**
 * Maximum supported operation type for hash table sizing
 * Should be updated if new operations are added
 */
#define NUMA_OP_HASH_TABLE_SIZE 128

/**
 * Safe pointer check with logging
 */
#define NUMA_CHECK_PTR(ptr, name) \
    do { \
        if (!(ptr)) { \
            NUMA_LOG_ERROR("NULL pointer: %s", (name)); \
            return NULL; \
        } \
    } while(0)

/**
 * Safe pointer check with custom return value
 */
#define NUMA_CHECK_PTR_RET(ptr, name, ret_val) \
    do { \
        if (!(ptr)) { \
            NUMA_LOG_ERROR("NULL pointer: %s", (name)); \
            return (ret_val); \
        } \
    } while(0)

/**
 * NUMA range validation
 */
#define NUMA_CHECK_RANGE(value, min_val, max_val, name) \
    do { \
        if ((value) < (min_val) || (value) > (max_val)) { \
            NUMA_LOG_ERROR("Value %d out of range [%d, %d]: %s", \
                          (int)(value), (int)(min_val), (int)(max_val), (name)); \
            return false; \
        } \
    } while(0)

// ============================================================================
// NUMA Performance Logging
// ============================================================================

/**
 * Performance timing macros for NUMA operations
 */
#define NUMA_PERF_LOG_START(operation) \
    NUMA_LOG_DEBUG("PERF: Starting %s", (operation))

#define NUMA_PERF_LOG_END(operation, duration_us) \
    NUMA_LOG_DEBUG("PERF: Completed %s in %ld μs", (operation), (long)(duration_us))

// ============================================================================
// NUMA Memory Logging
// ============================================================================

/**
 * Memory allocation logging
 */
#define NUMA_MEM_LOG_ALLOC(size, numa_node) \
    NUMA_COORD_LOG_DEBUG(numa_node, "Allocated %zu bytes", (size_t)(size))

#define NUMA_MEM_LOG_FREE(ptr, numa_node) \
    NUMA_COORD_LOG_DEBUG(numa_node, "Freed memory at %p", (void*)(ptr))

#define NUMA_MEM_LOG_BUFFER(name, size, numa_node) \
    NUMA_COORD_LOG_DEBUG(numa_node, "Buffer %s: %zu bytes", (name), (size_t)(size))
