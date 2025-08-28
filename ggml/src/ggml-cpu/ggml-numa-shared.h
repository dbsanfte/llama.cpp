/*
 * NUMA Shared Utilities and Macros
 * 
 * Performance-optimized NUMA utilities for llama.cpp with zero-overhead design:
 * - Debug logging controlled by GGML_NUMA_DEBUG (zero overhead when disabled)
 * - Performance timing controlled by g_numa_perf_enabled (zero overhead when disabled)
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
 * Combined execution strategy - combines node distribution and on-node execution
 */
typedef struct {
    ggml_numa_node_strategy_t node_strategy;      // How to distribute across nodes
    ggml_numa_on_node_strategy_t on_node_strategy; // How to execute within each node
} ggml_numa_execution_strategy_t;

// ============================================================================
// NUMA Logging Macros (Simplified versions without ggml-impl dependency)
// ============================================================================

/**
 * Environment variable-controlled debug logging system
 * Set GGML_NUMA_DEBUG=1 to enable debug output
 * Set GGML_NUMA_DEBUG=2 for verbose debug output
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
