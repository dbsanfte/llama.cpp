/*
 * NUMA Shared Utilities and Macros
 * 
 * Common logging, assertion, and utility macros for the NUMA system.
 * This header provides a clean interface for all NUMA components.
 */

#pragma once

#include "ggml-impl.h"
#include <assert.h>

// ============================================================================
// NUMA Logging Macros
// ============================================================================

/**
 * Debug logging with NUMA node context
 * Usage: NUMA_COORD_LOG_DEBUG(numa_node, "Message %d", value);
 */
#define NUMA_COORD_LOG_DEBUG(numa_node, ...) \
    GGML_LOG_DEBUG("NUMA[%d]: " __VA_ARGS__, (int)(numa_node))

/**
 * Info logging with NUMA node context
 */
#define NUMA_COORD_LOG_INFO(numa_node, ...) \
    GGML_LOG_INFO("NUMA[%d]: " __VA_ARGS__, (int)(numa_node))

/**
 * Warning logging with NUMA node context
 */
#define NUMA_COORD_LOG_WARN(numa_node, ...) \
    GGML_LOG_WARN("NUMA[%d]: " __VA_ARGS__, (int)(numa_node))

/**
 * Error logging with NUMA node context
 */
#define NUMA_COORD_LOG_ERROR(numa_node, ...) \
    GGML_LOG_ERROR("NUMA[%d]: " __VA_ARGS__, (int)(numa_node))

/**
 * General NUMA logging without node-specific context
 */
#define NUMA_LOG_DEBUG(...) GGML_LOG_DEBUG("NUMA: " __VA_ARGS__)
#define NUMA_LOG_INFO(...)  GGML_LOG_INFO("NUMA: " __VA_ARGS__)
#define NUMA_LOG_WARN(...)  GGML_LOG_WARN("NUMA: " __VA_ARGS__)
#define NUMA_LOG_ERROR(...) GGML_LOG_ERROR("NUMA: " __VA_ARGS__)

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
