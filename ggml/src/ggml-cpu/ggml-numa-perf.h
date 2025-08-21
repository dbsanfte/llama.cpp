/**
 * @file ggml-numa-perf.h
 * @brief NUMA Performance Instrumentation System
 * 
 * High-precision timing and performance analysis for NUMA operations
 * Provides detailed metrics for coordinator, executor, and kernel performance
 */

#ifndef GGML_NUMA_PERF_H
#define GGML_NUMA_PERF_H

#include <time.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// Performance measurement categories
typedef enum {
    NUMA_PERF_COORDINATOR_INIT,
    NUMA_PERF_COORDINATOR_DISPATCH,
    NUMA_PERF_COORDINATOR_BARRIER_WAIT,
    NUMA_PERF_COORDINATOR_CLEANUP,
    NUMA_PERF_EXECUTOR_QUERY,
    NUMA_PERF_EXECUTOR_STRATEGY,
    NUMA_PERF_EXECUTOR_KERNEL_EXEC,
    NUMA_PERF_EXECUTOR_FALLBACK,
    NUMA_PERF_KERNEL_NUMA_EXEC,
    NUMA_PERF_KERNEL_THREAD_WORK,
    NUMA_PERF_KERNEL_MEMORY_ACCESS,
    NUMA_PERF_OPERATION_TOTAL,
    NUMA_PERF_CATEGORY_COUNT
} numa_perf_category_t;

// Performance event structure
typedef struct {
    numa_perf_category_t category;
    const char* operation_name;
    const char* kernel_name;
    int numa_node;
    int thread_id;
    double duration_ns;
    uint64_t timestamp_ns;
    size_t data_size_bytes;
    int thread_count;
    bool cache_hit;
} numa_perf_event_t;

// Performance statistics aggregation
typedef struct {
    double total_time_ns;
    double min_time_ns;
    double max_time_ns;
    double avg_time_ns;
    uint64_t event_count;
    double throughput_gbps;  // Gigabytes per second
    double efficiency_score;
} numa_perf_stats_t;

// Performance context for thread-local measurements
typedef struct {
    bool enabled;
    uint64_t start_time_ns;
    numa_perf_category_t current_category;
    const char* current_operation;
    const char* current_kernel;
    int current_numa_node;
    size_t current_data_size;
    int current_thread_count;
} numa_perf_context_t;

// Global performance configuration
extern bool g_numa_perf_enabled;
extern bool g_numa_perf_detailed_logging;
extern bool g_numa_perf_aggregate_stats;

// ============================================================================
// Performance Measurement API
// ============================================================================

/**
 * Initialize performance measurement system
 */
bool ggml_numa_perf_init(void);

/**
 * Shutdown performance measurement system and print summary
 */
void ggml_numa_perf_shutdown(void);

/**
 * Enable/disable performance measurement
 */
void ggml_numa_perf_set_enabled(bool enabled);

/**
 * Enable/disable detailed logging
 */
void ggml_numa_perf_set_detailed_logging(bool enabled);

/**
 * Get high-precision timestamp
 */
static inline uint64_t ggml_numa_perf_get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/**
 * Start performance measurement for current thread
 */
void ggml_numa_perf_start(numa_perf_category_t category, 
                          const char* operation_name,
                          const char* kernel_name,
                          int numa_node,
                          size_t data_size_bytes,
                          int thread_count);

/**
 * End performance measurement and record event
 */
void ggml_numa_perf_end(void);

/**
 * Record a complete performance event
 */
void ggml_numa_perf_record_event(numa_perf_category_t category,
                                 const char* operation_name,
                                 const char* kernel_name,
                                 int numa_node,
                                 int thread_id,
                                 double duration_ns,
                                 size_t data_size_bytes,
                                 int thread_count);

/**
 * Get performance statistics for a category
 */
numa_perf_stats_t ggml_numa_perf_get_stats(numa_perf_category_t category);

/**
 * Print performance summary
 */
void ggml_numa_perf_print_summary(void);

/**
 * Print detailed performance report
 */
void ggml_numa_perf_print_detailed_report(void);

/**
 * Reset all performance statistics
 */
void ggml_numa_perf_reset(void);

// ============================================================================
// Convenience Macros for Easy Instrumentation
// ============================================================================

#define NUMA_PERF_START(category, op_name, kernel_name, numa_node, data_size, thread_count) \
    do { \
        if (g_numa_perf_enabled) { \
            ggml_numa_perf_start(category, op_name, kernel_name, numa_node, data_size, thread_count); \
        } \
    } while(0)

#define NUMA_PERF_END() \
    do { \
        if (g_numa_perf_enabled) { \
            ggml_numa_perf_end(); \
        } \
    } while(0)

#define NUMA_PERF_RECORD(category, op_name, kernel_name, numa_node, thread_id, duration_ns, data_size, thread_count) \
    do { \
        if (g_numa_perf_enabled) { \
            ggml_numa_perf_record_event(category, op_name, kernel_name, numa_node, thread_id, duration_ns, data_size, thread_count); \
        } \
    } while(0)

// Scoped performance measurement
#define NUMA_PERF_SCOPE(category, op_name, kernel_name, numa_node, data_size, thread_count) \
    numa_perf_scope_guard_t _perf_guard __attribute__((cleanup(numa_perf_scope_cleanup))) = { \
        .active = g_numa_perf_enabled \
    }; \
    if (_perf_guard.active) { \
        ggml_numa_perf_start(category, op_name, kernel_name, numa_node, data_size, thread_count); \
    }

// Cleanup function for scoped measurement
typedef struct {
    bool active;
} numa_perf_scope_guard_t;

static inline void numa_perf_scope_cleanup(numa_perf_scope_guard_t* guard) {
    if (guard->active) {
        ggml_numa_perf_end();
    }
}

// ============================================================================
// Category Names for Logging
// ============================================================================

static const char* numa_perf_category_names[] = {
    "CoordinatorInit",
    "CoordinatorDispatch", 
    "CoordinatorBarrierWait",
    "CoordinatorCleanup",
    "ExecutorQuery",
    "ExecutorStrategy",
    "ExecutorKernelExec",
    "ExecutorFallback",
    "KernelNumaExec",
    "KernelThreadWork",
    "KernelMemoryAccess",
    "OperationTotal"
};

static inline const char* ggml_numa_perf_category_name(numa_perf_category_t category) {
    if (category >= 0 && category < NUMA_PERF_CATEGORY_COUNT) {
        return numa_perf_category_names[category];
    }
    return "Unknown";
}

#ifdef __cplusplus
}
#endif

#endif // GGML_NUMA_PERF_H
