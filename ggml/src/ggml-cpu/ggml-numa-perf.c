/**
 * @file ggml-numa-perf.c
 * @brief NUMA Performance Instrumentation Implementation
 * 
 * High-precision timing and performance analysis for NUMA operations
 */

#include "ggml-numa-perf.h"
#include "ggml-numa-shared.h"  // For debug environment variable control
#include "ggml-impl.h"  // For GGML_LOG_* macros
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ============================================================================
// Global Configuration
// ============================================================================

bool g_numa_perf_enabled = false;
bool g_numa_perf_detailed_logging = false;
bool g_numa_perf_aggregate_stats = true;

// ============================================================================
// Internal Data Structures
// ============================================================================

#define NUMA_PERF_MAX_EVENTS 10000
#define NUMA_PERF_MAX_THREADS 128

// Global event storage
static numa_perf_event_t g_perf_events[NUMA_PERF_MAX_EVENTS];
static volatile uint64_t g_perf_event_count = 0;
static pthread_mutex_t g_perf_mutex = PTHREAD_MUTEX_INITIALIZER;

// Thread-local performance context
static __thread numa_perf_context_t g_perf_context = {0};

// Aggregated statistics per category
static numa_perf_stats_t g_perf_stats[NUMA_PERF_CATEGORY_COUNT] = {0};
static bool g_perf_initialized = false;

// ============================================================================
// Internal Helper Functions
// ============================================================================

static void update_stats(numa_perf_category_t category, double duration_ns, size_t data_size) {
    numa_perf_stats_t* stats = &g_perf_stats[category];
    
    if (stats->event_count == 0) {
        stats->min_time_ns = duration_ns;
        stats->max_time_ns = duration_ns;
        stats->total_time_ns = duration_ns;
    } else {
        if (duration_ns < stats->min_time_ns) stats->min_time_ns = duration_ns;
        if (duration_ns > stats->max_time_ns) stats->max_time_ns = duration_ns;
        stats->total_time_ns += duration_ns;
    }
    
    stats->event_count++;
    stats->avg_time_ns = stats->total_time_ns / stats->event_count;
    
    // Calculate throughput if data size is available
    if (data_size > 0 && duration_ns > 0) {
        double throughput_bps = (double)data_size / (duration_ns / 1e9);
        stats->throughput_gbps = throughput_bps / 1e9;
    }
    
    // Calculate efficiency score (inverse of normalized time)
    if (stats->avg_time_ns > 0) {
        stats->efficiency_score = 1.0 / (stats->avg_time_ns / 1e6); // Normalize to milliseconds
    }
}

static void log_performance_event(const numa_perf_event_t* event) {
    if (!g_numa_perf_detailed_logging) return;
    
    double duration_ms = event->duration_ns / 1e6;
    double throughput_gbps = 0.0;
    
    if (event->data_size_bytes > 0 && event->duration_ns > 0) {
        double throughput_bps = (double)event->data_size_bytes / (event->duration_ns / 1e9);
        throughput_gbps = throughput_bps / 1e9;
    }
    
    GGML_LOG_INFO("PERF: %s | %s | %s | Node:%d | Thread:%d | Duration:%.3fms | Size:%zuB | Threads:%d | Throughput:%.2fGB/s\n",
                  ggml_numa_perf_category_name(event->category),
                  event->operation_name ? event->operation_name : "N/A",
                  event->kernel_name ? event->kernel_name : "N/A",
                  event->numa_node,
                  event->thread_id,
                  duration_ms,
                  event->data_size_bytes,
                  event->thread_count,
                  throughput_gbps);
}

// ============================================================================
// Public API Implementation
// ============================================================================

bool ggml_numa_perf_init(void) {
    if (g_perf_initialized) {
        return true;
    }
    
    pthread_mutex_lock(&g_perf_mutex);
    if (!g_perf_initialized) {
        // Initialize event storage
        memset(g_perf_events, 0, sizeof(g_perf_events));
        g_perf_event_count = 0;
        
        // Initialize statistics
        memset(g_perf_stats, 0, sizeof(g_perf_stats));
        
        // Enable performance measurement with GGML_NUMA_PERF environment variable
        // This provides clean performance data independent from debug logging
        g_numa_perf_enabled = (ggml_numa_perf_enabled() >= 1);
        g_numa_perf_detailed_logging = (ggml_numa_perf_enabled() >= 2);  // Detailed mode
        
        g_perf_initialized = true;
        NUMA_LOG_DEBUG("NUMA Performance instrumentation initialized (enabled=%s, detailed=%s)",
                      g_numa_perf_enabled ? "true" : "false",
                      g_numa_perf_detailed_logging ? "true" : "false");
    }
    pthread_mutex_unlock(&g_perf_mutex);
    
    return true;
}

void ggml_numa_perf_shutdown(void) {
    if (!g_perf_initialized) return;
    
    pthread_mutex_lock(&g_perf_mutex);
    if (g_perf_initialized) {
        if (g_numa_perf_enabled) {
            ggml_numa_perf_print_summary();
        }
        g_perf_initialized = false;
        NUMA_LOG_DEBUG("NUMA Performance instrumentation shutdown");
    }
    pthread_mutex_unlock(&g_perf_mutex);
}

void ggml_numa_perf_set_enabled(bool enabled) {
    g_numa_perf_enabled = enabled;
    if (enabled) {
        ggml_numa_perf_init();
    }
}

void ggml_numa_perf_set_detailed_logging(bool enabled) {
    g_numa_perf_detailed_logging = enabled;
}

void ggml_numa_perf_start(numa_perf_category_t category, 
                          const char* operation_name,
                          const char* kernel_name,
                          int numa_node,
                          size_t data_size_bytes,
                          int thread_count) {
    if (!g_numa_perf_enabled) return;
    
    g_perf_context.enabled = true;
    g_perf_context.start_time_ns = ggml_numa_perf_get_time_ns();
    g_perf_context.current_category = category;
    g_perf_context.current_operation = operation_name;
    g_perf_context.current_kernel = kernel_name;
    g_perf_context.current_numa_node = numa_node;
    g_perf_context.current_data_size = data_size_bytes;
    g_perf_context.current_thread_count = thread_count;
}

void ggml_numa_perf_end(void) {
    if (!g_numa_perf_enabled || !g_perf_context.enabled) return;
    
    uint64_t end_time_ns = ggml_numa_perf_get_time_ns();
    double duration_ns = (double)(end_time_ns - g_perf_context.start_time_ns);
    
    // Get thread ID
    pthread_t pthread_id = pthread_self();
    int thread_id = (int)(uintptr_t)pthread_id % 10000; // Simple hash for display
    
    // Record the event
    ggml_numa_perf_record_event(
        g_perf_context.current_category,
        g_perf_context.current_operation,
        g_perf_context.current_kernel,
        g_perf_context.current_numa_node,
        thread_id,
        duration_ns,
        g_perf_context.current_data_size,
        g_perf_context.current_thread_count
    );
    
    g_perf_context.enabled = false;
}

void ggml_numa_perf_record_event(numa_perf_category_t category,
                                 const char* operation_name,
                                 const char* kernel_name,
                                 int numa_node,
                                 int thread_id,
                                 double duration_ns,
                                 size_t data_size_bytes,
                                 int thread_count) {
    if (!g_numa_perf_enabled) return;
    
    pthread_mutex_lock(&g_perf_mutex);
    
    // Store event if we have space
    if (g_perf_event_count < NUMA_PERF_MAX_EVENTS) {
        numa_perf_event_t* event = &g_perf_events[g_perf_event_count];
        event->category = category;
        event->operation_name = operation_name;
        event->kernel_name = kernel_name;
        event->numa_node = numa_node;
        event->thread_id = thread_id;
        event->duration_ns = duration_ns;
        event->timestamp_ns = ggml_numa_perf_get_time_ns();
        event->data_size_bytes = data_size_bytes;
        event->thread_count = thread_count;
        event->cache_hit = false; // TODO: Implement cache hit detection
        
        g_perf_event_count++;
        
        // Update aggregated statistics
        if (g_numa_perf_aggregate_stats) {
            update_stats(category, duration_ns, data_size_bytes);
        }
        
        // Log event if detailed logging is enabled
        log_performance_event(event);
    }
    
    pthread_mutex_unlock(&g_perf_mutex);
}

numa_perf_stats_t ggml_numa_perf_get_stats(numa_perf_category_t category) {
    if (category >= 0 && category < NUMA_PERF_CATEGORY_COUNT) {
        return g_perf_stats[category];
    }
    numa_perf_stats_t empty = {0};
    return empty;
}

void ggml_numa_perf_print_summary(void) {
    if (!g_perf_initialized) return;
    
    printf("\n" 
           "╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗\n"
           "║                                   NUMA PERFORMANCE SUMMARY                                          ║\n"
           "╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣\n");
    
    printf("║ Total Events: %-10lu  |  Detailed Logging: %-8s  |  Aggregate Stats: %-8s           ║\n",
           g_perf_event_count,
           g_numa_perf_detailed_logging ? "Enabled" : "Disabled",
           g_numa_perf_aggregate_stats ? "Enabled" : "Disabled");
    
    printf("╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║ %-24s │ %-8s │ %-12s │ %-12s │ %-12s │ %-10s ║\n",
           "Category", "Events", "Avg (ms)", "Min (ms)", "Max (ms)", "Efficiency");
    printf("╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣\n");
    
    for (int i = 0; i < NUMA_PERF_CATEGORY_COUNT; i++) {
        numa_perf_stats_t* stats = &g_perf_stats[i];
        if (stats->event_count > 0) {
            printf("║ %-24s │ %-8lu │ %12.3f │ %12.3f │ %12.3f │ %10.2f ║\n",
                   ggml_numa_perf_category_name(i),
                   stats->event_count,
                   stats->avg_time_ns / 1e6,
                   stats->min_time_ns / 1e6,
                   stats->max_time_ns / 1e6,
                   stats->efficiency_score);
        }
    }
    
    printf("╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝\n");
    
    // Calculate total execution time and breakdown
    double total_time_ns = 0;
    for (int i = 0; i < NUMA_PERF_CATEGORY_COUNT; i++) {
        total_time_ns += g_perf_stats[i].total_time_ns;
    }
    
    if (total_time_ns > 0) {
        printf("\nPerformance Breakdown:\n");
        for (int i = 0; i < NUMA_PERF_CATEGORY_COUNT; i++) {
            numa_perf_stats_t* stats = &g_perf_stats[i];
            if (stats->event_count > 0) {
                double percentage = (stats->total_time_ns / total_time_ns) * 100.0;
                printf("  %s: %.1f%% (%.3fms total)\n",
                       ggml_numa_perf_category_name(i),
                       percentage,
                       stats->total_time_ns / 1e6);
            }
        }
    }
    
    printf("\n");
}

void ggml_numa_perf_print_detailed_report(void) {
    if (!g_perf_initialized) return;
    
    printf("\n=== NUMA PERFORMANCE DETAILED REPORT ===\n");
    printf("Total Events: %lu\n\n", g_perf_event_count);
    
    pthread_mutex_lock(&g_perf_mutex);
    for (uint64_t i = 0; i < g_perf_event_count && i < NUMA_PERF_MAX_EVENTS; i++) {
        const numa_perf_event_t* event = &g_perf_events[i];
        double duration_ms = event->duration_ns / 1e6;
        double throughput_gbps = 0.0;
        
        if (event->data_size_bytes > 0 && event->duration_ns > 0) {
            double throughput_bps = (double)event->data_size_bytes / (event->duration_ns / 1e9);
            throughput_gbps = throughput_bps / 1e9;
        }
        
        printf("Event %lu: %s | %s | %s | Node:%d | Thread:%d | %.3fms | %zuB | %dthreads | %.2fGB/s\n",
               i + 1,
               ggml_numa_perf_category_name(event->category),
               event->operation_name ? event->operation_name : "N/A",
               event->kernel_name ? event->kernel_name : "N/A",
               event->numa_node,
               event->thread_id,
               duration_ms,
               event->data_size_bytes,
               event->thread_count,
               throughput_gbps);
    }
    pthread_mutex_unlock(&g_perf_mutex);
    
    printf("=== END DETAILED REPORT ===\n\n");
}

void ggml_numa_perf_reset(void) {
    if (!g_perf_initialized) return;
    
    pthread_mutex_lock(&g_perf_mutex);
    memset(g_perf_events, 0, sizeof(g_perf_events));
    memset(g_perf_stats, 0, sizeof(g_perf_stats));
    g_perf_event_count = 0;
    pthread_mutex_unlock(&g_perf_mutex);
    
    GGML_LOG_INFO("NUMA Performance statistics reset\n");
}
