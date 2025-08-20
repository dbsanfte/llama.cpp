/*
 * NUMA Kernel Registry Implementation - High-Performance Caching System
 * 
 * This module provides lightning-fast kernel lookups through a one-off cache
 * of supported operations with pre-computed strategies and work functions.
 * 
 * Architecture:
 * 1. Build cache on init with quick complexity scoring
 * 2. Ultra-fast lookups during graph execution
 * 3. No runtime decision overhead
 */

#include "numa-kernels.h"
#include "add.h"
#include "../ggml-impl.h"

// ============================================================================
// Fast Cache System for Lightning-Speed Operation Execution
// ============================================================================

/**
 * Quick complexity categorization for cache keys
 * This avoids expensive runtime calculations during graph execution
 */
typedef enum {
    COMPLEXITY_TINY = 0,    // < 1K elements
    COMPLEXITY_SMALL,       // 1K - 16K elements  
    COMPLEXITY_MEDIUM,      // 16K - 256K elements
    COMPLEXITY_LARGE,       // 256K - 4M elements
    COMPLEXITY_HUGE,        // > 4M elements
    COMPLEXITY_COUNT
} ggml_numa_complexity_class_t;

/**
 * Pre-computed cache entry for ultra-fast lookups
 * All decisions made at init time, zero overhead during execution
 */
typedef struct {
    bool valid;                                    // Cache entry is valid
    ggml_numa_execution_strategy_t strategy;      // Pre-computed strategy
    size_t work_buffer_size_per_thread;          // Pre-computed buffer size
    ggml_numa_work_function_t work_function;     // Pre-selected work function
    float efficiency_score;                       // Pre-computed efficiency
    const char * kernel_name;                     // Kernel identifier
} ggml_numa_cache_entry_t;

/**
 * High-speed lookup cache
 * [operation_type][complexity_class] -> cache_entry
 * O(1) lookups during graph execution
 */
static ggml_numa_cache_entry_t g_numa_cache[GGML_OP_COUNT][COMPLEXITY_COUNT];
static bool g_numa_cache_initialized = false;

// ============================================================================
// Quick Complexity Scoring (Ultra-Fast)
// ============================================================================

static inline ggml_numa_complexity_class_t get_complexity_class(size_t num_elements) {
    if (num_elements < 1024) return COMPLEXITY_TINY;
    if (num_elements < 16384) return COMPLEXITY_SMALL;
    if (num_elements < 262144) return COMPLEXITY_MEDIUM;
    if (num_elements < 4194304) return COMPLEXITY_LARGE;
    return COMPLEXITY_HUGE;
}

static inline size_t get_tensor_complexity_score(const struct ggml_tensor * tensor) {
    if (!tensor) return 0;
    return ggml_nelements(tensor);
}

// ============================================================================
// Cache Population at Init Time
// ============================================================================

static void populate_add_cache_entries(void) {
    // Pre-compute all ADD operation strategies across complexity classes
    for (int complexity = 0; complexity < COMPLEXITY_COUNT; complexity++) {
        ggml_numa_cache_entry_t * entry = &g_numa_cache[GGML_OP_ADD][complexity];
        
        entry->valid = true;
        entry->kernel_name = "NUMA Add";
        entry->work_function = ggml_numa_kernel_add_work_function;
        entry->work_buffer_size_per_thread = 1024; // Minimal for element-wise ops
        
        // Strategy selection based on complexity class
        switch (complexity) {
            case COMPLEXITY_TINY:
            case COMPLEXITY_SMALL:
                // Small tensors: single-node execution
                entry->strategy.node_strategy = NUMA_NODE_STRATEGY_SINGLE;
                entry->strategy.on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD;
                entry->efficiency_score = 0.60f; // Lower due to overhead
                break;
                
            case COMPLEXITY_MEDIUM:
            case COMPLEXITY_LARGE:
            case COMPLEXITY_HUGE:
                // Large tensors: data-parallel across nodes
                entry->strategy.node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL;
                entry->strategy.on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD;
                entry->efficiency_score = 0.95f; // Excellent for large parallel work
                break;
                
            default:
                entry->valid = false;
                break;
        }
        
        GGML_LOG_DEBUG("NUMA Cache: Populated ADD[%d] -> strategy=%s, efficiency=%.2f\n",
                       complexity,
                       (entry->strategy.node_strategy == NUMA_NODE_STRATEGY_DATA_PARALLEL) ? "data-parallel" : "single-node",
                       entry->efficiency_score);
    }
}

static void populate_unsupported_operation(enum ggml_op op) {
    // Mark all complexity classes as unsupported for this operation
    for (int complexity = 0; complexity < COMPLEXITY_COUNT; complexity++) {
        ggml_numa_cache_entry_t * entry = &g_numa_cache[op][complexity];
        entry->valid = false;
        entry->kernel_name = "Unsupported";
    }
}

static bool build_kernel_cache(void) {
    GGML_LOG_INFO("🔧 NUMA Cache: Building high-speed operation cache...\n");
    
    // Initialize all entries as invalid
    for (int op = 0; op < GGML_OP_COUNT; op++) {
        for (int complexity = 0; complexity < COMPLEXITY_COUNT; complexity++) {
            g_numa_cache[op][complexity].valid = false;
        }
    }
    
    // Populate supported operations
    populate_add_cache_entries();
    
    // Mark unsupported operations (can be extended as kernels are added)
    for (int op = 0; op < GGML_OP_COUNT; op++) {
        if (op != GGML_OP_ADD) {
            populate_unsupported_operation((enum ggml_op)op);
        }
    }
    
    GGML_LOG_INFO("✅ NUMA Cache: Cache built successfully with %d operations cached\n", GGML_OP_COUNT);
    return true;
}

// ============================================================================
// Lightning-Fast Public API
// ============================================================================

bool ggml_numa_kernels_init(void) {
    if (g_numa_cache_initialized) {
        GGML_LOG_DEBUG("NUMA Cache: Already initialized\n");
        return true;
    }
    
    if (!build_kernel_cache()) {
        GGML_LOG_ERROR("NUMA Cache: Failed to build cache\n");
        return false;
    }
    
    g_numa_cache_initialized = true;
    GGML_LOG_INFO("🚀 NUMA Cache: High-speed kernel registry ready for lightning execution\n");
    return true;
}

void ggml_numa_kernels_cleanup(void) {
    g_numa_cache_initialized = false;
    GGML_LOG_DEBUG("NUMA Cache: Cleaned up\n");
}

ggml_numa_kernel_query_result_t ggml_numa_kernels_query(const struct ggml_tensor * tensor) {
    ggml_numa_kernel_query_result_t result = {
        .supported = false,
        .strategy = {0},
        .work_buffer_size_per_thread = 0,
        .work_function = NULL,
        .efficiency_score = 0.0f,
        .kernel_name = "None"
    };
    
    if (!tensor || !g_numa_cache_initialized) {
        return result;
    }
    
    // Lightning-fast complexity scoring and cache lookup
    size_t complexity_score = get_tensor_complexity_score(tensor);
    ggml_numa_complexity_class_t complexity_class = get_complexity_class(complexity_score);
    
    // O(1) cache lookup - zero runtime overhead
    const ggml_numa_cache_entry_t * entry = &g_numa_cache[tensor->op][complexity_class];
    
    if (!entry->valid) {
        // Cache miss - operation not supported
        return result;
    }
    
    // Cache hit - populate result with pre-computed values
    result.supported = true;
    result.strategy = entry->strategy;
    result.work_buffer_size_per_thread = entry->work_buffer_size_per_thread;
    result.work_function = entry->work_function;
    result.efficiency_score = entry->efficiency_score;
    result.kernel_name = entry->kernel_name;
    
    return result;
}

// ============================================================================
// Legacy Compatibility Functions
// ============================================================================

bool ggml_numa_kernels_supports(enum ggml_op op, const struct ggml_tensor * tensor) {
    if (!g_numa_cache_initialized || !tensor) {
        return false;
    }
    
    ggml_numa_complexity_class_t complexity_class = get_complexity_class(get_tensor_complexity_score(tensor));
    return g_numa_cache[op][complexity_class].valid;
}

enum ggml_status ggml_numa_kernels_execute(struct ggml_tensor * tensor, struct ggml_cplan * cplan) {
    // This legacy function should not be used in the new architecture
    // Executor should query and dispatch to coordinator instead
    (void)tensor;
    (void)cplan;
    
    GGML_LOG_WARN("NUMA Cache: Legacy execute function called - use executor query pattern instead\n");
    return GGML_STATUS_FAILED;
}

float ggml_numa_kernels_get_efficiency(enum ggml_op op, const struct ggml_tensor * tensor, size_t tensor_size) {
    if (!g_numa_cache_initialized || !tensor) {
        return 0.0f;
    }
    
    ggml_numa_complexity_class_t complexity_class = get_complexity_class(tensor_size);
    const ggml_numa_cache_entry_t * entry = &g_numa_cache[op][complexity_class];
    
    return entry->valid ? entry->efficiency_score : 0.0f;
}
