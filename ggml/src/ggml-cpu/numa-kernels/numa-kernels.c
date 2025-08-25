/**
 * @file numa-kernels.c
 * @brief NUMA Kernel Registry Implementation - High-Performance Caching System
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
    if (num_elements < 1048576) return COMPLEXITY_MEDIUM;  // Increased to 1M elements (4MB for float32)
    if (num_elements < 16777216) return COMPLEXITY_LARGE;  // Increased to 16M elements (64MB for float32)
    if (num_elements < 67108864) return COMPLEXITY_HUGE;   // Up to 64M elements (256MB for float32)
    if (num_elements < 536870912) return COMPLEXITY_GIGANTIC_1GB;  // Up to 512M elements (~2GB for float32)
    if (num_elements < 1073741824) return COMPLEXITY_GIGANTIC_2GB; // Up to 1024M elements (~4GB for float32)
    if (num_elements < 2147483648UL) return COMPLEXITY_GIGANTIC_4GB; // Up to 2048M elements (~8GB for float32)
    if (num_elements < 4294967296UL) return COMPLEXITY_GIGANTIC_8GB; // Up to 4096M elements (~16GB for float32)
    return COMPLEXITY_GIGANTIC_16GB;
}

static inline size_t get_tensor_complexity_score(const struct ggml_tensor * tensor) {
    if (!tensor) return 0;
    return ggml_nelements(tensor);
}

// ============================================================================
// Cache Population at Init Time
// ============================================================================

static void populate_unsupported_operation(enum ggml_op op) {
    // Mark all complexity classes as unsupported for this operation
    for (int complexity = 0; complexity < COMPLEXITY_COUNT; complexity++) {
        ggml_numa_cache_entry_t * entry = &g_numa_cache[op][complexity];
        entry->valid = false;
        entry->kernel_name = "Unsupported";
    }
}

static bool build_kernel_cache(void) {
    NUMA_LOG_DEBUG("NUMA Cache: Building high-speed operation cache...");
    GGML_LOG_INFO("🔧 NUMA Cache: Building high-speed operation cache...\n");
    
    // Initialize all entries as invalid
    for (int op = 0; op < GGML_OP_COUNT; op++) {
        for (int complexity = 0; complexity < COMPLEXITY_COUNT; complexity++) {
            g_numa_cache[op][complexity].valid = false;
        }
    }
    
    // Populate supported operations - each kernel provides its own cache entries
    NUMA_LOG_DEBUG("NUMA Cache: Temporarily disabling ADD cache for debugging");
    GGML_LOG_DEBUG("NUMA Cache: Temporarily disabling ADD cache for debugging\n");
    ggml_numa_kernel_add_populate_cache(g_numa_cache[GGML_OP_ADD]); // ADD kernel
    
    // Verify ADD cache entries were populated
    for (int complexity = 0; complexity < COMPLEXITY_COUNT; complexity++) {
        ggml_numa_cache_entry_t * entry = &g_numa_cache[GGML_OP_ADD][complexity];
        NUMA_LOG_DEBUG("NUMA Cache: ADD[%d] valid=%s, kernel=%s", 
               complexity, entry->valid ? "true" : "false", 
               entry->valid ? entry->kernel_name : "N/A");
        GGML_LOG_DEBUG("NUMA Cache: ADD[%d] valid=%s, kernel=%s\n", 
                       complexity, entry->valid ? "true" : "false", 
                       entry->valid ? entry->kernel_name : "N/A");
    }
    
    // Mark unsupported operations (can be extended as kernels are added)
    for (int op = 0; op < GGML_OP_COUNT; op++) {
        if (op != GGML_OP_ADD) {  // Mark all operations except ADD as unsupported 
            populate_unsupported_operation((enum ggml_op)op);
        }
    }
    
    NUMA_LOG_DEBUG("NUMA Cache: Cache built successfully with %d operations cached", GGML_OP_COUNT);
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
    
    if (!tensor) {
        GGML_LOG_DEBUG("NUMA Query: Tensor is NULL\n");
        return result;
    }
    
    if (!g_numa_cache_initialized) {
        GGML_LOG_DEBUG("NUMA Query: Cache not initialized, initializing for op %s\n", ggml_op_name(tensor->op));
        if (!ggml_numa_kernels_init()) {
            GGML_LOG_DEBUG("NUMA Query: Failed to initialize kernel cache for op %s\n", ggml_op_name(tensor->op));
            return result;
        }
    }
    
    // Lightning-fast complexity scoring and cache lookup
    size_t complexity_score = get_tensor_complexity_score(tensor);
    ggml_numa_complexity_class_t complexity_class = get_complexity_class(complexity_score);
    
    NUMA_LOG_VERBOSE("op=%s, complexity_score=%zu, complexity_class=%d", 
           ggml_op_name(tensor->op), complexity_score, complexity_class);
    
    // O(1) cache lookup - zero runtime overhead
    const ggml_numa_cache_entry_t * entry = &g_numa_cache[tensor->op][complexity_class];
    
    NUMA_LOG_VERBOSE("Cache entry valid=%s for op=%s[%d]", 
           entry->valid ? "true" : "false", ggml_op_name(tensor->op), complexity_class);
    
    if (!entry->valid) {
        // Cache miss - operation not supported
        NUMA_LOG_DEBUG("Cache miss for op %s complexity %d", 
               ggml_op_name(tensor->op), complexity_class);
        GGML_LOG_DEBUG("NUMA Query: Cache miss for op %s complexity %d\n", 
                       ggml_op_name(tensor->op), complexity_class);
        return result;
    }
    
    // Cache hit - populate result with pre-computed values
    result.supported = true;
    result.strategy = entry->strategy;
    result.work_buffer_size_per_thread = entry->work_buffer_size_per_thread;
    result.work_function = entry->work_function;
    result.efficiency_score = entry->efficiency_score;
    result.kernel_name = entry->kernel_name;
    
    NUMA_LOG_DEBUG("Cache hit for op %s - kernel=%s, efficiency=%.2f, work_function=%p", 
           ggml_op_name(tensor->op), entry->kernel_name, entry->efficiency_score, entry->work_function);
    
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
