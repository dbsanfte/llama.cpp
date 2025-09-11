/**
 * @file numa-kernels.c
 * @brief NUMA Kernel Registry 
 * 
 * This module provides ultra-fast kernel lookups through direct array access.
 * 
 * Architecture:
 * 1. Array 1: g_kernel_cache[GGML_OP_COUNT] - Main storage (sparse, most NULL)
 * 2. Array 2: g_kernel_lookup[GGML_OP_COUNT] - Fast pointers (inference hot path)
 * 3. Kernel registration at startup populates both arrays
 * 4. Inference: Single memory access lookup_table[op_type]
 * 5. Performance: ~2-3 CPU cycles 
 */

#include "numa-kernels.h"
#include "add.h"
#include "mul.h"
#include "div.h"
#include "sub.h"
#include "mul_mat.h"
#include "rope.h"
#include "soft_max.h"
#include "glu.h"
#include "noop.h"
#include "reshape.h"
#include "transpose.h"
#include "view.h"
#include "permute.h"
#include "rms_norm.h"
#include "cpy.h"
#include "get_rows.h"
#include "../ggml-impl.h"
#include "../ggml-vec-numa.h"

// ============================================================================
// Direct Array Cache System - Maximum Performance
// ============================================================================

// Global kernel cache with direct array access - NO HASH COMPUTATION!
static ggml_numa_kernel_array_cache_t g_kernel_array_cache = {0};

/**
 * Initialize the global kernel array cache system
 * Called once at startup to prepare direct access arrays
 */
enum ggml_status ggml_numa_init_kernel_array_cache(void) {
    if (g_kernel_array_cache.cache_initialized) {
        return GGML_STATUS_SUCCESS;  // Already initialized
    }
    
    // Initialize Array 1: Main cache storage (clear all entries)
    memset(g_kernel_array_cache.cache_storage, 0, sizeof(g_kernel_array_cache.cache_storage));
    
    // Initialize Array 2: Fast lookup table (all pointers NULL)
    memset(g_kernel_array_cache.lookup_table, 0, sizeof(g_kernel_array_cache.lookup_table));
    
    g_kernel_array_cache.num_registered_ops = 0;
    g_kernel_array_cache.cache_initialized = true;
    
    NUMA_LOG_DEBUG("🚀 NUMA Kernel Cache: Direct array system initialized (size: %d operations)", GGML_OP_COUNT);
    return GGML_STATUS_SUCCESS;
}

/**
 * Register a kernel strategy with the direct array cache system
 * Called by each kernel at startup to populate both storage and lookup arrays
 */
enum ggml_status ggml_numa_register_kernel_strategy(
    enum ggml_op op_type, 
    const ggml_numa_kernel_strategy_array_t * strategy_array,
    const ggml_numa_kernel_work_funcs_t * work_funcs,
    const ggml_numa_kernel_aggregation_funcs_t * agg_funcs,
    ggml_numa_kernel_query_fn_t query_fn,
    ggml_numa_kernel_work_buffer_calc_fn_t work_buffer_calc_fn,
    bool supported,
    bool is_noop) {    if (!g_kernel_array_cache.cache_initialized) {
        enum ggml_status init_result = ggml_numa_init_kernel_array_cache();
        if (init_result != GGML_STATUS_SUCCESS) {
            return init_result;
        }
    }
    
    // Validate operation type bounds
    if (op_type >= GGML_OP_COUNT || op_type <= GGML_OP_NONE) {
        NUMA_LOG_ERROR("Invalid operation type %d (must be 1 <= op_type < %d)", 
                      (int)op_type, GGML_OP_COUNT);
        return GGML_STATUS_FAILED;
    }
    
    // Check for existing registration
    if (g_kernel_array_cache.lookup_table[op_type] != NULL) {
        NUMA_LOG_WARN("Operation %d already registered, overwriting", (int)op_type);
    }
    
    // Store in Array 1: Main cache storage
    ggml_numa_kernel_cache_entry_t * entry = &g_kernel_array_cache.cache_storage[op_type];
    entry->op_type = op_type;
    entry->supported = supported;               // Store the supported flag
    entry->is_noop = is_noop;                   // Store the no-op flag
    entry->query_fn = query_fn;                 // Store the query function pointer
    entry->work_buffer_calc_fn = work_buffer_calc_fn;  // Store the work buffer calculation function pointer
    
    if (strategy_array && strategy_array->valid) {
        entry->strategy_array = *strategy_array;
    } else {
        entry->strategy_array.valid = false;
    }
    
    if (work_funcs && work_funcs->valid) {
        entry->work_funcs = *work_funcs;
    } else {
        entry->work_funcs.valid = false;
    }
    
    if (agg_funcs && agg_funcs->valid) {
        entry->agg_funcs = *agg_funcs;
    } else {
        entry->agg_funcs.valid = false;
    }
    
    // Set Array 2: Fast lookup pointer (this is what we query in inference!)
    // Only set lookup pointer if kernel is supported
    if (supported) {
        g_kernel_array_cache.lookup_table[op_type] = entry;
    } else {
        g_kernel_array_cache.lookup_table[op_type] = NULL;  // Disabled kernel
        NUMA_LOG_DEBUG("🚫 Kernel for operation %d is disabled", (int)op_type);
    }
    
    g_kernel_array_cache.num_registered_ops++;
    
    NUMA_LOG_DEBUG("✅ Registered kernel strategy for operation %d (direct array access)", (int)op_type);
    return GGML_STATUS_SUCCESS;
}

/**
 * Check if a kernel is registered and supported
 */
bool ggml_numa_is_kernel_supported(enum ggml_op op_type) {
    if (!g_kernel_array_cache.cache_initialized) {
        return false;
    }
    
    if (op_type >= GGML_OP_COUNT) {
        return false;
    }
    
    // Check if there's a valid lookup entry (only set for supported kernels)
    return g_kernel_array_cache.lookup_table[op_type] != NULL;
}

/**
 * Check if a kernel is a no-op kernel that skips coordinator dispatch
 */
bool ggml_numa_is_kernel_noop(enum ggml_op op_type) {
    if (!g_kernel_array_cache.cache_initialized) {
        return false;
    }
    
    if (op_type >= GGML_OP_COUNT) {
        return false;
    }
    
    const ggml_numa_kernel_cache_entry_t * entry = g_kernel_array_cache.lookup_table[op_type];
    return entry != NULL && entry->is_noop;
}

/**
 * Direct array lookup - ultra-fast single memory access
 * Returns complete kernel information or NULL if unsupported
 * Performance: ~2-3 CPU cycles (single array access + NULL check)
 */
const ggml_numa_kernel_cache_entry_t * ggml_numa_lookup_kernel_direct(enum ggml_op op_type) {
    if (op_type >= GGML_OP_COUNT || op_type <= GGML_OP_NONE) {
        return NULL;  // Invalid operation type
    }
    
    if (!g_kernel_array_cache.cache_initialized) {
        return NULL;  // Cache not initialized
    }
    
    // Lightning-fast direct array access - this is the magic!
    return g_kernel_array_cache.lookup_table[op_type];  // Single memory access!
}

/**
 * Array-based strategy lookup - direct access using operation type as index
 * Returns execution strategy based on operation type and element count
 */
const ggml_numa_execution_strategy_t * ggml_numa_lookup_strategy_direct(
    enum ggml_op op_type,
    size_t element_count) {
    
    static ggml_numa_execution_strategy_t default_strategy = NUMA_STRATEGY_SINGLE_NODE;
    
    // Direct array access - no hash computation!
    const ggml_numa_kernel_cache_entry_t * entry = ggml_numa_lookup_kernel_direct(op_type);
    if (!entry || !entry->strategy_array.valid) {
        return &default_strategy;
    }
    
    // Use the same fast strategy selection as before
    static ggml_numa_execution_strategy_t selected_strategy;
    selected_strategy = numa_select_strategy_fast(&entry->strategy_array, element_count);
    return &selected_strategy;
}

/**
 * Array-based aggregation function lookup - direct access using operation type as index
 * Returns aggregation function for the given strategy, or NULL if not needed
 */
enum ggml_status (*ggml_numa_lookup_aggregation_direct(
    enum ggml_op op, const ggml_numa_execution_strategy_t * strategy
))(void *, int, struct ggml_tensor *, struct ggml_cplan *) {
    // Implementation would go here based on strategy
    // For now, most operations don't need aggregation
    GGML_UNUSED(op);
    GGML_UNUSED(strategy);
    return NULL;
}

/**
 * Get work function for specific strategy from cache entry
 * Ultra-fast O(1) lookup with strategy-based function selection
 */
ggml_numa_work_function_t ggml_numa_get_work_function_from_cache(
    const ggml_numa_kernel_cache_entry_t * cache_entry,
    const ggml_numa_execution_strategy_t * strategy) {
    
    if (!cache_entry || !cache_entry->work_funcs.valid) {
        return NULL;
    }
    
    // Select appropriate work function based on strategy
    if (*strategy == NUMA_STRATEGY_DATA_PARALLEL) {
        return cache_entry->work_funcs.data_parallel_fn;
    } else if (*strategy == NUMA_STRATEGY_SINGLE_THREAD) {
        return cache_entry->work_funcs.single_single_fn;
    } else {
        return cache_entry->work_funcs.single_multi_fn;
    }
}

/**
 * Get work buffer calculation function from cache entry
 * Returns NULL if operation doesn't need work buffers
 */
ggml_numa_kernel_work_buffer_calc_fn_t ggml_numa_get_work_buffer_calc_from_cache(
    const ggml_numa_kernel_cache_entry_t * cache_entry) {
    
    if (!cache_entry) {
        return NULL;
    }
    
    return cache_entry->work_buffer_calc_fn;
}

/**
 * Get kernel name from cache entry for debugging/logging
 * Returns static string, safe to use in logging
 */
const char * ggml_numa_get_kernel_name_from_cache(
    const ggml_numa_kernel_cache_entry_t * cache_entry) {
    
    if (!cache_entry) {
        return "Unknown";
    }
    
    return cache_entry->kernel_name;
}

/**
 * Array-based work function lookup - direct access using operation type as index
 * Returns work function pointer for execution based on operation and strategy
 */
ggml_numa_work_function_t ggml_numa_lookup_work_function_direct(
    enum ggml_op op_type,
    const ggml_numa_execution_strategy_t * strategy) {
    
    if (!strategy) {
        return NULL;
    }
    
    // Direct array access - no hash computation!
    const ggml_numa_kernel_cache_entry_t * entry = ggml_numa_lookup_kernel_direct(op_type);
    if (!entry || !entry->work_funcs.valid) {
        return NULL;
    }
    
    // Use the same fast function selection as before
    return numa_get_work_func_fast(&entry->work_funcs, *strategy);
}

// ============================================================================
// Kernel Initialization System
// ============================================================================

static bool g_numa_kernels_initialized = false;

/**
 * Initialize all NUMA kernels and register their strategies
 * Called once at system startup
 */
enum ggml_status ggml_numa_kernels_init(void) {
    // FORCED DEBUG: Always log NUMA kernels initialization
    NUMA_LOG_DEBUG("NUMA kernels initialization called, already_initialized=%s", 
                   g_numa_kernels_initialized ? "YES" : "NO");
    
    if (g_numa_kernels_initialized) {
        return GGML_STATUS_SUCCESS;  // Already initialized
    }
    
    // Initialize the direct array cache system first
    enum ggml_status cache_result = ggml_numa_init_kernel_array_cache();
    if (cache_result != GGML_STATUS_SUCCESS) {
        NUMA_LOG_ERROR("Failed to initialize kernel array cache");
        return cache_result;
    }
    
    // Initialize NUMA vector operations for SIMD transcendental functions
    ggml_vec_numa_init();
    NUMA_LOG_DEBUG("NUMA vector operations initialized with runtime SIMD dispatch");
    
    // Register each kernel using their own registration functions
    // This allows kernels to define their own strategies and function pointers
    
    // Register ROPE kernel 
    NUMA_REGISTER_KERNEL(rope);

    // Register MUL_MAT kernel
    NUMA_REGISTER_KERNEL(mul_mat);
    
    // Register the binary op kernels:
    NUMA_REGISTER_KERNEL(add); 
    NUMA_REGISTER_KERNEL(mul);
    NUMA_REGISTER_KERNEL(div);
    NUMA_REGISTER_KERNEL(sub);
    
    // Register data movement kernels:
    NUMA_REGISTER_KERNEL(cpy);
    NUMA_REGISTER_KERNEL(get_rows);
    
    // Register reduction kernels:
    NUMA_REGISTER_KERNEL(rms_norm);
    NUMA_REGISTER_KERNEL(soft_max);
    
    // Register activation kernels:
    NUMA_REGISTER_KERNEL(glu);
    
    // Register view operations (metadata-only, no-op kernels):
    NUMA_REGISTER_KERNEL(reshape);
    NUMA_REGISTER_KERNEL(transpose);
    NUMA_REGISTER_KERNEL(view);
    NUMA_REGISTER_KERNEL(permute);
    
    // Register NOOP kernel for performance testing
    NUMA_REGISTER_KERNEL(noop);
    
    g_numa_kernels_initialized = true;
    
    NUMA_LOG_DEBUG("✅ NUMA Kernels initialized with direct array cache system");
    NUMA_LOG_DEBUG("   Registered %zu operations using direct array access", 
                  g_kernel_array_cache.num_registered_ops);
    
    return GGML_STATUS_SUCCESS;
}

// ============================================================================
// Modern Direct Array Lookup System
// ============================================================================

/**
 * Cleanup function for the direct array cache system
 */
void ggml_numa_kernels_cleanup(void) {
    g_numa_kernels_initialized = false;
    g_kernel_array_cache.cache_initialized = false;
    g_kernel_array_cache.num_registered_ops = 0;
    
    // Clear both arrays
    memset(g_kernel_array_cache.cache_storage, 0, sizeof(g_kernel_array_cache.cache_storage));
    memset(g_kernel_array_cache.lookup_table, 0, sizeof(g_kernel_array_cache.lookup_table));
    
    NUMA_LOG_DEBUG("✅ NUMA Kernels: O(1) hash table system cleaned up");
}

ggml_numa_execution_strategy_t ggml_numa_kernels_query(const struct ggml_tensor * tensor) {
    ggml_numa_execution_strategy_t strategy = {0};
    
    if (!tensor) {
        GGML_LOG_DEBUG("NUMA Query: Tensor is NULL\n");
        return strategy;
    }
    
    // Ensure kernel system is initialized
    if (!g_numa_kernels_initialized) {
        GGML_LOG_DEBUG("NUMA Query: Initializing kernel system for op %s\n", ggml_op_name(tensor->op));
        enum ggml_status init_result = ggml_numa_kernels_init();
        if (init_result != GGML_STATUS_SUCCESS) {
            GGML_LOG_DEBUG("NUMA Query: Failed to initialize kernel system for op %s\n", ggml_op_name(tensor->op));
            return strategy;
        }
    }
    
    // Direct cache lookup - ultra-fast single memory access!
    const ggml_numa_kernel_cache_entry_t * entry = ggml_numa_lookup_kernel_direct(tensor->op);
    if (!entry || !entry->query_fn) {
        NUMA_LOG_DEBUG("NUMA Query: No kernel found for operation %s", ggml_op_name(tensor->op));
        return strategy;
    }
    
    // Call the kernel's query function directly through the cached pointer!
    // Note: Individual kernels now return only strategy, not full result structure
    NUMA_LOG_DEBUG("NUMA Query: Calling cached query function for operation %s", ggml_op_name(tensor->op));
    return entry->query_fn(tensor);
}
