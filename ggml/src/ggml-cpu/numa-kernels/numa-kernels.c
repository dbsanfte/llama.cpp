/**
 * @file numa-kernels.c
 * @brief NUMA Kernel Registry Implementation - Direct Array System
 * 
 * This module provides ultra-fast kernel lookups through direct array access.
 * No hash functions, no collisions, no complexity - just pure performance.
 * 
 * Architecture:
 * 1. Array 1: g_kernel_cache[GGML_OP_COUNT] - Main storage (sparse, most NULL)
 * 2. Array 2: g_kernel_lookup[GGML_OP_COUNT] - Fast pointers (inference hot path)
 * 3. Kernel registration at startup populates both arrays
 * 4. Inference: Single memory access lookup_table[op_type]
 * 5. Performance: ~2-3 CPU cycles vs ~5-8 for hash table
 */

#include "numa-kernels.h"
#include "add.h"
#include "mul.h"
#include "mul_mat.h"
#include "cpy.h"
#include "cont.h"
#include "rms_norm.h"
#include "rope.h"
#include "permute.h"
#include "glu.h"
#include "reshape.h"
#include "view.h"
#include "soft_max.h"
#include "transpose.h"
#include "noop.h"
#include "../ggml-impl.h"

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
    entry->supported = supported;  // Store the supported flag
    entry->query_fn = query_fn;    // Store the query function pointer
    entry->is_noop = is_noop;      // Store the no-op flag
    
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
    
    static ggml_numa_execution_strategy_t default_strategy = {
        .node_strategy = NUMA_NODE_STRATEGY_SINGLE,
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
    };
    
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
 * Returns aggregation function pointer based on operation and strategy
 */
enum ggml_status (*ggml_numa_lookup_aggregation_direct(
    enum ggml_op op_type,
    const ggml_numa_execution_strategy_t * strategy
))(void *, int, struct ggml_tensor *, struct ggml_cplan *) {
    
    if (!strategy) {
        return NULL;
    }
    
    // Direct array access - no hash computation!
    const ggml_numa_kernel_cache_entry_t * entry = ggml_numa_lookup_kernel_direct(op_type);
    if (!entry || !entry->agg_funcs.valid) {
        return NULL;
    }
    
    // Use the same fast function selection as before
    return numa_get_aggregation_func_fast(&entry->agg_funcs, strategy);
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
    return numa_get_work_func_fast(&entry->work_funcs, strategy);
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
    if (g_numa_kernels_initialized) {
        return GGML_STATUS_SUCCESS;  // Already initialized
    }
    
    // Initialize the direct array cache system first
    enum ggml_status cache_result = ggml_numa_init_kernel_array_cache();
    if (cache_result != GGML_STATUS_SUCCESS) {
        NUMA_LOG_ERROR("Failed to initialize kernel array cache");
        return cache_result;
    }
    
    // Register each kernel using their own registration functions
    // This allows kernels to define their own strategies and function pointers
    
    // TEMPORARILY DISABLED: All NUMA kernel registrations disabled to fix threading model
    // We need to fix coordinator threading before re-enabling kernels
    
    // Enable ADD kernel for multi-threaded data-parallel testing
    NUMA_REGISTER_KERNEL(add);
    
    // Enable MUL kernel for testing
    // Register MUL kernel
    NUMA_REGISTER_KERNEL(mul);

    // Register CPY kernel - DISABLED due to memory corruption issues
    //NUMA_REGISTER_KERNEL(cpy);

    // Register CONT kernel for tensor contiguity operations
    NUMA_REGISTER_KERNEL(cont);

    // Enable MUL_MAT kernel for matrix multiplication
    // Register MUL_MAT kernel
    NUMA_REGISTER_KERNEL(mul_mat);
    
    // Enable RMS_NORM kernel for normalization operations
    // Register RMS_NORM kernel
    NUMA_REGISTER_KERNEL(rms_norm);
    
    // Enable ROPE kernel for rotary position embeddings
    // Register ROPE kernel
    NUMA_REGISTER_KERNEL(rope);
    
    // Register PERMUTE kernel for tensor dimension permutation
    NUMA_REGISTER_KERNEL(permute);
    
    // Register GLU kernel for gated linear unit operations
    NUMA_REGISTER_KERNEL(glu);
    
    // Register RESHAPE kernel for tensor shape transformation
    NUMA_REGISTER_KERNEL(reshape);
    
    // Register VIEW kernel for tensor view operations
    NUMA_REGISTER_KERNEL(view);
    
    // Register SOFT_MAX kernel for softmax activation
    NUMA_REGISTER_KERNEL(soft_max);
    
    // Register TRANSPOSE kernel for tensor dimension swapping
    NUMA_REGISTER_KERNEL(transpose);
    
    // Register NOOP kernel for performance testing
    //ggml_numa_register_noop_kernels();
    
    g_numa_kernels_initialized = true;
    
    NUMA_LOG_DEBUG("✅ NUMA Kernels initialized with direct array cache system");
    NUMA_LOG_DEBUG("   Registered %zu operations using direct array access", 
                  g_kernel_array_cache.num_registered_ops);
    
    return GGML_STATUS_SUCCESS;
}

// ============================================================================
// Modern Direct Array Lookup System
// ============================================================================

/*
 * Modern kernel strategy lookup using direct array system
 * Ultra-fast single memory access for optimal performance
 */
ggml_numa_kernel_query_result_t ggml_numa_kernels_strategy_lookup(const struct ggml_tensor * tensor) {
    // Default fallback result for safety
    ggml_numa_kernel_query_result_t default_result = {
        .kernel_name = "Fallback-Standard",
        .aggregation_policy = GGML_NUMA_AGGREGATION_NONE,
        .work_buffer_size_per_thread = 0,
        .efficiency_score = 0.5f,
        .strategy = {
            .node_strategy = NUMA_NODE_STRATEGY_SINGLE,
            .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
        },
        .aggregation_function = NULL,
        .supported = false,
        .work_function = NULL
    };
    
    if (!tensor) {
        NUMA_LOG_ERROR("NULL tensor in strategy lookup");
        return default_result;
    }
    
    // Ensure kernel system is initialized
    if (!g_numa_kernels_initialized) {
        enum ggml_status init_result = ggml_numa_kernels_init();
        if (init_result != GGML_STATUS_SUCCESS) {
            NUMA_LOG_ERROR("Failed to initialize kernel system");
            return default_result;
        }
    }
    
    // Calculate total element count
    size_t total_elements = ggml_nelements(tensor);
    
    // Direct array strategy lookup - single memory access!
    const ggml_numa_execution_strategy_t * strategy = ggml_numa_lookup_strategy_direct(tensor->op, total_elements);
    if (!strategy) {
        NUMA_LOG_DEBUG("No strategy found for op %s", ggml_op_name(tensor->op));
        return default_result;
    }

    // Get function pointers using direct array lookups
    void * work_func = ggml_numa_lookup_work_function_direct(tensor->op, strategy);
    if (!work_func) {
        NUMA_LOG_DEBUG("No work function found for op %s", ggml_op_name(tensor->op));
        return default_result;
    }

    // Create optimized result based on strategy (only if we have valid functions)
    ggml_numa_kernel_query_result_t result = default_result;
    result.supported = true;
    result.strategy = *strategy;
    result.work_function = work_func;
    result.aggregation_function = ggml_numa_lookup_aggregation_direct(tensor->op, strategy);
    
    // Generic kernel information for direct array lookup
    result.kernel_name = "NUMA Generic (Direct Array Lookup)";
    result.efficiency_score = 0.85f;  // Conservative generic efficiency
    result.aggregation_policy = GGML_NUMA_AGGREGATION_NONE;
    
    NUMA_LOG_DEBUG("Direct array lookup: op=%d, elements=%zu, strategy=%s/%s, efficiency=%.2f", 
                  (int)tensor->op, total_elements,
                  strategy->node_strategy == NUMA_NODE_STRATEGY_SINGLE ? "single" : "data-parallel",
                  strategy->on_node_strategy == NUMA_ON_NODE_STRATEGY_SINGLE_THREAD ? "single-thread" : "multi-thread",
                  result.efficiency_score);
    
    return result;
}

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

ggml_numa_kernel_query_result_t ggml_numa_kernels_query(const struct ggml_tensor * tensor) {
    ggml_numa_kernel_query_result_t result = {
        .supported = false,
        .strategy = {0},
        .work_buffer_size_per_thread = 0,
        .work_function = NULL,
        .efficiency_score = 0.0f,
        .kernel_name = "Unsupported"
    };
    
    if (!tensor) {
        GGML_LOG_DEBUG("NUMA Query: Tensor is NULL\n");
        return result;
    }
    
    // Ensure kernel system is initialized
    if (!g_numa_kernels_initialized) {
        GGML_LOG_DEBUG("NUMA Query: Initializing kernel system for op %s\n", ggml_op_name(tensor->op));
        enum ggml_status init_result = ggml_numa_kernels_init();
        if (init_result != GGML_STATUS_SUCCESS) {
            GGML_LOG_DEBUG("NUMA Query: Failed to initialize kernel system for op %s\n", ggml_op_name(tensor->op));
            return result;
        }
    }
    
    // Direct cache lookup - ultra-fast single memory access!
    const ggml_numa_kernel_cache_entry_t * entry = ggml_numa_lookup_kernel_direct(tensor->op);
    if (!entry || !entry->query_fn) {
        NUMA_LOG_DEBUG("NUMA Query: No kernel found for operation %s", ggml_op_name(tensor->op));
        return result;
    }
    
    // Call the kernel's query function directly through the cached pointer!
    NUMA_LOG_DEBUG("NUMA Query: Calling cached query function for operation %s", ggml_op_name(tensor->op));
    return entry->query_fn(tensor);
}

// ============================================================================
// Force Strategy Helper Function
// ============================================================================

bool ggml_numa_apply_kernel_force_strategy(ggml_numa_kernel_query_result_t * result,
                                           const char * op_name,
                                           ggml_numa_work_function_t single_single_fn,
                                           ggml_numa_work_function_t single_multi_fn,
                                           ggml_numa_work_function_t data_parallel_fn) {
    if (!result) {
        NUMA_LOG_ERROR("Cannot apply force strategy: result pointer is NULL");
        return false;
    }
    
    // Apply force strategy override to execution strategy
    bool strategy_overridden = ggml_numa_apply_force_strategy_override(&result->strategy);
    
    if (strategy_overridden) {
        // Update work function and kernel name based on overridden strategy
        if (result->strategy.node_strategy == NUMA_NODE_STRATEGY_SINGLE) {
            if (result->strategy.on_node_strategy == NUMA_ON_NODE_STRATEGY_SINGLE_THREAD) {
                if (single_single_fn) {
                    result->work_function = single_single_fn;
                    result->kernel_name = "NUMA (Forced Single/Single)";
                }
            } else {
                if (single_multi_fn) {
                    result->work_function = single_multi_fn;
                    result->kernel_name = "NUMA (Forced Single/Multi)";
                }
            }
        } else {
            // Data parallel strategy
            if (data_parallel_fn) {
                result->work_function = data_parallel_fn;
                result->kernel_name = "NUMA (Forced Data-Parallel)";
            }
        }
        
        // Add operation name to kernel name for clarity
        static char kernel_name_buffer[256];
        snprintf(kernel_name_buffer, sizeof(kernel_name_buffer), "%s %s", op_name, result->kernel_name);
        result->kernel_name = kernel_name_buffer;
        
        NUMA_LOG_DEBUG("%s: Strategy overridden by NUMA_FORCE_STRATEGY", op_name);
    }
    
    return strategy_overridden;
}
