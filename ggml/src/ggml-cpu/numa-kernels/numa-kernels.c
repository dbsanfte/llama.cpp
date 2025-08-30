/**
 * @file numa-kernels.c
 * @brief NUMA Kernel Registry Implementation - O(1) Hash Table System
 * 
 * This module provides ultra-fast O(1) kernel strategy lookups through
 * hash table-based caching with simple threshold arrays.
 * 
 * Architecture:
 * 1. Global hash table: OP_TYPE -> strategy_array + function_pointers
 * 2. Kernel registration at startup provides threshold arrays
 * 3. O(1) lookups with simple threshold comparisons
 * 4. No complex search structures or cache levels
 */

#include "numa-kernels.h"
#include "add.h"
#include "mul.h"
#include "mul_mat.h"
#include "cpy.h"
#include "../ggml-impl.h"

// ============================================================================
// O(1) Hash Table Strategy Cache System
// ============================================================================

// Global strategy cache with O(1) hash table access
static ggml_numa_strategy_cache_t g_strategy_cache = {0};

/**
 * Initialize the global strategy cache system
 * Called once at startup to prepare hash tables
 */
enum ggml_status ggml_numa_init_strategy_cache(void) {
    if (g_strategy_cache.cache_initialized) {
        return GGML_STATUS_SUCCESS;  // Already initialized
    }
    
    // Initialize all hash table entries as invalid
    for (size_t i = 0; i < NUMA_OP_HASH_TABLE_SIZE; i++) {
        g_strategy_cache.entries[i].initialized = false;
        g_strategy_cache.entries[i].op_type = GGML_OP_NONE;
        g_strategy_cache.entries[i].strategy_array.valid = false;
        g_strategy_cache.entries[i].work_funcs.valid = false;
    }
    
    g_strategy_cache.num_registered_ops = 0;
    g_strategy_cache.cache_initialized = true;
    
    NUMA_LOG_DEBUG("🚀 NUMA Strategy Cache: O(1) hash table system initialized");
    return GGML_STATUS_SUCCESS;
}

/**
 * Register a kernel strategy with the cache system
 * Called by each kernel at startup to provide threshold arrays and function pointers
 */
enum ggml_status ggml_numa_register_kernel_strategy(
    enum ggml_op op_type,
    const ggml_numa_kernel_strategy_array_t * strategy_array,
    const ggml_numa_kernel_work_funcs_t * work_funcs,
    const ggml_numa_kernel_aggregation_funcs_t * agg_funcs) {
    
    if (!g_strategy_cache.cache_initialized) {
        enum ggml_status init_result = ggml_numa_init_strategy_cache();
        if (init_result != GGML_STATUS_SUCCESS) {
            return init_result;
        }
    }
    
    // Calculate hash table index - direct operation type mapping
    size_t hash_idx = numa_op_hash(op_type);
    if (hash_idx >= NUMA_OP_HASH_TABLE_SIZE) {
        NUMA_LOG_ERROR("Operation type %d exceeds hash table size %d", 
                      (int)op_type, NUMA_OP_HASH_TABLE_SIZE);
        return GGML_STATUS_FAILED;
    }
    
    // Check for existing registration
    if (g_strategy_cache.entries[hash_idx].initialized) {
        NUMA_LOG_WARN("Operation %d already registered, overwriting", (int)op_type);
    }
    
    // Store strategy array and function pointers
    ggml_numa_strategy_cache_entry_t * entry = &g_strategy_cache.entries[hash_idx];
    entry->op_type = op_type;
    entry->initialized = true;
    
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
    
    g_strategy_cache.num_registered_ops++;
    
    NUMA_LOG_DEBUG("✅ Registered kernel strategy for operation %d (hash_idx=%zu)", 
                  (int)op_type, hash_idx);
    return GGML_STATUS_SUCCESS;
}

/**
 * O(1) strategy lookup - lightning-fast hash table access  
 * Returns strategy based on operation type and element count
 */
const ggml_numa_execution_strategy_t * ggml_numa_lookup_strategy_fast(
    enum ggml_op op_type,
    size_t element_count) {
    
    static ggml_numa_execution_strategy_t default_strategy = {
        .node_strategy = NUMA_NODE_STRATEGY_SINGLE,
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
    };
    
    if (!g_strategy_cache.cache_initialized) {
        return &default_strategy;
    }
    
    // O(1) hash table lookup
    size_t hash_idx = numa_op_hash(op_type);
    if (hash_idx >= NUMA_OP_HASH_TABLE_SIZE) {
        return &default_strategy;
    }
    
    const ggml_numa_strategy_cache_entry_t * entry = &g_strategy_cache.entries[hash_idx];
    if (!entry->initialized || !entry->strategy_array.valid) {
        return &default_strategy;
    }
    
    // O(1) strategy selection using inline threshold comparison
    static ggml_numa_execution_strategy_t selected_strategy;
    selected_strategy = numa_select_strategy_fast(&entry->strategy_array, element_count);
    return &selected_strategy;
}

/**
 * O(1) aggregation function lookup - ultra-fast hash table access  
 * Returns function pointer for aggregation based on operation and strategy
 */
enum ggml_status (*ggml_numa_lookup_aggregation_fast(
    enum ggml_op op_type,
    const ggml_numa_execution_strategy_t * strategy
))(void *, int, struct ggml_tensor *, struct ggml_cplan *) {
    
    if (!g_strategy_cache.cache_initialized || !strategy) {
        return NULL;
    }
    
    // O(1) hash table lookup
    size_t hash_idx = numa_op_hash(op_type);
    if (hash_idx >= NUMA_OP_HASH_TABLE_SIZE) {
        return NULL;
    }
    
    const ggml_numa_strategy_cache_entry_t * entry = &g_strategy_cache.entries[hash_idx];
    if (!entry->initialized || !entry->agg_funcs.valid) {
        return NULL;
    }
    
    // O(1) function pointer selection using inline strategy lookup
    return numa_get_aggregation_func_fast(&entry->agg_funcs, strategy);
}

/**
 * O(1) work function lookup - ultra-fast hash table access
 * Returns work function pointer for execution based on operation and strategy
 */
ggml_numa_work_function_t ggml_numa_lookup_work_function_fast(
    enum ggml_op op_type,
    const ggml_numa_execution_strategy_t * strategy) {
    
    if (!g_strategy_cache.cache_initialized || !strategy) {
        return NULL;
    }
    
    // O(1) hash table lookup
    size_t hash_idx = numa_op_hash(op_type);
    if (hash_idx >= NUMA_OP_HASH_TABLE_SIZE) {
        return NULL;
    }
    
    const ggml_numa_strategy_cache_entry_t * entry = &g_strategy_cache.entries[hash_idx];
    if (!entry->initialized || !entry->work_funcs.valid) {
        return NULL;
    }
    
    // O(1) function pointer selection using inline strategy lookup
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
    
    // Initialize the O(1) strategy cache system first
    enum ggml_status cache_result = ggml_numa_init_strategy_cache();
    if (cache_result != GGML_STATUS_SUCCESS) {
        NUMA_LOG_ERROR("Failed to initialize strategy cache");
        return cache_result;
    }
    
    // Register each kernel using their own registration functions
    // This allows kernels to define their own strategies and function pointers
    
    // Register ADD kernel
    NUMA_REGISTER_KERNEL(add);
    
    // Register MUL kernel
    NUMA_REGISTER_KERNEL(mul);

    // Register CPY kernel
    //NUMA_REGISTER_KERNEL(cpy);

    // Register MUL_MAT kernel
    NUMA_REGISTER_KERNEL(mul_mat);
    
    g_numa_kernels_initialized = true;
    
    NUMA_LOG_DEBUG("✅ NUMA Kernels initialized with O(1) hash table strategy system");
    NUMA_LOG_DEBUG("   Registered %zu operations using simplified macro registration", 
                  g_strategy_cache.num_registered_ops);
    
    return GGML_STATUS_SUCCESS;
}

// ============================================================================
// Modern O(1) Hash Table Lookup System
// ============================================================================

/*
 * Modern kernel strategy lookup using O(1) hash table system
 * Ultra-fast threshold-based lookups for optimal performance
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
    
    // O(1) strategy lookup using hash table
    const ggml_numa_execution_strategy_t * strategy = ggml_numa_lookup_strategy_fast(tensor->op, total_elements);
    if (!strategy) {
        NUMA_LOG_DEBUG("No strategy found for op %s", ggml_op_name(tensor->op));
        return default_result;
    }

    // Get function pointers using O(1) lookups
    void * work_func = ggml_numa_lookup_work_function_fast(tensor->op, strategy);
    if (!work_func) {
        NUMA_LOG_DEBUG("No work function found for op %s", ggml_op_name(tensor->op));
        return default_result;
    }

    // Create optimized result based on strategy (only if we have valid functions)
    ggml_numa_kernel_query_result_t result = default_result;
    result.supported = true;
    result.strategy = *strategy;
    result.work_function = work_func;
    result.aggregation_function = ggml_numa_lookup_aggregation_fast(tensor->op, strategy);    switch (tensor->op) {
        case GGML_OP_ADD:
            result.kernel_name = "NUMA ADD (O(1) Fast-Lookup)";
            result.efficiency_score = 0.99f;
            result.aggregation_policy = GGML_NUMA_AGGREGATION_NONE;
            break;
            
        case GGML_OP_MUL_MAT:
            result.kernel_name = "NUMA MUL_MAT (O(1) Fast-Lookup)";
            result.efficiency_score = 0.92f;
            result.aggregation_policy = GGML_NUMA_AGGREGATION_NONE;
            break;
            
        default:
            result.kernel_name = "NUMA Generic (O(1) Fast-Lookup)";
            result.efficiency_score = 0.8f;
            break;
    }
    
    NUMA_LOG_DEBUG("O(1) Strategy lookup: op=%d, elements=%zu, strategy=%s/%s, efficiency=%.2f", 
                  (int)tensor->op, total_elements,
                  strategy->node_strategy == NUMA_NODE_STRATEGY_SINGLE ? "single" : "data-parallel",
                  strategy->on_node_strategy == NUMA_ON_NODE_STRATEGY_SINGLE_THREAD ? "single-thread" : "multi-thread",
                  result.efficiency_score);
    
    return result;
}

/**
 * Cleanup function for the O(1) hash table system
 */
void ggml_numa_kernels_cleanup(void) {
    g_numa_kernels_initialized = false;
    g_strategy_cache.cache_initialized = false;
    g_strategy_cache.num_registered_ops = 0;
    
    // Clear all hash table entries
    for (size_t i = 0; i < NUMA_OP_HASH_TABLE_SIZE; i++) {
        g_strategy_cache.entries[i].initialized = false;
        g_strategy_cache.entries[i].op_type = GGML_OP_NONE;
        g_strategy_cache.entries[i].strategy_array.valid = false;
        g_strategy_cache.entries[i].work_funcs.valid = false;
        g_strategy_cache.entries[i].agg_funcs.valid = false;
    }
    
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
    
    // Use lightning-fast strategy lookup with threshold caching
    switch (tensor->op) {
        case GGML_OP_ADD:
            return ggml_numa_kernel_add_query(tensor);
            
        case GGML_OP_MUL:
            return ggml_numa_kernel_mul_query(tensor);
            
        case GGML_OP_CPY:
            return ggml_numa_kernel_cpy_query(tensor);
            
        case GGML_OP_MUL_MAT:
            return ggml_numa_kernel_mul_mat_query(tensor);
            
        default:
            // Operation not supported by NUMA kernels
            NUMA_LOG_DEBUG("Operation %s not supported by NUMA kernels", ggml_op_name(tensor->op));
            break;
    }
    
    // If we reach here, the operation is not supported
    NUMA_LOG_DEBUG("NUMA Query: Operation %s not supported", ggml_op_name(tensor->op));
    return result;
}
