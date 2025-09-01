/*
 * NUMA Shared Utilities Implementation
 * 
 * Implements shared functionality for NUMA system including:
 * - Force strategy environment variable parsing and caching
 * - Strategy override functionality for comprehensive testing
 * - Shared utility functions used across NUMA components
 */

#include "ggml-numa-shared.h"
#include <string.h>
#include <stdlib.h>

// ============================================================================
// Force Strategy System Implementation
// ============================================================================

// Cached force strategy value - parsed once from environment
static ggml_numa_force_strategy_t g_force_strategy = NUMA_FORCE_STRATEGY_DISABLED;
static bool g_force_strategy_initialized = false;

/**
 * Parse and cache force strategy from NUMA_FORCE_STRATEGY environment variable
 * 
 * Environment variable values:
 * - Not set or "0": Disabled (normal threshold-based selection)
 * - "1": Force single-node, single-thread for all operations
 * - "2": Force single-node, multi-thread for all operations  
 * - "3": Force data-parallel, multi-thread for all operations
 * 
 * @return Parsed force strategy mode
 */
static ggml_numa_force_strategy_t ggml_numa_parse_force_strategy(void) {
    const char * env_value = getenv("NUMA_FORCE_STRATEGY");
    
    if (!env_value) {
        NUMA_LOG_DEBUG("NUMA Force Strategy: Environment variable not set, using threshold-based selection");
        return NUMA_FORCE_STRATEGY_DISABLED;
    }
    
    int strategy_int = atoi(env_value);
    
    switch (strategy_int) {
        case 0:
            NUMA_LOG_DEBUG("NUMA Force Strategy: Disabled (threshold-based selection)");
            return NUMA_FORCE_STRATEGY_DISABLED;
            
        case 1:
            NUMA_LOG_DEBUG("NUMA Force Strategy: Forcing SINGLE_NODE + SINGLE_THREAD for all operations");
            return NUMA_FORCE_STRATEGY_SINGLE_SINGLE;
            
        case 2:
            NUMA_LOG_DEBUG("NUMA Force Strategy: Forcing SINGLE_NODE + MULTI_THREAD for all operations");
            return NUMA_FORCE_STRATEGY_SINGLE_MULTI;
            
        case 3:
            NUMA_LOG_DEBUG("NUMA Force Strategy: Forcing DATA_PARALLEL + MULTI_THREAD for all operations");
            return NUMA_FORCE_STRATEGY_DATA_PARALLEL;
            
        default:
            NUMA_LOG_DEBUG("NUMA Force Strategy: Invalid value '%s', using threshold-based selection", env_value);
            return NUMA_FORCE_STRATEGY_DISABLED;
    }
}

ggml_numa_force_strategy_t ggml_numa_get_force_strategy(void) {
    if (!g_force_strategy_initialized) {
        g_force_strategy = ggml_numa_parse_force_strategy();
        g_force_strategy_initialized = true;
    }
    return g_force_strategy;
}

bool ggml_numa_apply_force_strategy_override(ggml_numa_execution_strategy_t * strategy) {
    if (!strategy) {
        NUMA_LOG_ERROR("Cannot apply force strategy override: strategy pointer is NULL");
        return false;
    }
    
    ggml_numa_force_strategy_t force_strategy = ggml_numa_get_force_strategy();
    
    // If force strategy is disabled, don't override
    if (force_strategy == NUMA_FORCE_STRATEGY_DISABLED) {
        return false;
    }
    
    // Store original strategy for logging
    ggml_numa_execution_strategy_t original = *strategy;
    
    // Apply force strategy override
    switch (force_strategy) {
        case NUMA_FORCE_STRATEGY_SINGLE_SINGLE:
            strategy->node_strategy = NUMA_NODE_STRATEGY_SINGLE;
            strategy->on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD;
            break;
            
        case NUMA_FORCE_STRATEGY_SINGLE_MULTI:
            strategy->node_strategy = NUMA_NODE_STRATEGY_SINGLE;
            strategy->on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD;
            break;
            
        case NUMA_FORCE_STRATEGY_DATA_PARALLEL:
            strategy->node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL;
            strategy->on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD;
            break;
            
        default:
            // This should never happen due to validation in parse function
            NUMA_LOG_ERROR("Invalid force strategy mode: %d", force_strategy);
            return false;
    }
    
    // Log the strategy override
    const char * original_node_str = (original.node_strategy == NUMA_NODE_STRATEGY_SINGLE) ? "SINGLE" : "DATA_PARALLEL";
    const char * original_thread_str = (original.on_node_strategy == NUMA_ON_NODE_STRATEGY_SINGLE_THREAD) ? "SINGLE_THREAD" : "MULTI_THREAD";
    const char * new_node_str = (strategy->node_strategy == NUMA_NODE_STRATEGY_SINGLE) ? "SINGLE" : "DATA_PARALLEL";
    const char * new_thread_str = (strategy->on_node_strategy == NUMA_ON_NODE_STRATEGY_SINGLE_THREAD) ? "SINGLE_THREAD" : "MULTI_THREAD";
    
    NUMA_LOG_DEBUG("NUMA Force Strategy Override: %s+%s -> %s+%s", 
                   original_node_str, original_thread_str, new_node_str, new_thread_str);
    
    return true;
}
