/*
 * Enhanced NUMA Dispatcher Integration
 * 
 * This shows how to integrate the ggml-native NUMA approach into our existing dispatcher
 * while maintaining backward compatibility with the fallback system.
 */

// Add to ggml-numa-operation-dispatch.c

#include "numa-native/ggml-numa-native.h"

// ============================================================================
// Enhanced Dispatcher Logic
// ============================================================================

/**
 * Enhanced operation dispatch that can choose between multiple strategies:
 * 1. ggml-native NUMA (new) - for quantized operations
 * 2. Traditional NUMA (existing) - for F32 operations that work
 * 3. Fallback (current) - for reliability when others fail
 */
static enum ggml_status ggml_numa_dispatch_operation_enhanced(
    struct ggml_tensor* tensor, 
    struct ggml_cplan* cplan) {
    
    if (!tensor) {
        return GGML_STATUS_FAILED;
    }
    
    // Strategy selection variables
    float efficiency = 0.0f;
    enum ggml_numa_node_strategy numa_strategy = NUMA_NODE_STRATEGY_SINGLE;
    ggml_numa_work_function_t work_function = NULL;
    bool use_enhanced = false;
    
    // Try enhanced dispatchers for specific operations
    switch (tensor->op) {
        case GGML_OP_MUL_MAT: {
            // Use enhanced MUL_MAT dispatcher
            use_enhanced = ggml_numa_dispatch_mulmat_enhanced(
                tensor, cplan, &efficiency, &numa_strategy, &work_function);
            break;
        }
        
        case GGML_OP_ADD: {
            // Check if we should use ggml-native for quantized ADD operations
            if (tensor->src[0] && ggml_is_quantized(tensor->src[0]->type)) {
                // Future: ggml_numa_dispatch_add_enhanced(...)
                use_enhanced = false;  // Not implemented yet
            } else {
                // Use traditional NUMA for F32 ADD (if working)
                efficiency = 0.95f;
                numa_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL;
                work_function = ggml_numa_work_function_add_chunk;  // Existing
                use_enhanced = true;
            }
            break;
        }
        
        // Add more operations as they're enhanced...
        
        default:
            use_enhanced = false;  // Use fallback for unimplemented operations
            break;
    }
    
    // If enhanced dispatch succeeded, execute with chosen strategy
    if (use_enhanced && work_function) {
        GGML_LOG_DEBUG("Using enhanced NUMA dispatch for %s (efficiency: %.2f, strategy: %d)\\n",
                      ggml_op_name(tensor->op), efficiency, numa_strategy);
        
        // Execute using the selected work function and strategy
        return ggml_numa_execute_with_strategy(tensor, cplan, numa_strategy, work_function);
    }
    
    // Fallback to proven ggml system for reliability
    GGML_LOG_DEBUG("Using ggml fallback for %s operation\\n", ggml_op_name(tensor->op));
    return ggml_numa_fallback_execute(tensor, cplan);
}

/**
 * Execute operation with the selected NUMA strategy
 */
static enum ggml_status ggml_numa_execute_with_strategy(
    struct ggml_tensor* tensor,
    struct ggml_cplan* cplan,
    enum ggml_numa_node_strategy strategy,
    ggml_numa_work_function_t work_function) {
    
    // Get the coordinator manager
    struct ggml_numa_coordinator_manager* manager = ggml_numa_coordinator_manager_get_global();
    if (!manager || !ggml_numa_coordinator_manager_is_active(manager)) {
        GGML_LOG_WARN("NUMA coordinator not available, falling back to ggml\\n");
        return ggml_numa_fallback_execute(tensor, cplan);
    }
    
    // Create work context
    ggml_numa_work_context_t context = {
        .tensor = tensor,
        .cplan = cplan,
        .numa_node = 0,  // Will be set by coordinator
        .max_numa_nodes = ggml_numa_coordinator_manager_get_numa_node_count(manager),
        .thread_count = cplan ? cplan->n_threads : 1,
        .threadpool = cplan ? cplan->threadpool : NULL
    };
    
    // Execute based on strategy
    switch (strategy) {
        case NUMA_NODE_STRATEGY_SINGLE:
            return ggml_numa_execute_single_node(manager, tensor, &context, work_function);
            
        case NUMA_NODE_STRATEGY_DATA_PARALLEL:
            return ggml_numa_execute_data_parallel(manager, tensor, &context, work_function);
            
        default:
            GGML_LOG_ERROR("Unknown NUMA strategy: %d\\n", strategy);
            return ggml_numa_fallback_execute(tensor, cplan);
    }
}

// ============================================================================
// Integration Points
// ============================================================================

/**
 * Modified main dispatch function that uses enhanced logic
 */
enum ggml_status ggml_numa_dispatch_operation(
    struct ggml_tensor* tensor, 
    struct ggml_cplan* cplan) {
    
    if (!tensor) {
        GGML_LOG_ERROR("Invalid tensor for NUMA dispatch\\n");
        return GGML_STATUS_FAILED;
    }
    
    // Feature flag: Enable/disable enhanced dispatch
    static bool enhanced_dispatch_enabled = true;  // Can be controlled via environment variable
    
    if (enhanced_dispatch_enabled) {
        // Try enhanced dispatch first
        enum ggml_status result = ggml_numa_dispatch_operation_enhanced(tensor, cplan);
        
        if (result == GGML_STATUS_SUCCESS) {
            return result;
        }
        
        // If enhanced dispatch fails, fall back to proven system
        GGML_LOG_WARN("Enhanced dispatch failed for %s, using fallback\\n", ggml_op_name(tensor->op));
    }
    
    // Always have fallback as the most reliable option
    return ggml_numa_fallback_execute(tensor, cplan);
}

/**
 * Enhanced work buffer size calculation
 */
size_t ggml_numa_get_work_buffer_size_enhanced(struct ggml_tensor* tensor) {
    if (!tensor) {
        return 0;
    }
    
    size_t buffer_size = 0;
    
    switch (tensor->op) {
        case GGML_OP_MUL_MAT:
            // Check if we would use ggml-native implementation
            if (tensor->src[0] && ggml_is_quantized(tensor->src[0]->type)) {
                buffer_size = ggml_numa_get_mulmat_work_buffer_size_native(tensor);
            } else {
                buffer_size = ggml_numa_get_work_buffer_size_traditional_mulmat(tensor);
            }
            break;
            
        // Add other operations...
        
        default:
            // Use existing work buffer calculation
            buffer_size = ggml_numa_get_work_buffer_size_existing(tensor);
            break;
    }
    
    GGML_LOG_DEBUG("Work buffer size for %s: %zu bytes\\n", ggml_op_name(tensor->op), buffer_size);
    return buffer_size;
}

// ============================================================================
// Gradual Migration Framework
// ============================================================================

/**
 * Operation readiness levels for enhanced NUMA
 */
enum ggml_numa_operation_readiness {
    NUMA_OP_NOT_READY,      // Use fallback only
    NUMA_OP_EXPERIMENTAL,   // ggml-native available but may fall back
    NUMA_OP_STABLE,         // ggml-native preferred
    NUMA_OP_PRODUCTION      // ggml-native only, no fallback needed
};

/**
 * Check readiness level for enhanced NUMA implementation
 */
static enum ggml_numa_operation_readiness ggml_numa_get_operation_readiness(enum ggml_op operation) {
    switch (operation) {
        case GGML_OP_MUL_MAT:
            return NUMA_OP_EXPERIMENTAL;  // ggml-native implemented but needs testing
            
        case GGML_OP_ADD:
            return NUMA_OP_NOT_READY;     // Traditional NUMA works for F32, needs ggml-native for quantized
            
        // Future operations will start as EXPERIMENTAL and graduate to STABLE/PRODUCTION
        
        default:
            return NUMA_OP_NOT_READY;
    }
}

/**
 * Decide whether to use enhanced implementation based on readiness and configuration
 */
static bool ggml_numa_should_use_enhanced_for_operation(
    enum ggml_op operation,
    struct ggml_tensor* tensor) {
    
    enum ggml_numa_operation_readiness readiness = ggml_numa_get_operation_readiness(operation);
    
    switch (readiness) {
        case NUMA_OP_NOT_READY:
            return false;
            
        case NUMA_OP_EXPERIMENTAL: {
            // Use enhanced for quantized operations (where traditional NUMA fails)
            bool is_quantized = tensor->src[0] && ggml_is_quantized(tensor->src[0]->type);
            return is_quantized;
        }
        
        case NUMA_OP_STABLE:
            return true;  // Always use enhanced when stable
            
        case NUMA_OP_PRODUCTION:
            return true;  // Production ready
            
        default:
            return false;
    }
}
