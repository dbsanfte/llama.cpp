/**
 * @file numa-kernels.c
 * @brief NUMA Kernel Registry Implementation - Modern Query-Based System
 * 
 * This module provides efficient kernel lookups through threshold-based
 * query functions for each supported operation.
 * 
 * Architecture:
 * 1. Simple initialization flag
 * 2. Direct kernel-specific query functions
 * 3. No pre-computed cache or complexity overhead
 */

#include "numa-kernels.h"
#include "add.h"
#include "mul_mat.h"
#include "../ggml-impl.h"

// ============================================================================
// Modern Query-Based Kernel Registry - Initialization Flag
// ============================================================================

static bool g_numa_kernels_initialized = false;

// ============================================================================
// Modern Query-Based NUMA Kernel System  
// ============================================================================

bool ggml_numa_kernels_init(void) {
    if (g_numa_kernels_initialized) {
        GGML_LOG_DEBUG("NUMA Kernels: Already initialized\n");
        return true;
    }
    
    NUMA_LOG_DEBUG("NUMA Kernels: Initializing query-based kernel system");
    GGML_LOG_INFO("🚀 NUMA Kernels: Initializing modern query-based system\n");
    
    // Mark as initialized - the query-based system doesn't need cache pre-population
    g_numa_kernels_initialized = true;
    
    NUMA_LOG_DEBUG("NUMA Kernels: Query-based kernel system ready");
    GGML_LOG_INFO("✅ NUMA Kernels: Query-based kernel system ready for operation\n");
    return true;
}

void ggml_numa_kernels_cleanup(void) {
    g_numa_kernels_initialized = false;
    GGML_LOG_DEBUG("NUMA Kernels: Cleaned up\n");
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
        if (!ggml_numa_kernels_init()) {
            GGML_LOG_DEBUG("NUMA Query: Failed to initialize kernel system for op %s\n", ggml_op_name(tensor->op));
            return result;
        }
    }
    
    // Use kernel-specific threshold-based query functions
    switch (tensor->op) {
        case GGML_OP_ADD:
            result = ggml_numa_kernel_add_query(tensor);
            if (result.supported) {
                NUMA_LOG_DEBUG("ADD query: Using threshold-based strategy - %s", result.kernel_name);
                return result;
            }
            break;
            
        case GGML_OP_MUL_MAT:
            result = ggml_numa_kernel_mul_mat_query(tensor);
            if (result.supported) {
                NUMA_LOG_DEBUG("MUL_MAT query: Using threshold-based strategy - %s", result.kernel_name);
                return result;
            }
            break;
            
        default:
            // Operation not supported by NUMA kernels
            NUMA_LOG_DEBUG("Operation %s not supported by NUMA kernels", ggml_op_name(tensor->op));
            break;
    }
    
    // If we reach here, the operation is not supported
    NUMA_LOG_DEBUG("NUMA Query: Operation %s not supported", ggml_op_name(tensor->op));
    return result;
}
