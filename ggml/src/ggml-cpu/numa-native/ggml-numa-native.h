#pragma once

#include "ggml.h"
#include "ggml-cpu.h"
#include "../ggml-numa-coordinator.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// ggml-Native NUMA Types and Enums
// ============================================================================

/**
 * Strategy for splitting tensors across NUMA nodes
 */
enum ggml_numa_split_strategy {
    NUMA_SPLIT_NONE,      // No splitting (broadcast)
    NUMA_SPLIT_ROWS,      // Split along row dimension  
    NUMA_SPLIT_ELEMENTS,  // Split along flattened elements
    NUMA_SPLIT_BATCHES    // Split along batch dimension
};

/**
 * ggml-Native NUMA operation context
 */
struct ggml_numa_native_context {
    struct ggml_context* ggml_ctx;           // Local ggml context
    struct ggml_cgraph* subgraph;            // NUMA-local computational subgraph
    struct ggml_cplan* plan;                 // Execution plan
    enum ggml_numa_split_strategy strategy;  // Tensor splitting strategy
    size_t work_buffer_size;                 // Required work buffer size
};

// ============================================================================
// ggml-Native NUMA Core Functions
// ============================================================================

/**
 * Create a NUMA-local view of a tensor using ggml's view system
 */
struct ggml_tensor* ggml_numa_create_tensor_view(
    struct ggml_context* ctx,
    struct ggml_tensor* original,
    int numa_node,
    int total_numa_nodes,
    enum ggml_numa_split_strategy strategy);

/**
 * Ensure tensor type compatibility using ggml's conversion system
 */
struct ggml_tensor* ggml_numa_ensure_type_compatibility(
    struct ggml_context* ctx,
    struct ggml_tensor* tensor,
    enum ggml_type target_type);

/**
 * Create a NUMA-local computational subgraph for any operation
 */
struct ggml_cgraph* ggml_numa_create_operation_subgraph(
    struct ggml_context* ctx,
    struct ggml_tensor* operation,
    int numa_node,
    int total_numa_nodes);

// ============================================================================
// Operation-Specific ggml-Native Implementations
// ============================================================================

/**
 * ggml-Native NUMA MUL_MAT work function
 */
int ggml_numa_work_function_mulmat_native(void* context);

/**
 * Enhanced MUL_MAT dispatcher with native/fallback selection
 */
bool ggml_numa_dispatch_mulmat_enhanced(
    struct ggml_tensor* tensor,
    struct ggml_cplan* cplan,
    float* efficiency,
    enum ggml_numa_node_strategy* strategy,
    ggml_numa_work_function_t* work_function);

/**
 * Work buffer size calculation for ggml-native operations
 */
size_t ggml_numa_get_mulmat_work_buffer_size_native(struct ggml_tensor* tensor);

// ============================================================================
// Future Operation Extensions
// ============================================================================

/**
 * Template for creating ggml-native NUMA operations
 */
#define GGML_NUMA_NATIVE_OPERATION(op_name) \\
    int ggml_numa_work_function_##op_name##_native(void* context); \\
    bool ggml_numa_dispatch_##op_name##_enhanced( \\
        struct ggml_tensor* tensor, \\
        struct ggml_cplan* cplan, \\
        float* efficiency, \\
        enum ggml_numa_node_strategy* strategy, \\
        ggml_numa_work_function_t* work_function); \\
    size_t ggml_numa_get_##op_name##_work_buffer_size_native(struct ggml_tensor* tensor);

// Declare future operations
GGML_NUMA_NATIVE_OPERATION(add)
GGML_NUMA_NATIVE_OPERATION(rms_norm)
GGML_NUMA_NATIVE_OPERATION(soft_max)

// ============================================================================
// Integration Utilities
// ============================================================================

/**
 * Determine if an operation should use ggml-native NUMA vs traditional NUMA
 */
bool ggml_numa_should_use_native_implementation(
    struct ggml_tensor* tensor,
    size_t tensor_size_threshold);

/**
 * Get optimal NUMA splitting strategy for a given operation and tensor
 */
enum ggml_numa_split_strategy ggml_numa_get_optimal_split_strategy(
    enum ggml_op operation,
    struct ggml_tensor* tensor);

/**
 * Validate that a tensor is suitable for NUMA splitting
 */
bool ggml_numa_validate_tensor_for_splitting(
    struct ggml_tensor* tensor,
    enum ggml_numa_split_strategy strategy,
    int numa_nodes);

#ifdef __cplusplus
}
#endif
