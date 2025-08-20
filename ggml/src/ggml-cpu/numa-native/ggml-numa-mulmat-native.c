/*
 * ggml-Native NUMA MUL_MAT Implementation
 * 
 * This demonstrates how to adapt our NUMA operations to work WITH ggml's
 * infrastructure rather than around it.
 */

#include "ggml-numa-native.h"
#include "ggml-cpu.h"
#include "ggml.h"

// ============================================================================
// Type-Aware NUMA Tensor Management
// ============================================================================

/**
 * Create a NUMA-local view of a tensor using ggml's view system
 * This respects ggml's stride calculations and memory layout
 */
static struct ggml_tensor* ggml_numa_create_tensor_view(
    struct ggml_context* ctx,
    struct ggml_tensor* original,
    int numa_node,
    int total_numa_nodes,
    enum ggml_numa_split_strategy strategy) {
    
    switch (strategy) {
        case NUMA_SPLIT_ROWS: {
            // Split along the row dimension (typical for matrix operations)
            int64_t total_rows = original->ne[1];
            int64_t rows_per_node = total_rows / total_numa_nodes;
            int64_t row_start = numa_node * rows_per_node;
            int64_t row_end = (numa_node == total_numa_nodes - 1) ? 
                             total_rows : row_start + rows_per_node;
            
            // Calculate offset using ggml's stride system
            size_t offset = row_start * original->nb[1];
            
            // Create dimensions for the view
            int64_t ne[GGML_MAX_DIMS] = {
                original->ne[0],                    // Width unchanged
                row_end - row_start,                // NUMA slice of rows
                original->ne[2],                    // Depth unchanged  
                original->ne[3]                     // Batch unchanged
            };
            
            // Use ggml's view system (preserves memory layout)
            return ggml_view_nd(ctx, original, 4, ne, original->nb, offset);
        }
        
        case NUMA_SPLIT_ELEMENTS: {
            // Split along the element dimension for vector operations
            int64_t total_elements = ggml_nelements(original);
            int64_t elements_per_node = total_elements / total_numa_nodes;
            int64_t elem_start = numa_node * elements_per_node;
            int64_t elem_end = (numa_node == total_numa_nodes - 1) ?
                              total_elements : elem_start + elements_per_node;
            
            size_t offset = elem_start * ggml_type_size(original->type);
            
            int64_t ne[GGML_MAX_DIMS] = {elem_end - elem_start, 1, 1, 1};
            
            return ggml_view_1d(ctx, original, ne[0], offset);
        }
        
        default:
            return original;  // No splitting
    }
}

/**
 * Ensure tensor compatibility using ggml's type conversion system
 */
static struct ggml_tensor* ggml_numa_ensure_type_compatibility(
    struct ggml_context* ctx,
    struct ggml_tensor* tensor,
    enum ggml_type target_type) {
    
    if (tensor->type == target_type) {
        return tensor;  // Already compatible
    }
    
    // Create a new tensor with the target type
    struct ggml_tensor* target_tensor = ggml_new_tensor(ctx, target_type, 
                                                       tensor->n_dims, tensor->ne);
    
    // Use ggml's copy operation (handles type conversion automatically)
    return ggml_cpy(ctx, tensor, target_tensor);
}

// ============================================================================
// ggml-Native NUMA MUL_MAT Implementation
// ============================================================================

/**
 * Create a NUMA-local computational subgraph for MUL_MAT operation
 */
static struct ggml_cgraph* ggml_numa_create_mulmat_subgraph(
    struct ggml_context* ctx,
    struct ggml_tensor* src0,
    struct ggml_tensor* src1,
    struct ggml_tensor* dst,
    int numa_node,
    int total_numa_nodes) {
    
    // Create computational graph
    struct ggml_cgraph* subgraph = ggml_new_graph(ctx);
    
    // Get type requirements using ggml's type system
    const struct ggml_type_traits_cpu* traits = ggml_get_type_traits_cpu(src0->type);
    enum ggml_type required_src1_type = traits->vec_dot_type;
    
    // Create NUMA-local views of the tensors
    struct ggml_tensor* local_src0 = ggml_numa_create_tensor_view(
        ctx, src0, numa_node, total_numa_nodes, NUMA_SPLIT_ROWS);
    
    struct ggml_tensor* local_src1 = ggml_numa_create_tensor_view(
        ctx, src1, numa_node, total_numa_nodes, NUMA_SPLIT_NONE);  // Broadcast across nodes
    
    struct ggml_tensor* local_dst = ggml_numa_create_tensor_view(
        ctx, dst, numa_node, total_numa_nodes, NUMA_SPLIT_ROWS);
    
    // Ensure type compatibility using ggml's conversion system
    local_src1 = ggml_numa_ensure_type_compatibility(ctx, local_src1, required_src1_type);
    
    // Create the MUL_MAT operation within ggml's graph system
    struct ggml_tensor* result = ggml_mul_mat(ctx, local_src0, local_src1);
    
    // Set the destination tensor data pointer to our local slice
    result->data = local_dst->data;
    
    // Add to the computational graph
    ggml_build_forward_expand(subgraph, result);
    
    return subgraph;
}

/**
 * ggml-Native NUMA work function for MUL_MAT operations
 */
static int ggml_numa_work_function_mulmat_native(void* context) {
    const ggml_numa_work_context_t* ctx = (const ggml_numa_work_context_t*)context;
    struct ggml_tensor* tensor = ctx->tensor;
    struct ggml_tensor* src0 = tensor->src[0];
    struct ggml_tensor* src1 = tensor->src[1];
    
    // Create local ggml context for this NUMA node's computation
    struct ggml_init_params init_params = {
        .mem_size = 32 * 1024 * 1024,  // 32MB for local operations and type conversions
        .mem_buffer = NULL,
        .no_alloc = false
    };
    
    struct ggml_context* local_ctx = ggml_init(init_params);
    if (!local_ctx) {
        GGML_LOG_ERROR("Failed to create local ggml context for NUMA node %d\n", ctx->numa_node);
        return -1;
    }
    
    // Create NUMA-local computational subgraph
    struct ggml_cgraph* subgraph = ggml_numa_create_mulmat_subgraph(
        local_ctx, src0, src1, tensor, ctx->numa_node, ctx->max_numa_nodes);
    
    if (!subgraph) {
        GGML_LOG_ERROR("Failed to create computational subgraph for NUMA node %d\n", ctx->numa_node);
        ggml_free(local_ctx);
        return -1;
    }
    
    // Create execution plan using ggml's planning system
    struct ggml_cplan plan = ggml_graph_plan(subgraph, ctx->thread_count, ctx->threadpool);
    
    // Allocate work buffer if needed (ggml handles the sizing automatically)
    if (plan.work_size > 0) {
        plan.work_data = malloc(plan.work_size);
        if (!plan.work_data) {
            GGML_LOG_ERROR("Failed to allocate work buffer (%zu bytes) for NUMA node %d\n", 
                          plan.work_size, ctx->numa_node);
            ggml_free(local_ctx);
            return -1;
        }
    }
    
    // Execute the subgraph using ggml's proven computation system
    enum ggml_status result = ggml_graph_compute(subgraph, &plan);
    
    // Cleanup
    if (plan.work_data) {
        free(plan.work_data);
    }
    ggml_free(local_ctx);
    
    if (result != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("ggml computation failed for NUMA node %d\n", ctx->numa_node);
        return -1;
    }
    
    GGML_LOG_DEBUG("ggml-native NUMA computation completed successfully for node %d\n", ctx->numa_node);
    return 0;
}

// ============================================================================
// Dispatcher Integration
// ============================================================================

/**
 * Enhanced dispatcher handler that can choose between native and fallback
 */
bool ggml_numa_dispatch_mulmat_enhanced(
    struct ggml_tensor* tensor,
    struct ggml_cplan* cplan,
    float* efficiency,
    enum ggml_numa_node_strategy* strategy,
    ggml_numa_work_function_t* work_function) {
    
    struct ggml_tensor* src0 = tensor->src[0];
    struct ggml_tensor* src1 = tensor->src[1];
    
    if (!src0 || !src1) {
        return false;
    }
    
    // Get type information using ggml's type system
    const struct ggml_type_traits_cpu* traits = ggml_get_type_traits_cpu(src0->type);
    bool quantized_operation = ggml_is_quantized(src0->type);
    
    // Calculate tensor size for strategy decision
    int64_t total_elements = ggml_nelements(tensor);
    bool large_tensor = total_elements > 32768;  // Threshold for NUMA benefit
    
    // Strategy selection based on tensor characteristics
    if (large_tensor && quantized_operation) {
        // Use ggml-native NUMA for large quantized operations
        *efficiency = 0.85f;  // Good efficiency with ggml integration overhead
        *strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL;
        *work_function = ggml_numa_work_function_mulmat_native;
        
        GGML_LOG_DEBUG("Selected ggml-native NUMA execution for %s operation (elements: %ld)\n",
                      ggml_type_name(src0->type), total_elements);
        return true;
        
    } else if (large_tensor && !quantized_operation) {
        // Use traditional NUMA for large F32 operations (if working)
        *efficiency = 0.90f;
        *strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL;
        *work_function = ggml_numa_work_function_mulmat_chunk;  // Existing implementation
        
        GGML_LOG_DEBUG("Selected traditional NUMA execution for F32 operation (elements: %ld)\n", total_elements);
        return true;
        
    } else {
        // Use single-node execution for small tensors
        *efficiency = 0.95f;
        *strategy = NUMA_NODE_STRATEGY_SINGLE;
        *work_function = ggml_numa_work_function_mulmat_single;
        
        GGML_LOG_DEBUG("Selected single-node execution for small tensor (elements: %ld)\n", total_elements);
        return true;
    }
}

/**
 * Work buffer size calculation for ggml-native operations
 */
size_t ggml_numa_get_mulmat_work_buffer_size_native(struct ggml_tensor* tensor) {
    struct ggml_tensor* src0 = tensor->src[0];
    struct ggml_tensor* src1 = tensor->src[1];
    
    if (!src0 || !src1) {
        return 0;
    }
    
    // Get type requirements
    const struct ggml_type_traits_cpu* traits = ggml_get_type_traits_cpu(src0->type);
    enum ggml_type vec_dot_type = traits->vec_dot_type;
    
    size_t work_size = 0;
    
    // Calculate conversion buffer size if needed
    if (src1->type != vec_dot_type) {
        // Size for type conversion (per NUMA node)
        int64_t elements_per_node = ggml_nelements(src1) / 2;  // Assume 2 NUMA nodes
        work_size += ggml_type_size(vec_dot_type) * elements_per_node;
    }
    
    // Add ggml context overhead (estimated)
    work_size += 16 * 1024 * 1024;  // 16MB for ggml context and temporary tensors
    
    GGML_LOG_DEBUG("ggml-native work buffer size: %zu bytes\n", work_size);
    return work_size;
}
