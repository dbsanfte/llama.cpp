# ggml-Native NUMA Implementation Design

## Overview
Instead of bypassing ggml's infrastructure, we can create a NUMA-aware layer that works **within** ggml's type system and computational graph framework.

## Core Principle: ggml-Native NUMA Operations

### Current Problem
```c
// Our current approach: Direct kernel bypass
vec_dot(ne00, &dst_col[ir0], 0, src0_row + ir0*nb01, 0, src1_col, 0, 1);
// ❌ Bypasses ggml's type system, quantization handling, memory management
```

### Proposed Solution
```c
// ggml-Native NUMA approach: Leverage ggml's infrastructure
struct ggml_numa_operation_context {
    struct ggml_context * ggml_ctx;           // ggml context for memory management
    struct ggml_cgraph * subgraph;            // NUMA-local computational subgraph
    const struct ggml_type_traits_cpu * traits;  // Type system integration
    struct ggml_cplan * local_plan;           // NUMA-local execution plan
};
```

## Key Design Changes

### 1. Type System Integration

Instead of manual type handling:
```c
// ❌ Current: Manual type conversion
if (src1->type != vec_dot_type && thread_data->work_buffer) {
    const struct ggml_type_traits_cpu * src1_traits = ggml_get_type_traits_cpu(vec_dot_type);
    ggml_from_float_t const from_float = src1_traits->from_float;
    // Manual conversion loop...
}
```

Use ggml's graph-based type conversion:
```c
// ✅ Proposed: ggml-native type conversion
struct ggml_tensor* ggml_numa_ensure_compatible_type(
    struct ggml_context* ctx,
    struct ggml_tensor* tensor,
    enum ggml_type target_type) {
    
    if (tensor->type == target_type) {
        return tensor;  // Already compatible
    }
    
    // Create ggml conversion operation (part of computational graph)
    struct ggml_tensor* converted = ggml_cpy(ctx, 
        tensor, 
        ggml_new_tensor(ctx, target_type, tensor->n_dims, tensor->ne));
    
    return converted;  // ggml handles the conversion automatically
}
```

### 2. NUMA-Aware Graph Splitting

Instead of direct operation calls:
```c
// ❌ Current: Direct operation execution bypassing graph
for (int64_t ir0 = ir0_start; ir0 < ir0_end; ir0++) {
    vec_dot(ne00, &dst_col[ir0], 0, src0_row + ir0*nb01, 0, src1_col, 0, 1);
}
```

Use NUMA-local subgraphs:
```c
// ✅ Proposed: NUMA-aware graph partitioning
struct ggml_cgraph* ggml_numa_create_local_subgraph(
    struct ggml_context* ctx,
    struct ggml_tensor* original_op,
    int numa_node,
    int total_numa_nodes) {
    
    struct ggml_cgraph* subgraph = ggml_new_graph(ctx);
    
    // Create NUMA-local view of tensors (slice data appropriately)
    struct ggml_tensor* local_src0 = ggml_numa_create_local_view(ctx, original_op->src[0], numa_node, total_numa_nodes);
    struct ggml_tensor* local_src1 = ggml_numa_create_local_view(ctx, original_op->src[1], numa_node, total_numa_nodes);
    struct ggml_tensor* local_dst = ggml_numa_create_local_view(ctx, original_op, numa_node, total_numa_nodes);
    
    // Ensure type compatibility using ggml's type system
    const struct ggml_type_traits_cpu* traits = ggml_get_type_traits_cpu(local_src0->type);
    local_src1 = ggml_numa_ensure_compatible_type(ctx, local_src1, traits->vec_dot_type);
    
    // Create the operation within ggml's graph system
    struct ggml_tensor* result = ggml_mul_mat(ctx, local_src0, local_src1);
    
    // Add to subgraph
    ggml_build_forward_expand(subgraph, result);
    
    return subgraph;
}
```

### 3. Memory Layout Compliance

Respect ggml's tensor structure:
```c
// ✅ Proposed: ggml-compliant tensor slicing
struct ggml_tensor* ggml_numa_create_local_view(
    struct ggml_context* ctx,
    struct ggml_tensor* tensor,
    int numa_node, 
    int total_numa_nodes) {
    
    // Calculate NUMA slice using ggml's dimension system
    int64_t total_rows = tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
    int64_t rows_per_node = total_rows / total_numa_nodes;
    int64_t row_start = numa_node * rows_per_node;
    int64_t row_end = (numa_node == total_numa_nodes - 1) ? total_rows : row_start + rows_per_node;
    
    // Create view using ggml's view system (respects stride calculations)
    size_t offset = row_start * tensor->nb[1];  // Use ggml's stride system
    
    int64_t ne[GGML_MAX_DIMS] = {tensor->ne[0], row_end - row_start, 1, 1};
    size_t  nb[GGML_MAX_DIMS] = {tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]};
    
    return ggml_view_nd(ctx, tensor, 4, ne, nb, offset);
}
```

### 4. Execution Framework

```c
// ✅ Proposed: ggml-native NUMA execution
static int ggml_numa_work_function_mulmat_native(void* context) {
    const ggml_numa_work_context_t* ctx = context;
    struct ggml_tensor* tensor = ctx->tensor;
    
    // Create NUMA-local ggml context
    struct ggml_init_params init_params = {
        .mem_size = 16 * 1024 * 1024,  // 16MB for local operations
        .mem_buffer = NULL,
        .no_alloc = false
    };
    struct ggml_context* local_ctx = ggml_init(init_params);
    
    // Create NUMA-local subgraph using ggml's infrastructure
    struct ggml_cgraph* subgraph = ggml_numa_create_local_subgraph(
        local_ctx, tensor, ctx->numa_node, ctx->max_numa_nodes);
    
    // Create execution plan using ggml's planning system
    struct ggml_cplan plan = ggml_graph_plan(subgraph, ctx->thread_count, ctx->threadpool);
    
    // Execute using ggml's proven computation system
    enum ggml_status result = ggml_graph_compute(subgraph, &plan);
    
    // Cleanup
    if (plan.work_data) free(plan.work_data);
    ggml_free(local_ctx);
    
    return (result == GGML_STATUS_SUCCESS) ? 0 : -1;
}
```

## Benefits of ggml-Native Approach

### ✅ **Leverages ggml's Infrastructure**
- Automatic type conversion and quantization handling
- Proper memory layout and stride calculations
- Backend compatibility (CPU/CUDA/etc.)
- Graph optimization and planning

### ✅ **Maintains NUMA Benefits**
- Data distribution across NUMA nodes
- Memory-local processing
- Thread affinity and coordination
- Performance scaling

### ✅ **Reliability**
- Uses proven ggml mathematical kernels
- Proper error handling and resource management
- Type safety and validation
- Memory leak prevention

## Implementation Strategy

### Phase 1: Hybrid Approach
- Keep current fallback for immediate reliability
- Implement ggml-native NUMA for specific operations (starting with F32 MUL_MAT)
- Gradual migration operation by operation

### Phase 2: Full Integration
- Replace direct kernel calls with ggml subgraph execution
- Integrate with ggml's work buffer planning
- Add comprehensive type system support

### Phase 3: Optimization
- NUMA-aware graph partitioning
- Memory affinity optimization
- Cross-NUMA synchronization minimization

## Code Structure

```
ggml/src/ggml-cpu/numa-native/
├── ggml-numa-graph.c          # NUMA-aware graph operations
├── ggml-numa-views.c          # Tensor view creation and slicing
├── ggml-numa-types.c          # Type system integration
├── ggml-numa-execution.c      # ggml-native execution framework
└── ggml-numa-planning.c       # Work buffer and plan management
```

This approach gives us the best of both worlds: NUMA performance benefits with ggml's reliability and type system integration.
