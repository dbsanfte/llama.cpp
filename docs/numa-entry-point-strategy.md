# NUMA-Native Graph Computation: Replacing the Entry Point

## Current Problem: Infinite Recursion
```
llama-context.cpp
  ↓
ggml_numa_dispatch_compute_graph()  ← WE ARE HERE
  ↓
Per-operation dispatch
  ↓
ggml_numa_fallback_execute()
  ↓
ggml_graph_compute() [NEW GRAPH!]
  ↓
ggml-cpu.c compute functions
  ↓
Could recurse back to dispatcher... 💥 INFINITE RECURSION
```

## Solution: Complete Graph Management
```
llama-context.cpp
  ↓
ggml_numa_dispatch_compute_graph()  ← WE BECOME THE GRAPH EXECUTOR
  ↓
NUMA-aware graph analysis and planning
  ↓
Direct operation execution using ggml building blocks
  ↓
NO RECURSION - we manage the entire computation ourselves
```

## Key Integration Points

### 1. llama-context.cpp Entry Point (Unchanged)
```cpp
// In llama-context.cpp::graph_compute()
#ifdef GGML_NUMA_MIRROR
    if (numa_coordinator && ggml_get_numa_strategy() != GGML_NUMA_STRATEGY_DISABLED) {
        int result = ggml_numa_dispatch_compute_graph(gf, n_threads);  // ← OUR FUNCTION
        if (result == 0) {
            return GGML_STATUS_SUCCESS;  // ← WE HANDLED EVERYTHING
        }
        // Fallback to backend scheduler if we fail
    }
#endif
```

### 2. Our Complete Graph Execution Function
```c
int ggml_numa_dispatch_compute_graph(struct ggml_cgraph * cgraph, int n_threads) {
    // Step 1: Initialize NUMA infrastructure
    if (!setup_numa_infrastructure(n_threads)) {
        return -1; // Let llama-context use backend scheduler
    }
    
    // Step 2: Analyze graph for NUMA opportunities
    struct numa_graph_plan * plan = analyze_graph_for_numa(cgraph);
    
    // Step 3: Execute graph using NUMA coordination + ggml building blocks
    enum ggml_status result = execute_numa_graph_plan(plan);
    
    // Step 4: Cleanup and return
    cleanup_numa_graph_plan(plan);
    return (result == GGML_STATUS_SUCCESS) ? 0 : -1;
}
```

## Implementation Strategy

### Phase 1: Graph Analysis
- Walk the computation graph and identify operations
- Determine which operations benefit from NUMA distribution
- Create execution plan with dependencies

### Phase 2: Direct Operation Execution
- Use ggml's type system and mathematical kernels directly
- No intermediate graph creation (avoiding recursion)
- NUMA coordinate the individual operations

### Phase 3: Building Blocks Usage
```c
// Instead of creating new graphs, use ggml building blocks directly:

// ✅ Use ggml's type system
const struct ggml_type_traits_cpu* traits = ggml_get_type_traits_cpu(src0->type);

// ✅ Use ggml's tensor views
struct ggml_tensor* numa_slice = ggml_view_nd(ctx, tensor, 4, ne, nb, offset);

// ✅ Use ggml's mathematical kernels directly
traits->vec_dot(ne00, dst_ptr, 0, src0_ptr, 0, src1_ptr, 0, 1);

// ✅ Use ggml's type conversion
traits->from_float(src_f32, dst_quantized, elements);
```

## Architecture Benefits

### ✅ No Recursion
- We become the complete graph executor
- No delegation to ggml_graph_compute()
- Direct control over all operations

### ✅ ggml Building Blocks
- Still use ggml's proven type system
- Still use ggml's mathematical kernels
- Still use ggml's memory management utilities

### ✅ NUMA Coordination
- Full control over work distribution
- Optimal memory placement
- Thread affinity management

### ✅ Fallback Safety
- If anything goes wrong, return -1
- llama-context automatically uses backend scheduler
- No system instability

## Key Files to Modify

### 1. Replace `ggml_numa_dispatch_compute_graph()`
- Current: Node-by-node dispatch with recursion risk
- New: Complete graph analysis and execution

### 2. Create Graph Analysis Module
- Analyze operation dependencies
- Identify NUMA opportunities
- Create optimal execution plan

### 3. Create Direct Execution Engine
- Execute operations using ggml building blocks
- No intermediate graph creation
- Full NUMA coordination

## Clean Integration Flow

```
Model Request
    ↓
llama-context.cpp::graph_compute()
    ↓
ggml_numa_dispatch_compute_graph() [OUR COMPLETE SYSTEM]
    ├── Graph Analysis
    ├── NUMA Planning  
    ├── Direct Execution (using ggml building blocks)
    └── Results (no recursion, no intermediate graphs)
    ↓
Success → return to llama-context
Failure → llama-context uses backend scheduler as fallback
```

This approach gives us:
- Complete control over graph execution
- All ggml type safety and mathematical correctness
- Full NUMA optimization capabilities  
- Clean fallback path
- No recursion traps
