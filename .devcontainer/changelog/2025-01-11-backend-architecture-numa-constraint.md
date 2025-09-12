# Backend Architecture NUMA Integration Constraint

**Date**: January 11, 2025  
**Issue**: Regular CPU backend cannot use NUMA coordinator due to circular dependencies  
**Resolution**: Implement simplified NUMA allocation strategy for base backend  

## Problem Discovery

While implementing coordinator integration for the regular CPU backend (`ggml-backend.cpp`), discovered a fundamental architectural constraint:

### Circular Dependency Issue
- **ggml-backend.cpp** is compiled into **ggml-base** target
- **NUMA coordinator functions** are compiled into **ggml-cpu** target  
- **ggml-cpu** depends on **ggml-base**
- This creates circular dependency: `ggml-base` → `ggml-cpu` → `ggml-base`

### Root Cause Analysis
The issue is triggered by inline functions in `ggml.h`:
- `tensor_data()` and `tensor_set_data()` are inline functions
- These functions call `ggml_is_numa()`, `ggml_numa_node_count()`, `ggml_numa_should_mirror()`
- When `ggml-backend.cpp` uses these inline functions, it generates references to NUMA functions
- These NUMA functions are only defined in `ggml-cpu.c` (part of ggml-cpu target)
- Results in undefined reference errors during linking

## Architecture-Aware Solution

### Two-Tier Buffer Strategy
Instead of trying to force coordinator integration everywhere, implement appropriate strategy per backend:

#### 1. Repack Buffer (in ggml-cpu target)
- ✅ **Full coordinator integration** - 3-tier strategy with coordinator awareness
- ✅ **Timing robustness** - handles coordinator startup gracefully  
- ✅ **Optimal node selection** - uses coordinator's active node information

#### 2. Regular CPU Backend (in ggml-base target)
- ✅ **Simple round-robin allocation** - distributes across all NUMA nodes
- ✅ **No coordinator dependency** - avoids circular dependency issues
- ✅ **Load balancing** - ensures even distribution across available nodes

## Implementation Details

### Regular CPU Backend Approach (`ggml-backend.cpp`)
```cpp
// Round-robin allocation across all available NUMA nodes
if (numa_available() != -1) {
    int max_node = numa_max_node();
    if (max_node > 0) {
        static int allocation_counter = 0;
        int numa_node = allocation_counter % (max_node + 1);
        allocation_counter++;
        data = numa_alloc_onnode(size, numa_node);
    } else {
        data = numa_alloc_local(size);  // Single node system
    }
}
```

### Benefits of Two-Tier Approach
1. **Architectural Integrity** - Respects existing build target dependencies
2. **Consistent Load Balancing** - Both backends distribute load across NUMA nodes
3. **Performance Preservation** - No performance penalty from architectural constraints
4. **Maintenance Simplicity** - Avoids complex dependency refactoring

## Key Insights

### What Doesn't Work
- ❌ Moving coordinator functions to ggml-base (breaks coordinator encapsulation)
- ❌ Using weak symbols (complexity, fragility)
- ❌ Refactoring inline functions (breaks existing NUMA mirroring)

### What Works Well  
- ✅ Architecture-aware strategy selection per backend
- ✅ Round-robin allocation for base backends (simple, effective)
- ✅ Full coordinator integration for CPU backends (optimal)

## Performance Characteristics

### Repack Buffer (Coordinator-Integrated)
- **Best**: Uses coordinator's exact thread-to-node mapping
- **Good**: Adapts to dynamic thread assignments
- **Robust**: Graceful fallback during startup timing

### Regular CPU Backend (Round-Robin)
- **Good**: Even distribution across all NUMA nodes
- **Simple**: No coordinator timing dependencies  
- **Reliable**: Consistent behavior across all scenarios

Both approaches ensure good NUMA bandwidth utilization without architectural compromises.

## Lessons Learned

1. **Architecture Constraints Matter** - Build target dependencies create real limitations
2. **Simple Can Be Better** - Round-robin allocation often sufficient for good performance
3. **Inline Function Dependencies** - Header-defined inline functions create unexpected linking requirements
4. **Two-Tier Strategy Works** - Different backends can use different strategies appropriately

This resolution maintains performance goals while respecting architectural boundaries.
