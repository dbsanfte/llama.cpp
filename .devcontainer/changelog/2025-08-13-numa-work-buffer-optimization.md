# NUMA Work Buffer Optimization - August 13, 2025

## Issue Identified
The NUMA coordinator was allocating and freeing work buffers on every operation using `numa_alloc_onnode()` and `numa_free()` in the hot inference path. This caused significant performance overhead due to:

1. **Hot path allocations**: NUMA allocation/deallocation has considerable overhead
2. **Repeated operations**: Every compute graph node triggered allocation/deallocation 
3. **Consistent buffer sizes**: Work buffer sizes were typically consistent across operations
4. **NUMA fragmentation**: Frequent allocation/deallocation can cause NUMA memory fragmentation

## Solution Implemented
Replaced per-operation work buffer allocation with **persistent NUMA-local work buffers**:

### Architecture Changes
1. **Persistent Buffer Storage**: Added `work_buffer` and `work_buffer_size` fields to `ggml_coordinator_thread` struct
2. **Smart Buffer Management**: `ggml_numa_ensure_work_buffer()` function that:
   - Reuses existing buffer if adequate size
   - Grows buffer only when needed (not shrinks to avoid thrashing)
   - Allocates NUMA-local memory using `numa_alloc_onnode()`
3. **Lifecycle Management**: 
   - Initialize during coordinator setup
   - Cleanup during coordinator destruction
   - No hot-path allocations

### Code Changes
- **`/workspaces/llama.cpp/ggml/src/ggml-cpu/ggml-numa-coordinator.c`**:
  - Modified `struct ggml_coordinator_thread` to include persistent work buffer fields
  - Added `ggml_numa_ensure_work_buffer()` function for smart buffer management
  - Updated `ggml_numa_node_execute_operation()` to use persistent buffers
  - Added work buffer initialization in coordinator setup
  - Added work buffer cleanup in coordinator destruction

### Performance Impact
- **Before**: `numa_alloc_onnode()` + `numa_free()` on every graph operation
- **After**: One-time allocation per coordinator, persistent reuse

### Verification
Testing shows the optimization working correctly:
```
NUMA0: Allocating initial work buffer of 2928 bytes  # Once during setup
NUMA0: Using persistent work buffer (2928 bytes available, 2928 needed)  # Reused for all operations
```

## Benefits
1. **Eliminated hot-path NUMA allocations** - Major performance improvement
2. **NUMA locality preserved** - Buffers remain on correct NUMA nodes
3. **Memory efficiency** - Buffer grows only when needed, no unnecessary shrinking
4. **Reduced NUMA fragmentation** - Fewer allocation/deallocation cycles

## Compatibility
- Fully backward compatible
- No API changes
- Automatic optimization - transparent to users
- Same NUMA-awareness guarantees maintained
