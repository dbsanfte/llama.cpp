# NUMA Persistent Work Buffer Optimization

**Date**: 2025-08-30  
**Author**: GitHub Copilot  
**Type**: Performance Optimization

## Summary
Optimized the NUMA executor fallback path to use persistent, auto-growing work buffers instead of allocating new buffers for every operation, eliminating allocation overhead and improving performance.

## Problem
The NUMA executor fallback path was allocating a new work buffer (using `malloc()` or `numa_alloc_onnode()`) for every single operation, causing significant allocation overhead. This was inefficient because:

1. **Per-operation allocation**: Each fallback operation created a fresh work buffer
2. **Memory management overhead**: Constant allocation/deallocation cycles
3. **Underutilized infrastructure**: The coordinator already had persistent work buffer capabilities that weren't being used by the executor

## Solution
Implemented persistent work buffer reuse in the NUMA executor fallback path:

### 1. Added Persistent Work Buffer Accessors
**File**: `ggml/src/ggml-cpu/ggml-numa-simple-coordinator.h`
```c
// Added accessor functions for persistent work buffer system
void * ggml_numa_simple_coordinator_get_fallback_work_buffer(size_t required_size);
size_t ggml_numa_simple_coordinator_get_fallback_work_buffer_size(void);
```

### 2. Implemented Auto-Growing Work Buffer System  
**File**: `ggml/src/ggml-cpu/ggml-numa-simple-coordinator.c`
```c
void * ggml_numa_simple_coordinator_get_fallback_work_buffer(size_t required_size) {
    if (g_numa_coordinator.fallback_work_buffer_size < required_size) {
        // Auto-grow the persistent buffer as needed
        if (g_numa_coordinator.fallback_work_buffer) {
            numa_free(g_numa_coordinator.fallback_work_buffer, g_numa_coordinator.fallback_work_buffer_size);
        }
        
        g_numa_coordinator.fallback_work_buffer = numa_alloc_onnode(required_size, 0);
        g_numa_coordinator.fallback_work_buffer_size = required_size;
        
        NUMA_LOG_DEBUG("Auto-grew fallback work buffer to %zu bytes on node 0", required_size);
    }
    
    return g_numa_coordinator.fallback_work_buffer;
}
```

### 3. Modified Executor Fallback Path
**File**: `ggml/src/ggml-cpu/ggml-numa-executor.c`

**Before** (per-operation allocation):
```c
// OLD: Allocate fresh buffer for every operation
work_buffer = malloc(buffer_size);
// ... use buffer
free(work_buffer);  // Cleanup every time
```

**After** (persistent buffer reuse):
```c
// NEW: Use persistent auto-growing buffer
work_buffer = ggml_numa_simple_coordinator_get_fallback_work_buffer(buffer_size);
// ... use buffer
// No cleanup needed - buffer persists for reuse
```

## Technical Benefits

### Performance Improvements
- **Eliminated allocation overhead**: No more per-operation `malloc()`/`numa_alloc_onnode()` calls
- **Reduced memory fragmentation**: Single persistent buffer vs. constant allocation/deallocation
- **Auto-growing efficiency**: Buffer grows to accommodate largest operation, then reuses that size

### Architecture Benefits  
- **Leveraged existing infrastructure**: Used coordinator's existing persistent buffer system
- **Clean separation of concerns**: Coordinator manages memory, executor uses it
- **Consistent with NUMA architecture**: Follows the executor → coordinator resource management pattern

### Memory Management
- **NUMA-aware allocation**: Persistent buffer allocated on node 0 for optimal access
- **Automatic cleanup**: Coordinator handles buffer lifecycle during shutdown
- **Size optimization**: Buffer only grows when needed, maintains optimal size

## Testing Results

### Mathematical Correctness
- ✅ All mathematical correctness tests pass
- ✅ No computational accuracy degradation
- ✅ Fallback path produces identical results

### Integration Testing
- ✅ Real-world inference with `llama-server` works correctly
- ✅ No memory leaks or corruption detected
- ✅ NUMA debug output shows proper buffer reuse

### Performance Characteristics
- ✅ No allocation overhead in fallback path
- ✅ `buffer=0 bytes/thread` output indicates efficient NUMA kernel usage
- ✅ Debug messages show persistent buffer allocation only when needed

## Code Changes

### Files Modified
- `ggml/src/ggml-cpu/ggml-numa-simple-coordinator.h` - Added accessor function declarations
- `ggml/src/ggml-cpu/ggml-numa-simple-coordinator.c` - Implemented persistent buffer accessor with auto-growing
- `ggml/src/ggml-cpu/ggml-numa-executor.c` - Modified fallback path to use persistent buffers

### Lines of Code
- **Added**: ~25 lines (accessor implementation + function declarations)
- **Modified**: ~15 lines (fallback path optimization)
- **Removed**: ~10 lines (per-operation allocation/cleanup code)

## Impact Assessment

### Performance Impact
- **Positive**: Eliminated per-operation allocation overhead in fallback path
- **Neutral**: No change to mathematical accuracy or NUMA kernel efficiency
- **Memory**: Slight increase in persistent memory usage, but eliminates allocation churn

### Compatibility
- **API compatibility**: No changes to public interfaces
- **Behavioral compatibility**: Identical computational results
- **Debug compatibility**: Enhanced debug messages show buffer management details

## Future Considerations

### Potential Extensions
1. **Memory usage monitoring**: Track buffer utilization patterns
2. **Size optimization**: Implement buffer shrinking for long-running processes
3. **Multi-node buffers**: Extend to per-node work buffers if needed

### Maintenance Notes
- Monitor buffer size growth patterns in production workloads
- Consider adding buffer size limits for memory-constrained environments
- Keep accessor functions synchronized with coordinator lifecycle

## Validation Commands

```bash
# Build and test the optimization
cmake --build build --parallel

# Verify mathematical correctness
./build/bin/test-numa-mathematical-correctness-add

# Test real-world integration  
./tests/run-numa-integration-test.sh --numa mirror

# Run comprehensive test suite
./tests/run-numa-tests.sh
```

## Debug Verification

```bash
# Check for persistent buffer usage
GGML_NUMA_DEBUG=2 ./build/bin/test-numa-mathematical-correctness-add 2>&1 | grep -i "buffer\|allocation"

# Verify no per-operation allocation overhead
GGML_NUMA_DEBUG=1 ./tests/run-numa-integration-test.sh --numa mirror
```

This optimization eliminates a significant performance bottleneck in the NUMA fallback path while maintaining full mathematical correctness and leveraging existing coordinator infrastructure.
