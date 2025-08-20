# Work Buffer Race Condition Fix - NUMA Coordinator

**Date**: 2024-12-19  
**Status**: ✅ COMPLETED  
**Test Result**: `test-numa-coordinator` now passes (was failing with exit code 134)

## Problem Summary

The NUMA coordinator had a critical race condition in work buffer management causing heap corruption and test failures with SIGABRT (exit code 134).

## Root Cause Analysis

**Race Condition Pattern:**
```
Thread A (coordinator): ggml_numa_ensure_work_buffer() calls numa_free(work_buffer)
Thread B (threadpool):  Concurrently accessing the same work_buffer for computation
Thread A (coordinator): numa_alloc_onnode() allocates new buffer at same/different address  
Thread B (threadpool):  Heap corruption due to accessing freed memory
```

**Heap Error Message:**
```
malloc(): unaligned tcache chunk detected
```

## Solution Implemented

**Per-Thread Work Buffer Architecture:**

1. **Removed shared work buffers** from coordinator struct
2. **Added thread-local storage** using `__thread` variables:
   ```c
   __thread void * ggml_thread_work_buffer = NULL;
   __thread size_t ggml_thread_work_buffer_size = 0;
   __thread int ggml_thread_work_buffer_numa_node = -1;
   ```

3. **Implemented partitioned allocation** with `ggml_numa_get_partitioned_work_buffer()`
4. **NUMA-local allocation** per thread using `numa_alloc_onnode()`
5. **Automatic cleanup** when threads exit

## Technical Implementation

**Key Changes:**
- `ggml/src/ggml-cpu/ggml-numa-coordinator.c`: Core per-thread work buffer system
- Replaced `ggml_numa_ensure_work_buffer()` calls with `ggml_numa_get_partitioned_work_buffer()`
- Updated coordinator initialization and cleanup to remove shared buffer management
- Modified public API functions for backward compatibility

**Memory Management:**
- Each thread allocates its own NUMA-local work buffer
- Buffers grow as needed and are reused for efficiency  
- No synchronization needed between threads
- Memory overhead acceptable per user confirmation

## Validation Results

**Before Fix:**
```
❌ test-numa-coordinator: FAILED (exit_code=134, 0.19s)
malloc(): unaligned tcache chunk detected
```

**After Fix:**
```
✅ test-numa-coordinator: PASSED (0.68s)
Total: 9/9 tests passed 🎉 ALL TESTS PASSED!
```

## Impact

- ✅ **Eliminated race condition**: No more heap corruption from concurrent buffer access
- ✅ **Improved thread safety**: Per-thread allocation eliminates cross-thread dependencies  
- ✅ **Maintained performance**: NUMA-local allocation preserves memory locality
- ✅ **Backward compatibility**: Public API functions still work with per-thread semantics

## Architecture Benefits

1. **Thread Safety**: Each thread owns its buffer, no synchronization needed
2. **NUMA Optimization**: Buffers allocated on local NUMA nodes
3. **Scalability**: Linear scaling with thread count, no shared resource bottlenecks
4. **Simplicity**: Eliminates complex buffer coordination logic

This fix resolves the critical stability issue and enables reliable NUMA coordinator operation in multi-threaded environments.
