# NUMA Coordinator Per-Thread Work Buffer Implementation

## Problem Analysis

**Root Cause**: Critical race condition in NUMA coordinator work buffer management causing heap corruption with "malloc(): unaligned tcache chunk detected" error.

### Issue Details

1. **Single shared work buffer**: Each coordinator thread has one work buffer shared by all threadpool threads
2. **Multi-threaded access**: Work functions execute using threadpools with multiple threads accessing same buffer
3. **Unsynchronized reallocation**: `ggml_numa_ensure_work_buffer()` performs `numa_free()` + `numa_alloc_onnode()` while other threads may be using the buffer
4. **Simultaneous access**: Multiple threadpool threads access the same work buffer concurrently during operations

### Failure Sequence

```
Thread A (coordinator): ggml_numa_ensure_work_buffer() calls numa_free(work_buffer)
Thread B (threadpool):   Still has pointer to work_buffer, continues using freed memory  
Thread A (coordinator): numa_alloc_onnode() returns new buffer
Thread B (threadpool):   Accesses freed memory → heap corruption → SIGABRT
```

## Solution

**Replace shared work buffer with per-thread work buffers.** Each threadpool thread gets its own NUMA-local work buffer.

### Implementation Strategy

1. **Remove shared work buffer** from `struct ggml_coordinator_thread`
2. **Add per-thread buffer management** in threadpool execution
3. **Use thread-local storage** or threadpool context for buffer access
4. **NUMA-local allocation** per thread using `numa_alloc_onnode()`

### Advantages

- **Zero race conditions** - each thread owns its buffer
- **Better cache locality** - thread's buffer stays on its core
- **No synchronization overhead** - no mutexes needed
- **Cleaner architecture** - no shared mutable state

### Code Changes

```c
// Remove from struct ggml_coordinator_thread:
// void * work_buffer;
// size_t work_buffer_size;

// Add per-thread buffer management in compute params:
// Each thread gets its own buffer allocated on first use
```

## Testing

This fix resolves the SIGABRT in `test-numa-coordinator` that was failing with exit code 134.

## Files Modified

- `ggml/src/ggml-cpu/ggml-numa-coordinator.c`: Per-thread work buffer implementation
