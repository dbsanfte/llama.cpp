# NUMA Coordinator Hang Issue - January 12, 2025

## Problem Identified 🚨

The NUMA performance test (`test-comprehensive-numa-performance`) is hanging during the coordinator warmup phase after recent changes.

## Hang Location 📍

The test hangs at:
```
🔥 Warmup coordinator...
DEBUG: Creating warmup graph...
DEBUG: Filling tensors...
DEBUG: Calling compute_graph...
[HANGS HERE - never returns from ggml_numa_coordinator_manager_compute_graph]
```

## Root Cause Analysis 🔍

### What's Happening
1. **Coordinator threads are created** and immediately wait for work items via `ggml_work_queue_dequeue()`
2. **Main thread calls** `ggml_numa_coordinator_manager_compute_graph()` 
3. **Function hangs** inside compute_graph and never returns a result code
4. **Coordinator threads wait indefinitely** in `ggml_cond_wait(&queue->work_available, &queue->queue_mutex)`

### Suspected Issue
The recent commit "iterate - remove hugepages, just use numa_alloc_on_node" (fe662482) introduced changes to:
- Coordinator initialization with problematic warmup signaling
- Work queue synchronization mechanisms
- Threading startup sequences

### Evidence
- **Fixed warmup signaling bug**: Removed spurious `ggml_cond_signal(&coord->work_queue.work_available)` calls that signaled work availability without actual work items
- **Still hanging**: Issue persists, suggesting deeper synchronization problem in work submission or completion tracking

## Attempted Fixes ❌

### 1. Removed Spurious Warmup Signaling
**Problem**: Coordinator initialization was signaling work availability without queuing actual work
**Fix**: Removed the problematic warmup loop that called `ggml_cond_signal()` without work items
**Result**: ❌ Still hanging

### 2. Added Debug Logging  
**Problem**: Hard to identify exact hang location
**Fix**: Added debug prints to pinpoint hang location in `ggml_numa_coordinator_manager_compute_graph`
**Result**: ✅ Confirmed hang location but issue persists

## Current State 📊

- ✅ **Core GGML functionality**: Basic tensor operations work
- ✅ **Build system**: Compiles successfully  
- ❌ **NUMA coordinator**: Hangs during initialization warmup
- ❌ **Performance tests**: Cannot complete due to coordinator hang

## Impact Assessment 💥

### Critical Issues
- **NUMA performance testing blocked**: Cannot validate NUMA coordinator improvements
- **Potential runtime regression**: Applications using NUMA coordination may hang
- **Development workflow disrupted**: Core testing infrastructure non-functional

### Working Components  
- Single-threaded operations work normally
- Non-NUMA code paths unaffected
- Basic tensor operations and math functions operational

## Recommended Actions 🎯

### Immediate (High Priority)
1. **Consider reverting fe662482** to last known working state
2. **Alternative**: Temporarily disable NUMA coordination in tests to unblock development
3. **Investigate synchronization primitives** in coordinator work queue implementation

### Medium Term
1. **Systematic debugging**: Use GDB to trace the exact deadlock conditions
2. **Unit test isolation**: Create minimal NUMA coordinator test to isolate the issue
3. **Review recent changes**: Carefully audit all threading and synchronization changes

### Long Term
1. **Robust testing**: Add coordinator startup/shutdown tests to prevent regressions
2. **Documentation**: Better document the complex coordinator initialization sequence
3. **Simplification**: Consider simplifying the coordinator state machine

## Technical Details 🔧

### Hang Pattern
```c
// Coordinator threads wait here:
while (queue->head == NULL && !atomic_load(&queue->shutdown_requested)) {
    ggml_cond_wait(&queue->work_available, &queue->queue_mutex);  // ← INFINITE WAIT
}

// Main thread waits here:  
while (true) {
    // ... check if work complete ...
    ggml_cond_wait(&mgr->main_sync_cond, &mgr->main_sync_mutex);  // ← INFINITE WAIT
}
```

### Key Files Affected
- `ggml/src/ggml-cpu/ggml-numa-coordinator.c` - Core coordinator logic
- `tests/test-comprehensive-numa-performance.cpp` - Failing test
- Recent commit fe662482 - Introduced the regression

This represents a critical regression that blocks NUMA development and testing workflows.
