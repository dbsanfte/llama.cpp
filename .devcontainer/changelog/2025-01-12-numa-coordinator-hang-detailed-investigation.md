# NUMA Coordinator Hang Investigation - January 12, 2025

## Issue Summary
After commit `fe662482` ("iterate - remove hugepages, just use numa_alloc_on_node"), the NUMA coordinator hangs during warmup phase in performance tests.

## Investigation Process
Used systematic GDB debugging and comprehensive logging to isolate the exact hang location and identify root cause.

## Root Cause Analysis

### Hang Location Identified
- **Exact Location**: `ggml_numa_coordinator_manager_wait_for_completion()` line 1710
- **Symptom**: Main thread hangs in `ggml_cond_wait(&mgr->main_sync_cond, &mgr->main_sync_mutex)`
- **Context**: Waiting for coordinator threads to signal work completion

### Debug Evidence
```
=== DEBUG: Entering wait_for_completion ===
=== DEBUG: Acquired main_sync_mutex ===
=== DEBUG: Checking 2 coordinators for pending work ===
=== DEBUG: Coordinator 0 has 1 pending items ===
=== DEBUG: Not all complete, waiting on condition ===
[HANGS HERE]
```

### Work Queue Analysis
- ✅ **Work Enqueue Working**: `ggml_work_queue_enqueue()` successfully adds items
- ✅ **Pending Counter Correct**: Shows 1 pending item as expected
- ✅ **Thread Creation Succeeds**: `ggml_thread_create()` returns 0 (success)
- ❌ **Coordinator Threads Not Running**: No debug output from coordinator thread functions

### Producer-Consumer Bug Confirmed
1. **Producer (Main Thread)**: Successfully creates work items and increments pending counter
2. **Consumer (Coordinator Threads)**: Created successfully but never execute their main loop
3. **Synchronization**: Main thread waits for condition variable that is never signaled

## Technical Root Cause
**Coordinator threads crash or exit immediately after creation**, before reaching their work processing loop. This causes:
- Work items remain in `pending_items` counter but are never processed
- `main_sync_cond` condition variable is never signaled
- Main thread hangs indefinitely in `ggml_cond_wait()`

## Evidence Supporting Thread Crash Theory
- Thread creation succeeds (no errors from `ggml_thread_create`)
- Manager reports "coordinator threads started successfully"
- Zero debug output from coordinator thread functions (should print immediately upon entry)
- GDB analysis shows only regular GGML threadpool threads, no NUMA coordinator threads visible

## Initial Debugging Fix Attempt
Fixed obvious segfault in debug logging (accessing `coordinator->numa_node` before `coordinator` variable was assigned), but coordinator threads still not executing.

## Status
**CRITICAL REGRESSION** - Hang blocks all NUMA coordinator development. 

## Next Steps Required
1. **Deep Thread Analysis**: Investigate why coordinator threads crash/exit immediately
2. **Memory Validation**: Check coordinator data structure integrity 
3. **Thread Function Audit**: Review coordinator thread function for initialization issues
4. **Synchronization Review**: Verify thread creation timing and startup coordination

## Impact
- All NUMA coordinator performance tests hang during warmup
- Development workflow completely blocked on NUMA features
- Critical regression from recent memory allocation changes

## Debug Tools Used
- GDB with breakpoints and backtraces
- Thread state analysis (`info threads`, `bt`)
- Comprehensive printf debugging of execution flow
- Work queue state inspection
- Producer-consumer synchronization tracing

## Files Modified During Investigation
- `ggml/src/ggml-cpu/ggml-numa-coordinator.c` - Added extensive debug logging
- `tests/test-comprehensive-numa-performance.cpp` - Debug output for hang isolation
