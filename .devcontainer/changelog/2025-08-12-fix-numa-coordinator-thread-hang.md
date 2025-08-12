# NUMA Coordinator Thread Hang Fix - August 12, 2025

## Problem Description

The NUMA coordinator was experiencing an infinite hang during initialization. When attempting to create and use the coordinator, threads would be reported as created but would never execute their work functions, causing the coordinator warmup process to wait indefinitely.

## Root Cause Analysis

Through progressive debugging with printf statements, GDB thread analysis, and git commit examination, the issue was identified as:

**Premature warmup signaling in coordinator manager creation**

The problematic code was in `ggml-numa-coordinator.c` in the `ggml_numa_coordinator_manager_new()` function:

```c
// This was causing the hang:
printf("DEBUG: Starting coordinator warmup...\n");
for (int node = 0; node < manager->numa_nodes; node++) {
    printf("DEBUG: Warming up node %d...\n", node);
    struct ggml_numa_coordinator *coord = &manager->coordinators[node];
    
    // Signal that work is available - THIS WAS THE PROBLEM
    pthread_cond_signal(&coord->work_available_cond);
}
printf("DEBUG: Coordinator warmup completed\n");
```

This warmup code was:
1. **Signaling condition variables before threads existed** - The warmup was happening in manager creation, but coordinator threads weren't created until later in `ggml_numa_coordinator_manager_start()`
2. **Creating a race condition** - Threads would be waiting on condition variables that had already been signaled before they started waiting
3. **Causing undefined behavior** - Signaling condition variables with no waiting threads leads to lost signals

## Solution

**Removed the problematic warmup section entirely** from the manager creation function. The coordinator threads now:

1. Are created properly in `ggml_numa_coordinator_manager_start()`
2. Execute their thread functions correctly 
3. Wait for actual work items rather than premature signals
4. Process work items and complete normally

## Verification

The fix was verified through multiple test runs:

- **test-numa-coordinator-functional**: All 12 test categories passed (115.5ms total)
- **Coordinator threads execute correctly**: Debug output shows threads processing work items
- **No infinite hangs**: Operations complete in reasonable time
- **Mathematical correctness maintained**: All operations produce correct results

## Technical Details

**Files Modified:**
- `ggml/src/ggml-cpu/ggml-numa-coordinator.c` - Removed warmup section from manager creation

**Threading Architecture:**
- Manager creates coordinators with proper initialization
- `ggml_numa_coordinator_manager_start()` creates actual pthread threads
- Threads wait for real work items via condition variables
- Proper producer-consumer synchronization maintained

**Git History:**
- Issue introduced in commit `fe662482` which added the premature warmup code
- Fixed by reverting to working thread lifecycle from commit `93594808`
- Clean solution removes unnecessary warmup entirely

## Lessons Learned

1. **Thread synchronization timing is critical** - Signaling before threads exist causes hangs
2. **Condition variable signals must have waiting threads** - Lost signals lead to indefinite waits  
3. **Warmup code should happen after thread creation** - Not during manager initialization
4. **Simple solutions are often best** - Removing unnecessary code resolved the issue

## Impact

This fix restores proper NUMA coordinator functionality:
- ✅ No more infinite hangs during coordinator startup
- ✅ Threads execute and process work items correctly
- ✅ All test suites pass with proper timing
- ✅ Ready for normal NUMA-aware workloads
