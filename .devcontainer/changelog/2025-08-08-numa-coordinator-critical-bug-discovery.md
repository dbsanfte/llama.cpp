# Critical NUMA Coordinator Bug Discovery - August 8, 2025

## Issue Summary

**CRITICAL BUG IDENTIFIED**: The NUMA coordinator is being automatically triggered for all GGML graph computations (including basic `ggml_graph_compute` calls) and is silently failing to compute results while reporting success.

## Evidence

### Test Results
- **Simple ADD operations**: Work correctly through coordinator ✅
- **MUL_MAT operations**: Hang in coordinator execution ❌
- **Basic ggml_graph_compute**: Routes through coordinator but produces zero results ❌

### Key Error Messages
```
FATAL: Thread received NULL state pointer
```

### Behavior Pattern
1. Coordinator reports successful graph computation
2. All result values are zero (incorrect)
3. No errors or failures reported
4. MUL_MAT specifically hangs during execution

## Root Cause Analysis

The coordinator has TWO distinct issues:
1. **Silent computation failure**: Operations complete but produce zero results
2. **MUL_MAT hanging**: Matrix multiplication operations deadlock during execution

## Critical Finding

The thread count mismatch we identified earlier (`nth = 1` vs multi-threaded threadpool) is likely part of the problem but not the complete solution. The coordinator has deeper state management issues evidenced by the "NULL state pointer" error.

## Technical Details

### Fixed So Far
- ✅ Added `n_threads` field to coordinator structure
- ✅ Set `nth = coordinator->n_threads` in compute params
- ✅ Fixed CMakeLists.txt include directories

### Still Broken
- ❌ Coordinator produces zero results for all operations
- ❌ MUL_MAT operations hang completely
- ❌ Thread state management ("NULL state pointer")

## Next Steps

1. **Immediate**: Investigate thread state initialization in coordinator
2. **Priority**: Fix the NULL state pointer issue
3. **Validation**: Verify MUL_MAT works after state management fix

## Test Cases Created

- `test-numa-debug-simple.cpp`: Proves ADD works through coordinator
- `test-numa-mulmat-debug.cpp`: Reproduces MUL_MAT hanging
- `test-basic-mulmat.cpp`: Shows coordinator interference with basic operations

This discovery shifts focus from "why does MUL_MAT hang" to "why does the coordinator fail to compute anything correctly while claiming success".
