# 2025-01-14: Fixed Fallback Threadpool Thread Count Mismatch

## Issue Summary
Fixed critical thread count mismatch between backend requests (112 threads) and fallback threadpool capacity (56 threads), which caused "cplan requested more threads than available" errors during inference.

## Root Cause Analysis
1. **Backend Thread Setting**: The CPU backend was configured with `std::thread::hardware_concurrency()` = 112 threads via `ggml_backend_cpu_set_n_threads()`
2. **Fallback Threadpool Creation**: The NUMA fallback threadpool was created with only `g_simple_coordinator.threads_per_node[0]` = 56 threads (one NUMA node's worth)
3. **Mismatch**: When falling back to CPU execution, the system requested 112 threads but the threadpool only had 56 available

## Technical Solution

### Files Modified
- `ggml/src/ggml-cpu/ggml-numa-simple-coordinator.c`:
  - Fixed fallback threadpool creation to use full system thread count
  - Updated `ggml_numa_simple_coordinator_get_fallback_thread_count()` to return actual threadpool size
- `ggml/src/ggml-cpu/ggml-cpu.c`:
  - Implemented missing `ggml_threadpool_get_n_threads()` function

### Key Changes

#### 1. Fallback Threadpool Thread Count Fix
```c
// OLD: Used only one NUMA node's thread count
fallback_tpp.n_threads = g_simple_coordinator.threads_per_node[0]; // 56 threads

// NEW: Use full system thread count for backend compatibility  
fallback_tpp.n_threads = optimized_tpp.n_threads; // 112 threads
```

#### 2. Added Missing API Function
```c
int ggml_threadpool_get_n_threads(struct ggml_threadpool * threadpool) {
    if (!threadpool) {
        return 1;
    }
    return threadpool->n_threads_max;
}
```

#### 3. Updated Thread Count Query Function
```c
int ggml_numa_simple_coordinator_get_fallback_thread_count(void) {
    if (!g_simple_coordinator.initialized || !g_simple_coordinator.fallback_threadpool) {
        return 1;
    }
    return ggml_threadpool_get_n_threads(g_simple_coordinator.fallback_threadpool);
}
```

## Testing Results

### Before Fix
```
📊 Fallback Execution: threads=112, threadpool=0x... (disposable=false)
cplan requested more threads (112) than available (56)
[Repeated error messages during inference]
```

### After Fix
```
📊 Fallback Execution: threads=112, threadpool=0x... (disposable=false)
✅ NUMA Executor: All 966 operations completed successfully
srv init: initializing slots, n_slots = 1
main: server is listening on http://0.0.0.0:8080
```

## Impact
- ✅ **Eliminated thread count mismatch errors** during fallback execution
- ✅ **Restored inference functionality** with full hardware concurrency support
- ✅ **Maintained NUMA optimization** while ensuring backend compatibility
- ✅ **No performance regression** - fallback threadpool can now utilize full system capacity

## Validation
- **Build**: Successful compilation with warnings only
- **Runtime**: llama-server starts successfully without errors
- **Inference**: Model loading and operation completion verified
- **Thread Count**: Fallback threadpool correctly reports 112 threads

## Notes
This fix ensures that the NUMA fallback threadpool can handle full backend thread requests while maintaining the NUMA optimization benefits. The fallback path now seamlessly supports the same thread counts as the primary execution path.
