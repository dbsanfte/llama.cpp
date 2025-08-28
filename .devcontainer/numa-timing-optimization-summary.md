# NUMA Simple Coordinator Timing Optimization

## Summary

Optimized the NUMA simple coordinator to conditionally execute `clock_gettime()` calls only when performance measurement is enabled, eliminating unnecessary timing overhead during normal operation.

## Changes Made

### 1. Conditional Timing Macros

Added new macros to `ggml-numa-simple-coordinator.c` that respect the `g_numa_perf_enabled` flag:

```c
// Conditional timing macros - only execute when performance measurement is enabled
#define NUMA_TIMING_START(var_name) \
    struct timespec var_name; \
    if (g_numa_perf_enabled) { \
        clock_gettime(CLOCK_MONOTONIC, &var_name); \
    }

#define NUMA_TIMING_END(start_var, end_var, result_ms_var) \
    struct timespec end_var; \
    double result_ms_var = 0.0; \
    if (g_numa_perf_enabled) { \
        clock_gettime(CLOCK_MONOTONIC, &end_var); \
        result_ms_var = (end_var.tv_sec - start_var.tv_sec) * 1000.0 + \
                        (end_var.tv_nsec - start_var.tv_nsec) / 1000000.0; \
    }
```

### 2. Replaced All Direct clock_gettime() Calls

Replaced 9 instances of direct `clock_gettime()` calls and manual timing calculation with conditional macros:

- **Dispatch worker timing**: Work execution timing in the numa_dispatch_worker function
- **Data-parallel coordination timing**: Overall coordination timing in execute_data_parallel
- **Barrier synchronization timing**: Barrier wait timing
- **Aggregation timing**: Custom aggregation function timing
- **Final coordination timing**: Total execution time calculation

### 3. Performance Impact

**Before optimization:**
- `clock_gettime()` called on every dispatch loop iteration
- Timing calculations performed even when not needed
- System call overhead in hot execution paths

**After optimization:**
- Zero timing overhead when performance measurement disabled
- Full timing functionality preserved when debug/performance measurement enabled
- Conditional execution respects existing NUMA performance framework

### 4. Behavior Verification

**Without GGML_NUMA_DEBUG (normal operation):**
- No timing debug messages appear
- No `clock_gettime()` system calls in dispatch loops
- Maximum performance with no measurement overhead

**With GGML_NUMA_DEBUG=1 (debug mode):**
- Full timing debug messages appear
- All timing calculations work correctly
- Example: "Dispatch thread 0 completed work with status 0 (3455.866ms)"
- Example: "Barrier wait completed in 3457.516ms"
- Example: "All dispatch NUMA work completed, final status: 0 (total coordination time: 3457.629ms)"

### 5. Integration with NUMA Performance Framework

The optimization leverages the existing `g_numa_perf_enabled` global variable from the NUMA performance system, ensuring consistency with other performance measurement controls throughout the codebase.

This approach follows the established pattern where performance measurement is controlled centrally and can be disabled completely for production workloads while still providing rich debugging information when needed.

## Validation

- ✅ All NUMA mathematical correctness tests pass
- ✅ Build system compiles successfully 
- ✅ Zero timing overhead in normal operation
- ✅ Full timing functionality in debug mode
- ✅ Integration with existing NUMA performance framework
- ✅ Backward compatibility maintained

## Performance Benefits

This optimization removes `clock_gettime()` system call overhead from every dispatch loop iteration, providing measurable performance improvement in high-frequency NUMA operations while preserving full debugging capabilities when needed.
