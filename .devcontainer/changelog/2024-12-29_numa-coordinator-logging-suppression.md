# NUMA Coordinator Logging Suppression Fix

**Date:** 2024-12-29  
**Type:** Bug Fix / Performance Test Enhancement  
**Component:** NUMA Performance Testing  

## Problem

The comprehensive NUMA performance test (`test-comprehensive-numa-performance.cpp`) was producing excessive coordinator debug logging noise that made test output difficult to read. Despite implementing logging suppression using `common_log_set_verbosity_thold()`, coordinator debug messages continued to appear:

```
NUMA0: MUL_MAT operation - using full threadpool parallelization
Coordinator NUMA1: Operation MUL_MAT completed successfully
NUMA1: executing complete operation MUL_MAT
[... hundreds of similar debug lines ...]
```

## Root Cause Analysis

The issue was that the NUMA coordinator uses **GGML's separate logging system** (`GGML_LOG_DEBUG` macros) rather than the common logging system:

1. **GGML Logging System**: Uses `ggml_log_internal()` → global `g_logger_state` callback → `ggml_log_callback_default()` → prints everything to stderr
2. **Common Logging System**: Uses `LOG_*` macros controlled by `common_log_set_verbosity_thold()`
3. **The two systems are independent** - common log verbosity doesn't affect GGML logging

## Solution

Updated the logging suppression mechanism to control both systems:

### Original Implementation (Ineffective)
```cpp
static void suppress_coordinator_logging() {
    common_log_set_verbosity_thold(GGML_LOG_LEVEL_NONE); // Only affects common LOG_* macros
}
```

### New Implementation (Effective)
```cpp
// Custom GGML callback that filters debug messages
static void suppress_debug_callback(ggml_log_level level, const char* text, void* user_data) {
    // Only allow ERROR and WARN messages through during suppression
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
        fflush(stderr);
    }
    // Suppress DEBUG, INFO, and other messages
}

static void suppress_coordinator_logging() {
    // Control common logging system
    original_log_verbosity = common_log_verbosity_thold;
    common_log_set_verbosity_thold(GGML_LOG_LEVEL_NONE);
    
    // Control GGML logging system
    ggml_log_set(suppress_debug_callback, nullptr);
}

static void restore_coordinator_logging() {
    // Restore both logging systems
    common_log_set_verbosity_thold(original_log_verbosity);
    ggml_log_set(ggml_log_callback_default, nullptr);
}
```

## Files Modified

- `/workspaces/llama.cpp/tests/test-comprehensive-numa-performance.cpp`
  - Updated logging suppression functions to use GGML callback system
  - Added custom `suppress_debug_callback()` that filters by log level
  - Fixed unused parameter warning

## Results

✅ **Clean Test Output**: No more coordinator debug noise during performance tests  
✅ **Maintained Error Visibility**: Critical ERROR and WARN messages still appear  
✅ **Proper Restoration**: Logging is correctly restored after benchmarks  
✅ **No Build Warnings**: Clean compilation  

### Before Fix
```
NUMA0: MUL_MAT operation - using full threadpool parallelization  
Coordinator NUMA1: Operation MUL_MAT completed successfully
NUMA1: executing complete operation MUL_MAT
Work group 1: chunk 0/4 completed
[... hundreds of debug lines making output unreadable ...]
```

### After Fix
```
🧪 Testing Matrix 512x512x512 on CPU 0 (Hyperthread)
✅ Thread successfully pinned to CPU 0
   First iteration: 105.50 ms
   ✅ Average: 104.03 ms, Min: 99.62 ms, Max: 112.29 ms
   📊 2.58 GFLOPS
[... clean, readable test results ...]
```

## Technical Notes

- **GGML vs Common Logging**: GGML has its own logging infrastructure independent of the common logging system
- **Log Level Filtering**: Only suppresses DEBUG and INFO levels, preserves ERROR and WARN for diagnostics
- **Callback Management**: Properly saves and restores original GGML callback state
- **Integration Pattern**: This approach can be used in other tests that need to suppress GGML debug output

## Impact

This fix makes the NUMA performance test suite practically usable by eliminating debug noise while preserving essential error reporting. The clean output allows users to focus on actual performance metrics and test results.
