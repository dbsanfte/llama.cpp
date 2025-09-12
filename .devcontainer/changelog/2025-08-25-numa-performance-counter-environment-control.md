# 2025-08-25: NUMA Performance Counter Environment Variable Control

## 🎯 Summary
Implemented environment variable control for NUMA performance counters and logging to eliminate performance overhead and output clutter in production while preserving diagnostic capabilities for development.

## ✅ Implementation Details

### Environment Variable Control System
- **`GGML_NUMA_DEBUG=0` or unset**: No performance counters, no performance summary (default)
- **`GGML_NUMA_DEBUG=1`**: Performance counters enabled, summary shows "Detailed Logging: Disabled"
- **`GGML_NUMA_DEBUG=2`**: Performance counters enabled, summary shows "Detailed Logging: Enabled"

### Files Modified
- **`ggml/src/ggml-cpu/ggml-numa-perf.c`**:
  - Added `#include "ggml-numa-shared.h"` for debug environment variable access
  - Modified initialization to check `ggml_numa_debug_enabled() >= 1` for performance tracking
  - Modified detailed logging to use `ggml_numa_debug_enabled() >= 2` for verbose mode
  - Converted initialization and shutdown messages to use `NUMA_LOG_DEBUG` macros
- **`ggml/src/ggml-cpu/ggml-numa-simple-coordinator.c`**:
  - Removed explicit `ggml_numa_perf_set_enabled(true)` call
  - Performance instrumentation now respects environment variable settings from perf_init()

### Performance Summary Output Control
The complete NUMA performance summary table:
```
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                   NUMA PERFORMANCE SUMMARY                                          ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣
║ Total Events: 10000       |  Detailed Logging: Disabled  |  Aggregate Stats: Enabled            ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣
║ Category                 │ Events   │ Avg (ms)     │ Min (ms)     │ Max (ms)     │ Efficiency ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣
║ ExecutorQuery            │ 5000     │        0.000 │        0.000 │        0.040 │    3750.52 ║
║ ExecutorKernelExec       │ 236      │        0.668 │        0.252 │        4.054 │       1.50 ║
║ ExecutorFallback         │ 4764     │        1.378 │        0.007 │       33.526 │       0.73 ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

Performance Breakdown:
  ExecutorQuery: 0.0% (1.333ms total)
  ExecutorKernelExec: 2.3% (157.588ms total)
  ExecutorFallback: 97.6% (6565.326ms total)
```

Is now completely disabled by default, eliminating visual clutter and performance measurement overhead.

## 🚀 Performance Impact
- **Zero overhead in production**: No performance counters collected when `GGML_NUMA_DEBUG` is unset
- **Clean output**: No performance summary tables cluttering inference results
- **Eliminates timing overhead**: No high-precision timing calls during normal operation
- **Preserves diagnostics**: Full performance analysis available when debugging is enabled

## 🧪 Testing Verification
- ✅ **Clean mode** (`GGML_NUMA_DEBUG` unset): No performance summary, no debug output
- ✅ **Debug mode** (`GGML_NUMA_DEBUG=1`): Performance summary shows with "Detailed Logging: Disabled"
- ✅ **Verbose mode** (`GGML_NUMA_DEBUG=2`): Performance summary shows with "Detailed Logging: Enabled"
- ✅ **llama-bench**: Completes with clean output in production mode
- ✅ **Environment switching**: Performance control responds immediately to environment variable changes

## 📋 Usage Examples

### Production Inference (Silent, Maximum Performance)
```bash
./build/bin/llama-bench -m model.gguf --numa mirror
# No performance summary, clean output
```

### Development Performance Analysis
```bash
GGML_NUMA_DEBUG=1 ./build/bin/llama-bench -m model.gguf --numa mirror
# Shows performance summary with standard logging
```

### Detailed Performance Troubleshooting
```bash
GGML_NUMA_DEBUG=2 ./build/bin/llama-bench -m model.gguf --numa mirror
# Shows performance summary with detailed logging enabled
```

## 🎉 Result
User requested feature: "disable these performance counters and this numa performance logging unless the debug environment variable above is set" has been **completely implemented**. NUMA performance instrumentation is now fully controlled by the `GGML_NUMA_DEBUG` environment variable, providing clean production output while preserving full diagnostic capabilities for development and troubleshooting.
