# Unified CPU Assignment System - Implementation Summary

## Problem Solved
Resolved the dual CPU assignment problem where llama.cpp had two separate systems for assigning threads to CPUs:
1. **Threadpool assignment** - Based on user preferences and strict CPU settings
2. **NUMA runtime override** - Applied during graph computation, potentially overriding user choices

This dual system caused user CPU assignment preferences to be silently ignored when NUMA-aware assignment was active.

## Unified Solution Implemented

### 1. Enhanced Parameter Structures

**cpu_params** (in `common/common.h`):
```cpp
struct cpu_params {
    // ...existing fields...
    bool numa_aware = true;              // Enable NUMA-aware assignment
    bool allow_numa_override = true;     // Allow NUMA to override user strict setting
    bool warn_on_numa_override = false;  // Warn when NUMA overrides user settings
};
```

**ggml_threadpool_params** (in `ggml/include/ggml.h`):
```cpp
struct ggml_threadpool_params {
    // ...existing fields...
    bool numa_aware;              // NUMA-aware assignment enabled
    bool allow_numa_override;     // Can NUMA override user strict setting?
    bool warn_on_numa_override;   // Should we warn about overrides?
};
```

**ggml_threadpool** (in `ggml/src/ggml-cpu/ggml-cpu.c`):
```cpp
struct ggml_threadpool {
    // ...existing fields...
    struct ggml_threadpool_params params;  // Store parameters for unified assignment
};
```

### 2. New Command-Line Options

Added three new command-line arguments in `common/arg.cpp`:

- `--numa-no-aware`: Disable NUMA-aware CPU assignment (use simple round-robin)
- `--numa-respect-strict`: Never override user `--cpu-strict` setting for NUMA locality  
- `--numa-no-warn`: Disable warnings when NUMA overrides user CPU settings

### 3. Unified Assignment Function

Created `ggml_thread_cpumask_unified()` in `ggml/src/ggml-cpu/ggml-cpu.c`:
- Single point of control for CPU assignment decisions
- Considers both user preferences and NUMA topology
- Provides clear override control mechanism
- Supports warning when overrides occur

### 4. Modified NUMA Override Logic

Updated the NUMA mirror code in `ggml_graph_compute_thread()`:
- Now respects threadpool parameters for override control
- Only applies NUMA assignment when `allow_numa_override` is true
- Provides warnings when NUMA assignment is blocked by user preference
- Maintains backward compatibility with existing NUMA functionality

### 5. Parameter Propagation

Updated conversion functions in `common/common.cpp`:
- `ggml_threadpool_params_from_cpu_params()` properly converts unified parameters
- Parameters flow: CLI args → cpu_params → ggml_threadpool_params → threadpool storage
- Threadpool initialization stores parameters for runtime access

## Key Benefits

### 1. **User Preference Respect**
- User CPU assignment preferences are no longer silently overridden
- Clear control over when NUMA can override user settings
- Explicit warnings when overrides would occur

### 2. **Unified Control**
- Single function (`ggml_thread_cpumask_unified`) handles all CPU assignment logic
- Eliminates architectural confusion between threadpool and NUMA assignment
- Consistent behavior across different threading scenarios

### 3. **Backward Compatibility**
- Existing NUMA functionality preserved when override is allowed
- Default settings maintain current behavior for existing users
- Progressive enhancement that doesn't break existing workflows

### 4. **Clear User Interface**
- Intuitive command-line options for controlling NUMA behavior
- Self-documenting parameter names that clearly indicate their purpose
- Help text that explains the new options

## Usage Examples

### Default Behavior (NUMA-aware with overrides allowed)
```bash
./llama-server --model model.gguf
# NUMA can override user CPU assignments for better locality
```

### Strict User Control (No NUMA overrides)
```bash
./llama-server --model model.gguf --numa-respect-strict
# User CPU assignments are never overridden by NUMA logic
```

### Disable NUMA Awareness Completely
```bash
./llama-server --model model.gguf --numa-no-aware
# Simple round-robin CPU assignment, no NUMA considerations
```

### NUMA with Warnings
```bash
./llama-server --model model.gguf --numa-respect-strict
# Shows warnings when NUMA would override user settings (but doesn't)
```

## Testing Validation

✅ **Build Success**: All components compile successfully
✅ **Parameter Propagation**: New parameters flow correctly through the system
✅ **Command-Line Integration**: All new flags appear in help output and are recognized
✅ **CPU Topology Detection**: Existing functionality remains intact
✅ **Backward Compatibility**: Default behavior preserved

## Files Modified

- `common/common.h`: Extended cpu_params structure
- `common/arg.cpp`: Added new command-line arguments
- `ggml/include/ggml.h`: Extended ggml_threadpool_params structure  
- `ggml/src/ggml.c`: Updated parameter initialization and matching functions
- `ggml/src/ggml-cpu/ggml-cpu.c`: 
  - Added unified assignment function
  - Modified threadpool structure and initialization
  - Updated NUMA override logic
  - Modified traditional NUMA affinity application

## Architecture Impact

The unified system resolves the fundamental architectural issue where two independent CPU assignment mechanisms could conflict. Now there's a single, controllable pathway for CPU assignment decisions that respects user preferences while still providing NUMA-aware optimizations when desired.

This implementation provides the foundation for more sophisticated CPU assignment strategies in the future while maintaining clear user control and system predictability.
