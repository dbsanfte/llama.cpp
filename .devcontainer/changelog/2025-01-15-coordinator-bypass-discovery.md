# NUMA Coordinator Bypass Discovery

**Date**: January 15, 2025  
**Author**: GitHub Copilot  
**Context**: User reported fallback operations still being triggered despite implementing dispatcher intercept system

## Problem Discovery

While implementing the dispatcher-first architecture, discovered that there are **two parallel execution paths** for NUMA operations:

### Path 1: Operation-Level (Dispatcher) - ✅ Working  
`ggml-cpu.c` → `ggml_numa_intercept_operation()` → **dispatcher** → coordinator  
- Individual operations are intercepted in `ggml_compute_forward()`
- Routed through sophisticated dispatcher handlers
- MUL_MAT operations use `ggml_numa_handler_mul_mat_enhanced`
- Proper NUMA chunking and coordination

### Path 2: Graph-Level (Coordinator Direct) - ❌ Bypassing Dispatcher
`llama-context.cpp` → `ggml_numa_coordinator_manager_compute_graph()` → **coordinator only**  
- Entire graph processed by coordinator's loop
- Operations fall back to `ggml_numa_fallback_execute()`
- No sophisticated dispatcher handlers involved
- This is where the fallback calls are coming from!

## Root Cause Analysis

The issue is in `/workspaces/llama.cpp/src/llama-context.cpp`:

```cpp
// Line 372: Direct coordinator access
numa_coordinator = ggml_numa_coordinator_manager_get_global_with_params(&tpp);

// Line 1449: Direct graph processing bypass
int result = ggml_numa_coordinator_manager_compute_graph(numa_coordinator, gf);
```

This completely bypasses our dispatcher system! Operations are processed by the coordinator's simple graph loop, which then uses basic fallback operations instead of our sophisticated dispatcher handlers.

## Architecture Comparison

**Current Broken State**:
```
llama-context.cpp → coordinator.compute_graph() → fallback operations
                                                   
ggml-cpu.c → dispatcher → coordinator (unused path)
```

**Intended Architecture**:
```
llama-context.cpp → dispatcher.process_graph() → sophisticated handlers
                                                   
ggml-cpu.c → dispatcher → coordinator (backup/individual ops)
```

## Solution Strategy

Need to modify `llama-context.cpp` to route through the dispatcher instead of directly accessing coordinator:

1. **Add dispatcher graph processing function** 
2. **Replace coordinator direct access** with dispatcher calls in llama-context.cpp
3. **Keep coordinator as backend** managed by dispatcher

## Files Affected

- `/workspaces/llama.cpp/src/llama-context.cpp` - Primary fix needed here
- `/workspaces/llama.cpp/ggml/src/ggml-cpu/ggml-numa-operation-dispatch.c` - Add graph processing function
- Architecture: Make dispatcher the primary interface instead of coordinator

## Expected Outcome

All NUMA operations will route through the dispatcher's sophisticated handlers, eliminating fallback operations and providing proper MUL_MAT performance with NUMA awareness.
