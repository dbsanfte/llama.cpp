# NUMA Virtual Node Functions Removal

**Date**: August 24, 2025
**Completed By**: AI Assistant with Human Guidance

## Summary

Removed obsolete virtual node functions from the NUMA simple coordinator that were confusing and unused. This cleanup simplifies the codebase and eliminates potential confusion for developers.

## Analysis of Removed Functions

### Functions Removed:
1. `ggml_numa_set_virtual_node(int node)` - Set virtual NUMA node
2. `ggml_numa_get_virtual_node(void)` - Get virtual NUMA node  
3. `g_virtual_numa_node` - Thread-local storage variable

### Usage Analysis:
- **`ggml_numa_set_virtual_node()`**: Called in exactly ONE place (line 225 in dispatch worker) but the value was never meaningfully used
- **`ggml_numa_get_virtual_node()`**: NEVER called anywhere in the codebase
- **`g_virtual_numa_node`**: Only referenced in `ggml_numa_get_current_node()`, which is also never called externally
- **Virtual node mechanism**: Completely redundant with the direct thread-local `ggml_current_numa_node` approach

## Why These Functions Were Obsolete

The virtual node system appears to be a legacy approach that was replaced by the more direct thread-local variable system:

### Old Approach (Removed):
```c
// Set virtual node
ggml_numa_set_virtual_node(numa_node);

// Later, get current node via virtual node indirection  
int ggml_numa_get_current_node(void) {
    if (g_virtual_numa_node >= 0) {
        return g_virtual_numa_node;  // Use virtual node
    }
    // Fallback to physical detection...
}
```

### Current Approach (Kept):
```c
// Set thread-local NUMA context directly
extern __thread int ggml_current_numa_node;
ggml_current_numa_node = numa_node;

// Kernels read directly from thread-local variable
extern __thread int ggml_current_numa_node; // Used in kernels
```

## Changes Made

### Removed from `ggml-numa-simple-coordinator.c`:
1. **Virtual node function call**: Removed `ggml_numa_set_virtual_node(numa_node);` from dispatch worker
2. **Function implementations**: Removed both `ggml_numa_set_virtual_node()` and `ggml_numa_get_virtual_node()`
3. **Thread-local variable**: Removed `static __thread int g_virtual_numa_node = -1;`
4. **Virtual node check**: Simplified `ggml_numa_get_current_node()` to remove virtual node indirection

### Removed from `ggml-numa-simple-coordinator.h`:
1. **Function declaration**: Removed `void ggml_numa_set_virtual_node(int node);` from public API

### Simplified Logic:
- `ggml_numa_get_current_node()` now directly uses physical NUMA node detection
- No more confusing indirection through virtual node layer
- Cleaner, more direct approach to NUMA node management

## Benefits

### Code Simplification:
- **Reduced complexity**: Eliminated unnecessary indirection layer
- **Clearer intent**: Direct thread-local approach is more obvious
- **Fewer functions**: Removed 2 unused/obsolete functions from public API

### Developer Experience:
- **Less confusion**: No more wondering what "virtual node" means vs "current node"
- **Consistent approach**: All NUMA context now uses direct thread-local variables
- **Easier debugging**: One less layer of indirection to track through

### Maintainability:
- **Reduced code surface**: Less code to maintain and test
- **Eliminated dead code**: Removed functions that served no purpose
- **Cleaner architecture**: Focused on single approach (thread-local variables)

## Technical Impact

- **No functional changes**: NUMA execution behavior unchanged
- **API cleanup**: Removed unused functions from public header
- **Performance**: Eliminated unnecessary function calls and indirection
- **Memory**: Reduced thread-local storage usage (removed `g_virtual_numa_node`)

## Files Modified

1. `ggml/src/ggml-cpu/ggml-numa-simple-coordinator.c`:
   - Removed virtual node function call from dispatch worker
   - Removed virtual node function implementations  
   - Removed `g_virtual_numa_node` variable
   - Simplified `ggml_numa_get_current_node()` function

2. `ggml/src/ggml-cpu/ggml-numa-simple-coordinator.h`:
   - Removed `ggml_numa_set_virtual_node()` function declaration

## Validation

- ✅ Build successful with no errors
- ✅ Mathematical correctness tests pass
- ✅ NUMA functionality unaffected
- ✅ No breaking changes to used APIs

## Next Steps

This cleanup paves the way for:
1. **Further simplification**: Look for other obsolete patterns in the codebase
2. **Consistent approach**: Ensure all NUMA context uses thread-local variables directly
3. **Documentation updates**: Update any documentation referring to removed functions

The codebase is now cleaner and less confusing, with a single consistent approach to NUMA node management through direct thread-local variables.
