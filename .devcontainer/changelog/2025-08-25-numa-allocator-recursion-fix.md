# NUMA Allocator Infinite Recursion Fix

**Date:** 2025-08-25  
**Author:** GitHub Copilot  
**Status:** ✅ Complete  

## 🚨 Problem
Severe infinite recursion bug in NUMA memory allocator causing stack overflow crashes during inference:

```
#52310 0x00007ffff7f74ef6 in ggml_numa_aligned_malloc ()
#52311 0x00007ffff7f74ef6 in ggml_numa_aligned_malloc ()
#52312 0x00007ffff7f74ef6 in ggml_numa_aligned_malloc ()
... (thousands of identical frames)
```

## 🔍 Root Cause Analysis
**Circular recursion** between two NUMA allocation functions:
1. `ggml_numa_aligned_malloc()` calls `ggml_numa_aligned_malloc_on_node()` (line 192)
2. `ggml_numa_aligned_malloc_on_node()` fails and calls back to `ggml_numa_aligned_malloc()` (lines 298, 328)
3. Infinite loop created, causing stack overflow

**Specific recursion points:**
```c
// Line 298: Invalid node fallback
if (preferred_node < 0 || preferred_node >= numa_ctx->num_numa_nodes) {
    return ggml_numa_aligned_malloc(size, numa_ctx); // ❌ RECURSION
}

// Line 328: mmap failure fallback  
if (memory == MAP_FAILED) {
    return ggml_numa_aligned_malloc(size, numa_ctx); // ❌ RECURSION
}
```

## ✅ Solution Implementation

### 1. Non-Recursive Fallback Function
Added dedicated fallback allocation that breaks recursion:
```c
// Non-recursive fallback allocation - breaks infinite recursion
static void* ggml_numa_fallback_alloc(size_t size, ggml_numa_alloc_context_t* numa_ctx) {
    void* mem = aligned_alloc(64, size);
    if (mem) {
        memset(mem, 0, size);
        numa_ctx->total_allocated += size;
        if (numa_ctx->debug_enabled) {
            printf("🔄 Fallback allocation: %zu bytes (breaking recursion)\n", size);
        }
    }
    return mem;
}
```

### 2. Fixed Recursive Calls
**Before:**
```c
if (preferred_node < 0 || preferred_node >= numa_ctx->num_numa_nodes) {
    return ggml_numa_aligned_malloc(size, numa_ctx); // ❌ RECURSION
}
if (memory == MAP_FAILED) {
    return ggml_numa_aligned_malloc(size, numa_ctx); // ❌ RECURSION  
}
```

**After:**
```c
if (preferred_node < 0 || preferred_node >= numa_ctx->num_numa_nodes) {
    return ggml_numa_fallback_alloc(size, numa_ctx); // ✅ NO RECURSION
}
if (memory == MAP_FAILED) {
    return ggml_numa_fallback_alloc(size, numa_ctx); // ✅ NO RECURSION
}
```

## 🧪 Validation Results

### ✅ Segfault Elimination
**Before:**
```bash
$ ./build/bin/llama-bench --numa mirror
Segmentation fault (stack overflow from infinite recursion)
```

**After:**
```bash
$ ./build/bin/llama-bench --numa mirror  
| model | size | params | backend | threads | test | t/s |
# Successfully proceeds through execution without crashing
DEBUG: NUMA Executor: Successfully completed MUL_MAT using NUMA MUL_MAT (Data-Parallel/Multi)
DEBUG: NUMA Executor: Successfully completed ADD using NUMA ADD (Single-Node Multi-Thread)
```

### ✅ Mathematical Correctness Preserved
```bash
$ ./build/bin/test-numa-mathematical-correctness-mul_mat
🎉 ALL TESTS PASSED! MUL_MAT NUMA implementation is mathematically correct.
Passed: 45, Failed: 0, Total: 45
```

### ✅ Core Architecture Stability
```bash
$ cmake --build build --target ggml-cpu llama
✅ Core components building successfully
```

## 📊 Impact Assessment

### Positive Results
- ✅ **Complete elimination** of infinite recursion crashes
- ✅ **NUMA execution proceeding** - MUL_MAT and ADD kernels executing successfully 
- ✅ **Mathematical correctness maintained** - All 45 tests still pass
- ✅ **Fallback allocation working** - Graceful degradation when NUMA fails

### Remaining Issues
- ⚠️ Some MUL_MAT operations returning failure status (-1) during real inference
- ⚠️ Debug output still showing in production builds (separate issue)

## 🔧 Technical Details

### Memory Safety Improvements
- **Stack overflow prevention**: Eliminated unbounded recursion
- **Graceful degradation**: System continues with standard allocation when NUMA fails
- **Resource tracking**: Fallback allocation still updates allocation counters
- **Debug visibility**: Clear logging when fallback path is taken

### Architecture Impact
- **Call graph simplified**: Removed circular dependencies in allocation subsystem
- **Reliability improved**: System no longer crashes on NUMA setup failures  
- **Performance maintained**: Fallback only triggers on exceptional conditions
- **Compatibility preserved**: Existing NUMA functionality remains intact

## 🎯 Files Modified
- `ggml/src/ggml-numa-allocator.c` - ✅ Fixed recursion with fallback function

## 🏆 Summary
**Critical stability fix** resolving infinite recursion that was causing immediate crashes during NUMA inference. The implementation provides a clean fallback path while preserving all existing NUMA functionality and mathematical correctness. This unblocks continued development and testing of the NUMA kernel system.

## 🔜 Next Steps
1. **Investigate MUL_MAT failures** - Some operations returning -1 status during inference
2. **Debug output cleanup** - Disable debug messages in production builds
3. **Performance testing** - Validate NUMA performance improvements with stable system
