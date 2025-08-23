# 2025-08-23: Cleanup Duplicate ADD Kernel Implementations

## Summary
Cleaned up duplicate ADD kernel implementations to resolve code duplication and confusion around which kernel was being used.

## Changes Made

### 🗑️ **Removed Duplicate/Unused Files**
- **Deleted**: `ggml/src/ggml-cpu/numa-kernels/add.c` (old 521-line version with migration logic)
- **Deleted**: `ggml/src/ggml-cpu/numa-kernels/add.h` (old header)
- **Deleted**: `ggml/src/ggml-cpu/numa-kernels/numa-kernels-new.c` (unused alternative registry)

### ✅ **Renamed Active Implementation**
- **Renamed**: `add-direct.c` → `add.c` (325-line simplified version)
- **Renamed**: `add-direct.h` → `add.h`
- **Strategy**: Keep the working kernel that uses pre-allocated NUMA-local data without migration

### 🔧 **Updated Function Names**
- All function names changed from `ggml_numa_kernel_add_direct_*` to `ggml_numa_kernel_add_*`
- Updated kernel cache names from `"NUMA ADD Direct (Strategy)"` to `"NUMA ADD (Strategy)"`
- Updated includes and references in `numa-kernels.c`

### 📦 **Updated Build Configuration**
- Removed duplicate entries from `ggml/src/ggml-cpu/CMakeLists.txt`
- Now builds only the single `add.c` and `add.h` files

## Verification

### ✅ **Tests Still Pass**
- **Mathematical Correctness**: ✅ `ADD_mathematical_equivalence: PASSED`
- **Performance Benchmark**: ✅ Shows 1.13x average speedup 
- **Broadcasting**: ❌ Pre-existing failure (unrelated to this cleanup)

### 🎯 **Cache Correctly Shows New Names**
```
NUMA Cache: ADD[0] valid=true, kernel=NUMA ADD (Single/Single)
NUMA Cache: ADD[1] valid=true, kernel=NUMA ADD (Single/Multi)
NUMA Cache: ADD[2] valid=true, kernel=NUMA ADD (Data-Parallel/Single)
NUMA Cache: ADD[3] valid=true, kernel=NUMA ADD (Data-Parallel/Multi)
```

## Impact
- **Eliminated confusion** about which ADD kernel was active
- **Reduced codebase size** by removing ~500 lines of unused code
- **Simplified maintenance** with single ADD implementation
- **No performance regression** - same kernel logic, just cleaner naming

## Architecture Decision
Chose to keep the "direct" approach (pre-allocated NUMA-local data) rather than the migration approach, as it's simpler and already working effectively with the NUMA mirror strategy.
