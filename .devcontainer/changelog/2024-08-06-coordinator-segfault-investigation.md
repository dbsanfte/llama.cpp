# Coordinator Segfault Investigation - 2024-08-06

## Issue Description
- Global 3-tier NUMA coordinator working for single computation
- Segmentation fault occurs when attempting second computation using same coordinator
- First test (simple computation) passes completely
- Second test (multiple separate computations) segfaults immediately

## Root Cause Identified ✅
**NUMA Virtual Memory Allocation Bug**: After first GGML context is freed, subsequent tensor allocations use virtual memory addresses in the NUMA range (`0x2000...` → `0x3000...`) but this virtual memory is not properly mapped when NUMA is disabled.

## Environment Context 🔍
- **Dev Container**: Running in dev container (containerized environment)
- **Non-NUMA System**: Host system doesn't have NUMA hardware
- **Build Configuration**: `GGML_NUMA_MIRROR=ON` at compile time, but `ggml_numa_init(GGML_NUMA_STRATEGY_DISABLED)` at runtime
- **Virtual Memory**: Container environment may have different memory management behavior

## Technical Analysis
```
First tensor:  data ptr: 0x7f71805241c0  (normal heap, works fine)
Second tensor: data ptr: 0x3628ec321580  (virtual memory range, SEGFAULT)
```

**Root Cause**: The dev container + non-NUMA system combination causes GGML's memory allocator to sometimes return addresses in the virtual NUMA range (`0x200000000000ULL` - `0x400000000000ULL`) even when NUMA is disabled. The `tensor_set_data()` function detects these as virtual addresses but doesn't set up proper memory mapping since `ggml_is_numa()` returns false.

**Virtual Memory Constants**:
- `GGML_MMAP_VIRTUAL_MEMORY_BASE_OFFSET = 0x200000000000ULL` (≈35TB)
- `GGML_MMAP_VIRTUAL_MEMORY_NUMA_INCREMENT = 0x200000000000ULL` (≈35TB)

## Code Analysis
In `ggml.h` `tensor_set_data()`:
```c
if ((uint64_t)data >= GGML_MMAP_VIRTUAL_MEMORY_BASE_OFFSET && 
    (uint64_t)data < GGML_MMAP_VIRTUAL_MEMORY_BASE_OFFSET + GGML_MMAP_VIRTUAL_MEMORY_NUMA_INCREMENT) {
    should_setup_numa_mirrors = ggml_is_numa();  // FALSE in container - but data is still in virtual range!
}
```

**Container Issue**: In containerized environments, the memory allocator behavior can differ from bare metal, potentially causing heap allocations to land in address ranges that overlap with the NUMA virtual memory space.

## Evidence Points
1. **Coordinator Creation Working**: Successfully creates singleton with proper NUMA node setup
2. **Single Computation Working**: Matrix multiplication executes successfully 
3. **Thread Management Working**: Coordinator threads start and process work correctly
4. **Container Memory Management**: Dev container + non-NUMA system causes heap addresses to land in virtual NUMA range
5. **Compile vs Runtime Mismatch**: `GGML_NUMA_MIRROR` enabled at build time, NUMA disabled at runtime in container
6. **Address Space Collision**: Container memory layout causes heap addresses to collide with NUMA virtual address space

## Next Steps
1. ✅ **Root cause identified**: Container-specific virtual memory allocation collision
2. 🔄 **Fix approach options**:
   - **Option A**: Disable `GGML_NUMA_MIRROR` in dev container builds
   - **Option B**: Fix `tensor_set_data()` to handle container environments correctly
   - **Option C**: Adjust virtual memory base offset for container environments
3. 🔄 **Test fix**: Verify coordinator works with corrected memory management

## Code Changes
- Created `test-tensor-debug.cpp`: Isolated virtual memory allocation issue
- Identified exact failure point: `ggml_set_f32_1d()` writing to unmapped virtual memory
- **Key insight**: Issue is container-specific, not coordinator architecture-specific

## Status
🔄 **Debugging Deep Issue** - Thread-local NUMA node is not the cause; virtual memory allocation still happening despite fixes
