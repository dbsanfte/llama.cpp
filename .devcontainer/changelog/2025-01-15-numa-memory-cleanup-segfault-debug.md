# NUMA Memory Cleanup Segfault Investigation and Fix
*Date: January 15, 2025*

## Problem Summary
During debugging of a segfault that occurred when running `llama-cli` with `--numa mirror` flag, we discovered a critical memory management bug in the NUMA repack buffer implementation.

## Root Cause Analysis

### Initial Investigation
- **Symptom**: Segfault occurring during cleanup phase after successful model inference
- **Location**: `llama_vocab::impl::~impl()` destructor → `std::unordered_map` cleanup → `std::string` destruction
- **Stack trace showed**: Memory corruption during standard library cleanup operations

### Technical Root Cause
The issue was in `ggml/src/ggml-cpu/repack.cpp` lines 1507-1516:

1. **NUMA Memory Allocation**: Code used `numa_alloc_onnode(size, numa_node)` to allocate NUMA-aware memory
2. **Improper Buffer Wrapping**: Memory was wrapped with `ggml_backend_cpu_buffer_from_ptr()` 
3. **Missing Cleanup**: `ggml_backend_cpu_buffer_from_ptr` has `free_buffer = NULL`, meaning NUMA memory was never freed with `numa_free()`
4. **Memory Corruption**: This created a memory leak/corruption where NUMA-allocated memory was never properly released

### Evidence
- Without `--numa mirror`: Program runs successfully and exits cleanly
- With `--numa mirror`: Consistent segfault during cleanup phase
- GDB stack trace confirmed crash in vocabulary destructor trying to clean up corrupted memory

## Attempted Solutions

### Solution 1: Global Registration with `atexit()`
- **Approach**: Created global registry tracking NUMA allocations, cleaned up at process exit
- **Result**: Failed - buffers are freed during normal operation, not at process exit
- **Issue**: Timing mismatch between buffer lifecycle and process exit

### Solution 2: Custom NUMA Buffer Interface
- **Approach**: Created custom `ggml_backend_buffer_i` interface with proper `free_buffer` callback
- **Implementation**: 
  - Custom context structure storing NUMA pointer and size
  - Custom `free_buffer` function calling `numa_free()`
  - Custom buffer creation function `ggml_backend_cpu_repack_buffer_from_numa_ptr()`
- **Current Status**: Under development - interface created but causing execution-phase segfaults

## Current Status
- ✅ Root cause identified: NUMA allocation/standard deallocation mismatch
- ✅ Segfault consistently reproduced and debugged
- ✅ Problem isolated to `--numa mirror` mode specifically  
- ⚠️ Custom buffer interface implemented but needs debugging
- ❌ Final fix not yet working

## Next Steps
1. Debug custom buffer interface implementation
2. Ensure all required interface functions are properly implemented
3. Test memory allocation/deallocation flow
4. Verify fix resolves segfault without introducing new issues

## Code Changes
- Modified `ggml/src/ggml-cpu/repack.cpp` with custom NUMA buffer interface
- Added proper `numa_free()` cleanup in custom buffer destructor
- Need to complete debugging of interface implementation

## Impact
This is a critical memory management bug that makes `--numa mirror` mode unusable due to consistent segfaults during cleanup. Fix is essential for NUMA-aware memory allocation functionality.
