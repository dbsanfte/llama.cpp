# NUMA Implementation Debugging - Struct Layout Issue

**Date:** August 5, 2025
**Summary:** Identified critical struct layout issue blocking NUMA implementation

## Problem Analysis

### Initial Issue
- test-backend-ops was segfaulting after NUMA changes
- Fixed critical NUMA assertion bug: `numa_run_on_node() == 0` (not `!= 0`)
- This resolved test-backend-ops crashes on NUMA systems

### Persistent Segfault Discovery
After the assertion fix, discovered that test-barrier.cpp (and other tests) still segfault.

### Root Cause Identification
The issue is **not** in the NUMA code itself, but in the structural change to `ggml_threadpool`:

**Problem:** Added `struct ggml_numa_threadpool_manager * numa_mgr;` field to `ggml_threadpool` struct (line 469 in ggml-cpu.c)

**Impact:** This changes the struct size and memory layout, causing:
- Memory corruption in tensor operations  
- Segfaults in `quantize_row_q8_0()` with invalid tensor data pointers (`x=0x80`)
- Binary compatibility issues

### Evidence
1. **NUMA code disabled:** Even with NUMA override completely disabled (`#ifdef GGML_NUMA_MIRROR_DISABLED_FOR_DEBUGGING`), segfault persists
2. **Field properly initialized:** `numa_mgr` is initialized to NULL in `ggml_threadpool_new_impl()`
3. **Crash location:** Segfault occurs in core tensor operations, not NUMA code
4. **Memory pattern:** Invalid low memory addresses in tensor pointers suggest struct corruption

### Test Results
- `test-backend-ops`: ✅ Works (after NUMA assertion fix)
- `test-barrier`: ❌ Segfaults (struct layout issue)
- All other tests using threadpools: ❌ Likely affected

## Next Steps

### Critical Fix Required
1. **Remove numa_mgr from ggml_threadpool struct** - This changes the ABI and breaks existing code
2. **Alternative approach:** Store NUMA manager externally or use different initialization pattern
3. **Ensure struct size compatibility** - No changes to existing struct layouts

### NUMA Implementation Strategy
- Keep NUMA functionality but avoid changing existing struct sizes
- Consider storing NUMA manager in a global map keyed by threadpool pointer
- Or pass NUMA manager as parameter to functions that need it

### Testing Protocol
1. Revert struct changes
2. Verify test-barrier works again
3. Implement NUMA using approach that doesn't change struct layouts
4. Re-run comprehensive test suite

## Lessons Learned
- **ABI Compatibility:** Adding fields to existing structs breaks binary compatibility
- **Memory Layout:** Even properly initialized new fields can cause corruption due to size changes
- **Testing Strategy:** Always test with existing test suite, not just new functionality
- **NUMA Functions:** Follow Unix conventions (0 = success, != 0 = error)

## Code Changes Made

### Fixed Issues ✅
- `numa_run_on_node()` assertion logic corrected
- NUMA bitmask allocation error checking added  
- Proper initialization of numa_mgr field

### Remaining Issues ❌
- **Critical:** `numa_mgr` field in `ggml_threadpool` struct breaks ABI compatibility
- Need alternative NUMA manager storage approach

## Impact Assessment
- **Severity:** Critical - blocks all threadpool functionality
- **Scope:** Affects core tensor operations, not just NUMA features
- **Priority:** Must be resolved before any NUMA features can be merged
