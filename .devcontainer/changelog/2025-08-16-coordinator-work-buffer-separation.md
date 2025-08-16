# NUMA Coordinator Work Buffer Calculation Separation

**Date:** August 16, 2025  
**Type:** Architecture Improvement  
**Impact:** Critical - Proper separation of concerns between dispatcher and coordinator

## Summary

Removed work buffer size calculation from the NUMA coordinator and enforced that all buffer size requirements must be provided by the dispatcher. This establishes a clear architectural boundary: **the dispatcher calculates, the coordinator executes**.

## What Was Done

### 1. Removed Coordinator Buffer Calculation
- **Deleted**: `ggml_numa_calculate_work_buffer_size()` function entirely from coordinator
- **Removed**: Forward declaration and all internal buffer size calculations
- **Reason**: Coordinator should not make decisions about buffer requirements

### 2. Updated Function Signatures
- **Modified**: `ggml_numa_coordinator_manager_submit_work()` to require `size_t required_buffer_size` parameter
- **Updated**: Both header declaration and implementation
- **Fixed**: All callers to provide buffer size (or 0 for legacy operations)

### 3. Fixed All Call Sites
- **Data parallel functions**: Set buffer size to 0 (let fallback system handle)
- **CGraph execution**: Set buffer size to 0 (legacy operations)
- **Dispatcher wrapper**: Pass 0 for legacy calls
- **Complex graph operations**: Set buffer size to 0

### 4. Preserved Dispatcher Buffer Calculation
- **Kept**: All buffer size calculation in `ggml-numa-operation-dispatch.c`
- **Verified**: Dispatcher still calculates 65536 bytes for MUL_MAT operations
- **Confirmed**: `submit_work_function()` correctly passes `work_size` to coordinator

## Architecture Before vs After

### Before (Problematic)
```
Dispatcher calculates work_size → Coordinator ignores it and calculates own → Mismatch
```

### After (Correct)
```
Dispatcher calculates work_size → Coordinator uses provided size → Consistency
```

## Technical Details

### Key Files Modified
1. **`ggml-numa-coordinator.h`**: Added `required_buffer_size` parameter to `submit_work()`
2. **`ggml-numa-coordinator.c`**: 
   - Removed `ggml_numa_calculate_work_buffer_size()` function (40+ lines)
   - Updated all callers to pass buffer size parameter
   - Set legacy operations to use buffer size 0
3. **`ggml-numa-operation-dispatch.c`**: Updated wrapper function for new signature

### Work Buffer Flow
1. **Dispatcher**: Calculates precise buffer requirements (e.g., 65536 bytes for 2x2 MUL_MAT)
2. **Coordinator**: Receives buffer size as parameter and uses it directly
3. **Auto-growth**: Coordinator ensures buffer meets dispatcher requirements
4. **Execution**: Operations get correctly sized buffers

## Impact Assessment

### ✅ **Improvements**
- **Clear separation of concerns**: Dispatcher decides, coordinator executes
- **Eliminated buffer size mismatches**: No more dual calculations
- **Consistent architecture**: All buffer requirements flow from dispatcher
- **Compilation success**: All function signatures consistent

### 🔄 **Status**
- **Buffer size communication**: ✅ **FIXED** - Coordinator now uses dispatcher's buffer size
- **MUL_MAT mathematical correctness**: ⚠️ **STILL NEEDS WORK** - Results still incorrect
- **Auto-growth system**: ✅ **WORKING** - Buffer growing from 1024 to 65536 bytes

### 📊 **Test Results**
- **Before**: MUL_MAT getting 16 bytes instead of calculated size
- **After**: MUL_MAT getting correct buffer size (65536 bytes) but results still wrong
- **Progress**: Infrastructure fixed, mathematical implementation needs debugging

## Next Steps

1. **Mathematical Correctness**: Debug why MUL_MAT operations return zeros despite correct buffer sizes
2. **Buffer Verification**: Add debug logging to confirm buffer contents and usage
3. **Kernel Investigation**: Examine if mathematical kernels are processing data correctly
4. **Legacy Cleanup**: Eventually provide proper buffer sizes for cgraph operations

## Design Principles Enforced

1. **Single Responsibility**: Coordinator only executes, never calculates requirements
2. **Explicit Dependencies**: All buffer requirements must be explicitly provided
3. **No Hidden Logic**: No "magic" buffer size calculations in coordinator
4. **Fail Fast**: Missing buffer sizes result in 0, making issues visible

This change establishes the proper architecture for the NUMA system where the intelligent dispatcher handles all computational requirements and the coordinator focuses purely on execution coordination.
