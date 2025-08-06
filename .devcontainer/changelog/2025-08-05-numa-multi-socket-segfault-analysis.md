# NUMA Multi-Socket Test Segfault Fix

**Date**: 2025-01-05  
**Issue**: `test-numa-multi-socket.cpp` and `test-barrier.cpp` segfaulting due to NUMA mirror setup issues  

## Root Cause Analysis

Through detailed debugging, we identified **two separate but related issues**:

### Issue 1: Context Allocation NUMA Mirror Setup
**Problem**: Context-allocated tensors (like in `test-barrier.cpp`) were not using the proper NUMA-aware `tensor_set_data()` function.

**Details**: 
- Context allocation directly assigned `tensor->__data[0] = tensor->__data[1] = data`
- Backend allocation used `tensor_set_data(tensor, addr)` which properly handles NUMA mirroring
- This inconsistency caused context-allocated tensors to have NULL or invalid NUMA mirror pointers

**Fix Applied**: Modified `ggml_new_tensor_impl()` in `ggml/src/ggml.c` to use `tensor_set_data()` for context-allocated tensors, ensuring consistent NUMA mirror setup across all allocation paths.

### Issue 2: Virtual Memory NUMA Mirror Allocation  
**Problem**: Backend allocation creates virtual memory addresses for NUMA mirrors but doesn't actually map the memory at those addresses.

**Details**:
- Backend tensors get addresses like `0x357719fec880` (virtual memory range)
- These addresses are calculated from `GGML_MMAP_VIRTUAL_MEMORY_BASE_OFFSET + offset`
- But the actual virtual memory mapping is not set up, causing segfaults when accessed

**Current Status**: Issue 1 fixed, Issue 2 requires further investigation of virtual memory mapping in backend allocation.

## Tests Fixed
- ✅ `test-barrier.cpp` - Works correctly after context allocation fix
- ❌ `test-numa-multi-socket.cpp` - Still segfaults due to backend virtual memory issue

## Key Insights
1. **Allocation Path Differences**: Context vs Backend allocation use different tensor setup methods
2. **Virtual Memory Requirements**: NUMA mirroring uses virtual memory addresses that require proper mapping
3. **Initialization Dependencies**: Proper `llama_backend_init()` + `llama_numa_init()` required before tensor operations

## Next Steps
1. Fix virtual memory mapping in backend allocation for NUMA mirrors
2. Add guard clauses to prevent NUMA operations without proper initialization
3. Document proper NUMA initialization patterns for tests

## Code Changes
- `ggml/src/ggml.c`: Use `tensor_set_data()` in context allocation path
- `ggml/src/ggml-cpu/ggml-cpu.c`: Add NUMA availability checks  
- `tests/test-barrier.cpp`: Add proper NUMA initialization
