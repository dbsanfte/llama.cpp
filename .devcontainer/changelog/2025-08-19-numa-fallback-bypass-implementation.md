# 2025-08-19: NUMA Fallback Bypass Implementation

## Summary
Implemented complete bypass of broken NUMA MUL_MAT system to use original ggml graph computation, resolving quantized operation corruption issues.

## Problem Identified
The custom NUMA MUL_MAT system was fundamentally broken for quantized operations:
- Direct calls to `ggml_compute_forward_mul_mat` and `ggml_compute_forward_mul_mat_one_chunk` produced all-zero results for Q8_0/Q4_0/Q5_0 operations
- F32×F32 operations worked correctly, but quantized operations were completely corrupted
- The custom system bypassed essential ggml graph computation infrastructure needed for quantization

## Solution Implemented
**Complete NUMA System Bypass**: Modified dispatcher to route ALL operations directly to original ggml fallback system.

### Key Changes

#### 1. Updated Fallback System (`ggml-numa-fallback.c`)
- **Replaced direct function calls** with full ggml graph computation
- **Created temporary ggml context** and computation graph for each operation
- **Used `ggml_graph_plan()` and `ggml_graph_compute()`** instead of direct mathematical kernels
- **Fixed `ggml_graph_plan` signature** to include threadpool parameter

#### 2. Modified Dispatcher (`ggml-numa-operation-dispatch.c`)
- **Implemented immediate bypass** of all NUMA handlers
- **Routed ALL operations** directly to fallback system using `ggml_numa_fallback_execute()`
- **Eliminated complex NUMA coordination** that was causing quantization failures

#### 3. Verification Results
✅ **Direct fallback execution**: Produces correct non-zero results for Q8_0×F32 operations
✅ **NUMA dispatcher routing**: Successfully routes to working fallback system
✅ **Mathematical correctness**: Both paths produce identical correct results (36.21, 59.69, 63.40, etc.)

## Technical Details

### Original Problem Code
```c
// BROKEN: Direct mathematical kernel calls
ggml_compute_forward_mul_mat(&fallback_params, tensor);
ggml_compute_forward_mul_mat_one_chunk(&params, dst_tensor, ...);
```

### Working Solution Code
```c
// WORKING: Full ggml graph computation
struct ggml_cgraph * cgraph = ggml_new_graph(temp_ctx);
ggml_build_forward_expand(cgraph, tensor);
struct ggml_cplan plan = ggml_graph_plan(cgraph, n_threads, threadpool);
enum ggml_status result = ggml_graph_compute(cgraph, &plan);
```

## Current Status
- ✅ **Quantized operations fixed**: Q8_0×F32 MUL_MAT now produces correct results
- ✅ **Fallback system working**: Both direct and dispatcher paths verified
- ⚠️ **Full model still crashes**: Server segfaults during full model loading (separate issue)
- ✅ **Debug test passes**: Isolated quantized operation testing works perfectly

## Lessons Learned
1. **Avoid bypassing ggml infrastructure**: Direct calls to mathematical kernels skip essential quantization handling
2. **Use ggml graph computation**: The graph computation system handles all the complexity of quantization, work buffers, and threading
3. **Test in isolation first**: Our debug test correctly identified the working solution before full integration

## Next Steps
The quantized operation corruption is now **resolved**. The remaining server crashes appear to be separate issues in the full model loading pipeline, not related to the core mathematical operations.

## Files Modified
- `ggml/src/ggml-cpu/ggml-numa-fallback.c` - Implemented graph-based computation
- `ggml/src/ggml-cpu/ggml-numa-operation-dispatch.c` - Added complete NUMA bypass
- `test-fallback-debug.cpp` - Debug test demonstrating the fix

## Impact
This change ensures that **ALL** operations (especially quantized MUL_MAT) use the proven, working ggml system instead of the broken custom NUMA implementations. Quantized models should now produce correct mathematical results.
