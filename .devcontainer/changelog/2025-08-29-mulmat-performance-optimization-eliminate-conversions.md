# MUL_MAT Performance Optimization - Eliminate Expensive F32 Conversions

**Date:** 2025-08-29  
**Type:** Performance Optimization  
**Impact:** Major Performance Improvement  
**Scope:** NUMA MUL_MAT Kernel Execution  

## Summary

Implemented critical performance optimization in MUL_MAT kernel to eliminate expensive F32→quantized conversions that were causing 1.674ms average overhead per operation. The optimization rejects MUL_MAT operations requiring conversion and allows them to fallback to the much faster reference implementation.

## Performance Impact

**Massive Performance Improvement:**

| Mode | Before | After | Improvement | Gap to Reference |
|------|--------|-------|-------------|------------------|
| **Prefill (pp16)** | 27.65 tps | **61.80 tps** | **+123%** | -72% vs 218.80 tps |
| **Text Gen (tg16)** | 3.32 tps | **4.09 tps** | **+23%** | -79% vs 19.85 tps |

**Performance Breakdown Changes:**
- **Before**: KernelNumaExec 62.5% (1,123 ops at 1.674ms avg each) - **ELIMINATED**
- **After**: ExecutorFallback 99.1% (fast reference implementation)

## Root Cause Analysis

**Problem:** MUL_MAT kernel was doing expensive F32→Q8_0 conversion on every operation:
- Q8_0 weights × F32 activations → converting F32 to Q8_0 for each matrix multiplication
- 1.674ms average per operation (vs ~0.04ms for reference)
- Per-operation buffer allocation and data conversion loops

**Debug Evidence:**
```
src0_type=8 (Q8_0), src1_type=0 (F32), vec_dot_type=8 (Q8_0), needs_conversion=YES
MUL_MAT buffer calc: buffer_size=1904
Converting F32->Q8_0 row[0]: src_ptr=..., dst_ptr=...
```

## Solution Implementation

**Added Performance Guard:**
```c
// PERFORMANCE OPTIMIZATION: Reject operations requiring expensive F32->quantized conversion
// For Q8_0 × F32 operations, fallback to reference implementation which is faster
const enum ggml_type vec_dot_type = type_traits->vec_dot_type;
if (src1->type != vec_dot_type && src1->type == GGML_TYPE_F32) {
    NUMA_LOG_DEBUG("MUL_MAT query: REJECTING - avoiding expensive F32->%d conversion (src0_type=%d, src1_type=%d)", 
                   vec_dot_type, src0->type, src1->type);
    result.supported = false;
    return result;
}
```

**Strategy:**
1. **Detect conversion operations** in query phase (zero overhead)
2. **Reject expensive conversions** before execution starts  
3. **Fallback to reference** which handles Q8_0 × F32 directly without conversion
4. **Keep NUMA acceleration** for operations that don't require conversion

## Files Modified

### ggml/src/ggml-cpu/numa-kernels/mul_mat.c
- **ggml_numa_kernel_mul_mat_query()**: Added performance guard to reject F32→quantized conversions

**Location:** Lines ~785-795 in query function  
**Change:** Early rejection of expensive conversion operations

## Validation Results

### Performance Verification
```bash
# Before optimization
./llama-bench --numa mirror -n 16 -p 16
# pp16: 27.65 tps, tg16: 3.32 tps

# After optimization  
./llama-bench --numa mirror -n 16 -p 16  
# pp16: 61.80 tps (+123%), tg16: 4.09 tps (+23%)
```

### Debug Verification
```bash
GGML_NUMA_DEBUG=1 ./llama-bench --numa mirror
# Shows: "MUL_MAT query: REJECTING - avoiding expensive F32->8 conversion"
```

### Performance Breakdown
```bash
GGML_NUMA_PERF=2 ./llama-bench --numa mirror
# Before: KernelNumaExec 62.5% (1.674ms avg)
# After:  ExecutorFallback 99.1% (fast reference)
```

## Technical Analysis

**Why Reference is Faster:**
- Reference implementation has direct Q8_0 × F32 vector dot functions
- No per-operation conversion overhead
- Optimized SIMD implementations for mixed-type operations
- No NUMA coordination overhead for operations that don't benefit

**NUMA Strategy:**
- **Keep accelerating**: Operations that benefit from NUMA parallelization (large ADD, etc.)
- **Smart fallback**: Operations where reference is faster due to specialized implementations
- **Zero overhead**: Decision made at query time, not execution time

## Next Steps

1. ✅ **Immediate**: Eliminated expensive conversion overhead
2. 🔄 **Medium-term**: Implement direct Q8_0 × F32 NUMA kernels (without conversion)
3. 🔄 **Long-term**: Add support for more operations (RMS_NORM, MUL, ROPE, etc.)

## Impact Assessment

**Positive:**
- Major performance improvement for current workloads
- Eliminated worst performance bottleneck  
- Maintained mathematical correctness (ADD kernels still accelerated)
- Smart fallback strategy preserves performance

**Remaining Work:**
- Still 72% gap to reference on prefill (mainly due to unsupported operations)
- Opportunity to add more NUMA-accelerated operations
- Potential for direct mixed-type NUMA kernels

---

**Category:** Performance Optimization  
**Priority:** High Impact  
**Dependencies:** None  
**Breaking Changes:** None (pure optimization)
