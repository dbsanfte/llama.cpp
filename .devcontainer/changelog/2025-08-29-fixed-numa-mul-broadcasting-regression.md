# 2025-08-29: Fixed NUMA MUL Broadcasting Regression

## Problem
The NUMA MUL kernel had a broadcasting regression that caused 4/20 test cases to fail. The issue was specifically with multi-dimensional broadcasting cases like Matrix [256x32] + Vector [1x32].

## Root Cause Analysis
1. **Incorrect element indexing**: The low-overhead MUL kernel used `src1_idx = i % src1_elements` for broadcasting
2. **Wrong tensor interpretation**: This treated broadcasting as simple cyclic repetition rather than proper multi-dimensional broadcasting
3. **Example failure**: For element[1] in [256x32] + [1x32]:
   - **Wrong**: Used `src1[1]` (10.100) → Result: 10.201
   - **Correct**: Should use `src1[0]` (10.000) → Result: 10.100

## Technical Details
**Tensor Layout Understanding:**
- src0: `ne=[256,32,1,1]` = 256 columns × 32 rows
- src1: `ne=[1,32,1,1]` = 1 column × 32 rows  
- Broadcasting: For each row r, `dst[r][c] = src0[r][c] * src1[r][0]`

**Broken Logic:**
```c
// OLD - Wrong broadcasting
const int64_t src1_idx = i % src1_elements;  // Treats as cyclic repetition
dst_data[i] = src0_data[i] * src1_data[src1_idx];
```

**Fixed Logic:**
```c
// NEW - Correct multi-dimensional broadcasting  
const int64_t row = i / ne0;       // Convert to 2D coordinates
const int64_t col = i % ne0;
const int64_t src1_row = row % ne11;  // Apply broadcasting with modulo
const int64_t src1_col = col % ne10;
const int64_t src1_idx = src1_row * ne10 + src1_col;  // Proper linear index
dst_data[i] = src0_data[i] * src1_data[src1_idx];
```

## Changes Made
1. **Fixed `ggml_numa_kernel_mul_execute_low_overhead()`** in `ggml/src/ggml-cpu/numa-kernels/mul.c`:
   - Replaced simple modulo indexing with proper 2D coordinate calculation
   - Added support for multi-dimensional broadcasting following ggml tensor conventions
   - Added debug logging for troubleshooting

2. **Verified mathematical equivalence** with reference implementation
3. **Cleaned up test debug output** in `tests/test-numa-mathematical-correctness-mul.cpp`

## Results
✅ **Before Fix**: 16/20 broadcasting cases passing, 4 failing with ~1% error  
✅ **After Fix**: 20/20 broadcasting cases passing, mathematical equivalence achieved
✅ **Complete Test Results**: 3/3 MUL tests passing (mathematical equivalence, quantization coverage, broadcasting regression)

## Files Modified
- `ggml/src/ggml-cpu/numa-kernels/mul.c` - Fixed broadcasting logic in low-overhead kernel
- `tests/test-numa-mathematical-correctness-mul.cpp` - Cleaned up temporary debug output

## Verification
```bash
./build/bin/test-numa-mathematical-correctness-mul
# Result: ✅ NUMA Mathematical Correctness: ALL TESTS PASSED
```

The NUMA MUL kernel now handles all broadcasting scenarios correctly and produces mathematically equivalent results to the reference implementation.
