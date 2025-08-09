# Apples-to-Apples Baseline Comparison & Coordinator Warmup

**Date:** 2024-12-29  
**Type:** Test Enhancement / Performance Fix  
**Component:** NUMA Performance Testing  

## Problem

The comprehensive NUMA performance test had two critical issues that made performance comparisons misleading:

### Issue 1: Incomparable Baseline Operations
- **Baseline tests**: Used direct matrix multiplication with `ggml_mul_mat()` 
- **Coordinator tests**: Used tensor element-wise operations with batch processing
- **Result**: Comparing different operation types made speedup measurements meaningless

### Issue 2: No Coordinator Warmup
- **Cold coordinator instances**: First test execution included initialization overhead
- **Inconsistent timing**: Setup costs contaminated benchmark measurements  
- **Result**: Artificially inflated execution times and reduced apparent performance

## Solution

### 1. Apples-to-Apples Baseline Testing
Completely rewrote the baseline test to use **identical operations** as coordinator tests:

#### Before: Different Operations
```cpp
// Baseline: Matrix multiplication
ggml_tensor* c = ggml_mul_mat(ctx, a, b);  // GEMM operations

// Coordinator: Tensor element-wise operations  
ggml_tensor* result = ggml_mul(ctx, a, b);  // Element-wise multiplication
```

#### After: Same Operations
```cpp
// Both baseline and coordinator now use identical tensor operations:
for (batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, tensor_size);
    ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, tensor_size);  
    ggml_tensor* result = ggml_mul(ctx, a, b); // Same element-wise ops
}
```

### 2. Coordinator Warmup Implementation
Added mandatory warmup runs to eliminate cold-start overhead:

```cpp
// WARMUP RUN - Execute once to initialize coordinator state
int warmup_result = ggml_numa_coordinator_manager_compute_graph(mgr, graph);
ggml_numa_coordinator_manager_wait_for_completion(mgr);

// Then execute actual benchmark iterations (timing starts after warmup)
for (int iter = 0; iter < iterations; iter++) {
    // Actual timed benchmarks...
}
```

## Implementation Details

### New Baseline Function
Created `run_single_core_baseline_tensor_ops()` that mirrors coordinator operations:
- **Same tensor creation**: 1D tensors with configurable size
- **Same operations**: Element-wise multiplication (`ggml_mul`)
- **Same batch processing**: Multiple tensors per iteration
- **Same computation graph**: Identical structure to coordinator tests
- **Same warmup**: Single warmup run before timing

### Updated Test Structure  
```cpp
void test_single_core_baseline() {
    printf("Using SAME operations as NUMA coordinator tests for apples-to-apples comparison\n");
    
    // Test each batch size used in coordinator tests
    for (int batch_size : g_test_config.batch_sizes) {
        BaselineResult result = run_single_core_baseline_tensor_ops(
            batch_size, g_test_config.tensor_size, g_test_config.baseline_iterations);
        // Now truly comparable to coordinator results
    }
}
```

### Enhanced BaselineResult Structure
```cpp
struct BaselineResult {
    int batch_size;        // Now tracks batch size for comparison
    int64_t tensor_size;   // Tracks tensor size used
    // ... existing fields
};
```

## Files Modified

- `/workspaces/llama.cpp/tests/test-comprehensive-numa-performance.cpp`
  - Added `run_single_core_baseline_tensor_ops()` function using identical operations as coordinator
  - Updated `test_single_core_baseline()` to test same batch sizes as coordinator tests
  - Added warmup run to `benchmark_matrix_multiplication_with_cpu_config()`
  - Enhanced `BaselineResult` structure with batch_size and tensor_size tracking
  - Updated baseline output format to match coordinator test parameters

## Results & Impact

### Before Fix: Misleading Comparison
```
❌ Baseline: Matrix multiplication GEMM operations
❌ Coordinator: Element-wise tensor operations  
❌ Speedup: Meaningless comparison of different operation types
```

### After Fix: True Performance Comparison
```
✅ Baseline (single-threaded tensor ops): 0.196 GOPS
✅ Auto-Optimized coordinator: 2.34 GOPS  
✅ Real speedup: 11.9x improvement (truly meaningful!)
```

### Quick Mode Example Results
```
BASELINE SUMMARY (Same Operations as Coordinator Tests):
Batch   Test           Avg(ms)   GOPS    Status
-----   -------------- -------  ------   ------
4       TensorOps-B4     21.41   0.196   ✅
8       TensorOps-B8     45.06   0.186   ✅

1. CPU MASK PERFORMANCE IMPACT
   Primary-Only               1.37 GOPS    (7.0x speedup over baseline)
   Hyperthreading             2.24 GOPS    (11.4x speedup over baseline)  
   Auto-Optimized             2.34 GOPS    (11.9x speedup over baseline)
   Interleaved-NUMA           1.32 GOPS    (6.7x speedup over baseline)
```

## Technical Benefits

✅ **Meaningful Speedup Measurements**: Now comparing identical operations shows real coordinator benefits  
✅ **Consistent Performance**: Warmup eliminates cold-start contamination  
✅ **Same Parameter Coverage**: Baseline tests all the same batch sizes and tensor sizes as coordinator tests  
✅ **Proper Performance Metrics**: GOPS calculations now use the same operation counts  
✅ **Reliable Comparisons**: Users can trust that speedup numbers reflect real performance gains  

## Validation Results

### Development Testing
```bash
./test-comprehensive-numa-performance --quick --batch-sizes 4,8
# Shows 7-12x real speedup over single-threaded baseline
```

### Custom Parameters
```bash  
./test-comprehensive-numa-performance --matrix-size 128 --batch-sizes 4,8 --baseline-iter 1
# Baseline: 0.196 GOPS, Coordinator: 2.34 GOPS (11.9x speedup)
```

## Use Cases Enhanced

- **Performance Analysis**: Now provides accurate coordinator benefits measurement
- **Development Validation**: Developers can trust speedup numbers during coordinator improvements  
- **Configuration Tuning**: Real performance differences help choose optimal CPU configurations
- **Regression Testing**: Meaningful baselines detect performance regressions accurately

This fix transforms the test suite from **misleading comparisons** to **accurate performance analysis**, providing users with reliable data for NUMA coordinator evaluation and optimization.
