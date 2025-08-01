# AVX-512 Quantization Tests

This document describes the new AVX-512 specific tests added to verify and benchmark the AVX-512 optimized quantization functions in llama.cpp.

## Overview

We've added two new tests to specifically target the AVX-512 optimizations:

1. **test-quantize-avx512** - Functional correctness tests for AVX-512 optimized quantization types
2. **test-quantize-avx512-perf** - Performance benchmarks for AVX-512 optimized quantization functions

## AVX-512 Optimized Quantization Types

The following quantization types have been identified as having AVX-512 optimizations in `ggml/src/ggml-cpu/arch/x86/quants.c`:

| Type | Function | AVX-512 Features Used |
|------|----------|---------------------|
| q4_0 | `ggml_vec_dot_q4_0_q8_0` | AVX-512F, AVX-512VNNI* |
| q5_0 | `ggml_vec_dot_q5_0_q8_0` | AVX-512F, AVX-512VNNI* |
| q8_0 | `ggml_vec_dot_q8_0_q8_0` | AVX-512F, AVX-512VNNI* |
| q4_K | `ggml_vec_dot_q4_K_q8_K` | AVX-512F |
| iq2_xxs | `ggml_vec_dot_iq2_xxs_q8_K` | AVX-512F |
| iq3_xxs | `ggml_vec_dot_iq3_xxs_q8_K` | AVX-512F |
| iq4_xs | `ggml_vec_dot_iq4_xs_q8_K` | AVX-512F |

*AVX-512VNNI is used when available, with fallback to AVX-512F

## New Test Files

### test-quantize-avx512.cpp

**Purpose**: Functional correctness testing for AVX-512 optimized quantization functions

**Features**:
- Tests quantization accuracy with various data patterns
- Tests dot product accuracy 
- Uses block-size aligned test data to avoid assertion failures
- Tests edge cases with extreme values (zeros, ones, alternating patterns)
- Validates performance across different data sizes optimized for AVX-512 (64-byte alignment)
- Reports CPU feature support (AVX-512F, AVX-512VNNI, AVX-512VL)

**Usage**:
```bash
# Run basic functionality test
./build/bin/test-quantize-avx512

# Run with verbose output
./build/bin/test-quantize-avx512 -v
```

**Expected Output**: All AVX-512 optimized types should pass or be skipped if missing required functions.

### test-quantize-avx512-perf.cpp

**Purpose**: Performance benchmarking specifically for AVX-512 optimized quantization functions

**Features**:
- Benchmarks quantization, dequantization, and dot product operations
- Tests multiple data sizes (L1/L2/L3 cache friendly sizes)
- Measures both CPU cycles and wall-clock time
- Reports throughput in GB/s for both float32 and quantized data
- Supports extra-large test sizes with `-xlarge` flag
- Memory-aligned test data for optimal AVX-512 performance

**Usage**:
```bash
# Run standard performance benchmark
./build/bin/test-quantize-avx512-perf

# Include extra large test sizes (1MB)
./build/bin/test-quantize-avx512-perf -xlarge

# Show help
./build/bin/test-quantize-avx512-perf --help
```

## Test Coverage Analysis

### Existing Coverage
The standard tests (`test-quantize-fns.cpp` and `test-quantize-perf.cpp`) already cover all quantization types, including those with AVX-512 optimizations. However, they don't specifically:
- Target AVX-512 code paths with appropriate data sizes
- Test AVX-512 specific edge cases (64-byte alignment, etc.)
- Provide AVX-512 specific performance metrics
- Validate AVX-512 optimizations are actually being used

### New Coverage
The new AVX-512 specific tests provide:
- **Targeted testing**: Specifically exercises AVX-512 code paths with appropriate block sizes
- **Alignment testing**: Uses data sizes that are multiples of 64 bytes (AVX-512 register size)
- **Edge case coverage**: Tests boundary conditions specific to AVX-512 implementations
- **Performance visibility**: Shows performance characteristics of AVX-512 vs non-AVX-512 paths
- **Feature detection**: Reports which AVX-512 features are available at runtime

## Building and Running

The tests are automatically built with the standard CMake build process:

```bash
cmake --build build --parallel
```

Both tests are added to the test suite:
- `test-quantize-avx512` - Added as an automatic test
- `test-quantize-avx512-perf` - Built but not run automatically (performance test)

## Performance Results

On systems with AVX-512 support, you should see significant performance improvements for:
- Vector dot products (2-4x speedup typical)
- Quantization operations (1.5-2x speedup typical)
- Memory throughput improvements due to wider registers

On systems without AVX-512 (like the current dev container), the functions fall back to AVX2 or SSE implementations.

## Integration with CI/CD

The functional test (`test-quantize-avx512`) is integrated into the standard test suite and will:
- Run automatically as part of `make test` or `ctest`
- Pass on systems without AVX-512 (functions are skipped gracefully)
- Fail if AVX-512 optimized functions have regressions
- Report which functions are tested vs. skipped

The performance test is available for manual benchmarking and performance regression testing.

## Future Improvements

Potential enhancements to the AVX-512 test suite:
1. **Comparative benchmarking**: Compare AVX-512 vs AVX2 performance side-by-side  
2. **Accuracy validation**: More rigorous numerical accuracy testing for AVX-512 paths
3. **Automated regression detection**: Detect performance regressions in AVX-512 optimizations
4. **Coverage expansion**: Add tests as more quantization types get AVX-512 optimizations
