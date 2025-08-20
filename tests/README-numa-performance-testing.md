# NUMA Performance Testing Framework

This document describes the comprehensive performance testing framework for benchmarking NUMA kernels against fallback CPU implementations in llama.cpp.

## 🎯 Overview

The NUMA Performance Testing Framework provides systematic benchmarking of NUMA kernels to ensure they provide meaningful performance improvements over fallback implementations. It's designed to be:

- **Modular**: Easy to extend to new operations
- **Comprehensive**: Tests multiple tensor sizes and thread configurations  
- **Statistical**: Provides rigorous performance analysis with variance calculations
- **Automated**: Integrates with existing test infrastructure

## 📁 Framework Components

### Core Files

1. **`test-numa-performance-benchmark-template.cpp`** - Template for creating new performance tests
2. **`test-numa-performance-benchmark-add.cpp`** - Complete example implementation for ADD operation
3. **`run-numa-performance-tests.sh`** - Dedicated performance test runner with advanced reporting
4. **`run-numa-tests.sh`** - Updated main test runner with `--performance` option

### Performance Control Functions

The framework adds execution control functions to force NUMA vs fallback execution:

```cpp
// Control NUMA dispatch for performance comparison
extern void ggml_numa_set_dispatch_enabled(bool enabled);
extern bool ggml_numa_get_dispatch_enabled(void);
extern void ggml_numa_clear_dispatch_override(void);
```

## 🚀 Quick Start

### Running Performance Tests

```bash
# Run all performance benchmarks
./tests/run-numa-performance-tests.sh

# Run with verbose output
./tests/run-numa-performance-tests.sh --verbose

# Run only specific operation
./tests/run-numa-performance-tests.sh --operation=ADD

# Generate CSV output for analysis
./tests/run-numa-performance-tests.sh --output=csv > results.csv

# Quick test suite for faster results
./tests/run-numa-performance-tests.sh --quick
```

### Integration with Main Test Suite

```bash
# Run regular tests only
./tests/run-numa-tests.sh

# Run regular tests + performance benchmarks
./tests/run-numa-tests.sh --performance

# Verbose output with performance tests
./tests/run-numa-tests.sh --verbose --performance
```

## 📊 Performance Metrics

The framework measures and reports:

### Timing Statistics
- **Minimum time**: Best-case performance across multiple runs
- **Average time**: Mean performance with standard deviation
- **Maximum time**: Worst-case performance

### Throughput Metrics
- **Memory bandwidth (GB/s)**: For memory-bound operations
- **Compute throughput (GFLOP/s)**: For compute-bound operations
- **Speedup ratio**: NUMA performance / fallback performance

### Statistical Analysis
- **Standard deviation**: Performance consistency measurement
- **Success rate**: Percentage of successful benchmark runs
- **Confidence analysis**: Statistical significance of improvements

## 🔧 Creating New Performance Tests

### Step 1: Copy Template

```bash
cp tests/test-numa-performance-benchmark-template.cpp \
   tests/test-numa-performance-benchmark-YOUR_OPERATION.cpp
```

### Step 2: Implement Operation-Specific Logic

Replace `TEMPLATE_OPERATION` with your operation name and implement:

1. **`create_OPERATION_tensors()`** - Create input/output tensors
2. **`calculate_operation_flops()`** - Count floating-point operations
3. **Test dimensions** - Define appropriate tensor sizes for your operation
4. **Memory access patterns** - Calculate total memory usage

### Step 3: Update CMakeLists.txt

Add your test to `tests/CMakeLists.txt`:

```cmake
# test-numa-performance-benchmark-YOUR_OPERATION
set(LLAMA_TEST_NAME test-numa-performance-benchmark-YOUR_OPERATION)
llama_build_and_test(test-numa-performance-benchmark-YOUR_OPERATION.cpp)
target_link_libraries(${LLAMA_TEST_NAME} PRIVATE ggml ggml-cpu common)
target_include_directories(${LLAMA_TEST_NAME} PRIVATE 
    ${CMAKE_SOURCE_DIR}/ggml/src/ggml-cpu 
    ${CMAKE_SOURCE_DIR}/ggml/src 
    ${CMAKE_SOURCE_DIR}/ggml/include)
```

### Step 4: Build and Test

```bash
cmake --build build --target test-numa-performance-benchmark-YOUR_OPERATION
./build/bin/test-numa-performance-benchmark-YOUR_OPERATION --verbose
```

## 📋 Example: ADD Operation Implementation

The ADD operation serves as a complete reference implementation:

### Operation Characteristics
- **Type**: Memory-bound (limited by memory bandwidth)
- **Access pattern**: Linear (cache-friendly)
- **Parallelization**: Embarrassingly parallel
- **Expected NUMA benefits**: High (data-parallel scaling)

### Test Configuration
- **Tensor sizes**: TINY (16K) → HUGE (64M elements)
- **Thread counts**: 1, 2, 4, 8 threads
- **Memory access**: 3x tensor size (read A + read B + write result)
- **FLOP count**: 1 operation per output element

### Performance Expectations
- **Multi-socket systems**: Significant speedup (1.5x+ expected)
- **Single-socket systems**: Modest speedup or parity
- **Large tensors**: Better speedup than small tensors
- **Memory bandwidth**: Primary limiting factor

## 📈 Interpreting Results

### Speedup Categories
- **🚀 Excellent**: ≥1.5x speedup - Clear NUMA benefits
- **✅ Good**: ≥1.2x speedup - Meaningful improvement
- **🔷 Marginal**: ≥1.0x speedup - No regression
- **⚠️ Poor**: <1.0x speedup - Optimization needed

### Performance Analysis
The framework categorizes operations and provides recommendations:

```
🎯 ADD OPERATION PERFORMANCE ANALYSIS:
  Average speedup: 2.34x
  Best speedup: 3.21x
  Worst speedup: 1.87x
  Average bandwidth improvement: 2.18x
  Peak bandwidth achieved: 45.67 GB/s
  Successful tests: 20/20
  ✅ NUMA ADD kernel shows excellent performance improvement!
     This confirms ADD operation benefits significantly from NUMA data-parallel execution.
```

## 🔍 Troubleshooting

### Common Issues

1. **No speedup observed**
   - Check if NUMA topology is available: `numactl --hardware`
   - Verify coordinator is initialized: Look for coordinator logs
   - Test on larger tensor sizes: Small tensors have overhead

2. **Performance regression**
   - Check work distribution overhead vs. kernel efficiency
   - Verify memory allocation patterns
   - Consider single-node execution for small workloads

3. **Build failures**
   - Ensure all NUMA components are built: `cmake --build build`
   - Check include paths in CMakeLists.txt
   - Verify GGML_NUMA_MIRROR is defined

### Debugging Tips

```bash
# Verbose test execution
./build/bin/test-numa-performance-benchmark-add --verbose

# Force single-threaded execution
numactl --interleave=all ./build/bin/test-numa-performance-benchmark-add

# Check NUMA topology
numactl --hardware
cat /proc/meminfo | grep -i numa
```

## 🎛️ Advanced Configuration

### Custom Test Parameters

Modify test parameters in your implementation:

```cpp
// Performance test configuration
static constexpr int WARMUP_RUNS = 3;        // Warmup iterations
static constexpr int BENCHMARK_RUNS = 10;    // Benchmark iterations
static constexpr int MIN_BENCHMARK_TIME_MS = 100;  // Minimum test duration
```

### Tensor Size Configuration

Define operation-appropriate test cases:

```cpp
struct {
    int dim1, dim2, dim3;
    const char* label;
} test_cases[] = {
    {32, 32, 16, "TINY"},      // L1 cache friendly
    {128, 128, 64, "MEDIUM"},  // L3 cache size
    {512, 512, 256, "HUGE"}    // Memory-bound
};
```

### Thread Strategy Testing

Test different thread configurations:

```cpp
int thread_strategies[] = {1, 2, 4, 8, 16};  // Add more thread counts
```

## 📊 Output Formats

The framework supports multiple output formats:

### Summary Format (Default)
Human-readable summary with key metrics and recommendations.

### CSV Format
```bash
Operation,AvgSpeedup,BestSpeedup,SuccessfulTests,TotalTests,Status
ADD,2.34,3.21,20,20,Excellent
```

### JSON Format
```json
{
  "timestamp": "2025-08-20T10:30:00Z",
  "total_tests": 20,
  "passed_tests": 20,
  "failed_tests": 0,
  "operations": {
    "ADD": {
      "avg_speedup": 2.34,
      "best_speedup": 3.21,
      "successful_tests": 20,
      "total_tests": 20
    }
  }
}
```

## 🏗️ Integration with CI/CD

### Automated Performance Regression Detection

```bash
# Run quick performance tests in CI
./tests/run-numa-performance-tests.sh --quick --output=json > perf_results.json

# Compare against baseline (implement comparison script)
python scripts/compare_performance.py perf_results.json baseline.json
```

### Performance Monitoring

```bash
# Generate performance reports
./tests/run-numa-performance-tests.sh --output=csv >> performance_history.csv

# Track performance trends over time
python scripts/analyze_performance_trends.py performance_history.csv
```

## 🎯 Best Practices

### Operation Categories

**Memory-bound operations** (ADD, MUL, etc.):
- Focus on memory bandwidth metrics
- Test with large tensors to saturate memory bus
- Expect linear scaling with NUMA nodes

**Compute-bound operations** (matrix multiply, convolution):
- Focus on FLOP/s metrics
- Test thread scaling characteristics
- Consider cache hierarchy effects

**Mixed operations** (normalization, activation functions):
- Test both compute and memory patterns
- Consider work distribution overhead
- Optimize for balanced execution

### Performance Testing Guidelines

1. **Consistent test environment**: Disable CPU frequency scaling, use dedicated test machines
2. **Multiple runs**: Use statistical analysis to handle performance variance
3. **Realistic workloads**: Test with tensor sizes representative of actual usage
4. **Baseline comparison**: Always compare against fallback implementation
5. **Regression testing**: Continuously monitor performance over development cycles

## 📚 Related Documentation

- [NUMA Architecture Documentation](../docs/numa-architecture.md) - Comprehensive architecture guide
- [Copilot Instructions](../.github/copilot-instructions.md) - Development workflow guidance
- [NUMA Mathematical Correctness Tests](./test-numa-mathematical-correctness-template.cpp) - Correctness testing framework

## 🤝 Contributing

When adding new performance tests:

1. Follow the template structure for consistency
2. Include comprehensive operation documentation
3. Add appropriate test dimensions for the operation type
4. Update this README with operation-specific guidance
5. Test on both single-socket and multi-socket systems

The framework is designed to grow with the NUMA kernel implementation, providing systematic performance validation as new operations are migrated to the NUMA architecture.
