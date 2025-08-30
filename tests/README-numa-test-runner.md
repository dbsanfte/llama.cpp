# NUMA Test Suite Runner

## Overview

The `run-numa-tests.sh` script is a comprehensive test runner for all NUMA-related tests in the llama.cpp project. It executes all NUMA tests sequentially and provides detailed reporting.

## Usage

From the project root directory:

```bash
# Make executable (if not already)
chmod +x tests/run-numa-tests.sh

# Run all NUMA tests
./tests/run-numa-tests.sh
```

## Features

### 🎯 Test Coverage
The script runs these NUMA tests in order:
1. **test-numa-coordinator** - Core coordinator functionality
2. **test-numa-coordinator-wait** - Wait-for-completion and execution strategies
3. **test-numa-dispatcher** - Operation dispatch and strategy analysis
4. **test-numa-mathematical-correctness-add** - ADD operation mathematical validation with 3-part testing
5. **test-numa-mathematical-correctness-mul** - MUL operation with broadcasting regression tests
6. **test-numa-mathematical-correctness-cpy** - CPY operation validation
7. **test-numa-mathematical-correctness-mul-mat** - Matrix multiplication with large tensor support

### 🔬 Mathematical Correctness Framework
Each mathematical correctness test follows a comprehensive 3-part structure:
1. **Mathematical Equivalence Testing** - Multi-dimensional tensors across various thread strategies
2. **Quantization Type Coverage** - Q8_0, Q4_0, Q5_0, F16, F32 validation (critical for model reliability)
3. **Regression Testing** - Operation-specific edge cases and previous bug scenarios

**Critical**: Quantization coverage prevents silent model inference failures in production environments.

### 🛡️ Safety Features
- **Timeout protection**: Each test limited to 5 minutes maximum
- **Error handling**: Continues running even if individual tests fail
- **Binary validation**: Checks if test binaries exist and are executable
- **Graceful interruption**: Handles Ctrl+C with proper cleanup

### 📊 Comprehensive Reporting
- **Real-time progress**: Shows current test being executed
- **Timing information**: Reports duration for each test
- **Detailed results**: Color-coded pass/fail status
- **Summary report**: Final statistics and recommendations

### 🏗️ System Integration
- **Automatic detection**: Finds build directory and test binaries
- **NUMA awareness**: Detects system NUMA capabilities
- **Dependency checking**: Verifies required commands are available

## Exit Codes

- **0**: All tests passed successfully
- **1**: One or more tests failed
- **130**: Script interrupted by user (Ctrl+C)

## Example Output

```
🧪 NUMA Test Suite Runner
========================================
Project: llama.cpp NUMA improvements
Build directory: /workspaces/llama.cpp/build
Total tests: 7

🔍 Checking system requirements...
🏗️  NUMA system information:
available: 1 nodes (0)
node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21
node 0 size: 31863 MB

🚀 Starting NUMA test suite...

🎯 Running test 1/7: test-numa-coordinator
=========================================
✅ PASSED (test-numa-coordinator) - Duration: 2.34s

🎯 Running test 2/7: test-numa-coordinator-wait
=========================================
✅ PASSED (test-numa-coordinator-wait) - Duration: 5.67s

🎯 Running test 3/7: test-numa-dispatcher
=========================================
✅ PASSED (test-numa-dispatcher) - Duration: 3.21s

🎯 Running test 4/7: test-numa-mathematical-correctness-add
=========================================
🧪 ADD Mathematical Equivalence: 20/20 multi-dimensional tests passed
🔬 ADD Quantization Coverage: 6/6 quantization types verified
✅ PASSED (test-numa-mathematical-correctness-add) - Duration: 4.12s

🎯 Running test 5/7: test-numa-mathematical-correctness-mul
=========================================
🧪 MUL Mathematical Equivalence: 20/20 multi-dimensional tests passed
🔬 MUL Quantization Coverage: 6/6 quantization types verified  
🔄 MUL Broadcasting Regression: 20/20 broadcasting scenarios passed
✅ PASSED (test-numa-mathematical-correctness-mul) - Duration: 6.78s

🎯 Running test 6/7: test-numa-mathematical-correctness-cpy
=========================================
🧪 CPY Mathematical Equivalence: 20/20 multi-dimensional tests passed
🔬 CPY Quantization Coverage: 8/8 quantization types verified
✅ PASSED (test-numa-mathematical-correctness-cpy) - Duration: 3.45s

🎯 Running test 7/7: test-numa-mathematical-correctness-mul-mat
=========================================
🧪 MUL_MAT Mathematical Equivalence: 16/16 matrix dimension tests passed
🔬 MUL_MAT Quantization Coverage: 12/12 quantization types verified
✅ PASSED (test-numa-mathematical-correctness-mul-mat) - Duration: 8.91s

========================================
📊 NUMA Test Suite Results
========================================
Total tests: 7
Passed: 7
Failed: 0

Detailed Results:
----------------
✅ test-numa-coordinator: PASSED (2.34s)
✅ test-numa-coordinator-wait: PASSED (5.67s)
✅ test-numa-dispatcher: PASSED (3.21s)
✅ test-numa-mathematical-correctness-add: PASSED (4.12s)
✅ test-numa-mathematical-correctness-mul: PASSED (6.78s)
✅ test-numa-mathematical-correctness-cpy: PASSED (3.45s)
✅ test-numa-mathematical-correctness-mul-mat: PASSED (8.91s)

🎉 All NUMA tests passed successfully!
The NUMA coordinator system and all mathematical operations are working correctly.
Mathematical equivalence verified across all quantization types.
```

## Integration with CI/CD

The script is designed for integration with continuous integration systems:

```bash
# In CI pipeline
if ./tests/run-numa-tests.sh; then
    echo "✅ NUMA tests passed - safe to merge"
else
    echo "❌ NUMA tests failed - review required"
    exit 1
fi
```

## Troubleshooting

### Missing Binaries
```
❌ Error: Test binary not found: /path/to/build/bin/test-numa-coordinator
Please ensure the test is built with CMake.
```
**Solution**: Run `cmake --build build` to build all tests.

### Timeout Issues
```
❌ TIMEOUT (test-numa-coordinator) - Exceeded 5 minutes
```
**Solution**: Check for infinite loops or deadlocks in the failing test.

### Permission Issues
```
❌ Error: Test binary not executable: /path/to/binary
```
**Solution**: The script will attempt to fix permissions automatically, or run `chmod +x build/bin/test-*`.

## Customization

### Adding New Tests
To include new NUMA tests in the runner:

1. Add the test binary name to the `NUMA_TESTS` array in the script:
```bash
NUMA_TESTS=(
    "test-numa-coordinator"
    "test-numa-coordinator-wait"
    "test-numa-dispatcher"
    "test-numa-mathematical-correctness-add"
    "test-numa-mathematical-correctness-mul"
    "test-numa-mathematical-correctness-cpy"
    "test-numa-mathematical-correctness-mul-mat"
    "test-numa-mathematical-correctness-your-operation"  # Add here
)
```

2. Ensure the test is built by CMake in the `build/bin/` directory

3. For mathematical correctness tests, follow the 3-part structure:
   - Mathematical equivalence testing (multi-dimensional, multi-threading)
   - Comprehensive quantization type coverage (Q8_0, Q4_0, Q5_0, F16, F32)
   - Regression testing (operation-specific edge cases)

4. Use the template: `cp tests/test-numa-mathematical-correctness-template.cpp tests/test-numa-mathematical-correctness-OPERATION.cpp`

### Adjusting Timeout
To change the 5-minute timeout per test:
```bash
# Find this line in the script:
timeout 300 "$test_binary" || exit_code=$?

# Change 300 to desired seconds (e.g., 600 for 10 minutes):
timeout 600 "$test_binary" || exit_code=$?
```

## Dependencies

Required system commands:
- `timeout` - For test timeouts
- `bc` - For duration calculations (optional, will show "N/A" if missing)

Optional commands for enhanced output:
- `numactl` - For NUMA system information
- `lscpu` - For CPU topology information

The script will warn about missing optional dependencies but continue execution.
