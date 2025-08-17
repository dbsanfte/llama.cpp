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
4. **test-numa-mathematical-correctness** - Mathematical validation framework

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
Total tests: 4

🔍 Checking system requirements...
🏗️  NUMA system information:
available: 1 nodes (0)
node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21
node 0 size: 31863 MB

🚀 Starting NUMA test suite...

🎯 Running test 1/4: test-numa-coordinator
=========================================
✅ PASSED (test-numa-coordinator) - Duration: 2.34s

🎯 Running test 2/4: test-numa-coordinator-wait
=========================================
✅ PASSED (test-numa-coordinator-wait) - Duration: 5.67s

🎯 Running test 3/4: test-numa-dispatcher
=========================================
✅ PASSED (test-numa-dispatcher) - Duration: 3.21s

🎯 Running test 4/4: test-numa-mathematical-correctness
=========================================
✅ PASSED (test-numa-mathematical-correctness) - Duration: 1.89s

========================================
📊 NUMA Test Suite Results
========================================
Total tests: 4
Passed: 4
Failed: 0

Detailed Results:
----------------
✅ test-numa-coordinator: PASSED (2.34s)
✅ test-numa-coordinator-wait: PASSED (5.67s)
✅ test-numa-dispatcher: PASSED (3.21s)
✅ test-numa-mathematical-correctness: PASSED (1.89s)

🎉 All NUMA tests passed successfully!
The NUMA coordinator system is working correctly.
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
    "test-numa-mathematical-correctness"
    "test-my-new-numa-feature"  # Add here
)
```

2. Ensure the test is built by CMake in the `build/bin/` directory

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
