# NUMA Integration Test Extraction and Modularity Improvements

**Date**: August 30, 2025  
**Author**: GitHub Copilot  
**Category**: Testing Infrastructure  

## Summary

Extracted the llama-server integration test logic from the main test orchestrator (`tests/run-numa-tests.sh`) into a standalone script (`tests/run-numa-integration-test.sh`) to improve modularity and enable independent execution of the integration test.

## Changes Made

### 1. New Standalone Integration Test Script

**File**: `tests/run-numa-integration-test.sh`
- **Purpose**: Standalone integration test for NUMA-enabled llama-server
- **Features**:
  - Can run independently or be called by main orchestrator
  - Supports `--verbose` and `--help` command line options
  - Includes comprehensive requirements checking
  - Provides detailed progress reporting and error handling
  - Uses same test logic as original but with better modularity

### 2. Main Test Orchestrator Updates

**File**: `tests/run-numa-tests.sh`
- **Removed**: Large `run_integration_test()` function (180+ lines)
- **Added**: Call to standalone integration test script with appropriate verbosity
- **Improvement**: Cleaner separation of concerns between unit tests and integration tests
- **Backward Compatibility**: Maintains existing command-line interface and behavior

### 3. Documentation Updates

**File**: `tests/README-numa-test-runner.md`
- **Added**: Section describing the standalone integration test
- **Updated**: Usage examples to show both orchestrator and standalone usage
- **Enhanced**: Overview to mention both test scripts

## Technical Details

### Integration Test Capabilities

The standalone integration test provides:
- **Real-world validation**: Tests actual llama-server with NUMA mirror mode
- **Model management**: Automatically downloads test model if needed (Qwen2.5-0.5B-Instruct)
- **End-to-end testing**: Validates complete inference pipeline from HTTP API to response
- **Deterministic testing**: Uses fixed parameters (temperature=0.0, seed=42) for reproducible results
- **Robust error handling**: Proper server cleanup, timeout management, and detailed logging

### Architecture Benefits

1. **Modularity**: Integration test can be run separately for targeted testing
2. **Maintainability**: Easier to modify integration test logic without affecting unit tests
3. **Reusability**: Integration test script can be imported by other test frameworks
4. **Debugging**: Easier to debug integration issues in isolation

### Usage Examples

```bash
# Run all tests (original behavior maintained)
./tests/run-numa-tests.sh

# Run only integration test
./tests/run-numa-integration-test.sh

# Debug integration test with verbose output
./tests/run-numa-integration-test.sh --verbose

# Help for integration test
./tests/run-numa-integration-test.sh --help
```

## Validation

### Testing Performed

1. **Syntax Validation**: Both scripts pass `bash -n` syntax checking
2. **Help Functions**: Both scripts properly display help information
3. **Requirements Checking**: Integration test properly validates dependencies
4. **Server Startup**: Integration test can successfully start llama-server
5. **Backward Compatibility**: Main orchestrator maintains existing interface

### Exit Codes

- **Integration Test**: Returns 0 on success, 1 on failure, 130 on interruption
- **Main Orchestrator**: Unchanged behavior - still calls integration test at end if unit tests pass

## Benefits

1. **Improved Development Workflow**: Developers can run integration tests separately during development
2. **Better CI/CD Integration**: Integration tests can be run in different CI stages
3. **Enhanced Debugging**: Easier to isolate and debug integration test issues
4. **Code Organization**: Cleaner separation between unit and integration testing logic
5. **Flexibility**: Integration test can be easily adapted for different testing scenarios

## Files Modified

- `tests/run-numa-integration-test.sh` (NEW - 350+ lines)
- `tests/run-numa-tests.sh` (MODIFIED - removed 180+ lines, added orchestrator call)
- `tests/README-numa-test-runner.md` (UPDATED - added integration test documentation)

## Backward Compatibility

✅ **Fully maintained** - All existing usage patterns continue to work:
- `./tests/run-numa-tests.sh` still runs all tests including integration
- Command-line options (`--verbose`, `--performance`) unchanged
- Exit codes and output formatting preserved
- Integration test still runs only if unit tests pass

## Future Enhancements

This modular approach enables future improvements:
- Multiple integration test scenarios (different models, configurations)
- Parallel execution of integration tests
- Integration with external test frameworks
- Easier CI pipeline customization
