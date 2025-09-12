# NUMA Integration Test Command-Line Option Enhancement

**Date**: August 30, 2025  
**Author**: GitHub Copilot  
**Category**: Testing Infrastructure Enhancement  

## Summary

Enhanced the standalone NUMA integration test script (`tests/run-numa-integration-test.sh`) to accept a configurable `--numa` command-line option, allowing users to specify different NUMA modes or run without NUMA entirely. The main test orchestrator continues to use `--numa mirror` by default to maintain existing behavior.

## Changes Made

### 1. Enhanced Command-Line Argument Parsing

**File**: `tests/run-numa-integration-test.sh`
- **Added**: `--numa <mode>` option support with proper argument validation
- **Supported formats**: `--numa mirror`, `--numa=distribute`, etc.
- **Validation**: Requires argument when `--numa` is specified
- **Error handling**: Clear error messages for missing arguments

### 2. Dynamic Server Configuration

**File**: `tests/run-numa-integration-test.sh`
- **Dynamic command**: llama-server starts with or without NUMA options based on user input
- **Conditional messaging**: Updates output messages to reflect NUMA vs non-NUMA testing
- **Flexible cleanup**: Process cleanup works for any llama-server configuration

### 3. Main Orchestrator Integration

**File**: `tests/run-numa-tests.sh`
- **Explicit NUMA mode**: Always passes `--numa mirror` to integration test
- **Backward compatibility**: Maintains existing behavior for the full test suite
- **Consistent verbosity**: Preserves verbose mode when calling integration test

### 4. Updated Documentation

**File**: `tests/README-numa-test-runner.md`
- **Enhanced usage examples**: Shows various NUMA modes and standalone usage
- **Feature descriptions**: Documents new configurable NUMA testing capabilities
- **Integration clarity**: Explains the difference between orchestrator and standalone usage

## Technical Details

### Command-Line Interface

The integration test now supports these usage patterns:

```bash
# Run without NUMA (baseline testing)
./tests/run-numa-integration-test.sh

# Run with specific NUMA modes
./tests/run-numa-integration-test.sh --numa mirror
./tests/run-numa-integration-test.sh --numa distribute
./tests/run-numa-integration-test.sh --numa isolate

# Combined with verbose output
./tests/run-numa-integration-test.sh --verbose --numa mirror

# Using equals syntax
./tests/run-numa-integration-test.sh --numa=mirror
```

### Dynamic Message Adaptation

The script now provides context-appropriate messages:

```bash
# Without NUMA
🧪 Integration Test with llama-server
🚀 Starting llama-server without NUMA options...
🎯 llama-server is working correctly!

# With NUMA
🧪 NUMA Integration Test with llama-server  
🚀 Starting llama-server with NUMA option: --numa mirror...
🎯 NUMA-enabled llama-server is working correctly!
```

### Argument Parsing Implementation

```bash
while [[ $# -gt 0 ]]; do
    case $1 in
        --numa)
            if [ -z "$2" ]; then
                echo "Error: --numa option requires an argument"
                exit 1
            fi
            NUMA_OPTION="--numa $2"
            shift 2
            ;;
        --numa=*)
            NUMA_OPTION="--numa ${1#*=}"
            shift
            ;;
    esac
done
```

### Server Command Construction

```bash
# Dynamic command based on NUMA_OPTION variable
"$BIN_DIR/llama-server" -m "$model_path" --host 0.0.0.0 $NUMA_OPTION --port $server_port
```

## Benefits

### 1. **Flexible Testing**
- Users can test different NUMA modes independently
- Baseline testing without NUMA options for comparison
- Easier debugging of NUMA-specific issues

### 2. **Development Workflow**
- Developers can quickly test specific NUMA configurations
- No need to modify scripts for different test scenarios
- Faster iteration during NUMA feature development

### 3. **CI/CD Integration**
- Different CI stages can test different NUMA modes
- Baseline and NUMA tests can run in parallel
- Better test coverage across configurations

### 4. **Backward Compatibility**
- Main orchestrator behavior unchanged (`--numa mirror` default)
- Existing scripts and documentation remain valid
- Gradual adoption of new features

## Validation

### Testing Performed

1. **Argument Parsing**: All option formats work correctly
2. **Error Handling**: Proper error messages for invalid usage
3. **NUMA Modes**: Successfully tested mirror, distribute, and isolate modes
4. **Non-NUMA Mode**: Baseline testing works without NUMA options
5. **Orchestrator Integration**: Main test suite still uses `--numa mirror`
6. **Help System**: Updated help text accurately describes new options

### Usage Examples Validated

```bash
# ✅ All these work correctly
./tests/run-numa-integration-test.sh --help
./tests/run-numa-integration-test.sh
./tests/run-numa-integration-test.sh --numa mirror
./tests/run-numa-integration-test.sh --numa=distribute
./tests/run-numa-integration-test.sh --verbose --numa isolate
./tests/run-numa-tests.sh  # Still uses --numa mirror internally
```

### Error Handling Verified

```bash
# ❌ Properly rejected with clear error messages
./tests/run-numa-integration-test.sh --numa          # Missing argument
./tests/run-numa-integration-test.sh --invalid      # Unknown option
```

## Files Modified

- `tests/run-numa-integration-test.sh` (ENHANCED - added configurable NUMA options and environment variable pass-through)
- `tests/run-numa-tests.sh` (MODIFIED - explicit --numa mirror for integration test)
- `tests/README-numa-test-runner.md` (UPDATED - enhanced documentation and examples)
- `.github/copilot-instructions.md` (UPDATED - replaced manual model validation with automated integration test)

## Backward Compatibility

✅ **Fully maintained** - All existing usage patterns continue to work:
- `./tests/run-numa-tests.sh` still runs full suite with NUMA mirror integration test
- Command-line options and behavior preserved
- Output format and exit codes unchanged
- Integration test default behavior (when called by orchestrator) unchanged

## Future Enhancements

This enhancement enables several future improvements:
- **Multi-mode testing**: Scripts that test all NUMA modes sequentially
- **Performance comparison**: Automated benchmarking across NUMA configurations
- **CI optimization**: Parallel testing of different NUMA modes
- **Advanced options**: Additional llama-server options beyond NUMA
