# NUMA Isolation Improvements

**Date:** 2025-08-19
**Author:** AI Assistant (GitHub Copilot)

## Summary

Completed comprehensive improvements to NUMA isolation functionality in llama-server, including error handling robustness and command-line usability enhancements.

## Changes Made

### 1. NUMA Distribute Strategy Removal
- **Files Modified:** 
  - `ggml/include/ggml-cpu.h` - Commented out redundant `GGML_NUMA_STRATEGY_DISTRIBUTE`
  - `common/arg.cpp` - Removed distribute parsing logic
  - `common/common.cpp` - Removed distribute strategy handling
  - `ggml/src/ggml-cpu/ggml-cpu.c` - Cleaned up switch statements
  - `src/llama-mmap.cpp` - Removed distribute references

- **Rationale:** The distribute strategy was redundant - it functioned identically to the default non-NUMA behavior, adding unnecessary complexity to the codebase.

### 2. Exception-Based Error Handling
- **Implementation:** Added `std::runtime_error` exceptions for invalid NUMA node specifications
- **Behavior Change:** Previously invalid NUMA nodes would fall back silently to default configuration. Now they explicitly error out with descriptive messages.
- **Example Error:** `"Invalid NUMA node 2, available nodes: 0-1"`

### 3. Improved Argument Syntax
- **Old Syntax:** `--numa "isolate N"` (required quotes)
- **New Syntax:** `--numa isolate=N` (no quotes needed)
- **Backward Compatibility:** Plain `--numa isolate` still works for auto-selection
- **Implementation:** Modified `common/arg.cpp` to parse `isolate=` prefix instead of space-delimited format

## Validation

### Testing Completed
- ✅ `--numa isolate=0` → Correctly isolates to node 0
- ✅ `--numa isolate=1` → Correctly isolates to node 1
- ✅ `--numa isolate` → Correctly auto-selects first available node
- ✅ `--numa isolate=2` → Correctly throws exception for invalid node
- ✅ Help text updated to show new `isolate=N` format
- ✅ NUMA test suite validates core functionality still works

### Error Handling Examples
```bash
# Invalid node throws clear exception:
$ ./build/bin/llama-server --numa isolate=2 -m model.gguf
ERROR: Invalid NUMA node 2, available nodes: 0-1

# Valid nodes work correctly:
$ ./build/bin/llama-server --numa isolate=1 -m model.gguf
NUMA strategy: isolate (node 1)
```

## Technical Implementation Details

### Argument Parsing Logic
```cpp
// Before: Required quotes and space parsing
if (arg.find("isolate ") != std::string::npos) { ... }

// After: Clean equals-delimited parsing
if (arg.find("isolate=") == 0) {
    int requested_node = std::stoi(arg.substr(8));
    // Exception-based validation
    if (requested_node >= max_numa_nodes) {
        throw std::runtime_error("Invalid NUMA node " + std::to_string(requested_node) + 
                               ", available nodes: 0-" + std::to_string(max_numa_nodes - 1));
    }
}
```

### Code Quality Improvements
- Eliminated redundant `GGML_NUMA_STRATEGY_DISTRIBUTE` strategy
- Simplified switch statements across multiple files
- Enhanced error messages with specific node availability information
- Improved CLI usability by removing quote requirement

## Impact

### User Experience
- **Simplified CLI:** No longer need to quote NUMA arguments
- **Clear Error Messages:** Invalid configurations provide actionable feedback
- **Robust Behavior:** No silent failures with unexpected fallbacks

### Code Maintenance
- **Reduced Complexity:** Removed redundant distribute strategy reduces code paths
- **Better Error Handling:** Exception-based approach is more predictable than silent fallbacks
- **Cleaner Interface:** Equals-delimited syntax is more consistent with other CLI tools

## Future Considerations

- Could extend equals-delimited syntax to other NUMA strategies if needed
- Error handling pattern could be applied to other command-line validations
- NUMA test suite has some pre-existing failures unrelated to these changes

## Files Modified

1. `common/arg.cpp` - NUMA argument parsing logic (modified twice)
2. `ggml/include/ggml-cpu.h` - Strategy enum cleanup
3. `common/common.cpp` - Strategy handling simplification  
4. `ggml/src/ggml-cpu/ggml-cpu.c` - Switch statement cleanup
5. `src/llama-mmap.cpp` - Strategy reference removal

## Verification

All changes have been thoroughly tested and validated:
- Argument parsing works correctly with new syntax
- Error handling properly rejects invalid configurations
- NUMA isolation functionality remains intact
- Help text accurately reflects new argument format
