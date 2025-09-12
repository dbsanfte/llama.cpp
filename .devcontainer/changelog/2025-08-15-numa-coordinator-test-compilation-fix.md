# NUMA Coordinator Test Compilation Fix - 2025-08-15

## Issue

The `test-numa-coordinator` was failing to compile with errors indicating that NUMA functions were not declared:

```
error: 'ggml_numa_graph_compute_with_virtual' was not declared in this scope
error: 'ggml_numa_graph_compute' was not declared in this scope
```

## Root Cause

The include directories in `tests/CMakeLists.txt` were in the wrong order for the `test-numa-coordinator` target. The test needs to prioritize the implementation headers in `ggml/src/ggml-cpu/` over the public headers in `ggml/include/` to access the full NUMA function declarations.

## Solution

Updated the include directory order in `tests/CMakeLists.txt` for the `test-numa-coordinator` target:

**Before:**
```cmake
target_include_directories(${LLAMA_TEST_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/ggml/include ${CMAKE_SOURCE_DIR}/ggml/src/ggml-cpu)
```

**After:**
```cmake
# Prioritize ggml/src/ggml-cpu/ to get the full implementation headers
target_include_directories(${LLAMA_TEST_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src/ggml-cpu ${CMAKE_SOURCE_DIR}/ggml/include)
```

This change matches the pattern already used by `test-numa-dispatcher` which was compiling successfully.

## Files Modified

- `/workspaces/llama.cpp/tests/CMakeLists.txt` - Fixed include directory order
- `/workspaces/llama.cpp/tests/test-numa-coordinator.cpp` - Added missing include for `ggml-numa-coordinator.h`

## Verification

✅ **Compilation**: The test now compiles successfully without undeclared function errors
✅ **Linking**: All NUMA function symbols are properly resolved during linking
🔄 **Runtime**: Test starts executing but encounters segmentation fault during NUMA operation (separate issue)

## Status

**Compilation Issue**: ✅ **RESOLVED**  
**Next Step**: Investigate runtime segmentation fault in NUMA coordinator execution

## Technical Notes

The issue occurred because:
1. The public headers in `ggml/include/` contain basic declarations
2. The implementation headers in `ggml/src/ggml-cpu/` contain the full NUMA function prototypes
3. Include order matters - implementation headers must be prioritized to override public headers
4. The `test-numa-dispatcher` already used the correct order, serving as a working reference

This fix ensures that tests can access the complete NUMA API for comprehensive testing of the NUMA coordination system.
