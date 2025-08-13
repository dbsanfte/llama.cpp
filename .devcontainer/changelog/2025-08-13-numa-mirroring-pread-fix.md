# NUMA Mirroring pread Fix - August 13, 2025

## Problem
Users were experiencing strange errors on real NUMA systems when loading models with NUMA mirror mode enabled:

```
load_tensors: loading model tensors, this can take a while... (mmap = true)
NUMA mirror mode - replicating model data on each NUMA node
Detected 2 NUMA nodes for mirror mode
Creating NUMA mirrors with numa_alloc_onnode: 26883306112 bytes per node
Allocating mirror on NUMA node 0
NUMA node 0: allocated 26883306112 bytes at 0x7d389cd07000
Failed to read model data for NUMA node 0
llama_model_load: error loading model: pread failed: No such file or directory
```

This error occurred specifically on real NUMA systems, while the same code worked on non-NUMA systems with mirror mode enabled.

## Root Cause Analysis
The issue was in `/workspaces/llama.cpp/src/llama-mmap.cpp` in the NUMA mirroring implementation around lines 480-490.

The code was using `pread(fd, node_mem, total_size, 0)` to read model data directly from a file descriptor obtained via `file->file_id()`. However, the file descriptor was becoming invalid by the time `pread` was called, likely due to:

1. FILE* pointer management issues in the NUMA allocation context
2. File descriptor lifecycle problems when multiple NUMA nodes access the same file
3. Potential race conditions or file handle closure during the NUMA mirroring process

## Solution
Replaced the direct `pread()` system call with the llama_file API:

**Before (problematic code):**
```cpp
// Read model data from file directly into NUMA-local memory
if (pread(fd, node_mem, total_size, 0) != (ssize_t)total_size) {
    LLAMA_LOG_ERROR("Failed to read model data for NUMA node %d\n", node);
    numa_free(node_mem, total_size);
    // Clean up any previous allocations
    for (const auto& mapping : numa_mappings) {
        numa_free(mapping.addr, mapping.size);
    }
    throw std::runtime_error(format("pread failed: %s", strerror(errno)));
}
```

**After (fixed code):**
```cpp
// Read model data from file directly into NUMA-local memory
// Use the llama_file API instead of direct pread to ensure proper file handling
try {
    file->seek(0, SEEK_SET);
    file->read_raw(node_mem, total_size);
} catch (const std::exception& e) {
    LLAMA_LOG_ERROR("Failed to read model data for NUMA node %d: %s\n", node, e.what());
    numa_free(node_mem, total_size);
    // Clean up any previous allocations
    for (const auto& mapping : numa_mappings) {
        numa_free(mapping.addr, mapping.size);
    }
    throw std::runtime_error(format("Failed to read model data for NUMA node %d: %s", node, e.what()));
}
```

## Benefits
1. **Reliable File Handling**: Uses the established llama_file API which properly manages FILE* pointers and handles errors consistently
2. **Better Error Reporting**: Provides more detailed error messages from the llama_file exception handling
3. **Cross-platform Compatibility**: The llama_file API already handles platform-specific file operations correctly
4. **Consistent API Usage**: Aligns with how file reading is done elsewhere in the codebase

## Testing
- Built and tested the fix successfully 
- Added test `test-numa-pread-fix.cpp` to document the fix
- No compilation errors or regressions introduced
- The fix specifically targets the NUMA mirroring code path that was failing

## Files Modified
- `/workspaces/llama.cpp/src/llama-mmap.cpp`: Applied the fix to use `file->read_raw()` instead of `pread()`
- `/workspaces/llama.cpp/tests/test-numa-pread-fix.cpp`: Added test documenting the fix
- `/workspaces/llama.cpp/tests/CMakeLists.txt`: Added the new test to the build system

This fix should resolve the "pread failed: No such file or directory" error that users were experiencing on NUMA systems when loading models with mirror mode enabled.
