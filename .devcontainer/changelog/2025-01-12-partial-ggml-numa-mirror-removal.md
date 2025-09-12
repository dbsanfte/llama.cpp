# Partial GGML_NUMA_MIRROR Removal - January 12, 2025

## Objective
Remove the `GGML_NUMA_MIRROR` compile-time flag entirely and make NUMA functionality always available with runtime decisions based on `--numa` options.

## Progress Made ✅

### 1. Core GGML Infrastructure Updated
- **ggml/include/ggml.h**: 
  - Tensor structure now unconditionally uses `__data[GGML_NUMA_MAX_NODES]` array
  - `tensor_data()` and `tensor_set_data()` functions always use NUMA logic with runtime checks
  - Added `ggml_numa_should_use_coordinator()` function declaration

- **ggml/src/ggml.c**:
  - Thread-local variables (`ggml_current_numa_node`) now always declared
  - Weak function implementations for NUMA functions unconditional
  - Tensor initialization uses unconditional NUMA-aware structure

- **ggml/src/ggml-cpu/ggml-cpu.c**:
  - Threading strategy now uses runtime decisions via `ggml_numa_should_use_coordinator()`
  - NUMA coordinator integration always available but runtime-controlled
  - Includes NUMA headers unconditionally on supported platforms

### 2. Build System Simplified
- **ggml/CMakeLists.txt**:
  - `GGML_NUMA` option now defaults to ON
  - `GGML_NUMA_MIRROR` option completely removed
  - Simplified build messages and compile definitions
  - Always links with libnuma when NUMA enabled

## Key Achievement 🎯
**Core GGML infrastructure successfully transformed** from compile-time conditionals to runtime decision-making. The fundamental NUMA coordinator and tensor data structures are now always available but controlled by runtime checks.

## Challenge Encountered 🚧

### llama-mmap.cpp Complexity
Attempted to remove `GGML_NUMA_MIRROR` conditionals from `src/llama-mmap.cpp` but encountered:

1. **Intricate conditional compilation structure**: The file has nested platform conditionals:
   ```cpp
   #ifdef _POSIX_MAPPED_FILES
       // POSIX implementations with GGML_NUMA_MIRROR blocks
   #elif defined(_WIN32)
       // Windows implementations
   #endif
   ```

2. **Multiple alternating code paths**: 
   - `#ifdef GGML_NUMA_MIRROR` and `#ifndef GGML_NUMA_MIRROR` blocks provide completely different implementations
   - NUMA mirror mode uses `numa_alloc_onnode()` and specialized allocation
   - Traditional mode uses standard `mmap()` with different memory management

3. **Platform-specific dependencies**: 
   - Linux-specific NUMA functions intermixed with general POSIX code
   - Windows-specific memory mapping code in separate branches
   - Risk of breaking cross-platform compatibility

## Lessons Learned 📚

1. **Systematic approach needed**: Large files with complex conditional compilation require careful analysis before transformation
2. **Platform boundaries matter**: Must preserve Linux/Windows/POSIX conditional structure while removing feature flags
3. **Runtime vs compile-time**: Need to merge alternate implementations into single codebase with runtime decisions
4. **Test incrementally**: Complex transformations should be done in smaller, testable chunks

## Current Status 📊

### Working Components
- ✅ **Core GGML tensor operations** work with runtime NUMA decisions
- ✅ **CPU backend threading** uses runtime coordinator logic  
- ✅ **Build system** simplified and functional
- ✅ **NUMA coordinator** always available for runtime use

### Deferred Components
- 🔄 **llama-mmap.cpp**: Requires more systematic approach to merge NUMA mirror and traditional mmap paths
- 🔄 **Other source files**: Additional files likely have similar `GGML_NUMA_MIRROR` usage patterns

## Next Steps (Future Work) 🔮

1. **Develop systematic approach** for complex conditional compilation removal:
   - Analyze all conditional blocks first
   - Identify merge vs replace patterns
   - Preserve platform-specific boundaries

2. **Create runtime decision framework** for memory mapping:
   - Single entry point that chooses NUMA mirror vs traditional mmap
   - Preserve both code paths but make runtime-controlled

3. **Continue with remaining files** using lessons learned from this attempt

## Technical Notes 🔧

- **Runtime decision function**: `ggml_numa_should_use_coordinator()` serves as the central decision point
- **Backward compatibility**: Traditional threading behavior preserved when no NUMA options specified
- **Platform support**: NUMA functionality remains Linux x86_64 specific via platform conditionals

This partial progress represents significant infrastructure work that enables future runtime NUMA decisions across the codebase.
