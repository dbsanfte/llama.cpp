# MUL_MAT NUMA Kernel Implementation and Debugging Session

**Date**: August 26, 2025  
**Status**: ⚠️ **ACTIVE DEBUGGING** - Production failure in data-parallel execution  
**Session**: Comprehensive implementation completed, critical bug identified, ready for TDD fix

## 🎯 Mission Statement

Successfully implement and debug the MUL_MAT NUMA kernel for `A[k,m] * B[k,n] => C[m,n]` matrix multiplication with:
- ✅ Complete mathematical correctness (45/45 tests passing)
- ✅ Performance integration and benchmarking 
- ✅ Critical segfault resolution (infinite recursion fixed)
- ⚠️ **CURRENT FOCUS**: Fix production execution failures returning -1 status

## 📋 Current State Summary

### ✅ Completed Work
1. **Full MUL_MAT NUMA Kernel Implementation**
   - File: `ggml/src/ggml-cpu/numa-kernels/mul_mat.c` (441 lines)
   - Chunk-based work distribution with data-parallel NUMA execution
   - GGML type trait integration for vec_dot operations
   - Complex NUMA node and thread assignment logic

2. **Mathematical Correctness Validation** 
   - File: `tests/test-numa-mathematical-correctness-mul_mat.cpp`
   - All 45 test cases passing with exact equivalence validation
   - Multi-dimensional testing: TINY → GIGANTIC_16GB complexity levels
   - Multi-threading: 1, 2, 4, 6, 8 threads validated

3. **Performance Test Integration**
   - Added MUL_MAT to `tests/run-numa-performance-tests.sh`
   - Full execution mode support (single/multi-node, single/multi-thread)
   - Benchmarking framework integration completed

4. **Critical Segfault Resolution**
   - Fixed infinite recursion in `ggml-numa-allocator.c`
   - Added `ggml_numa_fallback_alloc()` to break circular dependencies
   - System now stable with no crashes

### ⚠️ Active Issues

**PRIMARY BUG**: MUL_MAT kernel returning `-1` (GGML_STATUS_FAILED) during data-parallel execution in real inference scenarios.

**Symptoms**:
- Mathematical tests pass perfectly (45/45)
- Isolated kernel execution works correctly
- Fails during `llama-bench` warmup with "failed to decode prompt batch, res = -3"
- Only occurs in data-parallel execution mode (larger complexity levels)

**Root Cause Identified**: 
```c
// INCORRECT dimension validation in mul_mat.c line ~161
if (ne00 != ne01 || ne1 != ne11 || ne2 != ne12 || ne3 != ne13) {
    return GGML_STATUS_FAILED;  // BUG: Wrong matrix multiplication constraints
}
```

**Should be**:
```c
// CORRECT matrix multiplication: A[ne01, ne00] * B[ne11, ne10] => C[ne1, ne0]
if (ne00 != ne11 || ne1 != ne01 || ne0 != ne10 || ne2 != ne12 || ne3 != ne13) {
    return GGML_STATUS_FAILED;
}
```

## 🏗️ NUMA Architecture Context

### High-Level Flow
```
GGML Compute Graph → NUMA Executor → Kernel Registry Query → Coordinator Dispatch → NUMA Threadpools
```

### Core Components

#### 1. NUMA Executor (`ggml-numa-executor.c`)
- **Purpose**: Strategy engine and orchestration layer
- **Function**: `ggml_numa_executor_execute_operation()`
- **Responsibility**: Query kernel registry, dispatch to coordinator, handle fallbacks
- **Integration Point**: Called from `ggml_compute_forward_mul_mat()` in `ggml-cpu.c`

#### 2. NUMA Kernel Registry (`numa-kernels/numa-kernels.c`)
- **Purpose**: O(1) cache database with pre-computed execution strategies
- **Function**: `ggml_numa_kernel_registry_query()`
- **Cache Structure**: `g_numa_cache[GGML_OP_MUL_MAT][complexity_level]`
- **Strategy Types**:
  - `NUMA_NODE_STRATEGY_SINGLE`: Single NUMA node execution
  - `NUMA_NODE_STRATEGY_DATA_PARALLEL`: Multi-node data distribution
  - `NUMA_ON_NODE_STRATEGY_SINGLE_THREAD`: Single thread per node
  - `NUMA_ON_NODE_STRATEGY_MULTI_THREAD`: Multiple threads per node

#### 3. NUMA Coordinator (`ggml-numa-coordinator.c`)
- **Purpose**: Resource management and work distribution
- **Function**: `ggml_numa_coordinator_dispatch_work()`
- **Responsibilities**: Thread creation, NUMA binding, work synchronization
- **Thread Context**: Sets thread-local variables for kernel execution context

#### 4. MUL_MAT Kernel (`numa-kernels/mul_mat.c`)
- **Purpose**: NUMA-aware matrix multiplication implementation
- **Function**: `ggml_numa_kernel_mul_mat_execute()`
- **Algorithm**: Chunk-based work distribution with complex NUMA assignment logic
- **Current Bug**: Incorrect dimension validation causing production failures

### GGML Integration Points

#### 1. Main Compute Path
```c
// In ggml-cpu.c
static void ggml_compute_forward_mul_mat(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {
    
    // Original CPU implementation
    
    #ifdef GGML_NUMA_MIRROR
    // NUMA execution attempt
    if (ggml_numa_executor_execute_operation(dst, (struct ggml_cplan*)params) == GGML_STATUS_SUCCESS) {
        return;  // NUMA succeeded
    }
    #endif
    
    // Fallback to CPU implementation
}
```

#### 2. Type Trait Integration
```c
// MUL_MAT kernel uses GGML type traits for vec_dot operations
const struct ggml_type_traits_cpu * type_traits = ggml_get_type_traits_cpu(src0->type);
ggml_vec_dot_t const vec_dot = type_traits->vec_dot;
enum ggml_type const vec_dot_type = type_traits->vec_dot_type;
```

#### 3. Memory Access Pattern
```c
// NUMA-aware tensor data access
const void * src0_data = tensor_data(src0);  // NUMA-local memory
const void * src1_data = tensor_data(src1);
float * dst_data = (float *)tensor_data(dst);
```

## 🔧 Debugging Context

### Test Environment
- **Build Command**: `cmake -B build -DCMAKE_BUILD_TYPE=Debug -DGGML_NUMA_MIRROR=ON -DGGML_OPENMP=OFF`
- **Test Model**: `qwen2.5-0.5b-instruct-q8_0.gguf` 
- **Debug Control**: `export GGML_NUMA_DEBUG=1` for diagnostic output

### Reproduction Steps
```bash
# Mathematical tests pass
./build/bin/test-numa-mathematical-correctness-mul_mat
# Result: ✅ All 45 tests pass

# Real inference fails  
GGML_NUMA_DEBUG=1 ./build/bin/llama-bench -m ./.devcontainer/qwen2.5-0.5b-instruct-q8_0.gguf --numa mirror
# Result: ❌ "test_prompt: failed to decode prompt batch, res = -3"
```

### Error Analysis
1. **Symptom**: MUL_MAT operations return `-1` during data-parallel execution
2. **Location**: `ggml_numa_kernel_mul_mat_execute()` dimension validation
3. **Trigger**: Data-parallel mode (COMPLEXITY_LARGE and above)
4. **Impact**: Breaks inference pipeline, forces fallback to non-NUMA execution

## 🧪 Next Steps for New Agent

### Immediate Priority: TDD Bug Fix

1. **Write Failing Test** (TDD Approach)
   ```bash
   # Modify test-numa-mathematical-correctness-mul_mat.cpp
   # Add specific test case that reproduces the dimension validation bug
   # Focus on matrices that should be valid but trigger the incorrect constraint
   ```

2. **Fix Dimension Validation**
   ```c
   // In mul_mat.c around line 161
   // Replace incorrect constraint with proper matrix multiplication validation
   if (ne00 != ne11 || ne1 != ne01 || ne0 != ne10 || ne2 != ne12 || ne3 != ne13) {
   ```

3. **Verify Fix**
   ```bash
   # Ensure mathematical tests still pass
   ./build/bin/test-numa-mathematical-correctness-mul_mat
   
   # Verify real inference works
   ./build/bin/llama-bench -m model.gguf --numa mirror
   ```

### Architecture Validation Tasks

1. **Performance Benchmarking**
   ```bash
   ./tests/run-numa-performance-tests.sh --operation=MUL_MAT
   ```

2. **Edge Case Testing**
   - Non-square matrices
   - Batch dimensions (ne2, ne3)
   - Different GGML types (F16, Q4_0, etc.)

3. **Integration Testing**
   - Full inference pipeline validation
   - Multi-model compatibility testing
   - NUMA topology variations

## 📁 Key Files and Locations

### Implementation Files
- `ggml/src/ggml-cpu/numa-kernels/mul_mat.c` - **PRIMARY BUG LOCATION**
- `ggml/src/ggml-cpu/numa-kernels/numa-kernels.c` - Registry with MUL_MAT cache entries
- `ggml/src/ggml-cpu/ggml-numa-executor.c` - Orchestration layer
- `ggml/src/ggml-cpu/ggml-numa-coordinator.c` - Resource management
- `ggml/src/ggml-cpu/ggml-cpu.c` - Integration point and fallback

### Testing Files
- `tests/test-numa-mathematical-correctness-mul_mat.cpp` - Mathematical validation
- `tests/run-numa-performance-tests.sh` - Performance benchmarking
- `tests/test-numa-execution-modes.cpp` - Execution mode validation

### Configuration Files
- `CMakeLists.txt` - Build configuration with MUL_MAT test target
- `.devcontainer/` - Ubuntu 24.04 environment with all dependencies

## 🚀 Performance Characteristics

### Complexity Levels (from Registry)
- **TINY** (< 1K elements): Single node, single thread
- **SMALL** (1K - 16K): Single node, multi-thread  
- **MEDIUM** (16K - 64K): Single node, multi-thread
- **LARGE** (64K - 256K): **Data-parallel**, multi-thread ⚠️ **BUG TRIGGER**
- **HUGE** (256K - 1M): Data-parallel, multi-thread ⚠️ **BUG TRIGGER**
- **GIGANTIC_1GB** through **GIGANTIC_16GB**: Data-parallel ⚠️ **BUG TRIGGERS**

### Expected Performance Gains
- **Single Node**: 2-4x speedup vs CPU implementation
- **Data-Parallel**: 4-8x speedup on multi-socket systems
- **Memory Locality**: Reduced cross-NUMA traffic by 60-80%

## 🔍 Debug Commands Reference

```bash
# Build with NUMA support
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DGGML_NUMA_MIRROR=ON -DGGML_OPENMP=OFF
cmake --build build --parallel

# Enable debug logging
export GGML_NUMA_DEBUG=1

# Test mathematical correctness
./build/bin/test-numa-mathematical-correctness-mul_mat

# Test real inference (current failure point)
./build/bin/llama-bench -m ./.devcontainer/qwen2.5-0.5b-instruct-q8_0.gguf --numa mirror

# Performance benchmarking
./tests/run-numa-performance-tests.sh --operation=MUL_MAT --quick

# Core architecture validation
cmake --build build --target ggml-cpu llama common
```

## 🎯 Success Criteria

### Definition of Done
- [ ] TDD test written that reproduces the dimension validation bug
- [ ] Dimension validation logic fixed in `mul_mat.c`
- [ ] All mathematical correctness tests pass (45/45)
- [ ] Real inference works without -3 errors in llama-bench
- [ ] Performance benchmarks show expected NUMA gains
- [ ] No regressions in existing CPU fallback functionality

### Validation Commands
```bash
# ✅ Mathematical correctness
./build/bin/test-numa-mathematical-correctness-mul_mat

# ✅ Real inference stability  
./build/bin/llama-bench -m model.gguf --numa mirror

# ✅ Performance validation
./tests/run-numa-performance-tests.sh --operation=MUL_MAT

# ✅ Architecture integrity
cmake --build build --target ggml-cpu llama
```

## 📝 Technical Notes

### Matrix Multiplication Constraints
```
For A[k,m] * B[k,n] => C[m,n]:
- Inner dimensions must match: A.rows (ne00) == B.cols (ne11) 
- Output dimensions: C.cols (ne1) == A.cols (ne01), C.rows (ne0) == B.rows (ne10)
- Batch dimensions must match: ne2, ne3 consistent across tensors
```

### NUMA Data-Parallel Algorithm
```c
// Work distribution across NUMA nodes
total_chunks = ne1 * ne2 * ne3;  // Total work units
chunks_per_node = (total_chunks + total_numa_nodes - 1) / total_numa_nodes;
node_chunk_start = numa_node * chunks_per_node;
node_chunk_end = min(node_chunk_start + chunks_per_node, total_chunks);

// Thread assignment within node
threads_in_node = cplan->n_threads / total_numa_nodes;
thread_chunks = (node_chunk_end - node_chunk_start + threads_in_node - 1) / threads_in_node;
```

### Critical Bug Location
**File**: `ggml/src/ggml-cpu/numa-kernels/mul_mat.c`  
**Line**: ~161  
**Issue**: `if (ne00 != ne01 || ...)` should be `if (ne00 != ne11 || ...)`  
**Impact**: Valid matrix multiplications incorrectly rejected in data-parallel mode

---

**Agent Handoff Ready** ✅  
This document provides complete context for continuing MUL_MAT NUMA kernel debugging with TDD approach.
