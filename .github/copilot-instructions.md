# Copilot Instructions for llama.cpp

This document provides instructions for AI assistants working on the llama.cpp project with NUMA improvements.

## 🎯 Project Overview

This is a fork of llama.cpp with **NUMA-aware improvements** for better CPU threading and memory allocation:

- **NUMA Coordinator** - `ggml/src/ggml-cpu/ggml-numa-coordinator.c` - Thread/node assignment
- **Operation Dispatcher** - `ggml/src/ggml-cpu/ggml-numa-operation-dispatch.c` - Routes operations to coordinator
- **Fallback Operations** - `ggml/src/ggml-cpu/ggml-numa-fallback.c` - Non-NUMA operations
- **Dev Container** - Ubuntu 24.04 with pre-installed dependencies

**Goal**: Implement NUMA-aware dispatch/execution for all 85+ operations in `ggml/src/ggml-cpu/ops.h` to improve multi-socket performance. Main flow: `src/llama-context.cpp` → `ggml-cpu.c` → Dispatcher → Coordinator.

## 🔧 NUMA Operation Implementation Workflow

### Step 1: Analysis & Discovery
Find the operation and locate mathematical kernels:
```bash
grep -r "GGML_OP_YOUR_OPERATION" ggml/src/ggml-cpu/
```

**Kernel locations:**
- `ggml/src/ggml-cpu/ggml-cpu.c` - Main implementations
- `ggml/src/ggml-cpu/ops.h` - Operation interfaces
- `ggml/src/ggml-cpu/ggml-cpu-*` - Specialized kernels

**Target functions:** `ggml_compute_forward_your_operation()`, `ggml_compute_forward_your_operation_f32()`, etc.

**⚠️ Critical:** Extract pure mathematical kernels - avoid ggml threading logic that conflicts with NUMA coordinator!

**Operation suitability:**
- ✅ **Excellent**: Element-wise ops (GLU, ADD, MUL) - independent computations, linear memory access
- ⚠️ **Complex**: Matrix ops (MUL_MAT), reductions (SOFT_MAX) - need specialized splitting
- ❌ **Poor**: Global synchronization, complex dependencies

**🚀 SIMD Optimization Requirements:**
- **Always use SIMD**: Replace scalar operations with `ggml_vec_*` functions from `ggml/src/ggml-cpu/vec.h`
- **Common SIMD functions**: `ggml_vec_add_f32()`, `ggml_vec_dot_f32()`, `ggml_vec_scale_f32()`, `ggml_vec_cpy_f32()`
- **Performance impact**: SIMD provides significant speedup on modern CPUs with AVX2/AVX512 support
- **Mathematical equivalence**: SIMD operations must produce identical results to scalar reference

### Step 2: Implementation
### Step 2: Implementation

**Critical NUMA Data Slicing Pattern:**
```c
// For data-parallel operations, ALWAYS slice data across NUMA nodes
int numa_node = context->numa_node;
int max_numa_nodes = context->max_numa_nodes;

// Calculate this NUMA node's data slice
size_t total_elements = tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
size_t elements_per_node = total_elements / max_numa_nodes;
size_t numa_start = numa_node * elements_per_node;
size_t numa_end = (numa_node == max_numa_nodes - 1) ? total_elements : numa_start + elements_per_node;
```

**SIMD-Optimized Work Function Template:**
```c
static int ggml_numa_work_function_your_operation_chunk(void* context) {
    // 1. Extract context and calculate NUMA data slice
    const ggml_numa_work_context_t * ctx = context;
    
    // 2. Slice tensors for NUMA data parallelism (see pattern above)
    
    // 3. Use SIMD operations from vec.h instead of scalar loops:
    //    - Replace: for(i=0; i<n; i++) dst[i] = src0[i] + src1[i];
    //    - With: ggml_vec_add_f32(n, dst, src0, src1);
    
    // 4. Multi-thread within NUMA node using thread_start/thread_end
    
    return 0; // Success
}
```

**Execution Strategy Selection:**
- **Single-node execution**: Small tensors, poor cache locality, or sequential dependencies
- **Data-parallel execution**: Large tensors with good NUMA splitting characteristics
- **Always implement both** strategies like MUL_MAT pattern: `_single` and `_chunk` functions

**Dispatcher handler:**
```c
case GGML_OP_YOUR_OPERATION: {
    efficiency = 0.95f;  // High for element-wise, lower for complex
    strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL;  // Or SINGLE for small tensors
    work_function = ggml_numa_work_function_your_operation_chunk;
    break;
}
```

**⚠️ Critical Patterns for Proper NUMA Integration:**

**1. Execution Strategy Decision Logic:**
```c
// Use single-node for small tensors or poor splitting characteristics
if (total_elements < 32768 || tensor->ne[0] < 512) {
    strategy = NUMA_NODE_STRATEGY_SINGLE;
    work_function = ggml_numa_work_function_your_operation_single;
} else {
    strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL;
    work_function = ggml_numa_work_function_your_operation_chunk;
}
```

**2. SIMD Integration Examples:**
```c
// Element-wise addition (ADD pattern)
ggml_vec_add_f32(chunk_size, dst_chunk, src0_chunk, src1_chunk);

// Dot product for reductions (RMS_NORM pattern)  
float sum_squares = ggml_vec_dot_f32(row_size, row_data, row_data);

// Scaling operations (RMS_NORM pattern)
ggml_vec_scale_f32(row_size, dst_row, inv_rms);
```

**Required strategies for full parallelism:**
- `NUMA_NODE_STRATEGY_DATA_PARALLEL` - Distribute across NUMA nodes
- `NUMA_ON_NODE_STRATEGY_MULTI_THREAD` - Multi-thread within each node

### Step 3: Testing
Use the mathematical correctness template:
```bash
cp tests/test-numa-mathematical-correctness-template.cpp tests/test-numa-mathematical-correctness-YOUR_OPERATION.cpp
```

**Required tests:**
- Multi-dimensional: TINY → LARGE tensor sizes
- Multi-threading: 1, 2, 4, 6, 8 threads
- Mathematical equivalence: Exact comparison with reference
- Add to CMake and `tests/run-numa-tests.sh`
## 🏗️ Build Environment & Commands

**Always use the dev container** for consistency (Ubuntu 24.04 with all dependencies).

### Build Commands
```bash
# Configure debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DGGML_NUMA_MIRROR=ON -DGGML_OPENMP=OFF

# Build (let cmake auto-detect thread count)
cmake --build build --parallel
```

### Quick Model Validation
```bash
# Download test model
wget -c -O ./.devcontainer/qwen2.5-0.5b-instruct-q8_0.gguf https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf

# Test with NUMA mirror mode
./build/bin/llama-server -m ./.devcontainer/qwen2.5-0.5b-instruct-q8_0.gguf --host 0.0.0.0 --numa mirror --port 8080 &

curl -X POST http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "qwen2.5-0.5b-instruct", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 20}'
```

## 🧠 Key Technical Areas

### NUMA Data Slicing & SIMD Optimization Patterns

**Successful SIMD Implementations:**
- **ADD Operation**: Element-wise addition using `ggml_vec_add_f32()` with single-node execution strategy
- **RMS_NORM Operation**: Root mean square normalization using `ggml_vec_dot_f32()` + `ggml_vec_scale_f32()` with data-parallel execution

**Pattern: Element-wise Operations (ADD, MUL, etc.)**
```c
// Data slicing for NUMA parallelism
size_t total_elements = ne[0] * ne[1] * ne[2] * ne[3];
size_t elements_per_node = total_elements / max_numa_nodes;
size_t numa_start = numa_node * elements_per_node;
size_t numa_end = (numa_node == max_numa_nodes - 1) ? total_elements : numa_start + elements_per_node;

// SIMD operation on NUMA slice
ggml_vec_add_f32(numa_end - numa_start, dst + numa_start, src0 + numa_start, src1 + numa_start);
```

**Pattern: Reduction Operations (RMS_NORM, SOFT_MAX, etc.)**
```c
// Row-wise processing with NUMA distribution
for (int row = numa_start_row; row < numa_end_row; row++) {
    float* row_data = (float*)((char*)src + row * src->nb[1]);
    float* dst_row = (float*)((char*)dst + row * dst->nb[1]);
    
    // SIMD dot product for sum of squares
    float sum_squares = ggml_vec_dot_f32(row_size, row_data, row_data);
    float scale = 1.0f / sqrtf(sum_squares / row_size + eps);
    
    // SIMD scaling
    ggml_vec_scale_f32(row_size, dst_row, scale);
}
```

**Critical SIMD Requirements:**
- Always prefer `ggml_vec_*` over manual loops
- Ensure mathematical equivalence with reference implementation
- Validate with comprehensive multi-dimensional testing
- Handle edge cases (uneven NUMA splits, remainder elements)

### NUMA Memory Management
- **Files**: `ggml-numa-coordinator.c`, `ggml-cpu-numa-buffer.cpp`, `ggml.h`
- **NUMA mirroring**: Use `tensor_data()`/`tensor_set_data()` for NUMA-aware access
- **Memory allocation**: Always use `numa_alloc_onnode()` for local allocation
- **Thread mapping**: Each threadpool assigned to its own NUMA node

## � Debugging

```bash
# Strategy 1: Direct GDB backtrace
gdb --batch --ex run --ex bt --ex quit --args ./build/bin/test-program

# Strategy 2: Core dump analysis
echo 'core' | sudo tee /proc/sys/kernel/core_pattern
ulimit -c unlimited
# Run program, then analyze core dump:
gdb --batch --ex "file ./build/bin/program" --ex "core-file ./core" --ex "bt" --ex quit
```

## 🧪 Testing Requirements

**Critical**: ALL NUMA tests must pass before changes are complete.

```bash
# Final validation
./tests/run-numa-tests.sh  # Must return exit code 0
```

### Test Template Usage
```bash
# Copy template for new operations
cp tests/test-numa-mathematical-correctness-template.cpp tests/test-numa-mathematical-correctness-OPERATION.cpp

# Features: Multi-dimensional testing, multi-threading validation, exact mathematical comparison
# Always add tests to CMakeLists.txt and run-numa-tests.sh
```

## 💡 AI Agent Guidelines

### Critical Workflow
```bash
# 1. Verify clean starting state
./tests/run-numa-tests.sh && echo "✅ Starting clean"

# 2. Make changes incrementally with testing
cmake --build build --parallel
./build/bin/test-specific-component

# 3. Final validation
./tests/run-numa-tests.sh && echo "🎉 Complete!" || echo "❌ Fix failures"
```

### Best Practices
- **Always use dev container** - consistent environment
- **Test incrementally** - build and test after each change
- **Add comprehensive tests** - feature isn't done without tests
- **Validate with real models** - not just compilation
- **Check exit codes** - tests must properly signal failures
- **SIMD First**: Always use `ggml_vec_*` functions instead of scalar loops
- **NUMA Data Slicing**: Implement proper data distribution across NUMA nodes
- **Dual Strategy**: Create both single-node and data-parallel work functions for optimal performance

## 📋 Quick Reference

### Essential Files
```
ggml/src/ggml-cpu/ggml-numa-operation-dispatch.c  # Operation handlers
ggml/src/ggml-cpu/ggml-cpu.c                      # Mathematical kernels
tests/test-numa-mathematical-correctness-*.cpp    # Correctness tests
```

### Implementation Checklist
- [ ] Find mathematical kernel in `ggml-cpu.c`
- [ ] Extract pure mathematical operations (no ggml threading)
- [ ] Replace scalar loops with SIMD `ggml_vec_*` functions
- [ ] Implement NUMA data slicing pattern for data-parallel operations
- [ ] Create both single-node and data-parallel work functions when appropriate
- [ ] Add dispatcher handler with correct execution strategies
- [ ] Create test from template with multi-dimensional validation
- [ ] Add to CMake and test runner
- [ ] Verify `./tests/run-numa-tests.sh` passes

### Performance Commands
```bash
cmake --build build --target test-numa-mathematical-correctness-OPERATION
./build/bin/test-numa-mathematical-correctness-OPERATION
./tests/run-numa-tests.sh
```

## Changelog
Document completed tasks in `.devcontainer/changelog/YYYY-MM-DD-description.md` with current date.