# Copilot Instructions for llama.cpp

This document provides instructions for AI assistants working on the llama.cpp project with NUMA improvements.

## 🎯 Project Overview

This is a fork of llama.cpp with **NUMA-aware execution architecture** for optimal CPU threading and memory allocation on multi-socket systems:

- **NUMA Kernel Registry** - `ggml/src/ggml-cpu/numa-kernels/` - O(1) cache database with pre-computed strategies
- **NUMA Executor** - `ggml/src/ggml-cpu/ggml-numa-executor.c` - Strategy engine and work orchestration
- **NUMA Coordinator** - `ggml/src/ggml-cpu/ggml-numa-coordinator.c` - Resource management and work distribution
- **Dev Container** - Ubuntu 24.04 with pre-installed dependencies

**Goal**: Provide lightning-fast NUMA-aware execution for all operations through intelligent strategy selection and optimal resource utilization. 

**Architecture Flow**: `Compute Graph → Executor → Kernel Registry Query → Coordinator Dispatch → NUMA Threadpools`

## 📋 Architecture Documentation

For comprehensive architecture details, see `docs/numa-architecture.md` which covers:
- Component interfaces and responsibilities
- Execution strategies and data flow
- Performance characteristics and benchmarks
- Development guidelines and best practices

## 🔧 NUMA Kernel Implementation Workflow

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
- ✅ **Excellent**: Element-wise ops (ADD, MUL, GLU) - independent computations, linear memory access
- ⚠️ **Complex**: Matrix ops (MUL_MAT), reductions (SOFT_MAX, RMS_NORM) - need specialized splitting
- ❌ **Poor**: Global synchronization, complex dependencies

**🚀 SIMD Optimization Requirements:**
- **Always use SIMD**: Replace scalar operations with `ggml_vec_*` functions from `ggml/src/ggml-cpu/vec.h`
- **Common SIMD functions**: `ggml_vec_add_f32()`, `ggml_vec_dot_f32()`, `ggml_vec_scale_f32()`, `ggml_vec_cpy_f32()`
- **Performance impact**: SIMD provides significant speedup on modern CPUs with AVX2/AVX512 support
- **Mathematical equivalence**: SIMD operations must produce identical results to scalar reference

### Step 2: Implementation

**Critical NUMA Data Slicing Pattern:**
```c
// For data-parallel operations, ALWAYS slice data across NUMA nodes
int total_numa_nodes = context->total_numa_nodes;

// Calculate NUMA node's data slice
size_t total_elements = ggml_nelements(tensor);
size_t elements_per_node = total_elements / total_numa_nodes;
size_t numa_start = numa_node * elements_per_node;
size_t numa_end = (numa_node == total_numa_nodes - 1) ? total_elements : numa_start + elements_per_node;
```

**NUMA Kernel Implementation Pattern:**
```c
// Implement in numa-kernels/ directory
enum ggml_status ggml_numa_kernel_your_operation_execute(struct ggml_tensor * tensor, struct ggml_cplan * cplan) {
    // 1. Validate inputs
    NUMA_ASSERT(tensor != nullptr, "Tensor cannot be null");
    NUMA_ASSERT(cplan != nullptr, "Compute plan cannot be null");
    
    // 2. Extract tensor data and parameters
    const float * src0 = (const float *)tensor->src[0]->data;
    const float * src1 = (const float *)tensor->src[1]->data;
    float * dst = (float *)tensor->data;
    
    // 3. Use SIMD operations for performance
    ggml_vec_add_f32(ggml_nelements(tensor), dst, src0, src1);
    
    return GGML_STATUS_SUCCESS;
}
```

**Registry Integration:**
```c
// Add to numa-kernels.c cache population
static void populate_your_operation_cache_entries(void) {
    // TINY: < 32K elements
    g_numa_cache[GGML_OP_YOUR_OPERATION][COMPLEXITY_TINY] = (ggml_numa_kernel_cache_entry_t){
        .supported = true,
        .strategy = { .node_strategy = NUMA_NODE_STRATEGY_SINGLE, 
                     .on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD },
        .work_buffer_size_per_thread = 0,
        .work_function = ggml_numa_kernel_your_operation_execute,
        .efficiency_score = 0.95f,
        .kernel_name = "NUMA Your Operation (Single/Single)"
    };
    
    // LARGE: 16M - 256M elements  
    g_numa_cache[GGML_OP_YOUR_OPERATION][COMPLEXITY_LARGE] = (ggml_numa_kernel_cache_entry_t){
        .supported = true,
        .strategy = { .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL,
                     .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD },
        .work_buffer_size_per_thread = 1024,  // If needed
        .work_function = ggml_numa_kernel_your_operation_execute,
        .efficiency_score = 0.95f,
        .kernel_name = "NUMA Your Operation (Data-Parallel/Multi-Thread)"
    };
}
```

### Step 3: Testing
Use the mathematical correctness template:
```bash
cp tests/test-numa-mathematical-correctness-template.cpp tests/test-numa-mathematical-correctness-YOUR_OPERATION.cpp
```

**Required tests:**
- Multi-dimensional: TINY → LARGE tensor sizes
- Multi-threading: 1, 2, 4, 6, 8 threads  
- Mathematical equivalence: Exact comparison with reference
- Add to CMake and verify with `cmake --build build --target test-numa-mathematical-correctness-YOUR_OPERATION`

## 🏗️ Current Architecture Status

**✅ Implemented Components:**
- **NUMA Kernel Registry** - O(1) cache with complexity-based pre-computation
- **NUMA Executor** - Strategy engine with query-dispatch pattern
- **NUMA Coordinator** - Resource management and work distribution (cleaned of legacy cruft)

**✅ Supported Operations:**
- **ADD** - Element-wise addition with SIMD optimization

**🚀 Performance Characteristics:**
- **O(1) Strategy Lookups** - Pre-computed cache eliminates runtime overhead
- **NUMA-Aware Scheduling** - Optimal thread and memory placement
- **Cache-Optimized Execution** - Reduced memory bandwidth contention
- **Graceful Fallback** - Automatic fallback to CPU implementation when beneficial

## 🏗️ Build Environment & Commands

**Always use the dev container** for consistency (Ubuntu 24.04 with all dependencies).

### Build Commands
```bash
# Configure debug build with NUMA support
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DGGML_NUMA_MIRROR=ON -DGGML_OPENMP=OFF

# Build (let cmake auto-detect thread count)
cmake --build build --parallel
```

### Quick Model Validation
```bash
# Download test model
wget -c -O ./.devcontainer/qwen2.5-0.5b-instruct-q8_0.gguf https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf

# Test with NUMA in forced mirror mode in full coordinator run:
./build/bin/llama-server -m ./.devcontainer/qwen2.5-0.5b-instruct-q8_0.gguf --host 0.0.0.0 --numa mirror --port 8080 &

while ! curl --fail --silent http://localhost:8080/; do sleep 1; done

curl -X POST http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "qwen2.5-0.5b-instruct", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 20}'

# (Check the JSON response looks sane with a proper model response, not garbage)

# Kill the coordinator once done:
ps aux | grep llama-server | grep -v grep | awk '{print $2}' | xargs kill -9
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
# Test core components
cmake --build build --target test-numa-mathematical-correctness-add
cmake --build build --target test-numa-mathematical-correctness-rms-norm
```

### Test Template Usage
```bash
# Copy template for new operations
cp tests/test-numa-mathematical-correctness-template.cpp tests/test-numa-mathematical-correctness-OPERATION.cpp

# Features: Multi-dimensional testing, multi-threading validation, exact mathematical comparison
# Always add tests to CMakeLists.txt and verify builds successfully
```

## 💡 AI Agent Guidelines

### Critical Workflow
```bash
# 1. Verify clean starting state
cmake --build build --target ggml-cpu llama common && echo "✅ Core components building"

# 2. Make changes incrementally with testing
cmake --build build --parallel
./build/bin/test-numa-mathematical-correctness-OPERATION

# 3. Final validation - core architecture must build
cmake --build build --target ggml-cpu llama && echo "🎉 Complete!" || echo "❌ Fix failures"
```

### Best Practices
- **Always use dev container** - consistent environment
- **Test incrementally** - build and test after each change
- **Add comprehensive tests** - feature isn't done without tests
- **Validate with real models** - not just compilation
- **Check exit codes** - tests must properly signal failures
- **SIMD First**: Always use `ggml_vec_*` functions instead of scalar loops
- **Registry Integration**: Add cache entries for all complexity classes
- **Architecture Flow**: Follow Executor → Registry Query → Coordinator pattern

## 📋 Quick Reference

### Essential Files
```
ggml/src/ggml-cpu/numa-kernels/numa-kernels.c     # Kernel registry with O(1 cache
ggml/src/ggml-cpu/ggml-numa-executor.c            # Strategy engine and orchestration
ggml/src/ggml-cpu/ggml-numa-coordinator.c         # Resource management
ggml/src/ggml-cpu/ggml-cpu.c                      # Mathematical kernels (reference)
tests/test-numa-mathematical-correctness-*.cpp    # Correctness tests
docs/numa-architecture.md                         # Architecture documentation
```

### Implementation Checklist
- [ ] Find mathematical kernel in `ggml-cpu.c`
- [ ] Extract pure mathematical operations (no ggml threading)
- [ ] Replace scalar loops with SIMD `ggml_vec_*` functions
- [ ] Implement kernel function in `numa-kernels/` directory  
- [ ] Add cache entries to registry for all complexity classes
- [ ] Use `NUMA_ASSERT` for validation with proper coordinator signaling
- [ ] Create test from template with multi-dimensional validation
- [ ] Add to CMake and verify builds successfully
- [ ] Verify core architecture builds: `cmake --build build --target ggml-cpu llama`
- [ ] Add the new test to `tests/run-numa-tests.sh` and verify it and the entire suite passes


### Performance Commands
```bash
cmake --build build --target test-numa-mathematical-correctness-OPERATION
./build/bin/test-numa-mathematical-correctness-OPERATION
cmake --build build --target ggml-cpu llama common  # Core validation
```

## Changelog
Document completed tasks in `.devcontainer/changelog/YYYY-MM-DD-description.md`. Always use `date +%Y-%m-%d` to fetch the current date.