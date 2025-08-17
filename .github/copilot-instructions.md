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

### Step 2: Implementation
**Work function template:**
```c
static int ggml_numa_work_function_your_operation_chunk(void* context) {
    // Extract context, slice tensors for NUMA data parallelism
    // Execute mathematical kernel on assigned chunk
    // Return 0 for success, non-zero for failure
}
```

**Dispatcher handler:**
```c
case GGML_OP_YOUR_OPERATION: {
    efficiency = 0.95f;  // High for element-wise, lower for complex
    strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL;
    work_function = ggml_numa_work_function_your_operation_chunk;
    break;
}
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

## 📋 Quick Reference

### Essential Files
```
ggml/src/ggml-cpu/ggml-numa-operation-dispatch.c  # Operation handlers
ggml/src/ggml-cpu/ggml-cpu.c                      # Mathematical kernels
tests/test-numa-mathematical-correctness-*.cpp    # Correctness tests
```

### Implementation Checklist
- [ ] Find mathematical kernel in `ggml-cpu.c`
- [ ] Implement work function with tensor slicing
- [ ] Add dispatcher handler with correct strategies
- [ ] Create test from template
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