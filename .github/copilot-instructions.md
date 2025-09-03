# Copilot Instructions for llama.cpp

This document provides instructions for AI assistants working on the llama.cpp project with NUMA improvements.

## 🎯 Project Overview

This is a fork of llama.cpp with **NUMA-aware execution architecture** for optimal CPU inferencing in a NUMA environment.

### Pattern: Element-wise Operations (ADD, MUL, etc.)
```c
// Registry pattern - threshold-based strategy selection
ggml_numa_kernel_registration_info_t info = {
    .op_type = GGML_OP_ADD,
    .strategy_array = {
        .thresholds = {1024, 262144},  // 1K and 256K element thresholds
        .valid = true
    },
    .work_funcs = {
        .single_single_fn = ggml_numa_kernel_add_low_overhead_execute,
        .single_multi_fn = ggml_numa_kernel_add_low_overhead_execute, 
        .data_parallel_fn = ggml_numa_kernel_add_no_aggregation_execute,
        .valid = true
    },
    .agg_funcs = {
        .valid = false  // Element-wise operations don't need aggregation
    },
    .kernel_name = "NUMA ADD Kernel",
    .supported = true
};

// Data slicing for NUMA parallelism
size_t total_elements = ggml_nelements(tensor);
size_t numa_start = 0, numa_end = total_elements;

if (ggml_numa_is_data_parallel_execution) {
    size_t elements_per_node = total_elements / ggml_numa_total_nodes;
    numa_start = ggml_current_numa_node * elements_per_node;
    numa_end = (ggml_current_numa_node == ggml_numa_total_nodes - 1) ? 
               total_elements : numa_start + elements_per_node;
}

// SIMD operation on NUMA slice
ggml_vec_add_f32(numa_end - numa_start, dst + numa_start, src0 + numa_start, src1 + numa_start);
```ory allocation on multi-socket systems:

- **NUMA Kernel Registry** - `ggml/src/ggml-cpu/numa-kernels/` - O(1) cache database with direct function pointer dispatch
- **NUMA Executor** - `ggml/src/ggml-cpu/ggml-numa-executor.c` - Strategy engine and work orchestration
- **NUMA Coordinator** - `ggml/src/ggml-cpu/ggml-numa-simple-coordinator.c` - Resource management and work distribution
- **Dev Container** - Ubuntu 24.04 with pre-installed dependencies

**Goal**: Provide lightning-fast NUMA-aware execution for all operations through intelligent strategy selection and optimal resource utilization. 

**Architecture Flow**: `Compute Graph → Executor → Kernel Registry Direct Dispatch → Coordinator Dispatch → NUMA Threadpools`

## 📋 Architecture Documentation

For comprehensive architecture details, see `docs/numa-architecture.md` which covers:
- Component interfaces and responsibilities
- Execution strategies and data flow
- Performance characteristics and benchmarks
- Development guidelines and best practices

## 🔧 NUMA Kernel Implementation Workflow

NOTE: Remember to use Doxygen comments for all new files, functions and structures. For @author you should use David Sanftenberg.

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
- ⚠️ **Complex**: Matrix ops (MUL_MAT), reductions (RMS_NORM) - need specialized splitting
- 🔧 **View Operations**: RESHAPE, PERMUTE - metadata-only, minimal computation
- ❌ **Poor**: Global synchronization, complex dependencies

**🚀 SIMD Optimization Requirements:**
- **Always use SIMD**: Replace scalar operations with `ggml_vec_*` functions from `ggml/src/ggml-cpu/vec.h`
- **Common SIMD functions**: `ggml_vec_add_f32()`, `ggml_vec_dot_f32()`, `ggml_vec_scale_f32()`, `ggml_vec_cpy_f32()`
- **Performance impact**: SIMD provides significant speedup on modern CPUs with AVX2/AVX512 support
- **Mathematical equivalence**: SIMD operations must produce identical results to scalar reference

### Step 2: Template Selection & Implementation

**📚 NUMA Kernel Template System:**
Choose the appropriate template based on operation characteristics:

**🔹 Binary Element-wise Operations Template**: `numa-kernels/add.c`
- **Use for**: ADD, MUL, SUB, DIV and similar simple element-wise operations
- **Pattern**: Uniform memory access, simple data-parallel execution
- **Characteristics**: Single-pass algorithms, no aggregation needed
- **NUMA Strategy**: Direct element slicing across NUMA nodes

**🔹 Complex Operations Template**: `numa-kernels/mul_mat.c`
- **Use for**: Matrix multiplication, convolutions, complex transformations
- **Pattern**: Multidimensional slicing, specialized SIMD patterns  
- **Characteristics**: Non-uniform memory access, chunk-based processing
- **NUMA Strategy**: Sophisticated work distribution algorithms

**🔹 Reduction Operations Template**: `numa-kernels/rms_norm.c`
- **Use for**: Normalization, statistical operations, dimension-wise reductions
- **Pattern**: Row-wise/column-wise processing, potential aggregation
- **Characteristics**: Multi-pass algorithms, cache-optimized access patterns
- **NUMA Strategy**: Dimension-aware parallelization with optional aggregation

**🔹 View Operations Template**: `numa-kernels/reshape.c`
- **Use for**: RESHAPE, PERMUTE, VIEW, TRANSPOSE and similar metadata-only operations
- **Pattern**: No-op execution, metadata transformation only
- **Characteristics**: Zero computational overhead, single-node execution
- **NUMA Strategy**: `NUMA_NODE_STRATEGY_SINGLE` with `NUMA_ON_NODE_STRATEGY_SINGLE_THREAD`

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
// Implement in numa-kernels/ directory using appropriate template
enum ggml_status ggml_numa_kernel_your_operation_execute(void * work_context, struct ggml_compute_params * params) {
    struct ggml_tensor * tensor = (struct ggml_tensor *)work_context;
    
    // 1. Validate inputs
    NUMA_ASSERT(tensor != nullptr, "Tensor cannot be null");
    NUMA_ASSERT(params != nullptr, "Compute params cannot be null");
    
    // 2. Extract tensor data using shared memory approach
    const float * src0 = (const float *)tensor_data(tensor->src[0]);
    const float * src1 = (const float *)tensor->src[1] ? (const float *)tensor_data(tensor->src[1]) : NULL;
    
    // Use shared result tensor memory for direct writes (eliminates aggregation)
    extern __thread void * ggml_numa_shared_result_tensor_data;
    float * dst;
    if (ggml_numa_shared_result_tensor_data != NULL) {
        // Use shared result tensor memory - eliminates aggregation overhead
        dst = (float *)ggml_numa_shared_result_tensor_data;
    } else {
        // Fallback to local tensor data for compatibility
        dst = (float *)tensor_data(tensor);
    }
    
    // 3. Get NUMA execution context
    extern __thread int ggml_current_numa_node;
    extern __thread int ggml_numa_total_nodes_for_data_parallel; 
    extern __thread bool ggml_numa_is_data_parallel_execution;
    
    // 4. Calculate NUMA data slice for data-parallel execution
    size_t total_elements = ggml_nelements(tensor);
    size_t numa_start = 0, numa_end = total_elements;
    
    if (ggml_numa_is_data_parallel_execution) {
        size_t elements_per_node = total_elements / ggml_numa_total_nodes;
        numa_start = ggml_current_numa_node * elements_per_node;
        numa_end = (ggml_current_numa_node == ggml_numa_total_nodes - 1) ? 
                   total_elements : numa_start + elements_per_node;
    }
    
    // 5. Use SIMD operations for performance (on NUMA slice)
    ggml_vec_add_f32(numa_end - numa_start, dst + numa_start, src0 + numa_start, src1 + numa_start);
    
    NUMA_LOG_TRACE("Processed elements %zu-%zu on NUMA node %d", numa_start, numa_end, ggml_current_numa_node);
    
    return GGML_STATUS_SUCCESS;
}
```

**Registry Integration:**
```c
// Step 1: Create register function in your kernel .c file (e.g., add.c, mul.c, etc.)
ggml_numa_kernel_registration_info_t ggml_numa_kernel_your_operation_register(void) {
    ggml_numa_kernel_registration_info_t info = {0};
    
    info.op_type = GGML_OP_YOUR_OPERATION;
    info.supported = true;
    info.kernel_name = "NUMA Your Operation Kernel";
    
    // Strategy thresholds for operation
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_SINGLE] = 1024;      // Single thread below 1K elements
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_MULTI] = 262144;     // Multi-thread below 256K elements
    // Above 256K elements: data-parallel strategy
    info.strategy_array.valid = true;
    
    // Function pointers for different strategies
    info.work_funcs.single_single_fn = ggml_numa_kernel_your_operation_execute;
    info.work_funcs.single_multi_fn = ggml_numa_kernel_your_operation_execute;
    info.work_funcs.data_parallel_fn = ggml_numa_kernel_your_operation_execute;
    info.work_funcs.valid = true;
    
    // Query function pointer - enables direct dispatch without switch statements
    info.query_fn = (void*)ggml_numa_kernel_your_operation_query;
    
    // Most operations don't need aggregation functions
    info.agg_funcs.single_single_fn = NULL;
    info.agg_funcs.single_multi_fn = NULL; 
    info.agg_funcs.data_parallel_fn = NULL;
    info.agg_funcs.valid = false;
    
    return info;
}

// Step 2: Add function declarations to your kernel .h file (e.g., add.h, mul.h, etc.)
ggml_numa_kernel_registration_info_t ggml_numa_kernel_your_operation_register(void);
ggml_numa_kernel_query_result_t ggml_numa_kernel_your_operation_query(const struct ggml_tensor * tensor);

// Step 3: Enable in numa-kernels.c using NUMA_REGISTER_KERNEL macro
void ggml_numa_kernels_init(void) {
    // ... other kernels ...
    
    // Use NUMA_REGISTER_KERNEL macro for automatic registration with direct dispatch
    NUMA_REGISTER_KERNEL(your_operation);
}
```

**🚀 NEW ARCHITECTURE: Direct Function Pointer Dispatch**
- **Eliminated Switch Statement**: Registry now uses direct function pointer calls for O(1) kernel query dispatch
- **Automatic Registration**: `NUMA_REGISTER_KERNEL()` macro automatically populates query function pointers
- **Zero Maintenance Overhead**: Adding new kernels requires NO changes to central dispatch logic
- **Cache-Optimized Lookup**: Single array access followed by direct function call: `entry->query_fn(tensor)`

**🚨 CRITICAL: Always use NUMA_REGISTER_KERNEL() macro**
- **Pattern**: `NUMA_REGISTER_KERNEL(kernel_name)` in `numa-kernels.c`
- **Consistent architecture**: All kernels use the same registration mechanism
- **Automatic validation**: Macro includes error checking and debug logging
- **Direct Dispatch**: Macro automatically populates function pointers
```

### Step 3: Testing
Use the mathematical correctness template:
```bash
cp tests/test-numa-mathematical-correctness-template.cpp tests/test-numa-mathematical-correctness-YOUR_OPERATION.cpp
```

**Required tests:**
- Multi-dimensional: TINY → GIGANTIC_16GB tensor sizes (now includes GB-scale support)
- Multi-threading: 1, 2, 4, 6, 8, 15, 16, 31, 32, 64, 128 threads
- Hardware-specific Data Parallel: Data parallel tests with all numas available on the machine using max thread counts per numa node
- Mathematical equivalence: Exact comparison with reference
- Add to CMake and verify with `cmake --build build --target test-numa-mathematical-correctness-YOUR_OPERATION`

## 🏗️ Current Architecture Status

**✅ Implemented Components:**
- **NUMA Kernel Registry** - O(1) hash table with threshold-based strategy selection
- **NUMA Executor** - Strategy engine with registry query-dispatch pattern  
- **NUMA Coordinator** - Resource management and work distribution

**✅ Supported Operations:**
- **ADD** - Element-wise addition with SIMD optimization and shared memory approach
- **MUL** - Element-wise multiplication with optimized data-parallel execution
- **GLU** - Gated Linear Unit activation with specialized SIMD patterns
- **ROPE** - Rotary Position Embedding with complex mathematical transformations
- **PERMUTE** - Tensor axis permutation for view operations (no-op pattern)
- **RMS_NORM** - Root mean square normalization with row-wise NUMA distribution (critical for transformers)
- **MUL_MAT** - Matrix multiplication with chunk-based work distribution and type-specific SIMD operations
- **RESHAPE** - Tensor shape transformation for view operations (no-op metadata-only)

**🚀 Performance Characteristics:**
- **O(1) Strategy Lookups** - Direct function pointer dispatch eliminates all search overhead
- **Zero-Maintenance Dispatch** - No switch statements to update when adding new kernels
- **Threshold-Based Selection** - Simple element count thresholds for optimal strategy choice
- **NUMA-Aware Scheduling** - Optimal thread and memory placement
- **Shared Memory Optimization** - Direct memory writes eliminate aggregation overhead for most operations
- **3-Level Debug System** - Clean output at levels 1-2, detailed tracing at level 3
- **Work Function Architecture** - Clean separation between computation (work functions) and result combination (aggregation functions)
- **Registry-Based Scalability** - Easy addition of new kernels with consistent patterns

**📊 Current System Status:**
- **Total Registered Kernels**: 8 active (ADD, MUL, GLU, ROPE, PERMUTE, RMS_NORM, MUL_MAT, RESHAPE)
- **Kernel Implementation Files**: 18 files (9 headers + 9 implementations, plus system files)
- **Template Categories**: 4 types (Binary Element-wise, Complex, Reduction, View operations)
- **Registry Architecture**: NUMA_REGISTER_KERNEL() macro with automatic query dispatch
- **Test Coverage**: Mathematical correctness and performance benchmarks for all kernels

## 🏗️ Build Environment & Commands

**Always use the dev container** for consistency (Ubuntu 24.04 with all dependencies).

### Build Commands
```bash
# Configure debug build with NUMA support
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DGGML_NUMA_MIRROR=ON -DGGML_OPENMP=OFF

# Build (let cmake auto-detect thread count)
cmake --build build --parallel
```

### Debug Environment Variable Control

**NUMA Debug Logging Control**: All NUMA debug messages are controlled by the `GGML_NUMA_DEBUG` environment variable:

```bash
# Clean, silent operation (default) - no debug messages
unset GGML_NUMA_DEBUG
# OR
export GGML_NUMA_DEBUG=0

# Enable debug messages - shows NUMA strategy decisions and execution details
export GGML_NUMA_DEBUG=1

# Enable verbose debug messages - shows additional internal details
export GGML_NUMA_DEBUG=2

# Enable trace debug messages - very detailed, includes individual operations
export GGML_NUMA_DEBUG=3
```

**Debug Message Categories:**
- **Executor Debug**: Strategy selection, kernel queries, execution paths
- **Coordinator Debug**: Thread management, NUMA binding verification, dispatch decisions
- **Kernel Debug**: Mathematical operation details, data slicing, SIMD optimizations
- **Memory Debug**: NUMA allocation successes, memory mirroring operations
- **Trace Debug**: Very detailed per-operation logging, individual tensor rows, thread work distribution

**Performance Impact**: When `GGML_NUMA_DEBUG` is unset or 0, all debug output is completely disabled, eliminating printf overhead during inference loops.

**Usage Examples:**
```bash
# Production inference - silent, maximum performance
./build/bin/llama-server -m model.gguf --numa mirror

# Development debugging - rich diagnostic output
GGML_NUMA_DEBUG=1 ./build/bin/llama-bench -m model.gguf --numa mirror

# Detailed troubleshooting - verbose internal details
GGML_NUMA_DEBUG=2 ./build/bin/llama-server -m model.gguf --numa mirror

# Ultra-detailed troubleshooting - trace level with individual operations
GGML_NUMA_DEBUG=3 ./build/bin/llama-server -m model.gguf --numa mirror

# Integration testing with debug output
GGML_NUMA_DEBUG=1 ./tests/run-numa-integration-test.sh --verbose --numa mirror

# Performance testing with debug
GGML_NUMA_DEBUG=2 ./tests/run-numa-performance-tests.sh --operation=ADD
```
```

### Quick Model Validation

**Automated Integration Test (Recommended)**
```bash
# Run the automated integration test with NUMA mirror mode
./tests/run-numa-integration-test.sh --numa mirror

# Run with debug output for troubleshooting
GGML_NUMA_DEBUG=1 ./tests/run-numa-integration-test.sh --verbose --numa mirror

# Test different NUMA modes
./tests/run-numa-integration-test.sh --numa distribute
./tests/run-numa-integration-test.sh --numa isolate

# Run without NUMA for baseline comparison
./tests/run-numa-integration-test.sh
```

The integration test automatically:
- Downloads the test model if needed
- Starts llama-server with proper configuration
- Passes through all environment variables (GGML_NUMA_DEBUG, GGML_LOG_DEBUG, etc.)
- Waits for server and model loading
- Sends a test prompt and validates response
- Cleans up server processes
- Reports success/failure with clear messages

**Manual Process (For Advanced Debugging)**
```bash
# Download test model (if needed)
wget -c -O ./.devcontainer/qwen2.5-0.5b-instruct-q8_0.gguf https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf

# Manual test with NUMA mirror mode:
./build/bin/llama-server -m ./.devcontainer/qwen2.5-0.5b-instruct-q8_0.gguf --host 0.0.0.0 --numa mirror --port 8080 &

while ! curl --fail --silent http://localhost:8080/; do sleep 1; done

curl -X POST http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "qwen2.5-0.5b-instruct", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 20}'

# (Check the JSON response looks sane with a proper model response, not garbage)

# Kill the server once done:
ps aux | grep llama-server | grep -v grep | awk '{print $2}' | xargs kill -9
```

## 🧠 Key Technical Areas

### NUMA Data Slicing & SIMD Optimization Patterns

**Successful SIMD Implementations:**
- **ADD Operation**: Element-wise addition using `ggml_vec_add_f32()` with data-parallel execution and no-aggregation optimization

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
// Registry pattern - may need aggregation for reductions
ggml_numa_kernel_registration_info_t info = {
    .op_type = GGML_OP_RMS_NORM,
    .strategy_array = {
        .thresholds = {1024, 65536},  // Smaller thresholds for reductions
        .valid = true
    },
    .work_funcs = {
        .single_single_fn = ggml_numa_kernel_rms_norm_execute,
        .single_multi_fn = ggml_numa_kernel_rms_norm_execute,
        .data_parallel_fn = ggml_numa_kernel_rms_norm_execute,
        .valid = true
    },
    .agg_funcs = {
        .aggregation_fn = ggml_numa_kernel_rms_norm_aggregate,  // Custom aggregation
        .valid = true
    },
    .kernel_name = "NUMA RMS_NORM Kernel",
    .supported = true
};

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

### GB-Scale Tensor Infrastructure & Shared Memory Optimization

**Complexity Classification System:**
The NUMA framework now supports 10 complexity levels for optimal strategy selection:
- `COMPLEXITY_TINY` → `COMPLEXITY_HUGE`: Traditional levels for smaller tensors
- `COMPLEXITY_GIGANTIC_1GB` through `COMPLEXITY_GIGANTIC_16GB`: New GB-scale levels

**Shared Memory Optimization:**
For large tensors (1GB+), the shared memory approach provides significant performance improvements:
- **Direct memory writes** to final tensor memory locations
- **Zero-copy architecture** with proper NUMA memory locality
- **Eliminates aggregation overhead** by writing directly to shared result tensor

**Pattern: Shared Memory Kernel Implementation**
Note: F32 example is shown but all quant types in `quants.c` must be supported.

```c
// Shared memory kernel for GB-scale tensors
enum ggml_status ggml_numa_kernel_add_shared_memory_execute(void * work_context, struct ggml_compute_params * params) {
    struct ggml_tensor * tensor = (struct ggml_tensor *)work_context;
    
    // Get NUMA execution context from thread-local variables
    extern __thread void * ggml_numa_shared_result_tensor_data;
    extern __thread bool ggml_numa_is_data_parallel_execution;
    extern __thread int ggml_current_numa_node;
    
    // Use shared memory for direct writes in data-parallel mode
    float * dst_data;
    if (ggml_numa_shared_result_tensor_data != NULL) {
        dst_data = (float *)ggml_numa_shared_result_tensor_data;
    } else {
        dst_data = (float *)tensor_data(tensor);
    }
    
    // Direct SIMD operation on final tensor memory (no aggregation needed)
    ggml_vec_add_f32(numa_end - numa_start, dst_data + numa_start, src0 + numa_start, src1 + numa_start);
    
    return GGML_STATUS_SUCCESS;
}
```

**GB-Scale Tensor Dimension Calculation:**
```c
// For GB-scale test tensors, use cube root for balanced dimensions
size_t gb_elements = gb_size * 268435456; // ~1GB = 268M float32 elements
size_t cube_root = (size_t)cbrt((double)gb_elements);
tensor_dims = {cube_root, cube_root, cube_root, 1}; // Balanced 3D tensor
```

### NUMA Memory Management
- **Files**: `ggml-numa-simple-coordinator.c`, `ggml-cpu-numa-buffer.cpp`, `ggml.h`
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

# Run comprehensive test suite (includes integration test at end)
./tests/run-numa-tests.sh

# Run performance tests
./tests/run-numa-performance-tests.sh --operation=ADD --quick

# Run integration test separately
./tests/run-numa-integration-test.sh --numa mirror
```

### Test Template Usage
```bash
# Copy template for new operations
cp tests/test-numa-mathematical-correctness-template.cpp tests/test-numa-mathematical-correctness-OPERATION.cpp

# Features: 3-part test structure with comprehensive quantization coverage
# 1. Mathematical equivalence testing (multi-dimensional tensors, multi-threading validation)  
# 2. Quantization type coverage (Q8_0, Q4_0, Q5_0, F16, F32 - prevents silent model failures)
# 3. Regression testing (operation-specific edge cases and previous bug scenarios)
# Always add tests to CMakeLists.txt and verify builds successfully
```

**Critical**: ALL mathematical correctness tests MUST include comprehensive quantization type coverage. Missing quantization testing can lead to silent model inference failures in production. The template enforces this requirement with detailed TODOs and examples.

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
- **Architecture Flow**: Follow Executor → Registry Direct Dispatch → Coordinator pattern
- **Debug Control**: Use `GGML_NUMA_DEBUG=1` for development debugging, unset for performance testing
- **Kernel Registration**: Always use `NUMA_REGISTER_KERNEL()` macro, never legacy function-based registration
- **Strategy Selection**: Use `NUMA_SELECT_STRATEGY_FROM_CACHE()` macro for unified threshold-based strategy selection

### Strategy Selection Pattern
All kernels using cache-based thresholds should use the shared macro for consistency:
```c
// Get cache entry for this operation
const ggml_numa_kernel_cache_entry_t * cache_entry = ggml_numa_lookup_kernel_direct(tensor->op);

// Calculate total elements for threshold comparison
size_t total_elements = ggml_nelements(tensor);

// Use shared macro for unified strategy selection
ggml_numa_execution_strategy_t selected_strategy;
NUMA_SELECT_STRATEGY_FROM_CACHE(cache_entry, total_elements, selected_strategy);

// selected_strategy now contains the optimal strategy:
// - Below threshold[0]: Single node, single thread (fastest for tiny tensors)
// - Below threshold[1]: Single node, multi-thread (good for small-medium tensors)  
// - Above threshold[1]: Data-parallel across NUMA nodes (optimal for large tensors)
```

**Benefits:**
- **Single Source of Logic**: Strategy selection logic centralized in one macro
- **Consistent Behavior**: All kernels using this macro behave identically  
- **Maintainability**: Changes to strategy logic only need to be made in one place
- **Zero Performance Impact**: Macro expands to identical code at compile time

### Debug Message Implementation
When adding new NUMA components, always use the centralized debug control system:
```c
// Include the debug header
#include "ggml-numa-shared.h"

// Use controlled debug macros instead of printf
NUMA_LOG_DEBUG("Your debug message: %d\n", value);
NUMA_LOG_VERBOSE("Detailed debug info: %f\n", detail);

// Never use printf directly for debug messages - use controlled macros
// printf("Debug: ..."); // ❌ DON'T DO THIS
// NUMA_LOG_DEBUG("Debug: ..."); // ✅ DO THIS
```

## 📋 Quick Reference

### Essential Files
```
ggml/src/ggml-cpu/numa-kernels/numa-kernels.c     # Kernel registry
ggml/src/ggml-cpu/ggml-numa-executor.c            # Strategy engine and orchestration
ggml/src/ggml-cpu/ggml-numa-simple-coordinator.c  # Threadpool and NUMA management
ggml/src/ggml-cpu/ggml-numa-shared.h              # Debug control and shared utilities
ggml/src/ggml-cpu/ggml-numa-perf.c                # Performance instrumentation framework
ggml/src/ggml-cpu/ggml-cpu.c                      # Mathematical kernels (reference)
tests/test-numa-mathematical-correctness-*.cpp    # Correctness tests
tests/run-numa-performance-tests.sh               # Performance test orchestrator
tests/test-numa-execution-modes.cpp               # Runs individual op performance tests for the Performance test orchestrator across a variety of NUMA modes
docs/numa-architecture.md                         # Architecture documentation

# NUMA Kernel Templates (Choose Based on Operation Type)
ggml/src/ggml-cpu/numa-kernels/add.c              # Template: Binary element-wise operations
ggml/src/ggml-cpu/numa-kernels/mul_mat.c          # Template: Complex operations & matrix ops
ggml/src/ggml-cpu/numa-kernels/rms_norm.c         # Template: Reduction operations & normalizations
ggml/src/ggml-cpu/numa-kernels/reshape.c          # Template: View operations & metadata transformations
```

### Implementation Checklist
- [ ] Find mathematical kernel in `ggml-cpu.c`
- [ ] **Choose appropriate template**: Binary (add.c), Complex (mul_mat.c), Reduction (rms_norm.c), or View (reshape.c)
- [ ] Extract pure mathematical operations (no ggml threading)
- [ ] Replace scalar loops with SIMD `ggml_vec_*` functions
- [ ] **Copy template and adapt** for your operation type
- [ ] Implement kernel function in `numa-kernels/` directory following template patterns
- [ ] Use shared memory setup: check `ggml_numa_shared_result_tensor_data` for direct writes
- [ ] Create `ggml_numa_kernel_{operation}_register()` function that returns registration info
- [ ] Create `ggml_numa_kernel_{operation}_query()` function using `NUMA_SELECT_STRATEGY_FROM_CACHE()` macro
- [ ] Add function declarations to kernel header file (e.g., `add.h`, `mul.h`)
- [ ] Enable in `numa-kernels.c` using `NUMA_REGISTER_KERNEL(operation)` macro
- [ ] Use `NUMA_ASSERT` for validation with proper coordinator signaling
- [ ] Use `NUMA_LOG_DEBUG` macros instead of printf for debug messages
- [ ] Create test from template with multi-dimensional validation
- [ ] Add to CMake and verify builds successfully
- [ ] Verify core architecture builds: `cmake --build build --target ggml-cpu llama`
- [ ] Add the new test to `tests/run-numa-tests.sh` and verify it and the entire suite passes
- [ ] Run integration test to validate real-world functionality: `./tests/run-numa-integration-test.sh --numa mirror`


### Performance Commands
```bash
cmake --build build --target test-numa-mathematical-correctness-OPERATION
./build/bin/test-numa-mathematical-correctness-OPERATION
cmake --build build --target ggml-cpu llama common  # Core validation

# Run GB-scale performance tests
./tests/run-numa-performance-tests.sh --operation=OPERATION

# Run integration test with real model
./tests/run-numa-integration-test.sh --numa mirror
```

## Changelog
Document completed tasks in `.devcontainer/changelog/YYYY-MM-DD-description.md`. Always use `date +%Y-%m-%d` to fetch the current date.