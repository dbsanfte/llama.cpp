# Copilot Instructions for llama.cpp

This document provides instructions for AI assistants (GitHub Copilot, Claude, etc.) working on the llama.cpp project with NUMA improvements and development container setup.

## 🎯 Project Overview

This is a fork of llama.cpp with **NUMA-aware improvements** for better CPU threading and memory allocation. The project includes:

- **NUMA-aware Coordinator and Node/Thread assignment** - `ggml/src/ggml-cpu/ggml-numa-coordinator.c`
- **Work-in-Progress Operation Dispatcher to the Coordinator** - `ggml/src/ggml-cpu/ggml-numa-operation-dispatch.c`
- **Fallback Operations to older implementation for ops that aren't NUMA-parallelized yet to use the Coordinator** - `ggml/src/ggml-cpu/ggml-numa-fallback.c`
- **Development container** - Ubuntu 24.04 with all dependencies for consistent building

### Goal

Our goal is to implement a NUMA-aware Operation Dispatch and Coordinator-based Execution model for the llama.cpp project, improving performance on multi-socket systems. Our main consumers will be `src/llama-context.cpp` via `ggml/src/ggml-cpu/ggml-cpu.c`, which dispatches to our Dispatcher, which then Executes via the Coordinator via an interface.

We must implement the complete 85-ish set of arithmetic operations in `ggml/src/ggml-cpu/ops.h` in our dispatcher.

The plan for this can be found in: `.devcontainer/changelog/2025-08-14-operation-analysis-big-bang-strategy.md`

## 🔧 NUMA Operation Parallelization Workflow

This section documents the complete workflow for fully NUMA parallelizing operations like GLU, ROPE, MUL_MAT, etc. Follow this systematic approach for any new operation implementation.

### Step 1: Operation Analysis & Mathematical Kernel Discovery

#### 1.1 Identify the Operation
```bash
# Find the operation in the ggml operation list
grep -r "GGML_OP_YOUR_OPERATION" ggml/src/ggml-cpu/
# Look for existing implementations in ops.h and related files
```

#### 1.2 Locate Mathematical Kernels
Operations typically have mathematical kernels in these locations:
- **Primary location**: `ggml/src/ggml-cpu/ggml-cpu.c` - Main CPU implementations
- **Operation headers**: `ggml/src/ggml-cpu/ops.h` - Operation interfaces and utilities
- **Specialized kernels**: `ggml/src/ggml-cpu/ggml-cpu-*` files for specific optimizations

**Key functions to identify:**
```c
// Look for functions like:
ggml_compute_forward_your_operation()        // Main computation function
ggml_compute_forward_your_operation_f32()   // Float32 variant
ggml_your_operation_impl_*()                // Implementation variants
```

#### 1.3 Analyze Operation Characteristics
Determine if the operation is suitable for data parallelism:

**✅ Excellent candidates (Data Parallel):**
- Element-wise operations (GLU, ADD, MUL)
- Independent computations per output element
- Linear memory access patterns
- No inter-element dependencies

**⚠️ Complex candidates (Require careful analysis):**
- Matrix operations (MUL_MAT) - need specialized splitting
- Reduction operations (SOFT_MAX) - need coordination
- Sequence operations (ROPE) - may have positional dependencies

**❌ Poor candidates:**
- Operations requiring global synchronization
- Complex inter-element dependencies
- Operations that don't benefit from NUMA distribution

### Step 2: Current Implementation Assessment

#### 2.1 Check Existing NUMA Status
```bash
# Search for existing handler in dispatcher
grep -A 10 -B 5 "GGML_OP_YOUR_OPERATION" ggml/src/ggml-cpu/ggml-numa-operation-dispatch.c
```

Look for:
- Current strategy: `NUMA_NODE_STRATEGY_SINGLE` vs `NUMA_NODE_STRATEGY_DATA_PARALLEL`
- Efficiency estimate
- Work function assignment

#### 2.2 Analyze Fallback Usage
```bash
# Check if operation falls back to old implementation
grep -r "your_operation" ggml/src/ggml-cpu/ggml-numa-fallback.c
```

### Step 3: Mathematical Kernel Extraction & Understanding

#### 3.1 Study the Core Algorithm
Examine the main computation function:
```c
// Example: For GLU operation
static void ggml_compute_forward_glu_f32(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst) {
    
    // Key aspects to understand:
    // 1. Input tensor structure and dimensions
    // 2. Output tensor generation logic  
    // 3. Mathematical computation per element
    // 4. Memory access patterns
    // 5. Threading model (ith, nth parameters)
}
```

#### 3.2 Identify Tensor Access Patterns
```c
// Understand how the operation accesses data:
const float * src0_ptr = (float *)src0->data;  // Input tensors
float * dst_ptr = (float *)dst->data;          // Output tensor

// Memory stride and offset calculations
// Element indexing and addressing
// Batch processing logic
```

#### 3.3 Extract Pure Mathematical Logic
Isolate the core computation that can be parallelized:
```c
// Example: GLU core computation
for (int64_t i = 0; i < ne; i++) {
    const float x = src0_ptr[i];
    const float y = src0_ptr[i + ne];
    dst_ptr[i] = x * activation_function(y);  // Core math
}
```

### Step 4: NUMA Work Function Implementation

#### 4.1 Design Work Function Interface
```c
// Standard work function signature
static int ggml_numa_work_function_your_operation_chunk(void* context) {
    // Extract operation details from context
    // Handle tensor slicing for NUMA data parallelism
    // Execute mathematical kernel on assigned chunk
    // Return 0 for success, non-zero for failure
}
```

#### 4.2 Implement Tensor Slicing Logic
```c
// For data-parallel operations:
// 1. Calculate work distribution across NUMA nodes
int64_t total_elements = ggml_nelements(operation->dst);
int64_t elements_per_node = total_elements / num_numa_nodes;
int64_t start_idx = numa_node * elements_per_node;
int64_t end_idx = (numa_node == last_node) ? total_elements : start_idx + elements_per_node;

// 2. Adjust tensor pointers for local slice
float* local_src_ptr = src_ptr + start_idx;
float* local_dst_ptr = dst_ptr + start_idx;
int64_t local_elements = end_idx - start_idx;
```

#### 4.3 Integrate Mathematical Kernel
```c
// Option A: Call existing kernel with modified parameters
struct ggml_compute_params single_thread_params = {
    .ith = 0,           // Always 0 for single-threaded execution per node
    .nth = 1,           // Single thread per NUMA node
    .wsize = 0,
    .wdata = NULL,
    .threadpool = NULL
};

// Modify tensor to represent local slice and call existing function
ggml_compute_forward_your_operation_f32(&single_thread_params, local_tensor);

// Option B: Implement optimized NUMA-aware computation directly
// Use the extracted mathematical logic with local pointers
```

### Step 5: Dispatcher Integration

#### 5.1 Update Operation Handler
Add or modify the case in `ggml/src/ggml-cpu/ggml-numa-operation-dispatch.c`:

```c
case GGML_OP_YOUR_OPERATION: {
    // Set efficiency based on operation characteristics
    efficiency = 0.95f;  // High for element-wise, lower for complex ops
    
    // Choose appropriate strategy
    strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL;  // For most operations
    // strategy = NUMA_NODE_STRATEGY_SINGLE;     // For problematic operations
    
    // Assign work function
    work_function = ggml_numa_work_function_your_operation_chunk;
    break;
}
```

#### 5.2 Efficiency Guidelines
- **0.95-0.98**: Element-wise operations (GLU, ADD, MUL)
- **0.80-0.90**: Matrix operations with good parallelization (MUL_MAT)
- **0.60-0.80**: Operations with some coordination overhead (SOFT_MAX)
- **0.40-0.60**: Complex operations with significant NUMA overhead

### Step 6: Comprehensive Testing Implementation

#### 6.1 Create Mathematical Correctness Test
```bash
# Use the proven template approach
cp tests/test-numa-mathematical-correctness-template.cpp \
   tests/test-numa-mathematical-correctness-your-operation.cpp
```

#### 6.2 Customize Test Implementation
Key areas to modify in the template:

```cpp
// 1. Update class name and operation references
class NumaYourOperationMathematicalCorrectnessTestSuite

// 2. Define test cases with appropriate tensor dimensions
struct TestCase test_cases[] = {
    {"YOUR_OPERATION-TINY", 64, 64, 1, "your_operation"},
    {"YOUR_OPERATION-SMALL", 128, 128, 1, "your_operation"},
    {"YOUR_OPERATION-MEDIUM", 256, 256, 1, "your_operation"},
    // Add operation-specific test dimensions
};

// 3. Implement test logic for your specific operation
bool test_single_your_operation_case(/*...*/) {
    // Create appropriate tensors for your operation
    // Generate deterministic test data
    // Execute NUMA parallel version via ggml_numa_intercept_operation
    // Execute serial reference implementation
    // Compare results with compare_float_arrays()
}
```

#### 6.3 Add to Build System
Update `tests/CMakeLists.txt`:
```cmake
set(LLAMA_TEST_NAME test-numa-mathematical-correctness-your-operation)
llama_build_and_test(test-numa-mathematical-correctness-your-operation.cpp)
target_link_libraries(${LLAMA_TEST_NAME} PRIVATE ggml ggml-cpu common)
```

Update `tests/run-numa-tests.sh`:
```bash
NUMA_TESTS=(
    # ... existing tests ...
    "test-numa-mathematical-correctness-your-operation"
)
```

### Step 7: Validation & Performance Testing

#### 7.1 Mathematical Correctness Validation
```bash
# Build and run individual test
cmake --build build --target test-numa-mathematical-correctness-your-operation
./build/bin/test-numa-mathematical-correctness-your-operation

# Run comprehensive test suite
./tests/run-numa-tests.sh
```

#### 7.2 Performance Validation
```bash
# Test with real model to ensure no regressions
./build/bin/llama-server -m ./.devcontainer/qwen2.5-0.5b-instruct-q8_0.gguf \
  --host 0.0.0.0 --numa mirror --port 8080 &

# Verify functionality with actual inference
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5-0.5b-instruct", "messages": [{"role": "user", "content": "Test"}]}'
```

### Step 8: Documentation & Changelog

#### 8.1 Create Detailed Changelog
Document in `.devcontainer/changelog/YYYY-MM-DD-operation-numa-implementation.md`:
- Analysis findings
- Implementation details
- Performance characteristics
- Test results
- Future optimization opportunities

#### 8.2 Update Copilot Instructions
Add operation-specific notes to this document if needed.

## 🔍 Common Patterns & Best Practices

### Successful Operation Characteristics
- **GLU**: Element-wise with activation functions → Perfect data parallelism
- **ADD/MUL**: Simple element-wise → Excellent candidates  
- **RMS_NORM**: Row-wise normalization with reduction → Good data parallelism (0.85 efficiency)
- **MUL_MAT**: Matrix multiplication → Complex but highly parallel
- **SOFT_MAX**: Reduction operation → Requires careful coordination

### Work Function Patterns
```c
// Pattern 1: Simple element-wise operations
static int ggml_numa_work_function_elementwise_chunk(void* context) {
    // Direct mathematical kernel execution on tensor slice
    // High efficiency, minimal overhead
}

// Pattern 2: Complex operations requiring coordination
static int ggml_numa_work_function_complex_chunk(void* context) {
    // May require multiple phases
    // Inter-node communication/synchronization
    // Lower efficiency due to coordination overhead
}
```

### Testing Patterns
- **Multi-dimensional testing**: TINY → LARGE tensor sizes
- **Thread strategy validation**: 1, 2, 4, 6, 8 threads
- **Mathematical equivalence**: Exact comparison with reference
- **Performance regression**: Real model validation

### Debugging Strategies
```bash
# Enable debug output for NUMA operations
export GGML_NUMA_DEBUG=1

# Use GDB for segfaults
gdb --batch --ex run --ex bt --ex quit --args ./build/bin/test-your-operation

# Core dump analysis for threading issues
echo 'core' | sudo tee /proc/sys/kernel/core_pattern
ulimit -c unlimited
# Run test and analyze core dump with gdb
```

## 🏗️ Build Environment Setup

### Primary Development Method: Dev Container

**Always prefer the dev container for consistency**:

1. **Check if in container**: Look for `/.dockerenv` or check environment
2. **Start container**: If in VS Code, use "Dev Containers: Reopen in Container"
3. **Dependencies included**: All NUMA tools, build tools, debugging tools pre-installed

### Quick Build Commands

```bash
# Debug build - configure (pick up new tests)
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DGGML_NUMA_MIRROR=ON -DGGML_OPENMP=OFF

# Debug build - build
cmake --build build --parallel

# ^^ Note: Never limit the threads count of `--parallel`, just let cmake autodetect the number of cores and choose the max threadcount itself.
```

### Quick sanity check against a real model:
```bash
# Test llama-server with NUMA mirror mode for production-ready validation
wget -c -O ./.devcontainer/qwen2.5-0.5b-instruct-q8_0.gguf https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf

# Start server in background with NUMA mirror mode
./build/bin/llama-server -m ./.devcontainer/qwen2.5-0.5b-instruct-q8_0.gguf --host 0.0.0.0 --numa mirror --port 8080 &

# Wait for server to start up
while ! curl -s http://localhost:8080/; do sleep 1; done

# Test chat completion API
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-0.5b-instruct",
    "messages": [{"role": "user", "content": "Hello! Can you respond with just a short greeting?"}],
    "max_tokens": 20,
    "temperature": 0.1
  }'

# Verify JSON response contains a sensible greeting (e.g., "Hello!" or similar), e.g.:
#  ` "message":{"role":"assistant","content":"Hello!"} `

# Kill background server
ps aux | grep llama-server | grep -v grep | awk '{print $2}' | xargs kill -9
```

## 🧠 Key Areas of Focus

### 1. NUMA Memory Management
**Files**: `ggml/src/ggml-cpu/ggml-cpu.c`, `src/llama-mmap.cpp`, `ggml/src/ggml-numa-coordinator.c`, `ggml/src/ggml-cpu-numa-buffer.cpp`

- **NUMA mirroring**: `tensor_data()` and `tensor_set_data()` in `ggml/src/ggml.h` to handle numa-aware tensor data access and mirror across nodes, and numa-aware cache mirroring in `ggml-cpu-numa-buffer.cpp`
- **Thread-to-NUMA mapping**: In the numa coordinator, each worker threadpool gets assigned to its own numa node.
- **Memory allocation**: ALWAYS use `numa_alloc_onnode()` for local memory/buffer allocation on the current numa node, or indeed ALL numa nodes. Remember, we're trying to achieve data parallelism! Each numa node needs its own local copy of everything!

### 2. CPU Topology Detection
**Files**: `common/common.cpp`, `common/common.h`, `ggml/src/ggml-numa-coordinator.c`

- **Linux-specific**: `common.cpp` strategy reads `/sys/devices/system/cpu/` 
- **Hyperthreading detection**: Groups sibling threads correctly
- **Intel hybrid support**: Distinguishes P-cores from E-cores
- **NUMA awareness**: Incorporates NUMA node information into thread scheduling in `ggml-numa-coordinator.c`


### 3. Command-Line Interface
**Files**: `common/arg.cpp`

New command-line arguments for `llama-server` should be added to the file above.

## 🔧 Development Workflow

### Making Changes

1. **Identify the area**: NUMA allocation, CPU detection, CLI args, etc.
2. **Use dev container**: Ensure consistent environment
3. **Build incrementally**: Use `cmake --build build --parallel` for faster iteration
4. **Test immediately**: Run relevant tests in `tests/` which are built as CMake tests and output in `./build/bin`
5. **Check compilation**: Use `get_errors` tool to validate syntax

### Common Edit Patterns

#### Modifying NUMA Logic
1. Check `ggml-cpu.c`, `ggml-numa-coordinator.c` for thread computation changes and numa node logic
2. Update `llama-mmap.cpp`, `ggml-cpu-numa-buffer.cpp` for memory allocation
3. Write functional tests in `tests/test-numa-coordinator.cpp`
4. Verify tests pass and no regressions

#### Modifying Dispatcher/Operation Logic and Adding New NUMA Parallelized Operations
1. Check the operations list in `ggml/src/ggml-cpu/ops.h`
2. Check the existing fallback operations in `ggml/src/ggml-cpu/ggml-numa-fallback.c`
3. Implement the new NUMA-parallelized operation in `ggml/src/ggml-cpu/ggml-numa-operation-dispatch.c`
4. Write functional tests in `tests/test-numa-dispatcher.cpp`
5. Write a mathematical correctness test that compares the fallback "mathematical kernel" vs the new NUMA parallelization in `tests/test-numa-mathematical-correctness.cpp`
6. Verify tests pass and no regressions

### Debugging Approach

```bash
# Strategy 1: Catch a segfault directly and do a backtrace with gdb:
gdb --batch --ex run --ex bt --ex quit --args ./build/bin/${testFileName}

# Strategy 2: Get a core dump and fetch a backtrace from gdb (useful for multithreading issues where a segfault can't be cleanly caught directly):

## First, enable core dumps to current folder
echo 'core' | sudo tee /proc/sys/kernel/core_pattern
ulimit -c unlimited

## Next, run the program to generate a coredump
./build/bin/llama-cli -m ./.devcontainer/qwen2.5-0.5b-instruct-q8_0.gguf -v -no-cnv -n 1 -p "Repeat after me: Hello, world!" --numa mirror

"Segmentation fault: core dumped"

## Now debug the core dump in the local folder
gdb --batch --ex "file ./build/bin/llama-cli" --ex "core-file ./core" --ex "bt" --ex "info threads" --ex "thread apply all bt" --ex quit
```

## 📝 Code Standards

### Error Handling
- Always check return values for system calls
- Use `LOG_WRN()` / `GGML_LOG_WARN()` for warnings, `LOG_ERR()` / `GGML_LOG_ERROR()` for errors, `GGML_ASSERT()` for assertions.

### Platform Compatibility
- NUMA features are Linux-specific (`#if defined(__x86_64__) && defined(__linux__)`)

### Testing Guidelines
CMake tests live in the `tests/` folder and are built into `build/bin/`. Build and run tests as above.

**Important:** ALWAYS add tests to the `tests/` folder, never to the project root!
**Important:** ALWAYS use the CMake build/test apparatus for compiling tests!
**Important:** ALWAYS run `./tests/run-numa-tests.sh` as the final validation step!

1. Write tests in one of the following two files:
   - `tests/test-numa-dispatcher.cpp`
   - `tests/test-numa-coordinator.cpp`

2. Tests are already added to the CMake build system in `/tests/CMakeLists.txt`, but for reference, the schema looks like this, in case you need to update includes:
    ```c
    # test-numa-dispatcher
    set(LLAMA_TEST_NAME test-numa-dispatcher)
    llama_build_and_test(test-numa-dispatcher.cpp)
    target_link_libraries(${LLAMA_TEST_NAME} PRIVATE ggml ggml-cpu common) # includes may differ
    ```

3. Configure CMake again so it picks up your test:
   `cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DGGML_NUMA_MIRROR=ON -DGGML_OPENMP=OFF`

4. Build with CMake:
   `cmake --build build --target test-numa-dispatcher`

5. Run the tests:
   `./build/bin/test-numa-coordinator`
   `./build/bin/test-numa-dispatcher`

6. Verify sane test output, no errors, no overflowing variables, no hanging/segfaults, etc.

7. **Final validation with comprehensive test suite**:
   ```bash
   ./tests/run-numa-tests.sh
   ```
   This script runs all NUMA tests and provides timing information. It MUST return exit code 0.

A feature isn't done until it has comprehensive, working tests in one of these files!

### Mathematical Correctness Test Template

**IMPORTANT: Use the provided template instead of writing test frameworks from scratch!**

For testing mathematical correctness of new NUMA operations, use the comprehensive template:

- **Template file**: `tests/test-numa-mathematical-correctness-template.cpp`
- **Working example**: `tests/test-numa-mathematical-correctness.cpp` (MUL_MAT implementation)

#### Template Usage Instructions:

1. **Copy the template**:
   ```bash
   cp tests/test-numa-mathematical-correctness-template.cpp tests/test-numa-mathematical-correctness-YOUR_OPERATION.cpp
   ```

2. **Customize for your operation**:
   - Replace all instances of `TEMPLATE_OPERATION` with your operation name (e.g., `ADD`, `MUL`, `CONV`, etc.)
   - Update test dimensions in `test_cases[]` array to match your operation's requirements
   - Implement `test_single_YOUR_OPERATION_case()` with:
     - Appropriate tensor creation for your operation
     - Deterministic test data generation
     - NUMA operation execution via `ggml_numa_intercept_operation`
     - Reference implementation (serial computation or direct mathematical kernel)
     - Result comparison using `compare_float_arrays()`

3. **Add to CMake** (if needed):
   ```cmake
   set(LLAMA_TEST_NAME test-numa-mathematical-correctness-YOUR_OPERATION)
   llama_build_and_test(test-numa-mathematical-correctness-YOUR_OPERATION.cpp)
   target_link_libraries(${LLAMA_TEST_NAME} PRIVATE ggml ggml-cpu common)
   ```

4. **Build and test**:
   ```bash
   cmake --build build --target test-numa-mathematical-correctness-YOUR_OPERATION
   ./build/bin/test-numa-mathematical-correctness-YOUR_OPERATION
   ```

#### Template Features:

- **Multi-dimensional testing**: Tests across TINY, SMALL, MEDIUM, LARGE tensor sizes
- **Multi-threading strategies**: Tests with 1, 2, 4, 6, 8 threads to verify coordinator behavior
- **Comprehensive error reporting**: Detailed mismatch information with absolute and relative error tracking
- **Modular design**: Easy to extend for new operations
- **Mathematical equivalence validation**: Strict comparison between NUMA parallel and serial reference implementations

#### Key Design Principles:

- Tests must be **deterministic and reproducible**
- Each operation should be tested across **multiple tensor dimensions**
- **Multiple thread strategies** verify coordinator execution behavior
- **Mathematical equivalence should be exact** (within floating-point tolerance)
- **Comprehensive error reporting** helps debug failures
- Use `compare_float_arrays()` utility for consistent comparison logic

**DO NOT write mathematical correctness test frameworks from scratch - always start with the provided template!**

## 🐛 Common Issues and Solutions

### Build Issues
```bash
# Missing dependencies
apt list --installed | grep -E "(numa|hwloc|cmake)"

# Clean build
rm -rf build && cmake -B build

# Verbose build output
cmake --build build --parallel --verbose
```

## 📚 Key Documentation Files

- `.devcontainer/changelog/*.md` - Changelog and comprehensive technical documentation
- `.devcontainer/changelog/2025-08-14-operation-analysis-big-bang-strategy.md` - Numa coordinator/dispatcher design guide and project plan
- `.devcontainer/README.md` - Dev container usage guide
- `docs/build.md` - Official build instructions

## 🧪 NUMA Test Suite

### Running Tests

**ALWAYS run the NUMA test suite after making changes** to validate your work:

```bash
./tests/run-numa-tests.sh
```

This comprehensive test suite runs all NUMA-related tests and provides detailed timing information:
- `test-numa-coordinator` - Basic coordinator functionality
- `test-numa-coordinator-wait` - Advanced coordinator operations and strategy validation  
- `test-numa-dispatcher` - Operation dispatch and routing
- `test-numa-mathematical-correctness` - Mathematical accuracy validation

### Test Requirements

**A change is NOT complete until ALL tests pass.** The test script:
- ✅ Returns exit code 0 if all tests pass
- ❌ Returns exit code 1 if any test fails
- 📊 Shows precise timing for performance monitoring
- 🔍 Provides detailed failure information

### Adding New Tests

When adding features, **you MUST add corresponding tests**:

1. **Add tests to existing files** in `tests/`:
   - `test-numa-coordinator.cpp` - For coordinator functionality
   - `test-numa-dispatcher.cpp` - For operation dispatch features
   - `test-numa-mathematical-correctness.cpp` - For mathematical validation

2. **Ensure tests fail properly**:
   ```cpp
   // ❌ Bad: void function can't signal failure
   void test_my_feature() {
       if (condition_fails) {
           printf("❌ Test failed!\n");
           return; // Test appears to pass!
       }
   }
   
   // ✅ Good: bool function properly signals failure
   bool test_my_feature() {
       if (condition_fails) {
           printf("❌ Test failed!\n");
           return false; // Test actually fails
       }
       return true;
   }
   ```

3. **Update main() functions** to call your new tests and check return values:
   ```cpp
   int main() {
       bool all_passed = true;
       all_passed &= test_existing_feature();
       all_passed &= test_my_new_feature(); // Add your test here
       
       if (all_passed) {
           printf("🎉 ALL TESTS PASSED!\n");
           return 0;
       } else {
           printf("💥 Some tests failed.\n");
           return 1; // Critical: return error code
       }
   }
   ```

4. **Test compilation** - tests are built automatically with CMake:
   ```bash
   cmake --build build --target test-numa-coordinator
   cmake --build build --target test-numa-dispatcher
   ```

### Test Design Guidelines

- **Mathematical Correctness**: Always validate that NUMA operations produce identical results to fallback implementations
- **Error Conditions**: Test both success and failure paths
- **Performance**: Include tests that validate performance improvements don't break correctness
- **Thread Safety**: Validate that multi-threaded operations work correctly
- **Memory Safety**: Ensure no memory leaks or buffer overflows

## 🎯 Success Criteria for Changes

1. **Builds successfully** in dev container
2. **No compilation errors** across all modified files
3. **Test coverage** for new features
4. **✅ ALL NUMA TESTS PASS** - Run `./tests/run-numa-tests.sh` and verify exit code 0
5. **No failing tests** in `tests/` after changes

## 💡 Tips for AI Agents

1. **Always use the dev container** - it has all dependencies and correct environment
2. **Run tests FIRST and LAST** - `./tests/run-numa-tests.sh` before starting and after finishing
3. **Test incrementally** - build and test after each significant change
4. **Add tests for new features** - a feature isn't done until it has working tests
5. **Check multiple scenarios** - different thread counts, NUMA configurations
6. **Read existing code carefully** - NUMA and threading logic is subtle
7. **Check for platform-specific code** - many features are Linux-only
8. **Validate with real tests** - not just compilation success
9. **Check exit codes** - tests must return proper error codes on failure

### Critical Workflow:
```bash
# 1. Start with passing tests
./tests/run-numa-tests.sh && echo "✅ Starting with clean state"

# 2. Make your changes
# ... implement features ...

# 3. Build incrementally
cmake --build build --parallel

# 4. Test frequently during development
./build/bin/test-numa-coordinator
./build/bin/test-numa-dispatcher

# 5. Final validation - ALL tests must pass
./tests/run-numa-tests.sh && echo "🎉 Change is complete!" || echo "❌ Fix failures before finishing"
```

Remember: NUMA and CPU topology changes can have subtle effects. Always validate performance and correctness thoroughly before considering changes complete.

## 📋 Quick Reference: NUMA Operation Implementation

### Essential Files for Operation Implementation
```
ggml/src/ggml-cpu/ggml-numa-operation-dispatch.c  # Add operation handler
ggml/src/ggml-cpu/ggml-cpu.c                      # Find mathematical kernels
tests/test-numa-mathematical-correctness-*.cpp    # Create correctness tests
tests/CMakeLists.txt                              # Add test targets
tests/run-numa-tests.sh                           # Update test runner
```

### Standard Work Function Template
```c
static int ggml_numa_work_function_OPERATION_chunk(void* context) {
    struct ggml_numa_context* numa_context = (struct ggml_numa_context*)context;
    struct ggml_tensor* operation = numa_context->operation;
    
    // Extract operation parameters and tensor slicing logic
    // Call existing mathematical kernel or implement NUMA-optimized version
    // Return 0 for success, non-zero for failure
}
```

### Dispatcher Handler Template
```c
case GGML_OP_YOUR_OPERATION: {
    efficiency = 0.95f;  // Adjust based on operation type
    strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL;
    work_function = ggml_numa_work_function_your_operation_chunk;
    break;
}
```

### Test Implementation Checklist
- [ ] Copy template: `cp tests/test-numa-mathematical-correctness-template.cpp tests/test-numa-mathematical-correctness-OPERATION.cpp`
- [ ] Update class name and operation references
- [ ] Define operation-specific test cases and dimensions
- [ ] Implement tensor creation and reference computation
- [ ] Add CMake target in `tests/CMakeLists.txt`
- [ ] Update `tests/run-numa-tests.sh` test list
- [ ] Verify all tests pass: `./tests/run-numa-tests.sh`

### Performance Validation Commands
```bash
# Build specific test
cmake --build build --target test-numa-mathematical-correctness-OPERATION

# Run single test
./build/bin/test-numa-mathematical-correctness-OPERATION

# Full test suite
./tests/run-numa-tests.sh

# Real model validation
./build/bin/llama-server -m model.gguf --numa mirror --port 8080
```

## Changelog

After each task is complete, document what you did in a new markdown file in `.devcontainer/changelog`. Always look up the current date and include that in the filename. You can also search through markdown files in this folder for records of other similar past changes if you want to check past actions for more context on a task.