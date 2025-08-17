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

# Verify JSON response contains a sensible greeting (e.g., "Hello!" or similar)
# Kill background server
kill %1
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

## Changelog

After each task is complete, document what you did in a new markdown file in `.devcontainer/changelog`. Always look up the current date and include that in the filename. You can also search through markdown files in this folder for records of other similar past changes if you want to check past actions for more context on a task.