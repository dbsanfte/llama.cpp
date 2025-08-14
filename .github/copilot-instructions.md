# Copilot Instructions for llama.cpp

This document provides instructions for AI assistants (GitHub Copilot, Claude, etc.) working on the llama.cpp project with NUMA improvements and development container setup.

## 🎯 Project Overview

This is a fork of llama.cpp with **NUMA-aware improvements** for better CPU threading and memory allocation. The project includes:

- **NUMA-aware Coordinator and Node/Thread assignment** - `ggml/src/ggml-cpu/ggml-numa-coordinator.c`
- **Work-in-Progress dispatcher to the Coordinator** - `ggml/src/ggml-cpu/ggml-numa-operation-dispatch.c`
- **Development container** - Ubuntu 24.04 with all dependencies for consistent building

### Goal

Our goal is to implement a NUMA-aware scheduling and execution model for the llama.cpp project, improving performance on multi-socket systems. Our main consumer will be `src/llama-context.cpp` via `ggml/src/ggml-cpu/ggml-cpu.c`.

We must implement the complete 193-item set of arithmetic operations in `ggml/src/ggml-cpu/ops.h` in our dispatcher.

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
# Test the app against a real model for sanity, should exit with code 0
wget -c -O ./.devcontainer/qwen2.5-0.5b-instruct-q8_0.gguf https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf
./build/bin/llama-cli -m ./.devcontainer/qwen2.5-0.5b-instruct-q8_0.gguf -v -no-cnv -n 1 -p "Repeat after me: Hello, world!" || echo "failed!"
```

## 🧠 Key Areas of Focus

### 1. NUMA Memory Management
**Files**: `ggml/src/ggml-cpu/ggml-cpu.c`, `src/llama-mmap.cpp`, `ggml/src/ggml-numa-coordinator.c`, `ggml/src/ggml-cpu-numa-buffer.cpp`

- **NUMA mirroring**: `tensor_data()` and `tensor_set_data()` in `ggml/src/ggml.h` to handle numa-aware tensor data access and mirror across nodes, and numa-aware cache mirroring in `ggml-cpu-numa-buffer.cpp`
- **Thread-to-NUMA mapping**: In the numa coordinator, each worker threadpool gets assigned to its own numa node.
- **Memory allocation**: NEVER use `malloc()` to allocate memory/buffers. ALWAYS use `numa_alloc_onnode()` for local allocation on the current numa node.

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

#### Modifying Dispatcher/Operation Logic
1. Check the operations list in `ggml/src/ggml-cpu/ops.h`
2. Implement the operation in `ggml/src/ggml-cpu/ggml-numa-operation-dispatch.c`
3. Write functional tests in `tests/test-numa-dispatcher.cpp`
4. Verify tests pass and no regressions

### Debugging Approach

```bash
# Troubleshoot a segfault with gdb
gdb --batch --ex run --ex bt --ex quit --args ./build/bin/${testFileName}
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

A feature isn't done until it has comprehensive, working tests in one of these files!

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

## 🎯 Success Criteria for Changes

1. **Builds successfully** in dev container
2. **No compilation errors** across all modified files
3. **Test coverage** for new features
4. **No failing tests** in `tests/` after changes

## 💡 Tips for AI Agents

1. **Always use the dev container** - it has all dependencies and correct environment
2. **Test incrementally** - build and test after each significant change
3. **Check multiple scenarios** - different thread counts, NUMA configurations
4. **Read existing code carefully** - NUMA and threading logic is subtle
5. **Check for platform-specific code** - many features are Linux-only
6. **Validate with real tests** - not just compilation success

Remember: NUMA and CPU topology changes can have subtle effects. Always validate performance and correctness thoroughly before considering changes complete.

## Changelog

After each task is complete, document what you did in a new markdown file in `.devcontainer/changelog`. Always look up the current date and include that in the filename. You can also search through markdown files in this folder for records of other similar past changes if you want to check past actions for more context on a task.