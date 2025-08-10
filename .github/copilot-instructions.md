# Copilot Instructions for llama.cpp

This document provides instructions for AI assistants (GitHub Copilot, Claude, etc.) working on the llama.cpp project with NUMA improvements and development container setup.

## 🎯 Project Overview

This is a fork of llama.cpp with **NUMA-aware improvements** for better CPU threading and memory allocation. The project includes:

- **Fixed NUMA thread assignment** - Proper CPU topology detection instead of naive modulo arithmetic
- **Configurable hyperthreading** - Default enabled, user can disable with `--cpu-no-hyperthreading`
- **Intel hybrid CPU support** - Detects P-cores vs E-cores
- **Development container** - Ubuntu 24.04 with all dependencies for consistent building

## 🏗️ Build Environment Setup

### Primary Development Method: Dev Container

**Always prefer the dev container for consistency**:

1. **Check if in container**: Look for `/.dockerenv` or check environment
2. **Start container**: If in VS Code, use "Dev Containers: Reopen in Container"
3. **Dependencies included**: All NUMA tools, build tools, debugging tools pre-installed

### Quick Build Commands

```bash
# Debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DGGML_NUMA_MIRROR=ON -DGGML_OPENMP=ON
# Or equivalently: -DGGML_NUMA=ON (both flags are synonyms)
cmake --build build --parallel

# Run tests
ctest --list --output-on-failure

# Troubleshoot a segfault
gdb --batch --ex run --ex bt --ex quit --args ./build/bin/${testFileName}
```

Note: Never limit the threads count of `--parallel`, just let cmake autodetect the number of cores and choose the max threadcount itself.

## 🧠 Key Areas of Focus

### 1. NUMA Memory Management
**Files**: `ggml/src/ggml-cpu.c`, `src/llama-mmap.cpp`, `ggml/src/ggml-numa-coordinator.c`, `ggml/src/ggml-cpu-numa-buffer.cpp`

- **NUMA mirroring**: `tensor_data()` and `tensor_set_data()` in `ggml/src/ggml.h` to handle numa-aware tensor data access and mirror across nodes, and numa-aware cache mirroring in `ggml-cpu-numa-buffer.cpp`
- **Thread-to-NUMA mapping**: In the numa coordinator, each worker threadpool gets assigned to its own numa node
- **Memory allocation**: `numa_alloc_onnode()` for local allocation

### 2. CPU Topology Detection
**Files**: `common/common.cpp`, `common/common.h`, `ggml/src/ggml-numa-coordinator.c`

- **Linux-specific**: `common.cpp` strategy reads `/sys/devices/system/cpu/` 
- **Hyperthreading detection**: Groups sibling threads correctly
- **Intel hybrid support**: Distinguishes P-cores from E-cores
- **NUMA awareness**: Incorporates NUMA node information into thread scheduling in `ggml-numa-coordinator.c`

Key functions in `common`:
```cpp
detect_cpu_topology()           // Main topology detection
cpu_count_math_cpus()          // Count available CPUs with options
cpu_print_topology_info()     // Debug information display
```

Key functions in `ggml`:
```cpp

```

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

#### Adding New CPU Parameters
1. Update `cpu_params` struct in `common/common.h`
2. Add argument parsing in `common/arg.cpp`
3. Update `cpu_count_math_cpus()` logic in `common/common.cpp`
4. Test with `--cpu-topology` flag

#### Modifying NUMA Logic
1. Check `ggml-cpu.c` for thread computation changes
2. Update `llama-mmap.cpp` for memory allocation
3. Test on multi-NUMA system or simulate with `numactl`

#### CLI Changes
1. Add/modify arguments in `common/arg.cpp`
2. Update help text and descriptions
3. Test argument parsing with `--help`

### Debugging Approach

```bash
# Debug build for better symbols
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# Use GDB with VS Code integration
# Set breakpoints in VS Code, use "Debug llama-server" launch config

# Monitor system calls
strace -e sched_setaffinity,numa_alloc_onnode ./build/bin/llama-server 

# Check CPU affinity assignment
taskset -cp $(pgrep llama-server)
```

## 📝 Code Standards

### Error Handling
- Always check return values for system calls
- Use `LOG_WRN()` / `GGML_LOG_WARNING()` for warnings, `LOG_ERR()` / `GGML_LOG_ERROR()` for errors, `GGML_ASSERT()` for assertions.

### Platform Compatibility
- NUMA features are Linux-specific (`#if defined(__x86_64__) && defined(__linux__)`)
- Provide fallbacks for other platforms
- Test Windows compatibility doesn't break

### Testing Guidelines
CMake tests live in the `tests/` folder and are built into `build/bin/`. 

Important: ALWAYS add tests to the `tests/` folder, never to the project root. Important: ALWAYS use the CMake test apparatus for testing.

1. Write tests for your new features and add the `test-feature-name.cpp` to `tests/`
2. Add the test to the end of `/tests/CMakeLists.txt`:
    ```c
    # test-my-feature
    set(LLAMA_TEST_NAME test-my-feature)
    llama_build_and_test(test-my-feature.cpp)
    target_link_libraries(${LLAMA_TEST_NAME} PRIVATE ggml ggml-cpu common) # includes may differ
    ```
3. Configure CMake again so it picks up your test:
   `cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DGGML_NUMA_MIRROR=ON`
4. Build with CMake:
   `cmake --build build --target test-my-feature`
5. Run the test:
   `./build/bin/test-my-feature`

A feature isn't done until it has comprehensive, working tests!

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

- `NUMA_IMPROVEMENTS.md` - Comprehensive technical documentation
- `.devcontainer/README.md` - Dev container usage guide
- `.devcontainer/changelog/` - Change log folder for development tasks
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