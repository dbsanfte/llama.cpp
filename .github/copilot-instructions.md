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
```

Note: Never limit the threads count of `--parallel`, just let cmake autodetect the number of cores and choose the max threadcount itself.

## 🧠 Key Areas of Focus

### 1. NUMA Memory Management
**Files**: `ggml/src/ggml-cpu.c`, `src/llama-mmap.cpp`

- **NUMA mirroring**: Model weights duplicated across NUMA nodes
- **Thread-to-NUMA mapping**: Each thread accesses local memory
- **Memory allocation**: `numa_alloc_onnode()` for local allocation

### 2. CPU Topology Detection
**Files**: `common/common.cpp`, `common/common.h`

- **Linux-specific**: Reads `/sys/devices/system/cpu/` topology
- **Hyperthreading detection**: Groups sibling threads correctly
- **Intel hybrid support**: Distinguishes P-cores from E-cores

Key functions:
```cpp
detect_cpu_topology()           // Main topology detection
cpu_count_math_cpus()          // Count available CPUs with options
cpu_print_topology_info()     // Debug information display
```

### 3. Command-Line Interface
**Files**: `common/arg.cpp`

New command-line arguments for `llama-server` should be added to the file above.

## 🔧 Development Workflow

### Making Changes

1. **Identify the area**: NUMA allocation, CPU detection, CLI args, etc.
2. **Use dev container**: Ensure consistent environment
3. **Build incrementally**: Use `cmake --build build --parallel` for faster iteration
4. **Test immediately**: Run `./build/bin/llama-server --cpu-topology` after changes
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
- Use `LOG_WRN()` for warnings, `LOG_ERR()` for errors, `GGML_ASSERT()` for assertions.

### Platform Compatibility
- NUMA features are Linux-specific (`#if defined(__x86_64__) && defined(__linux__)`)
- Provide fallbacks for other platforms
- Test Windows compatibility doesn't break

### Testing Guidelines
1. Unit tests live in the `tests/` folder
2. Write tests with the Arrange, Act, Assert pattern
2. Ensure 90%+ coverage for new features
3. Run tests like this:
    ```bash
      set -e
      rm -rf build-ci-debug && mkdir build-ci-debug && cd build-ci-debug
      CMAKE_EXTRA="-DLLAMA_FATAL_WARNINGS=ON -DLLAMA_CURL=ON"
      time cmake -DCMAKE_BUILD_TYPE=Debug ${CMAKE_EXTRA} ..  2>&1 
      time make -j$(nproc) 2>&1 
      time ctest --list --output-on-failure 2>&1
    ```

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
- `docs/build.md` - Official build instructions
- `build-numa.sh` - Automated build and test script

## 🎯 Success Criteria for Changes

1. **Builds successfully** in dev container
2. **No compilation errors** across all modified files
3. **Unit test coverage** for new features
3. **No failing unit tests** after changes

## 💡 Tips for AI Agents

1. **Always use the dev container** - it has all dependencies and correct environment
2. **Test incrementally** - build and test after each significant change
3. **Check multiple scenarios** - different thread counts, NUMA configurations
4. **Read existing code carefully** - NUMA and threading logic is subtle
5. **Use the build script** - `./build-numa.sh` provides comprehensive testing
6. **Check for platform-specific code** - many features are Linux-only
7. **Validate with real workloads** - not just compilation success

Remember: NUMA and CPU topology changes can have subtle effects. Always validate performance and correctness thoroughly before considering changes complete.

## Changelog

After each task is complete, document what you did in a new markdown file in `.devcontainer/changelog`. You can also search through markdown files in this folder for records of other similar past changes if you want to check past actions for more context on a task.