# NUMA Distribute Strategy Removal

## Summary
Removed the `--numa distribute` strategy as it duplicated default non-NUMA behavior without providing additional value. The remaining NUMA strategies (`isolate`, `mirror`, `numactl`) provide distinct functionality.

## Changes Made

### 1. Enum Definition
**File**: `ggml/include/ggml-cpu.h`
- Removed `GGML_NUMA_STRATEGY_DISTRIBUTE = 1` from enum
- Added comment explaining removal to maintain historical context
- Preserved enum numbering to avoid ABI breaks

### 2. Command Line Interface  
**File**: `common/arg.cpp`
- Removed "distribute" from `--numa` argument parsing
- Updated help text to remove distribute option description
- Changed default behavior: empty `--numa` now defaults to isolate mode instead of distribute
- Updated `--numa-cache-strategy` description to reference mirror mode instead of distribute mode

### 3. Implementation Logic
**Files**: `common/common.cpp`, `ggml/src/ggml-cpu/ggml-cpu.c`, `src/llama-mmap.cpp`
- Removed all `GGML_NUMA_STRATEGY_DISTRIBUTE` case statements from switch blocks
- Simplified conditional logic by removing distribute-specific branches
- Updated memory mapping logic to remove distribute mode handling

### 4. Tools and Tests
**Files**: `tools/llama-bench/llama-bench.cpp`, `tests/test-numa-coordinator.cpp`
- Updated llama-bench to remove distribute option, default to isolate
- Updated test initialization to use mirror strategy instead of distribute
- Added mirror option to llama-bench for completeness

## Validation

### Command Line Behavior
```bash
# ❌ No longer accepted (correctly rejects)
./build/bin/llama-server --numa distribute
# error while handling argument "--numa": invalid value

# ✅ Still work correctly
./build/bin/llama-server --numa mirror
./build/bin/llama-server --numa "isolate 1"  
./build/bin/llama-server --numa numactl
```

### Help Text
Updated help now shows only meaningful NUMA strategies:
```
--numa TYPE                             attempt optimizations that help on some NUMA systems
                                        - isolate: only spawn threads on CPUs on the node that execution started on
                                        - isolate N: only spawn threads on CPUs on NUMA node N (if valid)
                                        - numactl: use the CPU map provided by numactl
                                        - mirror: enable coordinator data parallelism with NUMA-aware KV cache
```

## Rationale

The `distribute` strategy was redundant because:
1. **No unique functionality**: It behaved identically to the default threading without NUMA awareness
2. **Confusion potential**: Users might expect special NUMA-aware distribution that didn't exist
3. **Code complexity**: Maintaining separate code paths for equivalent behavior
4. **User clarity**: Remaining options (`isolate`, `mirror`, `numactl`) have distinct, useful purposes

## Remaining NUMA Strategies

- **`isolate`**: Restricts execution to current or specified NUMA node - useful for memory locality
- **`mirror`**: Enables NUMA coordinator with data parallelism and mirroring - performance optimization
- **`numactl`**: Respects external numactl CPU mapping - integration with system tools

## Build Status
- ✅ Core components compile successfully
- ⚠️ Some test files need updates for coordinator API changes (separate issue)
- ✅ Command line parsing works correctly
- ✅ Functional validation passes
