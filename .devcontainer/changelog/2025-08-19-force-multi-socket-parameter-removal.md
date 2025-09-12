# Force Multi-Socket Parameter Removal - 2025-08-19

## Overview
User requested complete removal of the `force_multi_socket` parameter from the NUMA coordinator system and all related test infrastructure. This parameter was originally intended for testing multi-socket behavior on single-node systems but was identified as a potential source of bugs in production.

## Changes Made

### Core System Updates
- **ggml-numa-coordinator.h**: Removed `force_multi_socket` parameter from function signatures
  - `ggml_numa_coordinator_manager_new(int n_threads)` (removed bool parameter)
  - `ggml_numa_coordinator_manager_get_global(int n_threads)` (removed bool parameter)

- **ggml-numa-coordinator.c**: Comprehensive removal of force_multi_socket logic
  - Removed `bool force_multi_socket` field from `struct ggml_numa_coordinator_manager`
  - Removed logic that forced 2 simulated NUMA nodes when `force_multi_socket=true && !numa_is_available`
  - Removed parameter from all function signatures and calls
  - Simplified coordinator initialization to only work with real NUMA hardware

- **ggml.h**: Removed `bool force_multi_socket` field from `struct ggml_threadpool_params`

- **ggml.c**: 
  - Removed `force_multi_socket` initialization in `ggml_threadpool_params_init()`
  - Removed `force_multi_socket` comparison in `ggml_threadpool_params_match()`

### NUMA Work Files Updates
Updated all extern declarations in NUMA work files to remove force_multi_socket parameter:
- `ggml-numa-mulmat.c`
- `ggml-numa-add.c` 
- `ggml-numa-rms-norm.c`
- `ggml-numa-flash-attn-ext.c`

### Operation Dispatch Updates
- **ggml-numa-operation-dispatch.c**: Updated all coordinator function calls to remove force_multi_socket parameter

### NUMA Strategy Cleanup
- **ggml-cpu.h**: Removed `GGML_NUMA_STRATEGY_MIRROR_FORCE = 5` enum value
- **ggml-cpu.c**: 
  - Removed MIRROR_FORCE from strategy switch cases
  - Updated `ggml_numa_should_dispatch()` to only enable for `GGML_NUMA_STRATEGY_MIRROR` on real NUMA systems
- **common/arg.cpp**: 
  - Removed `--numa mirror-force` command-line option
  - Updated help text to remove mirror-force documentation
- **common/common.cpp**: Removed MIRROR_FORCE cases from strategy name mapping and descriptions
- **src/llama-mmap.cpp**: Removed MIRROR_FORCE strategy handling and simplified single-node validation

### Test File Updates
Updated all NUMA tests to use real NUMA MIRROR strategy instead of force_multi_socket simulation:

#### Mathematical Correctness Tests
- **test-numa-mathematical-correctness-matmul.cpp**:
  - Added `ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR)` to main()
  - Updated coordinator call: `ggml_numa_coordinator_manager_get_global(8)` (removed bool parameter)
  - Changed log messages from "FORCE_MULTI_SOCKET" to "NUMA MIRROR"

- **test-numa-mathematical-correctness-rope.cpp**: Same updates as matmul
- **test-numa-mathematical-correctness-soft-max.cpp**: Same updates as matmul  
- **test-numa-mathematical-correctness-add.cpp**: Same updates as matmul
- **test-numa-mathematical-correctness-rms-norm.cpp**: Same updates as matmul

#### Coordinator Tests  
- **test-numa-coordinator-wait.cpp**:
  - Removed `params.force_multi_socket = true` assignment
  - Updated all 5 extern declarations to remove force_multi_socket parameter
  - Updated all 5 coordinator function calls to remove bool parameter

#### Dispatcher Tests
- **test-numa-dispatcher.cpp**:
  - Removed `tpp.force_multi_socket = true` assignment
  - Updated initialization to use regular MIRROR strategy

#### Timing Tests
- **test-numa-parallel-execution-timing.cpp**:
  - Replaced `setenv("GGML_NUMA_FORCE_MULTI_SOCKET", "1", 1)` with `ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR)`

#### Test Utils
- **numa-test-utils.h**:
  - Removed `bool force_multi_socket` field from `TestConfig` struct
  - Updated `default_test_config()` to not set force_multi_socket

## Impact Assessment

### Benefits
- **Simplified Architecture**: Removed testing-only parameter that was potentially causing production bugs
- **Real NUMA Focus**: Tests now run against actual NUMA hardware instead of simulated multi-socket behavior
- **Cleaner API**: Function signatures are simpler without the bool parameter
- **Production Safety**: Eliminates confusion between test mode and production mode

### Behavior Changes
- **NUMA Coordinator**: Now only activates on systems with real NUMA hardware when using MIRROR strategy
- **Tests**: All tests now require real multi-NUMA systems to properly test coordinator functionality
- **Command Line**: `--numa mirror-force` option no longer available (only `mirror` remains)
- **Single-Node Systems**: MIRROR strategy will properly error instead of falling back to simulation

## Compilation Status
✅ All targets compile successfully:
- Core libraries (ggml-base, ggml-cpu)
- Main binaries (llama-server) 
- Test binaries (test-numa-mathematical-correctness-matmul, test-numa-coordinator-wait)

## Validation Required
- **Real NUMA Testing**: All tests should be run on actual multi-NUMA hardware
- **Single-Node Validation**: Verify MIRROR strategy properly errors on single-node systems
- **Performance Testing**: Ensure removal doesn't impact real NUMA performance

## Files Modified (24 total)
**Core System (8 files)**:
- ggml/src/ggml-cpu/ggml-numa-coordinator.h
- ggml/src/ggml-cpu/ggml-numa-coordinator.c  
- ggml/include/ggml.h
- ggml/src/ggml.c
- ggml/include/ggml-cpu.h
- ggml/src/ggml-cpu/ggml-cpu.c
- ggml/src/ggml-cpu/ggml-numa-operation-dispatch.c
- src/llama-mmap.cpp

**NUMA Work Files (4 files)**:
- ggml/src/ggml-cpu/numa-work/ggml-numa-mulmat.c
- ggml/src/ggml-cpu/numa-work/ggml-numa-add.c
- ggml/src/ggml-cpu/numa-work/ggml-numa-rms-norm.c  
- ggml/src/ggml-cpu/numa-work/ggml-numa-flash-attn-ext.c

**CLI/Common (2 files)**:
- common/arg.cpp
- common/common.cpp

**Test Files (10 files)**:
- tests/test-numa-mathematical-correctness-matmul.cpp
- tests/test-numa-mathematical-correctness-rope.cpp
- tests/test-numa-mathematical-correctness-soft-max.cpp
- tests/test-numa-mathematical-correctness-add.cpp
- tests/test-numa-mathematical-correctness-rms-norm.cpp
- tests/test-numa-coordinator-wait.cpp
- tests/test-numa-dispatcher.cpp
- tests/test-numa-parallel-execution-timing.cpp
- tests/numa-test-utils.h

This represents a significant cleanup of the NUMA coordinator system, removing test-oriented complexity and focusing on production-ready real NUMA behavior.
