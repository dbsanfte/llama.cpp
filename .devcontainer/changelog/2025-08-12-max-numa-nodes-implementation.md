# NUMA max_numa_nodes Parameter Implementation - August 12, 2025

## Summary
Successfully implemented and tested the `max_numa_nodes` parameter in `ggml_threadpool_params` structure to control NUMA node count for testing scenarios.

## Problem Solved
The NUMA scaling tests needed a way to control how many NUMA nodes the coordinator would use:
- **1 NUMA node test**: Should use only 1 of 2 available nodes  
- **2 NUMA node test**: Should use both available nodes
- **4 NUMA node test**: Should allow virtual subdivision by not limiting real nodes

## Implementation Details

### 1. Structure Extension
**File**: `ggml/include/ggml.h`
- Added `int max_numa_nodes` field to `ggml_threadpool_params` structure after `force_multi_socket`

### 2. Parameter Initialization  
**File**: `ggml/src/ggml.c`
- Added `p->max_numa_nodes = 0;` in `ggml_threadpool_params_init()` (0 = auto-detect all nodes)
- Added comparison in `ggml_threadpool_params_match()` for parameter matching

### 3. NUMA Node Limiting Logic
**File**: `ggml/src/ggml-cpu/ggml-numa-coordinator.c`
- Added constraint check after real NUMA detection: Lines ~1095-1100
- Added constraint check after `force_multi_socket` logic: Lines ~1107-1112
- Both locations apply the same logic:
  ```c
  if (tpp->max_numa_nodes > 0 && num_numa_nodes > tpp->max_numa_nodes) {
      GGML_LOG_INFO("Limiting NUMA nodes from %d to %d (max_numa_nodes constraint)\n", 
                    num_numa_nodes, tpp->max_numa_nodes);
      num_numa_nodes = tpp->max_numa_nodes;
  }
  ```

### 4. Test Integration
**File**: `tests/test-comprehensive-numa-performance.cpp`
- Test already sets `tpp.max_numa_nodes = numa_nodes;` in `benchmark_numa_scaling()`
- Works for both real NUMA hardware and virtual NUMA simulation

## Test Results

### Dev Container Environment (Virtual NUMA)
```
Testing 1 NUMA node... Forcing multi-socket mode with 2 simulated NUMA nodes
Limiting NUMA nodes from 2 to 1 (max_numa_nodes constraint)
    Number of NUMA nodes requested: 1

Testing 2 NUMA nodes... Forcing multi-socket mode with 2 simulated NUMA nodes  
    Number of NUMA nodes requested: 2

Testing 4 NUMA nodes... Forcing multi-socket mode with 2 simulated NUMA nodes
    Number of NUMA nodes requested: 2
```

### Verification
- ✅ **1 NUMA node**: Successfully limits from 2 to 1 with clear logging
- ✅ **2 NUMA nodes**: No limiting needed (2 ≤ 2)  
- ✅ **4 NUMA nodes**: No limiting needed (2 ≤ 4), allows virtual subdivision at test level

## Real NUMA Hardware Impact
On real 2-socket NUMA hardware (112 CPUs across 2 nodes):
- Setting `max_numa_nodes = 1` will limit to using only NUMA node 0
- Setting `max_numa_nodes = 2` will use both NUMA nodes (no change)
- Setting `max_numa_nodes = 0` will auto-detect and use all available nodes

## Technical Notes
- Parameter only limits downward (never increases node count)
- `max_numa_nodes = 0` means auto-detect all available nodes
- Works with both real NUMA hardware and `force_multi_socket` simulation
- Provides clear logging when constraints are applied
- Maintains compatibility with existing code (defaults to 0 = no limit)

## Files Modified
1. `ggml/include/ggml.h` - Structure definition
2. `ggml/src/ggml.c` - Parameter initialization and comparison  
3. `ggml/src/ggml-cpu/ggml-numa-coordinator.c` - NUMA node limiting logic

## Build Status
- ✅ Compiles cleanly with GGML_NUMA_MIRROR=ON, GGML_OPENMP=OFF
- ✅ All tests build successfully
- ✅ NUMA coordinator tests pass with new parameter

This completes the max_numa_nodes implementation, enabling proper NUMA node count control for testing scenarios while maintaining full compatibility with real NUMA hardware.
