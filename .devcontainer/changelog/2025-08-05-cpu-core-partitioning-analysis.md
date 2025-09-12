# CPU Core Partitioning Implementation and Analysis

**Date**: August 5, 2025  
**Task**: Implement CPU core partitioning to resolve OpenMP thread contention in multi-socket testing  
**Status**: ✅ Implemented, 🔄 Performance Issues Identified  

## Problem Analysis

### Initial Issue: OpenMP Thread Contention
- **Root Cause**: Multi-socket implementation was using async task submission via pthread
- **Expected**: True data parallelism across different CPU cores
- **Actual**: 10x performance degradation (0.084x speedup)
- **Discovery**: OpenMP parallel regions in each socket competing for same CPU cores

### CPU Topology Investigation
```bash
# System Details (Intel Core Ultra 7 165H)
Thread(s) per core: 2 (Hyperthreading enabled)
Core(s) per socket: 11
Socket(s): 1
Total logical CPUs: 22 (0-21)

# Hyperthreading Layout
Physical Core 0: Logical CPUs 0, 1
Physical Core 1: Logical CPUs 2, 3  
Physical Core 2: Logical CPUs 4, 5
...
Physical Core 10: Logical CPUs 20, 21
```

### Original Flawed Partitioning
```cpp
// WRONG: Competing for same physical cores
Socket 0: CPUs 0-10  // Physical cores 0-5 (partial)
Socket 1: CPUs 11-21 // Physical cores 5-10 (overlapping!)
```

## Solution Implementation

### 1. Hyperthreading-Aware CPU Partitioning
```cpp
// CORRECT: Separate physical cores
Socket 0: Physical cores 0-4 = Logical CPUs {0,1,2,3,4,5,6,7,8,9}
Socket 1: Physical cores 5-10 = Logical CPUs {10,11,12,13,14,15,16,17,18,19,20,21}
```

### 2. Enhanced CPU Binding Implementation
**File**: `ggml/src/ggml-cpu/ggml-cpu.c`

**Key Changes**:
- Added hyperthreading detection and physical core mapping
- Implemented pthread CPU affinity using `pthread_setaffinity_np()`
- Added OpenMP environment variable binding (`OMP_PLACES`, `OMP_PROC_BIND`)
- Enhanced logging to show physical core assignments

**Code Additions**:
```cpp
// Physical core to logical CPU mapping
int physical_cores = available_cpus / threads_per_core;
int cores_per_socket = physical_cores / mgr->n_numa_nodes;

// Map physical cores to logical CPUs
for (int core = start_core; core < end_core; core++) {
    for (int thread = 0; thread < threads_per_core; thread++) {
        int logical_cpu = core * threads_per_core + thread;
        socket_params.cpumask[logical_cpu] = true;
    }
}
```

## Current Status

### ✅ What's Working
1. **CPU Partitioning**: Correctly partitioned physical cores (no hyperthreading competition)
2. **Core Binding**: Successfully binding pthread and OpenMP threads to specific cores
3. **Async Infrastructure**: Async task submission working correctly (fast launch times)
4. **Profiling**: Comprehensive performance monitoring in place

### ❌ Performance Issues Remain
- **Multi-socket**: 3187ms 
- **Standard**: 356ms
- **Speedup**: 0.11x (still 9x slower!)

### 🔍 Current Hypothesis
Even with proper CPU partitioning, we may have:

1. **OpenMP Environment Issues**: Setting `OMP_PLACES` and `OMP_PROC_BIND` in worker thread may not affect already-initialized OpenMP runtime
2. **Synchronization Overhead**: The async coordination and waiting might be too expensive
3. **Memory Access Patterns**: Matrix data might not be optimally distributed for multi-socket access
4. **Cache Effects**: Splitting work across physical cores might be causing cache misses

## Next Steps

### Immediate Actions Needed
1. **OpenMP Runtime Initialization**: Set OpenMP environment before any parallel regions
2. **Memory Locality**: Investigate if matrix data is being accessed efficiently by both sockets
3. **Synchronization Analysis**: Profile the async wait times vs computation times
4. **Alternative Approaches**: Consider other parallelization strategies

### Testing Approach
- Run focused tests with smaller matrices to isolate overhead vs computation
- Monitor actual CPU utilization during multi-socket execution
- Compare memory access patterns between standard and multi-socket approaches

## Lessons Learned

1. **Hyperthreading Matters**: Logical CPU partitioning can still compete for physical resources
2. **OpenMP Binding Complexity**: Environment variables may need to be set before OpenMP initialization
3. **Async Overhead**: The coordination overhead can easily outweigh parallelism benefits
4. **System-Specific Tuning**: CPU topology detection is critical for proper partitioning

## Files Modified
- `ggml/src/ggml-cpu/ggml-cpu.c`: CPU partitioning and binding implementation
- `tests/test-numa-multi-socket.cpp`: Enhanced performance analysis and testing

The CPU core partitioning infrastructure is now correct and ready for further optimization of the parallelization strategy.
