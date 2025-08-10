# NUMA Scaling Test Enhancement: Constant Thread Count Implementation

**Date**: August 10, 2025  
**Task**: Implement constant thread count allocation across NUMA configurations for better scaling comparison  
**Files Modified**: `/tests/test-comprehensive-numa-performance.cpp`  
**Test**: `test-comprehensive-numa-performance --quick`

## Problem Statement

The original NUMA scaling test implementation used variable thread counts across different NUMA configurations, making it difficult to analyze the effectiveness of the NUMA coordinator itself. The user requested a constant thread count approach where:

- **1 NUMA node**: 4 cores (2 physical + 2 hyperthreaded pairs)
- **2 NUMA nodes**: 8 cores (4 physical + 4 hyperthreaded pairs) 
- **4 NUMA nodes**: 16 cores (8 physical + 8 hyperthreaded pairs)

This allows better comparison of NUMA coordinator scaling effects by maintaining proportional thread usage.

## Solution Implemented

### 1. New Method: `create_virtual_numa_with_constant_threads()`

```cpp
void create_virtual_numa_with_constant_threads(bool cpumask[GGML_MAX_N_THREADS], int numa_nodes)
```

**Core Logic**:
- **Base allocation**: 4 cores per virtual NUMA node (physical + hyperthreads)
- **CPU topology awareness**: Groups physical cores with their hyperthreaded pairs
- **Constant scaling**: `numa_nodes * 4` total cores across all configurations
- **Intel Core Ultra 7 165H optimization**: Works with 11 physical cores + 11 hyperthreads

**Implementation Details**:
- Scans CPU topology to identify physical cores vs hyperthreads
- Groups cores into `[physical_core, hyperthread_pair]` vectors
- Allocates 2 physical cores per virtual NUMA (= 4 total cores with hyperthreads)
- Provides detailed logging of virtual NUMA setup

### 2. Updated `benchmark_numa_scaling()` Integration

**Before**: 
- 1 NUMA used all available cores (22 threads)
- Multi-NUMA used variable allocation

**After**:
- All NUMA configurations use constant thread count approach
- Consistent allocation: 1×4, 2×8, 4×16 threads

```cpp
// Simplified logic - always use constant thread approach
create_virtual_numa_with_constant_threads(tpp.cpumask, numa_nodes);
```

### 3. Fallback Strategy Enhanced

For systems without CPU topology detection:
- **Simple allocation**: `numa_nodes * 4` threads
- **Sequential CPU assignment**: CPUs 0-3 for 1 NUMA, 0-7 for 2 NUMA, etc.

## Test Results Verification

**System**: Intel Core Ultra 7 165H (11 physical + 11 hyperthreads = 22 logical CPUs)

### Virtual NUMA Setup Logging
```
Virtual NUMA constant-thread setup:
  Total system: 11 physical cores (22 logical CPUs)
  Base allocation: 4 cores per virtual NUMA (physical + hyperthreads)
  Testing X virtual NUMA nodes × 4 cores = Y total cores
```

### Thread Count Validation
| Configuration | Expected Threads | Actual Result | Status |
|---------------|------------------|---------------|---------|
| 1 NUMA node   | 4 threads       | 4.0 threads   | ✅ Correct |
| 2 NUMA nodes  | 8 threads       | 8.0 threads   | ✅ Correct |
| 4 NUMA nodes  | 16 threads      | 16.0 threads  | ✅ Correct |

### Performance Scaling Results
```
NUMA Nodes | Threads | Time(ms) | GOPS    | Speedup | Efficiency
-----------|---------|----------|---------|---------|----------
1          | 4       | 818.33   | 1.31    | 1.31   x | 100.0   %
2          | 8       | 467.42   | 2.30    | 2.30   x | 87.5    %
4          | 16      | 458.89   | 2.34    | 2.34   x | 44.6    %
```

## Key Improvements

1. **Consistent Thread Allocation**: Eliminates variable thread count confusion
2. **Better Scaling Analysis**: Can now properly measure NUMA coordinator effectiveness
3. **CPU Topology Intelligence**: Proper physical core and hyperthread pairing
4. **Detailed Logging**: Clear visibility into virtual NUMA setup process
5. **Divide-by-4 Strategy**: Implements user's specific requirement for core division

## Technical Notes

- **Physical Core Grouping**: Each virtual NUMA gets 2 physical cores + hyperthreads
- **Total Thread Proportionality**: 1 NUMA = 4 cores, 2 NUMA = 8 cores, 4 NUMA = 16 cores
- **NUMA Coordinator Integration**: Works with existing `ggml_numa_coordinator_manager` system
- **Fallback Compatibility**: Maintains functionality on systems without CPU topology detection

## Build Status
✅ **Compilation**: Clean build with only standard warnings  
✅ **Execution**: All thread counts verified correctly  
✅ **Performance**: Scaling analysis shows expected efficiency patterns

## Future Enhancements
- Could add configurable base thread count instead of fixed 4 cores per NUMA
- Potential for hybrid P-core/E-core awareness on Intel systems
- Option to test different thread count strategies within same test run

This implementation successfully provides the constant thread count NUMA scaling comparison requested, enabling accurate analysis of NUMA coordinator scaling effectiveness across different virtual node configurations.
