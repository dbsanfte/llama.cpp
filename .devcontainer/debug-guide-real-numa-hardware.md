# Debug Guide for Real NUMA Hardware Testing

**Date:** August 12, 2025  
**Issue:** "NUMA node 1: no NUMA-local CPUs found in optimized mask"  
**Hardware:** 2-socket server with 112 logical CPUs across 2 NUMA nodes  

## Debug Logging Added

I've added comprehensive debug logging to help diagnose the CPU mask creation and filtering issue on your real NUMA hardware. Here's what to look for:

### 1. CPU Mask Creation Phase

Look for these sections in the output:

```
================================================================================
                     Creating Optimal CPU Masks
================================================================================
   Input: X total threads, 2 NUMA nodes
   Original CPU mask: (none)
```

Then for each NUMA node:
```
NUMA node 0: has Y CPUs, assigning Z threads
   Available CPUs for node 0: [0,1,2,3,...,27,56,57,...,83]
   Primary cores: A, Hyperthreads: B
   Assigned primary CPU X to node 0
   Assigned hyperthread CPU Y to node 0
```

```
NUMA node 1: has Y CPUs, assigning Z threads  
   Available CPUs for node 1: [28,29,30,...,55,84,85,...,111]
   Primary cores: A, Hyperthreads: B
   Assigned primary CPU X to node 1
   Assigned hyperthread CPU Y to node 1
```

```
    Created NUMA-aware CPU mask with X CPUs total across 2 nodes
    Final global CPU mask: [0,2,4,6,...,28,30,32,...]
```

### 2. Coordinator Creation Phase

For each coordinator being created:

```
================================================================================
                Creating Coordinator for NUMA Node 0
================================================================================
   Processing NUMA node 0 in REAL NUMA mode (hardware node exists)
Filtering CPU mask for NUMA node 0 (real NUMA mode)
   Optimized mask before filtering: [0,2,4,6,8,...,28,30,32,...]
   CPUs belonging to NUMA node 0: [0,1,2,3,...,27,56,57,...,83]
   ✅ NUMA node 0: successfully filtered to X CPUs: [0,2,4,6,8,...]
✅ NUMA node 0 coordinator created successfully with Y threads
```

```
================================================================================
                Creating Coordinator for NUMA Node 1  
================================================================================
   Processing NUMA node 1 in REAL NUMA mode (hardware node exists)
Filtering CPU mask for NUMA node 1 (real NUMA mode)
   Optimized mask before filtering: [0,2,4,6,8,...,28,30,32,...]
   CPUs belonging to NUMA node 1: [28,29,30,...,55,84,85,...,111]
   ❌ NUMA node 1: no intersection found between optimized mask and node CPUs
   ❌ Filtered result would be: [] (empty)
NUMA node 1: no NUMA-local CPUs found in optimized mask, using original mask
```

## Key Things to Analyze

### 1. **CPU Topology Detection**
- Does it correctly detect CPUs for each NUMA node?
- Node 0 should show: `[0,1,2,3,...,27,56,57,...,83]`  
- Node 1 should show: `[28,29,30,...,55,84,85,...,111]`

### 2. **Primary vs Hyperthread Classification**
- Are CPUs correctly classified as primary cores vs hyperthreads?
- For your hardware, expect something like:
  - Node 0 primaries: `[0,2,4,...,26,56,58,...,82]`
  - Node 0 hyperthreads: `[1,3,5,...,27,57,59,...,83]`
  - Node 1 primaries: `[28,30,32,...,54,84,86,...,110]`  
  - Node 1 hyperthreads: `[29,31,33,...,55,85,87,...,111]`

### 3. **Global CPU Mask Creation**
- Does the final global mask include CPUs from both NUMA nodes?
- If requesting 112 threads, should have ~56 CPUs from each node
- Should see mix like: `[0,2,4,6,...,28,30,32,34,...]`

### 4. **CPU Mask Filtering**
- This is where the bug likely occurs
- When filtering global mask for NUMA node 1, does it find any intersection?
- **Expected**: Should find CPUs like `[28,30,32,34,...]` 
- **Bug**: Might show empty intersection `[]`

## Test Command

Run this on your real NUMA hardware:

```bash
cd /path/to/llama.cpp
./build/bin/test-comprehensive-numa-performance 2>&1 | tee numa_debug.log
```

## What to Report Back

Please share:

1. **The complete debug output** showing CPU mask creation and filtering
2. **Specific sections** where you see the ❌ errors
3. **CPU topology detection results** for both nodes
4. **Final global CPU mask** after creation
5. **Filtering results** for both NUMA nodes

## Likely Root Causes

Based on the debug output, we'll be able to identify:

1. **CPU assignment bug**: Global mask only includes CPUs from node 0
2. **Topology detection bug**: Wrong CPUs detected for node 1  
3. **Filtering logic bug**: Intersection logic fails incorrectly
4. **Thread siblings parsing bug**: Primary/hyperthread detection fails

The debug output will pinpoint exactly where the issue occurs!
