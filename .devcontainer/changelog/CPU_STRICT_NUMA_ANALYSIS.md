# Analysis: `--cpu-strict` and `--cpu-strict-batch` Interaction with NUMA Code

## Overview

The `--cpu-strict` and `--cpu-strict-batch` options control how CPU affinity masks are assigned to worker threads in llama.cpp. These options have important interactions with the NUMA-aware thread allocation system.

## How CPU Strict Placement Works

### 1. **Non-Strict Mode (`--cpu-strict 0`)**
- **Behavior**: All threads share the same CPU affinity mask
- **Implementation**: `ggml_thread_cpumask_next()` copies the global mask to each thread's local mask
- **Result**: Multiple threads can run on the same CPU cores
- **Use Case**: Simple workloads, systems without NUMA considerations

```c
// Non-strict: All threads get identical masks
if (!strict) {
    memcpy(local_mask, global_mask, GGML_MAX_N_THREADS);
    return;
}
```

### 2. **Strict Mode (`--cpu-strict 1`)**
- **Behavior**: Each thread gets assigned to a unique CPU core from the mask
- **Implementation**: `ggml_thread_cpumask_next()` distributes cores round-robin
- **Result**: One thread per CPU core, better cache locality
- **Use Case**: High-performance workloads, NUMA systems

```c
// Strict: Round-robin assignment of individual cores
int32_t base_idx = *iter;
for (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
    int32_t idx = base_idx + i;
    if (idx >= GGML_MAX_N_THREADS) {
        idx -= GGML_MAX_N_THREADS;  // Wrap around
    }
    if (global_mask[idx]) {
        local_mask[idx] = 1;  // Assign this single core
        *iter = idx + 1;      // Next thread gets next core
        return;
    }
}
```

## NUMA Integration and Conflicts

### 1. **NUMA Mirroring Override**

The NUMA code has its own strict placement logic that **overrides** user settings:

```c
// In ggml_graph_compute_thread() - NUMA MIRRORING section
if (cpuid == -1) {
    bool local_mask[GGML_MAX_N_THREADS];
    int iter = 0;
    for (int j = 0; j < thread_id; ++j) {
        ggml_thread_cpumask_next(cpumask, local_mask, true, &iter);  // ← ALWAYS strict=true
    }
    memset(local_mask, 0, sizeof(bool) * GGML_MAX_N_THREADS);
    ggml_thread_cpumask_next(cpumask, local_mask, true, &iter);     // ← ALWAYS strict=true
}
```

**Key Insight**: When NUMA mirroring is active, the system **forces strict placement** regardless of user's `--cpu-strict` setting.

### 2. **Two-Layer CPU Assignment**

The system has two CPU assignment layers:

1. **Threadpool Creation** (`ggml_threadpool_new_impl`):
   - Uses user's `--cpu-strict` setting
   - Assigns threads to CPU cores via `ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter)`

2. **NUMA Runtime Binding** (`ggml_graph_compute_thread`):
   - Overrides threadpool assignments when NUMA mirroring is enabled
   - Always uses strict placement for NUMA locality
   - Calls `sched_setaffinity()` directly with specific CPU cores

### 3. **Batch vs. Non-Batch Threading**

- `--cpu-strict`: Controls main inference threadpool
- `--cpu-strict-batch`: Controls batch processing threadpool
- Both can have different settings for different workload characteristics

## Potential Issues and Interactions

### 1. **Setting Conflicts**

**Scenario**: User sets `--cpu-strict 0` but NUMA mirroring is enabled
- **Result**: NUMA code overrides with strict=true
- **Impact**: User setting ignored, threads still get individual CPU assignments
- **Solution**: Document this behavior or modify NUMA code to respect user preference

### 2. **CPU Mask Validation**

```c
// From postprocess_cpu_params()
if (n_set && n_set < cpuparams.n_threads) {
    LOG_WRN("Not enough set bits in CPU mask (%d) to satisfy requested thread count: %d\n", 
            n_set, cpuparams.n_threads);
}
```

**Issue**: With strict placement, you need at least as many CPU cores as threads
- **Non-strict**: 4 threads can share 2 cores
- **Strict**: 4 threads need exactly 4 cores

### 3. **NUMA Node Boundaries**

When using CPU masks that span multiple NUMA nodes:
- **Non-strict**: Threads can migrate between NUMA nodes
- **Strict**: Each thread bound to specific core, but may not respect NUMA locality
- **NUMA mirroring**: Overrides both and enforces NUMA-aware strict placement

## Recommendations

### 1. **For Users**

- **Single NUMA node systems**: `--cpu-strict 1` for better performance
- **Multi-NUMA systems**: Let NUMA mirroring handle placement (it forces strict anyway)
- **CPU masks**: Ensure mask has enough cores for strict placement
- **Batch workloads**: Consider different settings for batch vs. non-batch

### 2. **For Code Improvements**

#### A. **Unified CPU Assignment Logic**
```c
// Proposed: Single function that respects both user preferences and NUMA requirements
static int assign_thread_cpu(int thread_id, const bool* global_mask, bool user_strict, 
                             bool numa_enabled, int numa_node) {
    if (numa_enabled && numa_node >= 0) {
        // NUMA-aware assignment (may override user_strict for locality)
        return assign_numa_local_cpu(thread_id, global_mask, numa_node);
    } else {
        // User preference controls
        return assign_cpu_by_preference(thread_id, global_mask, user_strict);
    }
}
```

#### B. **Better Conflict Detection**
```c
// Warn when NUMA overrides user settings
if (numa_enabled && !user_strict) {
    LOG_WRN("NUMA mirroring enabled - forcing strict CPU placement despite --cpu-strict 0\n");
}
```

#### C. **NUMA-Aware CPU Mask Validation**
```c
// Check if CPU mask has enough cores per NUMA node
bool validate_numa_cpu_mask(const bool* mask, int n_threads, int numa_nodes) {
    for (int node = 0; node < numa_nodes; node++) {
        int node_cores = count_cores_on_numa_node(mask, node);
        int threads_per_node = (n_threads + numa_nodes - 1) / numa_nodes;
        if (node_cores < threads_per_node) {
            LOG_WRN("NUMA node %d has only %d cores but needs %d threads\n", 
                    node, node_cores, threads_per_node);
            return false;
        }
    }
    return true;
}
```

## Testing Scenarios

### 1. **Verify Current Behavior**
```bash
# Test strict vs non-strict with CPU mask
./llama-cli --threads 4 --cpu-mask 0x0F --cpu-strict 0 --cpu-topology
./llama-cli --threads 4 --cpu-mask 0x0F --cpu-strict 1 --cpu-topology

# Test with insufficient cores (should warn)
./llama-cli --threads 8 --cpu-mask 0x0F --cpu-strict 1 --cpu-topology
```

### 2. **NUMA Interaction Testing**
```bash
# On multi-NUMA system, test if user settings are respected
numactl --membind=0 ./llama-cli --threads 4 --cpu-strict 0 --cpu-topology
numactl --membind=0 ./llama-cli --threads 4 --cpu-strict 1 --cpu-topology
```

## Summary

The `--cpu-strict` options provide important control over CPU thread assignment, but their interaction with NUMA code creates complexity:

1. **NUMA mirroring overrides user strict settings** for optimal locality
2. **Strict placement requires sufficient CPU cores** in the mask
3. **Batch and non-batch threadpools** can have different placement strategies
4. **Current implementation has two competing assignment layers** that may conflict

The system generally works well but could benefit from unified logic and better user feedback about when NUMA overrides their preferences.
