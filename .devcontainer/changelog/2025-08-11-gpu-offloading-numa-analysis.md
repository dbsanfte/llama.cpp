# GPU Offloading NUMA Analysis - Deep Dive

## Executive Summary

After conducting a comprehensive deep dive into the llama.cpp GPU offloading implementation, I've identified that **GPU offloading is currently NOT NUMA-aware in its thread allocation and execution model**, despite having excellent infrastructure in place for GPU-NUMA topology detection.

## Current GPU Offloading Architecture

### 1. GPU Backend Threading Model

**Key Finding**: GPU operations use a **single-threaded model per GPU backend** with CUDA streams for async operations:

- Each GPU backend has one `ggml_backend_cuda_context` 
- Uses `cudaStreamPerThread` for most operations
- One context per GPU device, one thread context per operation
- No explicit thread pool or multiple worker threads per GPU

**Code Evidence**:
```cpp
// From ggml-cuda.cu - all operations use per-thread streams
CUDA_CHECK(cudaMemcpyAsync(..., cudaStreamPerThread));
CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
```

### 2. Backend Scheduler (The Orchestrator)

The `ggml_backend_sched` in `/ggml/src/ggml-backend.cpp` is the central coordinator:

- **Splits computation graph** across CPU/GPU backends
- **Schedules data transfers** between backends  
- **Manages tensor copying** between devices
- **Controls execution order** with events/synchronization

**Critical Gap**: No NUMA awareness in the scheduler threading model.

### 3. GPU Device Assignment

**Current Logic** (from `llama-model.cpp`):
```cpp
const int i_gpu_start = std::max((int) hparams.n_layer - n_gpu_layers, (int) 0);
const int act_gpu_layers = devices.empty() ? 0 : std::min(n_gpu_layers, (int)n_layer + 1);

// Layer assignment based on memory splits, not NUMA topology
const int layer_gpu = std::upper_bound(splits.begin(), splits.begin() + n_devices(), 
                                      float(il - i_gpu_start)/act_gpu_layers) - splits.begin();
```

**Issue**: Layer assignment is based on **memory availability**, not NUMA proximity.

## NUMA-Awareness Infrastructure (Excellent but Unused)

### GPU-NUMA Detection System

The codebase has **sophisticated GPU-NUMA topology detection**:

**`detect_gpu_numa_affinity()`** in `common/common.cpp`:
- Reads `/sys/class/drm` for GPU hardware info
- Maps GPUs to NUMA nodes via PCI topology
- Detects GPU-local CPU cores
- Correlates with backend device information

**`gpu_numa_info` Structure**:
```cpp
struct gpu_numa_info {
    int gpu_id;
    int numa_node;                    // NUMA node GPU is connected to
    std::vector<int> local_cpu_cores; // CPU cores on same NUMA node
    std::string pci_bus_id;           // PCIe bus information
    bool affinity_detected;
    // ... backend correlation info
};
```

### Thread Binding Functions (Implemented but Unused)

**`bind_thread_to_gpu_numa_node()`**:
- Sets CPU affinity using `pthread_setaffinity_np()`
- Configures NUMA memory policy with `numa_set_membind()`
- Binds thread to GPU's local NUMA node

**`enforce_gpu_cpu_numa_affinity()`**:
- Validates NUMA node assignment
- Sets CPU_SET for local cores only
- Uses libnuma for memory locality

## Current Thread Coordination Analysis

### Where GPU Work Happens

1. **Main Application Thread**: Calls `llama_context::graph_compute()`
2. **Backend Scheduler**: `ggml_backend_sched_compute_splits()` orchestrates
3. **GPU Backend Thread**: Executes on calling thread (no separate worker threads)
4. **CUDA/Vulkan Async**: Uses device streams for parallel execution

### Threading Model Gaps

**Problem 1: No Dedicated GPU Threads**
- GPU operations execute on **caller's thread context**
- No thread pool dedicated to GPU operations
- **Thread affinity never set** for GPU operations

**Problem 2: Cross-NUMA Memory Access**
- CPU tensors allocated without NUMA awareness relative to target GPU
- Host-to-device transfers may cross NUMA boundaries
- No memory locality optimization

**Problem 3: Backend Scheduler NUMA-Blind**
- Splits work based on **device memory capacity**
- Ignores **thread placement** and **memory locality**
- No consideration of **cross-socket bandwidth**

## NUMA Impact Analysis

### Performance Issues

1. **Cross-Socket Memory Transfers**:
   ```
   CPU Thread on NUMA 0 → GPU on NUMA 1 Socket
   Host Memory on NUMA 0 → PCIe transfer across QPI/UPI links
   ```

2. **Bandwidth Bottlenecks**:
   - Inter-socket bandwidth: ~50-100 GB/s
   - Local NUMA bandwidth: ~200-400 GB/s
   - **2-4x performance penalty** for cross-socket access

3. **Cache Misses**:
   - CPU caches optimized for local NUMA node
   - Cross-socket access bypasses cache hierarchy

### Current State Summary

| Component | NUMA Aware | Notes |
|-----------|------------|-------|
| GPU Detection | ✅ | Excellent topology detection |
| Thread Binding Functions | ✅ | Implemented but unused |
| Layer Assignment | ❌ | Memory-based, not topology-based |
| Backend Scheduler | ❌ | No NUMA considerations |
| GPU Compute Threads | ❌ | No dedicated threads, no affinity |
| Memory Allocation | ❌ | No GPU-local memory strategy |

## Recommendations for NUMA-Aware GPU Offloading

### 1. GPU Thread Pool with NUMA Affinity

**Create dedicated worker threads per GPU**:
```cpp
struct numa_gpu_worker {
    int gpu_id;
    int numa_node;
    std::thread worker_thread;
    std::queue<gpu_task> task_queue;
    std::condition_variable task_cv;
    std::mutex task_mutex;
};

// Bind worker thread to GPU's NUMA node
void create_gpu_worker(int gpu_id, const gpu_numa_info& gpu_info) {
    workers[gpu_id].worker_thread = std::thread([gpu_id, gpu_info]() {
        bind_thread_to_gpu_numa_node(gpu_id, {gpu_info});
        // GPU worker loop...
    });
}
```

### 2. NUMA-Aware Backend Scheduler

**Modify `ggml_backend_sched_compute_splits()`**:
- Consider GPU-NUMA topology in layer assignment
- Prefer GPU backends on same NUMA node as current thread
- Implement cross-socket cost model

### 3. Host Memory Allocation Strategy

**GPU-Local Memory Allocation**:
```cpp
// Allocate host tensors on GPU's NUMA node
void* allocate_gpu_local_tensor(size_t size, int gpu_id) {
    const auto& gpu_info = get_gpu_numa_info(gpu_id);
    return allocate_gpu_numa_local_memory(size, gpu_info.numa_node);
}
```

### 4. Smart Layer-GPU Assignment

**Replace memory-based assignment with NUMA-aware logic**:
```cpp
int select_optimal_gpu_for_layer(int layer_id, const std::vector<gpu_numa_info>& gpus) {
    int current_numa = get_current_thread_numa_node();
    
    // Prefer GPU on same NUMA node
    for (const auto& gpu : gpus) {
        if (gpu.numa_node == current_numa && gpu.has_capacity_for_layer(layer_id)) {
            return gpu.gpu_id;
        }
    }
    
    // Fall back to closest NUMA node
    return find_closest_numa_gpu(current_numa, gpus);
}
```

## Implementation Priority

1. **High Priority**: GPU thread pool with NUMA affinity
2. **Medium Priority**: NUMA-aware layer assignment  
3. **Low Priority**: Host memory locality optimization
4. **Future**: Cross-socket bandwidth optimization

## Testing Strategy

- **Benchmark**: Multi-GPU inference across NUMA nodes
- **Measure**: Memory bandwidth utilization per NUMA node
- **Verify**: Thread placement with `numactl --hardware`
- **Monitor**: Cross-socket PCIe traffic

---

**Conclusion**: llama.cpp has excellent GPU-NUMA detection infrastructure but currently doesn't use it for actual GPU computation thread placement. The biggest impact would come from implementing dedicated GPU worker threads with proper NUMA affinity, followed by NUMA-aware layer assignment in the backend scheduler.
