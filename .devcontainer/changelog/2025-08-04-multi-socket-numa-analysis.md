# Multi-Socket NUMA Data Parallelism Analysis

## Summary

This document analyzes the feasibility of implementing true multi-socket data parallelism in GGML's CPU backend, leveraging multiple NUMA nodes to process different parts of matrices concurrently.

## Current Architecture

### Current NUMA Implementation
The current NUMA implementation in `llama.cpp` focuses on:

1. **Memory Mirroring**: Model weights are duplicated across all NUMA nodes in `llama-mmap.cpp`
2. **Thread Affinity**: Threads are assigned to NUMA nodes in round-robin fashion
3. **Single Threadpool**: All threads coordinate through one threadpool with barriers
4. **Node-Local Memory Access**: Each thread accesses its local NUMA copy of the model

### Current Chunking Strategy
From the attached threading guide and source analysis, GGML currently:

1. **Single-Node Processing**: All threads work on the same tensor operation sequentially 
2. **Work Chunking**: Matrix operations are divided into chunks using `ggml_threadpool_chunk_add()`
3. **Barrier Synchronization**: `ggml_barrier()` ensures all threads complete current node before proceeding
4. **Sequential Graph Execution**: Nodes in the computational graph are processed one at a time

### Key Code Locations
- **Threading**: `ggml/src/ggml-cpu/ggml-cpu.c` lines 3000-3500
- **Memory Mapping**: `src/llama-mmap.cpp` lines 280-500  
- **Matrix Multiplication**: `ggml/src/ggml-cpu/ggml-cpu.c` lines 1200-1500
- **Chunking Logic**: `ggml/src/ggml-cpu/ggml-cpu.c` lines 1400-1500

## Proposed Multi-Socket Architecture

### Concept: Socket-Level Data Parallelism

Instead of having all threads work on the same matrix operation, we can:

1. **Create Multiple Threadpools**: One threadpool per NUMA socket
2. **Distribute Matrix Operations**: Split large matrices across sockets
3. **Coordinate Results**: Aggregate results from each socket back on the main coordinator

### Implementation Strategy

#### 1. Multi-Threadpool Architecture

```c
struct ggml_numa_threadpool_manager {
    int n_sockets;
    struct ggml_threadpool* socket_pools[GGML_NUMA_MAX_NODES];
    struct ggml_threadpool* coordinator_pool;  // Main coordinator on socket 0
};
```

#### 2. Matrix Decomposition Strategy

For matrix multiplication `C = A @ B`, we can decompose by:

**Option A: Row-wise decomposition**
- Socket 0: Processes rows 0 to N/2 of A
- Socket 1: Processes rows N/2 to N of A
- Each socket computes its portion of C independently

**Option B: Column-wise decomposition**  
- Socket 0: Processes columns 0 to M/2 of B
- Socket 1: Processes columns M/2 to M of B
- Results need aggregation

#### 3. Memory Layout Considerations

The current NUMA mirroring already provides:
- Each socket has a complete copy of model weights
- Memory is allocated locally per socket
- This supports independent parallel computation

### Technical Implementation Plan

#### Phase 1: Multi-Threadpool Infrastructure

1. **Extend ggml_threadpool_params**:
```c
struct ggml_threadpool_params {
    // ... existing fields ...
    int numa_socket_id;           // Which socket this threadpool serves
    bool enable_socket_parallelism; // Enable multi-socket coordination
};
```

2. **Create Socket Manager**:
```c
struct ggml_socket_manager* ggml_socket_manager_new(int n_sockets);
void ggml_socket_manager_free(struct ggml_socket_manager* mgr);
```

#### Phase 2: Matrix Operation Decomposition

1. **Matrix Splitting Logic**:
```c
struct ggml_matrix_split {
    int socket_id;
    int64_t row_start, row_end;
    int64_t col_start, col_end;
    struct ggml_tensor* local_result;
};
```

2. **Modified Matrix Multiplication**:
```c
void ggml_compute_forward_mul_mat_multi_socket(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst,
    struct ggml_socket_manager * socket_mgr
);
```

#### Phase 3: Result Coordination

1. **Inter-Socket Communication**:
```c
// Use memory mapping or shared memory for coordination
struct ggml_socket_coordination {
    atomic_int sockets_completed;
    struct ggml_tensor* partial_results[GGML_NUMA_MAX_NODES];
};
```

2. **Result Aggregation**:
- Coordinator thread on socket 0 waits for all sockets
- Aggregates partial results into final tensor
- Continues with next graph node

### Integration with OpenMP

For systems with OpenMP, we can leverage OpenMP's NUMA affinity:

```c
#ifdef GGML_USE_OPENMP
void ggml_multi_socket_omp_compute() {
    #pragma omp parallel num_threads(total_threads)
    {
        int socket_id = omp_get_thread_num() / threads_per_socket;
        // Process matrix chunk for this socket
        process_matrix_chunk_for_socket(socket_id);
    }
    // Implicit barrier at end of parallel region
}
#endif
```

## Challenges and Considerations

### 1. Memory Bandwidth Limitations
- **Challenge**: Multiple sockets accessing memory simultaneously may saturate bandwidth
- **Mitigation**: Ensure each socket works primarily on its local memory copy

### 2. Load Balancing
- **Challenge**: Matrix sizes may not divide evenly across sockets
- **Mitigation**: Dynamic work stealing between sockets for remaining work

### 3. Overhead of Coordination
- **Challenge**: Inter-socket coordination overhead may negate benefits
- **Mitigation**: Only use multi-socket for operations above minimum size threshold

### 4. Graph-Level Dependencies
- **Challenge**: Some operations require results from previous nodes
- **Mitigation**: Fall back to single-socket for dependent operations

### 5. Implementation Complexity
- **Challenge**: Significantly increases codebase complexity
- **Mitigation**: Make it optional, fall back gracefully to current implementation

## Feasibility Assessment

### ✅ **Highly Feasible**
1. **Memory Infrastructure**: Already have NUMA-aware memory allocation
2. **Thread Affinity**: Thread-to-socket assignment already works
3. **Chunking Logic**: Existing chunk-based work distribution can be extended
4. **Barrier Synchronization**: Current barrier mechanisms can be adapted

### ⚠️ **Moderately Challenging**  
1. **Multi-Threadpool Coordination**: Need to coordinate between threadpools
2. **Result Aggregation**: Need efficient inter-socket communication
3. **Load Balancing**: Dynamic work distribution across heterogeneous sockets

### ❌ **Potential Blockers**
1. **Memory Bandwidth**: May not provide performance benefit if memory-bound
2. **Small Matrix Operations**: Overhead may exceed benefits for small operations
3. **Complex Graph Dependencies**: Some operations may not parallelize well

## Performance Expectations

### Scenarios Where Multi-Socket Would Help
1. **Large Matrix Multiplications**: Where computation >> coordination overhead
2. **Inference with Large Batch Sizes**: Multiple sequences processed independently
3. **Memory-Bandwidth Abundant Systems**: Where computation is the bottleneck

### Scenarios Where It May Not Help
1. **Small Models**: Overhead exceeds benefits
2. **Memory-Bound Operations**: Already saturating memory bandwidth
3. **Highly Sequential Operations**: Where dependencies prevent parallelization

## Recommendation

**This is a viable and promising approach** for the following reasons:

1. **Strong Foundation**: Current NUMA infrastructure provides excellent foundation
2. **Clear Benefits**: Large matrix operations would benefit significantly
3. **Manageable Complexity**: Can be implemented incrementally with fallbacks
4. **Real-World Impact**: Multi-socket systems are common in datacenter deployments

### Suggested Implementation Order

1. **Start Small**: Implement for `ggml_mul_mat` operations only
2. **Add Thresholds**: Only enable for matrices above certain size
3. **Measure Performance**: Validate benefits on real hardware before expanding
4. **Expand Gradually**: Add support for other operations if beneficial

The architecture is sound and the current codebase provides most of the necessary infrastructure. The main work would be implementing the multi-threadpool coordination and result aggregation logic.
