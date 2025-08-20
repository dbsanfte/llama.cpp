# NUMA Coordinator + ggml-Native Integration Architecture

## Overview
The ggml-native approach preserves and enhances our existing NUMA infrastructure by using ggml's computational graphs while maintaining our NUMA coordination and threadpool management.

## Key Insight: Threadpool Compatibility

```c
// ggml_graph_compute() accepts OUR threadpools!
enum ggml_status ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);

// Where cplan contains:
struct ggml_cplan {
    size_t work_size;
    uint8_t * work_data;
    int n_threads;
    struct ggml_threadpool * threadpool;  // <-- We can provide our NUMA threadpool here!
};
```

## Architecture Layers

### Layer 1: NUMA Coordinator Manager (Unchanged)
```
┌─────────────────────────────────────────────────────────────┐
│                NUMA Coordinator Manager                     │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   NUMA Node 0   │    │   NUMA Node 1   │                │
│  │   Coordinator   │    │   Coordinator   │                │
│  │                 │    │                 │                │
│  │  - 56 threads   │    │  - 56 threads   │                │
│  │  - CPU affinity │    │  - CPU affinity │                │
│  │  - Threadpool   │    │  - Threadpool   │                │
│  └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

### Layer 2: Enhanced Dispatcher (Modified)
```
┌─────────────────────────────────────────────────────────────┐
│                   Enhanced Dispatcher                       │
│                                                             │
│  Operation Analysis                                         │
│  ├── Quantized MUL_MAT    → ggml-Native NUMA               │
│  ├── F32 MUL_MAT          → Traditional NUMA                │
│  ├── Complex Operations   → Fallback                        │
│  └── Strategy Selection   → Single/Parallel/Graph           │
└─────────────────────────────────────────────────────────────┘
```

### Layer 3: Work Function Types (Enhanced)
```
┌─────────────────────────────────────────────────────────────┐
│                    Work Function Layer                      │
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Traditional   │  │  ggml-Native    │  │   Fallback   │ │
│  │   NUMA Work     │  │   NUMA Work     │  │   Work       │ │
│  │   Functions     │  │   Functions     │  │   Functions  │ │
│  │                 │  │                 │  │              │ │
│  │ Direct kernels  │  │ ggml subgraphs  │  │ Full ggml    │ │
│  │ Manual memory   │  │ + NUMA pools    │  │ computation  │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Integration Flow

### 1. Request Arrives
```
Model Inference Request
        ↓
   Dispatcher Analysis
        ↓
   Strategy Selection
```

### 2. NUMA Coordination (Unchanged)
```c
// Our existing coordinator infrastructure remains intact
struct ggml_numa_coordinator_manager* manager = ggml_numa_coordinator_manager_get_global();

// Work distribution across NUMA nodes (unchanged)
for (int node = 0; node < num_numa_nodes; node++) {
    ggml_numa_work_context_t context = {
        .tensor = tensor,
        .numa_node = node,
        .threadpool = coordinator->threadpool,  // Our NUMA-local threadpool
        // ...
    };
    
    // Dispatch work function (enhanced with new types)
    coordinator->work_function(&context);
}
```

### 3. ggml-Native Work Function (New)
```c
static int ggml_numa_work_function_mulmat_native(void* context) {
    const ggml_numa_work_context_t* ctx = context;
    
    // Step 1: Create local ggml context for graph operations
    struct ggml_context* local_ggml_ctx = ggml_init(init_params);
    
    // Step 2: Create NUMA-local subgraph using ggml's type system
    struct ggml_cgraph* subgraph = ggml_numa_create_mulmat_subgraph(
        local_ggml_ctx, src0, src1, dst, ctx->numa_node, ctx->max_numa_nodes);
    
    // Step 3: Create execution plan 
    struct ggml_cplan plan = ggml_graph_plan(subgraph, ctx->thread_count, NULL);
    
    // Step 4: USE OUR NUMA THREADPOOL (Key Integration Point!)
    plan.threadpool = ctx->threadpool;  // <-- Our NUMA-aware threadpool
    
    // Step 5: Execute using ggml's proven computation + our threading
    enum ggml_status result = ggml_graph_compute(subgraph, &plan);
    
    // Step 6: Cleanup
    ggml_free(local_ggml_ctx);
    return (result == GGML_STATUS_SUCCESS) ? 0 : -1;
}
```

## Preserved NUMA Benefits

### ✅ **Thread Affinity (Kept)**
- Our CPU binding and NUMA-local thread creation
- Optimal memory access patterns
- Cache locality preservation

### ✅ **Work Distribution (Enhanced)**
```c
// Our existing work distribution with ggml-native capabilities
switch (strategy) {
    case NUMA_NODE_STRATEGY_SINGLE:
        // Execute on single NUMA node with our threadpool
        return execute_single_node_ggml_native(work_function);
        
    case NUMA_NODE_STRATEGY_DATA_PARALLEL:
        // Distribute across NUMA nodes with our coordination
        return execute_data_parallel_ggml_native(work_function);
}
```

### ✅ **Memory Management (Enhanced)**
- Our NUMA-local memory allocation
- Enhanced with ggml's automatic work buffer sizing
- Optimal memory placement per NUMA node

## Code Changes Required

### Minimal Dispatcher Changes
```c
// Add to existing dispatcher
enum ggml_status ggml_numa_dispatch_operation(struct ggml_tensor* tensor, struct ggml_cplan* cplan) {
    // Enhanced strategy selection
    if (should_use_ggml_native(tensor)) {
        return execute_with_ggml_native_numa(tensor, cplan);
    } else if (should_use_traditional_numa(tensor)) {
        return execute_with_traditional_numa(tensor, cplan);
    } else {
        return ggml_numa_fallback_execute(tensor, cplan);
    }
}
```

### New Work Function Types
```c
// Traditional NUMA (unchanged)
static int ggml_numa_work_function_add_chunk(void* context);

// ggml-Native NUMA (new)
static int ggml_numa_work_function_add_native(void* context);

// Fallback (unchanged)  
static int ggml_numa_work_function_fallback(void* context);
```

## Benefits Matrix

| Component | Traditional NUMA | ggml-Native NUMA | Fallback |
|-----------|------------------|------------------|----------|
| **Coordinator** | ✅ Used | ✅ Used | ❌ Bypassed |
| **Our Threadpools** | ✅ Used | ✅ Used | ❌ ggml's pools |
| **CPU Affinity** | ✅ Preserved | ✅ Preserved | ❌ Lost |
| **Type System** | ⚠️ Manual | ✅ ggml's | ✅ ggml's |
| **Quantized Ops** | ❌ Broken | ✅ Working | ✅ Working |
| **Performance** | ✅ High | ✅ Very High | ⚠️ Lower |

## Migration Strategy

### Phase 1: Parallel Implementation
- Keep all existing infrastructure
- Add ggml-native work functions alongside traditional ones
- Dispatcher chooses based on operation and tensor type

### Phase 2: Gradual Migration
- Operation-by-operation migration to ggml-native
- Traditional NUMA for operations that work well
- ggml-native for quantized operations

### Phase 3: Optimization
- Enhance coordination between ggml graphs and NUMA threading
- Optimize memory allocation strategies
- Fine-tune work distribution algorithms

## Key Advantage: Zero Infrastructure Loss

**We keep everything that works:**
- ✅ NUMA node coordinators
- ✅ CPU affinity and thread binding  
- ✅ NUMA-aware threadpool management
- ✅ Work distribution strategies
- ✅ Memory locality optimization

**We add what was missing:**
- ✅ Proper quantized operation support
- ✅ Type system integration
- ✅ Automatic memory layout handling
- ✅ Graph-based computation benefits

This gives us a **truly hybrid system** that preserves our NUMA performance advantages while fixing the quantized operation correctness issues.
