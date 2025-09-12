# NUMA Coordinator Refactor Plan - August 13, 2025

## 🚨 **Critical Issue Identified**
Current NUMA coordinator crashes on ROPE operations due to naive parallelization approach:
```
Thread 35 "llama-server" received signal SIGSEGV, Segmentation fault.
0x00007ffff74c8346 in rope_yarn (cos_theta=0x0, sin_theta=0x4)
```

**Root Cause**: Attempting to execute operations in isolation without proper context, shared state, or initialization.

## 📋 **Comprehensive Refactor Plan**

### Phase 1: GGML Operations Audit & Classification
**Goal**: Catalog all GGML operations and classify them by parallelization characteristics

#### 1.1 Operation Discovery
- [ ] **Parse ggml-cpu.c**: Extract all `ggml_compute_forward_*` functions
- [ ] **Analyze ggml.h**: Identify all `GGML_OP_*` enum values  
- [ ] **Cross-reference**: Map operations to their compute functions
- [ ] **Document dependencies**: Identify operations requiring shared state/initialization

#### 1.2 Operation Classification Matrix
Create comprehensive classification for each operation:

```c
typedef enum {
    NUMA_OP_FULLY_PARALLELIZABLE,    // Can split across NUMA nodes safely
    NUMA_OP_LIMITED_PARALLEL,        // Parallel but with constraints  
    NUMA_OP_SEQUENTIAL_ONLY,         // Must run on single node
    NUMA_OP_REQUIRES_SPECIAL_HANDLING // Complex dependencies
} numa_op_parallelization_t;

typedef enum {
    NUMA_OP_NO_STATE,                // Stateless operation
    NUMA_OP_REQUIRES_CACHE,          // Needs cache initialization (e.g., ROPE)
    NUMA_OP_REQUIRES_CONTEXT,        // Needs execution context
    NUMA_OP_REQUIRES_SYNCHRONIZATION // Needs cross-node sync
} numa_op_state_requirements_t;
```

#### 1.3 Priority Operations to Analyze
High-impact operations that need immediate attention:
- [ ] **ROPE** (RoPE attention) - Currently crashing, needs cache init
- [ ] **MUL_MAT** (Matrix multiplication) - High parallelization potential
- [ ] **ADD/MUL/DIV** (Element-wise) - Good candidates for data parallelism
- [ ] **ATTENTION ops** - Complex, likely need special handling
- [ ] **NORM operations** - May require synchronization
- [ ] **ACTIVATION functions** (GELU, SILU, etc.) - Good parallel candidates

### Phase 2: Intelligent Operation Dispatcher Architecture

#### 2.1 Operation Registry System
```c
typedef struct ggml_numa_operation_handler {
    enum ggml_op op_type;
    numa_op_parallelization_t parallel_type;
    numa_op_state_requirements_t state_requirements;
    
    // Function pointers for different execution strategies
    ggml_numa_op_handler_fn single_node_handler;
    ggml_numa_op_handler_fn data_parallel_handler;
    ggml_numa_op_handler_fn custom_parallel_handler;
    
    // Initialization/cleanup functions
    ggml_numa_op_init_fn init_fn;
    ggml_numa_op_cleanup_fn cleanup_fn;
    
    // Performance characteristics
    size_t min_elements_for_parallelization;
    float estimated_parallel_efficiency;
} ggml_numa_operation_handler_t;
```

#### 2.2 Smart Dispatcher Logic
- [ ] **Operation Analysis**: Examine each graph node and determine optimal execution strategy
- [ ] **Dependency Resolution**: Identify operations that must be executed sequentially
- [ ] **Load Balancing**: Distribute parallelizable work optimally across NUMA nodes
- [ ] **Fallback Mechanisms**: Always provide single-node execution path

#### 2.3 Execution Strategy Framework
```c
typedef enum {
    NUMA_EXEC_SINGLE_NODE,           // Execute on NUMA node 0 only
    NUMA_EXEC_DATA_PARALLEL,         // Split data across nodes
    NUMA_EXEC_PIPELINE_PARALLEL,     // Pipeline different operations
    NUMA_EXEC_CUSTOM_STRATEGY        // Operation-specific strategy
} numa_execution_strategy_t;
```

### Phase 3: Operation-Specific Handlers

#### 3.1 ROPE Operation Handler (Priority 1 - Fixes crash)
**Requirements**: 
- Proper cache initialization with `ggml_rope_cache_init`
- Shared sin/cos lookup tables
- Thread-safe cache access

**Strategy**: 
- Initialize ROPE cache once per coordinator
- Execute ROPE operations on single node with proper context
- Consider data parallelism for large sequence lengths

#### 3.2 Matrix Operations Handler (Priority 2 - Performance)
**Operations**: `MUL_MAT`, `MUL_MAT_ID`
**Strategy**:
- Row-wise or column-wise data parallelism
- NUMA-local memory access patterns
- Optimal blocking for cache efficiency

#### 3.3 Element-wise Operations Handler (Priority 3)
**Operations**: `ADD`, `MUL`, `DIV`, `SUB`
**Strategy**:
- Simple data parallelism
- Chunk-based distribution across NUMA nodes
- Minimal synchronization required

#### 3.4 Attention Operations Handler (Priority 4)
**Operations**: Various attention-related ops
**Strategy**:
- Analyze attention patterns for parallelization opportunities
- May require specialized handling due to dependencies

### Phase 4: Implementation Strategy

#### 4.1 Refactor Current Coordinator
- [ ] **Replace naive operation execution** with dispatcher system
- [ ] **Add operation registry** and handler lookup
- [ ] **Implement fallback paths** for all operations
- [ ] **Add comprehensive logging** for debugging

#### 4.2 Gradual Migration
1. **Start with ROPE fix**: Implement single-node ROPE handler to fix crash
2. **Add matrix operations**: Focus on high-impact parallelizable operations  
3. **Expand coverage**: Add handlers for all remaining operations
4. **Optimize**: Fine-tune parallelization strategies based on performance data

#### 4.3 Testing Strategy
- [ ] **Operation-specific tests**: Test each handler independently
- [ ] **Integration tests**: Full inference pipeline testing
- [ ] **Performance benchmarks**: Measure improvement vs. single-threaded baseline
- [ ] **Stress testing**: Large models, long sequences, edge cases

### Phase 5: Advanced Optimizations

#### 5.1 Dynamic Strategy Selection
- Runtime profiling to choose optimal execution strategy
- Adaptive load balancing based on actual performance
- Machine learning-based strategy prediction

#### 5.2 Cross-Operation Optimizations
- Operation fusion opportunities
- Memory layout optimizations
- Cache-aware scheduling

## 🎯 **Immediate Action Items**

### Critical (Fix Crash)
1. [ ] **Implement ROPE single-node handler** - Fix segfault immediately
2. [ ] **Add operation classification system** - Prevent future naive parallelization
3. [ ] **Create dispatcher framework** - Route operations to appropriate handlers

### High Priority (Performance)
1. [ ] **Implement MUL_MAT handler** - Major performance impact
2. [ ] **Add element-wise operation handlers** - Low-hanging fruit
3. [ ] **Performance testing** - Validate improvements

### Medium Priority (Completeness)
1. [ ] **Complete operation audit** - Catalog all GGML operations
2. [ ] **Implement remaining handlers** - Full coverage
3. [ ] **Advanced optimizations** - Fine-tuning

## 📈 **Expected Outcomes**
- **Immediate**: No more crashes on ROPE operations
- **Short-term**: Intelligent parallelization of suitable operations
- **Long-term**: Significant performance improvements with full NUMA awareness

## ⚠️ **Risks & Mitigations**
- **Complexity**: Large refactor scope → Incremental implementation
- **Performance regression**: Over-engineering → Always provide fallback paths
- **Maintenance burden**: Too many handlers → Shared infrastructure and patterns
