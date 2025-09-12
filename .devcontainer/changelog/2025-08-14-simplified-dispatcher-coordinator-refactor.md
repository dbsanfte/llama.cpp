# Simplified Dispatcher-Coordinator Refactor - August 14, 2025

## 🎯 **Overview**
Based on insights from Gemini conversation and further analysis, we're implementing a **simplified dispatcher → coordinator architecture** that separates concerns cleanly:

- **Dispatcher**: Contains all parallelization intelligence, creates work items
- **Coordinator**: Simple execution engine that processes work items

This approach is much simpler than the complex operation registry system originally planned, while achieving the same performance goals.

## 🏗️ **Core Architecture**

### Work Item Abstraction
```c
typedef void (*ggml_task_t)(void *task_data);

typedef struct {
    ggml_task_t func;  // Function to execute
    void * data;       // Operation-specific payload
} ggml_work_item_t;
```

### Role Separation
- **Dispatcher**: Analyzes graph nodes, decides parallelization strategy, creates work items
- **Coordinator**: Dequeues work items, calls `item.func(item.data)`, manages NUMA threadpools

## 📋 **Step-by-Step Implementation Plan**

### Phase 1: Core Infrastructure + Testing Foundation (Priority: Critical)

#### Step 1.1: Create Comprehensive Test Suite Framework
- [ ] Create `test-numa-dispatcher-coordinator.c` as main test file
- [ ] Add test infrastructure (setup, teardown, assertions, test runner)
- [ ] Add basic work item tests (already implemented in test-work-item.c)
- [ ] Create mock NUMA environment for consistent testing
- [ ] Set up performance benchmarking framework

#### Step 1.2: Refactor Coordinator to Simple Execution Engine
- [ ] Replace complex work item system in `ggml-numa-coordinator.c`
- [ ] Implement simple `coordinator_enqueue(numa_node_id, work_item)` 
- [ ] Implement `coordinator_wait_completion()` for synchronization
- [ ] Keep existing NUMA threadpool management
- [ ] **ADD TESTS**: Coordinator enqueue/dequeue, threadpool management, synchronization
- [ ] **RUN FULL TEST SUITE**: Verify no regressions

#### Step 1.3: Create Basic Dispatcher Framework  
- [ ] Create `ggml-numa-dispatcher.c` as new file
- [ ] Implement `ggml_numa_dispatch_graph()` main entry point
- [ ] Add operation type switching logic (`switch(node->op)`)
- [ ] Implement fallback for non-parallelizable operations (primary node execution)
- [ ] **ADD TESTS**: Graph dispatching, operation routing, fallback behavior
- [ ] **RUN FULL TEST SUITE**: Verify dispatcher integrates correctly

### Phase 2: Operation Handlers + Continuous Testing (Priority: High)

#### Step 2.1: ROPE Handler (Critical - Fixes Crash)
- [ ] Create `rope_task_data_t` payload structure
- [ ] Implement `do_rope_chunk()` worker function (split by sequence length)
- [ ] Add `ggml_numa_dispatch_rope()` function in dispatcher
- [ ] **ADD TESTS**: ROPE chunking logic, cache initialization, sequence splitting
- [ ] **RUN FULL TEST SUITE**: Verify ROPE operations don't crash
- [ ] **INTEGRATION TEST**: Test with real ROPE operations from model

#### Step 2.2: Matrix Multiplication Handler (Performance)
- [ ] Create `matmul_task_data_t` payload structure
- [ ] Implement `do_matmul_chunk()` worker function (split by rows/columns)
- [ ] Add `ggml_numa_dispatch_matmul()` function
- [ ] **ADD TESTS**: Matrix chunking, dimension splitting, result aggregation
- [ ] **PERFORMANCE TESTS**: Benchmark vs single-threaded baseline
- [ ] **RUN FULL TEST SUITE**: Verify no regressions in performance

#### Step 2.3: Element-wise Operations Handler
- [ ] Create generic `elementwise_task_data_t` payload
- [ ] Implement worker functions for ADD, MUL, DIV, SUB (split along largest dimension)
- [ ] Add dispatcher functions for each element-wise operation
- [ ] **ADD TESTS**: Element-wise chunking, correctness verification
- [ ] **RUN FULL TEST SUITE**: Verify all operations work correctly

### Phase 3: Integration + System Testing (Priority: Medium)

#### Step 3.1: Wire Up New System in Main Path
- [ ] Update `ggml_graph_compute()` to call new dispatcher
- [ ] Remove old coordinator and dispatcher completely
- [ ] **ADD TESTS**: End-to-end graph computation, full inference pipeline
- [ ] **INTEGRATION TESTS**: Test with real models (qwen2.5-0.5b)
- [ ] **RUN FULL TEST SUITE**: Comprehensive system verification

#### Step 3.2: System Robustness Testing
- [ ] Add proper error handling in all components
- [ ] Implement fallback to single-threaded execution on errors
- [ ] Add extensive logging and debugging capabilities
- [ ] **ADD TESTS**: Error conditions, edge cases, memory constraints
- [ ] **STRESS TESTS**: Large models, long sequences, multi-NUMA systems
- [ ] **RUN FULL TEST SUITE**: Final comprehensive verification

### Phase 4: Performance Optimization (Priority: Low)

#### Step 4.1: Additional Operations + Testing
- [ ] Implement handlers for remaining operations (attention, normalization, activations)
- [ ] **ADD TESTS**: Each new operation with correctness and performance tests
- [ ] Profile and add handlers for remaining bottleneck operations
- [ ] **RUN FULL TEST SUITE**: After each new operation

#### Step 4.2: Performance Tuning + Validation
- [ ] Dynamic load balancing and optimal chunk size calculation
- [ ] Memory prefetching and thread affinity optimizations  
- [ ] **ADD TESTS**: Performance regression tests, optimization validation
- [ ] **FINAL TEST SUITE RUN**: Complete system verification

## 🛠️ **Implementation Strategy**

### Starting Point
1. **Complete replacement** of existing complex work item system
2. **Direct implementation** of simplified dispatcher-coordinator architecture
3. **No transition phase** - immediate adoption of new work item abstractions

### File Structure
```
ggml/src/
├── ggml-numa-dispatcher.c     # NEW: Parallelization intelligence
├── ggml-numa-coordinator.c    # MODIFIED: Simplified execution engine
├── ggml-work-item.h           # NEW: Work item abstractions
└── ggml-cpu.c                 # MODIFIED: Call dispatcher instead of coordinator
```

### Testing Strategy
- **Continuous Testing**: Add tests immediately after implementing each component
- **Test-Driven Development**: Write tests before or alongside implementation
- **Regression Prevention**: Run full test suite after every change
- **Multiple Test Types**: Unit tests, integration tests, performance tests, stress tests
- **Real-World Validation**: Test with actual models (qwen2.5-0.5b) at each step

## 🎯 **Success Criteria**

### Immediate (Phase 1-2)
- [ ] No crashes on ROPE operations
- [ ] Performance equal to or better than current implementation
- [ ] All existing functionality preserved

### Medium-term (Phase 3)
- [ ] Significant performance improvement on multi-NUMA systems
- [ ] Robust error handling and fallbacks
- [ ] Clean, maintainable codebase

### Long-term (Phase 4)
- [ ] Comprehensive operation coverage
- [ ] Optimal performance tuning
- [ ] Foundation for future NUMA optimizations

## 🚨 **Risk Mitigation**
- **Incremental implementation**: Can fall back to existing system at any point
- **Feature flags**: Easy to disable if issues arise
- **Comprehensive testing**: Catch issues early
- **Simple abstractions**: Easier to debug and maintain than complex registry system

## 📝 **Key Advantages of This Approach**
1. **Simplicity**: Work item abstraction is much simpler than operation registry
2. **Flexibility**: Easy to add new operations with just payload struct + worker function
3. **Maintainability**: Clear separation of concerns between dispatcher and coordinator
4. **Performance**: Leverages existing NUMA mirroring strategy for data locality
5. **Robustness**: Primary node fallback for non-parallelizable operations
