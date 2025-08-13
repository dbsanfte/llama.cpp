# NUMA Coordinator Refactor TODO - August 13, 2025

## 🚨 CRITICAL - Fix Segfault (Immediate)

### 1. Emergency ROPE Handler Implementation
- [ ] **Create single-node ROPE handler** to prevent crashes
  - Route all ROPE operations to NUMA node 0 only
  - Ensure proper ROPE cache initialization
  - Verify sin/cos theta pointers are valid
  - **Files to modify**: `ggml-numa-coordinator.c`
  - **Function**: Modify `ggml_numa_node_execute_operation()` to detect ROPE ops

### 2. Add Operation Safety Classification
- [ ] **Create operation whitelist/blacklist system**
  - Identify operations safe for naive parallelization
  - Blacklist problematic operations (ROPE, etc.) until proper handlers exist
  - Add safety checks before attempting NUMA parallelization
  - **Default**: Route unhandled operations to single node

## 🔧 HIGH PRIORITY - Core Infrastructure

### 3. Operation Registry System
- [ ] **Design and implement operation handler registry**
  - Create `ggml_numa_operation_handler` struct
  - Add operation registration system
  - Implement handler lookup by operation type
  - **Files to create**: `ggml-numa-operation-handlers.h/c`

### 4. Smart Dispatcher Implementation
- [ ] **Replace naive operation execution with intelligent dispatcher**
  - Analyze operation type before execution
  - Route to appropriate handler (single-node, data-parallel, custom)
  - Add comprehensive logging for debugging
  - Always provide fallback to single-node execution

### 5. GGML Operations Audit
- [ ] **Catalog all GGML operations from ggml-cpu.c**
  - Extract all `ggml_compute_forward_*` functions
  - Map `GGML_OP_*` enum values to compute functions
  - Classify by parallelization potential and requirements
  - **Deliverable**: Complete operation classification matrix

## 🚀 MEDIUM PRIORITY - Performance Optimizations

### 6. High-Impact Operation Handlers
- [ ] **Implement MUL_MAT handler** (matrix multiplication)
  - Row-wise or column-wise data parallelism
  - NUMA-local memory patterns
  - Performance benchmarking vs single-threaded
  
- [ ] **Implement element-wise operation handlers** (ADD, MUL, DIV, SUB)
  - Simple data parallelism with chunking
  - Minimal synchronization overhead
  - Easy wins for performance

### 7. Comprehensive Testing Framework
- [ ] **Create operation-specific tests**
  - Test each handler independently
  - Validate correctness vs single-threaded baseline
  - Performance regression detection
  - **Files to create**: `tests/test-numa-operation-handlers.cpp`

## 📊 ONGOING - Analysis & Optimization

### 8. Performance Monitoring
- [ ] **Add detailed performance metrics**
  - Operation-level timing
  - NUMA node utilization
  - Memory bandwidth utilization
  - Parallel efficiency measurements

### 9. Advanced Parallelization Strategies
- [ ] **Research complex operations** (Attention, LayerNorm, etc.)
  - Analyze mathematical structure for parallelization opportunities
  - Consider pipeline parallelism for dependent operations
  - Investigate operation fusion possibilities

## 🔄 REFACTOR PHASES

### Phase 1: Emergency Fix (Days 1-2)
1. Fix ROPE crash with single-node handler
2. Add operation safety checks
3. Prevent future crashes with unknown operations

### Phase 2: Infrastructure (Days 3-5) 
1. Build operation registry and dispatcher system
2. Complete GGML operations audit
3. Implement basic handler framework

### Phase 3: Performance (Days 6-10)
1. Implement high-impact operation handlers  
2. Add comprehensive testing
3. Performance optimization and tuning

### Phase 4: Completeness (Days 11+)
1. Handle all remaining operations
2. Advanced optimization strategies
3. Dynamic strategy selection

## ⚡ IMMEDIATE NEXT STEPS

1. **First**: Fix ROPE segfault with emergency single-node routing
2. **Then**: Build operation registry infrastructure  
3. **Finally**: Systematically add operation handlers

This approach ensures we:
- Fix the crash immediately ✅
- Build robust infrastructure for the long term ✅
- Deliver performance improvements incrementally ✅
- Maintain system stability throughout ✅
