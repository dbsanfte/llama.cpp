# Major Discovery: Complete ggml-cpu Operation Analysis & Big Bang Migration Strategy

**Date:** August 14, 2025  
**Context:** Task 4 - Dispatcher Architecture Analysis  
**Status:** 🔍 ANALYSIS COMPLETE - Strategy Pivot Required

## Critical Discovery: 193 Operations in ggml-cpu.c

### Investigation Summary

While analyzing integration points between `llama-context.cpp` and our dispatcher, discovered that `ggml-cpu.c` implements **193 unique operations** through a massive switch statement in `ggml_compute_forward()`. The existing NUMA coordinator was only partially implemented and doesn't handle the full operation set.

### ggml-cpu.c Architecture Analysis

**Main Flow:**
```
llama-context.cpp: graph_compute()
   ↓
ggml-cpu.c: ggml_graph_compute(cgraph, cplan) 
   ↓
ggml_graph_compute_thread() [per worker thread]
   ↓
for each node: ggml_compute_forward(params, tensor)
   ↓
switch(tensor->op): 193-operation dispatch table
```

**Key Integration Point:** `ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor)` - this is where ALL operations get dispatched.

**Current NUMA Integration:** ggml-cpu.c already has NUMA affinity code (`set_numa_thread_affinity()`) but uses its own threadpool system that conflicts with our coordinator.

## Strategic Decision: Big Bang Migration

### Why Selective Override Failed

Initial plan was to selectively override high-priority operations (ROPE, MUL_MAT) while falling back to existing implementations. However:

1. **Threadpool Conflicts**: ggml-cpu.c manages its own threadpool (`struct ggml_threadpool`) 
2. **NUMA Coordination Clash**: Cannot mix ggml-cpu threadpool with our NUMA coordinator threadpool
3. **Resource Conflicts**: Race conditions and synchronization nightmares between two threading systems
4. **Architectural Mess**: Trying to coordinate between two different concurrency models

### Big Bang Approach Benefits

1. **Clean Architecture**: Single threading/NUMA system under dispatcher control
2. **Complete NUMA Optimization**: Every operation can be NUMA-aware from day one
3. **Consistent Parallelization**: Dispatcher controls parallelization strategy per operation
4. **No Threading Conflicts**: Eliminates race conditions between competing threadpools
5. **Future-Proof**: Clean foundation for all future optimizations

## Complete Operation Migration TODO

### 193 Operations to Migrate (Alphabetical)

**Priority 1: Critical Operations (Crash-prone/Performance)**
- [ ] `GGML_OP_ROPE` - **URGENT**: Known crash source, sequence length splitting required
- [ ] `GGML_OP_MUL_MAT` - **HIGH**: Performance critical, heavy matrix computation
- [ ] `GGML_OP_FLASH_ATTN_EXT` - **HIGH**: Attention mechanism, performance critical
- [ ] `GGML_OP_MUL_MAT_ID` - **HIGH**: Matrix multiplication variant
- [ ] `GGML_OP_SOFT_MAX` - **HIGH**: Neural network activation
- [ ] `GGML_OP_RMS_NORM` - **HIGH**: Layer normalization

**Priority 2: Math Operations (Parallelizable)**
- [ ] `GGML_OP_ADD` - Element-wise addition
- [ ] `GGML_OP_ADD1` - Add scalar to tensor
- [ ] `GGML_OP_SUB` - Element-wise subtraction  
- [ ] `GGML_OP_MUL` - Element-wise multiplication
- [ ] `GGML_OP_DIV` - Element-wise division
- [ ] `GGML_OP_SQR` - Element-wise square
- [ ] `GGML_OP_SQRT` - Element-wise square root
- [ ] `GGML_OP_LOG` - Element-wise logarithm
- [ ] `GGML_OP_SIN` - Element-wise sine
- [ ] `GGML_OP_COS` - Element-wise cosine

**Priority 3: Neural Network Operations**
- [ ] `GGML_OP_GROUP_NORM` - Group normalization
- [ ] `GGML_OP_L2_NORM` - L2 normalization
- [ ] `GGML_OP_NORM` - General normalization
- [ ] `GGML_OP_RMS_NORM_BACK` - RMS norm backpropagation
- [ ] `GGML_OP_SOFT_MAX_BACK` - Softmax backpropagation
- [ ] `GGML_OP_CROSS_ENTROPY_LOSS` - Loss computation
- [ ] `GGML_OP_CROSS_ENTROPY_LOSS_BACK` - Loss backprop
- [ ] `GGML_OP_GLU` - Gated Linear Unit
- [ ] `GGML_OP_SILU_BACK` - SiLU backpropagation
- [ ] `GGML_OP_GATED_LINEAR_ATTN` - Gated attention

**Priority 4: Tensor Operations**
- [ ] `GGML_OP_DUP` - Tensor duplication
- [ ] `GGML_OP_CPY` - Tensor copy
- [ ] `GGML_OP_CONT` - Make tensor contiguous
- [ ] `GGML_OP_RESHAPE` - Tensor reshape
- [ ] `GGML_OP_VIEW` - Tensor view
- [ ] `GGML_OP_PERMUTE` - Tensor permutation
- [ ] `GGML_OP_TRANSPOSE` - Matrix transpose
- [ ] `GGML_OP_GET_ROWS` - Extract tensor rows
- [ ] `GGML_OP_GET_ROWS_BACK` - Get rows backprop
- [ ] `GGML_OP_SET_ROWS` - Set tensor rows
- [ ] `GGML_OP_REPEAT` - Tensor repetition
- [ ] `GGML_OP_REPEAT_BACK` - Repeat backprop
- [ ] `GGML_OP_CONCAT` - Tensor concatenation

**Priority 5: Specialized Operations**
- [ ] `GGML_OP_ROPE_BACK` - ROPE backpropagation
- [ ] `GGML_OP_OUT_PROD` - Outer product
- [ ] `GGML_OP_SCALE` - Tensor scaling
- [ ] `GGML_OP_SET` - Tensor assignment
- [ ] `GGML_OP_ACC` - Tensor accumulation
- [ ] `GGML_OP_SUM` - Tensor summation
- [ ] `GGML_OP_SUM_ROWS` - Row-wise summation
- [ ] `GGML_OP_MEAN` - Tensor mean
- [ ] `GGML_OP_ARGMAX` - Argmax operation
- [ ] `GGML_OP_COUNT_EQUAL` - Count equal elements

**Priority 6: Convolution Operations**
- [ ] `GGML_OP_CONV_2D` - 2D convolution
- [ ] `GGML_OP_CONV_2D_DW` - Depthwise convolution
- [ ] `GGML_OP_CONV_TRANSPOSE_1D` - 1D transpose convolution
- [ ] `GGML_OP_CONV_TRANSPOSE_2D` - 2D transpose convolution
- [ ] `GGML_OP_IM2COL` - Image to column
- [ ] `GGML_OP_IM2COL_BACK` - Im2col backprop
- [ ] `GGML_OP_POOL_1D` - 1D pooling
- [ ] `GGML_OP_POOL_2D` - 2D pooling
- [ ] `GGML_OP_POOL_2D_BACK` - 2D pooling backprop

**Priority 7: Advanced/Specialized Operations**
- [ ] `GGML_OP_FLASH_ATTN_BACK` - Flash attention backprop
- [ ] `GGML_OP_RWKV_WKV6` - RWKV WKV mechanism v6
- [ ] `GGML_OP_RWKV_WKV7` - RWKV WKV mechanism v7
- [ ] `GGML_OP_SSM_CONV` - State space model convolution
- [ ] `GGML_OP_SSM_SCAN` - State space model scan
- [ ] `GGML_OP_WIN_PART` - Window partition
- [ ] `GGML_OP_WIN_UNPART` - Window unpartition
- [ ] `GGML_OP_GET_REL_POS` - Get relative position
- [ ] `GGML_OP_ADD_REL_POS` - Add relative position
- [ ] `GGML_OP_TIMESTEP_EMBEDDING` - Timestep embedding

**Priority 8: Utility/Mask Operations**
- [ ] `GGML_OP_DIAG` - Diagonal extraction
- [ ] `GGML_OP_DIAG_MASK_INF` - Diagonal mask with infinity
- [ ] `GGML_OP_DIAG_MASK_ZERO` - Diagonal mask with zero
- [ ] `GGML_OP_CLAMP` - Value clamping
- [ ] `GGML_OP_LEAKY_RELU` - Leaky ReLU activation
- [ ] `GGML_OP_PAD` - Tensor padding  
- [ ] `GGML_OP_PAD_REFLECT_1D` - 1D reflection padding
- [ ] `GGML_OP_UPSCALE` - Tensor upscaling
- [ ] `GGML_OP_ARANGE` - Range generation
- [ ] `GGML_OP_ARGSORT` - Argument sorting
- [ ] `GGML_OP_COUNT` - Element counting
- [ ] `GGML_OP_ROLL` - Tensor rolling

**Priority 9: Custom/Optimization Operations**
- [ ] `GGML_OP_CUSTOM` - Custom operations
- [ ] `GGML_OP_MAP_CUSTOM1` - Custom mapping 1-ary
- [ ] `GGML_OP_MAP_CUSTOM2` - Custom mapping 2-ary  
- [ ] `GGML_OP_MAP_CUSTOM3` - Custom mapping 3-ary
- [ ] `GGML_OP_UNARY` - Generic unary operations
- [ ] `GGML_OP_OPT_STEP_ADAMW` - AdamW optimizer step

**Priority 10: Meta Operations**
- [ ] `GGML_OP_NONE` - No operation (pass-through)

## Dispatcher->Coordinator Interface Requirements

Based on 193 operation analysis, our interface must support:

### 1. Parallelization Flexibility
```c
typedef enum {
    GGML_PARALLEL_NONE,        // Single-threaded only
    GGML_PARALLEL_ELEMENT,     // Element-wise parallelizable
    GGML_PARALLEL_ROWS,        // Row-wise parallelizable  
    GGML_PARALLEL_SEQUENCE,    // Sequence-length splitting (ROPE)
    GGML_PARALLEL_BATCH,       // Batch dimension splitting
    GGML_PARALLEL_CUSTOM       // Operation-specific strategy
} ggml_parallel_strategy_t;
```

### 2. Work Item Data Structures
```c
typedef struct {
    struct ggml_compute_params * compute_params;
    struct ggml_tensor * tensor;
    ggml_parallel_strategy_t strategy;
    int chunk_start;
    int chunk_end;  
    void * operation_data;     // Operation-specific payload
} ggml_operation_work_item_data_t;
```

### 3. Operation Handler Registry
```c
typedef void (*ggml_operation_handler_t)(ggml_operation_work_item_data_t * work_data);

typedef struct {
    enum ggml_op operation;
    ggml_parallel_strategy_t default_strategy;
    ggml_operation_handler_t handler;
    bool requires_single_thread;
    bool requires_barrier;
} ggml_operation_info_t;
```

## Implementation Strategy

1. **Task 4**: Create dispatcher framework with operation registry and fallback system
2. **Task 5**: Implement ROPE handler (highest crash risk)
3. **Task 6**: Implement MUL_MAT handler (highest performance impact)
4. **Tasks 7-N**: Systematically migrate all 193 operations by priority

Each operation implementation will:
- Determine optimal parallelization strategy
- Split work appropriately for NUMA nodes
- Submit work items to coordinator 
- Handle operation-specific optimizations

## Expected Benefits

- **Complete NUMA Optimization**: All operations NUMA-aware
- **Unified Threading Model**: Single coordinator manages all parallelism
- **Crash Prevention**: Proper ROPE sequence splitting
- **Performance Gains**: Optimized parallelization per operation type
- **Clean Architecture**: No threadpool conflicts or race conditions

## Migration Commitment

This represents a significant undertaking - migrating 193 operations from ggml-cpu.c to our dispatcher-coordinator system. However, it provides the clean architectural foundation needed for robust NUMA-aware inference with no threading conflicts.

**Estimated Timeline:** 193 operations × ~2 hours avg = ~400 hours of implementation work, but can be parallelized across team members and incrementally tested.

**Success Criteria:** 
- All 193 operations migrated and tested
- Zero threading conflicts  
- ROPE crashes eliminated
- Performance parity or better vs original ggml-cpu.c
- Complete NUMA awareness across all operations
