# Copilot Instructions for llama.cpp

This document provides instructions for AI assistants working on the llama.cpp project with NUMA improvements.

## AI Agent Conversational Guidelines

As an AI agent working in this repo, you should observe the following behaviours:

* You are never overly sycophantic. You never reply with phrases like "You're absolutely right!" or similar. You maintain a neutral, fact-based conversational tone.

* You think critically about the user's asks and think deeply about hard technical problems. Sometimes the user is factually mistaken - if you believe this to be the case, you should reply with your perspective and ask for reconfirmation.

* You often make mistakes when calculating loop iterations and pointer arithmetic. You always think deeply about these hard technical problems. You favour reusing proven macros rather than writing loop arithmetic from scratch.

## 🎯 Project Overview

This is a fork of llama.cpp with **NUMA-aware execution architecture** for optimal CPU inferencing in a NUMA environment featuring kernel-based work buffer allocation:

- **NUMA Kernel Registry** - `ggml/src/ggml-cpu/numa-kernels/` - O(1) cache database with direct function pointer dispatch
- **NUMA Executor** - `ggml/src/ggml-cpu/ggml-numa-executor.c` - Strategy engine and work orchestration (work buffer calculation moved to kernels)
- **NUMA OpenMP Coordinator** - `ggml/src/ggml-cpu/ggml-numa-openmp-coordinator.c` - OpenMP-based resource management, work distribution, and kernel-based work buffer allocation
- **Dev Container** - Ubuntu 24.04 with pre-installed dependencies

**Goal**: Provide lightning-fast NUMA-aware execution for all operations through intelligent strategy selection and optimal resource utilization. 

**Architecture Flow**: `Compute Graph → Executor → Kernel Registry Direct Dispatch → OpenMP Coordinator Three-Strategy Dispatch → Kernel Work Buffer Allocation → OpenMP Parallel Regions`

## 🏗️ OpenMP Coordinator Architecture

The NUMA OpenMP coordinator uses a **three-strategy execution model** with OpenMP parallel regions for optimal performance:

### **Three Execution Strategies**

1. **Single-Thread/Single-Node**: `ggml_numa_openmp_execute_single_thread()`
   - **Use case**: Very small tensors (< 1K elements)
   - **Pattern**: One thread on target NUMA node, minimal overhead
   - **Data slicing**: No slicing - processes entire tensor
   - **OpenMP**: Single thread execution with CPU affinity

2. **Multi-Thread/Single-Node**: `ggml_numa_openmp_execute_single_node()`
   - **Use case**: Medium tensors (1K-256K elements)
   - **Pattern**: All threads on one NUMA node, shared memory locality
   - **Data slicing**: Thread-based slicing within single node
   - **OpenMP**: `#pragma omp parallel` region with NUMA-bound CPU affinity

3. **Multi-Thread/Multi-Node (Data-Parallel)**: `ggml_numa_openmp_execute_data_parallel()`
   - **Use case**: Large tensors (> 256K elements)
   - **Pattern**: All NUMA nodes participate, maximum parallelism
   - **Data slicing**: Both NUMA-level and thread-level slicing
   - **OpenMP**: Nested parallel regions or explicit thread binding per NUMA node

### **Thread-Local Context Variables**

The coordinator sets up thread-local variables that kernels use for data slicing:

```c
// NUMA node identification
extern __thread int ggml_current_numa_node;                    // Current NUMA node (0-based)

// Data-parallel execution context  
extern __thread bool ggml_numa_is_data_parallel_execution;     // True if multi-node execution
extern __thread int ggml_numa_total_nodes_for_data_parallel;   // Total nodes participating

// Shared memory optimization
extern __thread void * ggml_numa_shared_result_tensor_data;    // Direct result memory access
```

### **Data Slicing Mechanisms**

**NUMA-Level Slicing** (for data-parallel execution):
```c
if (ggml_numa_is_data_parallel_execution) {
    size_t total_elements = ggml_nelements(tensor);
    size_t elements_per_node = total_elements / ggml_numa_total_nodes_for_data_parallel;
    size_t numa_start = ggml_current_numa_node * elements_per_node;
    size_t numa_end = (ggml_current_numa_node == ggml_numa_total_nodes_for_data_parallel - 1) ? 
                      total_elements : numa_start + elements_per_node;
    
    // Process only this NUMA node's slice: numa_start to numa_end
}
```

**Thread-Level Slicing** (within each NUMA node):
```c
// Standard threading within the NUMA slice
int ith = params->ith;         // Thread index within this NUMA node
int nth = params->nth;         // Total threads on this NUMA node

size_t slice_elements = numa_end - numa_start;  // Elements for this NUMA node
size_t elements_per_thread = slice_elements / nth;
size_t thread_start = numa_start + (ith * elements_per_thread);
size_t thread_end = (ith == nth - 1) ? numa_end : thread_start + elements_per_thread;

// Process only this thread's slice: thread_start to thread_end
```

## 📋 Architecture Documentation

For comprehensive architecture details, see `docs/numa-architecture.md` which covers:
- Component interfaces and responsibilities
- Three-strategy execution model details
- Performance characteristics and benchmarks
- Development guidelines and best practices

## 🔧 NUMA Kernel Implementation Workflow

NOTE: Remember to use Doxygen comments for all new files, functions and structures. For @author you should use David Sanftenberg.

### Step 1: Analysis & Discovery
Find the operation and locate mathematical kernels:
```bash
grep -r "GGML_OP_YOUR_OPERATION" ggml/src/ggml-cpu/
```

**Kernel locations:**
- `ggml/src/ggml-cpu/ggml-cpu.c` - Main implementations
- `ggml/src/ggml-cpu/ops.h` - Operation interfaces
- `ggml/src/ggml-cpu/ggml-cpu-*` - Specialized kernels

**Target functions:** `ggml_compute_forward_your_operation()`, `ggml_compute_forward_your_operation_f32()`, etc.

**⚠️ Critical:** Extract pure mathematical kernels - avoid ggml threading logic that conflicts with NUMA coordinator! The NUMA coordinator does NOT use the ggml threading/threadpool scheme!!

**Operation Characteristics:**
- ✅ **Straightforward**: Element-wise ops (ADD, MUL, GLU) - independent computations, linear memory access
- ⚠️ **Complex**: Matrix ops (MUL_MAT), reductions (RMS_NORM) - need specialized splitting
- 🔧 **View Operations**: RESHAPE, PERMUTE - metadata-only, minimal/no computation (no-op)

**🚀 SIMD Optimization Requirements:**
- **Always use SIMD**: Replace scalar operations with `ggml_vec_*` functions from `ggml/src/ggml-cpu/vec.h`
- **Common SIMD functions**: `ggml_vec_add_f32()`, `ggml_vec_dot_f32()`, `ggml_vec_scale_f32()`, `ggml_vec_cpy_f32()`
- **Performance impact**: SIMD provides significant speedup on modern CPUs with AVX2/AVX512 support
- **Mathematical equivalence**: SIMD operations must produce identical results to scalar reference

**⚠️ Quantisation Type Support:** 
Ensure all tensor quant types supported by the underlying reference kernel are also supported in the NUMA kernel (F32, F16, Q8_0, Q4_0, Q5_0, etc). We must support everything in the NUMA kernel that the underlying reference kernel supports, because our goal is to replace it!

### Step 2: Modern Template Selection & Implementation with Composable Macros

**📚 NUMA Kernel Template System with Composable Macro Architecture:**
Choose the appropriate template and leverage our **Lego-like building block system** for efficient development:

**🔹 Element-wise Operations Template**: `numa-kernels/add.c` 
- **Use for**: ADD, MUL, SUB, DIV and similar simple element-wise operations
- **Pattern**: `NUMA_ROWWISE_KERNEL_SETUP()` composed macro for automatic slice setup
- **Characteristics**: Single-pass algorithms, no aggregation needed
- **Modern Implementation**: Full composable macro approach using atomic building blocks

**🔹 Sequence-wise Operations Template**: `numa-kernels/rope.c`
- **Use for**: ROPE, attention operations, sequence-based transformations  
- **Pattern**: **Hybrid approach** combining composable macros with custom logic
- **Characteristics**: Complex sequence processing (ne[1] * ne[2] * ne[3]), mathematical correctness requirements
- **Modern Implementation**: Basic composable macros for setup/validation + custom sequence-aware slicing

**🔹 Complex Operations Template**: `numa-kernels/mul_mat.c`
- **Use for**: Matrix multiplication, convolutions, complex transformations
- **Pattern**: Custom slicing with optional composable macro components
- **Characteristics**: Non-uniform memory access, chunk-based processing
- **Modern Implementation**: Multi-dimensional slicing with specialized access patterns

**🔹 Reduction Operations Template**: `numa-kernels/rms_norm.c`
- **Use for**: Normalization, statistical operations, dimension-wise reductions
- **Pattern**: `NUMA_ROWWISE_KERNEL_SETUP()` macro for row-wise processing 
- **Characteristics**: Multi-pass algorithms, cache-optimized access patterns  
- **Modern Implementation**: Full composable macro approach with proven validation

**🔹 View Operations Template**: `numa-kernels/reshape.c`
- **Use for**: RESHAPE, PERMUTE, VIEW, TRANSPOSE and similar metadata-only operations
- **Pattern**: Minimal setup, no actual computation
- **Characteristics**: Zero computational overhead, single-node execution
- **Modern Implementation**: `NUMA_OPENMP_STRATEGY_SINGLE_THREAD` with metadata-only logic

**🏗️ Revolutionary Composable Macro System:**
Our new **atomic building block architecture** provides Lego-like composability that eliminates boilerplate and ensures mathematical correctness:

**🎯 Success Story: ROPE Kernel Migration**
The ROPE kernel migration demonstrates the power of our hybrid approach:
- **Challenge**: Complex sequence-aware processing (ne[1] * ne[2] * ne[3]) with mathematical correctness requirements
- **Solution**: Hybrid approach combining composable macros with custom slicing logic
- **Results**: **32/32 tests passed (100% success rate)** - all ROPE variants, all execution strategies, all tensor sizes
- **Architecture**: Used `NUMA_INIT_CONTEXT`, `NUMA_VALIDATE_INPUTS`, `NUMA_BARRIER_AUTO` for setup/validation, preserved custom sequence-aware slicing for mathematical correctness
- **Lessons**: Complex kernels can successfully adopt composable benefits while preserving specialized mathematical logic

**🧱 Atomic Building Blocks (Direct Use):**
```c
// 1. Context initialization - Always first step
NUMA_INIT_CONTEXT(ctx, tensor, params);

// 2. Input validation - Critical for reliability
NUMA_VALIDATE_INPUTS(tensor, params);

// 3. Thread slicing - Fundamental operation
NUMA_SLICE_ROWS_ATOMIC(ctx, tensor, params);

// 4. Typed data access - Type-safe pointer extraction  
NUMA_GET_TYPED_POINTER(dst_data, tensor, float);
NUMA_GET_SOURCE_POINTER(src_data, tensor->src[0], float);

// 5. Synchronization - Essential for correctness
NUMA_BARRIER_AUTO();

// 6. Early exit handling - Performance optimization
NUMA_EARLY_EXIT_IF_NO_WORK(ctx);
```

**🏗️ Composed Templates (Recommended for Common Patterns):**
```c
// For simple row-wise operations (ADD, MUL, RMS_NORM)
NUMA_ROWWISE_KERNEL_SETUP(ctx, tensor, params, dst_data, float);

// For element-wise operations  
NUMA_ELEMENTWISE_KERNEL_SETUP(ctx, tensor, params, dst_data, float);

// For custom requirements
NUMA_CUSTOM_KERNEL_SETUP(ctx, tensor, params);
```

**🔧 Implementation Examples:**

**Simple Element-wise Operation (Full Composable Approach):**
```c
enum ggml_status ggml_numa_kernel_add_execute(void * work_context, struct ggml_compute_params * params) {
    struct ggml_tensor * tensor = (struct ggml_tensor *)work_context;
    
    // One-line setup handles everything
    ggml_numa_thread_context_t ctx;
    float * dst_data;
    NUMA_ROWWISE_KERNEL_SETUP(ctx, tensor, params, dst_data, float);
    
    // Extract source data and perform SIMD operation
    const float * src0_data, * src1_data;
    NUMA_GET_SOURCE_POINTER(src0_data, tensor->src[0], float);
    NUMA_GET_SOURCE_POINTER(src1_data, tensor->src[1], float);
    
    ggml_vec_add_f32(ctx.thread_end - ctx.thread_start,
                     dst_data + ctx.thread_start,
                     src0_data + ctx.thread_start, 
                     src1_data + ctx.thread_start);
    
    return GGML_STATUS_SUCCESS;
}
```

**Complex Operation (Hybrid Approach - ROPE Example):**
```c
enum ggml_status ggml_numa_kernel_rope_f32_execute(void * work_context, struct ggml_compute_params * params) {
    struct ggml_tensor * tensor = (struct ggml_tensor *)work_context;
    
    // Use composable macros for setup and validation
    ggml_numa_thread_context_t ctx;
    NUMA_INIT_CONTEXT(ctx, tensor, params);
    NUMA_VALIDATE_INPUTS(tensor, params);
    
    // Custom sequence-aware slicing for mathematical correctness
    const int64_t ne0 = tensor->ne[0], ne1 = tensor->ne[1], ne2 = tensor->ne[2], ne3 = tensor->ne[3];
    const int64_t nr = ne1 * ne2 * ne3;  // total sequences
    const int64_t ir0 = (nr * ctx.thread_id) / ctx.total_threads;
    const int64_t ir1 = (nr * (ctx.thread_id + 1)) / ctx.total_threads;
    
    if (ir0 >= ir1) {
        NUMA_BARRIER_AUTO();  // Still participate in barriers
        return GGML_STATUS_SUCCESS;
    }
    
    // Access data using composable macros
    float * dst_data;
    NUMA_GET_TYPED_POINTER(dst_data, tensor, float);
    const float * src_data;
    NUMA_GET_SOURCE_POINTER(src_data, tensor->src[0], float);
    
    // Custom ROPE processing logic for sequences ir0 to ir1
    for (int64_t i = ir0; i < ir1; i++) {
        // ... ROPE mathematical operations ...
    }
    
    return GGML_STATUS_SUCCESS;
}
```

**4D Rowwise Operation (Full Composable Approach with 4D Loop Pattern):**
```c
enum ggml_status ggml_numa_kernel_soft_max_execute(void * work_context, struct ggml_compute_params * params) {
    struct ggml_tensor * tensor = (struct ggml_tensor *)work_context;
    
    // Standard setup using composable macros
    ggml_numa_thread_context_t ctx;
    float * dst_data;
    NUMA_ROWWISE_KERNEL_SETUP(ctx, tensor, params, dst_data, float);
    
    const float * src_data;
    NUMA_GET_SOURCE_POINTER(src_data, tensor->src[0], float);
    
    // 4D rowwise loop pattern - processes outer dimensions completely,
    // distributes inner dimension (ne[1]) across threads using ctx.thread_start/thread_end
    NUMA_4D_ROWWISE_LOOP(tensor, ctx, {
        const int64_t row_offset = i03 * tensor->ne[2] * tensor->ne[1] * tensor->ne[0] +
                                   i02 * tensor->ne[1] * tensor->ne[0] +
                                   i01 * tensor->ne[0];
        
        // Softmax computation on the row from row_offset to row_offset + ne[0]
        ggml_vec_soft_max_f32(tensor->ne[0], dst_data + row_offset, src_data + row_offset);
    });
    
    return GGML_STATUS_SUCCESS;
}
```

**3D Threaded Operation (Multithreaded Type Conversion Pattern):**
```c
enum ggml_status ggml_numa_kernel_mul_mat_type_conversion(void * work_context, struct ggml_compute_params * params) {
    struct ggml_tensor * src1 = (struct ggml_tensor *)work_context;
    
    // Extract thread parameters for direct use in macro
    const int ith = params->ith;
    const int nth = params->nth;
    
    // Work buffer setup and type conversion function
    char * wdata = (char *)params->wdata;
    ggml_from_float_t const from_float = ggml_get_type_traits_cpu(target_type)->from_float;
    
    // 3D threaded loop pattern - distributes innermost dimension across threads
    NUMA_3D_THREADED_LOOP(src1, ith, nth, {
        const float * src1_row = (const float *)((char *)tensor_data(src1) + 
                                    i13*nb13 + i12*nb12 + i11*nb11);
        void * wdata_row = wdata + i13*nbw3 + i12*nbw2 + i11*nbw1;
        
        from_float(src1_row, wdata_row, ne10);
    });
    
    return GGML_STATUS_SUCCESS;
}
```

**Matrix Chunked Operation (Block-Tiled Processing Pattern):**
```c
enum ggml_status ggml_numa_kernel_mul_mat_computation(void * work_context, struct ggml_compute_params * params) {
    struct ggml_tensor * tensor = (struct ggml_tensor *)work_context;
    
    // Chunk parameters (from thread work distribution)
    const int64_t ir0_start = chunk_ir0_start, ir0_end = chunk_ir0_end;
    const int64_t ir1_start = chunk_ir1_start, ir1_end = chunk_ir1_end;
    const int64_t blck_0 = 16, blck_1 = 16;
    const int64_t num_rows_per_vec_dot = vec_dot_traits->nrows;
    
    // Matrix chunked loop pattern - processes blocks with vector dot optimization
    NUMA_MATRIX_CHUNKED_LOOP(ir0_start, ir0_end, ir1_start, ir1_end, 
                             blck_0, blck_1, num_rows_per_vec_dot, {
        // Calculate matrix coordinates from loop indices
        const int64_t i13 = (ir1 / (ne12 * ne1));
        const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
        const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);
        
        // Matrix computation with proper memory access patterns
        vec_dot_operation(src0_data, src1_data, dst_data, i11, i12, i13, iir0, ir1);
    });
    
    return GGML_STATUS_SUCCESS;
}
```

**🏆 Composable Macro Benefits:**
- **Lego-like Flexibility**: Mix and match atomic building blocks for any kernel complexity
- **Proven Patterns**: Composed templates handle 80% of common cases with one-line setup
- **Mathematical Correctness**: Hybrid approach preserves complex mathematical logic when needed  
- **Zero Maintenance**: Changes to core logic automatically propagate to all kernels
- **Consistent Behavior**: All kernels using composable macros behave identically
- **Performance**: Compile-time expansion with zero runtime overhead
- **Debugging**: Centralized debug logging in shared components
- **Barrier Safety**: Automatic OpenMP barrier handling prevents thread synchronization bugs

**🎯 When to Use Each Approach:**
- **Full Composable**: Simple operations (ADD, MUL, RMS_NORM) with standard row-wise processing
- **Hybrid Approach**: Complex operations (ROPE, matrix ops) requiring specialized mathematical logic
- **Atomic Building Blocks**: Advanced cases requiring custom control flow or unusual data access patterns

// 3. NUMA_GET_SHARED_DATA() - For accessing shared result memory
float * dst_data;
NUMA_GET_SHARED_DATA(tensor, dst_data, float);  // Handles shared memory logic automatically
```

**Shared Macro Benefits:**
- **Consistency**: All kernels use identical slice calculation logic
- **Error Prevention**: Built-in barrier handling and edge case management
- **Maintenance**: Changes to slicing logic only need updates in one place
- **Performance**: Compiled-time optimizations, zero runtime overhead
- **Debugging**: Centralized debug logging with `NUMA_LOG_SLICE_DEBUG()` macro

**NUMA Kernel Implementation Pattern with Composable Macros:**
```c
// Implement in numa-kernels/ directory using composable macros and appropriate template
enum ggml_status ggml_numa_kernel_your_operation_execute(void * work_context, struct ggml_compute_params * params) {
    struct ggml_tensor * tensor = (struct ggml_tensor *)work_context;
    
    // Choose implementation approach based on operation complexity:
    
    // APPROACH 1: Full Composable (Simple Operations - ADD, MUL, RMS_NORM)
    ggml_numa_thread_context_t ctx;
    float * dst_data;
    NUMA_ROWWISE_KERNEL_SETUP(ctx, tensor, params, dst_data, float);
    
    const float * src0_data, * src1_data;
    NUMA_GET_SOURCE_POINTER(src0_data, tensor->src[0], float);
    NUMA_GET_SOURCE_POINTER(src1_data, tensor->src[1], float);
    
    ggml_vec_add_f32(ctx.thread_end - ctx.thread_start,
                     dst_data + ctx.thread_start,
                     src0_data + ctx.thread_start, 
                     src1_data + ctx.thread_start);
    
    // APPROACH 2: Hybrid (Complex Operations - ROPE, Matrix Operations)
    // Use atomic building blocks for setup, custom logic for mathematical operations
    // ggml_numa_thread_context_t ctx;
    // NUMA_INIT_CONTEXT(ctx, tensor, params);
    // NUMA_VALIDATE_INPUTS(tensor, params);
    // 
    // // Custom mathematical logic here
    // const int64_t custom_start = calculate_custom_range_start(ctx);
    // const int64_t custom_end = calculate_custom_range_end(ctx);
    // 
    // if (custom_start >= custom_end) {
    //     NUMA_BARRIER_AUTO();
    //     return GGML_STATUS_SUCCESS;
    // }
    // 
    // float * dst_data;
    // NUMA_GET_TYPED_POINTER(dst_data, tensor, float);
    // // ... custom processing loop ...
    
    return GGML_STATUS_SUCCESS;
}
```

**Composable Macro Selection Guide:**
- **NUMA_ROWWISE_KERNEL_SETUP()**: Complete one-line setup for row-wise operations (ADD, MUL, RMS_NORM, etc.)
- **NUMA_ELEMENTWISE_KERNEL_SETUP()**: For element-wise operations with standard slicing
- **NUMA_CUSTOM_KERNEL_SETUP()**: For operations requiring manual control over setup process
- **Atomic building blocks**: For complex kernels requiring specialized mathematical logic (ROPE, matrix ops)
- **ggml_numa_thread_context_t**: Unified context structure with all thread and slice information  
- **Built-in barrier handling**: Threads with no work automatically participate in OpenMP barriers
- **Consistent debug logging**: `NUMA_LOG_TRACE()` provides standardized debug output

**Registry Integration:**
```c
// Step 1: Create register function in your kernel .c file (e.g., add.c, mul.c, etc.)
ggml_numa_kernel_registration_info_t ggml_numa_kernel_your_operation_register(void) {
    ggml_numa_kernel_registration_info_t info = {0};
    
    info.op_type = GGML_OP_YOUR_OPERATION;
    info.supported = true;
    info.kernel_name = "NUMA Your Operation Kernel";
    
    // Strategy thresholds for operation
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_SINGLE] = 1024;      // Single thread below 1K elements
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_MULTI] = 262144;     // Multi-thread below 256K elements
    // Above 256K elements: data-parallel strategy
    info.strategy_array.valid = true;
    
    // Function pointers for different strategies
    info.work_funcs.single_single_fn = ggml_numa_kernel_your_operation_execute;
    info.work_funcs.single_multi_fn = ggml_numa_kernel_your_operation_execute;
    info.work_funcs.data_parallel_fn = ggml_numa_kernel_your_operation_execute;
    info.work_funcs.valid = true;
    
    // Query function pointer - enables direct dispatch without switch statements
    info.query_fn = (void*)ggml_numa_kernel_your_operation_query;
    
    // Work buffer calculation function pointer - NEW ARCHITECTURE
    info.work_buffer_calc_fn = (void*)ggml_numa_kernel_your_operation_work_buffer_calc;
    
    // Most operations don't need aggregation functions
    info.agg_funcs.single_single_fn = NULL;
    info.agg_funcs.single_multi_fn = NULL; 
    info.agg_funcs.data_parallel_fn = NULL;
    info.agg_funcs.valid = false;
    
    return info;
}

// Step 2: Implement work buffer calculation function (if operation needs work buffers)
size_t ggml_numa_kernel_your_operation_work_buffer_calc(const struct ggml_tensor * tensor, int total_numa_nodes, int total_threads) {
    // Calculate per-thread work buffer size (e.g., cache, temporary arrays)
    const size_t cache_line_size_f32 = 16;  // CACHE_LINE_SIZE_F32 approximation
    const size_t per_thread_buffer = (tensor->ne[0] + cache_line_size_f32) * sizeof(float);
    
    // Return TOTAL work buffer size for ALL threads (coordinator will allocate this)
    return per_thread_buffer * total_threads;
}

// Step 3: Add function declarations to your kernel .h file (e.g., add.h, mul.h, etc.)
ggml_numa_kernel_registration_info_t ggml_numa_kernel_your_operation_register(void);
ggml_numa_execution_strategy_t ggml_numa_kernel_your_operation_query(const struct ggml_tensor * tensor);
size_t ggml_numa_kernel_your_operation_work_buffer_calc(const struct ggml_tensor * tensor, int total_numa_nodes, int total_threads);

// Step 4: Enable in numa-kernels.c using NUMA_REGISTER_KERNEL macro
void ggml_numa_kernels_init(void) {
    // ... other kernels ...
    
    // Use NUMA_REGISTER_KERNEL macro for automatic registration with direct dispatch
    NUMA_REGISTER_KERNEL(your_operation);
}
```

**🚀 NEW ARCHITECTURE: Direct Function Pointer Dispatch**
- **Eliminated Switch Statement**: Registry now uses direct function pointer calls for O(1) kernel query dispatch
- **Automatic Registration**: `NUMA_REGISTER_KERNEL()` macro automatically populates query function pointers
- **Zero Maintenance Overhead**: Adding new kernels requires NO changes to central dispatch logic
- **Cache-Optimized Lookup**: Single array access followed by direct function call: `entry->query_fn(tensor)`

**🏗️ NEW ARCHITECTURE: Kernel-Based Work Buffer Allocation**
- **Complexity Moved Out of Executor**: Work buffer calculation is now handled by individual kernels, not central executor switch statements
- **Kernel-Specific Logic**: Each kernel defines its own `work_buffer_calc_fn` that understands its specific memory requirements
- **Coordinator Allocation**: NUMA coordinator calls kernel's work buffer function and allocates total size for all threads
- **Thread-Local Access**: Work buffers are available to kernels via `params->wdata` with per-thread offsets
- **Simplified Executor**: Executor no longer needs operation-specific work buffer logic - just calls kernel's function
- **Scalable Design**: Adding new operations with work buffers requires no changes to executor code

**Work Buffer Implementation Pattern:**
```c
// Kernel defines its work buffer needs
size_t ggml_numa_kernel_rope_work_buffer_calc(const struct ggml_tensor * tensor, int total_numa_nodes, int total_threads) {
    const size_t ne0 = tensor->ne[0];
    const size_t cache_line_size_f32 = 16;  // CACHE_LINE_SIZE_F32 approximation
    const size_t per_thread_buffer = (ne0 + cache_line_size_f32) * sizeof(float);
    
    // Return TOTAL work buffer size for ALL threads
    return per_thread_buffer * total_threads;
}

// Coordinator allocates and kernel accesses
float * cache = (float *) params->wdata + (ne0 + CACHE_LINE_SIZE_F32) * ith;  // Thread-specific offset
```

**🚨 CRITICAL: Always use NUMA_REGISTER_KERNEL() macro**
- **Pattern**: `NUMA_REGISTER_KERNEL(kernel_name)` in `numa-kernels.c`
- **Consistent architecture**: All kernels use the same registration mechanism
- **Automatic validation**: Macro includes error checking and debug logging
- **Direct Dispatch**: Macro automatically populates function pointers
```

### Step 3: Testing
Use the mathematical correctness template:
```bash
cp tests/test-numa-mathematical-correctness-template.cpp tests/test-numa-mathematical-correctness-YOUR_OPERATION.cpp
```

**Required tests:**
- Multi-dimensional: TINY → GIGANTIC_16GB tensor sizes (now includes GB-scale support)
- Multi-threading: 1, 2, 4, 6, 8, 15, 16, 31, 32, 64, 128 threads
- Hardware-specific Data Parallel: Data parallel tests with all numas available on the machine using max thread counts per numa node
- Mathematical equivalence: Exact comparison with reference
- Add to CMake and verify with `cmake --build build --target test-numa-mathematical-correctness-YOUR_OPERATION`

## 🏗️ Current Architecture Status

**✅ Implemented Components:**
- **NUMA Kernel Registry** - O(1) hash table with threshold-based strategy selection
- **NUMA Executor** - Strategy engine with registry query-dispatch pattern  
- **NUMA Coordinator** - Resource management and work distribution

**✅ Supported Operations:**
- **ADD** - Element-wise addition with SIMD optimization using full composable macro approach (`NUMA_ROWWISE_KERNEL_SETUP`)
- **MUL** - Element-wise multiplication with optimized data-parallel execution using full composable macro approach
- **DIV** - Element-wise division with comprehensive quantization support using full composable macro approach  
- **SUB** - Element-wise subtraction with mathematical correctness validation using full composable macro approach
- **RMS_NORM** - Root Mean Square normalization using full composable macro approach with proven 100% test success rate
- **ROPE** - Rotary Position Embedding using **hybrid approach** combining composable macros (setup/validation) with custom sequence-aware slicing for mathematical correctness
- **NOOP** - No-operation for testing and debugging framework functionality

**🚀 Performance Characteristics:**
- **O(1) Strategy Lookups** - Direct function pointer dispatch eliminates all search overhead
- **Zero-Maintenance Dispatch** - No switch statements to update when adding new kernels
- **Threshold-Based Selection** - Simple element count thresholds for optimal strategy choice
- **NUMA-Aware Scheduling** - Optimal thread and memory placement
- **Shared Memory Optimization** - Direct memory writes eliminate aggregation overhead for most operations
- **3-Level Debug System** - Clean output at levels 1-2, detailed tracing at level 3
- **Shared Macro Architecture** - Consistent slicing logic across all kernels with centralized maintenance
- **Registry-Based Scalability** - Easy addition of new kernels with consistent patterns

**📊 Current System Status:**
- **Total Active Kernels**: 6 registered (ADD, MUL, DIV, SUB, RMS_NORM, ROPE, NOOP)  
- **Kernel Template Categories**: 5 types (Element-wise, Sequence-wise, Complex, Reduction, View operations)
- **Composable Macro System**: Revolutionary atomic building blocks with Lego-like composability for kernel development
  - **Atomic Building Blocks**: `NUMA_INIT_CONTEXT`, `NUMA_VALIDATE_INPUTS`, `NUMA_SLICE_ROWS_ATOMIC`, `NUMA_GET_TYPED_POINTER`, `NUMA_BARRIER_AUTO`, etc.
  - **Composed Templates**: `NUMA_ROWWISE_KERNEL_SETUP`, `NUMA_ELEMENTWISE_KERNEL_SETUP`, `NUMA_CUSTOM_KERNEL_SETUP` for common patterns
  - **Hybrid Approach**: Proven pattern for complex kernels (ROPE) combining composable macros with custom mathematical logic
- **Registry Architecture**: NUMA_REGISTER_KERNEL() macro with automatic query dispatch
- **Test Coverage**: Mathematical correctness and performance benchmarks with comprehensive test template, 100% success rate achieved for all implemented kernels

## 🏗️ Build Environment & Commands

**Always use the dev container** for consistency (Ubuntu 24.04 with all dependencies).

### Build Commands
```bash
# Configure debug build with NUMA support
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DGGML_NUMA_MIRROR=ON -DGGML_OPENMP=ON

# Build (let cmake auto-detect thread count)
cmake --build build --parallel
```

### Debug Environment Variable Control

**NUMA Debug Logging Control**: All NUMA debug messages are controlled by the `GGML_NUMA_DEBUG` environment variable:

```bash
# Clean, silent operation (default) - no debug messages
unset GGML_NUMA_DEBUG
# OR
export GGML_NUMA_DEBUG=0

# Enable debug messages - shows NUMA strategy decisions and execution details
export GGML_NUMA_DEBUG=1

# Enable verbose debug messages - shows additional internal details
export GGML_NUMA_DEBUG=2

# Enable trace debug messages - very detailed, includes individual operations
export GGML_NUMA_DEBUG=3
```

**Debug Message Categories:**
- **Executor Debug**: Strategy selection, kernel queries, execution paths
- **Coordinator Debug**: Thread management, NUMA binding verification, dispatch decisions
- **Kernel Debug**: Mathematical operation details, data slicing, SIMD optimizations
- **Memory Debug**: NUMA allocation successes, memory mirroring operations
- **Trace Debug**: Very detailed per-operation logging, individual tensor rows, thread work distribution

**Performance Impact**: When `GGML_NUMA_DEBUG` is unset or 0, all debug output is completely disabled, eliminating printf overhead during inference loops.

**Usage Examples:**
```bash
# Production inference - silent, maximum performance
./build/bin/llama-server -m model.gguf --numa mirror

# Development debugging - rich diagnostic output
GGML_NUMA_DEBUG=1 ./build/bin/llama-bench -m model.gguf --numa mirror

# Detailed troubleshooting - verbose internal details
GGML_NUMA_DEBUG=2 ./build/bin/llama-server -m model.gguf --numa mirror

# Ultra-detailed troubleshooting - trace level with individual operations
GGML_NUMA_DEBUG=3 ./build/bin/llama-server -m model.gguf --numa mirror

# Integration testing with debug output
GGML_NUMA_DEBUG=1 ./tests/run-numa-integration-test.sh --verbose --numa mirror

# Performance testing with debug
GGML_NUMA_DEBUG=2 ./tests/run-numa-performance-tests.sh --operation=ADD
```
```

### Quick Model Validation

**Automated Integration Test (Recommended)**
```bash
# Run the automated integration test with NUMA mirror mode
./tests/run-numa-integration-test.sh --numa mirror

# Run with debug output for troubleshooting
GGML_NUMA_DEBUG=1 ./tests/run-numa-integration-test.sh --verbose --numa mirror

# Test different NUMA modes
./tests/run-numa-integration-test.sh --numa distribute
./tests/run-numa-integration-test.sh --numa isolate

# Run without NUMA for baseline comparison
./tests/run-numa-integration-test.sh
```

The integration test automatically:
- Downloads the test model if needed
- Starts llama-server with proper configuration
- Passes through all environment variables (GGML_NUMA_DEBUG, GGML_LOG_DEBUG, etc.)
- Waits for server and model loading
- Sends a test prompt and validates response
- Cleans up server processes
- Reports success/failure with clear messages

**Manual Process (For Advanced Debugging)**
```bash
# Download test model (if needed)
wget -c -O ./.devcontainer/qwen2.5-0.5b-instruct-q8_0.gguf https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf

# Manual test with NUMA mirror mode:
./build/bin/llama-server -m ./.devcontainer/qwen2.5-0.5b-instruct-q8_0.gguf --host 0.0.0.0 --numa mirror --port 8080 &

while ! curl --fail --silent http://localhost:8080/; do sleep 1; done

curl -X POST http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "qwen2.5-0.5b-instruct", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 20}'

# (Check the JSON response looks sane with a proper model response, not garbage)

# Kill the server once done:
ps aux | grep llama-server | grep -v grep | awk '{print $2}' | xargs kill -9
```

## 🧠 Key Technical Areas

### NUMA Data Slicing & SIMD Optimization Patterns

**Successful SIMD Implementations:**
- **ADD Operation**: Element-wise addition using `ggml_vec_add_f32()` with data-parallel execution and no-aggregation optimization

**Pattern: Element-wise Operations (ADD, MUL, etc.)**
```c
// Data slicing for NUMA parallelism
size_t total_elements = ne[0] * ne[1] * ne[2] * ne[3];
size_t elements_per_node = total_elements / max_numa_nodes;
size_t numa_start = numa_node * elements_per_node;
size_t numa_end = (numa_node == max_numa_nodes - 1) ? total_elements : numa_start + elements_per_node;

// SIMD operation on NUMA slice
ggml_vec_add_f32(numa_end - numa_start, dst + numa_start, src0 + numa_start, src1 + numa_start);
```

**Pattern: Reduction Operations (RMS_NORM, SOFT_MAX, etc.)**
```c
// Registry pattern - may need aggregation for reductions
ggml_numa_kernel_registration_info_t info = {
    .op_type = GGML_OP_RMS_NORM,
    .strategy_array = {
        .thresholds = {1024, 65536},  // Smaller thresholds for reductions
        .valid = true
    },
    .work_funcs = {
        .single_single_fn = ggml_numa_kernel_rms_norm_execute,
        .single_multi_fn = ggml_numa_kernel_rms_norm_execute,
        .data_parallel_fn = ggml_numa_kernel_rms_norm_execute,
        .valid = true
    },
    .agg_funcs = {
**Critical SIMD Requirements:**
- Always prefer `ggml_vec_*` over manual loops
- Ensure mathematical equivalence with reference implementation
- Validate with comprehensive multi-dimensional testing
- Handle edge cases (uneven NUMA splits, remainder elements)

### GB-Scale Tensor Infrastructure & Shared Memory Optimization

**Shared Memory Optimization:**
For large tensors (1GB+), the shared memory approach provides significant performance improvements:
- **Direct memory writes** to final tensor memory locations
- **Zero-copy architecture** with proper NUMA memory locality
- **Eliminates aggregation overhead** by writing directly to shared result tensor

**Pattern: Shared Memory Kernel Implementation with Composable Macros**
Note: F32 example is shown but all quant types in `quants.c` must be supported.

```c
// Modern shared memory kernel using composable macros
enum ggml_status ggml_numa_kernel_add_execute(void * work_context, struct ggml_compute_params * params) {
    struct ggml_tensor * tensor = (struct ggml_tensor *)work_context;
    
    // Complete setup with composable macro - handles shared memory automatically
    ggml_numa_thread_context_t ctx;
    float * dst_data;
    NUMA_ROWWISE_KERNEL_SETUP(ctx, tensor, params, dst_data, float);
    
    // Extract source data using atomic building blocks
    const float * src0_data, * src1_data;
    NUMA_GET_SOURCE_POINTER(src0_data, tensor->src[0], float);
    NUMA_GET_SOURCE_POINTER(src1_data, tensor->src[1], float);
    
    // Direct SIMD operation on shared memory (no aggregation needed)
    ggml_vec_add_f32(ctx.thread_end - ctx.thread_start, 
                     dst_data + ctx.thread_start, 
                     src0_data + ctx.thread_start, 
                     src1_data + ctx.thread_start);
    
    return GGML_STATUS_SUCCESS;
}
```
**GB-Scale Tensor Dimension Calculation:**
```c
// For GB-scale test tensors, use cube root for balanced dimensions
size_t gb_elements = gb_size * 268435456; // ~1GB = 268M float32 elements
size_t cube_root = (size_t)cbrt((double)gb_elements);
tensor_dims = {cube_root, cube_root, cube_root, 1}; // Balanced 3D tensor
```

### NUMA Memory Management
- **Files**: `ggml-numa-openmp-coordinator.c`, `ggml-cpu-numa-buffer.cpp`, `ggml.h`
- **NUMA mirroring**: Use `tensor_data()`/`tensor_set_data()` for NUMA-aware access
- **Memory allocation**: Always use `numa_alloc_onnode()` for local allocation
- **OpenMP thread management**: CPU affinity and NUMA binding through OpenMP parallel regions

## � Debugging

```bash
# Strategy 1: Direct GDB backtrace
gdb --batch --ex run --ex bt --ex quit --args ./build/bin/test-program

# Strategy 2: Core dump analysis
echo 'core' | sudo tee /proc/sys/kernel/core_pattern
ulimit -c unlimited
# Run program, then analyze core dump:
gdb --batch --ex "file ./build/bin/program" --ex "core-file ./core" --ex "bt" --ex quit
```

## 🧪 Testing Requirements

**Critical**: ALL NUMA tests must pass before changes are complete.

```bash
# Test core components
cmake --build build --target test-numa-mathematical-correctness-add

# Run comprehensive test suite (includes integration test at end)
./tests/run-numa-tests.sh

# Run performance tests
./tests/run-numa-performance-tests.sh --operation=ADD --quick

# Run integration test separately
./tests/run-numa-integration-test.sh --numa mirror
```

### Test Template Usage
```bash
# Copy template for new operations
cp tests/test-numa-mathematical-correctness-template.cpp tests/test-numa-mathematical-correctness-OPERATION.cpp

# Features: 3-part test structure with comprehensive quantization coverage
# 1. Mathematical equivalence testing (multi-dimensional tensors, multi-threading validation)  
# 2. Quantization type coverage (Q8_0, Q4_0, Q5_0, F16, F32 - prevents silent model failures)
# 3. Regression testing (operation-specific edge cases and previous bug scenarios)
# Always add tests to CMakeLists.txt and verify builds successfully
```

**Critical**: ALL mathematical correctness tests MUST include comprehensive quantization type coverage. Missing quantization testing can lead to silent model inference failures in production. The template enforces this requirement with detailed TODOs and examples.

## 🏗️ OpenMP Coordinator Architecture Summary

The current NUMA architecture uses a **three-strategy execution model** with OpenMP parallel regions optimized for different computational workload sizes:

### **Strategy Selection & Data Slicing**

**Automatic Strategy Selection:**
```c
// Registry provides simple threshold-based strategy selection
if (total_elements < 1024) {
    // Strategy 1: Single-thread/single-node (minimal overhead)
    strategy = NUMA_STRATEGY_SINGLE_SINGLE;
} else if (total_elements < 262144) {
    // Strategy 2: Multi-thread/single-node (shared memory locality) 
    strategy = NUMA_STRATEGY_SINGLE_MULTI;
} else {
    // Strategy 3: Multi-thread/multi-node (maximum parallelism)
    strategy = NUMA_STRATEGY_DATA_PARALLEL;
}
```

**Modern Kernel Data Slicing with Composable Macros:**
```c
// Modern approach: Use composable macros for different complexity levels
enum ggml_status ggml_numa_kernel_operation_execute(void * work_context, struct ggml_compute_params * params) {
    struct ggml_tensor * tensor = (struct ggml_tensor *)work_context;
    
    // APPROACH 1: Full Composable (Simple operations - ADD, MUL, RMS_NORM)
    ggml_numa_thread_context_t ctx;
    float * dst_data;
    NUMA_ROWWISE_KERNEL_SETUP(ctx, tensor, params, dst_data, float);
    
    const float * src_data;
    NUMA_GET_SOURCE_POINTER(src_data, tensor->src[0], float);
    ggml_vec_operation(ctx.thread_end - ctx.thread_start, 
                       dst_data + ctx.thread_start, 
                       src_data + ctx.thread_start);
    
    // APPROACH 2: Hybrid (Complex operations - ROPE, matrix operations)
    // ggml_numa_thread_context_t ctx;
    // NUMA_INIT_CONTEXT(ctx, tensor, params);
    // NUMA_VALIDATE_INPUTS(tensor, params);
    // 
    // // Custom mathematical logic for sequence-aware processing
    // const int64_t custom_start = calculate_sequence_range_start(ctx);
    // const int64_t custom_end = calculate_sequence_range_end(ctx);
    // 
    // if (custom_start >= custom_end) {
    //     NUMA_BARRIER_AUTO();  // Still participate in barriers
    //     return GGML_STATUS_SUCCESS;
    // }
    // 
    // float * dst_data;
    // NUMA_GET_TYPED_POINTER(dst_data, tensor, float);
    // // ... custom processing loop for mathematical correctness ...
    
    return GGML_STATUS_SUCCESS;
}
```

**Composable Macro Architecture Advantages:**
- **Lego-like Flexibility**: Mix atomic building blocks for any complexity level
- **Proven Patterns**: Composed templates handle 80% of kernels with one-line setup  
- **Mathematical Correctness**: Hybrid approach preserves complex logic when needed (ROPE success: 32/32 tests passed)
- **Zero Maintenance**: Changes to atomic blocks automatically propagate everywhere
- **Consistent Behavior**: All composable components use identical underlying logic
- **Performance**: Compile-time macro expansion with zero runtime overhead

**Execution Flow:**
1. **Registry Query**: O(1) threshold lookup determines optimal strategy
2. **Coordinator Dispatch**: Maps strategy to one of three execution functions
3. **Thread-Local Setup**: Coordinator sets NUMA context variables for kernels
4. **Dual-Level Slicing**: Kernels adapt behavior based on execution strategy
5. **SIMD Execution**: Optimized computation on thread's data slice

**Performance Benefits:**
- **Automatic Optimization**: No manual tuning - thresholds handle strategy selection
- **Minimal Overhead**: Single-thread strategy for tiny tensors avoids threading costs
- **Shared Memory Locality**: Multi-thread single-node maximizes cache efficiency
- **Maximum Parallelism**: Data-parallel strategy utilizes all NUMA resources
- **Zero-Copy Architecture**: Shared memory optimization eliminates aggregation overhead

## 💡 AI Agent Guidelines

### Critical Workflow
```bash
# 1. Verify clean starting state
cmake --build build --target ggml-cpu llama common && echo "✅ Core components building"

# 2. Make changes incrementally with testing
cmake --build build --parallel
./build/bin/test-numa-mathematical-correctness-OPERATION

# 3. Final validation - core architecture must build
cmake --build build --target ggml-cpu llama && echo "🎉 Complete!" || echo "❌ Fix failures"
```

### Best Practices
- **Always use dev container** - consistent environment
- **Test incrementally** - build and test after each change
- **Add comprehensive tests** - feature isn't done without tests
- **Validate with real models** - not just compilation
- **Check exit codes** - tests must properly signal failures
- **SIMD First**: Always use `ggml_vec_*` functions instead of scalar loops
- **Use Composable Macros**: Leverage the revolutionary atomic building block system:
  - **Full Composable**: `NUMA_ROWWISE_KERNEL_SETUP()` for simple operations (ADD, MUL, RMS_NORM)
  - **Hybrid Approach**: Atomic building blocks + custom logic for complex operations (ROPE, matrix ops)
  - **Atomic Building Blocks**: `NUMA_INIT_CONTEXT`, `NUMA_VALIDATE_INPUTS`, `NUMA_BARRIER_AUTO` for advanced control
- **Registry Integration**: Add cache entries for all complexity classes
- **Architecture Flow**: Follow Executor → Registry Direct Dispatch → Coordinator pattern
- **Debug Control**: Use `GGML_NUMA_DEBUG=1` for development debugging, unset for performance testing
- **Kernel Registration**: Always use `NUMA_REGISTER_KERNEL()` macro, never legacy function-based registration
- **Strategy Selection**: Use `NUMA_SELECT_STRATEGY_FROM_CACHE()` macro for unified threshold-based strategy selection

### Modern Composable Macro Implementation Pattern
All new kernels should use the composable macro system for consistency and maintainability:
```c
// Choose appropriate approach based on operation complexity:

// APPROACH 1: Full Composable (Simple operations - ADD, MUL, RMS_NORM)
NUMA_ROWWISE_KERNEL_SETUP(ctx, tensor, params, dst_data, float);     // One-line complete setup

// APPROACH 2: Hybrid Approach (Complex operations - ROPE, matrix ops)  
NUMA_INIT_CONTEXT(ctx, tensor, params);                              // Context initialization
NUMA_VALIDATE_INPUTS(tensor, params);                                // Input validation
// ... custom mathematical logic ...
NUMA_BARRIER_AUTO();                                                 // Barrier handling

// APPROACH 3: Atomic Building Blocks (Advanced custom requirements)
NUMA_SLICE_ROWS_ATOMIC(ctx, tensor, params);                         // Manual slicing
NUMA_GET_TYPED_POINTER(dst_data, tensor, float);                     // Type-safe access
NUMA_EARLY_EXIT_IF_NO_WORK(ctx);                                     // Performance optimization
```

**Benefits:**
- **Lego-like Composability**: Mix and match atomic building blocks for any kernel complexity
- **Proven Patterns**: Composed templates handle 80% of common cases with one-line setup
- **Mathematical Correctness**: Hybrid approach preserves complex logic when needed (ROPE: 32/32 tests passed)
- **Zero Maintenance**: Changes to atomic blocks automatically propagate everywhere
- **Consistent Behavior**: All composable components use identical underlying logic
- **Zero Performance Impact**: Macros expand to identical code at compile time  
- **Built-in Safety**: Automatic barrier handling and edge case management

### Debug Message Implementation
When adding new NUMA components, always use the centralized debug control system:
```c
// Include the debug header
#include "ggml-numa-shared.h"

// Use controlled debug macros instead of printf
NUMA_LOG_DEBUG("Your debug message: %d\n", value);
NUMA_LOG_VERBOSE("Detailed debug info: %f\n", detail);

// Never use printf directly for debug messages - use controlled macros
// printf("Debug: ..."); // ❌ DON'T DO THIS
// NUMA_LOG_DEBUG("Debug: ..."); // ✅ DO THIS
```

## 📋 Quick Reference

### Essential Files
```
ggml/src/ggml-cpu/numa-kernels/numa-kernels.c     # Kernel registry
ggml/src/ggml-cpu/ggml-numa-executor.c            # Strategy engine and orchestration
ggml/src/ggml-cpu/ggml-numa-openmp-coordinator.c  # OpenMP-based NUMA management
ggml/src/ggml-cpu/ggml-numa-shared.h              # Debug control and shared utilities
ggml/src/ggml-cpu/ggml-numa-perf.c                # Performance instrumentation framework
ggml/src/ggml-cpu/ggml-cpu.c                      # Mathematical kernels (reference)
tests/test-numa-mathematical-correctness-*.cpp    # Correctness tests
tests/run-numa-performance-tests.sh               # Performance test orchestrator
tests/test-numa-execution-modes.cpp               # Runs individual op performance tests for the Performance test orchestrator across a variety of NUMA modes
docs/numa-architecture.md                         # Architecture documentation

# NUMA Kernel Templates (Choose Based on Operation Type)
ggml/src/ggml-cpu/numa-kernels/add.c              # Template: Element-wise operations (ADD, MUL, SUB, DIV)
ggml/src/ggml-cpu/numa-kernels/rope.c             # Template: Sequence-wise operations (ROPE, attention ops)
ggml/src/ggml-cpu/numa-kernels/mul_mat.c          # Template: Complex operations & matrix ops
ggml/src/ggml-cpu/numa-kernels/rms_norm.c         # Template: Reduction operations & normalizations
ggml/src/ggml-cpu/numa-kernels/reshape.c          # Template: View operations & metadata transformations
tests/test-numa-mathematical-correctness-template.cpp  # Comprehensive test template for new operations
```

### Implementation Checklist
- [ ] Find mathematical kernel in `ggml-cpu.c`
- [ ] **Choose appropriate template**: Element-wise (add.c), Sequence-wise (rope.c), Matrix (mul_mat.c), Reduction (rms_norm.c), or View (reshape.c)
- [ ] Extract pure mathematical operations (no ggml threading)
- [ ] Replace scalar loops with SIMD `ggml_vec_*` functions
- [ ] **Copy template and adapt** for your operation type
- [ ] Extract pure mathematical operations (no ggml threading)
- [ ] Replace scalar loops with SIMD `ggml_vec_*` functions
- [ ] **Choose implementation approach**:
  - [ ] **Full Composable**: For simple operations (ADD, MUL, RMS_NORM) use `NUMA_ROWWISE_KERNEL_SETUP()` 
  - [ ] **Hybrid Approach**: For complex operations (ROPE, matrix ops) use atomic building blocks + custom logic
- [ ] Implement kernel function in `numa-kernels/` directory using chosen approach:
  - [ ] Use `NUMA_ROWWISE_KERNEL_SETUP()` for simple row-wise operations  
  - [ ] Use `NUMA_ELEMENTWISE_KERNEL_SETUP()` for element-wise operations
  - [ ] Use atomic building blocks (`NUMA_INIT_CONTEXT`, `NUMA_VALIDATE_INPUTS`, etc.) for complex operations
  - [ ] Use `NUMA_GET_TYPED_POINTER()`/`NUMA_GET_SOURCE_POINTER()` for type-safe data access
- [ ] Ensure proper barrier handling with `NUMA_BARRIER_AUTO()` for custom implementations
- [ ] Check `ggml_numa_shared_result_tensor_data` for direct writes (shared memory optimization)
- [ ] Create `ggml_numa_kernel_{operation}_register()` function that returns registration info
- [ ] Create `ggml_numa_kernel_{operation}_query()` function using `NUMA_SELECT_STRATEGY_FROM_CACHE()` macro
- [ ] **Implement work buffer calculation function** if operation needs temporary storage (cache, arrays, etc.)
- [ ] Add function declarations to kernel header file (e.g., `add.h`, `mul.h`) including work buffer calc function
- [ ] Enable in `numa-kernels.c` using `NUMA_REGISTER_KERNEL(operation)` macro
- [ ] Use `NUMA_ASSERT` for validation with proper coordinator signaling
- [ ] Use `NUMA_LOG_DEBUG` macros instead of printf for debug messages
- [ ] Create test from mathematical correctness template with multi-dimensional validation
- [ ] Add to CMake and verify builds successfully
- [ ] Verify core architecture builds: `cmake --build build --target ggml-cpu llama`
- [ ] Add the new test to `tests/run-numa-tests.sh` and verify it and the entire suite passes
- [ ] Run integration test to validate real-world functionality: `./tests/run-numa-integration-test.sh --numa mirror`


### Performance Commands
```bash
cmake --build build --target test-numa-mathematical-correctness-OPERATION
./build/bin/test-numa-mathematical-correctness-OPERATION
cmake --build build --target ggml-cpu llama common  # Core validation

# Run GB-scale performance tests
./tests/run-numa-performance-tests.sh --operation=OPERATION

# Run integration test with real model
./tests/run-numa-integration-test.sh --numa mirror
```

## Changelog
Document completed tasks in `.devcontainer/changelog/YYYY-MM-DD-description.md`. Always use `date +%Y-%m-%d` to fetch the current date.