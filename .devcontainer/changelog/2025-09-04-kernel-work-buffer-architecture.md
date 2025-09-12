# Kernel-Based Work Buffer Allocation Architecture

**Date**: September 4, 2025  
**Type**: Architecture Redesign  
**Impact**: Major - Fundamental improvement to NUMA kernel work buffer management

## Summary

Implemented a comprehensive redesign of work buffer allocation, moving complexity from the executor to individual kernels. This architectural change eliminates central switch statements, improves scalability, and provides kernel-specific memory management autonomy.

## Key Changes

### 🏗️ **Work Buffer Architecture Redesign**

**Before (Executor-Centralized)**:
- Executor contained operation-specific work buffer calculation logic
- Central switch statements for each operation type
- Fixed per-thread work buffer sizes
- Maintenance overhead when adding new operations

**After (Kernel-Distributed)**:
- Each kernel defines its own `work_buffer_calc_fn`
- Kernels calculate total work buffer size for all threads
- Coordinator allocates and distributes work buffers with per-thread offsets
- Zero maintenance overhead for new operations

### 🚀 **Implementation Details**

#### **Kernel Registration Updates**
```c
// NEW: Work buffer calculation function pointer
info.work_buffer_calc_fn = (void*)ggml_numa_kernel_rope_work_buffer_calc;

// Example kernel-specific work buffer calculation
size_t ggml_numa_kernel_rope_work_buffer_calc(const struct ggml_tensor * tensor, int total_numa_nodes, int total_threads) {
    const size_t cache_line_size_f32 = 16;
    const size_t per_thread_buffer = (tensor->ne[0] + cache_line_size_f32) * sizeof(float);
    return per_thread_buffer * total_threads;  // Return TOTAL size
}
```

#### **Executor Simplification**
```c
// NEW: Simplified executor logic
enum ggml_status ggml_numa_executor_execute_tensor(struct ggml_tensor * tensor, struct ggml_cplan * cplan) {
    ggml_numa_kernel_query_result_t query_result = ggml_numa_kernels_query(tensor);
    
    // Kernel calculates its own work buffer requirements
    size_t work_buffer_size = 0;
    if (query_result.work_buffer_calc_fn) {
        work_buffer_size = query_result.work_buffer_calc_fn(tensor, total_numa_nodes, total_threads);
    }
    
    // Pass total size to coordinator
    return ggml_numa_simple_coordinator_compute_forward(tensor, strategy, work_function, agg_function, work_buffer_size, cplan);
}
```

#### **Coordinator Work Buffer Management**
```c
// NEW: Coordinator handles NUMA-aware allocation
enum ggml_status ggml_numa_simple_coordinator_compute_forward(
    struct ggml_tensor * tensor,
    ggml_numa_execution_strategy_t strategy,
    ggml_numa_work_function_t work_function,
    ggml_numa_aggregation_function_t agg_function,
    size_t total_work_buffer_size,  // NEW: Total for all threads
    struct ggml_cplan * cplan
) {
    void * work_buffer_data = NULL;
    if (total_work_buffer_size > 0) {
        work_buffer_data = allocate_numa_work_buffers(total_work_buffer_size, cplan->n_threads);
    }
    // ... dispatch to strategy functions
}
```

## Benefits

### **🎯 Scalability**
- **Zero maintenance overhead**: Adding new operations with work buffers requires no changes to executor
- **Kernel autonomy**: Each kernel manages its own memory requirements
- **Consistent patterns**: All kernels follow the same work buffer calculation interface

### **🚀 Performance**
- **NUMA-aware allocation**: Work buffers allocated on optimal NUMA nodes
- **Thread-local access**: Efficient per-thread work buffer offsets
- **Reduced complexity**: Elimination of central dispatch switch statements

### **🔧 Maintainability**
- **Localized logic**: Work buffer requirements co-located with kernel implementation
- **Type safety**: Kernel-specific work buffer calculation functions
- **Clear interfaces**: Well-defined function signatures for work buffer calculation

## Implementation Example: ROPE Kernel

The ROPE kernel implementation demonstrates the new architecture:

```c
// Work buffer calculation function
size_t ggml_numa_kernel_rope_work_buffer_calc(const struct ggml_tensor * tensor, int total_numa_nodes, int total_threads) {
    const size_t ne0 = tensor->ne[0];
    const size_t cache_line_size_f32 = 16;
    const size_t per_thread_buffer = (ne0 + cache_line_size_f32) * sizeof(float);
    return per_thread_buffer * total_threads;
}

// Kernel execution with work buffer access
enum ggml_status ggml_numa_kernel_rope_execute(void * work_context, struct ggml_compute_params * params) {
    // Access thread-specific work buffer portion
    float * cache = (float *) params->wdata + (ne0 + CACHE_LINE_SIZE_F32) * ith;
    
    // Use cache for ROPE calculations...
}

// Registration with work buffer function
ggml_numa_kernel_registration_info_t ggml_numa_kernel_rope_register(void) {
    ggml_numa_kernel_registration_info_t info = {0};
    // ... other setup ...
    info.work_buffer_calc_fn = (void*)ggml_numa_kernel_rope_work_buffer_calc;
    return info;
}
```

## Files Modified

### **Core Architecture**
- `ggml/src/ggml-cpu/ggml-numa-shared.h` - Added work buffer calc function pointer type
- `ggml/src/ggml-cpu/ggml-numa-executor.c` - Simplified executor with kernel-based work buffer calls
- `ggml/src/ggml-cpu/ggml-numa-simple-coordinator.c` - Enhanced work buffer allocation and distribution

### **Kernel Implementation**
- `ggml/src/ggml-cpu/numa-kernels/rope.c` - Complete ROPE kernel with work buffer calculation
- `ggml/src/ggml-cpu/numa-kernels/rope.h` - Work buffer function declarations

### **Documentation**
- `.github/copilot-instructions.md` - Updated with new work buffer architecture patterns
- `docs/numa-architecture.md` - Comprehensive architecture documentation updates

### **Testing**
- `tests/test-numa-mathematical-correctness-rope.cpp` - Updated to 3-stage execution methodology
- Fixed mathematical precision issues in ROPE cache initialization
- Achieved 100% test success rate (50/50 tests passing)

## Testing Results

**Before**: 70.4% success rate with mathematical mismatches and execution failures  
**After**: **100% success rate** - Complete mathematical correctness across all tensor sizes and execution strategies

**Key Fixes**:
1. **Test Methodology**: Updated from artificial thread constraints to proper 3-stage NUMA execution
2. **Mathematical Precision**: Fixed floating-point accumulation errors in ROPE cache initialization
3. **Work Buffer Allocation**: Resolved segmentation faults through proper kernel-based work buffer calculation

## Future Impact

This architectural change provides a solid foundation for:
- **Rapid kernel development**: Simplified work buffer management for new operations
- **Performance optimization**: NUMA-aware work buffer allocation patterns
- **Code maintainability**: Localized kernel logic without central dependencies
- **Testing reliability**: Consistent execution patterns aligned with coordinator strategies

The kernel-based work buffer architecture represents a significant step forward in NUMA execution efficiency and developer productivity.
