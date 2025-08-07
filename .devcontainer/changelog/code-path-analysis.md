# Code Path Analysis: Context/Tensor Lifecycle

## Problem Summary
- First context/tensor: Normal heap memory (`0x7fb94e0021c0`) ✅
- Second context/tensor: Virtual memory (`0x35aa12cab580`) ❌ SEGFAULT

## Code Path Mapping

### 1. First Context Creation: `ggml_init()`
```c
// ggml/src/ggml.c
struct ggml_context * ggml_init(struct ggml_init_params params) {
    // Allocate context memory
    ctx->mem_buffer = malloc(params.mem_size);  // Normal heap allocation
    ctx->mem_size = params.mem_size;
    // Initialize context state
}
```

### 2. First Tensor Creation: `ggml_new_tensor_2d()` → `ggml_new_tensor_impl()`
```c
// ggml/src/ggml.c:1616
static struct ggml_tensor * ggml_new_tensor_impl(...) {
    // Calculate data size
    size_t data_size = ggml_row_size(type, ne[0]) * ne[1];
    
    // Allocate tensor object + data in context memory pool
    struct ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_TENSOR, 
                                               GGML_TENSOR_SIZE + data_size);
    
    // Tensor pointer = context buffer + object offset
    struct ggml_tensor * result = (struct ggml_tensor *)((char *)ctx->mem_buffer + obj->offs);
    
    // Data pointer = tensor pointer + tensor struct size
    void * tensor_data_ptr = (char *)result + GGML_TENSOR_SIZE;
    
    // Set up tensor data pointers
    tensor_set_data(result, tensor_data_ptr);  // ⚠️ CRITICAL PATH
}
```

### 3. First Tensor Data Setup: `tensor_set_data()`
```c
// ggml/include/ggml.h:661
static inline void tensor_set_data(struct ggml_tensor * tensor, void * data) {
#ifdef GGML_NUMA_MIRROR
    // Check if data is in virtual memory range
    if ((uint64_t)data >= GGML_MMAP_VIRTUAL_MEMORY_BASE_OFFSET && 
        (uint64_t)data < GGML_MMAP_VIRTUAL_MEMORY_BASE_OFFSET + GGML_MMAP_VIRTUAL_MEMORY_NUMA_INCREMENT) {
        should_setup_numa_mirrors = ggml_is_numa() && ggml_numa_node_count() > 1;
    }
    
    // For first tensor: data = normal heap address
    // should_setup_numa_mirrors = false (NUMA disabled)
    tensor->__data[0] = data;      // Normal heap address
    tensor->__data[1] = data;      // Same normal heap address
#endif
}
```

### 4. First Context Cleanup: `ggml_free()`
```c
// ggml/src/ggml.c
void ggml_free(struct ggml_context * ctx) {
    if (ctx->mem_buffer) {
        free(ctx->mem_buffer);  // Free heap memory
        ctx->mem_buffer = NULL;
    }
    free(ctx);
}
```

## 🔍 CRITICAL QUESTION: What Changes Between First and Second Allocation?

### 5. Second Context Creation: `ggml_init()`
```c
// Same code path as #1
ctx->mem_buffer = malloc(params.mem_size);  // Should be normal heap again
```

### 6. Second Tensor Creation: `ggml_new_tensor_impl()`
```c
// Same calculation as #2
void * tensor_data_ptr = (char *)result + GGML_TENSOR_SIZE;
// BUT: result pointer is somehow in virtual memory range!
// tensor_data_ptr = 0x35aa12cab580 (virtual memory range)
```

## 🚨 ROOT CAUSE ANALYSIS

The issue is that `result` pointer itself is in virtual memory:
- **Expected**: `result` = `ctx->mem_buffer + obj->offs` = normal heap address
- **Actual**: `result` = some virtual memory address starting with `0x3...`

This suggests one of three possibilities:

### Hypothesis 1: Context Buffer Allocation Issue
```c
ctx->mem_buffer = malloc(params.mem_size);
```
**Question**: Is `malloc()` returning virtual memory addresses after first context is freed?

### Hypothesis 2: Object Allocation Issue  
```c
struct ggml_object * obj = ggml_new_object(ctx, ...);
struct ggml_tensor * result = (char *)ctx->mem_buffer + obj->offs;
```
**Question**: Is `obj->offs` being calculated incorrectly?

### Hypothesis 3: Memory Pool Corruption
**Question**: Is there some global state affecting memory allocation after first context?

## 🔬 INVESTIGATION TARGETS

1. **Check `ctx->mem_buffer` value in second context**
2. **Check `obj->offs` calculation in `ggml_new_object()`**  
3. **Look for global state that affects malloc() behavior**
4. **Check for memory pool reuse or caching mechanisms**

## 🎯 NEXT STEPS

1. Add debug output to see `ctx->mem_buffer` and `obj->offs` values
2. Trace `ggml_new_object()` implementation
3. Look for any global memory management state
4. Check if there's virtual memory pre-allocation happening
