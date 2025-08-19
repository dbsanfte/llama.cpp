# Enhanced NUMA_THREAD_LOG_DEBUG Macro Usage Examples

This document demonstrates the enhanced `NUMA_THREAD_LOG_DEBUG` macro with operation and function context.

## New Enhanced Format

The enhanced macro now includes operation name and function name for better debugging:

```c
🔧[NUMA{node}:T{thread}][{operation}:{function}] {your_message}
```

## Usage Patterns

### 1. NUMA_THREAD_LOG_DEBUG_AUTO (Recommended)
Automatically uses `__func__` for function name:

```c
enum ggml_status ggml_numa_work_function_add_chunk(void * work_context, struct ggml_compute_params * params) {
    NUMA_THREAD_LOG_DEBUG_AUTO("ADD", "Entering ADD work function\n");
    
    // Processing...
    NUMA_THREAD_LOG_DEBUG_AUTO("ADD", "Processing %ld elements\n", element_count);
    
    return GGML_STATUS_SUCCESS;
}
```

**Output**:
```
🔧[NUMA0:T0][ADD:ggml_numa_work_function_add_chunk] Entering ADD work function
🔧[NUMA0:T0][ADD:ggml_numa_work_function_add_chunk] Processing 1024 elements
```

### 2. NUMA_THREAD_LOG_DEBUG_TENSOR
Automatically extracts operation name from tensor:

```c
enum ggml_status process_tensor(struct ggml_tensor * tensor) {
    NUMA_THREAD_LOG_DEBUG_TENSOR(tensor, "Processing tensor with %d dimensions\n", tensor->n_dims);
    
    // For a MUL_MAT tensor, this outputs:
    // 🔧[NUMA1:T2][MUL_MAT:process_tensor] Processing tensor with 2 dimensions
    
    return GGML_STATUS_SUCCESS;
}
```

### 3. NUMA_THREAD_LOG_DEBUG_FUNC
For functions without known operation context:

```c
void * add_thread_kernel(void * arg) {
    NUMA_THREAD_LOG_DEBUG_FUNC(__func__, "Thread kernel starting\n");
    
    // Processing...
    NUMA_THREAD_LOG_DEBUG_FUNC(__func__, "Thread processing elements %d to %d\n", start, end);
    
    return NULL;
}
```

**Output**:
```
🔧[NUMA0:T1][UNKNOWN:add_thread_kernel] Thread kernel starting
🔧[NUMA0:T1][UNKNOWN:add_thread_kernel] Thread processing elements 512 to 1023
```

### 4. Manual NUMA_THREAD_LOG_DEBUG
Full control over operation and function names:

```c
void custom_operation_handler() {
    NUMA_THREAD_LOG_DEBUG("CUSTOM_OP", "my_handler", "Starting custom operation\n");
    
    // Processing...
    NUMA_THREAD_LOG_DEBUG("CUSTOM_OP", "helper_function", "Helper processing complete\n");
}
```

## Real-World Example: MUL_MAT Work Function

Here's how the enhanced logging appears in actual NUMA work functions:

```c
enum ggml_status ggml_numa_work_function_mul_mat_chunk(void * work_context, struct ggml_compute_params * params) {
    NUMA_THREAD_LOG_DEBUG_AUTO("MUL_MAT", "🔥 ENTERING MUL_MAT_chunk work function\n");
    
    // Validation
    NUMA_ASSERT(work_context && params);
    NUMA_THREAD_LOG_DEBUG_AUTO("MUL_MAT", "✅ Got valid context and params\n");
    
    // Get tensors
    ggml_numa_dispatcher_work_context_t * ctx = (ggml_numa_dispatcher_work_context_t *)work_context;
    struct ggml_tensor * dst = ctx->operation;
    NUMA_THREAD_LOG_DEBUG_TENSOR(dst, "📊 Processing tensor type: %s\n", ggml_type_name(dst->type));
    
    // Processing...
    NUMA_THREAD_LOG_DEBUG_AUTO("MUL_MAT", "🚀 Matrix multiplication complete\n");
    
    return GGML_STATUS_SUCCESS;
}
```

**Output**:
```
🔧[NUMA0:T0][MUL_MAT:ggml_numa_work_function_mul_mat_chunk] 🔥 ENTERING MUL_MAT_chunk work function
🔧[NUMA0:T0][MUL_MAT:ggml_numa_work_function_mul_mat_chunk] ✅ Got valid context and params  
🔧[NUMA0:T0][MUL_MAT:ggml_numa_work_function_mul_mat_chunk] 📊 Processing tensor type: f32
🔧[NUMA0:T0][MUL_MAT:ggml_numa_work_function_mul_mat_chunk] 🚀 Matrix multiplication complete
```

## Migration from Old Format

**Before** (old format):
```c
NUMA_THREAD_LOG_DEBUG("Processing operation\n");
```

**After** (new format - choose appropriate macro):
```c
NUMA_THREAD_LOG_DEBUG_AUTO("OPERATION_NAME", "Processing operation\n");
// or
NUMA_THREAD_LOG_DEBUG_TENSOR(tensor, "Processing operation\n");
// or  
NUMA_THREAD_LOG_DEBUG_FUNC(__func__, "Processing operation\n");
```

## Benefits

1. **Clear Operation Context**: Immediately know which GGML operation is being processed
2. **Function Tracing**: See exactly which function is executing  
3. **NUMA Awareness**: Track operations across different NUMA nodes
4. **Thread Identification**: Distinguish between different worker threads
5. **Debugging Efficiency**: Faster problem diagnosis in complex NUMA workflows

## Debugging Workflow

With the enhanced logging, debugging NUMA operations becomes much more efficient:

1. **Identify the operation**: `[MUL_MAT:...]` tells you it's a matrix multiplication
2. **Locate the function**: `[:ggml_numa_work_function_mul_mat_chunk]` shows the exact function
3. **Track NUMA placement**: `[NUMA1:...]` shows which NUMA node is processing
4. **Follow execution flow**: Chronological log entries show the execution path

This enhanced context makes it much easier to debug performance issues, correctness problems, and NUMA affinity concerns in the llama.cpp NUMA-aware operations.
