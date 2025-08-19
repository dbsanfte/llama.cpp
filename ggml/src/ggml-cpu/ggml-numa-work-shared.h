#pragma once

#include "ggml-numa-operation-dispatch.h"
#include "../include/ggml.h"

//
// Enhanced Debugging Macros for Work Functions
//

// Enhanced thread-aware NUMA logging macro with operation and function context
// Uses ERROR level to ensure visibility in threaded contexts where DEBUG may be filtered
// Parameters:
//   op_name: String name of the NUMA operation (e.g., "ADD", "RMS_NORM")
//   func_name: Function name context (use __func__ for automatic detection)
//   fmt: Format string for additional logging information
//   ...: Variable arguments for format string
#define NUMA_THREAD_LOG_DEBUG(op_name, func_name, fmt, ...) \
    do { \
        int current_numa = ggml_numa_get_current_node(); \
        int thread_id = 0; /* TODO: Get actual thread ID if available */ \
        GGML_LOG_ERROR("🔧[NUMA%d:T%d][%s:%s] " fmt, current_numa, thread_id, (op_name), (func_name), ##__VA_ARGS__); \
    } while(0)

// Simplified macro for cases where operation name is not available
// Uses only function name context
#define NUMA_THREAD_LOG_DEBUG_FUNC(func_name, fmt, ...) \
    NUMA_THREAD_LOG_DEBUG("UNKNOWN", func_name, fmt, ##__VA_ARGS__)

// Convenience macro that automatically uses __func__ for function name
#define NUMA_THREAD_LOG_DEBUG_AUTO(op_name, fmt, ...) \
    NUMA_THREAD_LOG_DEBUG(op_name, __func__, fmt, ##__VA_ARGS__)

// Utility function to extract operation name from tensor for logging
// Returns a string representation of the tensor's operation type
const char * ggml_numa_get_operation_name(const struct ggml_tensor * tensor);

// Enhanced macro that automatically extracts operation name from tensor
#define NUMA_THREAD_LOG_DEBUG_TENSOR(tensor, fmt, ...) \
    NUMA_THREAD_LOG_DEBUG(ggml_numa_get_operation_name(tensor), __func__, fmt, ##__VA_ARGS__)

// Work context structure for function pointer execution
typedef struct {
    struct ggml_tensor * operation;        // The operation to execute
    struct ggml_cplan * cplan;             // Compute plan for execution
    const char * operation_name;           // For debugging and logging
    void * additional_context;             // Operation-specific additional data
    size_t additional_context_size;        // Size of additional context data
} ggml_numa_dispatcher_work_context_t;

// Work context creation and management functions
ggml_numa_dispatcher_work_context_t * ggml_numa_dispatcher_create_work_context(
    struct ggml_tensor * operation,
    const char * operation_name,
    void * additional_context,
    size_t additional_context_size
);

void ggml_numa_dispatcher_free_work_context(ggml_numa_dispatcher_work_context_t * context);

// Buffer size calculation function for dispatcher work contexts
size_t ggml_numa_dispatcher_calculate_work_buffer_size(const struct ggml_tensor * operation);

// Thread data structures for multi-level parallelism
struct add_thread_data {
    // Source data
    const struct ggml_tensor * src0;
    const struct ggml_tensor * src1;
    const struct ggml_tensor * dst;
    // Tensor dimensions
    int64_t ne0, ne1, ne2, ne3;
    size_t nb0, nb1, nb2, nb3;
    size_t src1_nb0, src1_nb1, src1_nb2, src1_nb3;
    size_t dst_nb0, dst_nb1, dst_nb2, dst_nb3;
    // Thread-specific element range within NUMA node's assigned elements
    int64_t thread_start_elem;
    int64_t thread_end_elem;
    // For debugging
    int thread_id;
    int numa_node;
};

struct rms_norm_thread_data {
    const struct ggml_tensor * src0;
    const struct ggml_tensor * dst;
    int64_t ne00, ne01, ne02, ne03;
    size_t nb01, nb02, nb03;
    size_t nb1, nb2, nb3;  // destination strides
    float eps;
    int thread_start_row;
    int thread_end_row;
    int thread_id;
    int numa_node;
};

// Thread kernels
void * add_thread_kernel(void * arg);
void * rms_norm_thread_kernel(void * arg);

// Forward declarations for compute functions used by work functions
extern void ggml_compute_forward_soft_max(const struct ggml_compute_params * params, struct ggml_tensor * dst);
extern void ggml_compute_forward_add(const struct ggml_compute_params * params, struct ggml_tensor * dst);
extern void ggml_compute_forward_rms_norm(const struct ggml_compute_params * params, struct ggml_tensor * dst);
extern void ggml_compute_forward_rope(const struct ggml_compute_params * params, struct ggml_tensor * dst);
extern void ggml_compute_forward_glu(const struct ggml_compute_params * params, struct ggml_tensor * dst);
extern void ggml_compute_forward_flash_attn_ext(
    const struct ggml_compute_params * params,
    const struct ggml_tensor * q,
    const struct ggml_tensor * k,
    const struct ggml_tensor * v,
    const struct ggml_tensor * mask,
    struct ggml_tensor * dst);
extern void ggml_compute_forward_mul_mat_one_chunk(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst,
    const enum ggml_type type,
    const int64_t num_rows_per_vec_dot,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end);
