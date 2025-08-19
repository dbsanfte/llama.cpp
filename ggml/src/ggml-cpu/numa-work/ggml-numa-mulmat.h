#pragma once

#include "../ggml-numa-operation-dispatch.h"
#include "../ggml-numa-work-shared.h"
#include "../ggml-numa-work-shared.h"
#include "../../include/ggml.h"

// Thread data structure for MUL_MAT kernel
typedef struct {
    struct ggml_tensor * dst;           // Destination tensor
    void * work_buffer;                 // Buffer for type conversion (if needed)
    int64_t ir0_start, ir0_end;         // Row range for src0 
    int64_t ir1_start, ir1_end;         // Row range for src1/dst
    int numa_node;                      // NUMA node this thread is running on
    int thread_id;                      // Thread ID within NUMA node
} ggml_numa_mulmat_thread_data_t;

// MUL_MAT operation NUMA work functions  
enum ggml_status ggml_numa_work_function_mul_mat_single(void * work_context, struct ggml_compute_params * params);
enum ggml_status ggml_numa_work_function_mul_mat_chunking(void * work_context, struct ggml_compute_params * params);
enum ggml_status ggml_numa_work_function_mul_mat_chunk(void * work_context, struct ggml_compute_params * params);

// MUL_MAT computation kernel
void* mul_mat_thread_kernel(void* data);

// MUL_MAT-specific strategy analysis and dispatch
size_t ggml_numa_mul_mat_calculate_work_buffer_size(const struct ggml_tensor * operation);
enum ggml_status ggml_numa_mul_mat_analyze_strategy(
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context,
    ggml_numa_execution_strategy_t * strategy,
    ggml_numa_work_function_t * work_function,
    size_t * work_buffer_size
);

// Main MUL_MAT dispatch function - replaces dispatcher MUL_MAT logic
enum ggml_status ggml_numa_mul_mat_dispatch(
    struct ggml_numa_coordinator_manager * manager,
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context
);

// MUL_MAT operation handler (moved from dispatcher)
extern const ggml_numa_operation_handler_t ggml_numa_handler_mul_mat_enhanced;
