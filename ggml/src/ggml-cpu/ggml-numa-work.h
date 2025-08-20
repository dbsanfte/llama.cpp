/*
 * NUMA Work Functions Interface
 * 
 * This header defines the work function interfaces used by the NUMA CPU backend.
 * These functions contain the actual mathematical kernels and NUMA-aware execution logic.
 */

#pragma once

#include "ggml.h"
#include "ggml-impl.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * NUMA work context structure for individual operations
 * This is passed to work functions to specify NUMA node distribution and threading
 */
typedef struct {
    struct ggml_tensor * tensor;      // The operation tensor to process
    int numa_node;                    // Current NUMA node for this work
    int max_numa_nodes;               // Total NUMA nodes in the system
    int thread_start;                 // Start thread index within this NUMA node
    int thread_end;                   // End thread index within this NUMA node
    
    // Work distribution (calculated by caller)
    size_t work_start;               // Start element/row for this NUMA node
    size_t work_end;                 // End element/row for this NUMA node
} ggml_numa_work_context_t;

// ============================================================================
// NUMA Work Function Prototypes
// ============================================================================

/**
 * NUMA-aware GET_ROWS operation
 * Extracts rows from a tensor based on indices
 */
int ggml_numa_work_function_get_rows(void * context);

/**
 * NUMA-aware ADD operation
 * Element-wise addition of two tensors
 */
int ggml_numa_work_function_add(void * context);

/**
 * NUMA-aware MUL_MAT operation for regular (non-quantized) matrices
 * Matrix multiplication for float tensors
 */
int ggml_numa_work_function_mul_mat(void * context);

/**
 * NUMA-aware MUL_MAT operation for quantized matrices
 * Matrix multiplication for quantized tensors
 */
int ggml_numa_work_function_mul_mat_q(void * context);

/**
 * NUMA-aware RMS normalization
 * Root mean square normalization
 */
int ggml_numa_work_function_rms_norm(void * context);

/**
 * NUMA-aware softmax operation
 * Softmax activation function
 */
int ggml_numa_work_function_soft_max(void * context);

/**
 * NUMA-aware RoPE (Rotary Position Embedding) operation
 * Applies rotary position embeddings
 */
int ggml_numa_work_function_rope(void * context);

/**
 * NUMA-aware copy operation
 * Copies tensor data with NUMA awareness
 */
int ggml_numa_work_function_cpy(void * context);

#ifdef __cplusplus
}
#endif
