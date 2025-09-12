/**
 * @file rope.h
 * @brief NUMA ROPE kernel header with function prototypes and reusable macros
 * @author David Sanftenberg
 */

#pragma once

#include "numa-kernels.h"

// ============================================================================
// NUMA Work Distribution Macros
// ============================================================================

/**
 * Calculate NUMA work distribution for sequence-based operations like ROPE
 * @param total_sequences Total number of sequences (ne2)
 * @param ith Thread index within NUMA node
 * @param nth Total threads per NUMA node  
 * @param i2_start [OUT] Starting sequence index for this thread
 * @param i2_end [OUT] Ending sequence index for this thread
 * @param has_work [OUT] Whether this thread has any work to do
 */
#define NUMA_SEQUENCE_WORK_DISTRIBUTION(total_sequences, ith, nth, i2_start, i2_end, has_work) do { \
    /* Step 1: Divide work by NUMA nodes first */ \
    int numa_start_seq = 0, numa_end_seq = (total_sequences); \
    \
    if (ggml_numa_is_data_parallel_execution) { \
        const int64_t seqs_per_node = (total_sequences) / ggml_numa_total_nodes_for_data_parallel; \
        numa_start_seq = ggml_current_numa_node * seqs_per_node; \
        numa_end_seq = (ggml_current_numa_node == ggml_numa_total_nodes_for_data_parallel - 1) ? \
                       (total_sequences) : numa_start_seq + seqs_per_node; \
    } \
    \
    const int numa_sequences = numa_end_seq - numa_start_seq; \
    \
    /* Step 2: Divide remaining work among threads within this NUMA node */ \
    const int64_t seqs_per_thread = (numa_sequences + (nth) - 1) / (nth);  /* Ceiling division */ \
    const int thread_start_seq = (ith) * seqs_per_thread; \
    const int thread_end_seq = MIN(thread_start_seq + seqs_per_thread, numa_sequences); \
    \
    /* Step 3: Convert to global sequence indices */ \
    (i2_start) = numa_start_seq + thread_start_seq; \
    (i2_end) = numa_start_seq + thread_end_seq; \
    \
    /* Step 4: Check if this thread has work */ \
    (has_work) = (thread_start_seq < numa_sequences && (i2_start) < (i2_end)); \
    \
    /* Debug logging (first thread per NUMA node only) */ \
    if ((ith) == 0) { \
        NUMA_LOG_DEBUG("WORK DISTRIBUTION: total_sequences=%lld, NUMA node %d: numa_start_seq=%d numa_end_seq=%d numa_sequences=%d", \
                       (long long)(total_sequences), ggml_current_numa_node, numa_start_seq, numa_end_seq, numa_sequences); \
        NUMA_LOG_DEBUG("THREAD DISTRIBUTION: thread %d/%d local_range=[%d,%d) global_range=[%d,%d) has_work=%s", \
                       (ith), (nth), thread_start_seq, thread_end_seq, (i2_start), (i2_end), (has_work) ? "YES" : "NO"); \
    } \
} while(0)

// ============================================================================
// Memory Address Calculation Macros
// ============================================================================

/**
 * Calculate tensor element pointer with 4D strides
 * @param base_ptr Base pointer to tensor data
 * @param i3 Batch index
 * @param i2 Sequence index
 * @param i1 Head index
 * @param i0 Element index
 * @param nb3 Batch stride
 * @param nb2 Sequence stride
 * @param nb1 Head stride
 * @param nb0 Element stride
 * @return Pointer to the specific tensor element
 */
#define TENSOR_ELEMENT_PTR(base_ptr, i3, i2, i1, i0, nb3, nb2, nb1, nb0) \
    ((void*)((char*)(base_ptr) + (i3)*(nb3) + (i2)*(nb2) + (i1)*(nb1) + (i0)*(nb0)))

/**
 * Calculate typed tensor element pointer with 4D strides
 */
#define TENSOR_ELEMENT_PTR_TYPED(type, base_ptr, i3, i2, i1, i0, nb3, nb2, nb1, nb0) \
    ((type*)TENSOR_ELEMENT_PTR(base_ptr, i3, i2, i1, i0, nb3, nb2, nb1, nb0))

/**
 * Get source and destination pointers for ROPE computation
 * @param src_ptr [OUT] Source pointer
 * @param dst_ptr [OUT] Destination pointer
 * @param src_base Source tensor base
 * @param dst_base Destination tensor base
 * @param i3 Batch index
 * @param i2 Sequence index
 * @param i1 Head index
 * @param i0 Element index
 */
#define ROPE_GET_SRC_DST_PTRS(src_ptr, dst_ptr, src_base, dst_base, i3, i2, i1, i0, \
                              src_nb3, src_nb2, src_nb1, src_nb0, dst_nb3, dst_nb2, dst_nb1, dst_nb0) do { \
    (src_ptr) = TENSOR_ELEMENT_PTR_TYPED(const float, src_base, i3, i2, i1, i0, src_nb3, src_nb2, src_nb1, src_nb0); \
    (dst_ptr) = TENSOR_ELEMENT_PTR_TYPED(float, dst_base, i3, i2, i1, i0, dst_nb3, dst_nb2, dst_nb1, dst_nb0); \
} while(0)

// ============================================================================
// ROPE Computation Macros
// ============================================================================

/**
 * Standard ROPE rotation computation (adjacent pairs)
 * @param x0 First input value
 * @param x1 Second input value  
 * @param cos_theta Cosine value from cache
 * @param sin_theta Sine value from cache
 * @param result0 [OUT] First output value
 * @param result1 [OUT] Second output value
 */
#define ROPE_STANDARD_ROTATION(x0, x1, cos_theta, sin_theta, result0, result1) do { \
    (result0) = (x0) * (cos_theta) - (x1) * (sin_theta); \
    (result1) = (x0) * (sin_theta) + (x1) * (cos_theta); \
} while(0)

/**
 * NEOX ROPE rotation computation (half-dimension pairs)
 * @param x0 First input value (from src[0])
 * @param x1 Second input value (from src[n_dims/2])
 * @param cos_theta Cosine value from cache
 * @param sin_theta Sine value from cache
 * @param result0 [OUT] First output value (to dst[0])
 * @param result1 [OUT] Second output value (to dst[n_dims/2])
 */
#define ROPE_NEOX_ROTATION(x0, x1, cos_theta, sin_theta, result0, result1) \
    ROPE_STANDARD_ROTATION(x0, x1, cos_theta, sin_theta, result0, result1)

// ============================================================================
// Multi-dimensional Loop Macros
// ============================================================================

/**
 * Standard 4D tensor iteration loop with sequence range filtering
 * @param i3_start Starting batch index
 * @param i3_end Ending batch index
 * @param i2_start Starting sequence index (from work distribution)
 * @param i2_end Ending sequence index (from work distribution)
 * @param i1_start Starting head index
 * @param i1_end Ending head index
 * @param loop_body Code block to execute for each iteration
 */
/**
 * @brief Type-generic ROPE computation macros
 * These macros allow the same computation logic to work with both F32 and F16 types
 * by parameterizing the type conversion functions
 */

/**
 * ROPE dimension iteration (processes pairs)
 * @param n_dims Number of rotated dimensions
 * @param step Step size (usually 2 for pairs)
 * @param loop_body Code block to execute for each dimension pair
 */
#define ROPE_DIMENSION_LOOP(n_dims, step, loop_body) \
    for (int64_t i0 = 0; i0 < (n_dims); i0 += (step)) { \
        loop_body \
    }

// ============================================================================
// Common Data Structures for Refactored Implementation
// ============================================================================

/**
 * @brief Consolidated ROPE parameters structure
 * Eliminates the 60+ lines of duplicated parameter extraction in F32/F16 implementations
 */
typedef struct {
    // Tensor dimensions  
    int64_t ne0, ne1, ne2, ne3;
    int64_t n_dims;
    
    // Source tensor strides
    size_t src_nb0, src_nb1, src_nb2, src_nb3;
    
    // Destination tensor strides
    size_t dst_nb0, dst_nb1, dst_nb2, dst_nb3;
    
    // ROPE configuration
    int mode;
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    float sin_sign;
    float theta_scale;      // Added: calculated as powf(freq_base, -2.0f/n_dims)
    int64_t n_ctx_orig;
    int sections[4];        // Multi-rope sections
    
    // Optional multi-rope configuration
    bool is_mrope;
    bool is_vision;
    
    // Derived values
    bool is_neox;
    bool is_glm;
    
    // Work calculation
    int64_t total_rows;
} numa_rope_params_t;

/**
 * @brief Thread work distribution structure
 * Defines what work range each thread should process
 */
typedef struct {
    int ir0, ir1;        // Row range for this thread [ir0, ir1)
    int i2_start, i2_end; // Sequence range for debugging
    bool has_work;       // Whether thread has any work assigned
    int total_rows;      // Total rows across all threads
} numa_rope_thread_work_t;

// ============================================================================
// Type-Generic Computation Macros
// ============================================================================

/**
 * @brief Standard ROPE rotation computation (adjacent pairs)
 * Supports both F32 and F16 through type conversion macros
 */
#define ROPE_STANDARD_COMPUTE_CORE(TYPE, SRC_PTR, DST_PTR, TO_F32, FROM_F32, \
                                   cos_theta, sin_theta, trace_prefix) do { \
    const float x0 = TO_F32((SRC_PTR)[0]); \
    const float x1 = TO_F32((SRC_PTR)[1]); \
    \
    const float result0 = x0 * (cos_theta) - x1 * (sin_theta); \
    const float result1 = x0 * (sin_theta) + x1 * (cos_theta); \
    \
    (DST_PTR)[0] = FROM_F32(result0); \
    (DST_PTR)[1] = FROM_F32(result1); \
    \
    NUMA_LOG_TRACE("%s STANDARD: x0=%f x1=%f cos=%f sin=%f -> r0=%f r1=%f", \
                   trace_prefix, (double)x0, (double)x1, (double)(cos_theta), \
                   (double)(sin_theta), (double)result0, (double)result1); \
} while(0)

/**
 * @brief NEOX ROPE rotation computation (half-dimension pairs)
 * Supports both F32 and F16 through type conversion macros
 */
#define ROPE_NEOX_COMPUTE_CORE(TYPE, SRC_PTR, DST_PTR, TO_F32, FROM_F32, \
                               n_dims, cos_theta, sin_theta, trace_prefix) do { \
    const float x0 = TO_F32((SRC_PTR)[0]); \
    const float x1 = TO_F32((SRC_PTR)[(n_dims)/2]); \
    \
    const float result0 = x0 * (cos_theta) - x1 * (sin_theta); \
    const float result1 = x0 * (sin_theta) + x1 * (cos_theta); \
    \
    (DST_PTR)[0] = FROM_F32(result0); \
    (DST_PTR)[(n_dims)/2] = FROM_F32(result1); \
    \
    NUMA_LOG_TRACE("%s NEOX: x0=%f x1=%f cos=%f sin=%f -> r0=%f r1=%f", \
                   trace_prefix, (double)x0, (double)x1, (double)(cos_theta), \
                   (double)(sin_theta), (double)result0, (double)result1); \
} while(0)

/**
 * @brief Copy non-rotated elements (beyond n_dims)
 * Type-generic macro for copying unchanged elements
 */
#define ROPE_COPY_NONROTATED(TYPE, SRC_PTR, DST_PTR) do { \
    (DST_PTR)[0] = (SRC_PTR)[0]; \
    (DST_PTR)[1] = (SRC_PTR)[1]; \
} while(0)

// ============================================================================
// Debug and Trace Macros
// ============================================================================

/**
 * Log ROPE computation details with context
 */
#define ROPE_LOG_COMPUTATION(variant, thread_id, numa_node, seq, head, elem, x0, x1, cos_val, sin_val, result0, result1) \
    NUMA_LOG_TRACE("ROPE TRACE: Thread %d (NUMA %d) %s computation: seq=%d head=%d elem=%d inputs(x0=%f,x1=%f) cache(cos=%f,sin=%f) outputs(result0=%f,result1=%f)", \
                   (thread_id), (numa_node), (variant), (int)(seq), (int)(head), (int)(elem), (float)(x0), (float)(x1), (float)(cos_val), (float)(sin_val), (float)(result0), (float)(result1))

/**
 * Log memory writes with addresses
 */
#define ROPE_LOG_WRITES(variant, thread_id, numa_node, result0, dst_addr0, result1, dst_addr1) \
    NUMA_LOG_TRACE("ROPE TRACE: Thread %d (NUMA %d) %s writes: dst[0]=%f written to %p, dst[1]=%f written to %p", \
                   (thread_id), (numa_node), (variant), (float)(result0), (void*)(dst_addr0), (float)(result1), (void*)(dst_addr1))

/**
 * Log processing start for variant
 */
#define ROPE_LOG_VARIANT_START(variant, numa_node, thread_id, i2_start, i2_end) \
    if (i2 == (i2_start)) { \
        NUMA_LOG_DEBUG("%s ROPE: Starting processing on NUMA node %d, thread %d, sequence range [%d,%d)", \
                      (variant), (numa_node), (thread_id), (i2_start), (i2_end)); \
    }

// ============================================================================
// Helper Function Prototypes
// ============================================================================

/**
 * @brief Extract all ROPE parameters from tensor
 * Consolidates parameter extraction logic shared between F32/F16 implementations
 * @param dst Destination tensor containing ROPE operation parameters
 * @return Populated numa_rope_params_t structure
 */
static numa_rope_params_t extract_rope_params(const struct ggml_tensor* dst);

/**
 * @brief Calculate thread work distribution for ROPE operation
 * Determines which rows each thread should process
 * @param params ROPE parameters structure
 * @param ith Thread index within NUMA node
 * @param nth Total threads per NUMA node
 * @return Thread work assignment structure
 */
static numa_rope_thread_work_t calculate_rope_thread_work(
    const numa_rope_params_t* params, int ith, int nth);

/**
 * @brief Setup ROPE cache for specific position
 * Handles cache computation for given sequence position
 * @param cache Pre-allocated cache buffer
 * @param params ROPE parameters
 * @param i1 Head index
 * @param i2 Sequence index  
 * @param i3 Batch index
 * @param pos Position data array
 * @param freq_factors Frequency factors (optional)
 */
static void setup_rope_cache_for_position(
    float* cache, const numa_rope_params_t* params,
    int64_t i1, int64_t i2, int64_t i3,
    const int32_t* pos, const float* freq_factors);

/**
 * Main ROPE kernel execution function (supports F32 and F16)
 */
enum ggml_status ggml_numa_kernel_rope_execute(void * work_context, struct ggml_compute_params * params);

/**
 * ROPE F32 kernel execution function
 */
enum ggml_status ggml_numa_kernel_rope_f32_execute(void * work_context, struct ggml_compute_params * params);

/**
 * ROPE F16 kernel execution function
 */
enum ggml_status ggml_numa_kernel_rope_f16_execute(void * work_context, struct ggml_compute_params * params);

/**
 * Query function for ROPE kernel strategy selection
 */
ggml_numa_execution_strategy_t ggml_numa_kernel_rope_query(const struct ggml_tensor * tensor);

/**
 * Work buffer calculation function for ROPE kernels
 */
size_t ggml_numa_kernel_rope_work_buffer_calc(const struct ggml_tensor * tensor, int total_numa_nodes, int total_threads);

/**
 * Registration function for ROPE kernel
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_rope_register(void);

#pragma once

#include "numa-kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// NUMA ROPE Kernel Registration
// ============================================================================

/**
 * Register ROPE kernel with NUMA strategy array and work functions
 * Returns registration info for the NUMA kernel registry system
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_rope_register(void);

/**
 * Query ROPE kernel for optimal execution strategy
 * Returns strategy recommendation and kernel info for given tensor
 */
ggml_numa_execution_strategy_t ggml_numa_kernel_rope_query(const struct ggml_tensor * tensor);

/**
 * Calculate work buffer size for ROPE operation
 * Returns per-thread work buffer size in bytes
 */
size_t ggml_numa_kernel_rope_work_buffer_calc(const struct ggml_tensor * tensor, int total_numa_nodes, int total_threads);

// ============================================================================
// NUMA ROPE Kernel Work Functions
// ============================================================================

/**
 * NUMA ROPE kernel execution function
 * Supports all ROPE variants with NUMA-aware parallelization
 * 
 * @param work_context   Tensor containing ROPE operation parameters
 * @param params         Compute parameters (threading, NUMA context)
 * @return               GGML_STATUS_SUCCESS on success, error code on failure
 */
enum ggml_status ggml_numa_kernel_rope_execute(void * work_context, struct ggml_compute_params * params);

#ifdef __cplusplus
}
#endif
