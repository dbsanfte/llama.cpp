/**
 * @file soft_max.c  
 * @brief NUMA SOFT_MAX (Softmax Activation) Kernel Implementation
 * @author David Sanftenberg
 * 
 * ============================================================================
 * NUMA KERNEL: SOFT_MAX (Softmax Activation)
 * ============================================================================
 * 
 * Operation: y = softmax(x) with ALiBi attention bias support
 * 
 * Pattern: Reduction operation with row-wise processing
 * Features: 
 * - Numerical stability (max subtraction)
 * - ALiBi attention bias support with scale/max_bias
 * - Optional mask tensor handling (F16/F32)
 * - Shared memory optimization for direct result writes
 * - SIMD-optimized vector operations
 * 
 * Implementation: Hybrid approach with composable macros for setup/validation
 * and custom row-wise processing for mathematical correctness
 */

#include "soft_max.h"
#include "numa-kernels.h"
#include "../ggml-numa-shared.h"
#include "../ggml-numa-openmp-coordinator.h"
#include "../vec.h"
#include "../ops.h"
#include "ggml.h"

// Forward declaration of F32 implementation
static enum ggml_status ggml_numa_kernel_soft_max_f32_execute(void * work_context, struct ggml_compute_params * params);

/**
 * @brief Main SOFT_MAX kernel execute function - dispatches to type-specific implementations
 * @param work_context Pointer to tensor being processed  
 * @param params Compute parameters from coordinator
 * @return Status of the computation
 */
enum ggml_status ggml_numa_kernel_soft_max_execute(void * work_context, struct ggml_compute_params * params) {
    struct ggml_tensor * dst = (struct ggml_tensor *)work_context;
    
    // Dispatch to type-specific implementation
    switch (dst->type) {
        case GGML_TYPE_F32:
            return ggml_numa_kernel_soft_max_f32_execute(work_context, params);
        default:
            NUMA_ASSERT(false, "Unsupported tensor type for SOFT_MAX");
            return GGML_STATUS_FAILED;
    }
}

/**
 * @brief SOFT_MAX kernel implementation for F32 tensors using hybrid approach
 * @param work_context Pointer to tensor being processed
 * @param params Compute parameters from coordinator
 * @return Status of the computation
 */
static enum ggml_status ggml_numa_kernel_soft_max_f32_execute(void * work_context, struct ggml_compute_params * params) {
    struct ggml_tensor * dst = (struct ggml_tensor *)work_context;
    
    // Use composable macros for setup and validation (hybrid approach)
    ggml_numa_slice_context_t ctx;
    NUMA_INIT_CONTEXT(ctx, dst, params);
    NUMA_VALIDATE_INPUTS(dst, params);
    
    // Extract tensor parameters for SOFT_MAX operation
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];  // Optional mask tensor
    
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2]; 
    const int64_t ne03 = src0->ne[3];
    
    // Row-wise processing: total rows to process
    const int64_t nr = ne01 * ne02 * ne03;
    
    // Custom row-wise slicing for mathematical correctness (like ROPE)
    const int64_t dr = (nr + ctx.total_threads - 1) / ctx.total_threads;
    const int64_t ir0 = dr * ctx.thread_id;
    const int64_t ir1 = MIN(ir0 + dr, nr);
    
    if (ir0 >= ir1) {
        NUMA_BARRIER_AUTO();  // Still participate in barriers even with no work
        return GGML_STATUS_SUCCESS;
    }
    
    // Extract SOFT_MAX operation parameters
    float scale     = 1.0f;
    float max_bias  = 0.0f; 
    float m0        = powf(2.0f, -(max_bias) / scale);
    float m1        = powf(2.0f, -(max_bias / 2.0f) / scale);
    
    // Get operation parameters if available (ALiBi support)
    const float * op_params = (const float *)dst->op_params;
    if (op_params) {
        scale    = op_params[0];
        max_bias = op_params[1];
        m0       = powf(2.0f, -(max_bias) / scale);
        m1       = powf(2.0f, -(max_bias / 2.0f) / scale);
    }
    
    // Access data using composable macros
    float * dst_data;
    NUMA_GET_TYPED_POINTER(dst_data, dst, float);
    const float * src0_data;
    NUMA_GET_SOURCE_POINTER(src0_data, src0, float);
    
    // Optional mask tensor data
    const void * src1_data = NULL;
    if (src1) {
        src1_data = src1->data;
    }
    
    // Process assigned rows ir0 to ir1
    for (int64_t i = ir0; i < ir1; i++) {
        // Calculate row indices
        const int64_t i03_val = i / (ne02 * ne01);
        const int64_t i02_val = (i - i03_val * ne02 * ne01) / ne01;
        const int64_t i01_val = i - i03_val * ne02 * ne01 - i02_val * ne01;
        
        // Calculate data pointers for this row
        const float * sp = src0_data + i03_val * src0->nb[3]/sizeof(float) + 
                                      i02_val * src0->nb[2]/sizeof(float) +
                                      i01_val * src0->nb[1]/sizeof(float);
        
        float * dp = dst_data + i03_val * dst->nb[3]/sizeof(float) +
                               i02_val * dst->nb[2]/sizeof(float) +
                               i01_val * dst->nb[1]/sizeof(float);
        
        // Handle ALiBi bias and mask if present
        const char * mask_ptr = NULL;
        if (src1) {
            mask_ptr = (const char *)src1_data + i01_val * src1->nb[1] + 
                                                 i02_val * src1->nb[2] + 
                                                 i03_val * src1->nb[3];
        }
        
        // Copy source to destination first
        for (int j = 0; j < ne00; j++) {
            dp[j] = sp[j];
        }
        
        // Apply scale
        if (scale != 1.0f) {
            ggml_vec_scale_f32(ne00, dp, scale);
        }
        
        // Apply ALiBi bias if needed
        if (max_bias != 0.0f) {
            for (int j = 0; j < ne00; j++) {
                if (i01_val % 2 == 0) {
                    dp[j] += powf(m0, j + 1);
                } else {
                    dp[j] += powf(m1, j + 1);
                }
            }
        }
        
        // Apply mask if present
        if (mask_ptr) {
            if (src1->type == GGML_TYPE_F16) {
                const ggml_fp16_t * mask_f16 = (const ggml_fp16_t *)mask_ptr;
                for (int j = 0; j < ne00; j++) {
                    dp[j] += ggml_fp16_to_fp32(mask_f16[j]);
                }
            } else if (src1->type == GGML_TYPE_F32) {
                const float * mask_f32 = (const float *)mask_ptr;
                ggml_vec_add_f32(ne00, dp, dp, mask_f32);
            }
        }
        
        // Compute softmax with numerical stability
        // Step 1: Find maximum for numerical stability
        float max_val = dp[0];
        for (int j = 1; j < ne00; j++) {
            if (dp[j] > max_val) {
                max_val = dp[j];
            }
        }
        
        // Step 2: Subtract max and compute exp
        float exp_sum = 0.0f;
        for (int j = 0; j < ne00; j++) {
            dp[j] = expf(dp[j] - max_val);
            exp_sum += dp[j];
        }
        
        // Step 3: Normalize by sum
        if (exp_sum > 0.0f) {
            ggml_vec_scale_f32(ne00, dp, 1.0f / exp_sum);
        }
    }
    
    return GGML_STATUS_SUCCESS;
}

/**
 * @brief Calculate work buffer size for SOFT_MAX operation
 * @param tensor The destination tensor for SOFT_MAX operation
 * @param total_numa_nodes Total number of NUMA nodes participating
 * @param total_threads Total number of threads participating
 * @return Total work buffer size needed for all threads
 */
size_t ggml_numa_kernel_soft_max_work_buffer_calc(const struct ggml_tensor * tensor, int total_numa_nodes, int total_threads) {
    // SOFT_MAX doesn't need additional work buffers - operations are done in-place
    GGML_UNUSED(tensor);
    GGML_UNUSED(total_numa_nodes);
    GGML_UNUSED(total_threads);
    return 0;
}

/**
 * @brief Query function for SOFT_MAX NUMA kernel strategy selection
 * @param tensor Tensor to be processed
 * @return Recommended execution strategy based on tensor size
 */
ggml_numa_execution_strategy_t ggml_numa_kernel_soft_max_query(const struct ggml_tensor * tensor) {
    // Use strategy selection from cache with unified thresholds
    NUMA_SELECT_STRATEGY_FROM_CACHE(soft_max, tensor);
}

/**
 * @brief Register SOFT_MAX kernel with NUMA system
 * @return Registration information for SOFT_MAX kernel
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_soft_max_register(void) {
    ggml_numa_kernel_registration_info_t info = {0};
    
    info.op_type = GGML_OP_SOFT_MAX;
    info.supported = true;
    info.kernel_name = "NUMA SOFT_MAX Kernel";
    
    // Strategy thresholds optimized for reduction operations
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_SINGLE] = 1024;      // Single thread below 1K elements
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_MULTI] = 65536;      // Multi-thread below 64K elements
    // Above 64K elements: data-parallel strategy
    info.strategy_array.valid = true;
    
    // Function pointers for different strategies
    info.work_funcs.single_single_fn = ggml_numa_kernel_soft_max_execute;
    info.work_funcs.single_multi_fn = ggml_numa_kernel_soft_max_execute;
    info.work_funcs.data_parallel_fn = ggml_numa_kernel_soft_max_execute;
    info.work_funcs.valid = true;
    
    // Query function pointer - enables direct dispatch without switch statements
    info.query_fn = (void*)ggml_numa_kernel_soft_max_query;
    
    // Work buffer calculation function pointer
    info.work_buffer_calc_fn = (void*)ggml_numa_kernel_soft_max_work_buffer_calc;
    
    // SOFT_MAX doesn't need aggregation functions - operations are in-place
    info.agg_funcs.single_single_fn = NULL;
    info.agg_funcs.single_multi_fn = NULL; 
    info.agg_funcs.data_parallel_fn = NULL;
    info.agg_funcs.valid = false;
    
    return info;
}
