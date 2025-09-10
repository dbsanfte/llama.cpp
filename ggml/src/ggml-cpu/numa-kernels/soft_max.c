/**
 * @file soft_max.c
 * @brief NUMA-aware SOFT_MAX kernel implementation
 * @author David Sanftenberg
 * 
 * Implements high-performance NUMA-aware SOFT_MAX operation with ALiBi support.
 * Uses exact reference implementation pattern with work buffers for numerical stability.
 */

#include "soft_max.h"
#include "numa-kernels.h"
#include "ggml-numa-shared.h"
#include "../ggml-vec-numa.h"
#include <math.h>
#include <string.h>

// Constants
#define CACHE_LINE_SIZE_F32 16
#include <assert.h>

/**
 * @brief Execute SOFT_MAX kernel for F32 tensors using exact reference implementation pattern
 * @param work_context Pointer to destination tensor
 * @param params Compute parameters including thread info and work buffer
 * @return GGML_STATUS_SUCCESS on success
 */
enum ggml_status ggml_numa_kernel_soft_max_f32_execute(void * work_context, struct ggml_compute_params * params) {
    struct ggml_tensor * dst = (struct ggml_tensor *)work_context;
    
    // === SOFT_MAX with NUMA ROWWISE_KERNEL_SETUP for proper row distribution ===
    NUMA_ROWWISE_KERNEL_SETUP(ctx, dst, params, dst_data, float);
    
    // Get source tensors
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    
    // Get typed data using reference implementation pattern  
    const float * src_data = (const float *) tensor_data(src0);
    
    // Source 1 data if available (for ALiBi bias tensor)
    void * src1_data = src1 ? tensor_data(src1) : NULL;
    
    // Tensor dimensions  
    const int64_t ne00 = dst->ne[0];  // Elements per row (sequence length)
    
    // Get work buffer for this thread (using ggml thread ID for consistency with coordinator allocation)
    float * wp = (float *) params->wdata + (ne00 + CACHE_LINE_SIZE_F32) * params->ith;
    memset(wp, 0, ne00 * sizeof(float));  // Zero out work buffer to prevent stale data races
    
    // Source tensor nb arrays
    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];
    
    // Destination tensor nb arrays  
    const size_t nb1 = dst->nb[1];
    const size_t nb2 = dst->nb[2];
    const size_t nb3 = dst->nb[3];
    
    // Mask tensor dimensions and strides (if present)
    const int64_t ne12 = src1 ? src1->ne[2] : 1;
    const int64_t ne13 = src1 ? src1->ne[3] : 1;
    const size_t nb11 = src1 ? src1->nb[1] : 1;
    const size_t nb12 = src1 ? src1->nb[2] : 1;
    const size_t nb13 = src1 ? src1->nb[3] : 1;
    
    // Get operation parameters using ggml helper functions for better performance
    const float scale = ggml_get_op_params_f32(dst, 0);
    const float max_bias = ggml_get_op_params_f32(dst, 1);
    
    // DEBUG: Print scale and max_bias values (only from thread 0 to avoid spam)
    if (params->ith == 0) {
        NUMA_LOG_DEBUG("NUMA SOFT_MAX: scale=%.8f, max_bias=%.8f", scale, max_bias);
        NUMA_LOG_DEBUG("NUMA SOFT_MAX: params->ith=%d, params->nth=%d", params->ith, params->nth);
        NUMA_LOG_DEBUG("NUMA SOFT_MAX: ctx.thread_id=%d, ctx.total_threads=%d", ctx.thread_id, ctx.total_threads);
    }
    
    // ALiBi calculations (matching reference implementation)
    const uint32_t n_head = dst->ne[2];
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));
    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    
    // Check if mask uses F16
    const bool use_f16 = (src1 && src1->type == GGML_TYPE_F16);
    
    // SOFT_MAX specific: Process complete logical rows using NUMA row distribution
    // NUMA_ROWWISE_KERNEL_SETUP has already sliced rows properly: ctx.thread_start to ctx.thread_end
    NUMA_4D_ROWWISE_LOOP(dst, ctx, {
        const int64_t i11 = i01;
        const int64_t i12 = i02 % ne12;
        const int64_t i13 = i03 % ne13;

        // ALiBi slope calculation (matching reference exactly)
        const uint32_t h = i02; // head
        const float slope = (max_bias > 0.0f) ? 
            (h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1)) : 1.0f;

        // Source and destination pointers (matching reference)
        const float * sp = (const float *)((char *) src_data + i01*nb01 + i02*nb02 + i03*nb03);
        float * dp = (float *)((char *) dst_data  + i01*nb1  + i02*nb2  + i03*nb3);

        // Mask pointers (matching reference broadcast logic)
        ggml_fp16_t * mp_f16 = src1 ? (ggml_fp16_t *)((char *) src1_data + i11*nb11 + i12*nb12 + i13*nb13) : NULL;
        float       * mp_f32 = src1 ? (float       *)((char *) src1_data + i11*nb11 + i12*nb12 + i13*nb13) : NULL;

        // Step 1: Copy input to work buffer and scale (matching reference exactly)
        ggml_vec_cpy_f32(ne00, wp, sp);
        ggml_vec_scale_f32(ne00, wp, scale);
        
        // Step 2: Apply mask with ALiBi slope (matching reference exactly)
        if (mp_f32) {
            if (use_f16) {
                for (int i = 0; i < ne00; ++i) {
                    wp[i] += slope * GGML_CPU_FP16_TO_FP32(mp_f16[i]);
                }
            } else {
                for (int i = 0; i < ne00; ++i) {
                    wp[i] += slope * mp_f32[i];
                }
            }
        }

        // Step 3: Find max for numerical stability (matching reference)
        float max_val = -INFINITY;
        ggml_vec_max_f32(ne00, &max_val, wp);

        // Step 4: Apply softmax and normalize (matching reference exactly)
        ggml_float sum = ggml_vec_soft_max_f32(ne00, dp, wp, max_val);
        assert(sum > 0.0);

        sum = 1.0/sum;
        ggml_vec_scale_f32(ne00, dp, sum);
    });
    
    // CRITICAL: All threads must reach barrier for proper synchronization
    NUMA_BARRIER_AUTO(ctx);
    
    return GGML_STATUS_SUCCESS;
}

/**
 * @brief Query function for SOFT_MAX strategy selection
 * @param tensor The destination tensor for SOFT_MAX operation
 * @return Selected execution strategy based on tensor size
 */
ggml_numa_execution_strategy_t ggml_numa_kernel_soft_max_query(const struct ggml_tensor * tensor) {
    const size_t total_elements = ggml_nelements(tensor);
    
    // Use same thresholds as registration
    if (total_elements < 1024) {
        return NUMA_STRATEGY_SINGLE_THREAD;
    } else if (total_elements < 65536) {
        return NUMA_STRATEGY_SINGLE_NODE;
    } else {
        return NUMA_STRATEGY_DATA_PARALLEL;
    }
}

/**
 * @brief Calculate work buffer size for SOFT_MAX operation
 * @param tensor The destination tensor for SOFT_MAX operation
 * @param total_numa_nodes Total number of NUMA nodes participating
 * @param total_threads Total number of threads participating
 * @return Total work buffer size needed for all threads
 */
size_t ggml_numa_kernel_soft_max_work_buffer_calc(const struct ggml_tensor * tensor, int total_numa_nodes, int total_threads) {
    // Each thread needs a work buffer for one row (ne00 elements + cache line alignment)
    const size_t ne00 = tensor->ne[0];
    const size_t per_thread_buffer = (ne00 + CACHE_LINE_SIZE_F32);  // In float elements, not bytes
    
    // Suppress unused parameter warnings
    (void)total_numa_nodes;
    
    // Return TOTAL work buffer size for ALL threads (in bytes for coordinator allocation)
    return per_thread_buffer * total_threads * sizeof(float);
}

/**
 * @brief Register NUMA SOFT_MAX kernel with the kernel registry
 * @return Registration information for SOFT_MAX operation
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_soft_max_register(void) {
    ggml_numa_kernel_registration_info_t info = {0};
    
    info.op_type = GGML_OP_SOFT_MAX;
    info.supported = true;
    info.kernel_name = "NUMA SOFT_MAX Kernel with ALiBi support";
    
    // Strategy thresholds for SOFT_MAX operation
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_SINGLE] = 128;      // Single thread below 1K elements
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_MULTI] = 1024;     // Multi-thread below 64K elements
    // Above 64K elements: data-parallel strategy
    info.strategy_array.valid = true;
    
    // Function pointers for different strategies - all use same F32 implementation
    info.work_funcs.single_single_fn = ggml_numa_kernel_soft_max_f32_execute;
    info.work_funcs.single_multi_fn = ggml_numa_kernel_soft_max_f32_execute;
    info.work_funcs.data_parallel_fn = ggml_numa_kernel_soft_max_f32_execute;
    info.work_funcs.valid = true;
    
    // Query function pointer - enables direct dispatch without switch statements
    info.query_fn = (void*)ggml_numa_kernel_soft_max_query;
    
    // Work buffer calculation function pointer 
    info.work_buffer_calc_fn = (void*)ggml_numa_kernel_soft_max_work_buffer_calc;
    
    // SOFT_MAX doesn't need aggregation functions (probability distributions are complete per row)
    info.agg_funcs.single_single_fn = NULL;
    info.agg_funcs.single_multi_fn = NULL; 
    info.agg_funcs.data_parallel_fn = NULL;
    info.agg_funcs.valid = false;
    
    return info;
}
