/**
 * @file add.c
 * @brief NUMA-aware ADD operation with F16/BF16 support
 * @author David Sanftenberg
 */

#include "numa-kernels.h"
#include "add.h"
#include "ggml-cpu-impl.h"
#include "ggml-numa-shared.h"
#include "ggml-numa-openmp-coordinator.h"
#include "vec.h"

/**
 * @brief ADD kernel execution function supporting F32, F16, and BF16 types
 */
enum ggml_status ggml_numa_kernel_add_unified_execute(void * work_context, struct ggml_compute_params * params) {
    struct ggml_tensor * tensor = (struct ggml_tensor *)work_context;
    
    // Input validation
    if (!tensor || !tensor->src[0] || !tensor->src[1]) {
        return GGML_STATUS_FAILED;
    }
    
    // Handle different type combinations
    const enum ggml_type src0_type = tensor->src[0]->type;
    const enum ggml_type src1_type = tensor->src[1]->type;
    const enum ggml_type dst_type = tensor->type;
    
    // All F32 case
    if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
        NUMA_KERNEL_SETUP_BINARY_BROADCAST(ctx, tensor, params, float) {
            if (__is_scalar) {
                const float __scalar = __src1_data[0];
                for (size_t __i = ctx.thread_start; __i < ctx.thread_end; __i++) {
                    __dst_data[__i] = __src0_data[__i] + __scalar;
                }
            } else if (__is_same_shape) {
                ggml_vec_add_f32(ctx.thread_elements,
                               __dst_data + ctx.thread_start,
                               __src0_data + ctx.thread_start,
                               __src1_data + ctx.thread_start);
            } else {
                // Complex broadcasting case
                NUMA_COMPLEX_BROADCAST_LOOP(ctx, tensor, float, val0 + val1);
            }
        }
    }
    // All F16 case
    else if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
        // For F16, we need custom processing due to type conversions
        ggml_numa_refined_context_t ctx = {0};
        NUMA_SLICE_WORK_BY_ELEMENT(ctx, tensor, params);
        
        if (ctx.has_work) {
            ggml_fp16_t * dst_data = (ggml_fp16_t *)tensor_data(tensor);
            const ggml_fp16_t * src0_data = (const ggml_fp16_t *)tensor_data(tensor->src[0]);
            const ggml_fp16_t * src1_data = (const ggml_fp16_t *)tensor_data(tensor->src[1]);
            
            if (ggml_is_scalar(tensor->src[1])) {
                const float scalar_f32 = GGML_FP16_TO_FP32(src1_data[0]);
                for (size_t i = ctx.thread_start; i < ctx.thread_end; i++) {
                    float val0_f32 = GGML_FP16_TO_FP32(src0_data[i]);
                    dst_data[i] = GGML_FP32_TO_FP16(val0_f32 + scalar_f32);
                }
            } else {
                for (size_t i = ctx.thread_start; i < ctx.thread_end; i++) {
                    float val0_f32 = GGML_FP16_TO_FP32(src0_data[i]);
                    float val1_f32 = GGML_FP16_TO_FP32(src1_data[i]);
                    dst_data[i] = GGML_FP32_TO_FP16(val0_f32 + val1_f32);
                }
            }
        }
    }
    // All BF16 case
    else if (src0_type == GGML_TYPE_BF16 && src1_type == GGML_TYPE_BF16 && dst_type == GGML_TYPE_BF16) {
        // For BF16, we need custom processing due to type conversions
        ggml_numa_refined_context_t ctx = {0};
        NUMA_SLICE_WORK_BY_ELEMENT(ctx, tensor, params);
        
        if (ctx.has_work) {
            ggml_bf16_t * dst_data = (ggml_bf16_t *)tensor_data(tensor);
            const ggml_bf16_t * src0_data = (const ggml_bf16_t *)tensor_data(tensor->src[0]);
            const ggml_bf16_t * src1_data = (const ggml_bf16_t *)tensor_data(tensor->src[1]);
            
            if (ggml_is_scalar(tensor->src[1])) {
                const float scalar_f32 = GGML_BF16_TO_FP32(src1_data[0]);
                for (size_t i = ctx.thread_start; i < ctx.thread_end; i++) {
                    float val0_f32 = GGML_BF16_TO_FP32(src0_data[i]);
                    dst_data[i] = GGML_FP32_TO_BF16(val0_f32 + scalar_f32);
                }
            } else {
                for (size_t i = ctx.thread_start; i < ctx.thread_end; i++) {
                    float val0_f32 = GGML_BF16_TO_FP32(src0_data[i]);
                    float val1_f32 = GGML_BF16_TO_FP32(src1_data[i]);
                    dst_data[i] = GGML_FP32_TO_BF16(val0_f32 + val1_f32);
                }
            }
        }
    }
    // Mixed F16 + F32 -> F16 case
    else if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
        ggml_numa_refined_context_t ctx = {0};
        NUMA_SLICE_WORK_BY_ELEMENT(ctx, tensor, params);
        
        if (ctx.has_work) {
            ggml_fp16_t * dst_data = (ggml_fp16_t *)tensor_data(tensor);
            const ggml_fp16_t * src0_data = (const ggml_fp16_t *)tensor_data(tensor->src[0]);
            const float * src1_data = (const float *)tensor_data(tensor->src[1]);
            
            for (size_t i = ctx.thread_start; i < ctx.thread_end; i++) {
                float val0_f32 = GGML_FP16_TO_FP32(src0_data[i]);
                dst_data[i] = GGML_FP32_TO_FP16(val0_f32 + src1_data[i]);
            }
        }
    }
    // Mixed F16 + F32 -> F32 case
    else if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
        ggml_numa_refined_context_t ctx = {0};
        NUMA_SLICE_WORK_BY_ELEMENT(ctx, tensor, params);
        
        if (ctx.has_work) {
            float * dst_data = (float *)tensor_data(tensor);
            const ggml_fp16_t * src0_data = (const ggml_fp16_t *)tensor_data(tensor->src[0]);
            const float * src1_data = (const float *)tensor_data(tensor->src[1]);
            
            for (size_t i = ctx.thread_start; i < ctx.thread_end; i++) {
                float val0_f32 = GGML_FP16_TO_FP32(src0_data[i]);
                dst_data[i] = val0_f32 + src1_data[i];
            }
        }
    }
    // Mixed BF16 + F32 -> BF16 case
    else if (src0_type == GGML_TYPE_BF16 && src1_type == GGML_TYPE_F32 && dst_type == GGML_TYPE_BF16) {
        ggml_numa_refined_context_t ctx = {0};
        NUMA_SLICE_WORK_BY_ELEMENT(ctx, tensor, params);
        
        if (ctx.has_work) {
            ggml_bf16_t * dst_data = (ggml_bf16_t *)tensor_data(tensor);
            const ggml_bf16_t * src0_data = (const ggml_bf16_t *)tensor_data(tensor->src[0]);
            const float * src1_data = (const float *)tensor_data(tensor->src[1]);
            
            for (size_t i = ctx.thread_start; i < ctx.thread_end; i++) {
                float val0_f32 = GGML_BF16_TO_FP32(src0_data[i]);
                dst_data[i] = GGML_FP32_TO_BF16(val0_f32 + src1_data[i]);
            }
        }
    }
    // Mixed BF16 + F32 -> F32 case
    else if (src0_type == GGML_TYPE_BF16 && src1_type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
        ggml_numa_refined_context_t ctx = {0};
        NUMA_SLICE_WORK_BY_ELEMENT(ctx, tensor, params);
        
        if (ctx.has_work) {
            float * dst_data = (float *)tensor_data(tensor);
            const ggml_bf16_t * src0_data = (const ggml_bf16_t *)tensor_data(tensor->src[0]);
            const float * src1_data = (const float *)tensor_data(tensor->src[1]);
            
            for (size_t i = ctx.thread_start; i < ctx.thread_end; i++) {
                float val0_f32 = GGML_BF16_TO_FP32(src0_data[i]);
                dst_data[i] = val0_f32 + src1_data[i];
            }
        }
    }
    else {
        GGML_ABORT("Unsupported type combination in ADD kernel: dst=%s, src0=%s, src1=%s",
                   ggml_type_name(dst_type), ggml_type_name(src0_type), ggml_type_name(src1_type));
    }
    
    return GGML_STATUS_SUCCESS;
}

// Zero-boilerplate registration using the new macro system
NUMA_KERNEL_REGISTER_METADATA(
    add,                                   // op_name
    GGML_OP_ADD,                          // ggml_op_type  
    "NUMA ADD Kernel with F16/BF16 Support", // kernel_display_name
    1024,                                 // threshold_single_single 
    262144,                               // threshold_single_multi 
    ggml_numa_kernel_add_unified_execute  // execute_function
)