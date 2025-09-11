/**
 * @file get_rows.c
 * @brief NUMA-aware GET_ROWS kernel implementation with quantization support
 * @author David Sanftenberg
 * 
 * This kernel handles row extraction operations from a source tensor (src0) 
 * based on indices in an index tensor (src1). Supports all quantization types
 * with optimized NUMA-aware parallelization.
 */

#include "get_rows.h"
#include "numa-kernels.h"
#include "ggml-numa-shared.h"
#include "../ggml-vec-numa.h"
#include "../ggml-impl.h"
#include <stdlib.h>
#include <string.h>

/**
 * @brief NUMA GET_ROWS kernel execution with quantization support
 */
enum ggml_status ggml_numa_kernel_get_rows_execute(void * work_context, struct ggml_compute_params * params) {
    NUMA_ASSERT(work_context != NULL, "Work context cannot be null");
    NUMA_ASSERT(params != NULL, "Compute params cannot be null");
    
    struct ggml_tensor * dst = (struct ggml_tensor *)work_context;
    struct ggml_tensor * src0 = dst->src[0];  // Source data tensor
    struct ggml_tensor * src1 = dst->src[1];  // Index tensor
    
    NUMA_ASSERT(src0 != NULL, "Source tensor cannot be null");
    NUMA_ASSERT(src1 != NULL, "Index tensor cannot be null");
    NUMA_ASSERT(dst->op == GGML_OP_GET_ROWS, "Expected GET_ROWS operation");
    
    // Validate tensor types - dst should be F32, src1 should be I32
    NUMA_ASSERT(dst->type == GGML_TYPE_F32, "Destination must be F32");
    NUMA_ASSERT(src1->type == GGML_TYPE_I32, "Index tensor must be I32");
    
    // GET_ROWS uses row-wise parallelization - distribute rows across threads
    NUMA_ROWWISE_KERNEL_SETUP(ctx, dst, params, dst_data, float);
    
    // Extract tensor dimensions
    const int64_t nc = dst->ne[0];          // Number of columns (row width)
    const int32_t * indices = (const int32_t *)tensor_data(src1);
    
    // Source tensor properties
    const size_t src0_row_size = src0->nb[1];
    const int64_t src0_num_rows = src0->ne[1];
    
    // Process rows in this thread's range using composable macro system
    // The NUMA_ROWWISE_KERNEL_SETUP already calculates thread_start and thread_end correctly
    for (int64_t i = ctx.thread_start; i < ctx.thread_end; i++) {
        const int64_t src_row_idx = indices[i];
        
        // Bounds check on source row index - FAIL on invalid indices
        if (src_row_idx < 0 || src_row_idx >= src0_num_rows) {
            NUMA_LOG_ERROR("GET_ROWS: Index out of bounds: %lld (max: %lld)\n", 
                          (long long)src_row_idx, (long long)src0_num_rows);
            NUMA_ASSERT(false, "GET_ROWS: Index out of bounds");
            return GGML_STATUS_FAILED;
        }
        
        // Calculate pointers
        const char * src_row = (const char *)tensor_data(src0) + src_row_idx * src0_row_size;
        float * dst_row = dst_data + i * nc;
        
        // Handle different source quantization types
        switch (src0->type) {
            case GGML_TYPE_F32: {
                // F32 -> F32: Direct copy with SIMD optimization
                ggml_vec_cpy_f32(nc, dst_row, (const float *)src_row);
                break;
            }
            
            case GGML_TYPE_F16: {
                // F16 -> F32: Use optimized conversion
                const ggml_fp16_t * src_f16 = (const ggml_fp16_t *)src_row;
                for (int64_t j = 0; j < nc; j++) {
                    dst_row[j] = GGML_FP16_TO_FP32(src_f16[j]);
                }
                break;
            }
            
            case GGML_TYPE_BF16: {
                // BF16 -> F32: Use optimized conversion
                const ggml_bf16_t * src_bf16 = (const ggml_bf16_t *)src_row;
                for (int64_t j = 0; j < nc; j++) {
                    dst_row[j] = GGML_BF16_TO_FP32(src_bf16[j]);
                }
                break;
            }
            
            default: {
                // Quantized types: Use type traits for dequantization
                const struct ggml_type_traits * type_traits = ggml_get_type_traits(src0->type);
                if (type_traits && type_traits->to_float) {
                    type_traits->to_float(src_row, dst_row, nc);
                } else {
                    NUMA_LOG_ERROR("GET_ROWS: Unsupported source type: %d\n", src0->type);
                    NUMA_ASSERT(false, "GET_ROWS: Unsupported source type");
                    return GGML_STATUS_FAILED;
                }
                break;
            }
        }
    }
    
    // Explicit barrier required after NUMA_ROWWISE_KERNEL_SETUP
    NUMA_BARRIER_AUTO(ctx);
    return GGML_STATUS_SUCCESS;
}

// Generate all kernel support functions using the modern registration macro
// This replaces ~80 lines of boilerplate code with a single macro call!
NUMA_KERNEL_REGISTER_METADATA(
    get_rows,                               // op_name  
    GGML_OP_GET_ROWS,                      // ggml_op_type
    "NUMA GET_ROWS Kernel",                // kernel_display_name
    1024,                                  // threshold_single_single (Single thread below 1K elements)
    8192,                                // threshold_single_multi (Multi-thread below 256K elements) 
    ggml_numa_kernel_get_rows_execute      // execute_function
)
