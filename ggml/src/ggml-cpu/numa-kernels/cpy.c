/**
 * @file cpy.c
 * @brief NUMA-aware CPY/DUP kernel implementation with type conversion support
 * @author David Sanftenberg
 * 
 * This kernel handles tensor copying operations with optional type conversion.
 * It supports same-type copies as well as conversions between F32, F16, BF16,
 * and quantised types to F32.
 */

#include "cpy.h"
#include "numa-kernels.h"
#include "../ggml-numa-shared.h"
#include "../ggml-numa-openmp-coordinator.h"
#include "../ggml-cpu-impl.h"
#include "../ggml-impl.h"  // For MIN macro
#include "../ggml-vec-numa.h"
#include "../quants.h"
#include "../vec.h"
#include <stdlib.h>
#include <string.h>

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Check if tensors have the same shape for optimized copy path
 */
static inline bool ggml_numa_tensors_same_shape(const struct ggml_tensor * src, const struct ggml_tensor * dst) {
    return src->ne[0] == dst->ne[0] && src->ne[1] == dst->ne[1] && 
           src->ne[2] == dst->ne[2] && src->ne[3] == dst->ne[3];
}

/**
 * @brief Check if tensors are both contiguous for memcpy optimization
 */
static inline bool ggml_numa_tensors_both_contiguous(const struct ggml_tensor * src, const struct ggml_tensor * dst) {
    return ggml_is_contiguous(src) && ggml_is_contiguous(dst);
}

// ============================================================================
// Quantized CPY Kernel Implementation
// ============================================================================

/**
 * @brief CPY kernel for quantized to F32 dequantization
 */
static enum ggml_status ggml_numa_kernel_cpy_quantized_execute(
    struct ggml_tensor * dst, 
    struct ggml_tensor * src0, 
    struct ggml_compute_params * params) {
    
    // Quantized to F32 conversion - Use BLOCKWISE setup for proper quantized handling
    const struct ggml_type_traits * type_traits = ggml_get_type_traits(src0->type);
    if (!type_traits || !type_traits->to_float) {
        NUMA_LOG_DEBUG("CPY Quantized: No dequantization function for type %s\n", ggml_type_name(src0->type));
        return GGML_STATUS_SUCCESS;
    }
    
    const ggml_to_float_t dequantize_row_q = type_traits->to_float;
    const size_t qk = type_traits->blck_size;
    
    NUMA_LOG_TRACE("CPY Quantized START: Thread %d/%d, src_type=%s, dst_type=%s, qk=%zu", 
                   params->ith, params->nth, ggml_type_name(src0->type), ggml_type_name(dst->type), qk);
    NUMA_LOG_TRACE("CPY Quantized TENSORS: src=%p[%ld,%ld,%ld,%ld] dst=%p[%ld,%ld,%ld,%ld]",
                   tensor_data(src0), src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                   tensor_data(dst), dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    NUMA_LOG_TRACE("CPY Quantized STRIDES: src_nb=[%zu,%zu,%zu,%zu] dst_nb=[%zu,%zu,%zu,%zu]",
                   src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                   dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
    
    // Initialize context using atomic building blocks for flexibility
    NUMA_INIT_CONTEXT(ctx, dst, params);
    NUMA_VALIDATE_INPUTS(dst, params);
    
    // Determine slicing strategy based on tensor shape - same logic as standard kernel
    const bool same_shape = ggml_are_same_shape(src0, dst);
    
    // Use shared macro for consistent thread slicing setup
    NUMA_CPY_THREAD_SLICING_SETUP(ctx, dst, src0, same_shape);
    
    NUMA_EARLY_EXIT_IF_NO_WORK(ctx, GGML_STATUS_SUCCESS);
    
    NUMA_GET_TYPED_POINTER(dst_data, dst, float);
    const char * src_data = (const char *)tensor_data(src0);
    
    NUMA_LOG_TRACE("CPY Quantized SETUP: Thread %d, src_data=%p, dst_data=%p, same_shape=%d", 
                   ctx.thread_id, src_data, dst_data, same_shape);
    
    if (same_shape) {
        // Same-shape operations: use quantized rowwise loop for memory locality
        NUMA_4D_QUANTIZED_ROWWISE_LOOP(ctx, dst, src0, qk, {
            NUMA_LOG_TRACE("CPY Quantized BLOCK: Thread %d block(i03=%ld,i02=%ld,i01=%ld,blk=%ld) src_offset=%zu dst_offset=%zu src_ptr=%p dst_ptr=%p",
                           ctx.thread_id, i03, i02, i01, block_idx, src_block_offset, dst_block_offset,
                           (void*)(src_data + src_block_offset), (void*)(dst_data + dst_block_offset));
            
            // Read a few bytes from source to debug
            const char *src_ptr = src_data + src_block_offset;
            NUMA_LOG_TRACE("CPY Quantized SRC_DATA: Thread %d first 8 bytes: %02x %02x %02x %02x %02x %02x %02x %02x",
                           ctx.thread_id, (unsigned char)src_ptr[0], (unsigned char)src_ptr[1], 
                           (unsigned char)src_ptr[2], (unsigned char)src_ptr[3],
                           (unsigned char)src_ptr[4], (unsigned char)src_ptr[5],
                           (unsigned char)src_ptr[6], (unsigned char)src_ptr[7]);
            
            // Dequantize block: quantized source -> F32 destination  
            dequantize_row_q(
                (const void *)(src_data + src_block_offset),    // Source: quantized block
                dst_data + dst_block_offset,                    // Destination: F32 elements
                qk                                              // Elements to convert
            );
            
            // Log first few F32 values written to destination
            float *dst_ptr = dst_data + dst_block_offset;
            NUMA_LOG_TRACE("CPY Quantized DST_DATA: Thread %d first 4 floats: %.6f %.6f %.6f %.6f",
                           ctx.thread_id, dst_ptr[0], dst_ptr[1], dst_ptr[2], dst_ptr[3]);
        });
    } else {
        // Reshape operations: use element-wise processing with quantized blocks
        // Process elements assigned to this thread using linear indexing
        const int64_t total_elements = ggml_nelements(dst);
        
        // Source tensor dimensions for linear-to-4D conversion
        const int64_t src_ne0 = src0->ne[0];
        const int64_t src_ne1 = src0->ne[1]; 
        const int64_t src_ne2 = src0->ne[2];
        const int64_t src_ne3 = src0->ne[3];
        
        // Destination tensor dimensions for linear-to-4D conversion
        const int64_t dst_ne0 = dst->ne[0];
        const int64_t dst_ne1 = dst->ne[1]; 
        const int64_t dst_ne2 = dst->ne[2];
        const int64_t dst_ne3 = dst->ne[3];
        
        // Process all quantized blocks that contain elements in our thread's range
        for (int64_t idx = ctx.thread_start; idx < ctx.thread_end; idx++) {
            // Calculate source coordinates from linear index using shared macro
            int64_t i03_src, i02_src, i01_src, i00_src;
            NUMA_LINEAR_TO_4D_COORDS(idx, src_ne0, src_ne1, src_ne2, src_ne3, i00_src, i01_src, i02_src, i03_src);
            
            // Calculate destination coordinates from linear index using shared macro
            int64_t i03_dst, i02_dst, i01_dst, i00_dst;
            NUMA_LINEAR_TO_4D_COORDS(idx, dst_ne0, dst_ne1, dst_ne2, dst_ne3, i00_dst, i01_dst, i02_dst, i03_dst);
            
            // Determine which quantized block contains this source element
            const int64_t block_idx = i00_src / qk;
            const int64_t element_in_block = i00_src % qk;
            
            // Calculate source block offset
            const size_t src_block_offset = i03_src * src0->nb[3] + i02_src * src0->nb[2] + i01_src * src0->nb[1] + 
                                            block_idx * ggml_type_size(src0->type);
            
            // Calculate destination element offset
            const size_t dst_element_offset = i03_dst * dst->nb[3] + i02_dst * dst->nb[2] + i01_dst * dst->nb[1] + i00_dst * dst->nb[0];
            
            // Dequantize the entire block to temporary buffer, then extract the element we need
            float temp_block[qk];
            dequantize_row_q((const void *)(src_data + src_block_offset), temp_block, qk);
            
            // Copy the specific element from the dequantized block to the destination
            *((float *)((char *)dst_data + dst_element_offset)) = temp_block[element_in_block];
            
            if (idx < ctx.thread_start + 4) {
                NUMA_LOG_TRACE("CPY Quantized ELEM: Thread %d idx=%ld src_coords=(%ld,%ld,%ld,%ld) dst_coords=(%ld,%ld,%ld,%ld) block_idx=%ld elem_in_block=%ld value=%f",
                               ctx.thread_id, idx, i00_src, i01_src, i02_src, i03_src, i00_dst, i01_dst, i02_dst, i03_dst, 
                               block_idx, element_in_block, temp_block[element_in_block]);
            }
        }
    }
    
    NUMA_BARRIER_AUTO(ctx);
    NUMA_LOG_TRACE("CPY Quantized END: Thread %d completed", ctx.thread_id);
    return GGML_STATUS_SUCCESS;
}

// ============================================================================
// Non-Quantized CPY Kernel Implementation  
// ============================================================================

/**
 * @brief CPY kernel for non-quantized operations (same-type copy and standard conversions)
 */
static enum ggml_status ggml_numa_kernel_cpy_standard_execute(
    struct ggml_tensor * dst, 
    struct ggml_tensor * src0, 
    struct ggml_compute_params * params) {
    
    NUMA_LOG_TRACE("CPY Standard START: Thread %d/%d, src_type=%s, dst_type=%s", 
                   params->ith, params->nth, ggml_type_name(src0->type), ggml_type_name(dst->type));
    NUMA_LOG_TRACE("CPY Standard TENSORS: src=%p[%ld,%ld,%ld,%ld] dst=%p[%ld,%ld,%ld,%ld]",
                   tensor_data(src0), src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                   tensor_data(dst), dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    NUMA_LOG_TRACE("CPY Standard STRIDES: src_nb=[%zu,%zu,%zu,%zu] dst_nb=[%zu,%zu,%zu,%zu]",
                   src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                   dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
    
    // Use atomic building blocks for custom element-wise distribution (CPY needs linear element access for reshapes)
    NUMA_INIT_CONTEXT(ctx, dst, params);
    NUMA_VALIDATE_INPUTS(dst, params);
    
    // For CPY operations, we need to set up thread distribution manually based on the specific operation type
    // This is because CPY can be either row-wise (same shape) or element-wise (reshape)
    const bool same_shape = ggml_are_same_shape(src0, dst);
    
    // Use shared macro for consistent thread slicing setup
    NUMA_CPY_THREAD_SLICING_SETUP(ctx, dst, src0, same_shape);
    
    // Early exit if this thread has no work
    if (!ctx.has_work) {
        NUMA_LOG_TRACE("CPY Standard: Thread %d has no work, exiting early", ctx.thread_id);
        NUMA_BARRIER_AUTO(ctx);
        return GGML_STATUS_SUCCESS;
    }
    
    // Get tensor dimensions
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1]; 
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];
    
    const size_t nb0 = dst->nb[0];
    const size_t nb1 = dst->nb[1];
    const size_t nb2 = dst->nb[2];
    const size_t nb3 = dst->nb[3];
    
    const size_t nb00 = src0->nb[0];
    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];
    
    const size_t type_size = ggml_type_size(src0->type);
    
    NUMA_LOG_TRACE("CPY Standard DIMS: ne00=%ld ne01=%ld ne02=%ld ne03=%ld type_size=%zu",
                   ne00, ne01, ne02, ne03, type_size);
    
    // Handle same-type copy (optimized path)
    if (src0->type == dst->type) {
        NUMA_LOG_TRACE("CPY Standard: Same-type copy %s, ne00=%ld, type_size=%zu", 
                       ggml_type_name(src0->type), ne00, type_size);
        
        // Check if we can use simple contiguous copy
        bool is_contiguous = (ggml_is_contiguous(src0) && ggml_is_contiguous(dst));
        bool same_shape = ggml_are_same_shape(src0, dst);
        
        NUMA_LOG_TRACE("CPY Standard: is_contiguous=%d, same_shape=%d, single_thread=%d", 
                       is_contiguous, same_shape, (params->ith == 0 && params->nth == 1));
        
        if (is_contiguous && ggml_nelements(src0) > 1024 && params->ith == 0 && params->nth == 1) {
            // Use bulk memcpy for large contiguous tensors - ONLY in single-thread mode
            const size_t total_elements = ggml_nelements(src0);
            const size_t copy_size = total_elements * type_size;
            
            NUMA_LOG_TRACE("CPY Standard: Bulk copy %zu elements (total_size=%zu bytes)", 
                           total_elements, copy_size);
            
            const char * src_data = (const char *)tensor_data(src0);
            char * dst_data = (char *)tensor_data(dst);
            
            NUMA_LOG_TRACE("CPY Standard BULK: src_data=%p dst_data=%p copy_size=%zu", 
                           src_data, dst_data, copy_size);
            
            // Copy all data in one go for contiguous tensors (single-thread only)
            memcpy(dst_data, src_data, copy_size);
            
            NUMA_LOG_TRACE("CPY Standard BULK: Copy completed successfully");
            NUMA_BARRIER_AUTO(ctx);
        } else if (same_shape && nb00 == type_size && nb0 == type_size) {
            // Same shape optimization: copy by rows using source tensor dimensions
            NUMA_LOG_TRACE("CPY Standard: Same-shape row copy optimization");
            const size_t row_size = ggml_row_size(src0->type, ne00);
            
            NUMA_4D_ROWWISE_LOOP(src0, ctx, {
                const char * src_ptr = (const char *)tensor_data(src0) + i01*nb01 + i02*nb02 + i03*nb03;
                char * dst_ptr = (char *)tensor_data(dst) + i01*nb1 + i02*nb2 + i03*nb3;
                
                if (i01 < 4) { // Log first few rows
                    NUMA_LOG_TRACE("CPY Standard SAME_SHAPE: Thread %d row (%ld,%ld,%ld) src_ptr=%p dst_ptr=%p row_size=%zu",
                                   ctx.thread_id, i01, i02, i03, src_ptr, dst_ptr, row_size);
                }
                
                memcpy(dst_ptr, src_ptr, row_size);
            });
            NUMA_BARRIER_AUTO(ctx);
        } else {
            // General case: element-by-element copy with potential reshape handling
            NUMA_LOG_TRACE("CPY Standard: Element-wise copy, contiguous=%d, same_shape=%d", is_contiguous, same_shape);
            
            // Use pre-calculated context values (either from row slicing or element slicing)
            const int64_t start_idx = ctx.thread_start;
            const int64_t end_idx = ctx.thread_end;
            
            NUMA_LOG_TRACE("CPY Standard Same-Type: Thread %d processing elements %ld to %ld (elements=%ld)",
                           ctx.thread_id, start_idx, end_idx, ctx.thread_elements);
            
            // Use comprehensive same-type loop macro
            NUMA_CPY_SAME_TYPE_LOOP(ctx, src0, dst, start_idx, end_idx, type_size);
            
            NUMA_BARRIER_AUTO(ctx);
        }
    }
    // Handle type conversion
    else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
        NUMA_LOG_TRACE("CPY Standard: F16 to F32 conversion");
        
        // Use shared macro to set up element range for type conversion
        int64_t start_idx, end_idx;
        NUMA_CPY_ELEMENT_RANGE_SETUP(ctx, same_shape, ne00, start_idx, end_idx);
        
        // Use shared macro to early exit if no work
        NUMA_CPY_EARLY_EXIT_IF_NO_WORK(start_idx, end_idx, ctx, "F16->F32");
        
        NUMA_LOG_TRACE("CPY Standard F16->F32: Thread %d/%d processing elements %ld to %ld (total=%ld)", 
                       ctx.thread_id, ctx.total_threads, start_idx, end_idx, end_idx - start_idx);
        
        // Use comprehensive type conversion loop macro
        NUMA_CPY_TYPE_CONVERSION_LOOP(ctx, src0, dst, start_idx, end_idx, 
                                     ggml_fp16_t, float, GGML_FP16_TO_FP32(*src_ptr));
        
        NUMA_BARRIER_AUTO(ctx);
    } else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        NUMA_LOG_TRACE("CPY Standard: BF16 to F32 conversion");
        
        // Use shared macro to set up element range for type conversion
        int64_t start_idx, end_idx;
        NUMA_CPY_ELEMENT_RANGE_SETUP(ctx, same_shape, ne00, start_idx, end_idx);
        
        // Use shared macro to early exit if no work
        NUMA_CPY_EARLY_EXIT_IF_NO_WORK(start_idx, end_idx, ctx, "BF16->F32");
        
        // Use comprehensive type conversion loop macro
        NUMA_CPY_TYPE_CONVERSION_LOOP(ctx, src0, dst, start_idx, end_idx, 
                                     ggml_bf16_t, float, GGML_BF16_TO_FP32(*src_ptr));
        NUMA_BARRIER_AUTO(ctx);
    } else if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
        NUMA_LOG_TRACE("CPY Standard: F32 to F16 conversion");
        
        // Use shared macro to set up element range for type conversion
        int64_t start_idx, end_idx;
        NUMA_CPY_ELEMENT_RANGE_SETUP(ctx, same_shape, ne00, start_idx, end_idx);
        
        // Use shared macro to early exit if no work
        NUMA_CPY_EARLY_EXIT_IF_NO_WORK(start_idx, end_idx, ctx, "F32->F16");
        
        // Use comprehensive type conversion loop macro
        NUMA_CPY_TYPE_CONVERSION_LOOP(ctx, src0, dst, start_idx, end_idx, 
                                     float, ggml_fp16_t, GGML_FP32_TO_FP16(*src_ptr));
        NUMA_BARRIER_AUTO(ctx);
    } else if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_BF16) {
        NUMA_LOG_TRACE("CPY Standard: F32 to BF16 conversion");
        
        // Use shared macro to set up element range for type conversion
        int64_t start_idx, end_idx;
        NUMA_CPY_ELEMENT_RANGE_SETUP(ctx, same_shape, ne00, start_idx, end_idx);
        
        // Use shared macro to early exit if no work
        NUMA_CPY_EARLY_EXIT_IF_NO_WORK(start_idx, end_idx, ctx, "F32->BF16");
        
        // Use comprehensive type conversion loop macro
        NUMA_CPY_TYPE_CONVERSION_LOOP(ctx, src0, dst, start_idx, end_idx, 
                                     float, ggml_bf16_t, GGML_FP32_TO_BF16(*src_ptr));
        NUMA_BARRIER_AUTO(ctx);
    } else {
        // Unsupported type conversion - log but don't exit early for barrier safety
        NUMA_LOG_TRACE("CPY Standard: Unsupported type conversion from %s to %s", 
                       ggml_type_name(src0->type), ggml_type_name(dst->type));
        NUMA_BARRIER_AUTO(ctx);
    }
    
    NUMA_LOG_TRACE("CPY Standard END: Thread %d completed", ctx.thread_id);
    return GGML_STATUS_SUCCESS;
}

// ============================================================================
// Main CPY Dispatcher
// ============================================================================

/**
 * @brief Main CPY kernel execution function - dispatches to appropriate handler
 */
enum ggml_status ggml_numa_kernel_cpy_execute(void * work_context, struct ggml_compute_params * params) {
    NUMA_ASSERT(work_context != NULL, "Work context cannot be null");
    NUMA_ASSERT(params != NULL, "Compute params cannot be null");
    
    struct ggml_tensor * dst = (struct ggml_tensor *)work_context;
    NUMA_ASSERT(dst != NULL, "Destination tensor cannot be null");
    
    struct ggml_tensor * src0 = dst->src[0];
    NUMA_ASSERT(src0 != NULL, "Source tensor cannot be null");
    NUMA_ASSERT(dst->op == GGML_OP_CPY, "Expected CPY operation");
    NUMA_ASSERT(ggml_nelements(dst) == ggml_nelements(src0), "Element count must match");
    
    // Additional validation
    if (tensor_data(src0) == NULL || tensor_data(dst) == NULL) {
        NUMA_LOG_ERROR("CPY DISPATCH: Thread %d aborting due to null tensor data (src=%p, dst=%p)", 
                       params->ith, tensor_data(src0), tensor_data(dst));
        return GGML_STATUS_FAILED;
    }
    
    // Dispatch to appropriate handler based on source type
    if (ggml_is_quantized(src0->type) && dst->type == GGML_TYPE_F32) {
        NUMA_LOG_TRACE("CPY DISPATCH: Thread %d routing to QUANTIZED handler", params->ith);
        return ggml_numa_kernel_cpy_quantized_execute(dst, src0, params);
    } else {
        NUMA_LOG_TRACE("CPY DISPATCH: Thread %d routing to STANDARD handler", params->ith);
        return ggml_numa_kernel_cpy_standard_execute(dst, src0, params);
    }
}

// ============================================================================
// Kernel Registration - Using New Streamlined System
// ============================================================================

// Standard kernel registration - CPY operations don't need work buffers or aggregation
NUMA_KERNEL_REGISTER_METADATA(
    cpy,                                  // op_name
    GGML_OP_CPY,                          // ggml_op_type
    "NUMA CPY Kernel",                    // kernel_display_name
    256,                                 // threshold_single_single (Single thread below 1K elements)
    512,                               // threshold_single_multi (Multi-thread below 256K elements) 
    ggml_numa_kernel_cpy_execute          // execute_function
)
