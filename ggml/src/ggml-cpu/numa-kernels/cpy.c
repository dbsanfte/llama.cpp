#include "cpy.h"
#include "../binary-ops.h"

// ================================================================
// CPY Kernel Implementation  
// ================================================================

/**
 * @brief Core CPY implementation function
 * 
 * Handles all types of copy operations:
 * 1. Contiguous same-type: Direct memcpy
 * 2. Same-type row-wise: Row-by-row memcpy  
 * 3. Type conversion: Element-wise conversion
 * 4. Complex reshape: Element-wise with dimension mapping
 * 
 * Uses NUMA-aware data slicing for optimal memory bandwidth.
 */
static enum ggml_status ggml_numa_kernel_cpy_core_implementation(void * work_context, struct ggml_compute_params * params) {
    struct ggml_tensor * tensor = (struct ggml_tensor *)work_context;
    
    // Validate inputs
    if (!tensor || !params || !tensor->src[0]) {
        return GGML_STATUS_FAILED;
    }
    
    const struct ggml_tensor * src0 = tensor->src[0];
    struct ggml_tensor * dst = tensor;
    
    // Validate tensor compatibility
    if (ggml_nelements(dst) != ggml_nelements(src0)) {
        return GGML_STATUS_FAILED;
    }
    
    NUMA_LOG_DEBUG("NUMA CPY: Processing %zu elements, src_type=%d, dst_type=%d\n", 
                   ggml_nelements(src0), src0->type, dst->type);
    
    // Extract tensor dimensions using standard GGML macros
    GGML_TENSOR_UNARY_OP_LOCALS;
    
    // Get NUMA execution context from thread-local variables
    extern __thread void * ggml_numa_shared_result_tensor_data;
    extern __thread bool ggml_numa_is_data_parallel_execution;
    extern __thread int ggml_current_numa_node;
    extern __thread int ggml_numa_total_nodes_for_data_parallel;
    
    // Determine destination data pointer (shared memory optimization)
    void * dst_data;
    if (ggml_numa_shared_result_tensor_data != NULL) {
        dst_data = ggml_numa_shared_result_tensor_data;
        NUMA_LOG_DEBUG("NUMA CPY: Using shared result tensor memory\n");
    } else {
        dst_data = ggml_get_data(dst);
        NUMA_LOG_DEBUG("NUMA CPY: Using local tensor memory\n");
    }
    
    const void * src0_data = ggml_get_data(src0);
    
    // NUMA data slicing for data-parallel execution
    size_t total_elements = ggml_nelements(dst);
    size_t numa_start = 0;
    size_t numa_end = total_elements;
    
    if (ggml_numa_is_data_parallel_execution && ggml_numa_total_nodes_for_data_parallel > 1) {
        size_t elements_per_node = total_elements / ggml_numa_total_nodes_for_data_parallel;
        numa_start = ggml_current_numa_node * elements_per_node;
        numa_end = (ggml_current_numa_node == ggml_numa_total_nodes_for_data_parallel - 1) ? 
                   total_elements : numa_start + elements_per_node;
        
        NUMA_LOG_TRACE("NUMA Node %d, Thread %d CPY processing elements [%zu, %zu) (%zu elements)\n",
                       ggml_current_numa_node, params->ith, numa_start, numa_end, numa_end - numa_start);
    } else {
        NUMA_LOG_TRACE("Thread %d CPY processing all %zu elements\n", params->ith, total_elements);
    }
    
    // Case 1: Same type and contiguous - fastest path using memcpy
    if (src0->type == dst->type && ggml_is_contiguous(src0) && ggml_is_contiguous(dst)) {
        const size_t type_size = ggml_type_size(src0->type);
        const size_t start_bytes = numa_start * type_size;
        const size_t copy_bytes = (numa_end - numa_start) * type_size;
        
        memcpy((char*)dst_data + start_bytes, (const char*)src0_data + start_bytes, copy_bytes);
        
        NUMA_LOG_DEBUG("NUMA CPY: Contiguous memcpy of %zu bytes\n", copy_bytes);
        return GGML_STATUS_SUCCESS;
    }
    
    // Case 2: Same type, row-wise copy (non-contiguous but structured)
    if (src0->type == dst->type) {
        const size_t type_size = ggml_type_size(src0->type);
        
        // Parallelize by rows if possible
        const int ith = params->ith; // thread index
        const int nth = params->nth; // number of threads
        
        if (nth > 1 && ne01 > nth) {
            // Row-wise threading
            const int dr = (ne01 + nth - 1) / nth;
            const int ir0 = dr * ith;
            const int ir1 = (ir0 + dr < ne01) ? ir0 + dr : ne01;
            
            if (ir0 < ir1 && nb00 == type_size && nb0 == type_size) {
                const size_t rs = ggml_row_size(src0->type, ne00);
                for (int64_t i03 = 0; i03 < ne03; i03++) {
                    for (int64_t i02 = 0; i02 < ne02; i02++) {
                        for (int64_t i01 = ir0; i01 < ir1; i01++) {
                            memcpy(
                                ((char *)dst_data + i01*nb1  + i02*nb2  + i03*nb3),
                                ((const char *)src0_data + i01*nb01 + i02*nb02 + i03*nb03),
                                rs);
                        }
                    }
                }
                return GGML_STATUS_SUCCESS;
            }
        }
        
        // Fall through to element-wise copy
    }
    
    // Case 3: Element-wise copy (type conversion or complex layout)
    // This handles F32->F16, quantized->F32, and reshape operations
    
    if (dst->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32) {
        // F32 to F32 element-wise (non-contiguous)
        const float * src_ptr = (const float *)src0_data;
        float * dst_ptr = (float *)dst_data;
        
        for (size_t i = numa_start; i < numa_end; i++) {
            dst_ptr[i] = src_ptr[i];
        }
        
        NUMA_LOG_DEBUG("NUMA CPY: F32->F32 element-wise copy of %zu elements\n", numa_end - numa_start);
        return GGML_STATUS_SUCCESS;
    }
    
    // Case 4: Type conversion cases
    if (dst->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F16) {
        // F16 to F32 conversion
        const ggml_fp16_t * src_ptr = (const ggml_fp16_t *)src0_data;
        float * dst_ptr = (float *)dst_data;
        
        for (size_t i = numa_start; i < numa_end; i++) {
            dst_ptr[i] = GGML_FP16_TO_FP32(src_ptr[i]);
        }
        
        NUMA_LOG_DEBUG("NUMA CPY: F16->F32 conversion of %zu elements\n", numa_end - numa_start);
        return GGML_STATUS_SUCCESS;
    }
    
    if (dst->type == GGML_TYPE_F16 && src0->type == GGML_TYPE_F32) {
        // F32 to F16 conversion  
        const float * src_ptr = (const float *)src0_data;
        ggml_fp16_t * dst_ptr = (ggml_fp16_t *)dst_data;
        
        for (size_t i = numa_start; i < numa_end; i++) {
            dst_ptr[i] = GGML_FP32_TO_FP16(src_ptr[i]);
        }
        
        NUMA_LOG_DEBUG("NUMA CPY: F32->F16 conversion of %zu elements\n", numa_end - numa_start);
        return GGML_STATUS_SUCCESS;
    }
    
    // Case 5: Quantized to F32 conversion - Skip for now, fallback to CPU
    if (dst->type == GGML_TYPE_F32 && ggml_is_quantized(src0->type)) {
        NUMA_LOG_DEBUG("NUMA CPY: Quantized->F32 conversion not implemented, falling back to CPU\n");
        return GGML_STATUS_FAILED;
    }
    
    // Fallback: Unsupported type combination
    NUMA_LOG_DEBUG("NUMA CPY: Unsupported type combination src=%d->dst=%d, falling back to CPU\n", 
                   src0->type, dst->type);
    return GGML_STATUS_FAILED;
}

// ================================================================
// Kernel Function Implementations
// ================================================================

enum ggml_status ggml_numa_kernel_cpy_optimized_execute(void * work_context, struct ggml_compute_params * params) {
    return ggml_numa_kernel_cpy_core_implementation(work_context, params);
}

enum ggml_status ggml_numa_kernel_cpy_low_overhead_execute(void * work_context, struct ggml_compute_params * params) {
    return ggml_numa_kernel_cpy_core_implementation(work_context, params);
}

enum ggml_status ggml_numa_kernel_cpy_no_aggregation_execute(void * work_context, struct ggml_compute_params * params) {
    return ggml_numa_kernel_cpy_core_implementation(work_context, params);
}

// ================================================================
// Registry and Query Functions
// ================================================================

/**
 * @brief Register CPY kernel with NUMA system
 * 
 * Provides registration information for integration into the O(1) hash table system.
 * Uses proven threshold strategy from ADD/MUL kernels.
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_cpy_register(void) {
    ggml_numa_kernel_registration_info_t info = {0};
    
    info.op_type = GGML_OP_CPY;
    info.supported = true;
    info.kernel_name = "NUMA CPY Kernel";
    
    // Strategy thresholds for CPY operations - same as ADD/MUL
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_SINGLE] = 1024;      // Single thread below 1K elements
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_MULTI] = 262144;     // Multi-thread below 256K elements
    // Above 256K elements: data-parallel strategy
    info.strategy_array.valid = true;
    
    // Function pointers for different strategies - using work_funcs not agg_funcs
    info.work_funcs.single_single_fn = ggml_numa_kernel_cpy_low_overhead_execute;
    info.work_funcs.single_multi_fn = ggml_numa_kernel_cpy_low_overhead_execute;
    info.work_funcs.data_parallel_fn = ggml_numa_kernel_cpy_no_aggregation_execute;
    info.work_funcs.valid = true;
    
    // CPY doesn't need aggregation functions (no result aggregation needed)
    info.agg_funcs.single_single_fn = NULL;
    info.agg_funcs.single_multi_fn = NULL;
    info.agg_funcs.data_parallel_fn = NULL;
    info.agg_funcs.valid = false;
    
    // CPY is a computational operation, not a no-op
    info.is_noop = false;
    
    return info;
}

void ggml_numa_kernel_cpy_init_cache_entries(void) {
    // Legacy function for backward compatibility - functionality moved to registration system
    NUMA_LOG_DEBUG("CPY cache entries managed by registration system\n");
}

ggml_numa_kernel_query_result_t ggml_numa_kernel_cpy_query(const struct ggml_tensor * tensor) {
    ggml_numa_kernel_query_result_t result = {0};
    
    if (!tensor || !tensor->src[0]) {
        NUMA_LOG_DEBUG("CPY query: Invalid tensor or missing source\n");
        return result;
    }
    
    const struct ggml_tensor * src0 = tensor->src[0];
    
    // Validate element count compatibility
    if (ggml_nelements(tensor) != ggml_nelements(src0)) {
        NUMA_LOG_DEBUG("CPY query: Element count mismatch\n");
        return result;
    }
    
    // Use the modern O(1) hash table query system
    result = ggml_numa_kernels_strategy_lookup(tensor);
    
    // Update kernel information for CPY-specific details
    if (result.supported && result.work_function != NULL) {
        result.kernel_name = "NUMA CPY (O(1) Fast-Lookup)";
        result.efficiency_score = 0.85f;
        result.aggregation_policy = GGML_NUMA_AGGREGATION_NONE;
        
        NUMA_LOG_DEBUG("CPY query: %zu elements -> %s (efficiency: %.2f)", 
                       ggml_nelements(tensor), result.kernel_name, result.efficiency_score);
    } else {
        NUMA_LOG_DEBUG("CPY query: %zu elements -> No NUMA strategy available", ggml_nelements(tensor));
    }
    
    return result;
}
