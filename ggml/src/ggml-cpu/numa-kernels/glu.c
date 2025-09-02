#include "glu.h"
#include "numa-kernels.h"
#include "../ggml-numa-shared.h"
#include "../ggml-numa-simple-coordinator.h"
#include "../ggml-cpu-impl.h"
#include "../ggml-impl.h"
#include "../ops.h"
#include "../vec.h"
#include <assert.h>

// GLU Threshold-Based Strategy Selection

/**
 * Strategy threshold structure for GLU operations.
 * GLU operations benefit from data-parallel execution at moderate sizes.
 */
typedef struct {
    size_t element_threshold;
    ggml_numa_execution_strategy_t strategy;
    float efficiency_score;
} ggml_glu_strategy_threshold_t;

/**
 * GLU-specific strategy thresholds
 * Based on element count with moderate computational cost characteristics
 */
static const ggml_glu_strategy_threshold_t GLU_THRESHOLDS[] = {
    {
        .element_threshold = 16384,  // 16K elements - single thread
        .strategy = {
            .node_strategy = NUMA_NODE_STRATEGY_SINGLE,
            .on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD
        },
        .efficiency_score = 0.92f
    },
    {
        .element_threshold = 65536,  // 64K elements - multi-thread single node  
        .strategy = {
            .node_strategy = NUMA_NODE_STRATEGY_SINGLE,
            .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
        },
        .efficiency_score = 0.85f
    },
    {
        .element_threshold = SIZE_MAX,  // Above 64K elements - data-parallel strategy
        .strategy = {
            .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL,
            .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
        },
        .efficiency_score = 0.88f
    }
};

#define GLU_THRESHOLD_COUNT (sizeof(GLU_THRESHOLDS) / sizeof(GLU_THRESHOLDS[0]))

/**
 * @brief Execute GLU operation using NUMA kernel
 * 
 * This kernel implements GLU (Gated Linear Unit) operations including REGLU, SWIGLU, 
 * GEGLU, etc. using NUMA-aware execution for optimal performance on multi-socket systems.
 * 
 * GLU operations are binary element-wise operations that take two tensors (x and g) and
 * apply gated linear unit transformations: dst[i] = glu_variant(x[i], g[i])
 * 
 * The kernel supports data-parallel execution across NUMA nodes by slicing the data
 * and distributing computation to maximize memory locality.
 */
enum ggml_status ggml_numa_kernel_glu_execute(void * work_context, struct ggml_compute_params * params) {
    struct ggml_tensor * tensor = (struct ggml_tensor *)work_context;
    
    // Validate inputs
    NUMA_ASSERT(tensor != NULL, "Tensor cannot be null");
    NUMA_ASSERT(params != NULL, "Compute params cannot be null");
    
    // GLU operations require two source tensors
    NUMA_ASSERT(tensor->src[0] != NULL, "GLU source tensor 0 cannot be null");
    
    // Get GLU operation type to determine which SIMD function to use
    const enum ggml_glu_op glu_op = ggml_get_glu_op(tensor);
    
    // Get NUMA execution context from thread-local variables
    extern __thread int ggml_current_numa_node;
    extern __thread int ggml_numa_total_nodes_for_data_parallel; 
    extern __thread bool ggml_numa_is_data_parallel_execution;
    extern __thread void * ggml_numa_shared_result_tensor_data;
    
    // Extract tensor data using shared memory approach
    const struct ggml_tensor * src0 = tensor->src[0];
    const struct ggml_tensor * src1 = tensor->src[1];
    
    NUMA_ASSERT(src0 != NULL, "GLU src0 tensor cannot be null");
    
    // GLU can have either two separate tensors or one tensor split in half
    const bool has_separate_src1 = (src1 != NULL);
    
    // Use shared result tensor memory for direct writes (eliminates aggregation)
    float * dst_data;
    if (ggml_numa_shared_result_tensor_data != NULL) {
        // Use shared result tensor memory - eliminates aggregation overhead
        dst_data = (float *)ggml_numa_shared_result_tensor_data;
    } else {
        // Fallback to local tensor data for compatibility
        dst_data = (float *)tensor_data(tensor);
    }
    
    // Calculate tensor dimensions and row structure
    const int nc = has_separate_src1 ? src0->ne[0] : src0->ne[0] / 2;  // Columns (elements per row)
    const int nr = ggml_nrows(src0);  // Total number of rows
    
    NUMA_LOG_DEBUG("GLU operation: type=%d, nc=%d, nr=%d, has_separate_src1=%s, numa_node=%d\n", 
                   glu_op, nc, nr, has_separate_src1 ? "true" : "false", ggml_current_numa_node);
    
    // Calculate NUMA data slice for data-parallel execution
    int numa_start_row = 0, numa_end_row = nr;
    
    if (ggml_numa_is_data_parallel_execution) {
        // Distribute rows across NUMA nodes for optimal memory locality
        int rows_per_node = nr / ggml_numa_total_nodes_for_data_parallel;
        numa_start_row = ggml_current_numa_node * rows_per_node;
        numa_end_row = (ggml_current_numa_node == ggml_numa_total_nodes_for_data_parallel - 1) ? 
                       nr : numa_start_row + rows_per_node;
        
        NUMA_LOG_DEBUG("GLU NUMA data slice: rows %d-%d (of %d total) on node %d\n",
                       numa_start_row, numa_end_row, nr, ggml_current_numa_node);
    }
    
    // Get tensor stride information
    const size_t src0_stride = src0->nb[1];
    const size_t src1_stride = has_separate_src1 ? src1->nb[1] : src0->nb[1];
    const size_t dst_stride = tensor->nb[1];
    
    // Get 32-bit swapped parameter for single-tensor GLU operations
    const int32_t swapped = ggml_get_op_params_i32(tensor, 1);
    
    // Process each row in the NUMA slice
    for (int row = numa_start_row; row < numa_end_row; row++) {
        // Calculate row pointers
        const char * src0_row_data = (const char *)tensor_data(src0) + row * src0_stride;
        const char * src1_row_data;
        char * dst_row_data = (char *)dst_data + row * dst_stride;
        
        if (has_separate_src1) {
            // Two separate tensors
            src1_row_data = (const char *)tensor_data(src1) + row * src1_stride;
        } else {
            // Single tensor split in half - handle swapped parameter
            src1_row_data = src0_row_data;
        }
        
        // Cast to appropriate type pointers
        const float * src0_ptr = (const float *)src0_row_data;
        const float * src1_ptr = (const float *)src1_row_data;
        float * dst_ptr = (float *)dst_row_data;
        
        // Handle single tensor with swapped parameter
        if (!has_separate_src1) {
            // Adjust pointers based on swapped parameter
            src0_ptr += swapped ? nc : 0;  // First half or second half
            src1_ptr += swapped ? 0 : nc;  // Second half or first half
        }
        
        // Apply the appropriate GLU SIMD operation based on operation type
        switch (glu_op) {
            case GGML_GLU_OP_REGLU:
                ggml_vec_reglu_f32(nc, dst_ptr, src0_ptr, src1_ptr);
                break;
                
            case GGML_GLU_OP_SWIGLU:
                ggml_vec_swiglu_f32(nc, dst_ptr, src0_ptr, src1_ptr);
                break;
                
            case GGML_GLU_OP_GEGLU:
                ggml_vec_geglu_f32(nc, dst_ptr, src0_ptr, src1_ptr);
                break;
                
            case GGML_GLU_OP_GEGLU_ERF:
                ggml_vec_geglu_erf_f32(nc, dst_ptr, src0_ptr, src1_ptr);
                break;
                
            case GGML_GLU_OP_GEGLU_QUICK:
                ggml_vec_geglu_quick_f32(nc, dst_ptr, src0_ptr, src1_ptr);
                break;
                
            default:
                NUMA_ASSERT(false, "Unsupported GLU operation type");
                return GGML_STATUS_FAILED;
        }
    }
    
    NUMA_LOG_TRACE("GLU processed rows %d-%d on NUMA node %d\n", 
                   numa_start_row, numa_end_row, ggml_current_numa_node);
    
    return GGML_STATUS_SUCCESS;
}

/**
 * @brief Query GLU kernel efficiency and characteristics
 * 
 * Provides performance characteristics for GLU operations to help the NUMA executor
 * select the optimal execution strategy based on tensor size and system topology.
 */
ggml_numa_kernel_query_result_t ggml_numa_kernel_glu_query(const struct ggml_tensor * tensor) {
    ggml_numa_kernel_query_result_t result = { .supported = false };
    
    // Validate this is a GLU operation
    if (!tensor || tensor->op != GGML_OP_GLU) {
        return result;
    }
    
    // GLU operations require at least one source tensor
    if (!tensor->src[0]) {
        NUMA_LOG_DEBUG("GLU query: Missing source tensor");
        return result;
    }
    
    // Calculate total elements and rows for complexity assessment
    const int nc = tensor->src[1] ? tensor->src[0]->ne[0] : tensor->src[0]->ne[0] / 2;
    const int nr = ggml_nrows(tensor->src[0]);
    const size_t total_elements = (size_t)nc * nr;
    
    // Check if this kernel is actually registered and supported
    if (!ggml_numa_is_kernel_supported(GGML_OP_GLU)) {
        NUMA_LOG_DEBUG("GLU kernel not supported - registration disabled");
        result.supported = false;
        return result;
    }
    
    // GLU operations are memory-bound with moderate computational cost
    // Strategy selection based on element count thresholds
    
    result.supported = true;
    result.work_buffer_size_per_thread = 0;  // GLU doesn't require additional buffers
    result.work_function = ggml_numa_kernel_glu_execute;
    result.kernel_name = "NUMA GLU Kernel";
    result.aggregation_policy = GGML_NUMA_AGGREGATION_NONE;  // Element-wise operation, no aggregation needed
    result.aggregation_function = NULL;
    result.aggregation_user_data = NULL;
    
    // Find optimal strategy using threshold search
    const ggml_glu_strategy_threshold_t * selected_strategy;
    NUMA_SELECT_STRATEGY_BY_THRESHOLD(GLU_THRESHOLDS, GLU_THRESHOLD_COUNT, total_elements, selected_strategy);
    
    result.strategy = selected_strategy->strategy;
    result.efficiency_score = selected_strategy->efficiency_score;
    
    // Apply force strategy override if set
    ggml_numa_apply_kernel_force_strategy(&result, "GLU", 
                                          ggml_numa_kernel_glu_execute, 
                                          ggml_numa_kernel_glu_execute,
                                          ggml_numa_kernel_glu_execute);
    
    NUMA_LOG_TRACE("GLU query: elements=%zu, node_strategy=%d, on_node_strategy=%d, efficiency=%.2f\n",
                   total_elements, result.strategy.node_strategy, result.strategy.on_node_strategy, (double)result.efficiency_score);
    
    return result;
}

/**
 * @brief Register GLU NUMA kernel with the kernel registry
 * 
 * Configures GLU kernel registration information including strategy thresholds
 * and function pointers for the NUMA kernel system.
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_glu_register(void) {
    ggml_numa_kernel_registration_info_t info = {0};
    
    info.op_type = GGML_OP_GLU;
    info.supported = true;
    info.kernel_name = "NUMA GLU Kernel";
    
    // Strategy thresholds based on element count
    // GLU operations benefit from data-parallel execution at moderate sizes
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_SINGLE] = 16384;   // 16K elements - single thread
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_MULTI] = 65536;    // 64K elements - multi-thread single node
    // Above 64K elements: data-parallel strategy across NUMA nodes
    info.strategy_array.valid = true;
    
    // Function pointers for different execution strategies
    info.work_funcs.single_single_fn = ggml_numa_kernel_glu_execute;
    info.work_funcs.single_multi_fn = ggml_numa_kernel_glu_execute;
    info.work_funcs.data_parallel_fn = ggml_numa_kernel_glu_execute;
    info.work_funcs.valid = true;
    
    // GLU is an element-wise operation that doesn't require aggregation
    // The shared memory approach allows direct writes to final result
    info.agg_funcs.single_single_fn = NULL;
    info.agg_funcs.single_multi_fn = NULL; 
    info.agg_funcs.data_parallel_fn = NULL;
    info.agg_funcs.valid = false;
    
    // GLU is a computational operation, not a no-op
    info.is_noop = false;
    
    NUMA_LOG_DEBUG("Registered GLU kernel: thresholds=[%zu, %zu], supports_data_parallel=true\n",
                   info.strategy_array.thresholds[0], info.strategy_array.thresholds[1]);
    
    return info;
}
