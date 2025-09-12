/**
 * @file cont.c
 * @brief NUMA CONT kernel implementation as thin wrapper around CPY
 * 
 * CONT operations in ggml are identical to CPY operations in terms of actual
 * computation - both copy data from source to destination. The difference is
 * only semantic in the graph representation. This implementation leverages
 * the sophisticated CPY kernel as a thin wrapper.
 * 
 * @author David Sanftenberg
 * @date 2024
 */

#include "cont.h"
#include "cpy.h"  // Import CPY kernel functions
#include "numa-kernels.h"
#include "ggml-numa-openmp-coordinator.h"

/**
 * @brief NUMA CONT kernel execution function - thin wrapper around CPY
 * 
 * CONT operations are semantically identical to CPY operations in terms of
 * data movement. This implementation delegates to the sophisticated CPY kernel
 * which handles all quantization types, optimization strategies, and NUMA-aware
 * execution patterns.
 * 
 * @param work_context Tensor to process (cast to ggml_tensor*)
 * @param params Compute parameters with thread info
 * @return GGML_STATUS_SUCCESS on success, error code on failure
 */
enum ggml_status ggml_numa_kernel_cont_execute(void * work_context, struct ggml_compute_params * params) {
    // CONT is identical to CPY in terms of data movement
    // Delegate to the sophisticated CPY kernel implementation
    return ggml_numa_kernel_cpy_execute(work_context, params);
}

// Use the new streamlined registration system - CONT as standard operation
NUMA_KERNEL_REGISTER_METADATA(
    cont,                                 // op_name
    GGML_OP_CONT,                         // ggml_op_type
    "NUMA CONT Kernel (CPY wrapper)",     // kernel_display_name
    256,                                  // threshold_single_single (same as CPY)
    512,                                  // threshold_single_multi (same as CPY)
    ggml_numa_kernel_cont_execute         // execute_function
)
