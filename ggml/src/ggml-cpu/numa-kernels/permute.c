/**
 * @file permute.c
 * @brief NUMA kernel for PERMUTE operation implementation
 * 
 * This file implements a NUMA-aware PERMUTE kernel for tensor axis reordering 
 * operations. PERMUTE is a view operation that reorders tensor dimensions
 * according to a specified axis permutation without performing actual data movement.
 * 
 * @author David Sanftenberg
 * @date 2025
 */

#include "permute.h"
#include "../ggml-numa-shared.h"
#include "numa-kernels.h"
#include "../ggml-cpu-impl.h"

/**
 * @brief NUMA PERMUTE kernel execution function
 * 
 * This function provides a no-operation implementation for PERMUTE kernels.
 * PERMUTE is a view operation that only modifies tensor metadata and requires
 * no actual computation during execution.
 * 
 * @param work_context Pointer to the tensor being processed (cast from ggml_tensor*)
 * @param params Compute parameters containing thread information and work data
 * @return GGML_STATUS_SUCCESS on completion
 */
enum ggml_status ggml_numa_kernel_permute_execute(void * work_context, struct ggml_compute_params * params) {
    // PERMUTE is a metadata-only operation - no computation required
    // The permutation is handled by the ggml tensor system during graph construction
    
    // Basic validation for consistency with other NUMA kernels
    NUMA_ASSERT(work_context != NULL, "Work context cannot be null");
    NUMA_ASSERT(params != NULL, "Compute params cannot be null");
    
    struct ggml_tensor * tensor = (struct ggml_tensor *)work_context;
    NUMA_ASSERT(tensor != NULL, "Tensor cannot be null");
    NUMA_ASSERT(tensor->op == GGML_OP_PERMUTE, "Invalid operation type for PERMUTE kernel");
    
    // Log execution for debugging (minimal overhead)
    NUMA_LOG_TRACE("PERMUTE no-op kernel executing on thread %d/%d", params->ith, params->nth);
    
    // No computation needed - return success immediately
    return GGML_STATUS_SUCCESS;
}

/**
 * @brief Complete no-op implementation for PERMUTE kernel
 * 
 * PERMUTE is a metadata-only operation that should never be executed by the NUMA system.
 * This registration marks it as a no-op kernel, so the coordinator will skip execution.
 */
NUMA_KERNEL_REGISTER_METADATA_NOOP(
    permute,                               // kernel name
    GGML_OP_PERMUTE,                       // operation type  
    "NUMA PERMUTE No-Op Kernel"            // kernel description
)
