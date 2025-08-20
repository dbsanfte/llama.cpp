/*
 * Temporary stubs for old dispatcher functions
 * These allow the coordinator to compile while we transition to the new architecture
 */

#include "ggml-numa-operation-dispatch.h"
#include "ggml-numa-work-shared.h"
#include "ggml-numa-executor.h"
#include "ggml-impl.h"
#include "numa-work/ggml-numa-mulmat.h"

// Stub implementations to keep the coordinator compiling during transition

void ggml_numa_dispatch_init(void) {
    GGML_LOG_DEBUG("NUMA Dispatch: Using stub init (old system disabled)\n");
}

enum ggml_status ggml_numa_dispatch_operation(
    struct ggml_numa_coordinator_manager * manager,
    const struct ggml_tensor * tensor,
    const ggml_numa_work_context_t * context) {
    
    (void)manager;
    (void)tensor;
    (void)context;
    
    GGML_LOG_DEBUG("NUMA Dispatch: Using stub dispatch (old system disabled)\n");
    return GGML_STATUS_FAILED;  // Always fail to force fallback
}

void * ggml_numa_dispatch_get_work_buffer(int numa_node, size_t* buffer_size) {
    (void)numa_node;
    (void)buffer_size;
    GGML_LOG_DEBUG("NUMA Dispatch: Using stub work buffer (old system disabled)\n");
    return NULL;
}

void ggml_numa_dispatcher_free_work_context(ggml_numa_dispatcher_work_context_t * context) {
    (void)context;
    GGML_LOG_DEBUG("NUMA Dispatch: Using stub free context (old system disabled)\n");
}

// Bridge function: old signature -> new executor
// Bridge function - the one llama-context.cpp calls with old signature  
int ggml_numa_dispatch_compute_graph(struct ggml_cgraph * cgraph, int n_threads) {
    struct ggml_cplan cplan = {
        .n_threads = n_threads,
        .abort_callback = NULL,
        .abort_callback_data = NULL
    };
    
    return ggml_numa_executor_compute_graph(cgraph, &cplan);
}

// Legacy bridge function (if anything still calls the old name)
enum ggml_status ggml_numa_graph_compute(struct ggml_cgraph * cgraph, int n_threads) {
    int result = ggml_numa_dispatch_compute_graph(cgraph, n_threads);
    return result == 0 ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
}

// Legacy fallback function - just returns failure code for tests
enum ggml_status ggml_numa_execute_operation_fallback(struct ggml_tensor* tensor, struct ggml_cplan * cplan) {
    // This was called by old dispatcher tests when operations weren't supported
    // In our new system, we return GGML_STATUS_FAILED to indicate "not handled by NUMA, use regular CPU"
    (void)tensor; // suppress unused parameter warning
    (void)cplan; // suppress unused parameter warning
    return GGML_STATUS_FAILED; 
}

// Additional missing stubs
enum ggml_status ggml_numa_mul_mat_dispatch(
    struct ggml_numa_coordinator_manager * manager,
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context) {
    
    (void)manager; (void)operation; (void)context;
    GGML_LOG_DEBUG("NUMA MulMat: Using stub dispatch (old system disabled)\n");
    return GGML_STATUS_FAILED;
}

bool ggml_numa_dispatch_ensure_work_buffer(int numa_node, size_t required_size) {
    (void)numa_node; (void)required_size;
    GGML_LOG_DEBUG("NUMA Dispatch: Using stub work buffer ensure (old system disabled)\n");
    return true;  // Always succeed
}

enum ggml_status ggml_numa_intercept_operation(struct ggml_tensor * tensor, struct ggml_compute_params * params) {
    // Bridge to new executor - convert compute_params to cplan
    struct ggml_cplan cplan = {
        .n_threads = params->nth,
        .abort_callback = NULL,
        .abort_callback_data = NULL
    };
    
    return ggml_numa_executor_execute_tensor(tensor, &cplan);
}
