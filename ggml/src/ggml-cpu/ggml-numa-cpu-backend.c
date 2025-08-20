/*
 * NUMA CPU Backend - Simplified Entry Point
 * 
 * This provides internal functions for the NUMA executor system.
 * External entry points are handled by the bridge layer in ggml-numa-dispatch-stubs.c
 */

#include "ggml-numa-cpu-backend.h"
#include "ggml-numa-executor.h"
#include "ggml-numa-coordinator.h"
#include "ggml-impl.h"

// ============================================================================
// Internal Helper Functions
// ============================================================================

// Internal function for executor use
enum ggml_status ggml_numa_backend_compute_graph_internal(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan) {
    if (!cgraph) {
        GGML_LOG_ERROR("NUMA Backend: NULL cgraph provided\n");
        return GGML_STATUS_FAILED;
    }
    
    if (!cplan) {
        GGML_LOG_ERROR("NUMA Backend: NULL cplan provided\n");
        return GGML_STATUS_FAILED;
    }
    
    GGML_LOG_INFO("NUMA Backend: Computing graph with %d nodes using %d threads\n", 
                  cgraph->n_nodes, cplan->n_threads);
    
    // Initialize NUMA coordinator if needed
    struct ggml_numa_coordinator_manager * coordinator = 
        ggml_numa_coordinator_manager_get_global(cplan->n_threads);
    
    if (!coordinator) {
        GGML_LOG_ERROR("NUMA Backend: Failed to initialize NUMA coordinator\n");
        return GGML_STATUS_FAILED;
    }
    
    // Delegate to executor for actual computation
    return ggml_numa_executor_compute_graph(cgraph, cplan);
}
