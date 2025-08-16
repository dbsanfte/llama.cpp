/**
 * Example of how the dispatcher would use the new function pointer approach
 * This demonstrates the operation-agnostic coordinator architecture.
 */

#include "ggml/src/ggml-cpu/ggml-numa-coordinator.h"
#include "ggml.h"

// Example work context structure that the dispatcher would create
typedef struct {
    struct ggml_tensor * operation;      // The operation to execute
    struct ggml_cplan * cplan;           // Compute plan for execution
    const char * operation_name;         // For debugging
} example_work_context_t;

// Example work function that the dispatcher would provide
// This function has no knowledge of NUMA or threading - it just executes the operation
static enum ggml_status example_execute_operation(void * work_context, struct ggml_compute_params * params) {
    if (!work_context || !params) {
        return GGML_STATUS_FAILED;
    }
    
    example_work_context_t * ctx = (example_work_context_t *)work_context;
    
    // Log the execution (coordinator has no knowledge of what operation this is)
    printf("Executing operation %s with %d threads\n", 
           ctx->operation_name ? ctx->operation_name : "unknown", params->nth);
    
    // Update the compute plan with coordinator-provided parameters
    if (ctx->cplan) {
        ctx->cplan->n_threads = params->nth;
        ctx->cplan->work_size = params->wsize;
        ctx->cplan->work_data = params->wdata;
        ctx->cplan->threadpool = params->threadpool;
    }
    
    // Execute the operation using the fallback system
    // In a real implementation, this would call the appropriate operation function
    extern enum ggml_status ggml_numa_fallback_execute(struct ggml_tensor * tensor, struct ggml_cplan * plan);
    return ggml_numa_fallback_execute(ctx->operation, ctx->cplan);
}

// Example of how the dispatcher would submit work using function pointers
int example_dispatcher_submit_work(struct ggml_numa_coordinator_manager * mgr, 
                                   struct ggml_tensor * operation,
                                   const char * operation_name) {
    if (!mgr || !operation) return -1;
    
    // Create work context (dispatcher's responsibility)
    example_work_context_t * work_ctx = malloc(sizeof(example_work_context_t));
    if (!work_ctx) return -1;
    
    // Set up compute plan
    struct ggml_cplan * cplan = malloc(sizeof(struct ggml_cplan));
    if (!cplan) {
        free(work_ctx);
        return -1;
    }
    
    // Initialize work context
    work_ctx->operation = operation;
    work_ctx->cplan = cplan;
    work_ctx->operation_name = operation_name;
    
    // Determine execution strategy (dispatcher's decision)
    ggml_numa_execution_strategy_t strategy = {
        .node_strategy = NUMA_NODE_STRATEGY_SINGLE_NODE,
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
    };
    
    // Calculate buffer requirements (dispatcher's responsibility)
    size_t buffer_size = ggml_nelements(operation) * sizeof(float); // Example calculation
    
    // Submit work using function pointer approach
    // The coordinator will call example_execute_operation() with the provided context
    int work_id = ggml_numa_coordinator_manager_submit_work_function(
        mgr,
        example_execute_operation,  // Function pointer - coordinator just calls this
        work_ctx,                   // Context data - coordinator passes this to function
        -1,                         // Auto-select NUMA node
        strategy,                   // How to execute
        buffer_size                 // Buffer requirements
    );
    
    if (work_id < 0) {
        free(cplan);
        free(work_ctx);
        return -1;
    }
    
    printf("Submitted %s operation with work ID %d using function pointer approach\n", 
           operation_name, work_id);
    
    return work_id;
}

/*
 * KEY BENEFITS OF THIS APPROACH:
 * 
 * 1. COORDINATOR IS COMPLETELY OPERATION-AGNOSTIC
 *    - No switch statements on operation types
 *    - No operation-specific logic
 *    - Just calls the function pointer provided by dispatcher
 * 
 * 2. DISPATCHER MAKES ALL DECISIONS
 *    - Which function to call
 *    - What context data to provide
 *    - Buffer requirements
 *    - Execution strategy
 * 
 * 3. CLEAN SEPARATION OF CONCERNS
 *    - Coordinator: Generic execution engine
 *    - Dispatcher: Operation analysis and strategy
 *    - Work functions: Actual computation
 * 
 * 4. EASY TO EXTEND
 *    - Add new operations by creating new work functions
 *    - No need to modify coordinator at all
 *    - Dispatcher can provide any function pointer
 */
