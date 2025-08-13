#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-numa-coordinator.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("Testing NUMA coordinator startup...\n");
    
    // Create a simple context and graph
    struct ggml_init_params init_params = {
        .mem_size = 16 * 1024 * 1024,  // 16MB
        .mem_buffer = NULL,
        .no_alloc = false
    };
    
    struct ggml_context * ctx = ggml_init(init_params);
    if (!ctx) {
        printf("Failed to create context\n");
        return 1;
    }
    
    // Create a simple computation graph
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    struct ggml_tensor * result = ggml_mul_mat(ctx, a, b);
    
    // Fill tensors with some data
    for (int i = 0; i < 64 * 64; i++) {
        ((float*)a->data)[i] = 1.0f;
        ((float*)b->data)[i] = 2.0f;
    }
    
    ggml_build_forward_expand(graph, result);
    
    printf("Graph created with %d nodes\n", graph->n_nodes);
    
    // Create coordinator manager
    struct ggml_threadpool_params tpp = GGML_THREADPOOL_PARAMS_INITIALIZER;
    tpp.n_threads = 4;
    
    printf("Creating coordinator manager...\n");
    struct ggml_numa_coordinator_manager * mgr = 
        ggml_numa_coordinator_manager_new_with_params(&tpp, false);
    
    if (!mgr) {
        printf("Failed to create coordinator manager\n");
        ggml_free(ctx);
        return 1;
    }
    
    printf("Coordinator manager created successfully\n");
    
    // Try to compute the graph
    printf("Starting computation...\n");
    int result_code = ggml_numa_coordinator_manager_compute_graph(mgr, graph);
    
    if (result_code != 0) {
        printf("Computation failed with code %d\n", result_code);
        ggml_numa_coordinator_manager_free(mgr);
        ggml_free(ctx);
        return 1;
    }
    
    printf("Waiting for completion...\n");
    int wait_result = ggml_numa_coordinator_manager_wait_for_completion(mgr);
    
    if (wait_result != 0) {
        printf("Wait for completion failed with code %d\n", wait_result);
        ggml_numa_coordinator_manager_free(mgr);
        ggml_free(ctx);
        return 1;
    }
    
    printf("Computation completed successfully!\n");
    
    // Cleanup
    ggml_numa_coordinator_manager_free(mgr);
    ggml_free(ctx);
    
    return 0;
}
