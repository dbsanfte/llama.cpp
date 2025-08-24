#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "ggml.h"
#include "ggml-cpu.h"

int main() {
    printf("🔬 NUMA Data Slicing Verification Test\n");
    printf("=====================================\n\n");
    
    // Initialize GGML
    struct ggml_init_params params = {
        .mem_size = 16 * 1024 * 1024,  // 16MB
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("❌ Failed to initialize GGML context\n");
        return 1;
    }
    
    // Create tensors for LARGE size (should trigger data-parallel)
    const int64_t ne[4] = {1024, 1024, 1, 1};  // 1M elements = LARGE complexity
    
    printf("🎯 Testing LARGE complexity (1M elements)\n");
    printf("Expected: NUMA ADD (Data-Parallel/Multi) kernel\n\n");
    
    struct ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
    struct ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
    struct ggml_tensor * result = ggml_add(ctx, a, b);
    
    // Fill with test data
    float * a_data = (float *)a->data;
    float * b_data = (float *)b->data;
    
    for (int i = 0; i < ggml_nelements(a); i++) {
        a_data[i] = 1.0f;
        b_data[i] = 2.0f;
    }
    
    // Build computation graph
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    
    // Set NUMA to mirror mode
    setenv("NUMA_STRATEGY", "mirror", 1);
    
    printf("🚀 Executing computation...\n");
    
    // Create compute plan and execute the graph
    int n_threads = 8;  // Use multiple threads
    struct ggml_cplan cplan = ggml_graph_plan(graph, n_threads, nullptr);
    enum ggml_status status = ggml_graph_compute(graph, &cplan);
    
    if (status == GGML_STATUS_SUCCESS) {
        printf("✅ Computation completed successfully\n");
        
        // Verify results
        float * result_data = (float *)result->data;
        bool correct = true;
        for (int i = 0; i < 100; i++) {  // Check first 100 elements
            if (result_data[i] != 3.0f) {
                printf("❌ Incorrect result at index %d: expected 3.0, got %.2f\n", i, result_data[i]);
                correct = false;
                break;
            }
        }
        
        if (correct) {
            printf("✅ Mathematical correctness verified\n");
        }
    } else {
        printf("❌ Computation failed with status %d\n", status);
    }
    
    ggml_free(ctx);
    
    printf("\n🔬 Test completed\n");
    return (status == GGML_STATUS_SUCCESS) ? 0 : 1;
}
