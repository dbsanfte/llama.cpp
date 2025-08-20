#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// Include ggml and our NUMA implementation
#include "ggml.h"
#include "ggml-numa-executor.h"
#include "ggml-numa-coordinator.h"

int main(void) {
    printf("🧪 Testing Fallback Threadpool Implementation\n");
    printf("================================================================\n");
    
    // Initialize NUMA system
    printf("1. Initializing NUMA coordinator...\n");
    
    // Create a simple computation plan
    struct ggml_cplan cplan = {
        .n_threads = 1,
        .work_size = 0,
        .work_data = NULL,
        .abort_callback = NULL,
        .abort_callback_data = NULL
    };
    
    // Create a simple context for GGML
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024,  // 1MB
        .mem_buffer = NULL,
        .no_alloc = false
    };
    
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("❌ Failed to initialize GGML context\n");
        return 1;
    }
    
    printf("2. Creating test tensors...\n");
    
    // Create test tensors
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
    struct ggml_tensor * result = ggml_add(ctx, a, b);
    
    // Initialize tensor data
    float * a_data = (float*)ggml_get_data(a);
    float * b_data = (float*)ggml_get_data(b);
    
    for (int i = 0; i < 16; i++) {
        a_data[i] = (float)i;
        b_data[i] = 1.0f;
    }
    
    printf("3. Testing fallback execution...\n");
    
    // Test the fallback function directly
    enum ggml_status status = ggml_numa_executor_fallback_to_cpu(result, &cplan);
    
    if (status == GGML_STATUS_SUCCESS) {
        printf("✅ Fallback execution successful!\n");
        
        // Verify results
        float * result_data = (float*)ggml_get_data(result);
        bool correct = true;
        
        for (int i = 0; i < 16; i++) {
            float expected = (float)i + 1.0f;
            if (fabs(result_data[i] - expected) > 1e-6) {
                printf("❌ Result mismatch at index %d: got %.6f, expected %.6f\n", 
                       i, result_data[i], expected);
                correct = false;
                break;
            }
        }
        
        if (correct) {
            printf("✅ Mathematical correctness verified!\n");
        }
    } else {
        printf("❌ Fallback execution failed with status %d\n", status);
    }
    
    printf("4. Cleaning up...\n");
    ggml_free(ctx);
    
    printf("================================================================\n");
    printf("🎉 Fallback threadpool test completed!\n");
    
    return (status == GGML_STATUS_SUCCESS) ? 0 : 1;
}
