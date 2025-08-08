#include <iostream>
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-numa-coordinator.h"

int main() {
    std::cout << "🔍 Testing NUMA coordinator work distribution with multiple operations..." << std::endl;
    
    // Create context with more memory for multiple operations
    struct ggml_init_params params = {0};
    params.mem_size = 64 * 1024 * 1024;  // 64MB 
    params.mem_buffer = NULL;
    params.no_alloc = false;
    
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::cout << "❌ Failed to create context" << std::endl;
        return 1;
    }
    
    std::cout << "✅ Created context" << std::endl;
    
    // Create multiple small ADD operations to test work distribution
    const int NUM_OPS = 6;  // More operations than NUMA nodes
    struct ggml_tensor * results[NUM_OPS];
    
    for (int i = 0; i < NUM_OPS; i++) {
        struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
        struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
        results[i] = ggml_add(ctx, a, b);
        
        // Fill tensors with different test data for each operation
        float * a_data = (float*)ggml_get_data(a);
        float * b_data = (float*)ggml_get_data(b);
        
        for (int j = 0; j < 16; j++) {
            a_data[j] = (float)(i + 1);  // 1, 2, 3, 4, 5, 6
            b_data[j] = 10.0f;           // constant
        }
    }
    
    // Build graph with all operations
    struct ggml_cgraph * cgraph = ggml_new_graph(ctx);
    for (int i = 0; i < NUM_OPS; i++) {
        ggml_build_forward_expand(cgraph, results[i]);
    }
    
    std::cout << "✅ Created graph with " << NUM_OPS << " ADD operations" << std::endl;
    
    // Get coordinator manager 
    struct ggml_numa_coordinator_manager * mgr = 
        ggml_numa_coordinator_manager_get_global(4, true);
    
    if (!mgr) {
        std::cout << "❌ Failed to create coordinator manager" << std::endl;
        ggml_free(ctx);
        return 1;
    }
    
    std::cout << "✅ Created coordinator manager" << std::endl;
    
    // Compute the graph - should distribute work across NUMA nodes
    std::cout << "⏳ Computing graph with " << NUM_OPS << " operations..." << std::endl;
    int result = ggml_numa_coordinator_manager_compute_graph(mgr, cgraph);
    
    if (result == 0) {
        std::cout << "✅ Graph computed successfully!" << std::endl;
        
        // Verify results for each operation
        bool all_passed = true;
        for (int i = 0; i < NUM_OPS; i++) {
            float * c_data = (float*)ggml_get_data(results[i]);
            float expected = (float)(i + 1) + 10.0f;  // (i+1) + 10
            
            std::cout << "Operation " << i << " results (expected " << expected << "):" << std::endl;
            bool op_passed = true;
            for (int j = 0; j < 16; j++) {
                if (c_data[j] != expected) {
                    std::cout << "  c[" << j << "] = " << c_data[j] << " (❌)" << std::endl;
                    op_passed = false;
                } else if (j < 4) {  // Only show first few successes
                    std::cout << "  c[" << j << "] = " << c_data[j] << " (✅)" << std::endl;
                }
            }
            
            if (!op_passed) {
                all_passed = false;
                std::cout << "❌ Operation " << i << " verification failed!" << std::endl;
            } else {
                std::cout << "✅ Operation " << i << " verification passed!" << std::endl;
            }
        }
        
        if (all_passed) {
            std::cout << "✅ All operations verified successfully!" << std::endl;
        } else {
            result = 1;
        }
        
    } else {
        std::cout << "❌ Graph computation failed: " << result << std::endl;
    }
    
    ggml_free(ctx);
    return result;
}
