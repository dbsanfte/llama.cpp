#include <iostream>
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-numa-coordinator.h"

int main() {
    std::cout << "🔍 Testing basic NUMA coordinator functionality..." << std::endl;
    
    // Create simple context and tensor
    struct ggml_init_params params = {0};
    params.mem_size = 16 * 1024 * 1024;  // 16MB 
    params.mem_buffer = NULL;
    params.no_alloc = false;
    
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::cout << "❌ Failed to create context" << std::endl;
        return 1;
    }
    
    // Create simple ADD operation (not MUL_MAT to avoid the hang)
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
    struct ggml_tensor * c = ggml_add(ctx, a, b);
    
    // Fill tensors with test data
    float * a_data = (float*)ggml_get_data(a);
    float * b_data = (float*)ggml_get_data(b);
    
    for (int i = 0; i < 16; i++) {
        a_data[i] = 1.0f;
        b_data[i] = 2.0f;
    }
    
    // Build graph
    struct ggml_cgraph * cgraph = ggml_new_graph(ctx);
    ggml_build_forward_expand(cgraph, c);
    
    std::cout << "✅ Created graph with ADD operation" << std::endl;
    
    // Try to get coordinator manager
    struct ggml_numa_coordinator_manager * mgr = 
        ggml_numa_coordinator_manager_get_global(4, true);
    
    if (!mgr) {
        std::cout << "❌ Failed to create coordinator manager" << std::endl;
        ggml_free(ctx);
        return 1;
    }
    
    std::cout << "✅ Created coordinator manager" << std::endl;
    
    // Try to compute the graph
    std::cout << "⏳ Computing graph..." << std::endl;
    int result = ggml_numa_coordinator_manager_compute_graph(mgr, cgraph);
    
    if (result == 0) {
        std::cout << "✅ Graph computed successfully!" << std::endl;
        
        // Verify result
        float * c_data = (float*)ggml_get_data(c);
        bool verification_passed = true;
        
        std::cout << "Checking ADD results..." << std::endl;
        for (int i = 0; i < 16; i++) {
            float expected = 3.0f;
            float actual = c_data[i];
            std::cout << "c[" << i << "] = " << actual << " (expected " << expected << ")" << std::endl;
            if (actual != expected) {
                verification_passed = false;
            }
        }
        
        if (verification_passed) {
            std::cout << "✅ Result verification passed!" << std::endl;
        } else {
            std::cout << "❌ Result verification failed!" << std::endl;
            result = 1;
        }
        
    } else {
        std::cout << "❌ Graph computation failed: " << result << std::endl;
    }
    
    ggml_free(ctx);
    return result;
}
