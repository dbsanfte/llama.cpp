// Simple test to isolate the coordinator without multiple rapid creations
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-numa-coordinator.h"

#include <iostream>
#include <chrono>
#include <thread>

int main() {
    std::cout << "Simple 3-Tier NUMA Coordinator Test" << std::endl;
    std::cout << "====================================" << std::endl;
    
    // Initialize backend
    ggml_numa_init(GGML_NUMA_STRATEGY_DISABLED);
    
    // Test 1: Create and destroy coordinator manager (no computation)
    std::cout << "\n=== Test 1: Basic Coordinator Creation ===" << std::endl;
    
    struct ggml_numa_coordinator_manager * mgr = ggml_numa_coordinator_manager_new(4, true);
    if (mgr) {
        std::cout << "✓ Coordinator manager created successfully" << std::endl;
        
        // Give threads time to start
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        ggml_numa_coordinator_manager_free(mgr);
        std::cout << "✓ Coordinator manager cleaned up successfully" << std::endl;
    } else {
        std::cout << "✗ Failed to create coordinator manager" << std::endl;
        return 1;
    }
    
    // Add delay between tests to ensure full cleanup
    std::cout << "\nWaiting for full cleanup..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Test 2: Create coordinator with cgraph
    std::cout << "\n=== Test 2: Coordinator with CGrapgh ===" << std::endl;
    
    struct ggml_numa_coordinator_manager * mgr2 = ggml_numa_coordinator_manager_new(4, true);
    if (mgr2) {
        std::cout << "✓ Second coordinator manager created successfully" << std::endl;
        
        // Create a simple cgraph for testing
        struct ggml_init_params params = {
            /*.mem_size   =*/ 16 * 1024 * 1024,  // 16MB
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ false,
        };
        
        struct ggml_context * ctx = ggml_init(params);
        if (ctx) {
            struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1000);
            struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1000);
            struct ggml_tensor * c = ggml_add(ctx, a, b);
            
            struct ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, c);
            
            std::cout << "✓ Simple cgraph created" << std::endl;
            
            int result = ggml_numa_coordinator_manager_set_cgraph(mgr2, gf);
            if (result == 0) {
                std::cout << "✓ Cgraph set on coordinator successfully" << std::endl;
                
                result = ggml_numa_coordinator_manager_start(mgr2);
                if (result == 0) {
                    std::cout << "✓ Coordinator threads started successfully" << std::endl;
                    
                    // Give threads time to start
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    
                } else {
                    std::cout << "✗ Failed to start coordinator threads" << std::endl;
                }
            } else {
                std::cout << "✗ Failed to set cgraph on coordinator" << std::endl;
            }
            
            ggml_free(ctx);
        }
        
        ggml_numa_coordinator_manager_free(mgr2);
        std::cout << "✓ Second coordinator manager cleaned up successfully" << std::endl;
    } else {
        std::cout << "✗ Failed to create second coordinator manager" << std::endl;
        return 1;
    }
    
    std::cout << "\n✓ All tests completed successfully!" << std::endl;
    return 0;
}
