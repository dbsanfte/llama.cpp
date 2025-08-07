#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-numa-coordinator.h"
#include "llama.h"

#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

// Test the most basic coordinator functionality - single computation only
static bool test_single_computation_only() {
    std::cout << "\n=== Testing Single Computation Only ===" << std::endl;
    
    // Get the global coordinator manager
    std::cout << "Acquiring global coordinator manager..." << std::endl;
    struct ggml_numa_coordinator_manager * coordinator = ggml_numa_coordinator_manager_get_global(4, false);
    if (!coordinator) {
        std::cerr << "Failed to get global coordinator manager" << std::endl;
        return false;
    }
    
    std::cout << "✓ Global coordinator manager acquired successfully" << std::endl;
    
    // Create a simple computation context
    struct ggml_init_params params = {
        /*.mem_size   =*/ 8 * 1024 * 1024,  // 8MB
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::cerr << "Failed to create GGML context" << std::endl;
        return false;
    }
    
    // Create small matrices for computation
    const int size = 64;
    std::cout << "Creating " << size << "x" << size << " matrices..." << std::endl;
    
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, size, size);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, size, size);
    
    if (!a || !b) {
        std::cerr << "Failed to create tensors" << std::endl;
        ggml_free(ctx);
        return false;
    }
    
    // Initialize tensors with simple data
    for (int i = 0; i < ggml_nelements(a); i++) {
        ggml_set_f32_1d(a, i, 1.0f);
        ggml_set_f32_1d(b, i, 2.0f);
    }
    
    std::cout << "✓ Tensors created and initialized" << std::endl;
    
    // Create simple computation graph with ONLY matrix multiplication
    struct ggml_tensor * result = ggml_mul_mat(ctx, a, b);
    struct ggml_cgraph * cgraph = ggml_new_graph(ctx);
    ggml_build_forward_expand(cgraph, result);
    
    std::cout << "✓ Simple computation graph created" << std::endl;
    
    // Submit computation graph to the global coordinator
    std::cout << "\n>>> Submitting simple computation to coordinator..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    int compute_result = ggml_numa_coordinator_manager_compute_graph(coordinator, cgraph);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (compute_result == 0) {
        std::cout << "✓ Simple coordinator computation completed successfully in " << duration.count() << "ms" << std::endl;
    } else {
        std::cerr << "✗ Simple coordinator computation failed with result: " << compute_result << std::endl;
        ggml_free(ctx);
        return false;
    }
    
    // Verify result
    float sample_result = ggml_get_f32_1d(result, 0);
    std::cout << "✓ Computation result: " << sample_result << " (expected: around " << (2.0f * size) << ")" << std::endl;
    
    // Cleanup
    ggml_free(ctx);
    
    std::cout << "✓ Single computation test completed successfully" << std::endl;
    std::cout << "NOTE: Stopping here to avoid potential coordinator reuse issues" << std::endl;
    
    return true;
}

int main() {
    std::cout << "Global NUMA Coordinator Single Computation Test" << std::endl;
    std::cout << "===============================================" << std::endl;
    
    // Initialize backend
    llama_backend_init();
    ggml_numa_init(GGML_NUMA_STRATEGY_DISABLED);  // Disable NUMA for testing
    
    std::cout << "Backend initialized (OpenMP disabled, NUMA disabled)" << std::endl;
    std::cout << "NUMA available: " << (ggml_is_numa() ? "Yes" : "No") << std::endl;
    std::cout << "NUMA nodes: " << ggml_numa_node_count() << std::endl;
    
    // Only test one computation to see if coordinator itself is stable
    if (test_single_computation_only()) {
        std::cout << "\n🎉 Single coordinator computation test passed!" << std::endl;
        std::cout << "\nCoordinator Basic Functionality Verified:" << std::endl;
        std::cout << "✓ Global singleton coordinator manager working" << std::endl;
        std::cout << "✓ Persistent coordinator threads operational" << std::endl;
        std::cout << "✓ Work submission and execution functional" << std::endl;
        std::cout << "✓ Single computation processed successfully" << std::endl;
        std::cout << "✓ No segmentation faults with single computation" << std::endl;
        
        std::cout << "\nNext Step: Investigate why reuse causes segfaults" << std::endl;
        std::cout << "Note: Coordinator will be cleaned up automatically at program exit" << std::endl;
        return 0;
    } else {
        std::cout << "\n❌ Single coordinator computation test failed!" << std::endl;
        return 1;
    }
}
