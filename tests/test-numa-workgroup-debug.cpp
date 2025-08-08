/**
 * Debug test to investigate workgroup failures
 * Simplified test to isolate the data parallelism issue
 */

#include <chrono>
#include <iostream>

#include "ggml.h"
#include "ggml-cpu.h" 
#include "ggml-numa-coordinator.h"

void debug_data_parallelism() {
    std::cout << "🔍 Debugging Data Parallelism Workgroup Issue" << std::endl;
    std::cout << "===============================================" << std::endl;

    // Create GGML context
    struct ggml_init_params init_params = {
        64 * 1024 * 1024, // 64MB
        NULL,
        false,
    };
    
    struct ggml_context * ctx = ggml_init(init_params);
    if (!ctx) {
        std::cout << "❌ Failed to create context" << std::endl;
        return;
    }

    // Get NUMA coordinator manager
    struct ggml_numa_coordinator_manager * mgr = 
        ggml_numa_coordinator_manager_get_global(16, true);
    if (!mgr) {
        std::cout << "❌ Failed to create coordinator manager" << std::endl;
        ggml_free(ctx);
        return;
    }

    std::cout << "✅ Created coordinator manager" << std::endl;

    // Test 1: Small tensor (should work)
    std::cout << "\n🧪 Test 1: Small tensor (100K elements)" << std::endl;
    struct ggml_tensor * small_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100000);
    struct ggml_tensor * small_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100000);
    struct ggml_tensor * small_result = ggml_add(ctx, small_tensor, small_b);
    
    // Fill with data
    float * data_a = (float*)ggml_get_data(small_tensor);
    float * data_b = (float*)ggml_get_data(small_b);
    for (int i = 0; i < 100000; i++) {
        data_a[i] = 1.0f;
        data_b[i] = 2.0f;
    }
    
    // Create computation graph and test
    struct ggml_cgraph * small_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(small_graph, small_result);
    
    std::cout << "  Submitting small tensor computation..." << std::endl;
    int small_result_code = ggml_numa_coordinator_manager_compute_graph(mgr, small_graph);
    std::cout << "  Result code: " << small_result_code << std::endl;

    // Test 2: Large tensor (triggers data parallelism)
    std::cout << "\n🧪 Test 2: Large tensor (1M elements)" << std::endl;
    struct ggml_tensor * large_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1000000);
    struct ggml_tensor * large_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1000000);
    struct ggml_tensor * large_result = ggml_add(ctx, large_tensor, large_b);
    
    // Fill with data
    float * large_data_a = (float*)ggml_get_data(large_tensor);
    float * large_data_b = (float*)ggml_get_data(large_b);
    for (int i = 0; i < 1000000; i++) {
        large_data_a[i] = 1.0f;
        large_data_b[i] = 2.0f;
    }
    
    // Create computation graph and test
    struct ggml_cgraph * large_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(large_graph, large_result);
    
    std::cout << "  Submitting large tensor computation..." << std::endl;
    int large_result_code = ggml_numa_coordinator_manager_compute_graph(mgr, large_graph);
    std::cout << "  Result code: " << large_result_code << std::endl;

    // Test 3: Direct data parallelism API call
    std::cout << "\n🧪 Test 3: Direct data parallelism API call" << std::endl;
    std::cout << "  Calling ggml_numa_coordinator_manager_submit_data_parallel_work..." << std::endl;
    int work_group_id = ggml_numa_coordinator_manager_submit_data_parallel_work(mgr, large_result);
    std::cout << "  Returned ID: " << work_group_id << " (should be work group ID)" << std::endl;
    
    if (work_group_id >= 0) {
        std::cout << "  Calling ggml_numa_coordinator_manager_wait_for_work_group..." << std::endl;
        int wait_result = ggml_numa_coordinator_manager_wait_for_work_group(mgr, work_group_id);
        std::cout << "  Wait result: " << wait_result << std::endl;
    }

    std::cout << "\n📊 Analysis:" << std::endl;
    std::cout << "  - ggml_numa_coordinator_manager_submit_data_parallel_work returns work ID, not work group ID" << std::endl;
    std::cout << "  - ggml_numa_coordinator_manager_wait_for_work_group expects work group ID" << std::endl;
    std::cout << "  - This mismatch causes 'Work group X not found' errors" << std::endl;

    ggml_free(ctx);
}

int main() {
    debug_data_parallelism();
    return 0;
}
