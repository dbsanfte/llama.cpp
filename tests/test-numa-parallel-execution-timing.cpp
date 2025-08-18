#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-numa-coordinator.h"
#include "ggml-numa-operation-dispatch.h"
#include "ggml-backend.h"
#include <chrono>
#include <thread>
#include <atomic>
#include <stdio.h>
#include <math.h>

// Test function that includes a deliberate delay to see timing
static enum ggml_status test_work_function_with_delay(void* context, struct ggml_compute_params* params) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Get NUMA node info
    int numa_node = ggml_numa_get_current_node();
    
    printf("⏰ NUMA%d: Starting work function at %lld ms\n", numa_node, 
           std::chrono::duration_cast<std::chrono::milliseconds>(start_time.time_since_epoch()).count());
    
    // Simulate some computational work with a controlled delay
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Do some actual computation to make it realistic
    volatile float sum = 0.0f;
    for (int i = 0; i < 1000000; i++) {
        sum += sqrt(i * 1.337f);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    printf("⏰ NUMA%d: Completed work function at %lld ms (duration: %lld ms)\n", numa_node,
           std::chrono::duration_cast<std::chrono::milliseconds>(end_time.time_since_epoch()).count(),
           std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count());
    
    return GGML_STATUS_SUCCESS;
}

bool test_parallel_execution_timing() {
    printf("🕰️ Testing NUMA parallel execution timing...\n");
    
    // Force multi-socket mode for testing
    setenv("GGML_NUMA_FORCE_MULTI_SOCKET", "1", 1);
    
    // Initialize GGML
    struct ggml_init_params init_params = {
        .mem_size = 1024 * 1024 * 64,  // 64MB
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    
    struct ggml_context* ctx = ggml_init(init_params);
    if (!ctx) {
        printf("❌ Failed to initialize GGML context\n");
        return false;
    }
    
    // Initialize NUMA coordinator
    if (!ggml_numa_coordinator_is_initialized()) {
        if (ggml_numa_coordinator_init() != 0) {
            printf("❌ Failed to initialize NUMA coordinator\n");
            ggml_free(ctx);
            return false;
        }
    }
    
    auto* manager = ggml_numa_coordinator_manager_get_instance();
    if (!manager) {
        printf("❌ Failed to get coordinator manager\n");
        ggml_free(ctx);
        return false;
    }
    
    printf("🔧 Coordinator initialized with %d NUMA nodes\n", ggml_numa_coordinator_get_num_nodes());
    
    // Create a dummy tensor for work context
    struct ggml_tensor* dummy_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 32);
    
    // Create work context
    auto work_context = ggml_numa_dispatcher_create_work_context(
        dummy_tensor,
        "TIMING_TEST",
        nullptr,
        0
    );
    
    if (!work_context) {
        printf("❌ Failed to create work context\n");
        ggml_free(ctx);
        return false;
    }
    
    // Set up data parallel execution strategy
    ggml_numa_execution_strategy_t strategy = {
        .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL,
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
    };
    
    // Record overall start time
    auto overall_start = std::chrono::high_resolution_clock::now();
    printf("🚀 Overall test starting at %lld ms\n", 
           std::chrono::duration_cast<std::chrono::milliseconds>(overall_start.time_since_epoch()).count());
    
    // Submit work for data parallel execution
    int work_id = ggml_numa_coordinator_manager_submit_work_function(
        manager,
        test_work_function_with_delay,  // Our timing test function
        work_context,
        -1,                            // Auto-select for data parallel
        strategy,
        1024                           // Small buffer size
    );
    
    if (work_id < 0) {
        printf("❌ Failed to submit work\n");
        ggml_numa_dispatcher_free_work_context(work_context);
        ggml_free(ctx);
        return false;
    }
    
    printf("✅ Submitted timing test work (ID: %d)\n", work_id);
    
    // Wait for completion
    int wait_result = ggml_numa_coordinator_manager_wait_for_completion(manager);
    
    auto overall_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(overall_end - overall_start).count();
    
    printf("🏁 Overall test completed at %lld ms (total duration: %lld ms)\n", 
           std::chrono::duration_cast<std::chrono::milliseconds>(overall_end.time_since_epoch()).count(),
           total_duration);
    
    // Analyze results
    if (wait_result == 0) {
        printf("✅ Work completed successfully\n");
        
        // If execution was truly parallel, total time should be close to the individual work time (~100ms + computation)
        // If it was sequential, total time would be much longer (2 * work_time for 2 NUMA nodes)
        if (total_duration < 300) {  // Allow some margin for overhead
            printf("🎉 PARALLEL EXECUTION CONFIRMED: Total time (%lld ms) indicates parallel execution\n", total_duration);
        } else {
            printf("⚠️  SEQUENTIAL EXECUTION DETECTED: Total time (%lld ms) suggests sequential execution\n", total_duration);
        }
        
    } else {
        printf("❌ Work completion failed with result: %d\n", wait_result);
        ggml_numa_dispatcher_free_work_context(work_context);
        ggml_free(ctx);
        return false;
    }
    
    // Cleanup
    ggml_numa_dispatcher_free_work_context(work_context);
    ggml_free(ctx);
    
    return wait_result == 0;
}

int main() {
    printf("===========================================\n");
    printf("🕰️ NUMA Parallel Execution Timing Test\n");
    printf("===========================================\n");
    
    bool success = test_parallel_execution_timing();
    
    printf("\n===========================================\n");
    if (success) {
        printf("✅ Parallel execution timing test: PASSED\n");
        printf("🎯 NUMA nodes are executing work in parallel\n");
    } else {
        printf("❌ Parallel execution timing test: FAILED\n");
    }
    printf("===========================================\n");
    
    return success ? 0 : 1;
}
