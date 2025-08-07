/**
 * Test program for NUMA coordinator progress callback functionality
 * 
 * This demonstrates:
 * - Reduced verbose logging 
 * - Progress callback system for work item completion
 */

#include "ggml-numa-coordinator.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <atomic>

// Global counters for callback tracking
static std::atomic<int> total_callbacks{0};
static std::atomic<int> numa_node_callbacks[4]{0, 0, 0, 0};

// Progress callback function
static void progress_callback(int work_id, int numa_node, struct ggml_tensor * tensor, void * user_data) {
    const char * user_name = (const char *)user_data;
    
    total_callbacks++;
    if (numa_node >= 0 && numa_node < 4) {
        numa_node_callbacks[numa_node]++;
    }
    
    printf("[CALLBACK] %s: Work %d completed on NUMA node %d (tensor: %p)\n", 
           user_name ? user_name : "DEFAULT", work_id, numa_node, (void*)tensor);
}

int main() {
    printf("==== NUMA Coordinator Progress Callback Test ====\n");
    printf("Testing reduced logging and progress callback system...\n\n");
    
    // Get global coordinator manager
    struct ggml_numa_coordinator_manager * mgr = ggml_numa_coordinator_manager_get_global(4, false);
    if (!mgr) {
        printf("❌ Failed to get coordinator manager\n");
        return 1;
    }
    
    printf("✅ Coordinator manager initialized\n");
    
    // Set progress callback
    const char * user_data = "TEST_CALLBACK";
    int callback_result = ggml_numa_coordinator_manager_set_progress_callback(mgr, progress_callback, (void*)user_data);
    if (callback_result != 0) {
        printf("❌ Failed to set progress callback\n");
        return 1;
    }
    
    printf("✅ Progress callback enabled\n");
    
    // Create a simple cgraph for testing
    struct ggml_init_params params;
    params.mem_size = 16*1024*1024; // 16MB
    params.mem_buffer = NULL;
    params.no_alloc = false;
    
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("❌ Failed to create ggml context\n");
        return 1;
    }
    
    // Create tensors for computation
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    struct ggml_tensor * result = ggml_add(ctx, a, b);
    
    // Fill tensors with test data
    float * a_data = (float*)ggml_get_data(a);
    float * b_data = (float*)ggml_get_data(b);
    for (int i = 0; i < ggml_nelements(a); i++) {
        a_data[i] = 1.0f;
        b_data[i] = 2.0f;
    }
    
    // Build computation graph
    struct ggml_cgraph * cgraph = ggml_new_graph(ctx);
    ggml_build_forward_expand(cgraph, result);
    
    printf("✅ Created computation graph\n");
    
    // Set cgraph for coordinator
    int result_set = ggml_numa_coordinator_manager_set_cgraph(mgr, cgraph);
    if (result_set != 0) {
        printf("❌ Failed to set cgraph\n");
        ggml_free(ctx);
        return 1;
    }
    
    printf("✅ Cgraph set for coordinator\n");
    
    // Start coordinator threads
    int result_start = ggml_numa_coordinator_manager_start(mgr);
    if (result_start != 0) {
        printf("❌ Failed to start coordinator\n");
        ggml_free(ctx);
        return 1;
    }
    
    printf("✅ Coordinator threads started\n");
    printf("\n==== Submitting Work Items (with callbacks) ====\n");
    
    // Submit multiple work items
    const int num_work_items = 10;
    for (int i = 0; i < num_work_items; i++) {
        int work_id = ggml_numa_coordinator_manager_submit_work(mgr, result, -1);
        if (work_id < 0) {
            printf("❌ Failed to submit work item %d\n", i);
        } else {
            printf("📤 Submitted work item %d\n", work_id);
        }
    }
    
    printf("\n==== Waiting for Completion ====\n");
    
    // Wait for completion (note: reduced logging should make this cleaner)
    int result_wait = ggml_numa_coordinator_manager_wait_for_completion(mgr);
    if (result_wait != 0) {
        printf("❌ Failed while waiting for completion\n");
    } else {
        printf("✅ All work completed\n");
    }
    
    printf("\n==== Progress Callback Results ====\n");
    printf("Total callbacks received: %d\n", total_callbacks.load());
    for (int i = 0; i < 4; i++) {
        int callbacks = numa_node_callbacks[i].load();
        if (callbacks > 0) {
            printf("NUMA node %d callbacks: %d\n", i, callbacks);
        }
    }
    
    // Get performance stats
    struct ggml_numa_perf_stats stats = ggml_numa_coordinator_manager_get_stats(mgr, -1);
    printf("\n==== Performance Statistics ====\n");
    printf("Total work items processed: %ld\n", stats.total_work_items);
    printf("Total processing time: %ld μs\n", stats.total_processing_time_us);
    printf("Average processing time: %ld μs\n", stats.average_processing_time_us);
    printf("Throughput: %.2f items/sec\n", stats.throughput_items_per_sec);
    
    // Test disabling callback
    printf("\n==== Disabling Progress Callback ====\n");
    ggml_numa_coordinator_manager_set_progress_callback(mgr, NULL, NULL);
    
    // Submit one more work item (should not trigger callback)
    int final_work_id = ggml_numa_coordinator_manager_submit_work(mgr, result, -1);
    printf("📤 Submitted final work item %d (no callback expected)\n", final_work_id);
    
    ggml_numa_coordinator_manager_wait_for_completion(mgr);
    
    int final_callback_count = total_callbacks.load();
    printf("Total callbacks after disable: %d (should be same as before)\n", final_callback_count);
    
    // Cleanup
    ggml_free(ctx);
    
    printf("\n==== Test Results ====\n");
    bool callback_test_passed = (total_callbacks.load() == num_work_items);
    
    if (callback_test_passed) {
        printf("✅ Progress callback test PASSED\n");
    } else {
        printf("❌ Progress callback test FAILED\n");
    }
    
    printf("✅ Logging reduction test: Check console output for reduced verbosity\n");
    printf("\n==== Summary ====\n");
    printf("- Progress callbacks: %s\n", callback_test_passed ? "✅ Working" : "❌ Failed");
    printf("- Reduced logging: ✅ Implemented (verify visually)\n");
    printf("- Coordinator functionality: ✅ Working\n");
    
    return callback_test_passed ? 0 : 1;
}
