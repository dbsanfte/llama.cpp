/**
 * NUMA Coordinator Comprehensive Test Suite
 * 
 * This file tests various scenarios around waiting for work completion
 * AND execution strategy verification to identify bugs in the coordination mechanism.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <atomic>
#include <sched.h>

#ifdef __linux__
#include <numa.h>
#include <numaif.h>
#endif

#include "ggml.h"
#include "ggml-cpu.h" 
#include "ggml-cpu-impl.h"
#include "ggml-numa-operation-dispatch.h"
#include "ggml-numa-coordinator.h"

// Test work function that simulates different execution times
static enum ggml_status test_work_function_quick(void* context, struct ggml_compute_params* params) {
    (void)context; (void)params; // Suppress warnings
    printf("⚡ Quick work function executing...\n");
    usleep(10000); // 10ms work
    printf("⚡ Quick work function completed\n");
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status test_work_function_medium(void* context, struct ggml_compute_params* params) {
    (void)context; (void)params; // Suppress warnings
    printf("🔥 Medium work function executing...\n");
    usleep(50000); // 50ms work
    printf("🔥 Medium work function completed\n");
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status test_work_function_slow(void* context, struct ggml_compute_params* params) {
    (void)context; (void)params; // Suppress warnings
    printf("🐌 Slow work function executing...\n");
    usleep(200000); // 200ms work
    printf("🐌 Slow work function completed\n");
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status test_work_function_very_slow(void* context, struct ggml_compute_params* params) {
    (void)context; (void)params; // Suppress warnings
    printf("🚶 Very slow work function executing...\n");
    usleep(500000); // 500ms work
    printf("🚶 Very slow work function completed\n");
    return GGML_STATUS_SUCCESS;
}

// Test context data
struct test_work_context {
    int test_id;
    const char* test_name;
    std::atomic<int> execution_count{0};
};

static void print_test_header(const char* test_name) {
    printf("\n");
    printf("========================================================================\n");
    printf("--- Test: %s ---\n", test_name);
    printf("========================================================================\n");
}

static void print_test_result(const char* test_name, bool passed) {
    if (passed) {
        printf("✅ %s: PASS\n", test_name);
    } else {
        printf("❌ %s: FAIL\n", test_name);
    }
}

// Test 1: Single work item wait
static bool test_single_work_item_wait() {
    print_test_header("Single Work Item Wait");
    
    // Initialize NUMA with MIRROR mode for testing
    struct ggml_threadpool_params params = {};  // Initialize all to zero
    params.n_threads = -1;  // Auto-detect
    params.poll = 50;
    params.strict_cpu = false;
    params.paused = false;
    params.force_multi_socket = true;
    
    printf("🔧 Initializing coordinator with MIRROR mode...\n");
    ggml_numa_init_with_threadpool_params(GGML_NUMA_STRATEGY_MIRROR, &params);
    
    struct ggml_numa_coordinator_manager* mgr = ggml_numa_coordinator_manager_get_global_with_params(&params);
    if (!mgr) {
        printf("❌ Failed to get coordinator manager\n");
        return false;
    }
    
    printf("🎯 Creating test context...\n");
    struct test_work_context context = {1, "single_item"};
    
    printf("📤 Submitting single work item...\n");
    
    // Set up execution strategy
    ggml_numa_execution_strategy_t strategy = {
        .node_strategy = NUMA_NODE_STRATEGY_SINGLE,
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
    };
    
    int work_id = ggml_numa_coordinator_manager_submit_work_function(
        mgr,
        test_work_function_medium,
        &context,
        0,  // NUMA node 0
        strategy,
        1024  // Buffer size
    );
    
    if (work_id < 0) {
        printf("❌ Failed to submit work item\n");
        return false;
    }
    
    printf("⏰ Starting timer for wait test...\n");
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    printf("⏳ Waiting for completion...\n");
    int wait_result = ggml_numa_coordinator_manager_wait_for_completion(mgr);
    
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double elapsed_ms = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + 
                       (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;
    
    printf("⏱️  Wait completed in %.2f ms (result: %d)\n", elapsed_ms, wait_result);
    
    bool success = (wait_result == 0) && (elapsed_ms >= 40) && (elapsed_ms <= 200);
    printf("📊 Expected: 50-100ms, Actual: %.2fms\n", elapsed_ms);
    
    return success;
}

// Test 2: Multiple work items wait
static bool test_multiple_work_items_wait() {
    print_test_header("Multiple Work Items Wait");
    
    struct ggml_numa_coordinator_manager* mgr = ggml_numa_coordinator_manager_get_global(-1, true);
    if (!mgr) {
        printf("❌ Failed to get coordinator manager\n");
        return false;
    }
    
    struct test_work_context contexts[3] = {
        { 2, "multi_item_1" },
        { 3, "multi_item_2" },
        { 4, "multi_item_3" }
    };
    
    printf("📤 Submitting 3 work items...\n");
    int work_ids[3];
    
    // Set up execution strategy
    ggml_numa_execution_strategy_t strategy = {
        .node_strategy = NUMA_NODE_STRATEGY_SINGLE,
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
    };
    
    // Submit different duration work items
    work_ids[0] = ggml_numa_coordinator_manager_submit_work_function(
        mgr, test_work_function_quick, &contexts[0],
        0, strategy, 1024
    );
    
    work_ids[1] = ggml_numa_coordinator_manager_submit_work_function(
        mgr, test_work_function_medium, &contexts[1],
        0, strategy, 1024
    );
    
    work_ids[2] = ggml_numa_coordinator_manager_submit_work_function(
        mgr, test_work_function_slow, &contexts[2],
        0, strategy, 1024
    );
    
    bool all_submitted = true;
    for (int i = 0; i < 3; i++) {
        if (work_ids[i] < 0) {
            printf("❌ Failed to submit work item %d\n", i);
            all_submitted = false;
        }
    }
    
    if (!all_submitted) {
        return false;
    }
    
    printf("⏰ Starting timer for multiple items wait test...\n");
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    printf("⏳ Waiting for all items to complete...\n");
    int wait_result = ggml_numa_coordinator_manager_wait_for_completion(mgr);
    
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double elapsed_ms = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + 
                       (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;
    
    printf("⏱️  Multiple items wait completed in %.2f ms (result: %d)\n", elapsed_ms, wait_result);
    
    // Should take at least as long as the slowest item (200ms)
    bool success = (wait_result == 0) && (elapsed_ms >= 180) && (elapsed_ms <= 400);
    printf("📊 Expected: 200-300ms, Actual: %.2fms\n", elapsed_ms);
    
    return success;
}

// Test 3: Rapid succession wait test
static bool test_rapid_succession_wait() {
    print_test_header("Rapid Succession Wait");
    
    struct ggml_numa_coordinator_manager* mgr = ggml_numa_coordinator_manager_get_global(-1, true);
    if (!mgr) {
        printf("❌ Failed to get coordinator manager\n");
        return false;
    }
    
    printf("🚀 Testing rapid submission and wait cycles...\n");
    bool all_success = true;
    
    // Set up execution strategy
    ggml_numa_execution_strategy_t strategy = {
        .node_strategy = NUMA_NODE_STRATEGY_SINGLE,
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
    };
    
    for (int cycle = 0; cycle < 5; cycle++) {
        printf("📋 Cycle %d: Submit -> Wait -> Verify\n", cycle + 1);
        
        struct test_work_context context = { 10 + cycle, "rapid_cycle" };
        
        struct timespec cycle_start, cycle_end;
        clock_gettime(CLOCK_MONOTONIC, &cycle_start);
        
        int work_id = ggml_numa_coordinator_manager_submit_work_function(
            mgr, test_work_function_quick, &context,
            0, strategy, 1024
        );
        
        if (work_id < 0) {
            printf("❌ Cycle %d: Failed to submit work\n", cycle + 1);
            all_success = false;
            continue;
        }
        
        int wait_result = ggml_numa_coordinator_manager_wait_for_completion(mgr);
        
        clock_gettime(CLOCK_MONOTONIC, &cycle_end);
        double cycle_ms = (cycle_end.tv_sec - cycle_start.tv_sec) * 1000.0 + 
                         (cycle_end.tv_nsec - cycle_start.tv_nsec) / 1000000.0;
        
        if (wait_result != 0) {
            printf("❌ Cycle %d: Wait failed (result: %d)\n", cycle + 1, wait_result);
            all_success = false;
        } else if (cycle_ms > 50) {
            printf("⚠️  Cycle %d: Slower than expected (%.2fms)\n", cycle + 1, cycle_ms);
        } else {
            printf("✅ Cycle %d: Success (%.2fms)\n", cycle + 1, cycle_ms);
        }
    }
    
    return all_success;
}

// Test 4: Wait timeout behavior
static bool test_wait_timeout_behavior() {
    print_test_header("Wait Timeout Behavior");
    
    struct ggml_numa_coordinator_manager* mgr = ggml_numa_coordinator_manager_get_global(-1, true);
    if (!mgr) {
        printf("❌ Failed to get coordinator manager\n");
        return false;
    }
    
    printf("🐌 Submitting very slow work item...\n");
    struct test_work_context context = { 20, "timeout_test" };
    
    // Set up execution strategy
    ggml_numa_execution_strategy_t strategy = {
        .node_strategy = NUMA_NODE_STRATEGY_SINGLE,
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
    };
    
    int work_id = ggml_numa_coordinator_manager_submit_work_function(
        mgr, test_work_function_very_slow, &context,
        0, strategy, 1024
    );
    
    if (work_id < 0) {
        printf("❌ Failed to submit very slow work item\n");
        return false;
    }
    
    printf("⏰ Testing wait behavior with long-running work...\n");
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    // This should take ~500ms for the work to complete
    int wait_result = ggml_numa_coordinator_manager_wait_for_completion(mgr);
    
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double elapsed_ms = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + 
                       (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;
    
    printf("⏱️  Long wait completed in %.2f ms (result: %d)\n", elapsed_ms, wait_result);
    
    // Should wait for the full 500ms
    bool success = (wait_result == 0) && (elapsed_ms >= 480) && (elapsed_ms <= 700);
    printf("📊 Expected: 500-600ms, Actual: %.2fms\n", elapsed_ms);
    
    return success;
}

// Test 5: Immediate wait (no work submitted)
static bool test_immediate_wait_no_work() {
    print_test_header("Immediate Wait (No Work)");
    
    struct ggml_numa_coordinator_manager* mgr = ggml_numa_coordinator_manager_get_global(-1, true);
    if (!mgr) {
        printf("❌ Failed to get coordinator manager\n");
        return false;
    }
    
    printf("💨 Testing wait with no work submitted...\n");
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    int wait_result = ggml_numa_coordinator_manager_wait_for_completion(mgr);
    
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double elapsed_ms = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + 
                       (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;
    
    printf("⏱️  Immediate wait completed in %.2f ms (result: %d)\n", elapsed_ms, wait_result);
    
    // Should return immediately (< 5ms)
    bool success = (wait_result == 0) && (elapsed_ms < 10);
    printf("📊 Expected: <10ms, Actual: %.2fms\n", elapsed_ms);
    
    return success;
}

// Test 6: Concurrent waits from multiple threads
static struct {
    struct ggml_numa_coordinator_manager* mgr;
    std::atomic<int> wait_results[3];
    double wait_times[3];
    bool test_complete;
} concurrent_test_data;

static void* concurrent_wait_thread(void* arg) {
    int thread_id = *(int*)arg;
    
    printf("🧵 Thread %d: Starting concurrent wait...\n", thread_id);
    
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    int result = ggml_numa_coordinator_manager_wait_for_completion(concurrent_test_data.mgr);
    
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double elapsed_ms = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + 
                       (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;
    
    concurrent_test_data.wait_results[thread_id].store(result);
    concurrent_test_data.wait_times[thread_id] = elapsed_ms;
    
    printf("🧵 Thread %d: Wait completed in %.2fms (result: %d)\n", 
           thread_id, elapsed_ms, result);
    
    return NULL;
}

static bool test_concurrent_waits() {
    print_test_header("Concurrent Waits from Multiple Threads");
    
    concurrent_test_data.mgr = ggml_numa_coordinator_manager_get_global(-1, true);
    if (!concurrent_test_data.mgr) {
        printf("❌ Failed to get coordinator manager\n");
        return false;
    }
    
    // Initialize atomic variables
    for (int i = 0; i < 3; i++) {
        concurrent_test_data.wait_results[i].store(-999);
    }
    concurrent_test_data.test_complete = false;
    
    printf("🚀 Starting 3 concurrent wait threads...\n");
    pthread_t wait_threads[3];
    int thread_ids[3] = {0, 1, 2};
    
    // Start wait threads first
    for (int i = 0; i < 3; i++) {
        int result = pthread_create(&wait_threads[i], NULL, concurrent_wait_thread, &thread_ids[i]);
        if (result != 0) {
            printf("❌ Failed to create wait thread %d\n", i);
            return false;
        }
    }
    
    // Give threads time to start waiting
    usleep(50000); // 50ms
    
    printf("📤 Submitting work while threads are waiting...\n");
    struct test_work_context context = { 30, "concurrent_wait" };
    
    // Set up execution strategy
    ggml_numa_execution_strategy_t strategy = {
        .node_strategy = NUMA_NODE_STRATEGY_SINGLE,
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
    };
    
    int work_id = ggml_numa_coordinator_manager_submit_work_function(
        concurrent_test_data.mgr, test_work_function_medium, &context,
        0, strategy, 1024
    );
    
    if (work_id < 0) {
        printf("❌ Failed to submit work during concurrent wait test\n");
        return false;
    }
    
    printf("⏳ Waiting for all wait threads to complete...\n");
    bool all_success = true;
    
    // Join all wait threads
    for (int i = 0; i < 3; i++) {
        pthread_join(wait_threads[i], NULL);
        
        int result = concurrent_test_data.wait_results[i].load();
        double time_ms = concurrent_test_data.wait_times[i];
        
        if (result != 0) {
            printf("❌ Thread %d wait failed (result: %d)\n", i, result);
            all_success = false;
        } else if (time_ms < 40 || time_ms > 200) {
            printf("⚠️  Thread %d timing suspicious (%.2fms)\n", i, time_ms);
        } else {
            printf("✅ Thread %d wait successful (%.2fms)\n", i, time_ms);
        }
    }
    
    return all_success;
}

//
// Execution Strategy Verification Tests
//

// Test work function that tracks which NUMA node and thread count it's executed on
struct execution_tracking_context {
    std::atomic<int> numa_node_executions[4];  // Track executions per NUMA node
    std::atomic<int> total_executions;
    std::atomic<int> single_thread_executions;
    std::atomic<int> multi_thread_executions;
    std::atomic<int> max_concurrent_threads;
    pthread_mutex_t execution_mutex;
    
    execution_tracking_context() : total_executions(0), single_thread_executions(0), 
                                   multi_thread_executions(0), max_concurrent_threads(0) {
        for (int i = 0; i < 4; i++) {
            numa_node_executions[i] = 0;
        }
        pthread_mutex_init(&execution_mutex, NULL);
    }
    
    ~execution_tracking_context() {
        pthread_mutex_destroy(&execution_mutex);
    }
};

static enum ggml_status test_work_function_execution_tracking(void* context, struct ggml_compute_params* params) {
    execution_tracking_context* tracking = (execution_tracking_context*)context;
    
    // Get NUMA node with proper detection hierarchy
    int numa_node = 0;  // Default fallback
    
    // First: Check if we have virtual NUMA node information (for simulated environments)
    extern int ggml_numa_get_current_node(void);
    int virtual_node = ggml_numa_get_current_node();
    if (virtual_node >= 0) {
        // Use virtual NUMA node from coordinator (works in simulated environments)
        numa_node = virtual_node;
    } else {
        // Real NUMA system: detect actual NUMA node
        int cpu = sched_getcpu();
        if (cpu >= 0) {
#ifdef __linux__
            // Use proper NUMA detection if available
            if (numa_available() != -1) {
                numa_node = numa_node_of_cpu(cpu);
                // Fallback if numa_node_of_cpu fails
                if (numa_node < 0) {
                    numa_node = cpu % 2;  // Simple fallback for testing
                }
            } else {
                numa_node = cpu % 2;  // Simple mapping when NUMA not available
            }
#else
            numa_node = cpu % 2;  // Simple mapping for non-Linux systems
#endif
        }
        // If sched_getcpu() fails, numa_node stays 0 (default)
    }
    
    pthread_mutex_lock(&tracking->execution_mutex);
    
    // Track execution details
    tracking->total_executions++;
    tracking->numa_node_executions[numa_node]++;
    
    // Track threading strategy based on compute params
    if (params && params->nth > 1) {
        tracking->multi_thread_executions++;
        if (params->nth > tracking->max_concurrent_threads) {
            tracking->max_concurrent_threads = params->nth;
        }
    } else {
        tracking->single_thread_executions++;
    }
    
    printf("📊 Execution tracking: NUMA=%d, threads=%d, ith=%d\n", 
           numa_node, params ? params->nth : 1, params ? params->ith : 0);
    
    pthread_mutex_unlock(&tracking->execution_mutex);
    
    // Simulate some work
    usleep(25000); // 25ms work
    
    return GGML_STATUS_SUCCESS;
}

// Test: Single Node + Single Thread Strategy
bool test_execution_strategy_single_single() {
    printf("--- Test: Single Node + Single Thread Strategy ---\n");
    printf("Testing NUMA_NODE_STRATEGY_SINGLE + NUMA_ON_NODE_STRATEGY_SINGLE_THREAD...\n");
    
    execution_tracking_context tracking;
    
    // Get global coordinator manager with 2 NUMA nodes, 4 threads each
    extern struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_get_global(int n_threads, bool force_multi_socket);
    struct ggml_numa_coordinator_manager * mgr = ggml_numa_coordinator_manager_get_global(8, true);
    
    if (!mgr) {
        printf("  ❌ Failed to get coordinator manager\n");
        return false;
    }
    
    // Submit work with single node + single thread strategy
    ggml_numa_execution_strategy_t strategy = {
        .node_strategy = NUMA_NODE_STRATEGY_SINGLE,
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD
    };
    
    int work_id = ggml_numa_coordinator_manager_submit_work_function(
        mgr,
        test_work_function_execution_tracking,
        &tracking,
        0,  // Target NUMA node 0
        strategy,
        1024  // Small buffer size
    );
    
    if (work_id < 0) {
        printf("  ❌ Failed to submit work (ID: %d)\n", work_id);
        return false;
    }
    
    printf("  ✅ Work submitted (ID: %d) with SINGLE+SINGLE_THREAD strategy\n", work_id);
    
    // Wait for completion
    int wait_result = ggml_numa_coordinator_manager_wait_for_completion(mgr);
    if (wait_result != 0) {
        printf("  ❌ Wait failed (result: %d)\n", wait_result);
        return false;
    }
    
    // Verify execution strategy was followed
    printf("  📊 Execution Results:\n");
    printf("    Total executions: %d\n", tracking.total_executions.load());
    printf("    Single-thread executions: %d\n", tracking.single_thread_executions.load());
    printf("    Multi-thread executions: %d\n", tracking.multi_thread_executions.load());
    printf("    NUMA node 0 executions: %d\n", tracking.numa_node_executions[0].load());
    printf("    NUMA node 1 executions: %d\n", tracking.numa_node_executions[1].load());
    printf("    Max concurrent threads: %d\n", tracking.max_concurrent_threads.load());
    
    // Validate strategy compliance
    bool strategy_correct = true;
    
    if (tracking.total_executions.load() == 0) {
        printf("  ❌ No executions recorded\n");
        strategy_correct = false;
    }
    
    if (tracking.multi_thread_executions.load() > 0) {
        printf("  ❌ Multi-thread executions found in SINGLE_THREAD strategy\n");
        strategy_correct = false;
    }
    
    if (tracking.numa_node_executions[1].load() > 0) {
        printf("  ❌ Executions found on NUMA node 1 in SINGLE strategy (target was node 0)\n");
        strategy_correct = false;
    }
    
    if (strategy_correct) {
        printf("  ✅ Strategy SINGLE+SINGLE_THREAD correctly enforced\n");
    } else {
        printf("  ❌ Strategy SINGLE+SINGLE_THREAD not properly enforced\n");
    }
    
    return strategy_correct;
}

// Test: Single Node + Multi Thread Strategy
bool test_execution_strategy_single_multi() {
    printf("--- Test: Single Node + Multi Thread Strategy ---\n");
    printf("Testing NUMA_NODE_STRATEGY_SINGLE + NUMA_ON_NODE_STRATEGY_MULTI_THREAD...\n");
    
    execution_tracking_context tracking;
    
    extern struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_get_global(int n_threads, bool force_multi_socket);
    struct ggml_numa_coordinator_manager * mgr = ggml_numa_coordinator_manager_get_global(8, true);
    
    if (!mgr) {
        printf("  ❌ Failed to get coordinator manager\n");
        return false;
    }
    
    // Submit work with single node + multi thread strategy
    ggml_numa_execution_strategy_t strategy = {
        .node_strategy = NUMA_NODE_STRATEGY_SINGLE,
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
    };
    
    int work_id = ggml_numa_coordinator_manager_submit_work_function(
        mgr,
        test_work_function_execution_tracking,
        &tracking,
        1,  // Target NUMA node 1
        strategy,
        1024
    );
    
    if (work_id < 0) {
        printf("  ❌ Failed to submit work (ID: %d)\n", work_id);
        return false;
    }
    
    printf("  ✅ Work submitted (ID: %d) with SINGLE+MULTI_THREAD strategy\n", work_id);
    
    // Wait for completion
    int wait_result = ggml_numa_coordinator_manager_wait_for_completion(mgr);
    if (wait_result != 0) {
        printf("  ❌ Wait failed (result: %d)\n", wait_result);
        return false;
    }
    
    // Verify execution strategy was followed
    printf("  📊 Execution Results:\n");
    printf("    Total executions: %d\n", tracking.total_executions.load());
    printf("    Single-thread executions: %d\n", tracking.single_thread_executions.load());
    printf("    Multi-thread executions: %d\n", tracking.multi_thread_executions.load());
    printf("    NUMA node 0 executions: %d\n", tracking.numa_node_executions[0].load());
    printf("    NUMA node 1 executions: %d\n", tracking.numa_node_executions[1].load());
    printf("    Max concurrent threads: %d\n", tracking.max_concurrent_threads.load());
    
    // Validate strategy compliance
    bool strategy_correct = true;
    
    if (tracking.total_executions.load() == 0) {
        printf("  ❌ No executions recorded\n");
        strategy_correct = false;
    }
    
    if (tracking.numa_node_executions[0].load() > 0) {
        printf("  ❌ Executions found on NUMA node 0 in SINGLE strategy (target was node 1)\n");
        strategy_correct = false;
    }
    
    if (tracking.max_concurrent_threads.load() <= 1) {
        printf("  ❌ Multi-threading not detected in MULTI_THREAD strategy\n");
        strategy_correct = false;
    }
    
    if (strategy_correct) {
        printf("  ✅ Strategy SINGLE+MULTI_THREAD correctly enforced\n");
    } else {
        printf("  ❌ Strategy SINGLE+MULTI_THREAD not properly enforced\n");
    }
    
    return strategy_correct;
}

// Test: Data Parallel + Single Thread Strategy
bool test_execution_strategy_data_parallel_single() {
    printf("--- Test: Data Parallel + Single Thread Strategy ---\n");
    printf("Testing NUMA_NODE_STRATEGY_DATA_PARALLEL + NUMA_ON_NODE_STRATEGY_SINGLE_THREAD...\n");
    
    execution_tracking_context tracking;
    
    extern struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_get_global(int n_threads, bool force_multi_socket);
    struct ggml_numa_coordinator_manager * mgr = ggml_numa_coordinator_manager_get_global(8, true);
    
    if (!mgr) {
        printf("  ❌ Failed to get coordinator manager\n");
        return false;
    }
    
    // Submit work with data parallel + single thread strategy
    ggml_numa_execution_strategy_t strategy = {
        .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL,
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD
    };
    
    int work_id = ggml_numa_coordinator_manager_submit_work_function(
        mgr,
        test_work_function_execution_tracking,
        &tracking,
        -1,  // Auto-select for data parallelism
        strategy,
        1024
    );
    
    if (work_id < 0) {
        printf("  ❌ Failed to submit work (ID: %d)\n", work_id);
        return false;
    }
    
    printf("  ✅ Work submitted (ID: %d) with DATA_PARALLEL+SINGLE_THREAD strategy\n", work_id);
    
    // Wait for completion
    int wait_result = ggml_numa_coordinator_manager_wait_for_completion(mgr);
    if (wait_result != 0) {
        printf("  ❌ Wait failed (result: %d)\n", wait_result);
        return false;
    }
    
    // Verify execution strategy was followed
    printf("  📊 Execution Results:\n");
    printf("    Total executions: %d\n", tracking.total_executions.load());
    printf("    Single-thread executions: %d\n", tracking.single_thread_executions.load());
    printf("    Multi-thread executions: %d\n", tracking.multi_thread_executions.load());
    printf("    NUMA node 0 executions: %d\n", tracking.numa_node_executions[0].load());
    printf("    NUMA node 1 executions: %d\n", tracking.numa_node_executions[1].load());
    printf("    Max concurrent threads: %d\n", tracking.max_concurrent_threads.load());
    
    // Validate strategy compliance
    bool strategy_correct = true;
    
    if (tracking.total_executions.load() == 0) {
        printf("  ❌ No executions recorded\n");
        strategy_correct = false;
    }
    
    if (tracking.multi_thread_executions.load() > 0) {
        printf("  ❌ Multi-thread executions found in SINGLE_THREAD strategy\n");
        strategy_correct = false;
    }
    
    // For data parallel, we should see executions on multiple NUMA nodes
    int nodes_with_executions = 0;
    for (int i = 0; i < 2; i++) {
        if (tracking.numa_node_executions[i].load() > 0) {
            nodes_with_executions++;
        }
    }
    
    if (nodes_with_executions < 2) {
        printf("  ❌ Data parallelism not detected - only %d NUMA nodes used\n", nodes_with_executions);
        strategy_correct = false;
    }
    
    if (strategy_correct) {
        printf("  ✅ Strategy DATA_PARALLEL+SINGLE_THREAD correctly enforced\n");
    } else {
        printf("  ❌ Strategy DATA_PARALLEL+SINGLE_THREAD not properly enforced\n");
    }
    
    return strategy_correct;
}

// Test: Data Parallel + Multi Thread Strategy
bool test_execution_strategy_data_parallel_multi() {
    printf("--- Test: Data Parallel + Multi Thread Strategy ---\n");
    printf("Testing NUMA_NODE_STRATEGY_DATA_PARALLEL + NUMA_ON_NODE_STRATEGY_MULTI_THREAD...\n");
    
    execution_tracking_context tracking;
    
    extern struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_get_global(int n_threads, bool force_multi_socket);
    struct ggml_numa_coordinator_manager * mgr = ggml_numa_coordinator_manager_get_global(8, true);
    
    if (!mgr) {
        printf("  ❌ Failed to get coordinator manager\n");
        return false;
    }
    
    // Submit work with data parallel + multi thread strategy (most complex)
    ggml_numa_execution_strategy_t strategy = {
        .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL,
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
    };
    
    int work_id = ggml_numa_coordinator_manager_submit_work_function(
        mgr,
        test_work_function_execution_tracking,
        &tracking,
        -1,  // Auto-select for data parallelism
        strategy,
        1024
    );
    
    if (work_id < 0) {
        printf("  ❌ Failed to submit work (ID: %d)\n", work_id);
        return false;
    }
    
    printf("  ✅ Work submitted (ID: %d) with DATA_PARALLEL+MULTI_THREAD strategy\n", work_id);
    
    // Wait for completion
    int wait_result = ggml_numa_coordinator_manager_wait_for_completion(mgr);
    if (wait_result != 0) {
        printf("  ❌ Wait failed (result: %d)\n", wait_result);
        return false;
    }
    
    // Verify execution strategy was followed
    printf("  📊 Execution Results:\n");
    printf("    Total executions: %d\n", tracking.total_executions.load());
    printf("    Single-thread executions: %d\n", tracking.single_thread_executions.load());
    printf("    Multi-thread executions: %d\n", tracking.multi_thread_executions.load());
    printf("    NUMA node 0 executions: %d\n", tracking.numa_node_executions[0].load());
    printf("    NUMA node 1 executions: %d\n", tracking.numa_node_executions[1].load());
    printf("    Max concurrent threads: %d\n", tracking.max_concurrent_threads.load());
    
    // Validate strategy compliance
    bool strategy_correct = true;
    
    if (tracking.total_executions.load() == 0) {
        printf("  ❌ No executions recorded\n");
        strategy_correct = false;
    }
    
    // For data parallel + multi thread, we should see:
    // 1. Multiple NUMA nodes used
    int nodes_with_executions = 0;
    for (int i = 0; i < 2; i++) {
        if (tracking.numa_node_executions[i].load() > 0) {
            nodes_with_executions++;
        }
    }
    
    if (nodes_with_executions < 2) {
        printf("  ❌ Data parallelism not detected - only %d NUMA nodes used\n", nodes_with_executions);
        strategy_correct = false;
    }
    
    // 2. Multi-threading used
    if (tracking.max_concurrent_threads.load() <= 1) {
        printf("  ❌ Multi-threading not detected in MULTI_THREAD strategy\n");
        strategy_correct = false;
    }
    
    if (strategy_correct) {
        printf("  ✅ Strategy DATA_PARALLEL+MULTI_THREAD correctly enforced\n");
    } else {
        printf("  ❌ Strategy DATA_PARALLEL+MULTI_THREAD not properly enforced\n");
    }
    
    return strategy_correct;
}

// Test: Mixed Strategy Execution Verification
bool test_execution_strategy_mixed_workload() {
    printf("--- Test: Mixed Strategy Execution Verification ---\n");
    printf("Testing multiple work items with different execution strategies...\n");
    
    struct execution_tracking_context tracking_single;
    struct execution_tracking_context tracking_data_parallel;
    
    extern struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_get_global(int n_threads, bool force_multi_socket);
    struct ggml_numa_coordinator_manager * mgr = ggml_numa_coordinator_manager_get_global(8, true);
    
    if (!mgr) {
        printf("  ❌ Failed to get coordinator manager\n");
        return false;
    }
    
    // Submit work with different strategies simultaneously
    ggml_numa_execution_strategy_t single_strategy = {
        .node_strategy = NUMA_NODE_STRATEGY_SINGLE,
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
    };
    
    ggml_numa_execution_strategy_t data_parallel_strategy = {
        .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL,
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD
    };
    
    // Submit first work item
    int work_id1 = ggml_numa_coordinator_manager_submit_work_function(
        mgr,
        test_work_function_execution_tracking,
        &tracking_single,
        0,  // NUMA node 0
        single_strategy,
        1024
    );
    
    // Submit second work item
    int work_id2 = ggml_numa_coordinator_manager_submit_work_function(
        mgr,
        test_work_function_execution_tracking,
        &tracking_data_parallel,
        -1,  // Auto-select for data parallel
        data_parallel_strategy,
        1024
    );
    
    if (work_id1 < 0 || work_id2 < 0) {
        printf("  ❌ Failed to submit work (IDs: %d, %d)\n", work_id1, work_id2);
        return false;
    }
    
    printf("  ✅ Mixed workload submitted (IDs: %d, %d)\n", work_id1, work_id2);
    
    // Wait for completion
    int wait_result = ggml_numa_coordinator_manager_wait_for_completion(mgr);
    if (wait_result != 0) {
        printf("  ❌ Wait failed (result: %d)\n", wait_result);
        return false;
    }
    
    // Verify both strategies were executed correctly
    printf("  📊 Single Strategy Results:\n");
    printf("    NUMA node 0 executions: %d\n", tracking_single.numa_node_executions[0].load());
    printf("    NUMA node 1 executions: %d\n", tracking_single.numa_node_executions[1].load());
    printf("    Max concurrent threads: %d\n", tracking_single.max_concurrent_threads.load());
    
    printf("  📊 Data Parallel Strategy Results:\n");
    printf("    NUMA node 0 executions: %d\n", tracking_data_parallel.numa_node_executions[0].load());
    printf("    NUMA node 1 executions: %d\n", tracking_data_parallel.numa_node_executions[1].load());
    printf("    Max concurrent threads: %d\n", tracking_data_parallel.max_concurrent_threads.load());
    
    bool mixed_correct = true;
    
    // Validate single strategy (should only use node 0, with multi-threading)
    if (tracking_single.numa_node_executions[1].load() > 0) {
        printf("  ❌ Single strategy used wrong NUMA node\n");
        mixed_correct = false;
    }
    if (tracking_single.max_concurrent_threads.load() <= 1) {
        printf("  ❌ Single strategy didn't use multi-threading\n");
        mixed_correct = false;
    }
    
    // Validate data parallel strategy (should use multiple nodes, single threading)
    int nodes_used = 0;
    for (int i = 0; i < 2; i++) {
        if (tracking_data_parallel.numa_node_executions[i].load() > 0) {
            nodes_used++;
        }
    }
    if (nodes_used < 2) {
        printf("  ❌ Data parallel strategy didn't use multiple nodes\n");
        mixed_correct = false;
    }
    if (tracking_data_parallel.max_concurrent_threads.load() > 1) {
        printf("  ❌ Data parallel strategy used multi-threading when it shouldn't\n");
        mixed_correct = false;
    }
    
    if (mixed_correct) {
        printf("  ✅ Mixed strategy workload correctly executed\n");
    } else {
        printf("  ❌ Mixed strategy workload not properly executed\n");
    }
    
    return mixed_correct;
}

int main() {
    printf("🧪 NUMA Coordinator Comprehensive Test Suite\n");
    printf("============================================\n");
    printf("Testing coordinator wait-for-completion AND execution strategies...\n\n");
    
    struct {
        const char* name;
        bool (*test_func)();
        bool result;
    } tests[] = {
        {"Single Work Item Wait", test_single_work_item_wait, false},
        {"Multiple Work Items Wait", test_multiple_work_items_wait, false},
        {"Rapid Succession Wait", test_rapid_succession_wait, false},
        {"Wait Timeout Behavior", test_wait_timeout_behavior, false},
        {"Immediate Wait (No Work)", test_immediate_wait_no_work, false},
        {"Concurrent Waits", test_concurrent_waits, false}
    };
    
    int num_tests = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    
    printf("========================================================================\n");
    printf("                    WAIT-FOR-COMPLETION TESTS\n");
    printf("========================================================================\n");
    
    for (int i = 0; i < num_tests; i++) {
        printf("\n🎯 Running wait test %d/%d: %s\n", i + 1, num_tests, tests[i].name);
        tests[i].result = tests[i].test_func();
        print_test_result(tests[i].name, tests[i].result);
        
        if (tests[i].result) {
            passed++;
        }
        
        // Small delay between tests
        usleep(100000); // 100ms
    }
    
    printf("\n========================================================================\n");
    printf("                    EXECUTION STRATEGY TESTS\n");
    printf("========================================================================\n");
    
    // Run execution strategy tests (now return bool results)
    printf("\n🔍 Running execution strategy verification tests...\n");
    
    struct {
        const char* name;
        bool (*test_func)();
        bool result;
    } strategy_tests[] = {
        {"Single+SingleThread", test_execution_strategy_single_single, false},
        {"Single+MultiThread", test_execution_strategy_single_multi, false},
        {"DataParallel+SingleThread", test_execution_strategy_data_parallel_single, false},
        {"DataParallel+MultiThread", test_execution_strategy_data_parallel_multi, false},
        {"Mixed Strategy Workload", test_execution_strategy_mixed_workload, false}
    };
    
    int num_strategy_tests = sizeof(strategy_tests) / sizeof(strategy_tests[0]);
    int strategy_passed = 0;
    
    for (int i = 0; i < num_strategy_tests; i++) {
        printf("\n🎯 Running strategy test %d/%d: %s\n", i + 1, num_strategy_tests, strategy_tests[i].name);
        strategy_tests[i].result = strategy_tests[i].test_func();
        print_test_result(strategy_tests[i].name, strategy_tests[i].result);
        
        if (strategy_tests[i].result) {
            strategy_passed++;
        }
        
        // Delay between tests
        usleep(300000); // 300ms delay between strategy tests
    }
    
    printf("\n========================================================================\n");
    printf("                           Test Results Summary\n");
    printf("========================================================================\n");
    
    printf("WAIT-FOR-COMPLETION TESTS:\n");
    for (int i = 0; i < num_tests; i++) {
        printf("%-30s %s\n", tests[i].name, tests[i].result ? "✅ PASS" : "❌ FAIL");
    }
    
    printf("\nEXECUTION STRATEGY TESTS:\n");
    for (int i = 0; i < num_strategy_tests; i++) {
        printf("%-30s %s\n", strategy_tests[i].name, strategy_tests[i].result ? "✅ PASS" : "❌ FAIL");
    }
    
    printf("------------------------------------------------------------------------\n");
    printf("Wait Tests: %d/%d passed ", passed, num_tests);
    printf("| Strategy Tests: %d/%d passed\n", strategy_passed, num_strategy_tests);
    
    bool all_tests_passed = (passed == num_tests) && (strategy_passed == num_strategy_tests);
    
    if (all_tests_passed) {
        printf("🎉 ALL TESTS PASSED!\n");
    } else {
        if (passed != num_tests) {
            printf("⚠️  %d WAIT TESTS FAILED\n", num_tests - passed);
        }
        if (strategy_passed != num_strategy_tests) {
            printf("⚠️  %d STRATEGY TESTS FAILED\n", num_strategy_tests - strategy_passed);
        }
    }
    
    printf("📊 Strategy tests verify coordinator follows assigned execution patterns\n");
    printf("========================================================================\n");
    
    // Cleanup
    printf("\n🧹 Cleaning up coordinator...\n");
    // Note: ggml_numa_cleanup doesn't exist, coordinator cleanup happens automatically
    
    printf("✅ Comprehensive coordinator testing completed!\n");
    
    return all_tests_passed ? 0 : 1;
}
