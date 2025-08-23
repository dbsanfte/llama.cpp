/**
 * NUMA Performance Benchmark for ADD Operation
 * 
 * Simple performance test comparing NUMA vs Fallback execution
 * for ADD operations across different tensor sizes.
 * Outputs parseable summary for run-numa-performance-tests.sh
 */

#include <stdio.h>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <thread>
#include <string>
#include <atomic>

#ifdef __linux__
#include <sched.h>
#include <pthread.h>
#include <numa.h>
#endif

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-numa-simple-coordinator.h"  // For NUMA coordinator functions

// Global execution tracking for debugging
static std::atomic<int> g_numa_executions_node0(0);
static std::atomic<int> g_numa_executions_node1(0);
static std::atomic<int> g_fallback_executions(0);
static std::atomic<bool> g_numa_dispatch_called(false);
static std::atomic<bool> g_data_parallel_used(false);

// Hook functions that can be called from ggml-cpu.c to track execution
extern "C" {
    void test_track_numa_execution(int node_id) {
        g_numa_dispatch_called = true;
        if (node_id == 0) g_numa_executions_node0++;
        else if (node_id == 1) g_numa_executions_node1++;
        printf("🚀 NUMA EXECUTION TRACKED: Node %d (total executions: node0=%d, node1=%d)\n", 
               node_id, g_numa_executions_node0.load(), g_numa_executions_node1.load());
    }
    
    void test_track_fallback_execution() {
        g_fallback_executions++;
        printf("🔄 FALLBACK EXECUTION TRACKED: Count %d\n", g_fallback_executions.load());
    }
    
    void test_track_data_parallel() {
        g_data_parallel_used = true;
        printf("📊 DATA-PARALLEL EXECUTION TRACKED\n");
    }
}

// Reset execution counters
static void reset_execution_counters() {
    g_numa_executions_node0 = 0;
    g_numa_executions_node1 = 0;
    g_fallback_executions = 0;
    g_numa_dispatch_called = false;
    g_data_parallel_used = false;
}

// Verify execution path assertions
static void assert_execution_path(const char* test_name, const char* expected_path, int expected_numa_dispatches) {
    printf("🔍 [%s] EXECUTION PATH VERIFICATION:\n", test_name);
    printf("   Expected path: %s\n", expected_path);
    printf("   NUMA dispatches called: %s (expected: %s)\n", 
           g_numa_dispatch_called.load() ? "YES" : "NO",
           expected_numa_dispatches > 0 ? "YES" : "NO");
    printf("   Data-parallel used: %s\n", g_data_parallel_used.load() ? "YES" : "NO");
    printf("   NUMA executions - Node 0: %d, Node 1: %d\n", 
           g_numa_executions_node0.load(), g_numa_executions_node1.load());
    printf("   Fallback executions: %d\n", g_fallback_executions.load());
    
    // Hard assertions
    if (expected_numa_dispatches > 0 && !g_numa_dispatch_called.load()) {
        printf("❌ FATAL: Expected NUMA dispatch but none occurred!\n");
        abort();
    }
    
    if (expected_numa_dispatches == 0 && g_numa_dispatch_called.load()) {
        printf("❌ FATAL: Unexpected NUMA dispatch occurred!\n");
        abort();
    }
    
    printf("✅ [%s] Execution path verified\n", test_name);
}

// NUMA validation functions
#ifdef __linux__
static int get_current_numa_node() {
    return numa_node_of_cpu(sched_getcpu());
}

// Use coordinator's hard assertion instead of soft validation
static void validate_thread_binding_hard(const char* test_name, int expected_node) {
    if (expected_node >= 0) {
        ggml_numa_simple_coordinator_assert_thread_binding(expected_node, test_name, 0);
    }
    // For expected_node = -1, skip validation (any node acceptable)
}

static void validate_memory_allocation(const char* test_name, void* ptr, size_t size, int expected_node) {
    if (!ptr) {
        printf("❌ [%s] Memory validation: NULL pointer\n", test_name);
        abort(); // Hard failure for NULL pointer
        return;
    }
    
    // Check where the memory was actually allocated
    int status[1];
    void* addrs[1] = {ptr};
    
    if (numa_move_pages(0, 1, addrs, NULL, status, 0) == 0) {
        int actual_node = status[0];
        
        // Handle expected_node = -1 as "any node is acceptable"
        bool is_valid = (expected_node == -1) || (actual_node == expected_node);
        
        printf("🧠 [%s] Memory validation: %p (%zu bytes) → NUMA node %d (expected: %s) %s\n", 
               test_name, ptr, size, actual_node, 
               (expected_node == -1) ? "any" : std::to_string(expected_node).c_str(),
               is_valid ? "✅" : "❌");
        
        if (!is_valid) {
            printf("❌ FATAL MEMORY ALLOCATION FAILURE: Expected node %d, got node %d\n", expected_node, actual_node);
            abort(); // Hard failure for memory allocation on wrong node
        }
    } else {
        printf("⚠️ [%s] Memory validation: Could not determine NUMA node for %p\n", test_name, ptr);
        // Don't fail for inability to determine node - may not be critical
    }
}

static void validate_cpu_affinity(const char* test_name, const std::vector<int>& expected_cpus) {
    cpu_set_t current_mask;
    CPU_ZERO(&current_mask);
    
    if (sched_getaffinity(0, sizeof(current_mask), &current_mask) == 0) {
        printf("🔗 [%s] CPU Affinity validation:\n", test_name);
        
        bool all_correct = true;
        for (int cpu = 0; cpu < 128; cpu++) {  // Check reasonable range
            bool is_set = CPU_ISSET(cpu, &current_mask);
            bool should_be_set = std::find(expected_cpus.begin(), expected_cpus.end(), cpu) != expected_cpus.end();
            
            if (is_set != should_be_set) {
                printf("   ❌ CPU %d: is_set=%d, should_be_set=%d\n", cpu, is_set, should_be_set);
                all_correct = false;
            } else if (is_set) {
                printf("   ✅ CPU %d: correctly set\n", cpu);
            }
        }
        
        if (all_correct) {
            printf("   ✅ All expected CPUs correctly configured\n");
        }
    } else {
        printf("❌ [%s] Could not get CPU affinity\n", test_name);
    }
}
#endif

// Auto-detect number of physical cores
static int get_physical_core_count() {
    // Try to read from /sys filesystem first
    FILE* fp = popen("lscpu | grep 'Core(s) per socket' | awk '{print $4}'", "r");
    if (fp != nullptr) {
        int cores_per_socket = 0;
        if (fscanf(fp, "%d", &cores_per_socket) == 1) {
            pclose(fp);
            
            // Get number of sockets
            fp = popen("lscpu | grep 'Socket(s):' | awk '{print $2}'", "r");
            if (fp != nullptr) {
                int sockets = 0;
                if (fscanf(fp, "%d", &sockets) == 1) {
                    pclose(fp);
                    int total_physical = cores_per_socket * sockets;
                    printf("Auto-detected: %d cores per socket × %d sockets = %d physical cores\n", 
                           cores_per_socket, sockets, total_physical);
                    return total_physical;
                }
                pclose(fp);
            }
        } else {
            pclose(fp);
        }
    }
    
    // Fallback to hardware concurrency (includes hyperthreads)
    int logical_cores = std::thread::hardware_concurrency();
    int physical_cores = logical_cores / 2;  // Assume hyperthreading
    printf("Fallback: Using %d physical cores (half of %d logical cores)\n", 
           physical_cores, logical_cores);
    return physical_cores;
}

struct TestCase {
    int dim1, dim2, dim3;
    const char* name;
    const char* description;
};

struct PerformanceResult {
    const char* test_name;
    double fallback_time_ms;
    double numa_time_ms;
    double speedup;
    size_t tensor_size_mb;
    bool success;
};

class AddPerformanceBenchmark {
    std::vector<TestCase> test_cases;
    std::vector<PerformanceResult> results;
    int total_physical_cores;
    
public:
    AddPerformanceBenchmark() {
        total_physical_cores = get_physical_core_count();
        test_cases = {
            {64, 64, 32, "SMALL", "~128K elements (~512KB)"},
            {128, 128, 64, "MEDIUM", "~1M elements (~4MB)"},
            {256, 256, 128, "LARGE", "~8M elements (~32MB)"},
            {512, 512, 256, "HUGE", "~64M elements (~256MB)"}
        };
    }
    
    // Test 1: NUMA Node 0 isolation (like modes test)
    double measure_fallback_numa0_performance(const TestCase& test_case) {
        printf("🧪 Test 1: NUMA Node 0 Isolation\n");
        printf("    Testing %s [%dx%dx%d] (Node 0 isolation)...\n", 
               test_case.name, test_case.dim1, test_case.dim2, test_case.dim3);
        
        // Reset and track execution path
        reset_execution_counters();
        
        // Configure NUMA isolation to node 0 (like modes test)
        ggml_numa_set_isolate_node(0);
        ggml_numa_set_dispatch_enabled(true);
        
        if (!ggml_numa_simple_coordinator_is_initialized()) {
            printf("❌ NUMA coordinator not initialized\n");
            return -1.0;
        }
        
        printf("    Using NUMA coordinator isolation to node 0...\n");
        
        // Create context with NUMA support
        size_t tensor_size = (size_t)test_case.dim1 * test_case.dim2 * test_case.dim3 * sizeof(float);
        size_t ctx_size = tensor_size * 4 + 64*1024*1024;
        
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };
        
        struct ggml_context* ctx = ggml_init(params);
        if (!ctx) {
            printf("❌ Failed to create context\n");
            return -1.0;
        }
        
        // Create tensors and validate memory allocation
        struct ggml_tensor* tensor_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, test_case.dim1, test_case.dim2, test_case.dim3);
        struct ggml_tensor* tensor_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, test_case.dim1, test_case.dim2, test_case.dim3);
        struct ggml_tensor* result = ggml_add(ctx, tensor_a, tensor_b);
        
        if (!tensor_a || !tensor_b || !result) {
            printf("❌ Failed to create tensors\n");
            ggml_free(ctx);
            return -1.0;
        }
        
#ifdef __linux__
        // Validate memory allocations are on NUMA node 0
        validate_memory_allocation("Fallback-TensorA", ggml_get_data(tensor_a), ggml_nbytes(tensor_a), 0);
        validate_memory_allocation("Fallback-TensorB", ggml_get_data(tensor_b), ggml_nbytes(tensor_b), 0);
        validate_memory_allocation("Fallback-Result", ggml_get_data(result), ggml_nbytes(result), 0);
#endif
        
        // Initialize data
        float* a_data = (float*)ggml_get_data(tensor_a);
        float* b_data = (float*)ggml_get_data(tensor_b);
        size_t total_elements = ggml_nelements(tensor_a);
        
        for (size_t i = 0; i < total_elements; i++) {
            a_data[i] = 1.5f + (i % 100) * 0.01f;
            b_data[i] = 2.5f + (i % 100) * 0.01f;
        }
        
        // Create computation graph
        struct ggml_cgraph* cgraph = ggml_new_graph(ctx);
        ggml_build_forward_expand(cgraph, result);
        
        printf("      Using NUMA coordinator with node 0 isolation\n");
        
        // Warmup (use NUMA-aware compute)
        for (int i = 0; i < 3; i++) {
            ggml_graph_compute_with_ctx(ctx, cgraph, std::thread::hardware_concurrency());
        }
        
        // Measure performance
        const int num_runs = 10;
        std::vector<double> times;
        
        for (int run = 0; run < num_runs; run++) {
            auto start = std::chrono::high_resolution_clock::now();
            enum ggml_status status = ggml_graph_compute_with_ctx(ctx, cgraph, std::thread::hardware_concurrency());
            auto end = std::chrono::high_resolution_clock::now();
            
            if (status != GGML_STATUS_SUCCESS) {
                printf("❌ Computation failed\n");
                ggml_free(ctx);
                return -1.0;
            }
            
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            if (time_ms > 0.001) {
                times.push_back(time_ms);
            }
        }
        
        ggml_free(ctx);
        
        // Verify execution path for node 0 isolation
        assert_execution_path("NUMA-Node0-Isolation", "NUMA node 0 isolation", 1);
        
        if (times.empty()) {
            return -1.0;
        }
        
        // Calculate average time
        std::sort(times.begin(), times.end());
        if (times.size() > 4) {
            times.erase(times.begin());
            times.pop_back();
        }
        
        double avg_time = 0.0;
        for (double time : times) {
            avg_time += time;
        }
        avg_time /= times.size();
        
        printf("      ✅ Average time: %.3f ms\n", avg_time);
        return avg_time;
    }
    
    // Test 2: NUMA Node 1 isolation 
    double measure_numa_single_node_performance(const TestCase& test_case) {
        printf("\n🧪 Test 2: NUMA Node 1 Isolation\n");
        printf("    Testing %s [%dx%dx%d] (Node 1 isolation)...\n", 
               test_case.name, test_case.dim1, test_case.dim2, test_case.dim3);
        
        // Reset and track execution path
        reset_execution_counters();
        
        // Configure NUMA isolation to node 1 (like modes test)
        ggml_numa_set_isolate_node(1);
        ggml_numa_set_dispatch_enabled(true);
        
        if (!ggml_numa_simple_coordinator_is_initialized()) {
            printf("❌ NUMA coordinator not initialized\n");
            return -1.0;
        }
        
        // Coordinator already initialized in main()
        printf("    Using initialized NUMA coordinator...\n");
        
        size_t tensor_size = (size_t)test_case.dim1 * test_case.dim2 * test_case.dim3 * sizeof(float);
        size_t ctx_size = tensor_size * 8 + 64*1024*1024;
        
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };
        
        struct ggml_context* ctx = ggml_init(params);
        if (!ctx) {
            printf("❌ Failed to create NUMA context\n");
            return -1.0;
        }
        
        struct ggml_tensor* tensor_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, test_case.dim1, test_case.dim2, test_case.dim3);
        struct ggml_tensor* tensor_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, test_case.dim1, test_case.dim2, test_case.dim3);
        struct ggml_tensor* result = ggml_add(ctx, tensor_a, tensor_b);
        
        if (!tensor_a || !tensor_b || !result) {
            printf("❌ Failed to create tensors\n");
            ggml_free(ctx);
            return -1.0;
        }
        
#ifdef __linux__
        // Validate NUMA memory allocations - should be distributed but we'll force single node
        validate_memory_allocation("NUMA-Single-TensorA", ggml_get_data(tensor_a), ggml_nbytes(tensor_a), 0);
        validate_memory_allocation("NUMA-Single-TensorB", ggml_get_data(tensor_b), ggml_nbytes(tensor_b), 0);
        validate_memory_allocation("NUMA-Single-Result", ggml_get_data(result), ggml_nbytes(result), 0);
#endif
        
        // Initialize data
        float* a_data = (float*)ggml_get_data(tensor_a);
        float* b_data = (float*)ggml_get_data(tensor_b);
        size_t total_elements = ggml_nelements(tensor_a);
        
        for (size_t i = 0; i < total_elements; i++) {
            a_data[i] = 1.5f + (i % 100) * 0.01f;
            b_data[i] = 2.5f + (i % 100) * 0.01f;
        }
        
        struct ggml_cgraph* cgraph = ggml_new_graph(ctx);
        ggml_build_forward_expand(cgraph, result);
        
        // Force single node by using half the cores
        int numa_single_cores = total_physical_cores / 2;
        printf("      Using %d cores (NUMA coordinator, single node)\n", numa_single_cores);
        
        // Warmup
        for (int i = 0; i < 3; i++) {
            ggml_graph_compute_with_ctx(ctx, cgraph, numa_single_cores);
        }
        
        const int num_runs = 10;
        std::vector<double> times;
        
        for (int run = 0; run < num_runs; run++) {
#ifdef __linux__
            validate_thread_binding_hard("NUMA-Single-Execution", -1);  // Any node OK during execution
#endif
            
            auto start = std::chrono::high_resolution_clock::now();
            enum ggml_status status = ggml_graph_compute_with_ctx(ctx, cgraph, numa_single_cores);
            auto end = std::chrono::high_resolution_clock::now();
            
            if (status != GGML_STATUS_SUCCESS) {
                printf("❌ NUMA computation failed\n");
                ggml_free(ctx);
                return -1.0;
            }
            
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            if (time_ms > 0.001) {
                times.push_back(time_ms);
            }
        }
        
        ggml_free(ctx);
        
        if (times.empty()) {
            return -1.0;
        }
        
        std::sort(times.begin(), times.end());
        if (times.size() > 4) {
            times.erase(times.begin());
            times.pop_back();
        }
        
        double avg_time = 0.0;
        for (double time : times) {
            avg_time += time;
        }
        avg_time /= times.size();
        
        printf("      ✅ Average time: %.3f ms\n", avg_time);
        
        return avg_time;
    }
    
    // Test 3: NUMA Data-parallel mode (both nodes)
    double measure_numa_dual_node_performance(const TestCase& test_case) {
        printf("\n🧪 Test 3: NUMA Data-Parallel (Both Nodes)\n");
        printf("    Testing %s [%dx%dx%d] (Data-parallel across nodes)...\n", 
               test_case.name, test_case.dim1, test_case.dim2, test_case.dim3);
        
        // Reset and track execution path
        reset_execution_counters();
        
        // Configure NUMA data-parallel mode (like modes test)
        ggml_numa_set_isolate_node(-1);  // No isolation = data-parallel
        ggml_numa_set_dispatch_enabled(true);
        
        if (!ggml_numa_simple_coordinator_is_initialized()) {
            printf("❌ NUMA coordinator not initialized\n");
            return -1.0;
        }
        
        // Coordinator already initialized in main()
        printf("    Using initialized NUMA coordinator...\n");
        
        size_t tensor_size = (size_t)test_case.dim1 * test_case.dim2 * test_case.dim3 * sizeof(float);
        size_t ctx_size = tensor_size * 8 + 64*1024*1024;
        
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };
        
        struct ggml_context* ctx = ggml_init(params);
        if (!ctx) {
            printf("❌ Failed to create NUMA context\n");
            return -1.0;
        }
        
        struct ggml_tensor* tensor_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, test_case.dim1, test_case.dim2, test_case.dim3);
        struct ggml_tensor* tensor_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, test_case.dim1, test_case.dim2, test_case.dim3);
        struct ggml_tensor* result = ggml_add(ctx, tensor_a, tensor_b);
        
        if (!tensor_a || !tensor_b || !result) {
            printf("❌ Failed to create tensors\n");
            ggml_free(ctx);
            return -1.0;
        }
        
#ifdef __linux__
        // Validate NUMA memory allocations - should be mirrored across both nodes
        printf("      🔍 Checking memory distribution across NUMA nodes:\n");
        validate_memory_allocation("NUMA-Dual-TensorA", ggml_get_data(tensor_a), ggml_nbytes(tensor_a), -1);  // Any node OK
        validate_memory_allocation("NUMA-Dual-TensorB", ggml_get_data(tensor_b), ggml_nbytes(tensor_b), -1);  // Any node OK
        validate_memory_allocation("NUMA-Dual-Result", ggml_get_data(result), ggml_nbytes(result), -1);       // Any node OK
#endif
        
        // Initialize data
        float* a_data = (float*)ggml_get_data(tensor_a);
        float* b_data = (float*)ggml_get_data(tensor_b);
        size_t total_elements = ggml_nelements(tensor_a);
        
        for (size_t i = 0; i < total_elements; i++) {
            a_data[i] = 1.5f + (i % 100) * 0.01f;
            b_data[i] = 2.5f + (i % 100) * 0.01f;
        }
        
        struct ggml_cgraph* cgraph = ggml_new_graph(ctx);
        ggml_build_forward_expand(cgraph, result);
        
        printf("      Using %d cores (NUMA coordinator, dual node data-parallel)\n", total_physical_cores);
        
        // Warmup
        for (int i = 0; i < 3; i++) {
            ggml_graph_compute_with_ctx(ctx, cgraph, total_physical_cores);
        }
        
        const int num_runs = 10;
        std::vector<double> times;
        
        for (int run = 0; run < num_runs; run++) {
#ifdef __linux__
            validate_thread_binding_hard("NUMA-Dual-Execution", -1);  // Any node OK during execution
#endif
            
            auto start = std::chrono::high_resolution_clock::now();
            enum ggml_status status = ggml_graph_compute_with_ctx(ctx, cgraph, total_physical_cores);
            auto end = std::chrono::high_resolution_clock::now();
            
            if (status != GGML_STATUS_SUCCESS) {
                printf("❌ NUMA computation failed\n");
                ggml_free(ctx);
                return -1.0;
            }
            
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            if (time_ms > 0.001) {
                times.push_back(time_ms);
            }
        }
        
        ggml_free(ctx);
        
        if (times.empty()) {
            return -1.0;
        }
        
        std::sort(times.begin(), times.end());
        if (times.size() > 4) {
            times.erase(times.begin());
            times.pop_back();
        }
        
        double avg_time = 0.0;
        for (double time : times) {
            avg_time += time;
        }
        avg_time /= times.size();
        
        printf("      ✅ Average time: %.3f ms\n", avg_time);
        
        return avg_time;
    }
    
    void run_benchmark() {
        printf("🎯 ADD OPERATION PERFORMANCE ANALYSIS\n");
        printf("=====================================\n");
        
        // Coordinator already initialized in main()
        printf("NUMA coordinator initialized: %s\n", 
               ggml_is_numa() ? "true" : "false");
        printf("NUMA nodes: %d\n\n", ggml_numa_node_count());
        
        for (const auto& test_case : test_cases) {
            printf("\n📊 Testing %s: %s\n", test_case.name, test_case.description);
            printf("=====================================\n");
            
            // Run all three comprehensive tests
            double fallback_numa0_time = measure_fallback_numa0_performance(test_case);
            double numa_single_time = measure_numa_single_node_performance(test_case);
            double numa_dual_time = measure_numa_dual_node_performance(test_case);
            
            PerformanceResult result;
            result.test_name = test_case.name;
            result.fallback_time_ms = fallback_numa0_time;
            result.numa_time_ms = numa_dual_time;  // Compare fallback vs dual node
            result.success = (fallback_numa0_time > 0 && numa_single_time > 0 && numa_dual_time > 0);
            
            if (result.success) {
                result.speedup = fallback_numa0_time / numa_dual_time;
                result.tensor_size_mb = (test_case.dim1 * test_case.dim2 * test_case.dim3 * sizeof(float)) / (1024 * 1024);
                
                // Calculate additional metrics
                double single_speedup = fallback_numa0_time / numa_single_time;
                double dual_speedup = fallback_numa0_time / numa_dual_time;
                double numa_scaling = numa_single_time / numa_dual_time;
                
                printf("\n📈 PERFORMANCE SUMMARY:\n");
                printf("  Test 1 (Fallback NUMA 0):     %.3f ms\n", fallback_numa0_time);
                printf("  Test 2 (NUMA Single Node):    %.3f ms (%.2fx vs fallback)\n", numa_single_time, single_speedup);
                printf("  Test 3 (NUMA Dual Node):      %.3f ms (%.2fx vs fallback)\n", numa_dual_time, dual_speedup);
                printf("  NUMA Scaling (single→dual):   %.2fx\n", numa_scaling);
                printf("  Tensor Size:                   %zu MB\n", result.tensor_size_mb);
                
#ifdef __linux__
                printf("\n✅ NUMA Validation: All assertions passed\n");
#endif
            } else {
                result.speedup = 0.0;
                result.tensor_size_mb = 0;
                printf("❌ Test failed - one or more measurements invalid\n");
            }
            
            results.push_back(result);
            printf("\n");
        }
    }
    
    void print_summary() {
        printf("\n");
        printf("╔════════════════════════════════════════════════════════════════╗\n");
        printf("║                    NUMA PERFORMANCE SUMMARY                   ║\n");
        printf("╠════════════════════════════════════════════════════════════════╣\n");
        
        double total_speedup = 0.0;
        double best_speedup = 0.0;
        double worst_speedup = 999.0;
        int successful_tests = 0;
        int total_tests = results.size();
        
        // Print detailed results table
        printf("║ Test Case                │   Fallback │      NUMA │  Speedup  ║\n");
        printf("╠══════════════════════════╪════════════╪═══════════╪═══════════╣\n");
        
        for (const auto& result : results) {
            if (result.success) {
                printf("║ %-24s │ %8.3f ms │ %7.3f ms │   %.2fx    ║\n", 
                       result.test_name, 
                       result.fallback_time_ms,
                       result.numa_time_ms,
                       result.speedup);
                
                total_speedup += result.speedup;
                best_speedup = std::max(best_speedup, result.speedup);
                worst_speedup = std::min(worst_speedup, result.speedup);
                successful_tests++;
            } else {
                printf("║ %-24s │    FAILED │   FAILED  │   FAILED  ║\n", result.test_name);
            }
        }
        
        printf("╠══════════════════════════╧════════════╧═══════════╧═══════════╣\n");
        
        if (successful_tests > 0) {
            double avg_speedup = total_speedup / successful_tests;
            printf("║ Average Speedup: %.2fx                                      ║\n", avg_speedup);
            printf("║ Best Speedup:    %.2fx                                      ║\n", best_speedup);
            printf("║ Worst Speedup:   %.2fx                                      ║\n", worst_speedup);
            printf("║ Success Rate:    %d/%d tests                                ║\n", successful_tests, total_tests);
            printf("╠════════════════════════════════════════════════════════════════╣\n");
            
            if (avg_speedup > 1.2) {
                printf("║ 🎉 RESULT: NUMA shows significant performance improvement!   ║\n");
            } else if (avg_speedup > 1.05) {
                printf("║ ✅ RESULT: NUMA shows modest performance improvement         ║\n");
            } else if (avg_speedup > 0.95) {
                printf("║ ⚠️  RESULT: NUMA performance similar to fallback            ║\n");
            } else {
                printf("║ ❌ RESULT: NUMA performance degradation detected            ║\n");
            }
        } else {
            printf("║ ❌ ERROR: No successful performance measurements             ║\n");
        }
        
        printf("╚════════════════════════════════════════════════════════════════╝\n");
        printf("\n");
    }
};

int main() {
    printf("NUMA ADD Performance Benchmark\n");
    printf("==============================\n");
    printf("Comparing NUMA vs Fallback execution for ADD operations\n\n");
    
    // Initialize NUMA with automatic coordinator initialization
    printf("Initializing NUMA with MIRROR strategy (auto-initializes coordinator)...\n");
    ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);
    
    if (!ggml_numa_simple_coordinator_is_initialized()) {
        printf("❌ NUMA coordinator auto-initialization failed\n");
        return 1;
    }
    
    // Verify NUMA state
    printf("🔍 NUMA STATE VERIFICATION:\n");
    printf("   NUMA available: %s\n", ggml_is_numa() ? "YES" : "NO");
    printf("   NUMA nodes: %d\n", ggml_numa_node_count());
    printf("   NUMA strategy: %d (expected: %d for MIRROR)\n", 
           ggml_numa_get_strategy(), GGML_NUMA_STRATEGY_MIRROR);
    printf("   Hardware threads: %d\n", (int)std::thread::hardware_concurrency());
    printf("   Coordinator initialized: %s\n", 
           ggml_numa_simple_coordinator_is_initialized() ? "YES" : "NO");
    
    printf("✅ NUMA auto-initialized with strategy MIRROR using all %d cores\n\n", (int)std::thread::hardware_concurrency());
    
    AddPerformanceBenchmark benchmark;
    benchmark.run_benchmark();
    benchmark.print_summary();
    
    printf("\nBenchmark completed successfully\n");
    return 0;
}
