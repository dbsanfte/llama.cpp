#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-numa-simple-coordinator.h"
#include "upi-traffic-monitor.h"
#include <vector>
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <functional>
#include <fstream>
#include <cstdlib>

#ifdef __linux__
#include <numa.h>
#include <sched.h>
#include <sys/wait.h>
#endif

// Test configuration
struct TestConfig {
    const char* name;
    int dim1, dim2, dim3;
    const char* description;
};

// Test result storage
struct TestResult {
    std::string config_name;
    std::string description;
    double size_gb;
    double mode1_time_ms;  // NUMA Isolate Node 0
    double mode2_time_ms;  // NUMA Isolate Node 1  
    double mode3_time_ms;  // NUMA Mirror Dual-Socket
    double mode2_vs_mode1_ratio;
    double mode3_vs_mode1_ratio;
    double dual_socket_scaling;
};

static const TestConfig test_configs[] = {
    {"SMALL",    128, 128, 16,    "Small tensor (4 MB)"},
    {"MEDIUM",   256, 256, 16,    "Medium tensor (16 MB)"},
    {"LARGE",    512, 256, 64,    "Large tensor (64 MB)"},
    {"HUGE",     1024, 1024, 256,   "Huge tensor (1 GB) - Should force data-parallel"},
    {"MASSIVE",  1024, 1024, 512,   "Massive tensor (2 GB) - 2x HUGE complexity"},
    {"EXTREME",  1024, 1024, 1024,  "Extreme tensor (4 GB) - 4x HUGE complexity"},
    {"ULTRA",    1024, 2048, 1024,  "Ultra tensor (8 GB) - Dual 6-channel bandwidth test"},
    {"INSANE",   2048, 2048, 1024,  "Insane tensor (16 GB) - Maximum safe tensor size"}
};

class NumaExecutionModesTest {
private:
    int total_physical_cores;
    UpiTrafficMonitor upi_monitor;
    std::vector<TestResult> test_results;  // Store results for scaling table
    
    // Clear page cache to ensure fair performance comparison
    void clear_page_cache() {
        printf("      🗑️  Clearing page cache...\n");
        
        // Sync to flush any pending writes
        sync();
        
        // Use sudo to clear page cache, dentries, and inodes
        int result = system("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'");
        if (result == 0) {
            printf("      ✅ Page cache cleared successfully\n");
        } else {
            printf("      ⚠️  Page cache clear failed (continuing anyway)\n");
        }
        
        // Small delay to ensure cache clearing takes effect
        usleep(100000); // 100ms
    }
    
    int get_physical_core_count() {
        int cores = sysconf(_SC_NPROCESSORS_ONLN);
        printf("Detected %d total cores\n", cores);
        return cores;
    }
    
    void bind_to_numa_node_0() {
#ifdef __linux__
        cpu_set_t mask;
        CPU_ZERO(&mask);
        
        // Intel Xeon Gold 6238R: cores 0-27 are on NUMA node 0
        int cores_per_node = total_physical_cores / 2;
        for (int i = 0; i < cores_per_node; i++) {
            CPU_SET(i, &mask);
        }
        
        if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {
            printf("⚠️ Failed to set CPU affinity to NUMA node 0\n");
        } else {
            printf("✅ CPU affinity set to NUMA node 0 (cores 0-%d)\n", cores_per_node - 1);
        }
#endif
    }
    
    void bind_to_numa_node_1() {
#ifdef __linux__
        cpu_set_t mask;
        CPU_ZERO(&mask);
        
        // Intel Xeon Gold 6238R: cores 28-55 are on NUMA node 1  
        int cores_per_node = total_physical_cores / 2;
        for (int i = cores_per_node; i < total_physical_cores; i++) {
            CPU_SET(i, &mask);
        }
        
        if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {
            printf("⚠️ Failed to set CPU affinity to NUMA node 1\n");
        } else {
            printf("✅ CPU affinity set to NUMA node 1 (cores %d-%d)\n", cores_per_node, total_physical_cores - 1);
        }
#endif
    }
    
    void clear_cpu_affinity() {
#ifdef __linux__
        cpu_set_t mask;
        CPU_ZERO(&mask);
        
        // Allow all cores
        for (int i = 0; i < total_physical_cores; i++) {
            CPU_SET(i, &mask);
        }
        
        if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {
            printf("⚠️ Failed to clear CPU affinity\n");
        } else {
            printf("✅ CPU affinity cleared (all cores available)\n");
        }
#endif
    }
    
public:
    NumaExecutionModesTest() {
        total_physical_cores = get_physical_core_count();
        
        // Initialize UPI traffic monitoring
        if (!upi_monitor.initialize()) {
            printf("⚠️  UPI traffic monitoring not available (proceeding without UPI validation)\n");
        } else {
            printf("✅ UPI traffic monitoring initialized\n");
        }
    }
    
    // Helper function to run a test with UPI traffic monitoring
    double run_test_with_upi_monitoring(const TestConfig& /* config */, const std::string& mode_name,
                                       std::function<double()> test_function) {
        printf("\n🔍 Starting UPI monitoring for %s\n", mode_name.c_str());
        
        // Clear page cache before each test for fair comparison
        clear_page_cache();
        
        // Take baseline UPI snapshot
        auto upi_before = upi_monitor.take_snapshot();
        
        // Run the actual test
        double execution_time = test_function();
        
        // Take post-execution UPI snapshot
        auto upi_after = upi_monitor.take_snapshot();
        
        // Validate UPI traffic (5% threshold)
        bool upi_valid = upi_monitor.validate_numa_optimization(upi_before, upi_after, 5.0);
        
        // Print UPI report
        std::string upi_report = upi_monitor.get_traffic_report(upi_before, upi_after);
        printf("%s", upi_report.c_str());
        
        if (!upi_valid) {
            printf("❌ UPI VALIDATION FAILED: Cross-node traffic exceeded 5%% threshold!\n");
            printf("   This indicates NUMA optimizations are not effective.\n");
            return -1.0;  // Fail the test
        } else {
            printf("✅ UPI validation passed: Cross-node traffic within acceptable limits\n");
        }
        
        return execution_time;
    }
    
    // Mode 1: NUMA Isolate (node 0 only) - Implementation
    double test_fallback_node0_impl(const TestConfig& config) {
        printf("\n🔹 Mode 1: NUMA Isolate (NUMA node 0 only, %d threads)\n", total_physical_cores / 2);
        printf("      📍 Using NUMA coordinator with ISOLATE strategy on node 0\n");
        
        // Clear CPU affinity first
        clear_cpu_affinity();
        
        // Bind to node 0 BEFORE NUMA initialization for ISOLATE mode
        bind_to_numa_node_0();
        
        // Initialize NUMA with ISOLATE strategy specifying node 0 explicitly
        ggml_numa_init_with_node(GGML_NUMA_STRATEGY_ISOLATE, 0);
        printf("      ✅ NUMA coordinator initialized with ISOLATE strategy for node 0\n");
        
        // Force single-node execution by setting environment
        setenv("GGML_NUMA_NODES", "1", 1);
        printf("      🔧 NUMA coordinator: ISOLATE mode, node 0 only\n");
        
        size_t tensor_size = (size_t)config.dim1 * config.dim2 * config.dim3 * sizeof(float);
        // Use reasonable context size: 3x tensor size for the 3 tensors + 1GB overhead
        size_t ctx_size = tensor_size * 3 + 1024*1024*1024;
        
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
        
        struct ggml_tensor* tensor_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, config.dim1, config.dim2, config.dim3);
        struct ggml_tensor* tensor_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, config.dim1, config.dim2, config.dim3);
        struct ggml_tensor* result = ggml_add(ctx, tensor_a, tensor_b);
        
        if (!tensor_a || !tensor_b || !result) {
            printf("❌ Failed to create tensors\n");
            ggml_free(ctx);
            return -1.0;
        }
        
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
        
        printf("      Using %d cores (NUMA ISOLATE node 0 only)\n", total_physical_cores / 2);
        
        // Use NUMA coordinator execution instead of fallback
        printf("      🔧 DEBUG: Using NUMA coordinator execution (ISOLATE mode)\n");
        
        // Warmup
        for (int i = 0; i < 3; i++) {
            enum ggml_status status = ggml_graph_compute_with_ctx(ctx, cgraph, total_physical_cores / 2);
            if (status != GGML_STATUS_SUCCESS) {
                printf("❌ NUMA warmup failed\n");
                ggml_free(ctx);
                return -1.0;
            }
        }
        
        const int num_runs = 10;
        std::vector<double> times;
        
        printf("      🔧 DEBUG: About to run %d iterations using NUMA coordinator (ISOLATE MODE)\n", num_runs);
        
        for (int run = 0; run < num_runs; run++) {
            auto start = std::chrono::high_resolution_clock::now();
            enum ggml_status status = ggml_graph_compute_with_ctx(ctx, cgraph, total_physical_cores / 2);
            auto end = std::chrono::high_resolution_clock::now();
            
            if (status != GGML_STATUS_SUCCESS) {
                printf("❌ NUMA computation failed\n");
                ggml_free(ctx);
                return -1.0;
            }
            
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            printf("      🔧 DEBUG: NUMA ISOLATE run %d: %.3f ms\n", run+1, time_ms);
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
    
    // Mode 2: NUMA Isolate (node 1 only) - Implementation
    double test_numa_node0_only_impl(const TestConfig& config) {
        printf("\n🔹 Mode 2: NUMA Isolate (NUMA node 1 only, %d threads)\n", total_physical_cores / 2);
        printf("      📍 Using NUMA coordinator with ISOLATE strategy on node 1\n");
        
        // Clear CPU affinity first
        clear_cpu_affinity();
        
        // Bind to node 1 BEFORE NUMA initialization for ISOLATE mode
        bind_to_numa_node_1();
        
        // Initialize NUMA with ISOLATE strategy specifying node 1 explicitly
        ggml_numa_init_with_node(GGML_NUMA_STRATEGY_ISOLATE, 1);
        printf("      ✅ NUMA coordinator initialized with ISOLATE strategy for node 1\n");
        
        // Force single-node execution by setting environment
        setenv("GGML_NUMA_NODES", "1", 1);
        printf("      🔧 NUMA coordinator: ISOLATE mode, node 1 only\n");
        
        size_t tensor_size = (size_t)config.dim1 * config.dim2 * config.dim3 * sizeof(float);
        // Use reasonable context size: 3x tensor size for the 3 tensors + 1GB overhead
        size_t ctx_size = tensor_size * 3 + 1024*1024*1024;
        
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
        
        struct ggml_tensor* tensor_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, config.dim1, config.dim2, config.dim3);
        struct ggml_tensor* tensor_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, config.dim1, config.dim2, config.dim3);
        struct ggml_tensor* result = ggml_add(ctx, tensor_a, tensor_b);
        
        if (!tensor_a || !tensor_b || !result) {
            printf("❌ Failed to create tensors\n");
            ggml_free(ctx);
            return -1.0;
        }
        
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
        
        // Use NUMA coordinator with single node (node 1)
        printf("      🔧 Using NUMA coordinator with single-node execution on node 1 (%d cores)\n", total_physical_cores / 2);
        printf("      🔧 DEBUG: NUMA Mode 2 - will call ggml_graph_compute_with_ctx() on node 1\n");
        
        // Warmup
        for (int i = 0; i < 3; i++) {
            enum ggml_status warmup_status = ggml_graph_compute_with_ctx(ctx, cgraph, total_physical_cores / 2);
            printf("      🔧 DEBUG: Warmup %d status: %d\n", i+1, warmup_status);
        }
        
        const int num_runs = 10;
        std::vector<double> times;
        
        for (int run = 0; run < num_runs; run++) {
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
    
    // Mode 3: NUMA coordination (all nodes, all threads) - Implementation
    double test_numa_all_nodes_impl(const TestConfig& config) {
        printf("\n🔹 Mode 3: NUMA Coordinator (all nodes, %d threads)\n", total_physical_cores);
        printf("      📍 Using NUMA coordinator with multi-node data-parallel strategy\n");
        
        // Clear CPU affinity to allow all nodes
        clear_cpu_affinity();
        
        // Initialize NUMA with mirror strategy for multi-node execution
        ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);
        printf("      ✅ NUMA coordinator initialized with MIRROR strategy (multi-node)\n");
        
        size_t tensor_size = (size_t)config.dim1 * config.dim2 * config.dim3 * sizeof(float);
        // Use reasonable context size: 3x tensor size for the 3 tensors + 1GB overhead
        size_t ctx_size = tensor_size * 3 + 1024*1024*1024;
        
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
        
        struct ggml_tensor* tensor_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, config.dim1, config.dim2, config.dim3);
        struct ggml_tensor* tensor_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, config.dim1, config.dim2, config.dim3);
        struct ggml_tensor* result = ggml_add(ctx, tensor_a, tensor_b);
        
        if (!tensor_a || !tensor_b || !result) {
            printf("❌ Failed to create tensors\n");
            ggml_free(ctx);
            return -1.0;
        }
        
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
        
        printf("      🔧 Using NUMA coordinator with data-parallel execution (%d cores)\n", total_physical_cores);
        printf("      🔧 DEBUG: NUMA Mode 3 - will call ggml_graph_compute_with_ctx() with all nodes\n");
        
        // Warmup
        for (int i = 0; i < 3; i++) {
            enum ggml_status warmup_status = ggml_graph_compute_with_ctx(ctx, cgraph, total_physical_cores);
            printf("      🔧 DEBUG: Warmup %d status: %d\n", i+1, warmup_status);
        }
        
        const int num_runs = 10;
        std::vector<double> times;
        
        for (int run = 0; run < num_runs; run++) {
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
    
    void print_scaling_analysis_table() {
        if (test_results.empty()) {
            printf("\n⚠️  No test results available for scaling analysis\n");
            return;
        }
        
        printf("\n╔════════════════════════════════════════════════════════════════╗\n");
        printf("║                     NUMA SCALING ANALYSIS TABLE               ║\n");
        printf("╚════════════════════════════════════════════════════════════════╝\n");
        
        // Table header
        printf("┌─────────────┬─────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐\n");
        printf("│ Config      │ Size    │   Mode 1    │   Mode 2    │   Mode 3    │ Node Ratio  │ Dual-Socket │\n");
        printf("│             │ (GB)    │ Node 0 (ms) │ Node 1 (ms) │ Mirror (ms) │ (2 vs 1)    │ Scaling     │\n");
        printf("├─────────────┼─────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┤\n");
        
        // Table rows
        for (const auto& result : test_results) {
            printf("│ %-11s │ %7.3f │ %11.1f │ %11.1f │ %11.1f │ %11.2fx │ %11.2fx │\n",
                   result.config_name.c_str(),
                   result.size_gb,
                   result.mode1_time_ms,
                   result.mode2_time_ms,
                   result.mode3_time_ms,
                   result.mode2_vs_mode1_ratio,
                   result.dual_socket_scaling);
        }
        
        printf("└─────────────┴─────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘\n");
        
        // Analysis summary
        printf("\n📊 SCALING ANALYSIS INSIGHTS:\n");
        
        // Find best and worst scaling
        double best_dual_scaling = 0.0;
        double worst_dual_scaling = 10.0;
        std::string best_config, worst_config;
        
        for (const auto& result : test_results) {
            if (result.dual_socket_scaling > best_dual_scaling) {
                best_dual_scaling = result.dual_socket_scaling;
                best_config = result.config_name;
            }
            if (result.dual_socket_scaling < worst_dual_scaling) {
                worst_dual_scaling = result.dual_socket_scaling;
                worst_config = result.config_name;
            }
        }
        
        printf("• Best Dual-Socket Scaling: %.2fx (%s configuration)\n", best_dual_scaling, best_config.c_str());
        printf("• Worst Dual-Socket Scaling: %.2fx (%s configuration)\n", worst_dual_scaling, worst_config.c_str());
        
        // Memory bandwidth utilization analysis
        if (!test_results.empty()) {
            const auto& largest = test_results.back();  // Assume sorted by size
            double theoretical_bandwidth_gbps = 9.2;  // Dual-socket theoretical peak
            double data_transfer_gb = largest.size_gb * 3;  // Read 2 tensors, write 1
            double effective_bandwidth = (data_transfer_gb * 1000.0) / largest.mode3_time_ms;
            double utilization_percent = (effective_bandwidth / theoretical_bandwidth_gbps) * 100.0;
            
            printf("• Memory Bandwidth Utilization: %.1f GB/s (%.1f%% of theoretical 9.2 GB/s)\n", 
                   effective_bandwidth, utilization_percent);
        }
        
        // Scaling trend analysis  
        if (test_results.size() >= 2) {
            double scaling_improvement = test_results.back().dual_socket_scaling / test_results.front().dual_socket_scaling;
            printf("• Scaling Trend: %.2fx improvement from smallest to largest tensor\n", scaling_improvement);
            
            if (scaling_improvement > 1.2) {
                printf("  ✅ Excellent: Dual-socket benefits increase with tensor size\n");
            } else if (scaling_improvement > 1.0) {
                printf("  ✅ Good: Consistent dual-socket benefits across sizes\n");
            } else {
                printf("  ⚠️  Warning: Dual-socket benefits decrease with larger tensors\n");
            }
        }
    }
    
    void run_comparison() {
        // Enable NUMA debug logging
        setenv("GGML_DEBUG", "1", 1);
        setenv("NUMA_DEBUG", "1", 1);
        
        printf("🎯 NUMA EXECUTION MODES COMPARISON\n");
        printf("=====================================\n");
        printf("System: Intel Xeon Gold 6238R (%d cores, 2 NUMA nodes)\n\n", total_physical_cores);
        
        for (const auto& config : test_configs) {
            printf("\n📊 Testing %s: %s [%dx%dx%d]\n", 
                   config.name, config.description, config.dim1, config.dim2, config.dim3);
            printf("================================================\n");
            
            // Run tests with UPI monitoring
            double fallback_time = run_test_with_upi_monitoring(config, "Fallback Mode (Node 0)", 
                [this, &config]() { return test_fallback_node0_impl(config); });
                
            double numa_node0_time = run_test_with_upi_monitoring(config, "NUMA Single Node Mode", 
                [this, &config]() { return test_numa_node0_only_impl(config); });
                
            double numa_all_nodes_time = run_test_with_upi_monitoring(config, "NUMA All Nodes Mode", 
                [this, &config]() { return test_numa_all_nodes_impl(config); });
            
            if (fallback_time > 0 && numa_node0_time > 0 && numa_all_nodes_time > 0) {
                printf("\n📈 PERFORMANCE SUMMARY FOR %s:\n", config.name);
                printf("  Mode 1 (NUMA Isolate Node 0):     %.3f ms\n", fallback_time);
                printf("  Mode 2 (NUMA Isolate Node 1):     %.3f ms (%.2fx vs node 0)\n", 
                       numa_node0_time, fallback_time / numa_node0_time);
                printf("  Mode 3 (NUMA Mirror Dual-Socket): %.3f ms (%.2fx vs node 0)\n", 
                       numa_all_nodes_time, fallback_time / numa_all_nodes_time);
                printf("  Node Isolation Performance:       %.2fx (node 1 vs node 0)\n", 
                       fallback_time / numa_node0_time);
                printf("  Dual-Socket Scaling:              %.2fx (mirror vs single node)\n",
                       numa_node0_time / numa_all_nodes_time);
                
                // Store results for scaling analysis table
                TestResult result;
                result.config_name = config.name;
                result.description = config.description;
                result.size_gb = ((double)config.dim1 * config.dim2 * config.dim3 * sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
                result.mode1_time_ms = fallback_time;
                result.mode2_time_ms = numa_node0_time;
                result.mode3_time_ms = numa_all_nodes_time;
                result.mode2_vs_mode1_ratio = fallback_time / numa_node0_time;
                result.mode3_vs_mode1_ratio = fallback_time / numa_all_nodes_time;
                result.dual_socket_scaling = numa_node0_time / numa_all_nodes_time;
                test_results.push_back(result);
                
                // Critical diagnostic warnings
                if (abs(fallback_time - numa_node0_time) < 1.0) {
                    printf("⚠️  WARNING: Node 0 vs Node 1 performance identical - check ISOLATE implementation\n");
                }
                if (numa_all_nodes_time > numa_node0_time * 1.5) {
                    printf("🚨 CRITICAL: Mirror mode slower than single node - coordination overhead detected\n");
                }
                if (numa_all_nodes_time > fallback_time * 1.5) {
                    printf("🚨 CRITICAL: Mirror mode slower than node 0 - memory locality issue detected\n");
                }
            } else {
                printf("❌ Test failed for %s (Mode 2 may have crashed during node 1 initialization)\n", config.name);
            }
        }
        
        printf("\n╔════════════════════════════════════════════════════════════════╗\n");
        printf("║                    EXECUTION MODES COMPARISON                 ║\n");
        printf("╚════════════════════════════════════════════════════════════════╝\n");
        printf("Mode 1: NUMA Isolate node 0 (28 cores, NUMA optimized single socket)\n");
        printf("Mode 2: NUMA Isolate node 1 (28 cores, NUMA optimized single socket)\n");
        printf("Mode 3: NUMA Mirror data-parallel (56 cores, dual-socket coordination)\n");
        
        // Print comprehensive scaling analysis table
        print_scaling_analysis_table();
        
        printf("\n╔════════════════════════════════════════════════════════════════╗\n");
        printf("║                    DIAGNOSTIC EXPECTATIONS                    ║\n");
        printf("╚════════════════════════════════════════════════════════════════╝\n");
        printf("Expected Performance (Intel Xeon Gold 6238R, 12 DDR4-2933 channels):\n");
        printf("• Node 0 vs Node 1: ~1.0x (similar single-socket performance)\n");
        printf("• Mirror vs Single: ~2.0x faster (dual 6-channel bandwidth)\n");
        printf("• Memory Bandwidth: ~4.6 GB/s per socket, ~9.2 GB/s total\n");
        printf("\nIf Mode 2 missing: Node 1 isolation failed during initialization\n");
        printf("If Mirror slower: Memory locality bug - data not using both sockets\n");
    }
};

int main() {
    NumaExecutionModesTest test;
    test.run_comparison();
    return 0;
}
