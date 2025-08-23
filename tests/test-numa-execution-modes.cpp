#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-numa-coordinator.h"
#include "upi-traffic-monitor.h"
#include <chrono>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <functional>

#ifdef __linux__
#include <numa.h>
#include <sched.h>
#endif

// Test configuration
struct TestConfig {
    const char* name;
    int dim1, dim2, dim3;
    const char* description;
};

static const TestConfig test_configs[] = {
    {"SMALL",  128, 128, 16,   "Small tensor (4 MB)"},
    {"MEDIUM", 256, 256, 16,   "Medium tensor (16 MB)"},
    {"LARGE",  512, 256, 64,   "Large tensor (64 MB)"},
    {"HUGE",   512, 512, 256,  "Huge tensor (256 MB)"}
};

class NumaExecutionModesTest {
private:
    int total_physical_cores;
    UpiTrafficMonitor upi_monitor;
    
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
    
    // Mode 1: Fallback (node 0, all threads) - Implementation
    double test_fallback_node0_impl(const TestConfig& config) {
        printf("\n🔹 Mode 1: Fallback (NUMA node 0 only, %d threads)\n", total_physical_cores);
        
        // Bind process to NUMA node 0
        bind_to_numa_node_0();
        
        size_t tensor_size = (size_t)config.dim1 * config.dim2 * config.dim3 * sizeof(float);
        size_t ctx_size = tensor_size * 8 + 64*1024*1024;
        
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
        
        printf("      Using %d cores (fallback, CPU-bound to node 0)\n", total_physical_cores);
        
        // Use fallback computation (ggml_graph_plan + ggml_graph_compute)
        struct ggml_cplan cplan = ggml_graph_plan(cgraph, total_physical_cores, NULL);
        
        // Warmup
        for (int i = 0; i < 3; i++) {
            ggml_graph_compute(cgraph, &cplan);
        }
        
        const int num_runs = 10;
        std::vector<double> times;
        
        for (int run = 0; run < num_runs; run++) {
            auto start = std::chrono::high_resolution_clock::now();
            enum ggml_status status = ggml_graph_compute(cgraph, &cplan);
            auto end = std::chrono::high_resolution_clock::now();
            
            if (status != GGML_STATUS_SUCCESS) {
                printf("❌ Fallback computation failed\n");
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
    
    // Mode 2: NUMA coordination (node 0 only, all threads) - Implementation
    double test_numa_node0_only_impl(const TestConfig& config) {
        printf("\n🔹 Mode 2: NUMA Coordinator (NUMA node 0 only, %d threads)\n", total_physical_cores);
        
        // Clear CPU affinity but force NUMA coordinator to use single node
        clear_cpu_affinity();
        
        // Initialize NUMA
        ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);
        
        size_t tensor_size = (size_t)config.dim1 * config.dim2 * config.dim3 * sizeof(float);
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
        
        // Force single node by using half the cores (simulating node 0 only)
        int single_node_cores = total_physical_cores / 2;
        printf("      Using %d cores (NUMA coordinator, single node simulation)\n", single_node_cores);
        
        // Warmup
        for (int i = 0; i < 3; i++) {
            ggml_graph_compute_with_ctx(ctx, cgraph, single_node_cores);
        }
        
        const int num_runs = 10;
        std::vector<double> times;
        
        for (int run = 0; run < num_runs; run++) {
            auto start = std::chrono::high_resolution_clock::now();
            enum ggml_status status = ggml_graph_compute_with_ctx(ctx, cgraph, single_node_cores);
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
        
        // Clear CPU affinity to allow all nodes
        clear_cpu_affinity();
        
        // Initialize NUMA
        ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);
        
        size_t tensor_size = (size_t)config.dim1 * config.dim2 * config.dim3 * sizeof(float);
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
        
        printf("      Using %d cores (NUMA coordinator, data-parallel across all nodes)\n", total_physical_cores);
        
        // Warmup
        for (int i = 0; i < 3; i++) {
            ggml_graph_compute_with_ctx(ctx, cgraph, total_physical_cores);
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
    
    void run_comparison() {
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
                printf("  Mode 1 (Fallback Node 0):         %.3f ms\n", fallback_time);
                printf("  Mode 2 (NUMA Single Node):        %.3f ms (%.2fx vs fallback)\n", 
                       numa_node0_time, fallback_time / numa_node0_time);
                printf("  Mode 3 (NUMA All Nodes):          %.3f ms (%.2fx vs fallback)\n", 
                       numa_all_nodes_time, fallback_time / numa_all_nodes_time);
                printf("  NUMA Scaling (single→all nodes):  %.2fx\n", 
                       numa_node0_time / numa_all_nodes_time);
            } else {
                printf("❌ Test failed for %s\n", config.name);
            }
        }
        
        printf("\n╔════════════════════════════════════════════════════════════════╗\n");
        printf("║                    EXECUTION MODES COMPARISON                 ║\n");
        printf("╚════════════════════════════════════════════════════════════════╝\n");
        printf("Mode 1: Fallback bound to node 0 (28 cores, no NUMA coordinator)\n");
        printf("Mode 2: NUMA coordinator single node (28 cores, NUMA optimized)\n");
        printf("Mode 3: NUMA coordinator all nodes (56 cores, data-parallel)\n");
    }
};

int main() {
    NumaExecutionModesTest test;
    test.run_comparison();
    return 0;
}
