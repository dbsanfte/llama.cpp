#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-numa-simple-coordinator.h"
#include <chrono>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <thread>

// Test case structure
struct TestCase {
    int dim1, dim2, dim3;
    const char* name;
    const char* description;
    bool expect_numa_benefit;
};

class NumaModesComparison {
private:
    std::vector<TestCase> test_cases;
    
public:
    NumaModesComparison() {
        test_cases = {
            {64, 64, 32, "SMALL", "~128K elements (~512KB)", false},      // Cache-friendly
            {128, 128, 64, "MEDIUM", "~1M elements (~4MB)", false},       // L3 cache size
            {256, 256, 128, "LARGE", "~8M elements (~32MB)", true},       // Beyond cache
            {512, 512, 256, "HUGE", "~64M elements (~256MB)", true}       // Memory-bound
        };
    }
    
    // Test Mode 1: NUMA isolated to node 0
    double measure_numa_node0_performance(const TestCase& test_case) {
        printf("🧪 Mode 1: NUMA Node 0 Only\n");
        printf("    Testing %s [%dx%dx%d] (Single NUMA node 0)...\n", 
               test_case.name, test_case.dim1, test_case.dim2, test_case.dim3);
        
        // Force isolation to node 0
        ggml_numa_set_isolate_node(0);
        ggml_numa_set_dispatch_enabled(true);
        
        if (!ggml_numa_simple_coordinator_is_initialized()) {
            printf("❌ NUMA coordinator not initialized\n");
            return -1.0;
        }
        
        size_t tensor_size = (size_t)test_case.dim1 * test_case.dim2 * test_case.dim3 * sizeof(float);
        size_t ctx_size = tensor_size * 4 + 64*1024*1024;
        
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };
        
        struct ggml_context* ctx = ggml_init(params);
        if (!ctx) {
            printf("❌ Failed to create NUMA node 0 context\n");
            return -1.0;
        }
        
        // Create tensors using NUMA mirroring
        struct ggml_tensor* tensor_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, test_case.dim1, test_case.dim2, test_case.dim3);
        struct ggml_tensor* tensor_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, test_case.dim1, test_case.dim2, test_case.dim3);
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
        
        // Warmup
        for (int i = 0; i < 3; i++) {
            ggml_graph_compute_with_ctx(ctx, cgraph, std::thread::hardware_concurrency());
        }
        
        const int num_runs = 10;
        std::vector<double> times;
        
        for (int run = 0; run < num_runs; run++) {
            auto start = std::chrono::high_resolution_clock::now();
            
            enum ggml_status status = ggml_graph_compute_with_ctx(ctx, cgraph, std::thread::hardware_concurrency());
            
            auto end = std::chrono::high_resolution_clock::now();
            
            if (status != GGML_STATUS_SUCCESS) {
                printf("❌ NUMA node 0 computation failed\n");
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
        
        // Remove outliers and average
        std::sort(times.begin(), times.end());
        if (times.size() > 4) {
            times.erase(times.begin());
            times.pop_back();
        }
        
        double avg_time = 0.0;
        for (double t : times) avg_time += t;
        avg_time /= times.size();
        
        printf("      ✅ NUMA Node 0 average: %.3f ms\n", avg_time);
        return avg_time;
    }
    
    // Test Mode 2: NUMA isolated to node 1
    double measure_numa_node1_performance(const TestCase& test_case) {
        printf("🧪 Mode 2: NUMA Node 1 Only\n");
        printf("    Testing %s [%dx%dx%d] (Single NUMA node 1)...\n", 
               test_case.name, test_case.dim1, test_case.dim2, test_case.dim3);
        
        // Force isolation to node 1
        ggml_numa_set_isolate_node(1);
        ggml_numa_set_dispatch_enabled(true);
        
        if (!ggml_numa_simple_coordinator_is_initialized()) {
            printf("❌ NUMA coordinator not initialized\n");
            return -1.0;
        }
        
        size_t tensor_size = (size_t)test_case.dim1 * test_case.dim2 * test_case.dim3 * sizeof(float);
        size_t ctx_size = tensor_size * 4 + 64*1024*1024;
        
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };
        
        struct ggml_context* ctx = ggml_init(params);
        if (!ctx) {
            printf("❌ Failed to create NUMA node 1 context\n");
            return -1.0;
        }
        
        // Create tensors using NUMA mirroring
        struct ggml_tensor* tensor_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, test_case.dim1, test_case.dim2, test_case.dim3);
        struct ggml_tensor* tensor_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, test_case.dim1, test_case.dim2, test_case.dim3);
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
        
        // Warmup
        for (int i = 0; i < 3; i++) {
            ggml_graph_compute_with_ctx(ctx, cgraph, std::thread::hardware_concurrency());
        }
        
        const int num_runs = 10;
        std::vector<double> times;
        
        for (int run = 0; run < num_runs; run++) {
            auto start = std::chrono::high_resolution_clock::now();
            
            enum ggml_status status = ggml_graph_compute_with_ctx(ctx, cgraph, std::thread::hardware_concurrency());
            
            auto end = std::chrono::high_resolution_clock::now();
            
            if (status != GGML_STATUS_SUCCESS) {
                printf("❌ NUMA node 1 computation failed\n");
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
        
        // Remove outliers and average
        std::sort(times.begin(), times.end());
        if (times.size() > 4) {
            times.erase(times.begin());
            times.pop_back();
        }
        
        double avg_time = 0.0;
        for (double t : times) avg_time += t;
        avg_time /= times.size();
        
        printf("      ✅ NUMA Node 1 average: %.3f ms\n", avg_time);
        return avg_time;
    }
    
    // Test Mode 3: Data-parallel across both nodes
    double measure_numa_dataparallel_performance(const TestCase& test_case) {
        printf("🧪 Mode 3: Data-Parallel (Both Nodes)\n");
        printf("    Testing %s [%dx%dx%d] (Data-parallel across nodes 0+1)...\n", 
               test_case.name, test_case.dim1, test_case.dim2, test_case.dim3);
        
        // Remove isolation to allow both nodes
        ggml_numa_set_isolate_node(-1);
        ggml_numa_set_dispatch_enabled(true);
        
        if (!ggml_numa_simple_coordinator_is_initialized()) {
            printf("❌ NUMA coordinator not initialized\n");
            return -1.0;
        }
        
        size_t tensor_size = (size_t)test_case.dim1 * test_case.dim2 * test_case.dim3 * sizeof(float);
        size_t ctx_size = tensor_size * 4 + 64*1024*1024;
        
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };
        
        struct ggml_context* ctx = ggml_init(params);
        if (!ctx) {
            printf("❌ Failed to create data-parallel context\n");
            return -1.0;
        }
        
        // Create tensors using NUMA mirroring
        struct ggml_tensor* tensor_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, test_case.dim1, test_case.dim2, test_case.dim3);
        struct ggml_tensor* tensor_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, test_case.dim1, test_case.dim2, test_case.dim3);
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
        
        // Warmup
        for (int i = 0; i < 3; i++) {
            ggml_graph_compute_with_ctx(ctx, cgraph, std::thread::hardware_concurrency());
        }
        
        const int num_runs = 10;
        std::vector<double> times;
        
        for (int run = 0; run < num_runs; run++) {
            auto start = std::chrono::high_resolution_clock::now();
            
            enum ggml_status status = ggml_graph_compute_with_ctx(ctx, cgraph, std::thread::hardware_concurrency());
            
            auto end = std::chrono::high_resolution_clock::now();
            
            if (status != GGML_STATUS_SUCCESS) {
                printf("❌ Data-parallel computation failed\n");
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
        
        // Remove outliers and average
        std::sort(times.begin(), times.end());
        if (times.size() > 4) {
            times.erase(times.begin());
            times.pop_back();
        }
        
        double avg_time = 0.0;
        for (double t : times) avg_time += t;
        avg_time /= times.size();
        
        printf("      ✅ Data-parallel average: %.3f ms\n", avg_time);
        return avg_time;
    }
    
    void run_comparison() {
        printf("╔════════════════════════════════════════════════════════════════╗\n");
        printf("║                    NUMA MODES COMPARISON BENCHMARK             ║\n");
        printf("╠════════════════════════════════════════════════════════════════╣\n");
        printf("║ Comparing three NUMA execution modes:                        ║\n");
        printf("║ 1. NUMA Node 0 Only   - Single socket execution             ║\n");
        printf("║ 2. NUMA Node 1 Only   - Single socket execution             ║\n");
        printf("║ 3. Data-Parallel      - Dual socket execution               ║\n");
        printf("╚════════════════════════════════════════════════════════════════╝\n\n");
        
        // Initialize NUMA coordinator
        if (!ggml_numa_simple_coordinator_is_initialized()) {
            printf("Initializing NUMA coordinator...\n");
            
            // Use default threadpool parameters
            struct ggml_threadpool_params tpp = ggml_threadpool_params_default(std::thread::hardware_concurrency());
            
            if (!ggml_numa_simple_coordinator_init(&tpp)) {
                printf("❌ Failed to initialize NUMA coordinator\n");
                return;
            }
        }
        
        printf("NUMA Configuration:\n");
        printf("  Total NUMA nodes: %d\n", ggml_numa_node_count());
        printf("  Hardware threads: %d\n\n", (int)std::thread::hardware_concurrency());
        
        struct Result {
            double node0_time;
            double node1_time;
            double dataparallel_time;
            double best_single_node;
            double dataparallel_speedup;
        };
        
        std::vector<Result> results;
        
        for (const auto& test_case : test_cases) {
            printf("═══════════════════════════════════════════════════════════════\n");
            printf("Testing %s: %s\n", test_case.name, test_case.description);
            printf("Tensor size: ~%.1f MB\n", 
                   (double)(test_case.dim1 * test_case.dim2 * test_case.dim3 * sizeof(float)) / (1024.0 * 1024.0));
            printf("═══════════════════════════════════════════════════════════════\n");
            
            Result result = {};
            
            // Test Mode 1: NUMA Node 0 Only
            result.node0_time = measure_numa_node0_performance(test_case);
            printf("\n");
            
            // Test Mode 2: NUMA Node 1 Only
            result.node1_time = measure_numa_node1_performance(test_case);
            printf("\n");
            
            // Test Mode 3: Data-Parallel
            result.dataparallel_time = measure_numa_dataparallel_performance(test_case);
            printf("\n");
            
            if (result.node0_time > 0 && result.node1_time > 0 && result.dataparallel_time > 0) {
                result.best_single_node = std::min(result.node0_time, result.node1_time);
                result.dataparallel_speedup = result.best_single_node / result.dataparallel_time;
                
                printf("📊 COMPARISON RESULTS:\n");
                printf("    NUMA Node 0:      %.3f ms\n", result.node0_time);
                printf("    NUMA Node 1:      %.3f ms\n", result.node1_time);
                printf("    Data-Parallel:    %.3f ms\n", result.dataparallel_time);
                printf("    Best Single Node: %.3f ms\n", result.best_single_node);
                printf("    Speedup vs Best:  %.2fx %s\n", 
                       result.dataparallel_speedup,
                       result.dataparallel_speedup > 1.0 ? "✅" : "⚠️");
                
                if (test_case.expect_numa_benefit && result.dataparallel_speedup < 1.5) {
                    printf("    ⚠️  Expected significant NUMA benefit but got %.2fx\n", result.dataparallel_speedup);
                } else if (!test_case.expect_numa_benefit && result.dataparallel_speedup > 2.0) {
                    printf("    🎉 Unexpected NUMA benefit: %.2fx speedup!\n", result.dataparallel_speedup);
                } else {
                    printf("    ✅ Results match expectations\n");
                }
            } else {
                printf("❌ Some tests failed for %s\n", test_case.name);
            }
            
            results.push_back(result);
            printf("\n");
        }
        
        // Summary table
        printf("╔════════════════════════════════════════════════════════════════╗\n");
        printf("║                      NUMA MODES SUMMARY                       ║\n");
        printf("╠════════════════════════════════════════════════════════════════╣\n");
        printf("║ Case      │ Node 0   │ Node 1   │ Data-Par │ Speedup │ Status ║\n");
        printf("╠═══════════╪══════════╪══════════╪══════════╪═════════╪════════╣\n");
        
        for (size_t i = 0; i < test_cases.size() && i < results.size(); i++) {
            const auto& tc = test_cases[i];
            const auto& res = results[i];
            
            if (res.dataparallel_time > 0) {
                const char* status = "";
                if (tc.expect_numa_benefit) {
                    status = res.dataparallel_speedup >= 1.5 ? "  ✅  " : "  ⚠️  ";
                } else {
                    status = res.dataparallel_speedup < 2.0 ? "  ✅  " : "  🎉  ";
                }
                
                printf("║ %-9s │ %7.2f │ %7.2f │ %7.2f │  %5.2fx │%s║\n",
                       tc.name, res.node0_time, res.node1_time, res.dataparallel_time,
                       res.dataparallel_speedup, status);
            } else {
                printf("║ %-9s │   FAIL   │   FAIL   │   FAIL   │  FAIL  │  ❌  ║\n", tc.name);
            }
        }
        
        printf("╚════════════════════════════════════════════════════════════════╝\n");
        
        // Calculate overall statistics
        double total_speedup = 0.0;
        int valid_tests = 0;
        double best_speedup = 0.0;
        double worst_speedup = 100.0;
        
        for (const auto& res : results) {
            if (res.dataparallel_speedup > 0) {
                total_speedup += res.dataparallel_speedup;
                valid_tests++;
                best_speedup = std::max(best_speedup, res.dataparallel_speedup);
                worst_speedup = std::min(worst_speedup, res.dataparallel_speedup);
            }
        }
        
        if (valid_tests > 0) {
            printf("\n📈 OVERALL STATISTICS:\n");
            printf("    Average Speedup: %.2fx\n", total_speedup / valid_tests);
            printf("    Best Speedup:    %.2fx\n", best_speedup);
            printf("    Worst Speedup:   %.2fx\n", worst_speedup);
            printf("    Success Rate:    %d/%d tests\n", valid_tests, (int)test_cases.size());
            
            int beneficial_tests = 0;
            for (const auto& res : results) {
                if (res.dataparallel_speedup >= 1.1) beneficial_tests++;
            }
            printf("    Tests w/ Benefit: %d/%d\n", beneficial_tests, valid_tests);
            
            if (total_speedup / valid_tests >= 1.5) {
                printf("\n🎉 CONCLUSION: Data-parallel NUMA provides significant performance benefit\n");
            } else if (total_speedup / valid_tests >= 1.1) {
                printf("\n✅ CONCLUSION: Data-parallel NUMA provides moderate performance benefit\n");
            } else {
                printf("\n⚠️  CONCLUSION: Data-parallel NUMA overhead may exceed benefits for these operations\n");
            }
        }
        
        printf("\nNUMA modes comparison completed.\n");
    }
};

int main() {
    printf("🚀 Starting NUMA Modes Comparison Benchmark\n\n");
    
    NumaModesComparison benchmark;
    benchmark.run_comparison();
    
    return 0;
}
