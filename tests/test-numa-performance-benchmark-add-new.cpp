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

#include "ggml.h"
#include "ggml-cpu.h"

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
    
public:
    AddPerformanceBenchmark() {
        test_cases = {
            {64, 64, 32, "SMALL", "~128K elements (~512KB)"},
            {128, 128, 64, "MEDIUM", "~1M elements (~4MB)"},
            {256, 256, 128, "LARGE", "~8M elements (~32MB)"},
            {512, 512, 256, "HUGE", "~64M elements (~256MB)"}
        };
    }
    
    double measure_performance(const TestCase& test_case, bool use_numa) {
        printf("    Testing %s [%dx%dx%d] (%s mode)...\n", 
               test_case.name, test_case.dim1, test_case.dim2, test_case.dim3,
               use_numa ? "NUMA" : "Fallback");
        
        // Create context
        size_t tensor_size = (size_t)test_case.dim1 * test_case.dim2 * test_case.dim3 * sizeof(float);
        size_t ctx_size = tensor_size * 8 + 64*1024*1024; // Extra space
        
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };
        
        struct ggml_context* ctx = ggml_init(params);
        if (!ctx) {
            printf("Failed to create context\n");
            return -1.0;
        }
        
        // Create tensors
        struct ggml_tensor* tensor_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, test_case.dim1, test_case.dim2, test_case.dim3);
        struct ggml_tensor* tensor_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, test_case.dim1, test_case.dim2, test_case.dim3);
        struct ggml_tensor* result = ggml_add(ctx, tensor_a, tensor_b);
        
        if (!tensor_a || !tensor_b || !result) {
            printf("Failed to create tensors\n");
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
        
        // Set execution mode
        if (use_numa) {
            ggml_numa_set_dispatch_enabled(true);
        } else {
            ggml_numa_set_dispatch_enabled(false);
        }
        
        // Create computation graph
        struct ggml_cgraph* cgraph = ggml_new_graph(ctx);
        ggml_build_forward_expand(cgraph, result);
        
        // Warmup
        for (int i = 0; i < 3; i++) {
            ggml_graph_compute_with_ctx(ctx, cgraph, 8);
        }
        
        // Measure performance
        const int num_runs = 10;
        std::vector<double> times;
        
        for (int run = 0; run < num_runs; run++) {
            auto start = std::chrono::high_resolution_clock::now();
            enum ggml_status status = ggml_graph_compute_with_ctx(ctx, cgraph, 8);
            auto end = std::chrono::high_resolution_clock::now();
            
            if (status != GGML_STATUS_SUCCESS) {
                printf("Computation failed\n");
                ggml_free(ctx);
                return -1.0;
            }
            
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            if (time_ms > 0.001) { // Filter unrealistic times
                times.push_back(time_ms);
            }
        }
        
        ggml_free(ctx);
        
        if (times.empty()) {
            return -1.0;
        }
        
        // Calculate average time (remove outliers)
        std::sort(times.begin(), times.end());
        if (times.size() > 4) {
            times.erase(times.begin()); // Remove fastest
            times.pop_back(); // Remove slowest
        }
        
        double avg_time = 0.0;
        for (double time : times) {
            avg_time += time;
        }
        avg_time /= times.size();
        
        printf("      Average time: %.3f ms\n", avg_time);
        return avg_time;
    }
    
    void run_benchmark() {
        printf("🎯 ADD OPERATION PERFORMANCE ANALYSIS\n");
        printf("=====================================\n");
        
        // Initialize NUMA first
        printf("Initializing NUMA with MIRROR strategy...\n");
        ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);
        printf("NUMA nodes: %d, enabled: %s\n\n", 
               ggml_numa_node_count(), ggml_is_numa() ? "true" : "false");
        
        for (const auto& test_case : test_cases) {
            printf("📊 Testing %s: %s\n", test_case.name, test_case.description);
            
            double fallback_time = measure_performance(test_case, false);
            double numa_time = measure_performance(test_case, true);
            
            PerformanceResult result;
            result.test_name = test_case.name;
            result.fallback_time_ms = fallback_time;
            result.numa_time_ms = numa_time;
            result.success = (fallback_time > 0 && numa_time > 0);
            
            if (result.success) {
                result.speedup = fallback_time / numa_time;
                result.tensor_size_mb = (test_case.dim1 * test_case.dim2 * test_case.dim3 * sizeof(float)) / (1024 * 1024);
                
                printf("  Fallback: %.3f ms, NUMA: %.3f ms, Speedup=%.2fx\n", 
                       fallback_time, numa_time, result.speedup);
            } else {
                result.speedup = 0.0;
                result.tensor_size_mb = 0;
                printf("  Test failed\n");
            }
            
            results.push_back(result);
            printf("\n");
        }
    }
    
    void print_summary() {
        printf("=== Performance Summary ===\n");
        
        double total_speedup = 0.0;
        double best_speedup = 0.0;
        int successful_tests = 0;
        int total_tests = results.size();
        
        for (const auto& result : results) {
            if (result.success) {
                total_speedup += result.speedup;
                best_speedup = std::max(best_speedup, result.speedup);
                successful_tests++;
            }
        }
        
        if (successful_tests > 0) {
            double avg_speedup = total_speedup / successful_tests;
            printf("Average speedup: %.2fx\n", avg_speedup);
            printf("Best speedup: %.2fx\n", best_speedup);
            printf("Successful tests: %d/%d\n", successful_tests, total_tests);
            
            if (avg_speedup > 1.2) {
                printf("🎉 NUMA shows significant performance improvement!\n");
            } else if (avg_speedup > 1.05) {
                printf("✅ NUMA shows modest performance improvement\n");
            } else {
                printf("⚠️  NUMA performance similar to fallback\n");
            }
        } else {
            printf("❌ No successful performance measurements\n");
        }
    }
};

int main() {
    printf("NUMA ADD Performance Benchmark\n");
    printf("==============================\n");
    printf("Comparing NUMA vs Fallback execution for ADD operations\n\n");
    
    AddPerformanceBenchmark benchmark;
    benchmark.run_benchmark();
    benchmark.print_summary();
    
    printf("\nBenchmark completed successfully\n");
    return 0;
}
