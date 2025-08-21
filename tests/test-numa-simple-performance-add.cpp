/**
 * Simple NUMA Performance Test for ADD Operation
 * 
 * This test measures the performance difference between NUMA and fallback execution
 * for the ADD operation across different tensor sizes, following the working
 * test framework pattern.
 */

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <chrono>

// GGML includes
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-numa-executor.h"
#include "ggml-numa-simple-coordinator.h"

// Performance result structure
struct PerformanceResult {
    std::string test_name;
    bool numa_enabled;
    double avg_time_ms;
    size_t tensor_elements;
    double throughput_gb_per_sec;
};

class SimpleNumaPerformanceTestSuite {
private:
    std::vector<PerformanceResult> results;
    
    // High-resolution timing utilities
    using high_res_clock = std::chrono::high_resolution_clock;
    using time_point = std::chrono::time_point<high_res_clock>;
    
    time_point get_time() const {
        return high_res_clock::now();
    }
    
    double get_duration_ms(const time_point& start, const time_point& end) const {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return static_cast<double>(duration.count()) / 1000.0; // Convert to milliseconds
    }
    
    // Test a single ADD case with performance measurement
    bool test_single_ADD_performance(int dim1, int dim2, int dim3, bool use_numa, const char* size_label) {
        printf("    ⏱️  Testing %s: ADD performance with dimensions [%d,%d,%d] (%s)\n", 
               size_label, dim1, dim2, dim3, use_numa ? "NUMA" : "Fallback");
        
        // Create test context with sufficient memory
        struct ggml_init_params params;
        params.mem_size = std::max((size_t)(512 * 1024 * 1024), (size_t)(dim1 * dim2 * dim3) * sizeof(float) * 8); // Scale memory with tensor size
        params.mem_buffer = nullptr;
        params.no_alloc = false;
        
        struct ggml_context* ctx = ggml_init(params);
        if (!ctx) {
            printf("❌ Failed to initialize GGML context\n");
            return false;
        }
        
        // Create tensors with the specified dimensions
        struct ggml_tensor* a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim1, dim2, dim3);
        struct ggml_tensor* b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim1, dim2, dim3);
        struct ggml_tensor* result = ggml_add(ctx, a, b);
        
        if (!a || !b || !result) {
            printf("❌ Failed to create tensors\n");
            ggml_free(ctx);
            return false;
        }
        
        // Initialize input data
        float* a_data = (float*)ggml_get_data(a);
        float* b_data = (float*)ggml_get_data(b);
        size_t total_elements = ggml_nelements(a);
        
        for (size_t i = 0; i < total_elements; i++) {
            a_data[i] = (float)(i % 100) / 100.0f;
            b_data[i] = (float)((i + 1) % 100) / 100.0f;
        }
        
        // Set execution mode
        ggml_numa_set_dispatch_enabled(use_numa);
        
        // Create computation graph
        struct ggml_cgraph* cgraph = ggml_new_graph(ctx);
        ggml_build_forward_expand(cgraph, result);
        
        // Warmup runs (2 iterations)
        for (int i = 0; i < 2; i++) {
            ggml_graph_compute_with_ctx(ctx, cgraph, 1);
        }
        
        // Performance measurement runs
        const int num_runs = 5;
        std::vector<double> times;
        
        for (int run = 0; run < num_runs; run++) {
            auto start = get_time();
            ggml_graph_compute_with_ctx(ctx, cgraph, 1);
            auto end = get_time();
            times.push_back(get_duration_ms(start, end));
        }
        
        // Calculate average time
        double total_time = 0.0;
        for (double time : times) {
            total_time += time;
        }
        double avg_time_ms = total_time / times.size();
        
        // Calculate throughput (3 arrays: read A, read B, write result = 3 * elements * 4 bytes)
        size_t bytes_processed = 3 * total_elements * sizeof(float);
        double throughput_gb_per_sec = (bytes_processed / (1024.0 * 1024.0 * 1024.0)) / (avg_time_ms / 1000.0);
        
        printf("      ⏱️  Average time: %.3f ms, Throughput: %.2f GB/s\n", avg_time_ms, throughput_gb_per_sec);
        
        // Store result
        PerformanceResult result_data;
        result_data.test_name = std::string(size_label);
        result_data.numa_enabled = use_numa;
        result_data.avg_time_ms = avg_time_ms;
        result_data.tensor_elements = total_elements;
        result_data.throughput_gb_per_sec = throughput_gb_per_sec;
        results.push_back(result_data);
        
        ggml_free(ctx);
        return true;
    }
    
public:
    void test_ADD_performance() {
        printf("=== NUMA Performance Test for ADD Operation ===\n");
        
        // Initialize NUMA with MIRROR strategy for optimal performance
        printf("🪞 Initializing NUMA with MIRROR strategy...\n");
        ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);
        
        // Test dimensions: increasing complexity
        struct {
            const char* name;
            int dim1, dim2, dim3;
        } test_cases[] = {
            {"TINY",   64,   8,  8},     // ~32K elements
            {"SMALL",  128,  16, 16},    // ~512K elements
            {"MEDIUM", 256,  32, 16},    // ~2M elements
            {"LARGE",  512,  64, 16},    // ~8M elements
            {"HUGE",   1024, 64, 16}     // ~16M elements
        };
        
        printf("📊 Testing ADD performance across complexity levels...\n\n");
        
        for (const auto& test_case : test_cases) {
            printf("  🔍 %s complexity [%d,%d,%d]:\n", test_case.name, test_case.dim1, test_case.dim2, test_case.dim3);
            
            // Test fallback performance
            test_single_ADD_performance(test_case.dim1, test_case.dim2, test_case.dim3, false, test_case.name);
            
            // Test NUMA performance
            test_single_ADD_performance(test_case.dim1, test_case.dim2, test_case.dim3, true, test_case.name);
            
            printf("\n");
        }
    }
    
    void print_performance_summary() {
        printf("=== Performance Summary ===\n");
        printf("%-10s %-12s %-12s %-15s %-10s\n", "Size", "Fallback(ms)", "NUMA(ms)", "Throughput(GB/s)", "Speedup");
        printf("%-10s %-12s %-12s %-15s %-10s\n", "----", "-----------", "-------", "--------------", "-------");
        
        // Group results by test case (fallback vs NUMA pairs)
        for (size_t i = 0; i < results.size(); i += 2) {
            if (i + 1 < results.size()) {
                auto& fallback = results[i];     // Fallback result
                auto& numa = results[i + 1];     // NUMA result
                
                double speedup = fallback.avg_time_ms / numa.avg_time_ms;
                
                printf("%-10s %-12.3f %-12.3f %-15.2f %-9.2fx\n", 
                       fallback.test_name.c_str(),
                       fallback.avg_time_ms,
                       numa.avg_time_ms,
                       numa.throughput_gb_per_sec,
                       speedup);
            }
        }
        
        // Calculate overall statistics
        double total_speedup = 0.0;
        int pairs = 0;
        
        for (size_t i = 0; i < results.size(); i += 2) {
            if (i + 1 < results.size()) {
                auto& fallback = results[i];
                auto& numa = results[i + 1];
                total_speedup += fallback.avg_time_ms / numa.avg_time_ms;
                pairs++;
            }
        }
        
        if (pairs > 0) {
            double avg_speedup = total_speedup / pairs;
            printf("\nAverage speedup: %.2fx\n", avg_speedup);
            
            if (avg_speedup > 1.2) {
                printf("🎉 NUMA shows significant performance improvement!\n");
            } else if (avg_speedup > 1.05) {
                printf("✅ NUMA shows modest performance improvement\n");
            } else {
                printf("⚠️  NUMA performance similar to fallback\n");
            }
        }
    }
    
    void run_all_tests() {
        printf("🚀 Starting NUMA Performance Tests...\n\n");
        
        test_ADD_performance();
        print_performance_summary();
        
        printf("\n🏁 Performance testing complete!\n");
    }
};

int main() {
    try {
        SimpleNumaPerformanceTestSuite test_suite;
        test_suite.run_all_tests();
        return 0;
    } catch (const std::exception& e) {
        printf("❌ Test failed with exception: %s\n", e.what());
        return 1;
    } catch (...) {
        printf("❌ Test failed with unknown exception\n");
        return 1;
    }
}
