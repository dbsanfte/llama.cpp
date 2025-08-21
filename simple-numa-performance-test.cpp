/**
 * Simple NUMA Performance Test
 * 
 * This test measures the performance difference between NUMA and fallback execution
 * for the ADD operation across different tensor sizes.
 */

#include <cstdio>
#include <chrono>
#include <vector>
#include <cstring>

#include "ggml.h"
#include "ggml-cpu.h"

// Force execution control functions
extern void ggml_numa_set_dispatch_enabled(bool enabled);
extern bool ggml_numa_get_dispatch_enabled(void);

struct PerformanceResult {
    const char* test_name;
    int64_t elements;
    double numa_time_ms;
    double fallback_time_ms;
    double speedup_ratio;
};

class SimpleNumaPerformanceTest {
private:
    std::vector<PerformanceResult> results;
    
    // High-resolution timing
    using high_res_clock = std::chrono::high_resolution_clock;
    using time_point = std::chrono::time_point<high_res_clock>;
    
    time_point get_time() const {
        return high_res_clock::now();
    }
    
    double get_duration_ms(const time_point& start, const time_point& end) const {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return static_cast<double>(duration.count()) / 1000.0; // Convert to milliseconds
    }
    
    double run_add_test(int64_t elements, bool use_numa, int iterations = 5) {
        // Initialize GGML
        struct ggml_init_params params = {
            .mem_size   = 16 * 1024 * 1024,  // 16 MB
            .mem_buffer = NULL,
            .no_alloc   = false,
        };
        
        struct ggml_context* ctx = ggml_init(params);
        if (!ctx) {
            printf("❌ Failed to initialize GGML context\n");
            return -1.0;
        }
        
        // Create tensors based on element count
        // For simplicity, create 1D tensors
        struct ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, elements);
        struct ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, elements);
        struct ggml_tensor* result = ggml_add(ctx, a, b);
        
        // Initialize data
        float* a_data = (float*)a->data;
        float* b_data = (float*)b->data;
        for (int64_t i = 0; i < elements; i++) {
            a_data[i] = (float)(i % 100) / 100.0f;
            b_data[i] = (float)((i + 1) % 100) / 100.0f;
        }
        
        // Set execution mode
        ggml_numa_set_dispatch_enabled(use_numa);
        
        // Warmup runs
        for (int i = 0; i < 2; i++) {
            ggml_graph_compute_with_ctx(ctx, ggml_new_graph(ctx), 1);
        }
        
        // Timed runs
        std::vector<double> times;
        for (int iter = 0; iter < iterations; iter++) {
            auto start = get_time();
            ggml_graph_compute_with_ctx(ctx, ggml_new_graph(ctx), 1);
            auto end = get_time();
            times.push_back(get_duration_ms(start, end));
        }
        
        // Calculate average time
        double total_time = 0.0;
        for (double time : times) {
            total_time += time;
        }
        double avg_time = total_time / times.size();
        
        ggml_free(ctx);
        return avg_time;
    }
    
public:
    void run_performance_suite() {
        printf("=== Simple NUMA Performance Test ===\n");
        printf("Testing ADD operation with different tensor sizes\n\n");
        
        // Initialize NUMA
        ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);
        
        // Test different sizes
        std::vector<std::pair<const char*, int64_t>> test_cases = {
            {"TINY", 1024},           // 1K elements
            {"SMALL", 16384},         // 16K elements  
            {"MEDIUM", 262144},       // 256K elements
            {"LARGE", 4194304},       // 4M elements
            {"HUGE", 16777216}        // 16M elements
        };
        
        printf("%-10s %12s %15s %15s %10s\n", "Size", "Elements", "NUMA (ms)", "Fallback (ms)", "Speedup");
        printf("%-10s %12s %15s %15s %10s\n", "----", "--------", "--------", "----------", "-------");
        
        for (auto& test_case : test_cases) {
            const char* size_name = test_case.first;
            int64_t elements = test_case.second;
            
            // Test NUMA performance
            double numa_time = run_add_test(elements, true);
            
            // Test fallback performance  
            double fallback_time = run_add_test(elements, false);
            
            if (numa_time > 0 && fallback_time > 0) {
                double speedup = fallback_time / numa_time;
                
                printf("%-10s %12ld %15.3f %15.3f %9.2fx\n", 
                       size_name, elements, numa_time, fallback_time, speedup);
                       
                results.push_back({size_name, elements, numa_time, fallback_time, speedup});
            } else {
                printf("%-10s %12ld %15s %15s %10s\n", 
                       size_name, elements, "ERROR", "ERROR", "N/A");
            }
        }
        
        printf("\n=== Performance Summary ===\n");
        double total_speedup = 0.0;
        int valid_tests = 0;
        
        for (const auto& result : results) {
            total_speedup += result.speedup_ratio;
            valid_tests++;
        }
        
        if (valid_tests > 0) {
            double avg_speedup = total_speedup / valid_tests;
            printf("Average speedup: %.2fx\n", avg_speedup);
            
            if (avg_speedup > 1.1) {
                printf("🎉 NUMA improvements are providing significant performance benefits!\n");
            } else if (avg_speedup > 1.0) {
                printf("✅ NUMA improvements are providing modest performance benefits\n");
            } else {
                printf("⚠️  NUMA performance is similar to fallback - may need optimization\n");
            }
        }
        
        printf("\n");
    }
};

int main() {
    SimpleNumaPerformanceTest test;
    test.run_performance_suite();
    return 0;
}
