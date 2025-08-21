/**
 * NUMA Performance Analysis Test
 * Focused performance measurement with detailed timing breakdown
 */

#include "../ggml/src/ggml-cpu/ggml-numa-perf.h"
#include "../ggml/src/ggml-cpu/ggml-numa-executor.h"
#include "../ggml/src/ggml-cpu/ggml-numa-simple-coordinator.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <chrono>
#include <vector>

static bool g_verbose = false;

static void run_performance_analysis() {
    printf("🚀 NUMA Performance Analysis\n");
    printf("==========================\n\n");
    
    // Initialize performance instrumentation
    ggml_numa_perf_init();  // Initialize with default settings
    ggml_numa_perf_set_enabled(true);  // Enable measurement
    ggml_numa_perf_set_detailed_logging(false);  // Reduce verbosity
    
    // Test different tensor sizes to analyze scaling
    struct {
        const char* name;
        int ne[4];
        size_t expected_bytes;
    } test_cases[] = {
        {"TINY (L1 cache)", {32, 32, 16, 1}, 64 * 1024},
        {"SMALL (L2 cache)", {128, 128, 16, 1}, 1024 * 1024},
        {"MEDIUM (L3 cache)", {512, 256, 16, 1}, 16 * 1024 * 1024},
        {"LARGE (Main memory)", {1024, 1024, 16, 1}, 64 * 1024 * 1024}
    };
    
    for (int i = 0; i < 4; i++) {
        printf("📊 Testing %s [%d,%d,%d] - %zu MB\n", 
               test_cases[i].name, 
               test_cases[i].ne[0], test_cases[i].ne[1], test_cases[i].ne[2],
               test_cases[i].expected_bytes / (1024 * 1024));
        
        // Create context and tensors
        struct ggml_init_params params;
        params.mem_size = test_cases[i].expected_bytes * 8;  // 8x for safety
        params.mem_buffer = NULL;
        params.no_alloc = false;
        struct ggml_context * ctx = ggml_init(params);
        
        // Create ADD operation
        struct ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 
                                                    test_cases[i].ne[0], test_cases[i].ne[1], 
                                                    test_cases[i].ne[2], test_cases[i].ne[3]);
        struct ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 
                                                    test_cases[i].ne[0], test_cases[i].ne[1], 
                                                    test_cases[i].ne[2], test_cases[i].ne[3]);
        struct ggml_tensor * result = ggml_add(ctx, a, b);
        
        // Build compute graph
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, result);
        
        // Test with multiple iterations
        const int iterations = 5;
        printf("   Running %d iterations...\n", iterations);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int iter = 0; iter < iterations; iter++) {
            ggml_numa_perf_reset();  // Reset for clean measurement
            
            // Execute the computation
            struct ggml_cplan plan = ggml_graph_plan(gf, 4, nullptr);  // 4 threads, no threadpool
            ggml_graph_compute(gf, &plan);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
        
        printf("   ⏱️  Total time: %.3f ms (%.3f ms/iteration)\n", 
               total_ms, total_ms / iterations);
        
        // Print performance breakdown
        printf("   📈 Performance Breakdown:\n");
        ggml_numa_perf_print_summary();
        printf("\n");
        
        ggml_free(ctx);
    }
    
    printf("✅ Performance analysis complete!\n");
}

int main() {
    printf("NUMA Performance Analysis Tool\n");
    printf("==============================\n\n");
    
    // Initialize NUMA
    ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);
    
    run_performance_analysis();
    
    return 0;
}
