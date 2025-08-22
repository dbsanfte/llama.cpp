/**
 * NUMA ADD Performance Diagnostic Tool
 * 
 * Systematically isolates and measures different components of NUMA execution
 * to identify performance bottlenecks in the ADD operation pipeline.
 */

#include <stdio.h>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#include "ggml.h"
#include "ggml-cpu.h"

class NumaAddDiagnostic {
public:
    void run_comprehensive_diagnostic() {
        printf("🔍 NUMA ADD Performance Diagnostic\n");
        printf("===================================\n\n");
        
        // Initialize NUMA
        printf("Initializing NUMA with MIRROR strategy...\n");
        ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);
        printf("NUMA nodes: %d, enabled: %s\n\n", 
               ggml_numa_node_count(), ggml_is_numa() ? "true" : "false");
        
        // Test different tensor sizes to isolate the issue
        test_tensor_size(128, 128, 64, "MEDIUM");    // 4MB
        test_tensor_size(256, 256, 128, "LARGE");    // 32MB  
        test_tensor_size(512, 512, 256, "HUGE");     // 256MB
    }
    
private:
    void test_tensor_size(int dim1, int dim2, int dim3, const char* size_name) {
        printf("🎯 Diagnostic for %s tensor [%dx%dx%d]\n", size_name, dim1, dim2, dim3);
        printf("=====================================\n");
        
        size_t tensor_size_mb = (dim1 * dim2 * dim3 * sizeof(float)) / (1024 * 1024);
        printf("Tensor size: %zu MB\n", tensor_size_mb);
        
        // Test 1: Pure fallback timing
        double fallback_time = measure_pure_fallback(dim1, dim2, dim3);
        printf("✅ Pure Fallback: %.3f ms\n", fallback_time);
        
        // Test 2: NUMA with minimal overhead 
        double numa_time = measure_numa_execution(dim1, dim2, dim3);
        printf("✅ NUMA Execution: %.3f ms\n", numa_time);
        
        // Test 3: Break down NUMA overhead
        analyze_numa_overhead(dim1, dim2, dim3);
        
        // Summary
        if (fallback_time > 0 && numa_time > 0) {
            double speedup = fallback_time / numa_time;
            printf("📊 Speedup: %.2fx", speedup);
            if (speedup > 1.2) {
                printf(" (GOOD)\n");
            } else if (speedup > 1.05) {
                printf(" (MARGINAL)\n");  
            } else {
                printf(" (POOR)\n");
            }
        }
        printf("\n");
    }
    
    double measure_pure_fallback(int dim1, int dim2, int dim3) {
        // Create context without NUMA overhead
        size_t tensor_size = (size_t)dim1 * dim2 * dim3 * sizeof(float);
        size_t ctx_size = tensor_size * 4 + 64*1024*1024;
        
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };
        
        struct ggml_context* ctx = ggml_init(params);
        if (!ctx) return -1.0;
        
        struct ggml_tensor* a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim1, dim2, dim3);
        struct ggml_tensor* b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim1, dim2, dim3);
        struct ggml_tensor* result = ggml_add(ctx, a, b);
        
        if (!a || !b || !result) {
            ggml_free(ctx);
            return -1.0;
        }
        
        // Initialize data
        float* a_data = (float*)ggml_get_data(a);
        float* b_data = (float*)ggml_get_data(b);
        size_t total_elements = ggml_nelements(a);
        
        for (size_t i = 0; i < total_elements; i++) {
            a_data[i] = 1.0f + (i % 100) * 0.01f;
            b_data[i] = 2.0f + (i % 100) * 0.01f;
        }
        
        struct ggml_cgraph* cgraph = ggml_new_graph(ctx);
        ggml_build_forward_expand(cgraph, result);
        struct ggml_cplan cplan = ggml_graph_plan(cgraph, 8, NULL);
        
        // Warmup
        for (int i = 0; i < 3; i++) {
            ggml_graph_compute(cgraph, &cplan);
        }
        
        // Measure
        const int num_runs = 5;
        std::vector<double> times;
        
        for (int run = 0; run < num_runs; run++) {
            auto start = std::chrono::high_resolution_clock::now();
            ggml_graph_compute(cgraph, &cplan);
            auto end = std::chrono::high_resolution_clock::now();
            
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            times.push_back(time_ms);
        }
        
        ggml_free(ctx);
        
        // Return median time
        std::sort(times.begin(), times.end());
        return times[times.size() / 2];
    }
    
    double measure_numa_execution(int dim1, int dim2, int dim3) {
        // Create NUMA context
        size_t tensor_size = (size_t)dim1 * dim2 * dim3 * sizeof(float);
        size_t ctx_size = tensor_size * 4 + 64*1024*1024;
        
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };
        
        struct ggml_context* ctx = ggml_init(params);
        if (!ctx) return -1.0;
        
        struct ggml_tensor* a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim1, dim2, dim3);
        struct ggml_tensor* b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim1, dim2, dim3);
        struct ggml_tensor* result = ggml_add(ctx, a, b);
        
        if (!a || !b || !result) {
            ggml_free(ctx);
            return -1.0;
        }
        
        // Initialize data
        float* a_data = (float*)ggml_get_data(a);
        float* b_data = (float*)ggml_get_data(b);
        size_t total_elements = ggml_nelements(a);
        
        for (size_t i = 0; i < total_elements; i++) {
            a_data[i] = 1.0f + (i % 100) * 0.01f;
            b_data[i] = 2.0f + (i % 100) * 0.01f;
        }
        
        ggml_numa_set_dispatch_enabled(true);
        
        struct ggml_cgraph* cgraph = ggml_new_graph(ctx);
        ggml_build_forward_expand(cgraph, result);
        
        // Warmup
        for (int i = 0; i < 3; i++) {
            ggml_graph_compute_with_ctx(ctx, cgraph, 8);
        }
        
        // Measure
        const int num_runs = 5;
        std::vector<double> times;
        
        for (int run = 0; run < num_runs; run++) {
            auto start = std::chrono::high_resolution_clock::now();
            ggml_graph_compute_with_ctx(ctx, cgraph, 8);
            auto end = std::chrono::high_resolution_clock::now();
            
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            times.push_back(time_ms);
        }
        
        ggml_free(ctx);
        
        // Return median time
        std::sort(times.begin(), times.end());
        return times[times.size() / 2];
    }
    
    void analyze_numa_overhead(int dim1, int dim2, int dim3) {
        printf("🔍 NUMA Overhead Analysis:\n");
        
        // Create a minimal test to see where time is spent
        size_t tensor_size = (size_t)dim1 * dim2 * dim3 * sizeof(float);
        size_t ctx_size = tensor_size * 4 + 64*1024*1024;
        
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };
        
        struct ggml_context* ctx = ggml_init(params);
        if (!ctx) return;
        
        struct ggml_tensor* a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim1, dim2, dim3);
        struct ggml_tensor* b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim1, dim2, dim3);
        struct ggml_tensor* result = ggml_add(ctx, a, b);
        
        if (!a || !b || !result) {
            ggml_free(ctx);
            return;
        }
        
        // Initialize data quickly
        float* a_data = (float*)ggml_get_data(a);
        float* b_data = (float*)ggml_get_data(b);
        size_t total_elements = ggml_nelements(a);
        
        for (size_t i = 0; i < total_elements; i++) {
            a_data[i] = 1.0f;
            b_data[i] = 2.0f;
        }
        
        ggml_numa_set_dispatch_enabled(true);
        
        struct ggml_cgraph* cgraph = ggml_new_graph(ctx);
        ggml_build_forward_expand(cgraph, result);
        
        // Warmup once
        ggml_graph_compute_with_ctx(ctx, cgraph, 8);
        
        // Single timing run with debug output enabled - let it show overhead
        printf("   Running single NUMA execution with debug output...\n");
        auto start = std::chrono::high_resolution_clock::now();
        ggml_graph_compute_with_ctx(ctx, cgraph, 8);
        auto end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double, std::milli>(end - start).count();
        
        printf("   Total NUMA execution time: %.3f ms\n", total_time);
        
        ggml_free(ctx);
    }
};

int main() {
    NumaAddDiagnostic diagnostic;
    diagnostic.run_comprehensive_diagnostic();
    
    printf("🏁 Diagnostic complete!\n");
    return 0;
}
