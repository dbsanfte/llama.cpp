#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "common.h"

#include <chrono>
#include <vector>
#include <iostream>
#include <thread>
#include <sched.h>

using namespace std::chrono;

// Simple baseline test focusing on the concept
struct SimpleResult {
    std::string test_name;
    int cpu_id;
    double avg_time_ms;
    double gflops;
    bool success;
};

bool pin_to_cpu(int cpu_id) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    return sched_setaffinity(0, sizeof(cpuset), &cpuset) == 0;
#else
    return true; // Skip pinning on non-Linux
#endif
}

SimpleResult run_simple_benchmark(int cpu_id, int size = 512, int iterations = 5) {
    SimpleResult result = {};
    result.cpu_id = cpu_id;
    result.test_name = "Matrix " + std::to_string(size) + "x" + std::to_string(size);
    result.success = false;
    
    printf("🧪 Testing %s on CPU %d\n", result.test_name.c_str(), cpu_id);
    
    if (!pin_to_cpu(cpu_id)) {
        printf("⚠️  Failed to pin to CPU %d\n", cpu_id);
    }
    
    // Simple direct memory allocation approach
    size_t elements = (size_t)size * size;
    size_t bytes = elements * sizeof(float);
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ 3 * bytes + 1024*1024, // Space for 3 matrices + overhead
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        printf("❌ Failed to create context\n");
        return result;
    }
    
    // Create simple square matrices
    struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, size, size);
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, size, size);
    struct ggml_tensor* c = ggml_mul_mat(ctx, a, b);
    
    // Initialize with simple pattern (proven working approach)
    float* a_data = (float*)ggml_get_data(a);
    float* b_data = (float*)ggml_get_data(b);
    
    // Use exact same pattern as working test-basic-mulmat.cpp
    for (size_t i = 0; i < elements; i++) {
        a_data[i] = 1.0f; // All 1s for simple math
    }
    for (size_t i = 0; i < elements; i++) {
        b_data[i] = 2.0f; // All 2s for simple math  
    }
    
    // Create graph (proven pattern)
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);
    
    // Use direct ggml computation instead of backend (proven pattern)
    struct ggml_cplan cplan = ggml_graph_plan(graph, 1, NULL); // 1 thread, no threadpool
    
    // Test if computation works at all
    enum ggml_status status = ggml_graph_compute(graph, &cplan);
    if (status != GGML_STATUS_SUCCESS) {
        printf("❌ Basic computation failed with status: %d\n", status);
        ggml_free(ctx);
        return result;
    }
    
    printf("✅ Computation works, running benchmark...\n");
    
    std::vector<double> times;
    for (int i = 0; i < iterations; i++) {
        auto start = high_resolution_clock::now();
        ggml_graph_compute(graph, &cplan);
        auto end = high_resolution_clock::now();
        
        double time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        times.push_back(time_ms);
    }
    
    ggml_free(ctx);
    
    if (!times.empty()) {
        result.success = true;
        result.avg_time_ms = 0.0;
        for (double t : times) result.avg_time_ms += t;
        result.avg_time_ms /= times.size();
        
        // Calculate GFLOPS (approximate)
        double ops = 2.0 * size * size * size; // Matrix multiply ops
        result.gflops = (ops / (result.avg_time_ms / 1000.0)) / 1e9;
        
        printf("   ✅ Avg: %.2f ms, %.2f GFLOPS\n", result.avg_time_ms, result.gflops);
    }
    
    return result;
}

int main() {
    printf("Simple Single-Core Baseline Test\n");
    printf("=================================\n\n");
    
    // Test on just a few CPUs to establish baseline
    std::vector<int> test_cpus = {0, 1, 11, 12}; // Mix of different CPUs
    std::vector<SimpleResult> results;
    
    for (int cpu : test_cpus) {
        SimpleResult result = run_simple_benchmark(cpu, 512, 3);
        results.push_back(result);
        printf("\n");
    }
    
    // Summary
    printf("BASELINE SUMMARY:\n");
    printf("================\n");
    printf("CPU    Test           Avg(ms)  GFLOPS   Status\n");
    printf("-----  -------------- -------  -------  ------\n");
    
    double best_gflops = 0.0;
    for (const auto& r : results) {
        printf("%-3d    %-14s %7.2f  %7.2f  %s\n",
               r.cpu_id, r.test_name.c_str(), r.avg_time_ms, r.gflops,
               r.success ? "✅" : "❌");
        if (r.success && r.gflops > best_gflops) {
            best_gflops = r.gflops;
        }
    }
    
    printf("\n📊 Best single-core performance: %.2f GFLOPS\n", best_gflops);
    printf("💡 This will be the baseline for multi-core comparisons\n");
    
    return 0;
}
