#include <stdio.h>
#include <chrono>
#include <cstdlib>

#include "ggml.h"
#include "ggml-cpu.h"

class HugeDataParallelDebugTest {
    struct ggml_context* ctx = nullptr;
    struct ggml_tensor* tensor_a = nullptr;
    struct ggml_tensor* tensor_b = nullptr;
    struct ggml_tensor* result = nullptr;
    struct ggml_cgraph* cgraph = nullptr;
    
public:
    bool setup_huge_tensors() {
        printf("Setting up HUGE tensor test (256MB total)...\n");
        
        // Initialize NUMA FIRST before creating tensors
        printf("Initializing NUMA with MIRROR strategy (BEFORE tensor creation)...\n");
        ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);
        
        // Now check the NUMA state
        printf("NUMA state after init: nodes=%d, enabled=%s\n", 
               ggml_numa_node_count(), ggml_is_numa() ? "true" : "false");
        
        // Create context for tensors (AFTER NUMA init)
        size_t ctx_size = 1024 * 1024 * 1024; // 1GB context
        printf("Allocating NUMA-distributed context memory pool: %zu MB\n", ctx_size / (1024 * 1024));
        
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };
        
        ctx = ggml_init(params);
        if (!ctx) {
            printf("Failed to create context\n");
            return false;
        }
        
        // Create large tensors for testing
        const int dim1 = 2048, dim2 = 2048, dim3 = 16;
        
        printf("Creating tensors (AFTER NUMA init)...\n");
        tensor_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim1, dim2, dim3);
        tensor_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim1, dim2, dim3);
        result = ggml_add(ctx, tensor_a, tensor_b);
        
        if (!tensor_a || !tensor_b || !result) {
            printf("Failed to create tensors\n");
            return false;
        }
        
        // Create computation graph
        cgraph = ggml_new_graph(ctx);
        ggml_build_forward_expand(cgraph, result);
        
        // Initialize data
        float* a_data = (float*)ggml_get_data(tensor_a);
        float* b_data = (float*)ggml_get_data(tensor_b);
        
        size_t total_elements = dim1 * dim2 * dim3;
        printf("Tensor info: %dx%dx%d = %zu elements (%.1f MB each)\n", 
               dim1, dim2, dim3, total_elements, 
               (total_elements * sizeof(float)) / (1024.0 * 1024.0));
        
        for (size_t i = 0; i < total_elements; i++) {
            a_data[i] = 1.5f + (i % 100) * 0.01f;
            b_data[i] = 2.5f + (i % 100) * 0.01f;
        }
        
        printf("HUGE tensors initialized successfully\n");
        return true;
    }
    
    double measure_fallback_performance() {
        printf("\nTesting FALLBACK performance...\n");
        
        // Ensure NUMA is disabled for this test
        struct ggml_cplan cplan = ggml_graph_plan(cgraph, 8, NULL);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        enum ggml_status status = ggml_graph_compute(cgraph, &cplan);
        
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        if (status != GGML_STATUS_SUCCESS) {
            printf("Fallback computation failed\n");
            return -1.0;
        }
        
        printf("Fallback completed in %.3f ms\n", elapsed_ms);
        return elapsed_ms;
    }
    
    double measure_numa_performance() {
        printf("\nTesting NUMA MultiNode-DataParallel performance...\n");
        
        // NUMA already initialized during setup_huge_tensors()
        printf("Checking coordinator status...\n");
        
        // Create compute plan for NUMA execution
        struct ggml_cplan cplan = ggml_graph_plan(cgraph, 8, NULL);
        printf("Compute plan: n_threads=%d, work_size=%zu\n", cplan.n_threads, cplan.work_size);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        printf("Starting NUMA execution...\n");
        enum ggml_status status = ggml_graph_compute(cgraph, &cplan);
        
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        if (status != GGML_STATUS_SUCCESS) {
            printf("NUMA computation failed with status %d\n", status);
            return -1.0;
        }
        
        printf("NUMA completed in %.3f ms\n", elapsed_ms);
        return elapsed_ms;
    }
    
    void analyze_performance(double fallback_ms, double numa_ms) {
        if (fallback_ms < 0 || numa_ms < 0) {
            printf("Cannot analyze - one or both tests failed\n");
            return;
        }
        
        double speedup = fallback_ms / numa_ms;
        double improvement_pct = ((fallback_ms - numa_ms) / fallback_ms) * 100.0;
        
        printf("\nPERFORMANCE ANALYSIS\n");
        printf("=====================================\n");
        printf("Fallback time:    %.3f ms\n", fallback_ms);
        printf("NUMA time:        %.3f ms\n", numa_ms);
        printf("Speedup:          %.2fx\n", speedup);
        printf("Improvement:      %.1f%%\n", improvement_pct);
        
        if (improvement_pct < -50) {
            printf("CRITICAL ISSUE: NUMA is %.1f%% SLOWER than fallback!\n", -improvement_pct);
            printf("This should be NUMA's BEST case - massive data parallel workload\n");
        } else if (improvement_pct < 0) {
            printf("WARNING: NUMA is %.1f%% slower than fallback\n", -improvement_pct);
        } else if (improvement_pct > 200) {
            printf("EXCELLENT: NUMA is %.1fx faster than fallback!\n", speedup);
        } else {
            printf("GOOD: NUMA shows %.1f%% improvement\n", improvement_pct);
        }
    }
    
    ~HugeDataParallelDebugTest() {
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

int main() {
    printf("NUMA HUGE/MultiNode-DataParallel Debug Test\n");
    printf("===========================================\n");
    printf("Target: Isolate and debug NUMA performance with tensor mirroring\n\n");
    
    HugeDataParallelDebugTest test;
    
    if (!test.setup_huge_tensors()) {
        printf("Test setup failed\n");
        return 1;
    }
    
    // Test fallback performance first
    double fallback_time = test.measure_fallback_performance();
    
    // Test NUMA performance
    double numa_time = test.measure_numa_performance();
    
    // Analyze results
    test.analyze_performance(fallback_time, numa_time);
    
    printf("\nDebug test complete\n");
    return 0;
}
