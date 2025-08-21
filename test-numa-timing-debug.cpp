/**
 * Quick NUMA timing debug test
 */

#include <cstdio>
#include <cstring>
#include <cmath>
#include <time.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-numa-executor.h"
#include "ggml-numa-simple-coordinator.h"

int main() {
    printf("=== NUMA Timing Debug Test ===\n");
    
    // Initialize NUMA with MIRROR strategy
    ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);
    
    // Create a medium-sized tensor for testing
    size_t ctx_size = 32 * 1024 * 1024; // 32MB
    struct ggml_init_params params = {
        .mem_size = ctx_size,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        printf("❌ Failed to create ggml context\n");
        return 1;
    }
    
    // Create tensors for ADD operation
    int dim = 1024; // 1K x 1K = 1M elements = 4MB
    struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, dim);
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, dim);
    struct ggml_tensor* result = ggml_add(ctx, a, b);
    
    // Initialize data
    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    for (int i = 0; i < dim * dim; i++) {
        a_data[i] = 1.0f;
        b_data[i] = 2.0f;
    }
    
    printf("📊 Testing tensor: %dx%d (%.1f MB)\n", dim, dim, (dim * dim * sizeof(float)) / (1024.0 * 1024.0));
    
    // Test NUMA execution with debug output enabled
    printf("\n🎯 Running NUMA execution...\n");
    
    // Create computation graph
    struct ggml_cgraph* cgraph = ggml_new_graph(ctx);
    ggml_build_forward_expand(cgraph, result);
    
    // Execute with detailed timing
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    enum ggml_status status = ggml_graph_compute_with_ctx(ctx, cgraph, 8);
    
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double total_time_ms = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + 
                          (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;
    
    printf("✅ NUMA execution completed with status %d in %.3fms\n", status, total_time_ms);
    
    // Verify result
    float* result_data = (float*)result->data;
    bool correct = true;
    for (int i = 0; i < 100; i++) { // Check first 100 elements
        if (fabs(result_data[i] - 3.0f) > 0.001f) {
            correct = false;
            break;
        }
    }
    
    printf("🔍 Result verification: %s\n", correct ? "✅ PASSED" : "❌ FAILED");
    
    ggml_free(ctx);
    return 0;
}
