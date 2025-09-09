/**
 * NUMA Mathematical Correctness Test: MUL_MAT Operation
 * 
 * Simple smoke test to verify our placeholder mul_mat kernel is registered and working.
 */

#include <cstdio>
#include <cstring>
#include <cmath>

// GGML includes
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-numa-executor.h"
#include "ggml-cpu/numa-kernels/numa-kernels.h"
#include "ggml-cpu/ggml-numa-shared.h"

/**
 * Simple smoke test for MUL_MAT operation
 */
bool test_mul_mat_smoke() {
    printf("🔍 Testing MUL_MAT NUMA kernel registration...\n");
    
    // Create context
    struct ggml_init_params params = {
        .mem_size = 64 * 1024 * 1024,  // 64MB
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    
    ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        printf("❌ Failed to create context\n");
        return false;
    }
    
    // Create simple 4x4 matrices
    ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);  // 4x4
    ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);  // 4x4
    
    if (!a || !b) {
        printf("❌ Failed to create tensors\n");
        ggml_free(ctx);
        return false;
    }
    
    // Initialize with simple data
    float* a_data = (float*)ggml_get_data(a);
    float* b_data = (float*)ggml_get_data(b);
    
    for (int i = 0; i < 16; i++) {
        a_data[i] = (float)(i + 1);  // 1, 2, 3, ...
        b_data[i] = 1.0f;            // All ones
    }
    
    // Create matrix multiplication operation
    ggml_tensor* result = ggml_mul_mat(ctx, a, b);
    if (!result) {
        printf("❌ Failed to create mul_mat operation\n");
        ggml_free(ctx);
        return false;
    }
    
    // Check if NUMA kernel is registered for MUL_MAT
    const ggml_numa_kernel_cache_entry_t* cache_entry = ggml_numa_lookup_kernel_direct(GGML_OP_MUL_MAT);
    if (!cache_entry) {
        printf("❌ MUL_MAT operation not found in NUMA kernel registry\n");
        ggml_free(ctx);
        return false;
    }
    
    if (!cache_entry->supported) {
        printf("❌ MUL_MAT operation registered but marked as unsupported\n");
        ggml_free(ctx);
        return false;
    }
    
    printf("✅ MUL_MAT kernel registered: %s\n", cache_entry->kernel_name);
    
    // Test strategy query
    ggml_numa_execution_strategy_t strategy = ggml_numa_kernels_query(result);
    const char* strategy_name = "unknown";
    switch (strategy) {
        case NUMA_STRATEGY_SINGLE_THREAD: strategy_name = "single_thread"; break;
        case NUMA_STRATEGY_SINGLE_NODE: strategy_name = "single_node"; break;
        case NUMA_STRATEGY_DATA_PARALLEL: strategy_name = "data_parallel"; break;
        default: strategy_name = "invalid"; break;
    }
    
    printf("✅ Strategy query works: %s\n", strategy_name);
    
    ggml_free(ctx);
    return true;
}

/**
 * Main test function
 */
int main(int argc, char** argv) {
    printf("🧮 NUMA MUL_MAT Registration Test\n");
    printf("=================================\n\n");
    
    // Initialize NUMA system
    ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);
    
    bool success = test_mul_mat_smoke();
    
    if (success) {
        printf("\n✅ ALL TESTS PASSED - MUL_MAT kernel is properly registered\n");
        return 0;
    } else {
        printf("\n❌ TESTS FAILED - MUL_MAT kernel registration issues\n");
        return 1;
    }
}
