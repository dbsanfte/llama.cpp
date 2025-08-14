#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-numa-operation-dispatch.h"
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

void test_rope_operation(int64_t n_embd, int64_t n_seq, int64_t n_batch, const char* test_name) {
    printf("Testing %s ROPE: [%ld, %ld, %ld] = %ld elements\n", 
           test_name, n_embd, n_seq, n_batch, n_embd * n_seq * n_batch);
    
    // Initialize backend
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        printf("❌ Failed to initialize CPU backend\n");
        return;
    }

    // Create context
    struct ggml_init_params params = {
        .mem_size = 64 * 1024 * 1024,  // 64MB for large tensors
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("❌ Failed to initialize context\n");
        ggml_backend_free(backend);
        return;
    }
    
    // Create ROPE operation tensors
    struct ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_seq, n_batch);
    struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_batch);
    
    // Create ROPE operation with basic parameters
    struct ggml_tensor * rope_result = ggml_rope_ext(
        ctx, input, pos, NULL,
        n_embd,     // n_dims (all dimensions rotated)  
        0,          // mode (standard ROPE)
        0,          // n_ctx_orig
        10000.0f,   // freq_base
        1.0f,       // freq_scale
        0.0f,       // ext_factor
        1.0f,       // attn_factor
        1.0f,       // beta_fast
        1.0f        // beta_slow
    );
    
    // Try NUMA graph computation
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, rope_result);
    
    printf("Attempting NUMA graph computation with virtual NUMA enabled...\n");
    enum ggml_status numa_result = ggml_numa_graph_compute_with_virtual(gf, 4, true);
    
    if (numa_result == GGML_STATUS_SUCCESS) {
        printf("✅ %s ROPE completed via NUMA dispatch!\n", test_name);
    } else {
        printf("⚠️  %s ROPE NUMA dispatch failed, falling back to standard computation\n", test_name);
        
        // Fallback to standard computation
        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_alloc_graph(allocr, gf);
        
        // Initialize test data  
        std::vector<float> input_data(n_embd * n_seq * n_batch, 1.0f);
        std::vector<int32_t> pos_data(n_batch, 0);
        
        ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
        ggml_backend_tensor_set(pos, pos_data.data(), 0, pos_data.size() * sizeof(int32_t));
        
        int compute_result = ggml_backend_graph_compute(backend, gf);
        if (compute_result == GGML_STATUS_SUCCESS) {
            printf("✅ %s ROPE completed via standard backend\n", test_name);
        } else {
            printf("❌ %s ROPE computation failed\n", test_name);
        }
        
        ggml_gallocr_free(allocr);
    }
    
    // Cleanup
    ggml_free(ctx);
    ggml_backend_free(backend);
}

int main() {
    printf("🚀 Testing Phase 2: ROPE NUMA Selective Migration...\n");
    
    // Enable NUMA first
    printf("Initializing NUMA with DISTRIBUTE strategy...\n");
    ggml_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);
    
    // Test small operation (should use fallback)
    printf("\n--- Test 1: Small ROPE Operation (should use fallback) ---\n");
    test_rope_operation(64, 32, 1, "Small");
    
    // Test large operation (should use NUMA)
    printf("\n--- Test 2: Large ROPE Operation (should use NUMA dispatch) ---\n");
    test_rope_operation(512, 512, 8, "Large"); // 512*512*8 = ~2M elements > 100K threshold
    
    printf("\n🎉 Phase 2 ROPE NUMA selective migration test completed!\n");
    return 0;
}
