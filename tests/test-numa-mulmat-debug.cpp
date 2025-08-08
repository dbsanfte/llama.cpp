#include <iostream>
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-numa-coordinator.h"

int main() {
    std::cout << "🔍 Testing NUMA coordinator with MUL_MAT operation..." << std::endl;
    
    // Create context
    struct ggml_init_params params = {0};
    params.mem_size = 64 * 1024 * 1024;  // 64MB 
    params.mem_buffer = NULL;
    params.no_alloc = false;
    
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::cout << "❌ Failed to create context" << std::endl;
        return 1;
    }
    
    // Create matrices for multiplication: C = A * B
    const int64_t M = 32, N = 32, K = 32;  // Smaller matrices to isolate the issue
    struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);  // M x K
    struct ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);  // K x N
    
    // Fill with simple patterns
    float * A_data = (float*)ggml_get_data(A);
    float * B_data = (float*)ggml_get_data(B);
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            A_data[i * K + j] = 1.0f; // All 1s for simple math
        }
    }
    
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            B_data[i * N + j] = 2.0f; // All 2s for simple math
        }
    }
    
    // Matrix multiply - result should be K*2 = 64 in each element
    struct ggml_tensor * C = ggml_mul_mat(ctx, A, B);
    
    // Build graph
    struct ggml_cgraph * cgraph = ggml_new_graph(ctx);
    ggml_build_forward_expand(cgraph, C);
    
    std::cout << "✅ Created graph with MUL_MAT operation (" << M << "x" << K << " * " << K << "x" << N << ")" << std::endl;
    
    // Get coordinator manager
    struct ggml_numa_coordinator_manager * mgr = 
        ggml_numa_coordinator_manager_get_global(4, true);
    
    if (!mgr) {
        std::cout << "❌ Failed to create coordinator manager" << std::endl;
        ggml_free(ctx);
        return 1;
    }
    
    std::cout << "✅ Created coordinator manager" << std::endl;
    
    // Try to compute the graph
    std::cout << "⏳ Computing MUL_MAT graph (this might hang)..." << std::endl;
    int result = ggml_numa_coordinator_manager_compute_graph(mgr, cgraph);
    
    if (result == 0) {
        std::cout << "✅ MUL_MAT computed successfully!" << std::endl;
        
        // Verify result
        float * C_data = (float*)ggml_get_data(C);
        float expected = (float)(K * 2); // 32 * 2 = 64
        bool verification_passed = true;
        
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float actual = C_data[i * N + j];
                if (std::abs(actual - expected) > 0.1f) {
                    std::cout << "❌ Verification failed at (" << i << "," << j << "): expected=" << expected << ", actual=" << actual << std::endl;
                    verification_passed = false;
                    break;
                }
            }
            if (!verification_passed) break;
        }
        
        if (verification_passed) {
            std::cout << "✅ MUL_MAT result verification passed!" << std::endl;
        } else {
            result = 1;
        }
        
    } else {
        std::cout << "❌ MUL_MAT computation failed: " << result << std::endl;
    }
    
    ggml_free(ctx);
    return result;
}
