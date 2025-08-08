#include <iostream>
#include "ggml.h"
#include "ggml-cpu.h"

int main() {
    std::cout << "🔍 Testing basic MUL_MAT with ggml_graph_compute..." << std::endl;
    
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
    const int64_t M = 8, N = 8, K = 8;  // Very small matrices
    struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);  // M x K
    struct ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);  // K x N
    
    std::cout << "✅ Created tensors A(" << M << "x" << K << ") and B(" << K << "x" << N << ")" << std::endl;
    
    // Fill with simple patterns
    float * A_data = (float*)ggml_get_data(A);
    float * B_data = (float*)ggml_get_data(B);
    
    for (int i = 0; i < M * K; i++) {
        A_data[i] = 1.0f; // All 1s for simple math
    }
    
    for (int i = 0; i < K * N; i++) {
        B_data[i] = 2.0f; // All 2s for simple math  
    }
    
    std::cout << "✅ Filled tensors with data" << std::endl;
    
    // Matrix multiply - result should be K*2 = 16 in each element
    struct ggml_tensor * C = ggml_mul_mat(ctx, A, B);
    
    // Build graph
    struct ggml_cgraph * cgraph = ggml_new_graph(ctx);
    ggml_build_forward_expand(cgraph, C);
    
    std::cout << "✅ Created computation graph" << std::endl;
    
    // Create compute plan
    struct ggml_cplan cplan = ggml_graph_plan(cgraph, 1, NULL); // 1 thread, no threadpool
    
    std::cout << "✅ Created compute plan" << std::endl;
    
    // Try standard GGML computation
    std::cout << "⏳ Computing with ggml_graph_compute (single-threaded)..." << std::endl;
    enum ggml_status result = ggml_graph_compute(cgraph, &cplan);
    
    if (result == GGML_STATUS_SUCCESS) {
        std::cout << "✅ MUL_MAT computed successfully!" << std::endl;
        
        // Verify result
        float * C_data = (float*)ggml_get_data(C);
        float expected = (float)(K * 2); // 8 * 2 = 16
        bool verification_passed = true;
        
        std::cout << "Checking results..." << std::endl;
        for (int i = 0; i < M && i < 3; i++) {
            for (int j = 0; j < N && j < 3; j++) {
                float actual = C_data[i * N + j];
                std::cout << "C[" << i << "," << j << "] = " << actual << " (expected " << expected << ")" << std::endl;
                if (std::abs(actual - expected) > 0.1f) {
                    std::cout << "❌ Verification failed at (" << i << "," << j << "): expected=" << expected << ", actual=" << actual << std::endl;
                    verification_passed = false;
                }
            }
        }
        
        if (verification_passed) {
            std::cout << "✅ MUL_MAT result verification passed!" << std::endl;
        } else {
            result = GGML_STATUS_FAILED;
        }
        
    } else {
        std::cout << "❌ MUL_MAT computation failed: " << result << std::endl;
    }
    
    ggml_free(ctx);
    return (result == GGML_STATUS_SUCCESS) ? 0 : 1;
}
