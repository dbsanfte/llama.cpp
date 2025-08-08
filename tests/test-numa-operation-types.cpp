#include <iostream>
#include "ggml.h" 
#include "ggml-cpu.h"
#include "ggml-numa-coordinator.h"

int main() {
    std::cout << "🔍 Testing different operations to isolate MUL_MAT hanging issue..." << std::endl;
    
    // Create context
    struct ggml_init_params params = {0};
    params.mem_size = 32 * 1024 * 1024;  
    params.mem_buffer = NULL;
    params.no_alloc = false;
    
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::cout << "❌ Failed to create context" << std::endl;
        return 1;
    }
    
    // Test different operations
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 8);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 8);
    
    // Fill with test data
    float * a_data = (float*)ggml_get_data(a);
    float * b_data = (float*)ggml_get_data(b);
    
    for (int i = 0; i < 64; i++) {
        a_data[i] = 1.0f;
        b_data[i] = 2.0f;
    }
    
    // Get coordinator manager
    struct ggml_numa_coordinator_manager * mgr = 
        ggml_numa_coordinator_manager_get_global(4, true);
    
    if (!mgr) {
        std::cout << "❌ Failed to create coordinator manager" << std::endl;
        ggml_free(ctx);
        return 1;
    }
    
    // Test 1: ADD (known to work partially)
    std::cout << "\n🧪 Test 1: ADD operation" << std::endl;
    struct ggml_tensor * add_result = ggml_add(ctx, a, b);
    struct ggml_cgraph * add_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(add_graph, add_result);
    
    int result = ggml_numa_coordinator_manager_compute_graph(mgr, add_graph);
    std::cout << (result == 0 ? "✅ ADD completed" : "❌ ADD failed") << std::endl;
    
    // Test 2: MUL (element-wise multiplication - simpler than MUL_MAT)
    std::cout << "\n🧪 Test 2: MUL (element-wise) operation" << std::endl;
    struct ggml_tensor * mul_result = ggml_mul(ctx, a, b);
    struct ggml_cgraph * mul_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(mul_graph, mul_result);
    
    result = ggml_numa_coordinator_manager_compute_graph(mgr, mul_graph);
    std::cout << (result == 0 ? "✅ MUL completed" : "❌ MUL failed") << std::endl;
    
    // Test 3: Small MUL_MAT (this might hang)
    std::cout << "\n🧪 Test 3: MUL_MAT (small 8x8) operation - MIGHT HANG" << std::endl;
    std::cout << "If this hangs, we know MUL_MAT has specific threading issues..." << std::endl;
    
    // Create smaller matrices for MUL_MAT: C = A * B^T
    struct ggml_tensor * c_small = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 8);
    struct ggml_tensor * d_small = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 8);
    
    // Fill with simple data for matrix multiplication
    float * c_data = (float*)ggml_get_data(c_small);
    float * d_data = (float*)ggml_get_data(d_small);
    
    for (int i = 0; i < 64; i++) {
        c_data[i] = 1.0f;
        d_data[i] = 1.0f;
    }
    
    struct ggml_tensor * mulmat_result = ggml_mul_mat(ctx, c_small, d_small);
    struct ggml_cgraph * mulmat_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(mulmat_graph, mulmat_result);
    
    // Use a timeout approach - if it takes too long, we know it's hanging
    std::cout << "Starting MUL_MAT computation (timeout will indicate hang)..." << std::endl;
    result = ggml_numa_coordinator_manager_compute_graph(mgr, mulmat_graph);
    std::cout << (result == 0 ? "✅ MUL_MAT completed!" : "❌ MUL_MAT failed") << std::endl;
    
    ggml_free(ctx);
    return 0;
}
