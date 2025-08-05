#include "ggml.h"
#include "ggml-cpu.h"
#include <iostream>
#include <vector>

int main() {
    std::cout << "Minimal NUMA Test - Testing Basic Backend Operations" << std::endl;
    
    // Create CPU backend
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        std::cerr << "Failed to create CPU backend" << std::endl;
        return 1;
    }
    std::cout << "✓ CPU backend created" << std::endl;
    
    // Create context
    struct ggml_init_params params = {
        /*.mem_size   =*/ 1024 * 1024,  // 1MB
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::cerr << "Failed to create GGML context" << std::endl;
        ggml_backend_free(backend);
        return 1;
    }
    std::cout << "✓ GGML context created" << std::endl;
    
    // Create simple 2x2 tensor
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
    if (!a) {
        std::cerr << "Failed to create tensor" << std::endl;
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }
    std::cout << "✓ Tensor created: " << a->ne[0] << "x" << a->ne[1] << std::endl;
    
    // Allocate buffer
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        std::cerr << "Failed to allocate buffer" << std::endl;
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }
    std::cout << "✓ Buffer allocated" << std::endl;
    
    // Try to set data
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::cout << "About to set tensor data..." << std::endl;
    ggml_backend_tensor_set(a, data.data(), 0, ggml_nbytes(a));
    std::cout << "✓ Tensor data set successfully!" << std::endl;
    
    // Clean up
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    
    std::cout << "✅ Minimal test passed!" << std::endl;
    return 0;
}
