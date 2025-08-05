#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include <iostream>

int main() {
    std::cout << "Simple tensor allocation test" << std::endl;
    
    // Create CPU backend
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        std::cerr << "Failed to create CPU backend" << std::endl;
        return 1;
    }
    std::cout << "✓ CPU backend created" << std::endl;
    
    // Create context
    struct ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024,  // 1MB
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,  // Backend handles allocation
    };
    
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::cerr << "Failed to create context" << std::endl;
        ggml_backend_free(backend);
        return 1;
    }
    std::cout << "✓ Context created" << std::endl;
    
    // Create simple tensor
    struct ggml_tensor * tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
    if (!tensor) {
        std::cerr << "Failed to create tensor" << std::endl;
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }
    std::cout << "✓ Tensor created: " << tensor->ne[0] << "x" << tensor->ne[1] << std::endl;
    
    // Check tensor data before allocation
    std::cout << "Before allocation:" << std::endl;
#ifdef GGML_NUMA_MIRROR
    std::cout << "  tensor->__data[0] = " << tensor->__data[0] << std::endl;
    std::cout << "  tensor->__data[1] = " << tensor->__data[1] << std::endl;
#else
    std::cout << "  tensor->data = " << tensor->data << std::endl;
#endif
    
    // Allocate backend buffer
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        std::cerr << "Failed to allocate backend buffer" << std::endl;
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }
    std::cout << "✓ Backend buffer allocated" << std::endl;
    
    // Check tensor data after allocation
    std::cout << "After allocation:" << std::endl;
#ifdef GGML_NUMA_MIRROR
    std::cout << "  tensor->__data[0] = " << tensor->__data[0] << std::endl;
    std::cout << "  tensor->__data[1] = " << tensor->__data[1] << std::endl;
#else
    std::cout << "  tensor->data = " << tensor->data << std::endl;
#endif
    
    // Test tensor_data() function
    void* data_ptr = tensor_data(tensor);
    std::cout << "  tensor_data() returns = " << data_ptr << std::endl;
    
    if (data_ptr == nullptr) {
        std::cerr << "❌ tensor_data() returned NULL!" << std::endl;
    } else {
        std::cout << "✅ tensor_data() returned valid pointer" << std::endl;
    }
    
    // Cleanup
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    
    return data_ptr ? 0 : 1;
}
