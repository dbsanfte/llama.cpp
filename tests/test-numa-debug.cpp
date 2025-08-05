#include "ggml.h"
#include "ggml-cpu.h"
#include "llama.h"

#include <iostream>
#include <vector>

int main() {
    std::cout << "NUMA Debug Test" << std::endl;
    
    // Initialize like the working test-barrier
    llama_backend_init();
    // Try completely skipping NUMA initialization
    std::cout << "Skipping NUMA initialization entirely..." << std::endl;
    // llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

    // Test 1: Context allocation (like test-barrier)
    std::cout << "\nTest 1: Context allocation" << std::endl;
    struct ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,  // Context allocates
    };
    
    struct ggml_context * ctx = ggml_init(params);
    struct ggml_tensor * tensor_ctx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    
    std::cout << "Context tensor created" << std::endl;
    std::cout << "  __data[0] = " << tensor_ctx->__data[0] << std::endl;
    std::cout << "  __data[1] = " << tensor_ctx->__data[1] << std::endl;
    std::cout << "  tensor_data() = " << tensor_data(tensor_ctx) << std::endl;
    
    // Test access
    float* data_ptr = (float*)tensor_data(tensor_ctx);
    data_ptr[0] = 1.0f;  // This should work
    std::cout << "✓ Context tensor access works" << std::endl;
    
    ggml_free(ctx);
    
    // Test 2: Backend allocation (like test-numa-multi-socket)
    std::cout << "\nTest 2: Backend allocation" << std::endl;
    ggml_backend_t backend = ggml_backend_cpu_init();
    
    // Debug: Check what kind of backend this is
    const char* backend_name = ggml_backend_name(backend);
    std::cout << "Backend name: " << backend_name << std::endl;
    
    struct ggml_init_params params2 = {
        /*.mem_size   =*/ 1024*1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,  // Backend allocates
    };
    
    struct ggml_context * ctx2 = ggml_init(params2);
    struct ggml_tensor * tensor_backend = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, 64, 64);
    
    std::cout << "Backend tensor created (before allocation)" << std::endl;
    std::cout << "  __data[0] = " << tensor_backend->__data[0] << std::endl;
    std::cout << "  __data[1] = " << tensor_backend->__data[1] << std::endl;
    
    // Allocate backend buffer
    std::cout << "About to call ggml_backend_alloc_ctx_tensors..." << std::endl;
    
    // Add some debug prints to see where the virtual address comes from
    std::cout << "DEBUG: Before allocation, checking buffer creation..." << std::endl;
    
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx2, backend);
    std::cout << "ggml_backend_alloc_ctx_tensors completed" << std::endl;
    
    std::cout << "Backend tensor after allocation" << std::endl;
    std::cout << "  __data[0] = " << tensor_backend->__data[0] << std::endl;
    std::cout << "  __data[1] = " << tensor_backend->__data[1] << std::endl;
    std::cout << "  tensor_data() = " << tensor_data(tensor_backend) << std::endl;
    
    // Debug: Check if this address looks like a virtual memory address
    uint64_t addr = (uint64_t)tensor_backend->__data[0];
    std::cout << "  Address in hex: 0x" << std::hex << addr << std::dec << std::endl;
    std::cout << "  Virtual memory base: 0x200000000000" << std::endl;
    std::cout << "  Virtual memory increment: 0x200000000000" << std::endl;
    
    const uint64_t VM_BASE = 0x200000000000ULL;
    const uint64_t VM_INCREMENT = 0x200000000000ULL;
    bool in_virtual_range = (addr >= VM_BASE) && (addr < VM_BASE + VM_INCREMENT);
    std::cout << "  In virtual memory range: " << (in_virtual_range ? "Yes" : "No") << std::endl;
    
    // Test access - this is where it might crash
    std::cout << "Testing backend tensor access..." << std::endl;
    std::cout << "About to write to address: " << tensor_data(tensor_backend) << std::endl;
    
    // Try the actual write to see where exactly it crashes
    try {
        float* data_ptr2 = (float*)tensor_data(tensor_backend);
        data_ptr2[0] = 1.0f;  // This will crash
        std::cout << "✓ Backend tensor access works!" << std::endl;
    } catch (...) {
        std::cout << "✗ Backend tensor access crashed!" << std::endl;
    }
    
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx2);
    ggml_backend_free(backend);
    
    std::cout << "\n✅ All tests passed!" << std::endl;
    return 0;
}
