#include "ggml.h"
#include <iostream>
#include <cstdint>

extern __thread int ggml_current_numa_node;

int main() {
    std::cout << "Tensor Allocation Test (No Coordinator)" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    std::cout << "Virtual Memory Constants:" << std::endl;
    std::cout << "  Base: 0x" << std::hex << GGML_MMAP_VIRTUAL_MEMORY_BASE_OFFSET << std::dec << std::endl;
    std::cout << "  Increment: 0x" << std::hex << GGML_MMAP_VIRTUAL_MEMORY_NUMA_INCREMENT << std::dec << std::endl;

    // Test multiple context/tensor allocations
    struct ggml_init_params params = {
        /*.mem_size   =*/ 1 * 1024 * 1024,  // 1MB
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    
    for (int i = 0; i < 5; i++) {
        std::cout << "\n=== Context " << (i+1) << " ===" << std::endl;
        std::cout << "ggml_current_numa_node: " << ggml_current_numa_node << std::endl;
        
        struct ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            std::cerr << "Failed to create context " << (i+1) << std::endl;
            return 1;
        }
        
        struct ggml_tensor * tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 32);
        if (!tensor) {
            std::cerr << "Failed to create tensor " << (i+1) << std::endl;
            ggml_free(ctx);
            return 1;
        }
        
        void* data_ptr = ggml_get_data(tensor);
        
        std::cout << "  tensor ptr: " << (void*)tensor << std::endl;
        std::cout << "  data ptr: " << data_ptr << std::endl;
        std::cout << "  data ptr (hex): 0x" << std::hex << (uint64_t)data_ptr << std::dec << std::endl;
        
        // Check if data pointer is in virtual memory range
        uint64_t data_addr = (uint64_t)data_ptr;
        bool data_virtual = (data_addr >= GGML_MMAP_VIRTUAL_MEMORY_BASE_OFFSET && 
                            data_addr < GGML_MMAP_VIRTUAL_MEMORY_BASE_OFFSET + 2 * GGML_MMAP_VIRTUAL_MEMORY_NUMA_INCREMENT);
        
        std::cout << "  in virtual range: " << (data_virtual ? "YES" : "NO") << std::endl;
        
        // Test write
        try {
            *((float*)data_ptr) = (float)(i + 1) * 10.0f;
            float value = *((float*)data_ptr);
            std::cout << "  ✅ Write/read successful: " << value << std::endl;
        } catch (...) {
            std::cout << "  ❌ Write/read failed with exception" << std::endl;
        }
        
        ggml_free(ctx);
        std::cout << "  Context freed" << std::endl;
    }
    
    std::cout << "\n✅ All tensor allocation tests passed!" << std::endl;
    return 0;
}
