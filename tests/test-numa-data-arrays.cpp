#include "ggml.h"
#include <iostream>
#include <cstdint>
#include <iomanip>

extern __thread int ggml_current_numa_node;

int main() {
    std::cout << "NUMA Data Arrays Test" << std::endl;
    std::cout << "====================" << std::endl;
    
    std::cout << "Virtual Memory Constants:" << std::endl;
    std::cout << "  Base: 0x" << std::hex << GGML_MMAP_VIRTUAL_MEMORY_BASE_OFFSET << std::dec << std::endl;
    std::cout << "  Increment: 0x" << std::hex << GGML_MMAP_VIRTUAL_MEMORY_NUMA_INCREMENT << std::dec << std::endl;

    // Create contexts
    struct ggml_init_params params = {
        /*.mem_size   =*/ 1 * 1024 * 1024,  // 1MB
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    
    std::cout << "\nCreating first context..." << std::endl;
    struct ggml_context * ctx1 = ggml_init(params);
    std::cout << "ggml_current_numa_node: " << ggml_current_numa_node << std::endl;
    
    struct ggml_tensor * tensor1 = ggml_new_tensor_2d(ctx1, GGML_TYPE_F32, 16, 16);
    
    std::cout << "First tensor:" << std::endl;
    std::cout << "  tensor ptr: " << (void*)tensor1 << std::endl;
    std::cout << "  __data[0]: " << tensor1->__data[0] << std::endl;
    std::cout << "  __data[1]: " << tensor1->__data[1] << std::endl;
    std::cout << "  ggml_get_data(): " << ggml_get_data(tensor1) << std::endl;
    std::cout << "  normal calc (tensor + GGML_TENSOR_SIZE): " << (void*)((char*)tensor1 + GGML_TENSOR_SIZE) << std::endl;
    
    ggml_free(ctx1);
    
    std::cout << "\nCreating second context..." << std::endl;
    struct ggml_context * ctx2 = ggml_init(params);
    std::cout << "ggml_current_numa_node: " << ggml_current_numa_node << std::endl;
    
    struct ggml_tensor * tensor2 = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, 16, 16);
    
    std::cout << "Second tensor:" << std::endl;
    std::cout << "  tensor ptr: " << (void*)tensor2 << std::endl;
    std::cout << "  __data[0]: " << tensor2->__data[0] << std::endl;
    std::cout << "  __data[1]: " << tensor2->__data[1] << std::endl;
    std::cout << "  ggml_get_data(): " << ggml_get_data(tensor2) << std::endl;
    std::cout << "  normal calc (tensor + GGML_TENSOR_SIZE): " << (void*)((char*)tensor2 + GGML_TENSOR_SIZE) << std::endl;
    
    // Check which array element is actually being returned
    void* returned_data = ggml_get_data(tensor2);
    if (returned_data == tensor2->__data[0]) {
        std::cout << "  ggml_get_data() == __data[0]" << std::endl;
    } else if (returned_data == tensor2->__data[1]) {
        std::cout << "  ggml_get_data() == __data[1]" << std::endl;
    } else {
        std::cout << "  ggml_get_data() matches neither __data[0] nor __data[1]!" << std::endl;
    }
    
    ggml_free(ctx2);
    
    return 0;
}
