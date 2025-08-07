#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-numa-coordinator.h"
#include "llama.h"

#include <iostream>
#include <iomanip>

// Deep debug to trace where virtual memory addresses come from
static void trace_context_allocation() {
    extern __thread int ggml_current_numa_node;
    
    std::cout << "\n=== DEEP TRACE: Context Allocation ===" << std::endl;
    std::cout << "Initial ggml_current_numa_node: " << ggml_current_numa_node << std::endl;
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ 1 * 1024 * 1024,  // 1MB
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    
    std::cout << "Creating context with mem_size: " << params.mem_size << std::endl;
    struct ggml_context * ctx = ggml_init(params);
    
    if (!ctx) {
        std::cerr << "Failed to create context" << std::endl;
        return;
    }
    
    std::cout << "Context created:" << std::endl;
    std::cout << "  ctx ptr: " << (void*)ctx << std::endl;
    
    // Check if context pointer itself is in virtual range
    uint64_t ctx_addr = (uint64_t)ctx;
    bool ctx_in_virtual_range = (ctx_addr >= GGML_MMAP_VIRTUAL_MEMORY_BASE_OFFSET && 
                                ctx_addr < GGML_MMAP_VIRTUAL_MEMORY_BASE_OFFSET + GGML_MMAP_VIRTUAL_MEMORY_NUMA_INCREMENT);
    std::cout << "  ctx ptr in virtual range: " << (ctx_in_virtual_range ? "YES" : "NO") << std::endl;
    
    // Now create a tensor and trace the allocation
    std::cout << "\nCreating tensor..." << std::endl;
    std::cout << "ggml_current_numa_node before tensor creation: " << ggml_current_numa_node << std::endl;
    
    struct ggml_tensor * tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 16);
    if (!tensor) {
        std::cerr << "Failed to create tensor" << std::endl;
        ggml_free(ctx);
        return;
    }
    
    std::cout << "ggml_current_numa_node after tensor creation: " << ggml_current_numa_node << std::endl;
    
    std::cout << "Tensor created:" << std::endl;
    std::cout << "  tensor ptr: " << (void*)tensor << std::endl;
    std::cout << "  tensor ptr (hex): 0x" << std::hex << (uint64_t)tensor << std::dec << std::endl;
    
    // Check if tensor pointer itself is in virtual range
    uint64_t tensor_addr = (uint64_t)tensor;
    bool tensor_in_virtual_range = (tensor_addr >= GGML_MMAP_VIRTUAL_MEMORY_BASE_OFFSET && 
                                   tensor_addr < GGML_MMAP_VIRTUAL_MEMORY_BASE_OFFSET + GGML_MMAP_VIRTUAL_MEMORY_NUMA_INCREMENT);
    std::cout << "  tensor ptr in virtual range: " << (tensor_in_virtual_range ? "YES" : "NO") << std::endl;
    
    // Get tensor data pointer
    void* data_ptr = ggml_get_data(tensor);
    std::cout << "  data ptr: " << data_ptr << std::endl;
    std::cout << "  data ptr (hex): 0x" << std::hex << (uint64_t)data_ptr << std::dec << std::endl;
    
    // Check if data pointer is in virtual range
    uint64_t data_addr = (uint64_t)data_ptr;
    bool data_in_virtual_range = (data_addr >= GGML_MMAP_VIRTUAL_MEMORY_BASE_OFFSET && 
                                 data_addr < GGML_MMAP_VIRTUAL_MEMORY_BASE_OFFSET + GGML_MMAP_VIRTUAL_MEMORY_NUMA_INCREMENT);
    std::cout << "  data ptr in virtual range: " << (data_in_virtual_range ? "YES" : "NO") << std::endl;
    
    // Calculate offset between tensor and data
    uint64_t offset = (uint64_t)data_ptr - (uint64_t)tensor;
    std::cout << "  data offset from tensor: " << offset << " bytes" << std::endl;
    std::cout << "  expected offset (GGML_TENSOR_SIZE): " << GGML_TENSOR_SIZE << " bytes" << std::endl;
    
    // Test writing to see if this triggers the segfault
    if (data_ptr) {
        std::cout << "  Testing write to tensor data..." << std::endl;
        if (data_in_virtual_range) {
            std::cout << "  ⚠️  WARNING: Data is in virtual memory range - this may segfault" << std::endl;
        }
        
        try {
            std::cout << "  ggml_current_numa_node during write test: " << ggml_current_numa_node << std::endl;
            ggml_set_f32_1d(tensor, 0, 42.0f);
            float read_back = ggml_get_f32_1d(tensor, 0);
            std::cout << "  ✅ Write/read test successful: wrote 42.0, read back " << read_back << std::endl;
            std::cout << "  ggml_current_numa_node after write test: " << ggml_current_numa_node << std::endl;
        } catch (...) {
            std::cerr << "  ❌ Exception during write/read test!" << std::endl;
        }
    } else {
        std::cerr << "  ❌ Data pointer is NULL!" << std::endl;
    }
    
    std::cout << "ggml_current_numa_node before context free: " << ggml_current_numa_node << std::endl;
    ggml_free(ctx);
    std::cout << "ggml_current_numa_node after context free: " << ggml_current_numa_node << std::endl;
}

int main() {
    std::cout << "Deep Context/Tensor Allocation Trace" << std::endl;
    std::cout << "====================================" << std::endl;
    
    // Initialize backend
    llama_backend_init();
    ggml_numa_init(GGML_NUMA_STRATEGY_DISABLED);
    
    std::cout << "Virtual Memory Constants:" << std::endl;
    std::cout << "  Base: 0x" << std::hex << GGML_MMAP_VIRTUAL_MEMORY_BASE_OFFSET << std::dec << std::endl;
    std::cout << "  Increment: 0x" << std::hex << GGML_MMAP_VIRTUAL_MEMORY_NUMA_INCREMENT << std::dec << std::endl;
    
    // Test 1: First allocation
    std::cout << "\n=== FIRST ALLOCATION ===" << std::endl;
    trace_context_allocation();
    
    // Test 2: Second allocation (this should show the issue)
    std::cout << "\n=== SECOND ALLOCATION ===" << std::endl;
    trace_context_allocation();
    
    std::cout << "\n=== Trace Complete ===" << std::endl;
    return 0;
}
