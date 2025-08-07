#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-numa-coordinator.h"
#include "llama.h"

#include <iostream>
#include <iomanip>

// Test to debug tensor data allocation issues
static void debug_tensor_allocation() {
    std::cout << "\n=== Debugging Tensor Data Allocation ===" << std::endl;
    
    // Check virtual memory constants
    std::cout << "Virtual Memory Constants:" << std::endl;
    std::cout << "  GGML_MMAP_VIRTUAL_MEMORY_BASE_OFFSET: 0x" << std::hex << GGML_MMAP_VIRTUAL_MEMORY_BASE_OFFSET << std::dec << std::endl;
    std::cout << "  GGML_MMAP_VIRTUAL_MEMORY_NUMA_INCREMENT: 0x" << std::hex << GGML_MMAP_VIRTUAL_MEMORY_NUMA_INCREMENT << std::dec << std::endl;
    
    // Check NUMA status
    std::cout << "\nNUMA Status:" << std::endl;
    std::cout << "  ggml_is_numa(): " << (ggml_is_numa() ? "true" : "false") << std::endl;
    std::cout << "  ggml_numa_node_count(): " << ggml_numa_node_count() << std::endl;
    
    // Check current NUMA node (thread-local variable)
    extern __thread int ggml_current_numa_node;
    std::cout << "  ggml_current_numa_node: " << ggml_current_numa_node << std::endl;
    
    std::cout << "\n--- Creating First Context and Tensor ---" << std::endl;
    
    struct ggml_init_params params1 = {
        /*.mem_size   =*/ 2 * 1024 * 1024,  // 2MB
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    
    struct ggml_context * ctx1 = ggml_init(params1);
    if (!ctx1) {
        std::cerr << "Failed to create first context" << std::endl;
        return;
    }
    
    struct ggml_tensor * tensor1 = ggml_new_tensor_2d(ctx1, GGML_TYPE_F32, 32, 32);
    if (!tensor1) {
        std::cerr << "Failed to create first tensor" << std::endl;
        ggml_free(ctx1);
        return;
    }
    
    void* data1 = ggml_get_data(tensor1);
    std::cout << "First tensor:" << std::endl;
    std::cout << "  tensor ptr: " << (void*)tensor1 << std::endl;
    std::cout << "  data ptr: " << data1 << std::endl;
    std::cout << "  data raw (hex): 0x" << std::hex << (uint64_t)data1 << std::dec << std::endl;
    
    // Check if data address is in virtual memory range
    uint64_t data_addr = (uint64_t)data1;
    bool in_virtual_range = (data_addr >= GGML_MMAP_VIRTUAL_MEMORY_BASE_OFFSET && 
                           data_addr < GGML_MMAP_VIRTUAL_MEMORY_BASE_OFFSET + GGML_MMAP_VIRTUAL_MEMORY_NUMA_INCREMENT);
    std::cout << "  in virtual memory range: " << (in_virtual_range ? "YES" : "NO") << std::endl;
    
    // Test writing to tensor
    if (data1) {
        std::cout << "  Testing write to first tensor..." << std::endl;
        ggml_set_f32_1d(tensor1, 0, 42.0f);
        float read_back = ggml_get_f32_1d(tensor1, 0);
        std::cout << "  Write/read test: wrote 42.0, read back " << read_back << std::endl;
    }
    
    ggml_free(ctx1);
    std::cout << "  First context freed" << std::endl;
    
    std::cout << "\n--- Creating Second Context and Tensor ---" << std::endl;
    
    struct ggml_init_params params2 = {
        /*.mem_size   =*/ 2 * 1024 * 1024,  // 2MB
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    
    struct ggml_context * ctx2 = ggml_init(params2);
    if (!ctx2) {
        std::cerr << "Failed to create second context" << std::endl;
        return;
    }
    
    struct ggml_tensor * tensor2 = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, 32, 32);
    if (!tensor2) {
        std::cerr << "Failed to create second tensor" << std::endl;
        ggml_free(ctx2);
        return;
    }
    
    void* data2 = ggml_get_data(tensor2);
    std::cout << "Second tensor:" << std::endl;
    std::cout << "  tensor ptr: " << (void*)tensor2 << std::endl;
    std::cout << "  data ptr: " << data2 << std::endl;
    std::cout << "  data raw (hex): 0x" << std::hex << (uint64_t)data2 << std::dec << std::endl;
    
    // Debug the virtual memory range checks
    uint64_t base_offset = GGML_MMAP_VIRTUAL_MEMORY_BASE_OFFSET;
    uint64_t numa_increment = GGML_MMAP_VIRTUAL_MEMORY_NUMA_INCREMENT;
    uint64_t data2_addr = (uint64_t)data2;
    
    std::cout << "  Virtual memory range analysis:" << std::endl;
    std::cout << "    Base offset: 0x" << std::hex << base_offset << std::dec << std::endl;
    std::cout << "    NUMA increment: 0x" << std::hex << numa_increment << std::dec << std::endl;
    std::cout << "    Range 1: 0x" << std::hex << base_offset << " to 0x" << (base_offset + numa_increment) << std::dec << std::endl;
    std::cout << "    Range 2: 0x" << std::hex << (base_offset + numa_increment) << " to 0x" << (base_offset + 2*numa_increment) << std::dec << std::endl;
    
    bool in_range1 = (data2_addr >= base_offset && data2_addr < (base_offset + numa_increment));
    bool in_range2 = (data2_addr >= (base_offset + numa_increment) && data2_addr < (base_offset + 2*numa_increment));
    
    std::cout << "    Address 0x" << std::hex << data2_addr << std::dec << " in range 1: " << (in_range1 ? "YES" : "NO") << std::endl;
    std::cout << "    Address 0x" << std::hex << data2_addr << std::dec << " in range 2: " << (in_range2 ? "YES" : "NO") << std::endl;
    
    if (in_range2) {
        uint64_t adjusted_addr = data2_addr - numa_increment;
        std::cout << "    Would be adjusted to: 0x" << std::hex << adjusted_addr << std::dec << std::endl;
    }
    
    // Check if data address is in virtual memory range
    uint64_t data2_check_addr = (uint64_t)data2;
    in_virtual_range = (data2_check_addr >= GGML_MMAP_VIRTUAL_MEMORY_BASE_OFFSET && 
                       data2_check_addr < GGML_MMAP_VIRTUAL_MEMORY_BASE_OFFSET + GGML_MMAP_VIRTUAL_MEMORY_NUMA_INCREMENT);
    std::cout << "  in virtual memory range: " << (in_virtual_range ? "YES" : "NO") << std::endl;
    
    // Test writing to tensor
    if (data2) {
        std::cout << "  Testing write to second tensor..." << std::endl;
        try {
            ggml_set_f32_1d(tensor2, 0, 84.0f);
            float read_back = ggml_get_f32_1d(tensor2, 0);
            std::cout << "  Write/read test: wrote 84.0, read back " << read_back << std::endl;
        } catch (...) {
            std::cerr << "  ❌ Exception during write/read test!" << std::endl;
        }
    } else {
        std::cerr << "  ❌ Second tensor data is NULL!" << std::endl;
    }
    
    ggml_free(ctx2);
    std::cout << "  Second context freed" << std::endl;
}

static void debug_tensor_after_coordinator() {
    std::cout << "\n=== Debugging Tensor After Coordinator Use ===" << std::endl;
    
    // Get coordinator 
    struct ggml_numa_coordinator_manager * coordinator = ggml_numa_coordinator_manager_get_global(2, false);
    if (!coordinator) {
        std::cerr << "Failed to get coordinator" << std::endl;
        return;
    }
    
    std::cout << "✓ Coordinator acquired" << std::endl;
    
    // Create simple computation using coordinator
    struct ggml_init_params params = {
        /*.mem_size   =*/ 4 * 1024 * 1024,  // 4MB
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::cerr << "Failed to create context" << std::endl;
        return;
    }
    
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 16);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 16);
    
    std::cout << "Before coordinator computation:" << std::endl;
    std::cout << "  ggml_current_numa_node: " << ggml_current_numa_node << std::endl;
    std::cout << "  tensor a data: " << ggml_get_data(a) << std::endl;
    std::cout << "  tensor b data: " << ggml_get_data(b) << std::endl;
    
    // Initialize tensors
    for (int i = 0; i < ggml_nelements(a); i++) {
        ggml_set_f32_1d(a, i, 1.0f);
        ggml_set_f32_1d(b, i, 2.0f);
    }
    
    // Create computation graph
    struct ggml_tensor * result = ggml_mul_mat(ctx, a, b);
    struct ggml_cgraph * cgraph = ggml_new_graph(ctx);
    ggml_build_forward_expand(cgraph, result);
    
    // Submit to coordinator
    std::cout << "Submitting to coordinator..." << std::endl;
    int compute_result = ggml_numa_coordinator_manager_compute_graph(coordinator, cgraph);
    
    std::cout << "After coordinator computation:" << std::endl;
    std::cout << "  ggml_current_numa_node: " << ggml_current_numa_node << std::endl;
    
    // CRITICAL: Reset the current NUMA node to default after coordinator use
    std::cout << "  Resetting ggml_current_numa_node to -1..." << std::endl;
    ggml_current_numa_node = -1;
    std::cout << "  compute result: " << compute_result << std::endl;
    
    if (compute_result == 0) {
        float sample_result = ggml_get_f32_1d(result, 0);
        std::cout << "  computation result: " << sample_result << std::endl;
    }
    
    ggml_free(ctx);
    
    // NOW try to create a new context and tensor
    std::cout << "\nCreating new context after coordinator use..." << std::endl;
    std::cout << "Current ggml_current_numa_node before new context: " << ggml_current_numa_node << std::endl;
    
    struct ggml_init_params new_params = {
        /*.mem_size   =*/ 2 * 1024 * 1024,  // 2MB
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    
    struct ggml_context * new_ctx = ggml_init(new_params);
    if (!new_ctx) {
        std::cerr << "❌ Failed to create new context after coordinator!" << std::endl;
        return;
    }
    
    std::cout << "✓ New context created successfully" << std::endl;
    std::cout << "  ggml_current_numa_node: " << ggml_current_numa_node << std::endl;
    
    struct ggml_tensor * new_tensor = ggml_new_tensor_2d(new_ctx, GGML_TYPE_F32, 24, 24);
    if (!new_tensor) {
        std::cerr << "❌ Failed to create new tensor!" << std::endl;
        ggml_free(new_ctx);
        return;
    }
    
    void* new_data = ggml_get_data(new_tensor);
    std::cout << "New tensor after coordinator:" << std::endl;
    std::cout << "  tensor ptr: " << (void*)new_tensor << std::endl;
    std::cout << "  data ptr: " << new_data << std::endl;
    std::cout << "  data raw (hex): 0x" << std::hex << (uint64_t)new_data << std::dec << std::endl;
    
    if (new_data) {
        std::cout << "Testing write to new tensor..." << std::endl;
        try {
            ggml_set_f32_1d(new_tensor, 0, 123.0f);
            float read_back = ggml_get_f32_1d(new_tensor, 0);
            std::cout << "✓ Write/read test successful: wrote 123.0, read back " << read_back << std::endl;
        } catch (...) {
            std::cerr << "❌ Exception during write/read test!" << std::endl;
        }
    } else {
        std::cerr << "❌ New tensor data is NULL!" << std::endl;
    }
    
    ggml_free(new_ctx);
}

int main() {
    std::cout << "Tensor Data Allocation Debug Test" << std::endl;
    std::cout << "==================================" << std::endl;
    
    // Initialize backend
    llama_backend_init();
    ggml_numa_init(GGML_NUMA_STRATEGY_DISABLED);  // Disable NUMA
    
    std::cout << "Backend initialized (OpenMP disabled, NUMA disabled)" << std::endl;
    
    // Test 1: Basic tensor allocation
    debug_tensor_allocation();
    
    // Test 2: Tensor allocation after coordinator use
    debug_tensor_after_coordinator();
    
    std::cout << "\n=== Debug Test Complete ===" << std::endl;
    
    return 0;
}
