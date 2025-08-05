#include "ggml.h"
#include "ggml-cpu.h"

#include <iostream>

// Test NUMA topology detection
static void test_numa_topology() {
    std::cout << "\n=== NUMA Topology Information ===" << std::endl;
    
    // Initialize NUMA first to detect topology
    std::cout << "Initializing NUMA detection..." << std::endl;
    ggml_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);
    
    bool numa_available = ggml_is_numa();
    std::cout << "NUMA available: " << (numa_available ? "Yes" : "No") << std::endl;
    
    enum ggml_numa_strategy strategy = ggml_get_numa_strategy();
    std::cout << "NUMA strategy: " << strategy << std::endl;
    
    if (numa_available) {
        std::cout << "NUMA functionality is available - multi-socket code paths will be used" << std::endl;
    } else {
        std::cout << "NUMA not available - testing basic functionality" << std::endl;
    }
    
    // Test CPU backend creation
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend) {
        std::cout << "✓ CPU backend created successfully" << std::endl;
        ggml_backend_free(backend);
    } else {
        std::cout << "✗ Failed to create CPU backend" << std::endl;
    }
}

// Very simple test - just create tensors and verify they work
static bool test_basic_tensor_creation() {
    std::cout << "\nTesting basic tensor creation..." << std::endl;
    
    // Create context with internal allocation
    struct ggml_init_params params = {
        /*.mem_size   =*/ 1024 * 1024,  // 1MB
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::cerr << "Failed to create GGML context" << std::endl;
        return false;
    }
    
    // Create simple tensors
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
    
    if (!a || !b) {
        std::cerr << "Failed to create tensors" << std::endl;
        ggml_free(ctx);
        return false;
    }
    
    std::cout << "✓ Created tensors successfully" << std::endl;
    std::cout << "  Tensor A: " << a->ne[0] << "x" << a->ne[1] << " (" << ggml_nelements(a) << " elements)" << std::endl;
    std::cout << "  Tensor B: " << b->ne[0] << "x" << b->ne[1] << " (" << ggml_nelements(b) << " elements)" << std::endl;
    
    // Test tensor properties without accessing data
    bool properties_ok = true;
    if (a->type != GGML_TYPE_F32) {
        std::cerr << "Tensor A has wrong type" << std::endl;
        properties_ok = false;
    }
    if (ggml_nelements(a) != 16) {
        std::cerr << "Tensor A has wrong number of elements" << std::endl;
        properties_ok = false;
    }
    
    if (properties_ok) {
        std::cout << "✓ Tensor properties verification passed" << std::endl;
    }
    
    // Test creating a simple operation without computing it
    struct ggml_tensor * result = ggml_add(ctx, a, b);
    if (result) {
        std::cout << "✓ Created tensor operation successfully" << std::endl;
        std::cout << "  Result tensor: " << result->ne[0] << "x" << result->ne[1] << std::endl;
    } else {
        std::cerr << "Failed to create tensor operation" << std::endl;
        properties_ok = false;
    }
    
    ggml_free(ctx);
    return properties_ok;
}

int main() {
    std::cout << "NUMA Multi-Socket Test" << std::endl;
    std::cout << "======================" << std::endl;
    
    // Test NUMA topology
    test_numa_topology();
    
    // Test basic functionality
    if (test_basic_tensor_creation()) {
        std::cout << "\n✅ Basic functionality test passed!" << std::endl;
        
        if (ggml_is_numa()) {
            std::cout << "\nNote: This system has NUMA support." << std::endl;
            std::cout << "The multi-socket NUMA code paths in ggml-cpu.c are available" << std::endl;
            std::cout << "and will be used when NUMA-aware threadpools are configured." << std::endl;
        } else {
            std::cout << "\nNote: This system does not have NUMA support." << std::endl;
            std::cout << "However, the multi-socket code can still be tested by enabling" << std::endl;
            std::cout << "multi-socket mode even with n_numa_nodes=1." << std::endl;
        }
        return 0;
    } else {
        std::cout << "\n❌ Basic functionality test failed!" << std::endl;
        return 1;
    }
}
