#include "../common/common.h"
#include "../ggml/include/ggml.h"

#include <iostream>

int main() {
    std::cout << "=== GPU-NUMA Thread Binding Test ===\n";
    
    // Test 1: Test GPU-NUMA detection
    std::cout << "\n--- Testing GPU-NUMA Detection ---\n";
    auto gpu_infos = detect_gpu_numa_affinity();
    
    std::cout << "Detected " << gpu_infos.size() << " GPU(s)\n";
    for (size_t i = 0; i < gpu_infos.size(); i++) {
        const auto& gpu = gpu_infos[i];
        std::cout << "GPU " << i << ":\n";
        std::cout << "  - ID: " << gpu.gpu_id << "\n";
        std::cout << "  - Device: " << gpu.device_name << "\n";
        std::cout << "  - NUMA node: " << gpu.numa_node << "\n";
        std::cout << "  - Virtual: " << (gpu.is_virtual_gpu ? "yes" : "no") << "\n";
        std::cout << "  - Backend available: " << (gpu.backend_available ? "yes" : "no") << "\n";
        if (gpu.backend_available) {
            std::cout << "  - Backend name: " << gpu.backend_name << "\n";
        }
        std::cout << "  - Local CPU cores: " << gpu.local_cpu_cores.size() << "\n";
    }
    
    // Test 2: Test binding function for each detected GPU
    std::cout << "\n--- Testing GPU-NUMA Thread Binding ---\n";
    for (size_t i = 0; i < gpu_infos.size(); i++) {
        if (gpu_infos[i].backend_available) {
            std::cout << "Testing binding for GPU " << gpu_infos[i].gpu_id << "...\n";
            bool success = bind_current_thread_to_gpu_numa(gpu_infos[i].gpu_id);
            std::cout << "  Result: " << (success ? "✅ SUCCESS" : "❌ FAILED") << "\n";
            
#ifdef GGML_NUMA_MIRROR
            extern __thread int ggml_current_numa_node;
            std::cout << "  Current NUMA node for tensor_data(): " << ggml_current_numa_node << "\n";
#endif
        }
    }
    
    // Test 3: Test tensor_data() NUMA awareness
#ifdef GGML_NUMA_MIRROR
    std::cout << "\n--- Testing tensor_data() NUMA Awareness ---\n";
    
    // Create a simple tensor to test with
    struct ggml_init_params params = {};
    params.mem_size = 1024*1024;  // 1MB
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    
    struct ggml_context* ctx = ggml_init(params);
    if (ctx) {
        struct ggml_tensor* test_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100);
        if (test_tensor) {
            void* data_ptr = tensor_data(test_tensor);
            std::cout << "tensor_data() returned pointer: " << data_ptr << "\n";
            std::cout << "✅ tensor_data() function accessible\n";
        } else {
            std::cout << "❌ Failed to create test tensor\n";
        }
        ggml_free(ctx);
    } else {
        std::cout << "❌ Failed to create GGML context\n";
    }
#else
    std::cout << "\n--- NUMA Mirroring Not Enabled ---\n";
    std::cout << "GGML_NUMA_MIRROR not defined - standard tensor_data() will be used\n";
#endif
    
    std::cout << "\n✅ GPU-NUMA thread binding test completed\n";
    return 0;
}
