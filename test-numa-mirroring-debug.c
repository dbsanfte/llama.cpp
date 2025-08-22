#include <stdio.h>
#include "ggml.h"
#include "ggml-cpu.h"

int main() {
    printf("🔍 Testing NUMA mirroring state\n");
    printf("===============================\n");
    
    // Initialize NUMA with MIRROR strategy
    printf("🪞 Initializing NUMA with MIRROR strategy...\n");
    ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);
    
    // Check what the current state is
    enum ggml_numa_strategy strategy = ggml_get_numa_strategy();
    printf("📊 Current NUMA strategy: %d\n", strategy);
    printf("   DISABLED=0, ISOLATE=2, NUMACTL=3, MIRROR=4\n");
    
#ifdef GGML_NUMA_MIRROR
    // Check if mirroring should be active
    extern bool ggml_numa_should_mirror(void);
    bool should_mirror = ggml_numa_should_mirror();
    printf("🔍 ggml_numa_should_mirror(): %s\n", should_mirror ? "TRUE" : "FALSE");
    
    extern int ggml_numa_node_count(void);
    int node_count = ggml_numa_node_count();
    printf("🔍 ggml_numa_node_count(): %d\n", node_count);
    
    // Create a test context and tensor to see if data is mirrored
    printf("\n🧪 Testing tensor data mirroring...\n");
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024, // 1MB
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        printf("❌ Failed to create context\n");
        return 1;
    }
    
    struct ggml_tensor* test_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    if (!test_tensor) {
        printf("❌ Failed to create tensor\n");
        return 1;
    }
    
    // Check the tensor data pointers for different NUMA nodes
    printf("🔍 Tensor data pointers:\n");
    for (int i = 0; i < node_count && i < 8; i++) {
        void* data_ptr = test_tensor->__data[i];
        printf("   Node %d: %p\n", i, data_ptr);
    }
    
    // Check tensor_data() function behavior
    void* tensor_data_ptr = tensor_data(test_tensor);
    printf("🔍 tensor_data() returns: %p\n", tensor_data_ptr);
    
    ggml_free(ctx);
    
#else
    printf("❌ GGML_NUMA_MIRROR not defined - mirroring not compiled in\n");
#endif
    
    return 0;
}
