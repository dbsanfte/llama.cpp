#include <stdio.h>
#include "ggml.h"
#include "ggml-cpu.h"

int main() {
    printf("🔍 Testing NUMA tensor data access during execution\n");
    printf("=================================================\n");
    
    // Initialize NUMA with MIRROR strategy
    printf("🪞 Initializing NUMA with MIRROR strategy...\n");
    ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);
    
#ifdef GGML_NUMA_MIRROR
    // Check if mirroring should be active
    extern bool ggml_numa_should_mirror(void);
    bool should_mirror = ggml_numa_should_mirror();
    printf("🔍 ggml_numa_should_mirror(): %s\n", should_mirror ? "TRUE" : "FALSE");
    
    extern int ggml_numa_node_count(void);
    int node_count = ggml_numa_node_count();
    printf("🔍 ggml_numa_node_count(): %d\n", node_count);
    
    // Create a test context and tensor to see if data is mirrored
    printf("\n🧪 Creating large tensor to trigger mirroring...\n");
    struct ggml_init_params params = {
        .mem_size = 100 * 1024 * 1024, // 100MB - large enough to trigger NUMA distribution
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        printf("❌ Failed to create context\n");
        return 1;
    }
    
    // Create large tensors (like in our ADD test)
    struct ggml_tensor* src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 2048); // 16MB
    struct ggml_tensor* src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 2048); // 16MB
    struct ggml_tensor* dst = ggml_add(ctx, src0, src1);
    
    if (!src0 || !src1 || !dst) {
        printf("❌ Failed to create tensors\n");
        return 1;
    }
    
    printf("📊 Created tensors: src0=%p, src1=%p, dst=%p\n", src0, src1, dst);
    
    // Initialize data
    float* src0_data = (float*)ggml_get_data(src0);
    float* src1_data = (float*)ggml_get_data(src1);
    for (int i = 0; i < 100; i++) { // Just initialize a few values
        src0_data[i] = 1.0f;
        src1_data[i] = 2.0f;
    }
    
    // Check the tensor data pointers for different NUMA nodes
    printf("\n🔍 src0 tensor data pointers:\n");
    for (int i = 0; i < node_count && i < 8; i++) {
        void* data_ptr = src0->__data[i];
        printf("   Node %d: %p\n", i, data_ptr);
    }
    
    printf("\n🔍 src1 tensor data pointers:\n");
    for (int i = 0; i < node_count && i < 8; i++) {
        void* data_ptr = src1->__data[i];
        printf("   Node %d: %p\n", i, data_ptr);
    }
    
    printf("\n🔍 dst tensor data pointers:\n");
    for (int i = 0; i < node_count && i < 8; i++) {
        void* data_ptr = dst->__data[i];
        printf("   Node %d: %p\n", i, data_ptr);
    }
    
    // Test tensor_data() function from different simulated NUMA contexts
    extern __thread int ggml_current_numa_node;
    printf("\n🔍 Testing tensor_data() from different NUMA contexts:\n");
    
    for (int node = 0; node < node_count; node++) {
        ggml_current_numa_node = node;
        void* tensor_data_ptr = tensor_data(src0);
        printf("   From node %d: tensor_data(src0) = %p\n", node, tensor_data_ptr);
    }
    
    // Reset
    ggml_current_numa_node = -1;
    
    ggml_free(ctx);
    
#else
    printf("❌ GGML_NUMA_MIRROR not defined - mirroring not compiled in\n");
#endif
    
    return 0;
}
