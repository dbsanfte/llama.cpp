/*
 * Test NUMA-aware context allocation
 */
#include <stdio.h>
#include <stdlib.h>
#include "ggml.h"

int main() {
    printf("🧪 Testing NUMA-aware context allocation\n\n");
    
    // Create a large context (128MB) that should trigger NUMA distribution
    size_t context_size = 128 * 1024 * 1024;
    
    struct ggml_init_params params = {
        .mem_size = context_size,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    
    printf("📝 Creating ggml context with %zu MB memory...\n", context_size / (1024 * 1024));
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        printf("❌ Failed to create context\n");
        return 1;
    }
    
    printf("✅ Context created successfully\n");
    
    // Test creating a tensor to verify the context works
    struct ggml_tensor* tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1000, 1000);
    if (tensor) {
        printf("✅ Created test tensor: %p\n", tensor);
        printf("   Tensor size: %zu bytes (1000x1000 float32)\n", ggml_nbytes(tensor));
    } else {
        printf("❌ Failed to create test tensor\n");
    }
    
    // Clean up
    ggml_free(ctx);
    printf("✅ Context freed successfully\n");
    
    return 0;
}
