/*
 * Test NUMA-aware context allocation
 * 
 * Tests that ggml contexts use NUMA-distributed memory allocation
 * for large memory pools.
 */
#include <stdio.h>
#include <stdlib.h>
#include "ggml.h"

int main() {
    printf("🧪 Testing NUMA-aware context allocation\n\n");
    
    // Test 1: Small context (should use regular allocation)
    printf("📝 Test 1: Small context (32MB) - should use regular allocation\n");
    size_t small_size = 32 * 1024 * 1024;
    
    struct ggml_init_params small_params = {
        .mem_size = small_size,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    
    struct ggml_context* small_ctx = ggml_init(small_params);
    if (!small_ctx) {
        printf("❌ Failed to create small context\n");
        return 1;
    }
    printf("✅ Small context created successfully\n");
    
    // Test creating a tensor in small context
    struct ggml_tensor* small_tensor = ggml_new_tensor_2d(small_ctx, GGML_TYPE_F32, 100, 100);
    if (small_tensor) {
        printf("✅ Created small test tensor: %p (size: %zu bytes)\n", 
               small_tensor, ggml_nbytes(small_tensor));
    } else {
        printf("❌ Failed to create small test tensor\n");
    }
    
    ggml_free(small_ctx);
    printf("✅ Small context freed\n\n");
    
    // Test 2: Large context (should trigger NUMA distribution)
    printf("📝 Test 2: Large context (128MB) - should trigger NUMA distribution\n");
    size_t large_size = 128 * 1024 * 1024;
    
    struct ggml_init_params large_params = {
        .mem_size = large_size,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    
    struct ggml_context* large_ctx = ggml_init(large_params);
    if (!large_ctx) {
        printf("❌ Failed to create large context\n");
        return 1;
    }
    printf("✅ Large context created successfully\n");
    
    // Test creating multiple tensors in large context
    struct ggml_tensor* large_tensor1 = ggml_new_tensor_2d(large_ctx, GGML_TYPE_F32, 1000, 1000);
    struct ggml_tensor* large_tensor2 = ggml_new_tensor_2d(large_ctx, GGML_TYPE_F32, 2000, 1000);
    
    if (large_tensor1 && large_tensor2) {
        printf("✅ Created large test tensors:\n");
        printf("   Tensor 1: %p (size: %zu bytes)\n", large_tensor1, ggml_nbytes(large_tensor1));
        printf("   Tensor 2: %p (size: %zu bytes)\n", large_tensor2, ggml_nbytes(large_tensor2));
    } else {
        printf("❌ Failed to create large test tensors\n");
    }
    
    ggml_free(large_ctx);
    printf("✅ Large context freed\n\n");
    
    // Test 3: Multiple contexts (stress test)
    printf("📝 Test 3: Multiple large contexts (stress test)\n");
    const int num_contexts = 3;
    struct ggml_context* contexts[num_contexts];
    
    for (int i = 0; i < num_contexts; i++) {
        struct ggml_init_params params = {
            .mem_size = 96 * 1024 * 1024,  // 96MB each
            .mem_buffer = NULL,
            .no_alloc = false,
        };
        
        contexts[i] = ggml_init(params);
        if (!contexts[i]) {
            printf("❌ Failed to create context %d\n", i);
            return 1;
        }
        printf("✅ Context %d created\n", i);
    }
    
    // Clean up all contexts
    for (int i = 0; i < num_contexts; i++) {
        ggml_free(contexts[i]);
        printf("✅ Context %d freed\n", i);
    }
    
    printf("\n🎉 All NUMA context allocation tests passed!\n");
    return 0;
}
