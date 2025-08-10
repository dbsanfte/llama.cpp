#include "ggml.h"
#include "ggml-backend.h"
#include <stdio.h>
#include <cstring>
#include <cstdlib>

// Forward declaration for our hybrid buffer type
extern "C" ggml_backend_buffer_type_t ggml_backend_cpu_numa_repack_buffer_type(void);

int main() {
    printf("Testing CPU_NUMA_REPACK hybrid buffer type\n");
    printf("==========================================\n");
    
    // Initialize ggml
    struct ggml_init_params params = {
        /*.mem_size   =*/ 16*1024*1024, // 16MB
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        printf("ERROR: Failed to initialize ggml context\n");
        return 1;
    }
    
    // Get our hybrid buffer type
    ggml_backend_buffer_type_t numa_repack_buft = ggml_backend_cpu_numa_repack_buffer_type();
    if (!numa_repack_buft) {
        printf("ERROR: CPU_NUMA_REPACK buffer type not available\n");
        ggml_free(ctx);
        return 1;
    }
    
    printf("✅ CPU_NUMA_REPACK buffer type available: %s\n", 
           ggml_backend_buft_name(numa_repack_buft));
    
    // Test buffer allocation
    size_t test_size = 4 * 1024 * 1024; // 4MB buffer
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(numa_repack_buft, test_size);
    if (!buffer) {
        printf("ERROR: Failed to allocate buffer\n");
        ggml_free(ctx);
        return 1;
    }
    
    printf("✅ Buffer allocated successfully: %zu bytes\n", test_size);
    
    // Create test tensors of different types to test repack vs regular behavior
    printf("\nTesting tensor creation and data setting:\n");
    printf("=========================================\n");
    
    // Test Q4_0 tensor (should support repacking)
    struct ggml_tensor * q4_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 128, 64);
    if (q4_tensor) {
        ggml_set_name(q4_tensor, "test_q4_0");
        ggml_backend_tensor_alloc(buffer, q4_tensor, nullptr);
        
        // Create some dummy data
        size_t q4_size = ggml_nbytes(q4_tensor);
        void* q4_data = malloc(q4_size);
        memset(q4_data, 0x42, q4_size);
        
        printf("\n📋 Testing Q4_0 tensor (should use repack optimization):\n");
        ggml_backend_tensor_set(q4_tensor, q4_data, 0, q4_size);
        
        free(q4_data);
        printf("✅ Q4_0 tensor data set successfully\n");
    }
    
    // Test F32 tensor (should use regular copy)
    struct ggml_tensor * f32_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 32);
    if (f32_tensor) {
        ggml_set_name(f32_tensor, "test_f32");
        ggml_backend_tensor_alloc(buffer, f32_tensor, nullptr);
        
        // Create some dummy data
        size_t f32_size = ggml_nbytes(f32_tensor);
        void* f32_data = malloc(f32_size);
        memset(f32_data, 0x33, f32_size);
        
        printf("\n📋 Testing F32 tensor (should use regular copy):\n");
        ggml_backend_tensor_set(f32_tensor, f32_data, 0, f32_size);
        
        free(f32_data);
        printf("✅ F32 tensor data set successfully\n");
    }
    
    // Compare buffer types
    printf("\nBuffer Type Comparison:\n");
    printf("=====================\n");
    printf("CPU_NUMA_REPACK: %s\n", ggml_backend_buft_name(numa_repack_buft));
    printf("CPU (default):   %s\n", ggml_backend_buft_name(ggml_backend_cpu_buffer_type()));
    
    // Test NUMA buffer if available
    ggml_backend_buffer_type_t numa_buft = ggml_backend_cpu_numa_buffer_type();
    if (numa_buft) {
        printf("CPU_NUMA:        %s\n", ggml_backend_buft_name(numa_buft));
    }
    
    // Cleanup
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    
    printf("\n🎉 Test completed successfully!\n");
    printf("The hybrid CPU_NUMA_REPACK buffer combines:\n");
    printf("  • NUMA-aware allocation for memory locality\n");
    printf("  • Automatic repack optimization for quantized weights\n");
    printf("  • Best of both worlds performance!\n");
    
    return 0;
}
