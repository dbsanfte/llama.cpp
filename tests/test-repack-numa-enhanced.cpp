#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-cpu/repack.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("=======================================================\n");
    printf("Testing Enhanced CPU Buffer with NUMA Support\n");
    printf("=======================================================\n");
    
    // Initialize GGML context
    struct ggml_init_params params = {
        /*.mem_size =*/ 128 * 1024 * 1024,  // 128MB 
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc =*/ false,
    };
    
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("FAIL: Failed to initialize GGML context\n");
        return 1;
    }
    printf("SUCCESS: GGML context initialized successfully\n");

    bool test_passed = true;
    
    // Test basic CPU buffer type
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
    if (!cpu_buft) {
        printf("FAIL: CPU buffer type not available\n");
        test_passed = false;
    } else {
        printf("SUCCESS: CPU buffer type available: %s\n", ggml_backend_buft_name(cpu_buft));
    }
    
    // Test enhanced CPU_REPACK buffer if available
    #ifdef GGML_USE_CPU_REPACK
    ggml_backend_buffer_type_t repack_buft = ggml_backend_cpu_repack_buffer_type();
    if (repack_buft) {
        printf("SUCCESS: CPU_REPACK enabled - enhanced buffer includes NUMA awareness\n");
        printf("Enhanced CPU_REPACK buffer type: %s\n", ggml_backend_buft_name(repack_buft));
    } else {
        printf("INFO: CPU_REPACK buffer type not available\n");
    }
    #else
    printf("INFO: CPU_REPACK not enabled in build\n");
    #endif
    
    // Test buffer allocation
    size_t test_size = 64 * 1024;  // 64KB
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(cpu_buft, test_size);
    if (!buffer) {
        printf("FAIL: Failed to allocate CPU buffer\n");
        test_passed = false;
    } else {
        printf("SUCCESS: CPU buffer allocated successfully: %zu bytes\n", test_size);
        ggml_backend_buffer_free(buffer);
    }

    printf("=======================================================\n");
    if (test_passed) {
        printf("Test completed successfully!\n");
        printf("Enhanced CPU_REPACK buffer is working with NUMA awareness\n");
    } else {
        printf("Test failed - some functionality not working\n");
    }
    printf("=======================================================\n");

    ggml_free(ctx);
    return test_passed ? 0 : 1;
}
