#include "ggml.h"
#include "ggml-backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef GGML_NUMA_MIRROR
#include <numa.h>
#endif

// Test to verify CPU buffer allocation (with NUMA enhancement if available)
int main() {
    printf("=======================================================\n");
    printf("Testing CPU Buffer Enhancement\n");
    printf("=======================================================\n");
    
#ifdef GGML_NUMA_MIRROR
    printf("NUMA mirroring enabled in build\n");
    
    // Check if NUMA is available on the system
    if (numa_available() >= 0) {
        printf("NUMA support is available on this system\n");
        printf("Number of NUMA nodes: %d\n", numa_num_configured_nodes());
    } else {
        printf("NUMA support is not available on this system\n");
    }
#else
    printf("NUMA mirroring not enabled in build\n");
#endif
    
    // Get CPU buffer type
    ggml_backend_buffer_type_t cpu_buffer_type = ggml_backend_cpu_buffer_type();
    
    if (!cpu_buffer_type) {
        printf("FAILED: Could not get CPU buffer type\n");
        return 1;
    }
    
    printf("Successfully obtained CPU buffer type\n");
    
    // Test basic buffer allocation with different sizes
    size_t test_sizes[] = {1024, 4096, 8192, 16384};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    for (int i = 0; i < num_tests; i++) {
        size_t size = test_sizes[i];
        printf("Testing allocation of %zu bytes...\n", size);
        
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(cpu_buffer_type, size);
        
        if (!buffer) {
            printf("FAILED: Could not allocate buffer of size %zu\n", size);
            return 1;
        }
        
        // Get buffer base pointer
        void* data = ggml_backend_buffer_get_base(buffer);
        if (!data) {
            printf("FAILED: Buffer base pointer is null for size %zu\n", size);
            ggml_backend_buffer_free(buffer);
            return 1;
        }
        
        // Test that we can write to the buffer
        memset(data, 0xFF, size);
        
        // Verify the write
        unsigned char* byte_data = (unsigned char*)data;
        for (size_t j = 0; j < size; j++) {
            if (byte_data[j] != 0xFF) {
                printf("FAILED: Memory write verification failed at offset %zu\n", j);
                ggml_backend_buffer_free(buffer);
                return 1;
            }
        }
        
        // Free the buffer
        ggml_backend_buffer_free(buffer);
        
        printf("SUCCESS: %zu byte buffer allocation and access works\n", size);
    }
    
    printf("=======================================================\n");
    printf("ALL CPU BUFFER TESTS PASSED\n");
    printf("=======================================================\n");
    
    return 0;
}
