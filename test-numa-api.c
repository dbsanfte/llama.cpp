#include <stdio.h>
#include "ggml-numa-allocator.h"

int main() {
    printf("Testing NUMA allocator API...\n");
    
    // Test allocation
    void* ptr = ggml_numa_alloc_context_memory(1024, NULL);
    if (ptr) {
        printf("✅ ggml_numa_alloc_context_memory works\n");
        
        // Test check function
        bool is_numa_allocated = ggml_numa_is_numa_allocated(ptr);
        printf("✅ ggml_numa_is_numa_allocated returned: %s\n", is_numa_allocated ? "true" : "false");
        
        // Test free
        ggml_numa_free(ptr);
        printf("✅ ggml_numa_free works\n");
    } else {
        printf("❌ ggml_numa_alloc_context_memory failed\n");
    }
    
    return 0;
}
