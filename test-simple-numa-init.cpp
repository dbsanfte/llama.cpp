#include <stdio.h>

extern "C" {
#include "ggml.h"
#include "ggml-cpu.h"
    bool ggml_numa_simple_coordinator_is_initialized(void);
}

int main() {
    printf("Testing simple NUMA initialization...\n");
    
    // Test direct coordinator initialization
    
    printf("Before init: coordinator initialized = %s\n", 
           ggml_numa_simple_coordinator_is_initialized() ? "YES" : "NO");
    
    // Initialize NUMA
    printf("Calling ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR)...\n");
    ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);
    
    printf("After init: coordinator initialized = %s\n", 
           ggml_numa_simple_coordinator_is_initialized() ? "YES" : "NO");
    
    printf("NUMA enabled: %s\n", ggml_is_numa() ? "YES" : "NO");
    printf("NUMA node count: %d\n", ggml_numa_node_count());
    
    return 0;
}
