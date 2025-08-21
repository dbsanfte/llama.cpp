/**
 * Minimal NUMA coordinator debug test
 */

#include <cstdio>
#include "ggml.h"
#include "ggml-cpu.h"

extern "C" {
    extern bool ggml_numa_simple_coordinator_is_initialized(void);
}

int main() {
    printf("=== NUMA Coordinator Debug Test ===\n");
    
    printf("1. Before NUMA init:\n");
    bool coord_init_before = ggml_numa_simple_coordinator_is_initialized();
    printf("   Coordinator initialized: %s\n", coord_init_before ? "YES" : "NO");
    
    printf("\n2. Calling ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR)...\n");
    ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);
    
    printf("\n3. After NUMA init:\n");
    bool coord_init_after = ggml_numa_simple_coordinator_is_initialized();
    printf("   Coordinator initialized: %s\n", coord_init_after ? "YES" : "NO");
    
    if (!coord_init_after) {
        printf("\n❌ NUMA coordinator failed to initialize!\n");
        printf("   This explains why NUMA performance is poor - we're not using NUMA at all.\n");
        return 1;
    } else {
        printf("\n✅ NUMA coordinator successfully initialized!\n");
        return 0;
    }
}
