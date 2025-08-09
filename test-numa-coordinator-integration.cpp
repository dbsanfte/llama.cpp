#include <stdio.h>
#include "ggml-cpu.h"
#include "ggml.h"

int main() {
    printf("Testing NUMA Coordinator Integration\n");
    printf("=====================================\n");
    
    // Create CPU params similar to what llama-server would use
    struct cpu_params cpuparams;
    cpuparams.n_threads = 4;
    cpuparams.numa_aware = true;
    cpuparams.use_hyperthreading = true;
    cpuparams.use_efficiency_cores = true;
    cpuparams.strict_cpu = false;
    cpuparams.priority = GGML_SCHED_PRIO_NORMAL;
    cpuparams.poll = 50;
    cpuparams.mask_valid = false;
    cpuparams.allow_numa_override = true;
    cpuparams.warn_on_numa_override = true;
    
    // Convert to threadpool params
    struct ggml_threadpool_params tpp = ggml_threadpool_params_from_cpu_params(cpuparams);
    
    printf("1. Testing basic NUMA initialization...\n");
    ggml_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);
    printf("   - NUMA enabled: %s\n", ggml_is_numa() ? "Yes" : "No");
    printf("   - Node count: %d\n", ggml_numa_node_count());
    printf("   - Strategy: %d\n", (int)ggml_get_numa_strategy());
    
    printf("\n2. Testing threadpool parameter initialization...\n");
    ggml_numa_init_with_threadpool_params(GGML_NUMA_STRATEGY_DISTRIBUTE, &tpp);
    printf("   - NUMA enabled: %s\n", ggml_is_numa() ? "Yes" : "No");
    printf("   - Node count: %d\n", ggml_numa_node_count());
    printf("   - Strategy: %d\n", (int)ggml_get_numa_strategy());
    
    printf("\n3. Testing isolation strategy...\n");
    ggml_numa_init_with_threadpool_params(GGML_NUMA_STRATEGY_ISOLATE, &tpp);
    printf("   - NUMA enabled: %s\n", ggml_is_numa() ? "Yes" : "No");
    printf("   - Node count: %d\n", ggml_numa_node_count());
    printf("   - Strategy: %d\n", (int)ggml_get_numa_strategy());
    
    printf("\n4. Testing disabled strategy...\n");
    ggml_numa_init_with_threadpool_params(GGML_NUMA_STRATEGY_DISABLED, &tpp);
    printf("   - NUMA enabled: %s\n", ggml_is_numa() ? "Yes" : "No");
    printf("   - Node count: %d\n", ggml_numa_node_count());
    printf("   - Strategy: %d\n", (int)ggml_get_numa_strategy());
    
    printf("\n✅ NUMA Coordinator Integration Test Complete\n");
    return 0;
}
