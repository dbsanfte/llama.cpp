/**
 * NUMA Coordinator Integration Test
 * 
 * Tests the integration between common_params, ggml-cpu NUMA functions,
 * and the NUMA coordinator to ensure parameters flow correctly.
 */

#include <iostream>
#include <cstring>
#include <cassert>

#include "ggml.h"
#include "ggml-cpu.h"
#include "../common/common.h"  // For cpu_params and conversion functions

// Test helper to print NUMA state
static void print_numa_state(const char* test_name) {
    std::cout << test_name << ":\n";
    std::cout << "   - NUMA enabled: " << (ggml_is_numa() ? "Yes" : "No") << "\n";
    std::cout << "   - Node count: " << ggml_numa_node_count() << "\n"; 
    std::cout << "   - Strategy: " << (int)ggml_get_numa_strategy() << "\n";
}

int main() {
    std::cout << "Testing NUMA Coordinator Integration\n";
    std::cout << "=====================================\n";
    
    // Create CPU params similar to what applications would use
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
    
    std::cout << "1. Testing basic NUMA initialization...\n";
    ggml_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);
    print_numa_state("   Basic init");
    
    std::cout << "\n2. Testing threadpool parameter initialization...\n";
    ggml_numa_init_with_threadpool_params(GGML_NUMA_STRATEGY_DISTRIBUTE, &tpp);
    print_numa_state("   With threadpool params");
    
    std::cout << "\n3. Testing isolation strategy...\n";
    ggml_numa_init_with_threadpool_params(GGML_NUMA_STRATEGY_ISOLATE, &tpp);
    print_numa_state("   Isolation strategy");
    
    std::cout << "\n4. Testing disabled strategy...\n";
    ggml_numa_init_with_threadpool_params(GGML_NUMA_STRATEGY_DISABLED, &tpp);
    print_numa_state("   Disabled strategy");
    
    std::cout << "\n5. Testing with node isolation...\n";
    ggml_numa_init_with_node(GGML_NUMA_STRATEGY_ISOLATE, 0);
    print_numa_state("   Node isolation");
    
    // Test parameter validation
    std::cout << "\n6. Parameter validation tests...\n";
    
    // Test that threadpool params are correctly converted
    assert(tpp.n_threads == cpuparams.n_threads);
    assert(tpp.numa_aware == cpuparams.numa_aware);
    assert(tpp.allow_numa_override == cpuparams.allow_numa_override);
    assert(tpp.warn_on_numa_override == cpuparams.warn_on_numa_override);
    assert(tpp.prio == cpuparams.priority);  // Note: prio in tpp, priority in cpuparams
    assert(tpp.poll == cpuparams.poll);
    assert(tpp.strict_cpu == cpuparams.strict_cpu);
    std::cout << "   ✓ Parameter conversion validation passed\n";
    
    // Test NUMA strategy consistency
    ggml_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);
    assert(ggml_get_numa_strategy() == GGML_NUMA_STRATEGY_DISTRIBUTE);
    
    ggml_numa_init(GGML_NUMA_STRATEGY_ISOLATE);
    assert(ggml_get_numa_strategy() == GGML_NUMA_STRATEGY_ISOLATE);
    
    ggml_numa_init(GGML_NUMA_STRATEGY_DISABLED);
    assert(ggml_get_numa_strategy() == GGML_NUMA_STRATEGY_DISABLED);
    std::cout << "   ✓ Strategy consistency validation passed\n";
    
    // Test node count consistency
    int node_count = ggml_numa_node_count();
    assert(node_count >= 1);  // Should always be at least 1
    std::cout << "   ✓ Node count validation passed (detected " << node_count << " nodes)\n";
    
    std::cout << "\n✅ NUMA Coordinator Integration Test Complete\n";
    std::cout << "All tests passed successfully!\n";
    
    return 0;
}
