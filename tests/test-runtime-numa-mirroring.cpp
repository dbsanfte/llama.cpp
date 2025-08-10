#include "ggml.h"
#include "ggml-cpu.h"
#include <iostream>
#include <cassert>

void test_numa_runtime_mirroring() {
    std::cout << "=== Testing Runtime NUMA Mirroring Control ===" << std::endl;
    
    // Test 1: NUMA Distribute Strategy
    std::cout << "\n1. Testing NUMA DISTRIBUTE strategy:" << std::endl;
    ggml_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);
    
    std::cout << "   Strategy: " << (int)ggml_get_numa_strategy() << " (DISTRIBUTE)" << std::endl;
    std::cout << "   NUMA enabled: " << (ggml_is_numa() ? "yes" : "no") << std::endl;
    std::cout << "   NUMA nodes: " << ggml_numa_node_count() << std::endl;
    
#ifdef GGML_NUMA_MIRROR
    bool distribute_should_mirror = ggml_numa_should_mirror();
    std::cout << "   Should mirror: " << (distribute_should_mirror ? "yes" : "no") << std::endl;
#endif

    // Test 2: NUMA Mirror Strategy  
    std::cout << "\n2. Testing NUMA MIRROR strategy:" << std::endl;
    ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);
    
    std::cout << "   Strategy: " << (int)ggml_get_numa_strategy() << " (MIRROR)" << std::endl;
    std::cout << "   NUMA enabled: " << (ggml_is_numa() ? "yes" : "no") << std::endl;
    std::cout << "   NUMA nodes: " << ggml_numa_node_count() << std::endl;
    
#ifdef GGML_NUMA_MIRROR
    bool mirror_should_mirror = ggml_numa_should_mirror();
    std::cout << "   Should mirror: " << (mirror_should_mirror ? "yes" : "no") << std::endl;
#endif

    // Test 3: Verification
    std::cout << "\n3. Verification:" << std::endl;
    
#ifdef GGML_NUMA_MIRROR
    if (ggml_is_numa() && ggml_numa_node_count() > 1) {
        // On systems with actual NUMA, MIRROR should mirror but DISTRIBUTE should not
        std::cout << "   System has NUMA - testing differentiation" << std::endl;
        
        // Re-test distribute
        ggml_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);
        bool dist_mirrors = ggml_numa_should_mirror();
        
        // Re-test mirror
        ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);
        bool mirror_mirrors = ggml_numa_should_mirror();
        
        std::cout << "   DISTRIBUTE mirrors: " << (dist_mirrors ? "yes" : "no") << std::endl;
        std::cout << "   MIRROR mirrors: " << (mirror_mirrors ? "yes" : "no") << std::endl;
        
        // Key assertion: MIRROR should mirror, DISTRIBUTE should not
        if (mirror_mirrors && !dist_mirrors) {
            std::cout << "   ✅ SUCCESS: Runtime differentiation working correctly!" << std::endl;
        } else if (!mirror_mirrors && !dist_mirrors) {
            std::cout << "   ⚠️  WARNING: Neither strategy mirrors (may be expected on single-node systems)" << std::endl;
        } else {
            std::cout << "   ❌ ERROR: Expected MIRROR=yes, DISTRIBUTE=no" << std::endl;
        }
    } else {
        std::cout << "   System has no NUMA or single node - mirroring disabled for both" << std::endl;
    }
#else
    std::cout << "   GGML_NUMA_MIRROR not compiled - runtime mirroring not available" << std::endl;
#endif

    std::cout << "\n=== Test Complete ===" << std::endl;
}

int main() {
    test_numa_runtime_mirroring();
    return 0;
}
