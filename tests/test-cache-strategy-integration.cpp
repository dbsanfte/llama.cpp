#include "ggml.h"
#include "ggml-cpu.h"
#include <iostream>
#include <cassert>

// Cache strategy constants (matching the NUMA buffer)
#define TEST_CACHE_STRATEGY_DISABLED 0
#define TEST_CACHE_STRATEGY_EAGER    1
#define TEST_CACHE_STRATEGY_LAZY     2
#define TEST_CACHE_STRATEGY_DELTA    3
#define TEST_CACHE_STRATEGY_PARTIAL  4

void test_cache_strategy_integration() {
    std::cout << "=== Testing Cache Strategy Integration ===" << std::endl;
    
    // Test each cache strategy with different NUMA modes
    const struct {
        const char* strategy_name;
        int strategy_value;
    } strategies[] = {
        {"DISABLED", TEST_CACHE_STRATEGY_DISABLED},
        {"EAGER", TEST_CACHE_STRATEGY_EAGER},
        {"LAZY", TEST_CACHE_STRATEGY_LAZY},
        {"DELTA", TEST_CACHE_STRATEGY_DELTA},
        {"PARTIAL", TEST_CACHE_STRATEGY_PARTIAL}
    };
    
    std::cout << "\n1. Testing cache strategy setting and retrieval:" << std::endl;
    
    for (int i = 0; i < 5; i++) {
        // Set the cache strategy
        ggml_numa_set_cache_strategy(strategies[i].strategy_value);
        
        // Retrieve it
        int retrieved = ggml_numa_get_cache_strategy();
        
        std::cout << "   " << strategies[i].strategy_name 
                  << ": Set=" << strategies[i].strategy_value 
                  << ", Got=" << retrieved 
                  << (retrieved == strategies[i].strategy_value ? " ✓" : " ❌") << std::endl;
        
        assert(retrieved == strategies[i].strategy_value);
    }
    
    std::cout << "\n2. Testing interaction with NUMA mirroring:" << std::endl;
    
    // Test DISTRIBUTE with different cache strategies
    std::cout << "\n   DISTRIBUTE mode:" << std::endl;
    ggml_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);
    
    for (int i = 0; i < 5; i++) {
        ggml_numa_set_cache_strategy(strategies[i].strategy_value);
        
#ifdef GGML_NUMA_MIRROR
        bool should_mirror = ggml_numa_should_mirror();
        std::cout << "     " << strategies[i].strategy_name 
                  << ": Mirror=" << (should_mirror ? "yes" : "no");
        
        // DISTRIBUTE should never mirror regardless of cache strategy
        if (!should_mirror) {
            std::cout << " ✓" << std::endl;
        } else {
            std::cout << " ❌" << std::endl;
        }
        assert(!should_mirror);
#else
        std::cout << "     " << strategies[i].strategy_name 
                  << ": NUMA_MIRROR not compiled" << std::endl;
#endif
    }
    
    // Test MIRROR with different cache strategies
    std::cout << "\n   MIRROR mode:" << std::endl;
    ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);
    
    for (int i = 0; i < 5; i++) {
        ggml_numa_set_cache_strategy(strategies[i].strategy_value);
        
#ifdef GGML_NUMA_MIRROR
        bool should_mirror = ggml_numa_should_mirror();
        std::cout << "     " << strategies[i].strategy_name 
                  << ": Mirror=" << (should_mirror ? "yes" : "no");
        
        // MIRROR should mirror when NUMA is available (which it is in our test)
        if (ggml_is_numa() && ggml_numa_node_count() > 1) {
            if (should_mirror) {
                std::cout << " ✓" << std::endl;
            } else {
                std::cout << " ❌" << std::endl;
            }
            assert(should_mirror);
        } else {
            std::cout << " (single node - expected)" << std::endl;
        }
#else
        std::cout << "     " << strategies[i].strategy_name 
                  << ": NUMA_MIRROR not compiled" << std::endl;
#endif
    }
    
    std::cout << "\n3. Final verification:" << std::endl;
    std::cout << "   ✅ Cache strategy setting/getting works correctly" << std::endl;
    std::cout << "   ✅ DISTRIBUTE never mirrors (correct)" << std::endl;
    std::cout << "   ✅ MIRROR mirrors when appropriate (correct)" << std::endl;
    std::cout << "   ✅ Cache strategies don't interfere with mirroring logic" << std::endl;
    
    std::cout << "\n=== Cache Strategy Integration Test Complete ===" << std::endl;
}

int main() {
    test_cache_strategy_integration();
    return 0;
}
