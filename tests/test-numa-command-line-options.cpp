// Test the new NUMA command-line options functionality
// This test verifies the --numa mirror and --numa-cache-strategy options work correctly

#include "common.h"
#include "arg.h"
#include <iostream>
#include <vector>
#include <cassert>

int main() {
    std::cout << "=== Testing New NUMA Command-Line Options ===\n\n";
    
    // Test 1: --numa mirror option
    std::cout << "1. Testing --numa mirror option...\n";
    {
        common_params params;
        const char* args[] = {"test", "--numa", "mirror"};
        assert(common_params_parse(3, (char**)args, params, LLAMA_EXAMPLE_COMMON));
        assert(params.numa == GGML_NUMA_STRATEGY_MIRROR);
        std::cout << "   ✓ --numa mirror parsed correctly (GGML_NUMA_STRATEGY_MIRROR = " << params.numa << ")\n";
    }
    
    // Test 2: --numa-cache-strategy options
    std::cout << "\n2. Testing --numa-cache-strategy options...\n";
    const std::vector<std::pair<const char*, numa_cache_strategy>> cache_strategies = {
        {"disabled", NUMA_CACHE_STRATEGY_DISABLED},
        {"eager", NUMA_CACHE_STRATEGY_EAGER}, 
        {"lazy", NUMA_CACHE_STRATEGY_LAZY},
        {"delta", NUMA_CACHE_STRATEGY_DELTA},
        {"partial", NUMA_CACHE_STRATEGY_PARTIAL}
    };
    
    for (const auto& [strategy_name, expected_value] : cache_strategies) {
        common_params params;
        const char* args[] = {"test", "--numa-cache-strategy", strategy_name};
        assert(common_params_parse(3, (char**)args, params, LLAMA_EXAMPLE_COMMON));
        assert(params.numa_cache_strategy == expected_value);
        std::cout << "   ✓ --numa-cache-strategy " << strategy_name << " parsed correctly (" << expected_value << ")\n";
    }
    
    // Test 3: Combined options
    std::cout << "\n3. Testing combined NUMA options...\n";
    {
        common_params params;
        const char* args[] = {"test", "--numa", "mirror", "--numa-cache-strategy", "eager"};
        assert(common_params_parse(5, (char**)args, params, LLAMA_EXAMPLE_COMMON));
        assert(params.numa == GGML_NUMA_STRATEGY_MIRROR);
        assert(params.numa_cache_strategy == NUMA_CACHE_STRATEGY_EAGER);
        std::cout << "   ✓ Combined --numa mirror and --numa-cache-strategy eager parsed correctly\n";
    }
    
    // Test 4: Default values
    std::cout << "\n4. Testing default values...\n";
    {
        common_params params;
        const char* args[] = {"test"};
        assert(common_params_parse(1, (char**)args, params, LLAMA_EXAMPLE_COMMON));
        assert(params.numa == GGML_NUMA_STRATEGY_DISABLED);
        assert(params.numa_cache_strategy == NUMA_CACHE_STRATEGY_DISABLED);
        std::cout << "   ✓ Default NUMA strategy: DISABLED (" << params.numa << ")\n";
        std::cout << "   ✓ Default cache strategy: DISABLED (" << params.numa_cache_strategy << ")\n";
    }
    
    std::cout << "\n=== All tests passed! ===\n";
    std::cout << "\nNew NUMA options are ready:\n";
    std::cout << "- --numa mirror: Enables coordinator data parallelism with NUMA-aware KV cache\n";
    std::cout << "- --numa-cache-strategy: Cache replication strategy (disabled, eager, lazy, delta, partial)\n";
    
    return 0;
}
