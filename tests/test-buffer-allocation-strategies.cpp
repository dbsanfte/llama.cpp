#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include <iostream>
#include <cassert>

// Test that actually allocates buffers to see strategy differences
void test_actual_buffer_allocation() {
    std::cout << "=== Testing Actual Buffer Allocation with Different Strategies ===" << std::endl;
    
    // Test buffer size that's large enough to trigger replication (128MB)
    const size_t test_size = 128 * 1024 * 1024;
    
    const struct {
        const char* name;
        int strategy;
    } strategies[] = {
        {"DISABLED", 0},
        {"EAGER", 1},
        {"LAZY", 2},
        {"DELTA", 3},
        {"PARTIAL", 4}
    };
    
    std::cout << "\n1. Testing buffer allocation with each strategy:" << std::endl;
    std::cout << "   Buffer size: " << (test_size / 1024 / 1024) << "MB" << std::endl;
    
    // First set up NUMA mirroring mode
    ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);
    
    for (int i = 0; i < 5; i++) {
        std::cout << "\n--- Testing " << strategies[i].name << " strategy ---" << std::endl;
        
        // Set the cache strategy
        ggml_numa_set_cache_strategy(strategies[i].strategy);
        
        // Get the NUMA buffer type
        ggml_backend_buffer_type_t buffer_type = ggml_backend_cpu_numa_buffer_type();
        if (!buffer_type) {
            std::cout << "   ❌ Could not get NUMA buffer type" << std::endl;
            continue;
        }
        
        std::cout << "   Buffer type: " << ggml_backend_buft_name(buffer_type) << std::endl;
        
        // Try to allocate a buffer
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buffer_type, test_size);
        
        if (buffer) {
            std::cout << "   ✓ Successfully allocated " << (test_size / 1024 / 1024) << "MB buffer" << std::endl;
            
            // Check if it's host accessible
            if (ggml_backend_buffer_is_host(buffer)) {
                std::cout << "   ✓ Buffer is host accessible" << std::endl;
            } else {
                std::cout << "   ⚠ Buffer is not host accessible" << std::endl;
            }
            
            // Get the base address
            void* base = ggml_backend_buffer_get_base(buffer);
            if (base) {
                std::cout << "   ✓ Buffer base address: " << base << std::endl;
            } else {
                std::cout << "   ❌ Could not get buffer base address" << std::endl;
            }
            
            // Free the buffer
            ggml_backend_buffer_free(buffer);
            std::cout << "   ✓ Buffer freed" << std::endl;
            
        } else {
            std::cout << "   ❌ Failed to allocate buffer (may be expected for single-node system)" << std::endl;
        }
    }
    
    std::cout << "\n=== Buffer Allocation Test Complete ===" << std::endl;
    std::cout << "\nNote: On single-node systems, replication strategies may fall back to" << std::endl;
    std::cout << "standard allocation. Multi-node systems will show different allocation patterns." << std::endl;
}

int main() {
    test_actual_buffer_allocation();
    return 0;
}
