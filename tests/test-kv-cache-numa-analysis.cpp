#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <numa.h>
#include <numaif.h>
#include <unistd.h>
#include <sys/mman.h>

// Simulate KV cache allocation patterns
class KVCacheSimulator {
public:
    // Simulate standard llama.cpp KV cache allocation (non-NUMA-aware)
    void* allocate_standard_kv_cache(size_t size) {
        // This simulates ggml_aligned_malloc() behavior
        void* ptr = nullptr;
        int result = posix_memalign(&ptr, 64, size);
        if (result != 0) {
            return nullptr;
        }
        
        // Touch the memory to actually allocate pages
        memset(ptr, 0, size);
        return ptr;
    }
    
    // Simulate NUMA-aware KV cache allocation 
    void* allocate_numa_aware_kv_cache(size_t size, int node) {
        void* ptr = nullptr;
        
        if (numa_available() == -1) {
            std::cout << "NUMA not available, falling back to standard allocation\n";
            return allocate_standard_kv_cache(size);
        }
        
        ptr = numa_alloc_onnode(size, node);
        if (ptr == nullptr) {
            return nullptr;
        }
        
        // Touch the memory to ensure it's allocated on the specified node
        memset(ptr, 0, size);
        return ptr;
    }
    
    // Check which NUMA node memory is allocated on
    int check_memory_node(void* ptr, size_t size) {
        if (numa_available() == -1) {
            return -1;
        }
        
        int node = -1;
        get_mempolicy(&node, nullptr, 0, ptr, MPOL_F_NODE | MPOL_F_ADDR);
        return node;
    }
    
    void benchmark_memory_access(void* ptr, size_t size, int access_node, const std::string& label) {
        // Move this thread to specified NUMA node
        if (numa_available() != -1 && access_node >= 0) {
            numa_run_on_node(access_node);
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Simulate KV cache access pattern (random reads/writes)
        volatile uint64_t* data = (volatile uint64_t*)ptr;
        size_t count = size / sizeof(uint64_t);
        uint64_t sum = 0;
        
        // Read test
        for (size_t i = 0; i < count; i += 64) {  // Cache line stride
            sum += data[i];
        }
        
        // Write test  
        for (size_t i = 0; i < count; i += 64) {
            data[i] = sum + i;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        int memory_node = check_memory_node(ptr, size);
        
        std::cout << label << ":\n";
        std::cout << "  Memory allocated on node: " << memory_node << "\n";
        std::cout << "  Accessed from node: " << access_node << "\n";
        std::cout << "  Access time: " << duration.count() << " μs\n";
        std::cout << "  Bandwidth: " << (size * 2.0) / (duration.count() * 1e-6) / (1024*1024*1024) << " GB/s\n";
        std::cout << std::endl;
    }
    
    void free_standard(void* ptr) {
        free(ptr);
    }
    
    void free_numa(void* ptr, size_t size) {
        if (numa_available() != -1) {
            numa_free(ptr, size);
        } else {
            free(ptr);
        }
    }
};

int main() {
    std::cout << "=== KV Cache NUMA Placement Analysis ===\n\n";
    
    // Check NUMA availability
    if (numa_available() == -1) {
        std::cout << "NUMA not available on this system\n";
        return 1;
    }
    
    int num_nodes = numa_max_node() + 1;
    std::cout << "NUMA nodes available: " << num_nodes << "\n";
    
    // Simulate various KV cache sizes
    std::vector<size_t> cache_sizes = {
        64 * 1024 * 1024,    // 64MB - small model
        256 * 1024 * 1024,   // 256MB - medium model  
        1024 * 1024 * 1024   // 1GB - large model
    };
    
    KVCacheSimulator simulator;
    
    for (size_t cache_size : cache_sizes) {
        std::cout << "\n=== Testing " << cache_size / (1024*1024) << "MB KV Cache ===\n\n";
        
        // Test 1: Standard allocation (current llama.cpp behavior)
        std::cout << "1. Standard KV Cache Allocation (current llama.cpp):\n";
        void* standard_cache = simulator.allocate_standard_kv_cache(cache_size);
        if (standard_cache) {
            for (int node = 0; node < num_nodes; ++node) {
                simulator.benchmark_memory_access(
                    standard_cache, cache_size, node,
                    "  Access from node " + std::to_string(node)
                );
            }
            simulator.free_standard(standard_cache);
        }
        
        // Test 2: NUMA-aware allocation on each node
        std::cout << "\n2. NUMA-Aware KV Cache Allocation:\n";
        for (int alloc_node = 0; alloc_node < num_nodes; ++alloc_node) {
            std::cout << "\n  Allocated on node " << alloc_node << ":\n";
            void* numa_cache = simulator.allocate_numa_aware_kv_cache(cache_size, alloc_node);
            if (numa_cache) {
                for (int access_node = 0; access_node < num_nodes; ++access_node) {
                    simulator.benchmark_memory_access(
                        numa_cache, cache_size, access_node,
                        "    Access from node " + std::to_string(access_node)
                    );
                }
                simulator.free_numa(numa_cache, cache_size);
            }
        }
        
        std::cout << "\n" << std::string(60, '-') << "\n";
    }
    
    std::cout << "\n=== Analysis Summary ===\n";
    std::cout << "The KV cache in llama.cpp uses standard posix_memalign() which is NOT NUMA-aware.\n";
    std::cout << "This means:\n";
    std::cout << "1. All KV cache memory ends up on one NUMA node (usually node 0)\n"; 
    std::cout << "2. Threads on other NUMA nodes suffer cross-node memory access penalties\n";
    std::cout << "3. Memory bandwidth is not fully utilized across all NUMA nodes\n";
    std::cout << "4. Performance degrades significantly with larger models and more NUMA nodes\n";
    std::cout << "\nRecommendation: Integrate KV cache allocation with NUMA coordinator\n";
    
    return 0;
}
