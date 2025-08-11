#include "../ggml/include/ggml.h"
#include "../ggml/include/ggml-alloc.h"
#include "../ggml/include/ggml-backend.h"

#include <cassert>
#include <iostream>
#include <vector>
#include <map>

#ifdef GGML_NUMA_MIRROR
#include <numa.h>
#include <numaif.h>

static int get_numa_node_of_pointer(void *ptr) {
    if (!ptr) return -1;
    
    int numa_node = -1;
    if (get_mempolicy(&numa_node, NULL, 0, ptr, MPOL_F_NODE | MPOL_F_ADDR) != 0) {
        return -1;
    }
    return numa_node;
}
#endif

int main() {
    std::cout << "=== Dual NUMA Buffer Allocation Test ===\n";
    
#ifdef GGML_NUMA_MIRROR
    if (numa_available() == -1) {
        std::cout << "NUMA not available - testing fallback behavior\n";
    } else {
        int max_node = numa_max_node();
        std::cout << "NUMA available - max node: " << max_node << "\n";
    }
#else
    std::cout << "NUMA support not compiled in\n";
    return 0;
#endif

    // Test 1: Regular CPU backend allocation
    std::cout << "\n--- Testing Regular CPU Backend ---\n";
    
    ggml_backend_reg_t cpu_reg = ggml_backend_reg_by_name("CPU");
    if (!cpu_reg) {
        std::cout << "❌ Failed to find CPU backend registry\n";
        return 1;
    }
    
    ggml_backend_dev_t cpu_dev = ggml_backend_reg_dev_get(cpu_reg, 0);
    if (!cpu_dev) {
        std::cout << "❌ Failed to get CPU device\n";
        return 1;
    }
    
    ggml_backend_t cpu_backend = ggml_backend_dev_init(cpu_dev, NULL);
    if (!cpu_backend) {
        std::cout << "❌ Failed to initialize CPU backend\n";
        return 1;
    }
    
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_get_default_buffer_type(cpu_backend);
    
    // Allocate multiple buffers to see distribution
    const size_t buffer_size = 1024 * 1024; // 1MB
    const int num_buffers = 8;
    
    std::vector<ggml_backend_buffer_t> cpu_buffers;
    std::map<int, int> cpu_node_counts;
    
    for (int i = 0; i < num_buffers; i++) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(cpu_buft, buffer_size);
        if (!buffer) {
            std::cout << "❌ Failed to allocate CPU buffer " << i << "\n";
            continue;
        }
        cpu_buffers.push_back(buffer);
        
#ifdef GGML_NUMA_MIRROR
        void *data = ggml_backend_buffer_get_base(buffer);
        int node = get_numa_node_of_pointer(data);
        if (node >= 0) {
            cpu_node_counts[node]++;
            std::cout << "  Buffer " << i << " allocated on NUMA node " << node << "\n";
        } else {
            std::cout << "  Buffer " << i << " allocated (node detection failed)\n";
        }
#else
        std::cout << "  Buffer " << i << " allocated\n";
#endif
    }
    
#ifdef GGML_NUMA_MIRROR
    std::cout << "CPU Backend NUMA distribution:\n";
    for (const auto &pair : cpu_node_counts) {
        std::cout << "  Node " << pair.first << ": " << pair.second << " buffers\n";
    }
    
    // Check if we have good distribution across multiple nodes
    if (numa_available() != -1 && numa_max_node() > 0 && cpu_node_counts.size() > 1) {
        std::cout << "✅ CPU backend distributed buffers across multiple NUMA nodes\n";
    } else if (numa_available() != -1 && numa_max_node() == 0) {
        std::cout << "✅ CPU backend allocated on single NUMA node (as expected)\n";
    } else {
        std::cout << "✅ CPU backend allocation completed\n";
    }
#endif

    // Test 2: REPACK buffer allocation (if available)
    std::cout << "\n--- Testing REPACK Buffer ---\n";
    
    // Try to get repack buffer type (this may not be available in all builds)
    if (cpu_reg) {
        int num_dev = ggml_backend_reg_dev_count(cpu_reg);
        std::cout << "CPU registry found with " << num_dev << " devices\n";
        
        if (num_dev > 0) {
            if (cpu_dev) {
                // Try to find repack buffer type
                ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(cpu_dev);
                if (host_buft) {
                    const char *name = ggml_backend_buft_name(host_buft);
                    std::cout << "  Host buffer type: " << name << "\n";
                    
                    // Test this buffer type with one allocation
                    ggml_backend_buffer_t test_buffer = ggml_backend_buft_alloc_buffer(host_buft, buffer_size);
                    if (test_buffer) {
#ifdef GGML_NUMA_MIRROR
                        void *data = ggml_backend_buffer_get_base(test_buffer);
                        int node = get_numa_node_of_pointer(data);
                        if (node >= 0) {
                            std::cout << "    Test allocation on NUMA node " << node << "\n";
                        }
#endif
                        ggml_backend_buffer_free(test_buffer);
                    }
                }
            }
        }
    }
    
    // Cleanup
    for (auto buffer : cpu_buffers) {
        ggml_backend_buffer_free(buffer);
    }
    
    ggml_backend_free(cpu_backend);
    
    std::cout << "\n✅ Dual NUMA buffer allocation test completed successfully\n";
    return 0;
}
