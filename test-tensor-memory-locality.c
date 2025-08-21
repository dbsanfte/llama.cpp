#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <numa.h>
#include <numaif.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sched.h>

void check_memory_node(void* ptr, size_t size, const char* name) {
    if (!ptr) {
        printf("❌ %s: NULL pointer\n", name);
        return;
    }
    
    printf("\n🔍 Checking %s memory locality:\n", name);
    printf("  Address: %p, Size: %zu bytes\n", ptr, size);
    
    // Check page allocation across nodes
    void* page_start = (void*)((uintptr_t)ptr & ~0xFFFUL); // Page align
    size_t page_count = (size + 4095) / 4096;
    
    long pages_per_node[8] = {0}; // Support up to 8 NUMA nodes
    
    for (size_t i = 0; i < page_count && i < 10; i++) { // Check first 10 pages
        void* page_addr = (char*)page_start + i * 4096;
        int node = -1;
        
        if (get_mempolicy(&node, NULL, 0, page_addr, MPOL_F_NODE | MPOL_F_ADDR) == 0) {
            if (node >= 0 && node < 8) {
                pages_per_node[node]++;
            }
            if (i < 3) { // Show first few pages
                printf("  Page %zu (addr %p): NUMA node %d\n", i, page_addr, node);
            }
        }
    }
    
    printf("  Page distribution: ");
    for (int node = 0; node < 8; node++) {
        if (pages_per_node[node] > 0) {
            printf("Node%d=%ld ", node, pages_per_node[node]);
        }
    }
    printf("\n");
}

int main() {
    printf("🔍 Testing NUMA locality of different allocation methods\n");
    printf("====================================================\n");
    
    if (numa_available() < 0) {
        printf("❌ NUMA not available\n");
        return 1;
    }
    
    printf("🏗️  NUMA topology: %d nodes available\n", numa_max_node() + 1);
    printf("🖥️  Current CPU: %d (NUMA node %d)\n", sched_getcpu(), numa_node_of_cpu(sched_getcpu()));
    
    size_t test_size = 1024 * 1024; // 1MB
    
    // Test 1: Regular malloc (what ggml_aligned_malloc might use)
    void* regular_mem = aligned_alloc(64, test_size);
    check_memory_node(regular_mem, test_size, "aligned_alloc (like ggml)");
    
    // Test 2: NUMA-aware allocation 
    void* numa_mem = numa_alloc_onnode(test_size, 0);
    check_memory_node(numa_mem, test_size, "numa_alloc_onnode(node=0)");
    
    void* numa_mem1 = numa_alloc_onnode(test_size, 1);
    check_memory_node(numa_mem1, test_size, "numa_alloc_onnode(node=1)");
    
    // Test 3: First-touch policy test
    void* ft_mem = aligned_alloc(64, test_size);
    printf("\n🎯 Testing first-touch policy on aligned_alloc memory:\n");
    printf("  Before first-touch:\n");
    check_memory_node(ft_mem, test_size, "  pre-first-touch");
    
    // Bind to node 1 and first-touch
    numa_run_on_node(1);
    struct bitmask *mask = numa_allocate_nodemask();
    numa_bitmask_setbit(mask, 1);
    numa_set_membind(mask);
    
    // First touch every page
    volatile char* ptr = (volatile char*)ft_mem;
    for (size_t i = 0; i < test_size; i += 4096) {
        ptr[i] = 0x42; // Write to trigger page allocation
    }
    
    printf("  After first-touch on node 1:\n");
    check_memory_node(ft_mem, test_size, "  post-first-touch");
    
    numa_free_nodemask(mask);
    
    // Cleanup
    free(regular_mem);
    numa_free(numa_mem, test_size);
    numa_free(numa_mem1, test_size);
    free(ft_mem);
    
    return 0;
}
