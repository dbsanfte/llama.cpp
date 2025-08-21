#include <stdio.h>
#include <stdlib.h>
#include <numaif.h>
#include <numa.h>
#include <unistd.h>
#include <sys/mman.h>

void check_memory_node(void* ptr, size_t size, const char* name) {
    int node = -1;
    int status = get_mempolicy(&node, NULL, 0, ptr, MPOL_F_NODE | MPOL_F_ADDR);
    if (status == 0) {
        printf("✓ %s memory is allocated on NUMA node %d\n", name, node);
    } else {
        printf("✗ Failed to get memory policy for %s\n", name);
    }
}

int main() {
    printf("🔍 NUMA Memory Allocation Test\n");
    printf("=============================\n");
    
    // Check how large allocations behave with numactl --membind=0
    size_t test_size = 256 * 1024 * 1024;  // 256MB like our tensors
    
    printf("Allocating %zu MB with malloc...\n", test_size / (1024*1024));
    void* malloc_ptr = malloc(test_size);
    if (malloc_ptr) {
        check_memory_node(malloc_ptr, test_size, "malloc");
        
        // Touch the memory to force allocation
        printf("Touching memory to force allocation...\n");
        memset(malloc_ptr, 1, test_size);
        check_memory_node(malloc_ptr, test_size, "malloc (after touch)");
        
        free(malloc_ptr);
    }
    
    printf("\nAllocating %zu MB with numa_alloc_onnode(0)...\n", test_size / (1024*1024));
    void* numa_ptr = numa_alloc_onnode(test_size, 0);
    if (numa_ptr) {
        check_memory_node(numa_ptr, test_size, "numa_alloc_onnode(0)");
        numa_free(numa_ptr, test_size);
    }
    
    printf("\nAllocating %zu MB with numa_alloc_onnode(1)...\n", test_size / (1024*1024));
    numa_ptr = numa_alloc_onnode(test_size, 1);
    if (numa_ptr) {
        check_memory_node(numa_ptr, test_size, "numa_alloc_onnode(1)");
        numa_free(numa_ptr, test_size);
    }
    
    return 0;
}
