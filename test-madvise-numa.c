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
    
    printf("🔍 %s memory locality:\n", name);
    
    // Check first few pages
    for (int i = 0; i < 3; i++) {
        void* page_addr = (char*)ptr + i * 4096;
        int node = -1;
        
        if (get_mempolicy(&node, NULL, 0, page_addr, MPOL_F_NODE | MPOL_F_ADDR) == 0) {
            printf("  Page %d: NUMA node %d\n", i, node);
        }
    }
}

int main() {
    printf("🧪 Testing madvise-based page migration for NUMA\n");
    printf("================================================\n");
    
    size_t test_size = 16 * 4096; // 16 pages = 64KB
    
    // Allocate memory (will go to current node)
    void* mem = aligned_alloc(4096, test_size);
    if (!mem) {
        printf("❌ Failed to allocate memory\n");
        return 1;
    }
    
    // Touch all pages to allocate them
    for (size_t i = 0; i < test_size; i += 4096) {
        *((volatile char*)mem + i) = 0xFF;
    }
    
    printf("🔍 Initial state (allocated on node %d):\n", numa_node_of_cpu(sched_getcpu()));
    check_memory_node(mem, test_size, "Initial");
    
    // Try to migrate pages to node 0 using madvise + first-touch
    printf("\n🔄 Attempting page migration to node 0...\n");
    
    // Unbind current memory policy
    numa_set_membind(numa_all_nodes_ptr);
    
    // Bind to node 0
    numa_run_on_node(0);
    struct bitmask *mask = numa_allocate_nodemask();
    numa_bitmask_setbit(mask, 0);
    numa_set_membind(mask);
    
    // Method 1: Try madvise MADV_DONTNEED to unmap pages
    printf("  Trying MADV_DONTNEED...\n");
    if (madvise(mem, test_size, MADV_DONTNEED) == 0) {
        printf("  ✅ MADV_DONTNEED succeeded\n");
        
        // First-touch to reallocate on node 0
        for (size_t i = 0; i < test_size; i += 4096) {
            *((volatile char*)mem + i) = 0xAA;
        }
        
        check_memory_node(mem, test_size, "After MADV_DONTNEED + first-touch");
    } else {
        printf("  ❌ MADV_DONTNEED failed\n");
    }
    
    // Method 2: Try move_pages() system call
    printf("\n  Trying move_pages() system call...\n");
    
    void* pages[4];
    int target_nodes[4] = {0, 0, 0, 0};
    int status[4];
    
    for (int i = 0; i < 4; i++) {
        pages[i] = (char*)mem + i * 4096;
    }
    
    if (move_pages(0, 4, pages, target_nodes, status, MPOL_MF_MOVE) == 0) {
        printf("  ✅ move_pages() succeeded\n");
        for (int i = 0; i < 4; i++) {
            printf("    Page %d status: %d\n", i, status[i]);
        }
        check_memory_node(mem, test_size, "After move_pages()");
    } else {
        printf("  ❌ move_pages() failed\n");
        perror("move_pages");
    }
    
    numa_free_nodemask(mask);
    free(mem);
    
    return 0;
}
