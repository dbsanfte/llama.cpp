#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <numa.h>
#include <numaif.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sched.h>

void check_tensor_memory_locality(void* tensor_data, size_t size, const char* name) {
    printf("\nChecking %s tensor memory locality:\n", name);
    
    // Check first 20 pages
    long pages_per_node[8] = {0};
    int pages_checked = 0;
    
    for (size_t i = 0; i < size && pages_checked < 20; i += 4096, pages_checked++) {
        void* page_addr = (char*)tensor_data + i;
        int node = -1;
        
        if (get_mempolicy(&node, NULL, 0, page_addr, MPOL_F_NODE | MPOL_F_ADDR) == 0) {
            if (node >= 0 && node < 8) {
                pages_per_node[node]++;
            }
            if (pages_checked < 5) {
                printf("  Page %d (offset %zu): NUMA node %d\n", pages_checked, i, node);
            }
        }
    }
    
    printf("  Distribution of %d pages: ", pages_checked);
    for (int node = 0; node < 8; node++) {
        if (pages_per_node[node] > 0) {
            printf("Node%d=%ld(%.1f%%) ", node, pages_per_node[node], 
                   100.0 * pages_per_node[node] / pages_checked);
        }
    }
    printf("\n");
    
    // Performance implications
    int current_cpu = sched_getcpu();
    int current_node = numa_node_of_cpu(current_cpu);
    printf("  Current CPU: %d (NUMA node %d)\n", current_cpu, current_node);
    
    long local_pages = pages_per_node[current_node];
    long remote_pages = pages_checked - local_pages;
    if (remote_pages > 0) {
        printf("  WARNING: CROSS-NODE ACCESS: %ld/%d pages (%.1f%%) are remote\n", 
               remote_pages, pages_checked, 100.0 * remote_pages / pages_checked);
    } else {
        printf("  OK: LOCAL ACCESS: All pages are on current NUMA node\n");
    }
}

// Simulate ggml tensor allocation
void* simulate_ggml_tensor_allocation(size_t total_size) {
    // This simulates what ggml_aligned_malloc does
    void* memory_pool = aligned_alloc(64, total_size);
    if (!memory_pool) return NULL;
    
    // Touch the memory to allocate pages (simulates tensor initialization)
    volatile char* ptr = (volatile char*)memory_pool;
    for (size_t i = 0; i < total_size; i += 4096) {
        ptr[i] = 0xFF;
    }
    
    return memory_pool;
}

int main() {
    printf("Analyzing REAL tensor memory locality in our setup\n");
    printf("=================================================\n");
    
    if (numa_available() < 0) {
        printf("NUMA not available\n");
        return 1;
    }
    
    // Simulate the test setup: large tensors like in our test
    size_t tensor_size = 256 * 1024 * 1024; // 256MB like our HUGE test
    
    printf("Simulating ggml context allocation (256MB tensor)...\n");
    
    // Test from different starting NUMA nodes
    printf("\n1. Allocating from NUMA node 0:\n");
    numa_run_on_node(0);
    void* tensor_a = simulate_ggml_tensor_allocation(tensor_size);
    check_tensor_memory_locality(tensor_a, tensor_size, "Tensor A (allocated from node 0)");
    
    printf("\n2. Allocating from NUMA node 1:\n");
    numa_run_on_node(1);
    void* tensor_b = simulate_ggml_tensor_allocation(tensor_size);
    check_tensor_memory_locality(tensor_b, tensor_size, "Tensor B (allocated from node 1)");
    
    // Test cross-node access scenario
    printf("\nTesting cross-node access pattern:\n");
    printf("  Scenario: Tensor allocated on node 1, accessed from node 0\n");
    
    numa_run_on_node(0); // Switch to node 0
    check_tensor_memory_locality(tensor_b, tensor_size, "Cross-node access");
    
    // Test our first-touch approach
    printf("\nTesting first-touch migration on cross-node tensor:\n");
    struct bitmask *mask = numa_allocate_nodemask();
    numa_bitmask_setbit(mask, 0);
    numa_set_membind(mask);
    
    // Attempt first-touch on a slice (simulate kernel behavior)
    size_t slice_size = tensor_size / 2; // First half
    volatile float* slice_ptr = (volatile float*)tensor_b;
    printf("  Attempting first-touch on slice (%zu bytes)...\n", slice_size);
    
    for (size_t i = 0; i < slice_size / sizeof(float); i += 1024) {
        slice_ptr[i] = 1.0f;
    }
    
    check_tensor_memory_locality(tensor_b, tensor_size, "After first-touch attempt");
    
    numa_free_nodemask(mask);
    free(tensor_a);
    free(tensor_b);
    
    return 0;
}
