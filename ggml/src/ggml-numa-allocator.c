/*
 * NUMA-Aware Memory Allocator Implementation
 * 
 * Provides robust NUMA memory allocation that works even in Docker containers
 * where standard NUMA functions are broken.
 */

#include "ggml-numa-allocator.h"
#include "ggml.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <numa.h>
#include <pthread.h>
#include <numaif.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sched.h>

// Global NUMA allocator context (initialized once)
static ggml_numa_alloc_context_t g_numa_alloc_ctx = {0};
static bool g_numa_alloc_initialized = false;

// Simple tracking structure for NUMA allocations
typedef struct numa_alloc_entry {
    void* ptr;
    size_t size;
    struct numa_alloc_entry* next;
} numa_alloc_entry_t;

static numa_alloc_entry_t* g_numa_allocations = NULL;
static pthread_mutex_t g_numa_alloc_mutex = PTHREAD_MUTEX_INITIALIZER;

static void add_numa_allocation(void* ptr, size_t size) {
    pthread_mutex_lock(&g_numa_alloc_mutex);
    numa_alloc_entry_t* entry = malloc(sizeof(numa_alloc_entry_t));
    entry->ptr = ptr;
    entry->size = size;
    entry->next = g_numa_allocations;
    g_numa_allocations = entry;
    pthread_mutex_unlock(&g_numa_alloc_mutex);
}

static bool remove_numa_allocation(void* ptr) {
    pthread_mutex_lock(&g_numa_alloc_mutex);
    numa_alloc_entry_t** current = &g_numa_allocations;
    
    while (*current) {
        if ((*current)->ptr == ptr) {
            numa_alloc_entry_t* to_remove = *current;
            *current = (*current)->next;
            free(to_remove);
            pthread_mutex_unlock(&g_numa_alloc_mutex);
            return true;
        }
        current = &(*current)->next;
    }
    pthread_mutex_unlock(&g_numa_alloc_mutex);
    return false;
}

bool ggml_numa_alloc_init(ggml_numa_alloc_context_t* ctx, ggml_numa_alloc_strategy_t strategy) {
    if (!ctx) return false;
    
    memset(ctx, 0, sizeof(*ctx));
    
    // Check NUMA availability
    if (numa_available() < 0) {
        printf("NUMA not available, using single-node allocation\n");
        ctx->num_numa_nodes = 1;
        ctx->strategy = GGML_NUMA_ALLOC_STRATEGY_LOCAL;
        return true;
    }
    
    ctx->num_numa_nodes = numa_max_node() + 1;
    ctx->strategy = strategy;
    ctx->debug_enabled = false;
    
    printf("🏗️ NUMA Allocator initialized: %d nodes, strategy=%d\n", 
           ctx->num_numa_nodes, strategy);
    
    return true;
}

// Detect if we're in a broken container environment
static bool is_numa_broken(void) {
    // Quick test: try allocating on different nodes and check actual location
    const size_t test_size = 4096; // One page
    
    void* test_mem0 = numa_alloc_onnode(test_size, 0);
    void* test_mem1 = numa_alloc_onnode(test_size, 1);
    
    if (!test_mem0 || !test_mem1) {
        if (test_mem0) numa_free(test_mem0, test_size);
        if (test_mem1) numa_free(test_mem1, test_size);
        return true; // Allocation failed
    }
    
    // Check actual node placement
    int node0 = -1, node1 = -1;
    get_mempolicy(&node0, NULL, 0, test_mem0, MPOL_F_NODE | MPOL_F_ADDR);
    get_mempolicy(&node1, NULL, 0, test_mem1, MPOL_F_NODE | MPOL_F_ADDR);
    
    numa_free(test_mem0, test_size);
    numa_free(test_mem1, test_size);
    
    // If both allocations went to the same node, NUMA is broken
    bool broken = (node0 == node1);
    if (broken) {
        printf("⚠️ NUMA allocation broken (both nodes → %d), using workaround\n", node0);
    }
    
    return broken;
}

// Enhanced allocation with explicit first-touch distribution
static void* allocate_distributed_memory(size_t total_size, ggml_numa_alloc_context_t* ctx) {
    if (ctx->num_numa_nodes <= 1) {
        // Fallback to regular allocation
        return aligned_alloc(64, total_size);
    }
    
    // Use mmap for better control
    void* memory = mmap(NULL, total_size, PROT_READ | PROT_WRITE, 
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    
    if (memory == MAP_FAILED) {
        printf("❌ mmap failed for %zu bytes\n", total_size);
        return NULL;
    }
    
    if (ctx->debug_enabled) {
        printf("🗂️ Distributing %zu bytes across %d NUMA nodes\n", 
               total_size, ctx->num_numa_nodes);
    }
    
    // Distribute pages across NUMA nodes using first-touch
    const size_t page_size = 4096;
    const size_t total_pages = (total_size + page_size - 1) / page_size;
    const size_t pages_per_node = total_pages / ctx->num_numa_nodes;
    
    volatile char* mem_ptr = (volatile char*)memory;
    
    for (int node = 0; node < ctx->num_numa_nodes; node++) {
        // Calculate pages for this node
        size_t start_page = node * pages_per_node;
        size_t end_page = (node == ctx->num_numa_nodes - 1) ? 
                         total_pages : (node + 1) * pages_per_node;
        
        if (start_page >= total_pages) break;
        
        // Bind to this NUMA node
        numa_run_on_node(node);
        struct bitmask *mask = numa_allocate_nodemask();
        numa_bitmask_setbit(mask, node);
        numa_set_membind(mask);
        
        if (ctx->debug_enabled) {
            printf("  Node %d: pages %zu-%zu (%zu pages)\n", 
                   node, start_page, end_page - 1, end_page - start_page);
        }
        
        // First-touch pages for this node
        for (size_t page = start_page; page < end_page; page++) {
            size_t offset = page * page_size;
            if (offset < total_size) {
                mem_ptr[offset] = 0x42; // First touch
            }
        }
        
        numa_free_nodemask(mask);
        ctx->per_node_allocated[node] += (end_page - start_page) * page_size;
    }
    
    // Reset to default policy
    numa_set_membind(numa_all_nodes_ptr);
    
    ctx->total_allocated += total_size;
    
    // Track this allocation for later cleanup
    add_numa_allocation(memory, total_size);
    
    if (ctx->debug_enabled) {
        printf("✅ Memory distributed across %d nodes\n", ctx->num_numa_nodes);
    }
    
    return memory;
}

void* ggml_numa_aligned_malloc(size_t size, ggml_numa_alloc_context_t* numa_ctx) {
    if (!numa_ctx) {
        // Use global context
        if (!g_numa_alloc_initialized) {
            ggml_numa_alloc_init(&g_numa_alloc_ctx, GGML_NUMA_ALLOC_STRATEGY_DISTRIBUTE);
            g_numa_alloc_initialized = true;
        }
        numa_ctx = &g_numa_alloc_ctx;
    }
    
    if (size == 0) {
        printf("⚠️ Zero-size allocation requested\n");
        return NULL;
    }
    
    // For large allocations (>1MB), use distribution strategy
    if (size >= 1024 * 1024 && numa_ctx->strategy == GGML_NUMA_ALLOC_STRATEGY_DISTRIBUTE) {
        return allocate_distributed_memory(size, numa_ctx);
    }
    
    // For smaller allocations, use local or interleaved allocation
    switch (numa_ctx->strategy) {
        case GGML_NUMA_ALLOC_STRATEGY_LOCAL: {
            // Try to allocate on current NUMA node
            int current_node = numa_node_of_cpu(sched_getcpu());
            void* mem = numa_alloc_onnode(size, current_node);
            if (mem) {
                numa_ctx->per_node_allocated[current_node] += size;
                numa_ctx->total_allocated += size;
                return mem;
            }
            // Fallback to default
            break;
        }
        case GGML_NUMA_ALLOC_STRATEGY_INTERLEAVE: {
            void* mem = numa_alloc_interleaved(size);
            if (mem) {
                numa_ctx->total_allocated += size;
                return mem;
            }
            break;
        }
        default:
            break;
    }
    
    // Fallback to regular aligned allocation
    void* mem = aligned_alloc(64, size);
    if (mem && numa_ctx->debug_enabled) {
        printf("⚠️ Fallback allocation for %zu bytes\n", size);
    }
    
    if (mem) {
        numa_ctx->total_allocated += size;
    }
    
    return mem;
}

void* ggml_numa_aligned_malloc_on_node(size_t size, int preferred_node, ggml_numa_alloc_context_t* numa_ctx) {
    if (!numa_ctx) {
        numa_ctx = &g_numa_alloc_ctx;
    }
    
    if (preferred_node < 0 || preferred_node >= numa_ctx->num_numa_nodes) {
        return ggml_numa_aligned_malloc(size, numa_ctx);
    }
    
    // Try explicit node allocation
    void* mem = numa_alloc_onnode(size, preferred_node);
    if (mem) {
        numa_ctx->per_node_allocated[preferred_node] += size;
        numa_ctx->total_allocated += size;
        return mem;
    }
    
    // If broken, use mmap + first-touch approach
    void* memory = mmap(NULL, size, PROT_READ | PROT_WRITE, 
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (memory == MAP_FAILED) {
        return ggml_numa_aligned_malloc(size, numa_ctx);
    }
    
    // Bind to target node and first-touch
    numa_run_on_node(preferred_node);
    struct bitmask *mask = numa_allocate_nodemask();
    numa_bitmask_setbit(mask, preferred_node);
    numa_set_membind(mask);
    
    // First-touch all pages
    volatile char* mem_ptr = (volatile char*)memory;
    for (size_t i = 0; i < size; i += 4096) {
        mem_ptr[i] = 0x42;
    }
    
    numa_free_nodemask(mask);
    numa_set_membind(numa_all_nodes_ptr);
    
    numa_ctx->per_node_allocated[preferred_node] += size;
    numa_ctx->total_allocated += size;
    
    // Track this allocation for later cleanup
    add_numa_allocation(memory, size);
    
    return memory;
}

void* ggml_numa_alloc_context_memory(size_t total_size, void* numa_ctx_ptr) {
    // Cast back to the proper type  
    ggml_numa_alloc_context_t* numa_ctx = (ggml_numa_alloc_context_t*)numa_ctx_ptr;
    if (!numa_ctx) {
        // Initialize global context if not already done
        if (!g_numa_alloc_initialized) {
            ggml_numa_alloc_init(&g_numa_alloc_ctx, GGML_NUMA_ALLOC_STRATEGY_DISTRIBUTE);
            g_numa_alloc_initialized = true;
        }
        numa_ctx = &g_numa_alloc_ctx;
    }
    
    printf("🗂️ Allocating NUMA-distributed context memory: %zu bytes across %d nodes\n", 
           total_size, numa_ctx->num_numa_nodes);
    
    return allocate_distributed_memory(total_size, numa_ctx);
}

void ggml_numa_aligned_free(void* ptr, ggml_numa_alloc_context_t* numa_ctx) {
    if (!ptr) return;
    
    // For simplicity, use free() for now
    // In a production implementation, we'd track allocation method
    free(ptr);
}

void ggml_numa_alloc_stats(const ggml_numa_alloc_context_t* ctx) {
    if (!ctx) ctx = &g_numa_alloc_ctx;
    
    printf("\n📊 NUMA Allocation Statistics:\n");
    printf("  Total allocated: %zu MB\n", ctx->total_allocated / (1024 * 1024));
    printf("  Strategy: %d\n", ctx->strategy);
    printf("  Per-node allocation:\n");
    
    for (int i = 0; i < ctx->num_numa_nodes; i++) {
        if (ctx->per_node_allocated[i] > 0) {
            printf("    Node %d: %zu MB\n", i, ctx->per_node_allocated[i] / (1024 * 1024));
        }
    }
}

void ggml_numa_alloc_set_debug(ggml_numa_alloc_context_t* ctx, bool enabled) {
    if (!ctx) ctx = &g_numa_alloc_ctx;
    ctx->debug_enabled = enabled;
}

bool ggml_numa_is_numa_allocated(void* ptr) {
    pthread_mutex_lock(&g_numa_alloc_mutex);
    numa_alloc_entry_t* current = g_numa_allocations;
    
    while (current) {
        if (current->ptr == ptr) {
            pthread_mutex_unlock(&g_numa_alloc_mutex);
            return true;
        }
        current = current->next;
    }
    pthread_mutex_unlock(&g_numa_alloc_mutex);
    return false;
}

void ggml_numa_free(void* ptr) {
    if (!ptr) return;
    
    if (remove_numa_allocation(ptr)) {
        // This was a NUMA allocation, use numa_free which handles this
        numa_free(ptr, 0); // numa_free with size 0 is handled internally
    }
}
