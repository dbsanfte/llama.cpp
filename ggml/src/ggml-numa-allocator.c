/*
 * NUMA-Aware Memory Allocator Implementation
 * 
 * Provides robust NUMA memory allocation that works even in Docker containers
 * where standard NUMA functions are broken.
 */

#include "ggml-numa-allocator.h"
#include "ggml.h"
#include "ggml-cpu.h"  // For enum ggml_numa_strategy
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

#ifndef GGML_NUMA_MAX_NODES
#define GGML_NUMA_MAX_NODES 8
#endif

// Global NUMA allocator context (initialized once)
static ggml_numa_alloc_context_t g_numa_alloc_ctx = {0};
static bool g_numa_alloc_initialized = false;

// Simple tracking structure for NUMA allocations
typedef struct numa_alloc_entry {
    void* ptr;
    size_t size;
    bool is_mirror;
    void* mirror_copies[GGML_NUMA_MAX_NODES];  // For MIRROR allocations, track all node copies
    int num_mirror_copies;
    struct numa_alloc_entry* next;
} numa_alloc_entry_t;

static numa_alloc_entry_t* g_numa_allocations = NULL;
static pthread_mutex_t g_numa_alloc_mutex = PTHREAD_MUTEX_INITIALIZER;

static void add_numa_allocation(void* ptr, size_t size) {
    pthread_mutex_lock(&g_numa_alloc_mutex);
    numa_alloc_entry_t* entry = malloc(sizeof(numa_alloc_entry_t));
    entry->ptr = ptr;
    entry->size = size;
    entry->is_mirror = false;
    entry->num_mirror_copies = 0;
    memset(entry->mirror_copies, 0, sizeof(entry->mirror_copies));
    entry->next = g_numa_allocations;
    g_numa_allocations = entry;
    pthread_mutex_unlock(&g_numa_alloc_mutex);
}

static void add_numa_mirror_allocation(void* primary_ptr, void** mirror_copies, int num_copies, size_t size) {
    pthread_mutex_lock(&g_numa_alloc_mutex);
    numa_alloc_entry_t* entry = malloc(sizeof(numa_alloc_entry_t));
    entry->ptr = primary_ptr;
    entry->size = size;
    entry->is_mirror = true;
    entry->num_mirror_copies = num_copies;
    for (int i = 0; i < num_copies && i < GGML_NUMA_MAX_NODES; i++) {
        entry->mirror_copies[i] = mirror_copies[i];
    }
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
            
            // If this is a mirror allocation, free all copies
            if (to_remove->is_mirror) {
                printf("🪞 Freeing MIRROR allocation with %d copies\n", to_remove->num_mirror_copies);
                for (int i = 0; i < to_remove->num_mirror_copies; i++) {
                    if (to_remove->mirror_copies[i]) {
                        numa_free(to_remove->mirror_copies[i], to_remove->size);
                        printf("  ✅ Freed mirror copy %d: %p\n", i, to_remove->mirror_copies[i]);
                    }
                }
            }
            
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

// Utility function to get the NUMA node of a memory address
int get_memory_numa_node(void* ptr) {
    if (!ptr) return -1;
    
    int node = -1;
    int result = get_mempolicy(&node, NULL, 0, ptr, MPOL_F_NODE | MPOL_F_ADDR);
    if (result != 0) {
        return -1; // Error getting memory policy
    }
    return node;
}

// Assert that memory is allocated on the expected NUMA node
static void assert_numa_allocation(void* ptr, int expected_node, const char* context) {
    if (!ptr) return;
    
    int actual_node = get_memory_numa_node(ptr);
    if (actual_node != expected_node) {
        printf("❌ NUMA ASSERTION FAILED in %s: expected node %d, got node %d for ptr %p\n", 
               context, expected_node, actual_node, ptr);
        printf("   This is a fatal error - NUMA allocations MUST be on the correct node\n");
        fflush(stdout);
        fflush(stderr);
        abort(); // Always abort on NUMA assertion failure, like GGML_ASSERT
    } else {
        printf("✅ NUMA ASSERTION PASSED in %s: ptr %p correctly allocated on node %d\n",
               context, ptr, actual_node);
    }
}

void* ggml_numa_aligned_malloc(size_t size, ggml_numa_alloc_context_t* numa_ctx) {
    if (!numa_ctx) {
        // Use global context
        if (!g_numa_alloc_initialized) {
            ggml_numa_alloc_init(&g_numa_alloc_ctx, GGML_NUMA_ALLOC_STRATEGY_LOCAL);
            g_numa_alloc_initialized = true;
        }
        numa_ctx = &g_numa_alloc_ctx;
    }
    
    if (size == 0) {
        printf("⚠️ Zero-size allocation requested\n");
        return NULL;
    }
    
    // Get current NUMA strategy to determine allocation behavior
    // Use NUMA state functions from ggml.h
    extern int ggml_numa_get_strategy(void);
    extern int ggml_numa_get_isolate_node(void);
    
    int strategy = ggml_numa_get_strategy();
    
    switch (numa_ctx->strategy) {
        case GGML_NUMA_ALLOC_STRATEGY_LOCAL: {
            // For ISOLATE mode, allocate on the specific isolation node
            // For other modes, allocate on current node
            int target_node;
            if (strategy == GGML_NUMA_STRATEGY_ISOLATE) {
                target_node = ggml_numa_get_isolate_node();
                if (target_node < 0) {
                    printf("⚠️ ISOLATE mode but no isolation node specified, using current node\n");
                    target_node = numa_node_of_cpu(sched_getcpu());
                }
            } else {
                target_node = numa_node_of_cpu(sched_getcpu());
            }
            
            // Use the specific node allocation function
            void* mem = ggml_numa_aligned_malloc_on_node(size, target_node, numa_ctx);
            if (mem) {
                if (numa_ctx->debug_enabled) {
                    printf("✅ LOCAL allocation: %zu bytes on node %d\n", size, target_node);
                }
                return mem;
            }
            break;
        }
        
        case GGML_NUMA_ALLOC_STRATEGY_MIRROR: {
            // For MIRROR mode, we need to allocate copies on ALL nodes
            // This is fundamentally different - we return the node 0 copy but ensure all exist
            printf("🪞 MIRROR allocation: Creating %zu bytes on all %d nodes\n", 
                   size, numa_ctx->num_numa_nodes);
            
            // Allocate on each node and track all copies
            void* primary_mem = NULL;
            void* mirror_copies[GGML_NUMA_MAX_NODES] = {0};
            int successful_copies = 0;
            
            for (int node = 0; node < numa_ctx->num_numa_nodes && node < GGML_NUMA_MAX_NODES; node++) {
                void* node_mem = ggml_numa_aligned_malloc_on_node(size, node, numa_ctx);
                if (!node_mem) {
                    printf("❌ MIRROR allocation failed on node %d\n", node);
                    // Clean up already-allocated nodes
                    for (int cleanup = 0; cleanup < successful_copies; cleanup++) {
                        if (mirror_copies[cleanup]) {
                            numa_free(mirror_copies[cleanup], size);
                        }
                    }
                    return NULL;
                }
                
                // HARD ASSERTION: Verify each mirror copy is on correct node
                int actual_node = get_memory_numa_node(node_mem);
                if (actual_node != node) {
                    printf("❌ MIRROR ALLOCATION FAILURE: Copy %d wanted node %d, got node %d for ptr %p\n", 
                           successful_copies, node, actual_node, node_mem);
                    printf("   This is a fatal error - MIRROR copies MUST be on their assigned nodes\n");
                    fflush(stdout);
                    fflush(stderr);
                    
                    // Clean up all allocations including this failed one
                    numa_free(node_mem, size);
                    for (int cleanup = 0; cleanup < successful_copies; cleanup++) {
                        if (mirror_copies[cleanup]) {
                            numa_free(mirror_copies[cleanup], size);
                        }
                    }
                    abort(); // Always abort on MIRROR allocation failure
                }
                
                mirror_copies[node] = node_mem;
                successful_copies++;
                
                if (node == 0) {
                    primary_mem = node_mem; // Return the first node's copy
                }
                
                if (numa_ctx->debug_enabled) {
                    printf("  ✅ MIRROR copy created on node %d: %p\n", node, node_mem);
                }
            }
            
            if (primary_mem) {
                // Track this as a mirror allocation
                add_numa_mirror_allocation(primary_mem, mirror_copies, successful_copies, size);
                
                if (numa_ctx->debug_enabled) {
                    printf("✅ MIRROR allocation complete, returning primary copy: %p\n", primary_mem);
                }
            }
            
            return primary_mem;
        }
        
        default:
            printf("⚠️ Unknown NUMA allocator strategy: %d\n", numa_ctx->strategy);
            break;
    }
    
    // Fallback to regular aligned allocation - ensure it's zeroed
    void* mem = aligned_alloc(64, size);
    if (mem) {
        memset(mem, 0, size); // Ensure memory is zeroed
        
        // Hard assertion: Verify which node the fallback allocation landed on
        int actual_node = get_memory_numa_node(mem);
        if (numa_ctx->debug_enabled || actual_node >= 0) {
            printf("⚠️ Fallback allocation for %zu bytes (zeroed) - ended up on node %d\n", size, actual_node);
        }
        
        // For fallback allocations, we cannot control NUMA placement, 
        // but we should document which node it ended up on for debugging
        
        numa_ctx->total_allocated += size;
    }
    
    return mem;
}void* ggml_numa_aligned_malloc_on_node(size_t size, int preferred_node, ggml_numa_alloc_context_t* numa_ctx) {
    if (!numa_ctx) {
        numa_ctx = &g_numa_alloc_ctx;
    }
    
    if (preferred_node < 0 || preferred_node >= numa_ctx->num_numa_nodes) {
        return ggml_numa_aligned_malloc(size, numa_ctx);
    }
    
    // Check if NUMA is available at all
    if (numa_available() < 0 || numa_ctx->num_numa_nodes <= 1) {
        // No NUMA support or single node - use regular allocation
        void* mem = aligned_alloc(64, size);
        if (mem) {
            memset(mem, 0, size);
            numa_ctx->total_allocated += size;
            if (numa_ctx->debug_enabled) {
                printf("📍 Single node allocation: %zu bytes (no NUMA)\n", size);
            }
            
            // For single node systems, still verify the allocation worked
            if (numa_available() >= 0) {
                int actual_node = get_memory_numa_node(mem);
                if (numa_ctx->debug_enabled) {
                    printf("✅ Single node verification: ptr %p on node %d\n", mem, actual_node);
                }
            }
        }
        return mem;
    }
    
    // Use mmap + first-touch approach (more reliable in containers)
    void* memory = mmap(NULL, size, PROT_READ | PROT_WRITE, 
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (memory == MAP_FAILED) {
        printf("❌ mmap failed for %zu bytes on node %d\n", size, preferred_node);
        return ggml_numa_aligned_malloc(size, numa_ctx);
    }
    
    // Bind current thread to target node for first-touch
    int old_node = numa_node_of_cpu(sched_getcpu());
    numa_run_on_node(preferred_node);
    
    struct bitmask *old_mask = numa_get_membind();
    struct bitmask *mask = numa_allocate_nodemask();
    numa_bitmask_setbit(mask, preferred_node);
    numa_set_membind(mask);
    
    // First-touch all pages to establish NUMA locality
    volatile char* mem_ptr = (volatile char*)memory;
    const size_t page_size = 4096;
    for (size_t i = 0; i < size; i += page_size) {
        mem_ptr[i] = 0; // Zero and establish NUMA placement
    }
    
    // Restore original NUMA policy
    numa_set_membind(old_mask);
    numa_free_nodemask(mask);
    numa_free_nodemask(old_mask);
    
    // Verify allocation is on correct node
    int actual_node = get_memory_numa_node(memory);
    if (actual_node == preferred_node) {
        numa_ctx->per_node_allocated[preferred_node] += size;
        numa_ctx->total_allocated += size;
        
        // Track this allocation for later cleanup
        add_numa_allocation(memory, size);
        
        if (numa_ctx->debug_enabled) {
            printf("✅ Node allocation successful: %zu bytes on node %d (ptr: %p)\n", 
                   size, preferred_node, memory);
        }
        
        return memory;
    } else {
        // HARD FAILURE - NUMA allocation MUST be on correct node
        printf("❌ NUMA ALLOCATION FAILURE: wanted node %d, got node %d for ptr %p\n", 
               preferred_node, actual_node, memory);
        printf("   This is a fatal error - NUMA allocations MUST be on the correct node\n");
        fflush(stdout);
        fflush(stderr);
        
        // Clean up the failed allocation
        munmap(memory, size);
        abort(); // Always abort on NUMA allocation failure
    }
}

void* ggml_numa_alloc_context_memory(size_t total_size, void* numa_ctx_ptr) {
    // Cast back to the proper type  
    ggml_numa_alloc_context_t* numa_ctx = (ggml_numa_alloc_context_t*)numa_ctx_ptr;
    if (!numa_ctx) {
        // Initialize global context if not already done
        if (!g_numa_alloc_initialized) {
            // Check the current NUMA strategy to choose appropriate allocation strategy
            extern int ggml_numa_get_strategy(void);
            extern int ggml_numa_get_isolate_node(void);
            
            int strategy = ggml_numa_get_strategy();
            ggml_numa_alloc_strategy_t alloc_strategy;
            
            if (strategy == 2) { // GGML_NUMA_STRATEGY_ISOLATE = 2
                // For isolate mode, allocate everything on the isolation node
                alloc_strategy = GGML_NUMA_ALLOC_STRATEGY_LOCAL;
                printf("🏗️ NUMA Allocator: Using LOCAL strategy for ISOLATE mode (strategy %d)\n", strategy);
            } else if (strategy == 4) { // GGML_NUMA_STRATEGY_MIRROR = 4
                // For mirror mode, create copies on all nodes
                alloc_strategy = GGML_NUMA_ALLOC_STRATEGY_MIRROR;
                printf("🏗️ NUMA Allocator: Using MIRROR strategy for MIRROR mode (strategy %d)\n", strategy);
            } else {
                // For disabled/other modes, use local allocation
                alloc_strategy = GGML_NUMA_ALLOC_STRATEGY_LOCAL;
                printf("🏗️ NUMA Allocator: Using LOCAL strategy for strategy %d (default)\n", strategy);
            }
            
            ggml_numa_alloc_init(&g_numa_alloc_ctx, alloc_strategy);
            g_numa_alloc_initialized = true;
        }
        numa_ctx = &g_numa_alloc_ctx;
    }
    
    // Check strategy and allocate accordingly
    int strategy = ggml_numa_get_strategy();
    
    if (strategy == 2) { // GGML_NUMA_STRATEGY_ISOLATE = 2
        int isolate_node = ggml_numa_get_isolate_node();
        if (isolate_node >= 0) {
            printf("🗂️ Allocating NUMA context memory on isolation node %d: %zu bytes\n", 
                   isolate_node, total_size);
            // For isolate mode, allocate everything on the specified node
            return ggml_numa_aligned_malloc_on_node(total_size, isolate_node, numa_ctx);
        }
    } else if (strategy == GGML_NUMA_STRATEGY_MIRROR) {
        // For mirror mode, we still allocate the context on the first node
        // Individual tensors will be mirrored via tensor_set_data_numa_mirror()
        printf("🗂️ Allocating NUMA context memory pool for MIRROR mode: %zu MB\n", total_size / (1024 * 1024));
        return ggml_numa_aligned_malloc_on_node(total_size, 0, numa_ctx);
    }
    
    // Default: allocate on node 0 
    printf("🗂️ Allocating NUMA context memory on default node: %zu bytes\n", total_size);
    return ggml_numa_aligned_malloc_on_node(total_size, 0, numa_ctx);
}

void ggml_numa_aligned_free(void* ptr, ggml_numa_alloc_context_t* numa_ctx) {
    if (!ptr) return;
    
    // Use the proper NUMA-aware free function
    ggml_numa_free(ptr);
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
        // remove_numa_allocation already handled freeing mirror copies if needed
        // For non-mirror allocations, we still need to free the original pointer
        // But we can't call numa_free here because mirror copies already freed all
        printf("✅ NUMA allocation freed: %p\n", ptr);
    } else {
        // This wasn't tracked as a NUMA allocation, use regular free
        free(ptr);
    }
}

// Public interface for NUMA allocation assertions
void ggml_numa_assert_allocation(void* ptr, int expected_node, const char* context) {
    assert_numa_allocation(ptr, expected_node, context);
}

// Force linking of NUMA allocator symbols (prevent dead code elimination)
void ggml_force_link_numa_allocator_symbols(void) {
    // This function ensures that NUMA allocator symbols are linked
    // even if not directly called, preventing linker optimization
    volatile void* symbols[] = {
        (void*)ggml_numa_alloc_context_memory,
        (void*)ggml_numa_is_numa_allocated,
        (void*)ggml_numa_free
    };
    (void)symbols; // Suppress unused variable warning
}
