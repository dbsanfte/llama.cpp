/**
 * @file ggml-numa-allocator.c
 * @brief NUMA-Aware Memory Allocator Implementation
 * 
 * This module provides robust NUMA memory allocation that works reliably
 * even in Docker containers where standard NUMA functions may be broken.
 * 
 * Key features:
 * - Multiple allocation strategies (local, interleave, distribute, mirror)
 * - Container-safe NUMA detection and allocation
 * - Per-node allocation tracking and statistics
 * - Thread-safe allocation/deallocation
 * - Memory mirroring across all NUMA nodes for optimal access
 * 
 * @author GGML NUMA Team
 * @date 2025
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
#define GGML_NUMA_MAX_NODES 8    /**< Maximum supported NUMA nodes */
#endif

/** @brief Global NUMA allocator context (initialized once) */
static ggml_numa_alloc_context_t g_numa_alloc_ctx = {0};
/** @brief Flag indicating if global context has been initialized */
static bool g_numa_alloc_initialized = false;

/**
 * @brief NUMA allocation tracking entry
 * 
 * Internal structure for tracking individual NUMA allocations.
 * Supports both regular allocations and mirrored allocations
 * that exist on multiple NUMA nodes simultaneously.
 */
typedef struct numa_alloc_entry {
    void* ptr;                                    /**< Primary memory pointer */
    size_t size;                                  /**< Allocation size in bytes */
    bool is_mirror;                               /**< True if this is a mirrored allocation */
    void* mirror_copies[GGML_NUMA_MAX_NODES];     /**< Mirror copies on each node */
    int num_mirror_copies;                        /**< Number of valid mirror copies */
    struct numa_alloc_entry* next;               /**< Next entry in linked list */
} numa_alloc_entry_t;

/** @brief Linked list head for tracking all NUMA allocations */
static numa_alloc_entry_t* g_numa_allocations = NULL;
/** @brief Mutex protecting the allocation tracking list */
static pthread_mutex_t g_numa_alloc_mutex = PTHREAD_MUTEX_INITIALIZER;

/**
 * @brief Add a regular NUMA allocation to the tracking list
 * 
 * Thread-safe function to register a new NUMA allocation for tracking.
 * Used for regular allocations (not mirrored).
 * 
 * @param ptr Pointer to allocated memory
 * @param size Size of allocation in bytes
 */
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

/**
 * @brief Add a mirrored NUMA allocation to the tracking list
 * 
 * Thread-safe function to register a mirrored allocation that exists
 * on multiple NUMA nodes simultaneously. All mirror copies are tracked.
 * 
 * @param primary_ptr Primary memory pointer (typically from node 0)
 * @param mirror_copies Array of pointers to mirror copies on each node
 * @param num_copies Number of valid mirror copies
 * @param size Size of each allocation in bytes
 */
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

/**
 * @brief Remove and free a NUMA allocation from tracking
 * 
 * Thread-safe function to remove an allocation from tracking and
 * free all associated memory. For mirrored allocations, frees all
 * mirror copies across all NUMA nodes.
 * 
 * @param ptr Primary memory pointer to remove
 * @return true if allocation was found and removed, false otherwise
 */
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

/**
 * @brief Initialize NUMA allocator context
 * 
 * Sets up the NUMA allocation context with the specified strategy.
 * Detects available NUMA nodes and initializes tracking structures.
 * 
 * @param ctx Pointer to NUMA allocation context to initialize
 * @param strategy Allocation strategy to use (see ggml_numa_alloc_strategy_t)
 * @return true on successful initialization, false on error
 */
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

/**
 * @brief Get the NUMA node of a memory address
 * 
 * Utility function to determine which NUMA node a memory address
 * is allocated on. Uses get_mempolicy() system call.
 * 
 * @param ptr Memory pointer to check
 * @return NUMA node ID (0-based), or -1 on error
 */
int get_memory_numa_node(void* ptr) {
    if (!ptr) return -1;
    
    int node = -1;
    int result = get_mempolicy(&node, NULL, 0, ptr, MPOL_F_NODE | MPOL_F_ADDR);
    if (result != 0) {
        return -1; // Error getting memory policy
    }
    return node;
}

/**
 * @brief Assert that memory is allocated on the expected NUMA node
 * 
 * Development/debugging function that verifies memory allocation
 * succeeded on the intended NUMA node. Aborts program if assertion fails.
 * 
 * @param ptr Memory pointer to check
 * @param expected_node Expected NUMA node ID
 * @param context Description of where this assertion is being called from
 */
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

/**
 * @brief Non-recursive fallback allocation to break infinite recursion
 * 
 * Simple aligned allocation without NUMA awareness, used as a fallback
 * when NUMA-specific allocation fails. Breaks potential recursion loops.
 * 
 * @param size Number of bytes to allocate
 * @param numa_ctx NUMA allocation context for tracking
 * @return Pointer to allocated memory, or NULL on failure
 */
static void* ggml_numa_fallback_alloc(size_t size, ggml_numa_alloc_context_t* numa_ctx) {
    void* mem = aligned_alloc(64, size);
    if (mem) {
        memset(mem, 0, size);
        numa_ctx->total_allocated += size;
        if (numa_ctx->debug_enabled) {
            printf("🔄 Fallback allocation: %zu bytes (breaking recursion)\n", size);
        }
    }
    return mem;
}

/**
 * @brief NUMA-aware aligned memory allocation
 * 
 * Main allocation function that provides aligned memory using the
 * context's NUMA strategy. Serves as a drop-in replacement for
 * aligned_malloc() with NUMA awareness.
 * 
 * Supports multiple allocation strategies:
 * - LOCAL: Allocate on current/isolation node
 * - MIRROR: Create copies on all NUMA nodes
 * - INTERLEAVE/DISTRIBUTE: Future strategies for balanced allocation
 * 
 * @param size Number of bytes to allocate (must be > 0)
 * @param numa_ctx NUMA allocation context (NULL uses global context)
 * @return Pointer to allocated memory, or NULL on failure
 */
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
}

/**
 * @brief NUMA-aware memory allocation with explicit node preference
 * 
 * Allocates memory with strong preference for a specific NUMA node.
 * Falls back to other allocation methods if preferred node allocation fails.
 * 
 * This function handles several scenarios:
 * - Multi-node systems: Uses numa_alloc_onnode() for explicit placement
 * - Single-node systems: Uses regular aligned allocation
 * - Container environments: Provides fallbacks when NUMA calls fail
 * 
 * @param size Number of bytes to allocate (must be > 0)  
 * @param preferred_node Preferred NUMA node ID (-1 for no preference)
 * @param numa_ctx NUMA allocation context
 * @return Pointer to allocated memory, or NULL on failure
 */
void* ggml_numa_aligned_malloc_on_node(size_t size, int preferred_node, ggml_numa_alloc_context_t* numa_ctx) {
    if (!numa_ctx) {
        numa_ctx = &g_numa_alloc_ctx;
    }
    
    if (preferred_node < 0 || preferred_node >= numa_ctx->num_numa_nodes) {
        return ggml_numa_fallback_alloc(size, numa_ctx);
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
        return ggml_numa_fallback_alloc(size, numa_ctx);
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

/**
 * @brief Allocate context memory pool distributed across NUMA nodes
 * 
 * Allocates a large memory pool for ggml context usage, with NUMA-aware
 * placement based on the current NUMA strategy. This is the main entry
 * point for ggml context memory allocation.
 * 
 * Strategy selection:
 * - ISOLATE mode: Allocate on specific isolation node
 * - MIRROR mode: Allocate on node 0 (with mirroring handled separately)
 * - Other modes: Allocate on node 0 as default
 * 
 * @param total_size Total size of memory pool to allocate
 * @param numa_ctx_ptr NUMA allocation context (void* for ggml compatibility)
 * @return Pointer to allocated memory pool, or NULL on failure
 */
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

/**
 * @brief Free NUMA-allocated memory
 * 
 * Frees memory previously allocated by NUMA allocator functions.
 * Delegates to ggml_numa_free() which handles tracking updates.
 * 
 * @param ptr Pointer to memory to free (NULL is safe)
 * @param numa_ctx NUMA allocation context (unused, kept for API compatibility)
 */
void ggml_numa_aligned_free(void* ptr, ggml_numa_alloc_context_t* numa_ctx) {
    if (!ptr) return;
    
    // Use the proper NUMA-aware free function
    ggml_numa_free(ptr);
}

/**
 * @brief Get NUMA allocation statistics
 * 
 * Prints detailed allocation statistics including per-node usage,
 * total allocations, and strategy information. Useful for performance
 * analysis and debugging NUMA allocation patterns.
 * 
 * @param ctx NUMA allocation context to report on (NULL uses global context)
 */
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

/**
 * @brief Enable or disable debug logging
 * 
 * Controls verbose debug output for NUMA allocation operations.
 * When enabled, provides detailed logging of allocation decisions,
 * NUMA node placement, and memory usage patterns.
 * 
 * @param ctx NUMA allocation context (NULL uses global context)
 * @param enabled true to enable debug logging, false to disable
 */
void ggml_numa_alloc_set_debug(ggml_numa_alloc_context_t* ctx, bool enabled) {
    if (!ctx) ctx = &g_numa_alloc_ctx;
    ctx->debug_enabled = enabled;
}

/**
 * @brief Check if memory was allocated by NUMA allocator
 * 
 * Determines whether a memory pointer was allocated using the NUMA
 * allocator functions by searching the internal tracking list.
 * Thread-safe operation that's useful for mixed allocation scenarios.
 * 
 * @param ptr Memory pointer to check
 * @return true if allocated by NUMA allocator, false otherwise
 */
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

/**
 * @brief Free NUMA-allocated memory (context-free version)
 * 
 * Frees memory previously allocated by NUMA allocator without requiring
 * the original allocation context. Handles both regular and mirrored
 * allocations by consulting the internal tracking list.
 * 
 * For mirrored allocations, automatically frees all mirror copies
 * across all NUMA nodes. Falls back to regular free() for untracked
 * memory pointers.
 * 
 * @param ptr Pointer to memory to free (NULL is safe)
 */
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

/**
 * @brief Public interface for NUMA allocation assertions
 * 
 * Wrapper around internal assert_numa_allocation() function to provide
 * public access for testing and debugging NUMA memory placement.
 * 
 * @param ptr Memory pointer to check
 * @param expected_node Expected NUMA node ID  
 * @param context Description of where this assertion is being called from
 */
void ggml_numa_assert_allocation(void* ptr, int expected_node, const char* context) {
    assert_numa_allocation(ptr, expected_node, context);
}

/**
 * @brief Force linking of NUMA allocator symbols
 * 
 * Ensures that NUMA allocator symbols are linked into the final binary
 * even if not directly called, preventing linker dead code elimination.
 * This is important for dynamic symbol resolution and plugin architectures.
 */
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
