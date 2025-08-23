/*
 * NUMA-Aware Memory Allocator for ggml
 * 
 * Provides NUMA-aware allocation functions that work even in Docker containers
 * where standard numa_alloc_onnode() is broken.
 */

#ifndef GGML_NUMA_ALLOCATOR_H
#define GGML_NUMA_ALLOCATOR_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// NUMA allocation strategy
typedef enum {
    GGML_NUMA_ALLOC_STRATEGY_LOCAL,          // Allocate on current/specified node (for ISOLATE)
    GGML_NUMA_ALLOC_STRATEGY_MIRROR          // Create copies on all nodes (for MIRROR mode)
} ggml_numa_alloc_strategy_t;

// NUMA allocation context
typedef struct {
    ggml_numa_alloc_strategy_t strategy;
    int num_numa_nodes;
    size_t total_allocated;
    size_t per_node_allocated[8];  // Support up to 8 NUMA nodes
    bool debug_enabled;
} ggml_numa_alloc_context_t;

// Initialize NUMA allocator
bool ggml_numa_alloc_init(ggml_numa_alloc_context_t* ctx, ggml_numa_alloc_strategy_t strategy);

// NUMA-aware aligned malloc replacement
void* ggml_numa_aligned_malloc(size_t size, ggml_numa_alloc_context_t* numa_ctx);

// NUMA-aware malloc with explicit node preference
void* ggml_numa_aligned_malloc_on_node(size_t size, int preferred_node, ggml_numa_alloc_context_t* numa_ctx);

// NUMA-aware context memory pool allocation (distributed across nodes)
void* ggml_numa_alloc_context_memory(size_t total_size, void* numa_ctx);

// Free NUMA-allocated memory
void ggml_numa_aligned_free(void* ptr, ggml_numa_alloc_context_t* numa_ctx);

// Get allocation statistics
void ggml_numa_alloc_stats(const ggml_numa_alloc_context_t* ctx);

// Enable/disable debug logging
void ggml_numa_alloc_set_debug(ggml_numa_alloc_context_t* ctx, bool enabled);

// Check if memory was allocated by NUMA allocator
bool ggml_numa_is_numa_allocated(void* ptr);

// Free NUMA-allocated memory
void ggml_numa_free(void* ptr);

// Assert that memory allocation is on the expected NUMA node
void ggml_numa_assert_allocation(void* ptr, int expected_node, const char* context);

// Get the NUMA node of a memory address  
int get_memory_numa_node(void* ptr);

// Force linking of NUMA allocator symbols (internal use)
void ggml_force_link_numa_allocator_symbols(void);

#ifdef __cplusplus
}
#endif

#endif // GGML_NUMA_ALLOCATOR_H
