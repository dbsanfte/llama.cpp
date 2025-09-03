/**
 * @file ggml-numa-allocator.h
 * @brief NUMA-Aware Memory Allocator for ggml
 * 
 * This module provides NUMA-aware allocation functions that work reliably
 * even in Docker containers where standard numa_alloc_onnode() may be broken.
 * 
 * The allocator supports multiple strategies for NUMA memory placement:
 * - Interleaving across all nodes for uniform access
 * - Distributing chunks across nodes for parallel workloads  
 * - Local allocation on current node for cache efficiency
 * - Explicit per-node allocation for fine-grained control
 * 
 * @author David Sanftenberg
 * @date 2025
 */

#ifndef GGML_NUMA_ALLOCATOR_H
#define GGML_NUMA_ALLOCATOR_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief NUMA allocation strategy types
 * 
 * Defines the available strategies for NUMA memory allocation.
 * Each strategy optimizes for different use cases and access patterns.
 */
typedef enum {
    GGML_NUMA_ALLOC_STRATEGY_DEFAULT = 0,    /**< Use system default (usually broken in containers) */
    GGML_NUMA_ALLOC_STRATEGY_INTERLEAVE,     /**< Interleave across all nodes for uniform access */
    GGML_NUMA_ALLOC_STRATEGY_DISTRIBUTE,     /**< Distribute chunks across nodes for parallel workloads */
    GGML_NUMA_ALLOC_STRATEGY_LOCAL,          /**< Allocate on current node for cache efficiency */
    GGML_NUMA_ALLOC_STRATEGY_EXPLICIT        /**< Explicit per-node allocation for fine-grained control */
} ggml_numa_alloc_strategy_t;

/**
 * @brief NUMA allocation context structure
 * 
 * Maintains state and statistics for NUMA-aware memory allocation.
 * Tracks allocation strategy, per-node usage, and debug settings.
 */
typedef struct {
    ggml_numa_alloc_strategy_t strategy;     /**< Current allocation strategy in use */
    int num_numa_nodes;                      /**< Number of available NUMA nodes */
    size_t total_allocated;                  /**< Total bytes allocated across all nodes */
    size_t per_node_allocated[8];            /**< Per-node allocation tracking (max 8 nodes) */
    bool debug_enabled;                      /**< Enable debug logging for allocations */
} ggml_numa_alloc_context_t;

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
bool ggml_numa_alloc_init(ggml_numa_alloc_context_t* ctx, ggml_numa_alloc_strategy_t strategy);

/**
 * @brief NUMA-aware aligned memory allocation
 * 
 * Allocates aligned memory using the context's NUMA strategy.
 * Provides a drop-in replacement for aligned_malloc with NUMA awareness.
 * 
 * @param size Number of bytes to allocate
 * @param numa_ctx NUMA allocation context (NULL for default behavior)
 * @return Pointer to allocated memory, or NULL on failure
 */
void* ggml_numa_aligned_malloc(size_t size, ggml_numa_alloc_context_t* numa_ctx);

/**
 * @brief NUMA-aware memory allocation with explicit node preference
 * 
 * Allocates memory with preference for a specific NUMA node.
 * Falls back to other nodes if preferred node allocation fails.
 * 
 * @param size Number of bytes to allocate
 * @param preferred_node Preferred NUMA node ID (-1 for no preference)
 * @param numa_ctx NUMA allocation context
 * @return Pointer to allocated memory, or NULL on failure
 */
void* ggml_numa_aligned_malloc_on_node(size_t size, int preferred_node, ggml_numa_alloc_context_t* numa_ctx);

/**
 * @brief Allocate context memory pool distributed across NUMA nodes
 * 
 * Allocates a large memory pool and distributes it across available
 * NUMA nodes for optimal parallel access patterns.
 * 
 * @param total_size Total size of memory pool to allocate
 * @param numa_ctx NUMA allocation context
 * @return Pointer to allocated memory pool, or NULL on failure
 */
void* ggml_numa_alloc_context_memory(size_t total_size, ggml_numa_alloc_context_t* numa_ctx);

/**
 * @brief Free NUMA-allocated memory
 * 
 * Frees memory previously allocated by NUMA allocator functions.
 * Updates allocation tracking in the context.
 * 
 * @param ptr Pointer to memory to free (NULL is safe)
 * @param numa_ctx NUMA allocation context used for allocation
 */
void ggml_numa_aligned_free(void* ptr, ggml_numa_alloc_context_t* numa_ctx);

/**
 * @brief Get NUMA allocation statistics
 * 
 * Prints detailed allocation statistics including per-node usage,
 * total allocations, and strategy effectiveness.
 * 
 * @param ctx NUMA allocation context to report on
 */
void ggml_numa_alloc_stats(const ggml_numa_alloc_context_t* ctx);

/**
 * @brief Enable or disable debug logging
 * 
 * Controls verbose debug output for NUMA allocation operations.
 * Useful for debugging allocation patterns and NUMA effectiveness.
 * 
 * @param ctx NUMA allocation context
 * @param enabled true to enable debug logging, false to disable
 */
void ggml_numa_alloc_set_debug(ggml_numa_alloc_context_t* ctx, bool enabled);

#ifdef __cplusplus
}
#endif

/**
 * @brief Check if memory was allocated by NUMA allocator
 * 
 * Determines whether a memory pointer was allocated using the
 * NUMA allocator functions. Useful for mixed allocation scenarios.
 * 
 * @param ptr Memory pointer to check
 * @return true if allocated by NUMA allocator, false otherwise
 */
bool ggml_numa_is_numa_allocated(void* ptr);

/**
 * @brief Free NUMA-allocated memory (context-free version)
 * 
 * Frees memory previously allocated by NUMA allocator without
 * requiring the original allocation context. Less efficient than
 * ggml_numa_aligned_free() but more convenient.
 * 
 * @param ptr Pointer to memory to free (NULL is safe)
 */
void ggml_numa_free(void* ptr);

#endif // GGML_NUMA_ALLOCATOR_H
