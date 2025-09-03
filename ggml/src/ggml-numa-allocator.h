/**
 * @file ggml-numa-allocator.h
 * @brief NUMA-Aware Memory Allocator for ggml (Internal Source Header)
 * 
 * This is the internal source header for NUMA-aware allocation functions.
 * It provides a simplified interface compared to the public header, focusing
 * on the core allocation strategies used internally by ggml.
 * 
 * @author GGML NUMA Team
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
 * @brief NUMA allocation strategy types (Internal)
 * 
 * Simplified allocation strategies for internal ggml use.
 * This is a subset of the full strategy enumeration.
 */
typedef enum {
    GGML_NUMA_ALLOC_STRATEGY_LOCAL,          /**< Allocate on current/specified node (for ISOLATE) */
    GGML_NUMA_ALLOC_STRATEGY_MIRROR          /**< Create copies on all nodes (for MIRROR mode) */
} ggml_numa_alloc_strategy_t;

/**
 * @brief NUMA allocation context structure (Internal)
 * 
 * Simplified context structure for internal ggml NUMA allocation tracking.
 */
typedef struct {
    ggml_numa_alloc_strategy_t strategy;     /**< Current allocation strategy */
    int num_numa_nodes;                      /**< Number of available NUMA nodes */
    size_t total_allocated;                  /**< Total bytes allocated across all nodes */
    size_t per_node_allocated[8];            /**< Per-node allocation tracking (max 8 nodes) */
    bool debug_enabled;                      /**< Enable debug logging */
} ggml_numa_alloc_context_t;

/**
 * @brief Initialize NUMA allocator context
 * 
 * @param ctx Pointer to NUMA allocation context to initialize
 * @param strategy Allocation strategy to use
 * @return true on successful initialization, false on error
 */
bool ggml_numa_alloc_init(ggml_numa_alloc_context_t* ctx, ggml_numa_alloc_strategy_t strategy);

/**
 * @brief NUMA-aware aligned memory allocation
 * 
 * @param size Number of bytes to allocate
 * @param numa_ctx NUMA allocation context (NULL for default)
 * @return Pointer to allocated memory, or NULL on failure
 */
void* ggml_numa_aligned_malloc(size_t size, ggml_numa_alloc_context_t* numa_ctx);

/**
 * @brief NUMA-aware memory allocation with explicit node preference
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
 * @param total_size Total size of memory pool to allocate
 * @param numa_ctx NUMA allocation context (void* for compatibility)
 * @return Pointer to allocated memory pool, or NULL on failure
 */
void* ggml_numa_alloc_context_memory(size_t total_size, void* numa_ctx);

/**
 * @brief Free NUMA-allocated memory
 * 
 * @param ptr Pointer to memory to free (NULL is safe)
 * @param numa_ctx NUMA allocation context used for allocation
 */
void ggml_numa_aligned_free(void* ptr, ggml_numa_alloc_context_t* numa_ctx);

/**
 * @brief Get NUMA allocation statistics
 * 
 * @param ctx NUMA allocation context to report on
 */
void ggml_numa_alloc_stats(const ggml_numa_alloc_context_t* ctx);

/**
 * @brief Enable or disable debug logging
 * 
 * @param ctx NUMA allocation context
 * @param enabled true to enable debug logging, false to disable
 */
void ggml_numa_alloc_set_debug(ggml_numa_alloc_context_t* ctx, bool enabled);

/**
 * @brief Check if memory was allocated by NUMA allocator
 * 
 * @param ptr Memory pointer to check
 * @return true if allocated by NUMA allocator, false otherwise
 */
bool ggml_numa_is_numa_allocated(void* ptr);

/**
 * @brief Free NUMA-allocated memory (context-free version)
 * 
 * @param ptr Pointer to memory to free (NULL is safe)
 */
void ggml_numa_free(void* ptr);

/**
 * @brief Assert that memory allocation is on the expected NUMA node
 * 
 * @param ptr Memory pointer to check
 * @param expected_node Expected NUMA node ID
 * @param context Description of where this assertion is being called from
 */
void ggml_numa_assert_allocation(void* ptr, int expected_node, const char* context);

/**
 * @brief Get the NUMA node of a memory address
 * 
 * @param ptr Memory pointer to check
 * @return NUMA node ID (0-based), or -1 on error
 */
int get_memory_numa_node(void* ptr);

/**
 * @brief Force linking of NUMA allocator symbols (internal use)
 * 
 * Prevents linker dead code elimination of NUMA allocator symbols.
 */
void ggml_force_link_numa_allocator_symbols(void);

#ifdef __cplusplus
}
#endif

#endif // GGML_NUMA_ALLOCATOR_H
