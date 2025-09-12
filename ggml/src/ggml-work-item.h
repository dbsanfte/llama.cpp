/**
 * Simplified Work Item Abstraction for NUMA Dispatcher-Coordinator Architecture
 * 
 * This provides the simple function pointer + payload pattern suggested by Gemini
 * for clean separation between dispatcher (intelligence) and coordinator (execution).
 * 
 * Enhanced to support all 193 ggml operations with flexible parallelization strategies.
 */

#ifndef GGML_WORK_ITEM_H
#define GGML_WORK_ITEM_H

#include <stdbool.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Parallelization strategy for operations
 * Determines how work can be divided across NUMA nodes and threads
 */
typedef enum {
    GGML_PARALLEL_NONE,        // Single-threaded only, no parallelization
    GGML_PARALLEL_ELEMENT,     // Element-wise parallelizable (ADD, MUL, etc.)
    GGML_PARALLEL_ROWS,        // Row-wise parallelizable (matrix operations)
    GGML_PARALLEL_SEQUENCE,    // Sequence-length splitting (ROPE, attention)
    GGML_PARALLEL_BATCH,       // Batch dimension splitting
    GGML_PARALLEL_CUSTOM       // Operation-specific strategy
} ggml_parallel_strategy_t;

/**
 * Work synchronization requirements
 * Determines coordination needs between work items
 */
typedef enum {
    GGML_SYNC_NONE,           // No synchronization needed
    GGML_SYNC_BARRIER,        // Requires barrier after completion
    GGML_SYNC_SEQUENTIAL,     // Must execute in sequence
    GGML_SYNC_NUMA_LOCAL      // Must complete locally before cross-NUMA
} ggml_sync_requirement_t;

/**
 * Simple task function pointer type
 * Takes a generic payload pointer and executes the task
 */
typedef void (*ggml_task_t)(void *task_data);

/**
 * Enhanced work item for all 193 operations
 * Contains function pointer, data payload, and execution metadata
 */
typedef struct ggml_work_item_t {
    ggml_task_t func;                    // Function to execute
    void * data;                         // Operation-specific payload
    size_t data_size;                    // Size of payload for cleanup
    int work_id;                         // Unique work ID for tracking
    int target_numa_node;                // Target NUMA node (-1 for any)
    
    // Enhanced metadata for complex operations
    ggml_parallel_strategy_t strategy;   // How this work can be parallelized
    ggml_sync_requirement_t sync_req;    // Synchronization requirements
    int chunk_start;                     // Start index for this chunk
    int chunk_end;                       // End index for this chunk
    int total_chunks;                    // Total chunks in this batch
    int priority;                        // Execution priority (0=highest)
    
    struct ggml_work_item_t * next;      // Next item in queue (for internal use)
} ggml_work_item_t;

/**
 * Work completion callback type
 * Called when a work item finishes execution
 */
typedef void (*ggml_work_completion_callback_t)(int work_id, int numa_node, void *user_data);

/**
 * Synchronization barrier for coordinating work completion
 * Used to wait for groups of work items to finish
 */
typedef struct {
    atomic_int remaining_work;      // Number of work items still pending
    atomic_bool completed;          // All work in barrier is done
    int barrier_id;                 // Unique barrier ID
} ggml_work_barrier_t;

/**
 * Work batch - collection of related work items that should be executed together
 * Useful for operations that need to be split across NUMA nodes
 */
typedef struct {
    ggml_work_item_t * items;       // Array of work items
    int num_items;                  // Number of work items in batch
    ggml_work_barrier_t * barrier;  // Optional barrier for synchronization
    int batch_id;                   // Unique batch ID
} ggml_work_batch_t;

// =============================================================================
// Work Item Creation and Management Functions
// =============================================================================

/**
 * Create a simple work item with function pointer and payload
 * 
 * @param func Function to execute
 * @param data Payload data (will be copied if data_size > 0)
 * @param data_size Size of payload data (0 if data is just a pointer)
 * @param target_numa_node Target NUMA node (-1 for any available)
 * @return Work item or NULL on failure
 */
ggml_work_item_t * ggml_work_item_new(ggml_task_t func, void * data, size_t data_size, int target_numa_node);

/**
 * Create an enhanced work item with full operation metadata
 * For complex operations that need specific parallelization and synchronization
 * 
 * @param func Function to execute
 * @param data Payload data (will be copied if data_size > 0)
 * @param data_size Size of payload data
 * @param target_numa_node Target NUMA node (-1 for any)
 * @param strategy Parallelization strategy
 * @param sync_req Synchronization requirements
 * @param chunk_start Start index for this work chunk
 * @param chunk_end End index for this work chunk  
 * @param total_chunks Total number of chunks in batch
 * @param priority Execution priority (0=highest)
 * @return Enhanced work item or NULL on failure
 */
ggml_work_item_t * ggml_work_item_new_enhanced(
    ggml_task_t func, 
    void * data, 
    size_t data_size, 
    int target_numa_node,
    ggml_parallel_strategy_t strategy,
    ggml_sync_requirement_t sync_req,
    int chunk_start,
    int chunk_end,
    int total_chunks,
    int priority
);

/**
 * Free a work item and its payload data
 * 
 * @param item Work item to free
 */
void ggml_work_item_free(ggml_work_item_t * item);

/**
 * Create a work batch containing multiple related work items
 * 
 * @param items Array of work items
 * @param num_items Number of items in array
 * @param use_barrier Whether to create a barrier for synchronization
 * @return Work batch or NULL on failure
 */
ggml_work_batch_t * ggml_work_batch_new(ggml_work_item_t * items, int num_items, bool use_barrier);

/**
 * Free a work batch and all its work items
 * 
 * @param batch Work batch to free
 */
void ggml_work_batch_free(ggml_work_batch_t * batch);

// =============================================================================
// Synchronization Functions
// =============================================================================

/**
 * Create a work barrier for synchronizing multiple work items
 * 
 * @param num_work_items Number of work items this barrier will wait for
 * @return Barrier or NULL on failure
 */
ggml_work_barrier_t * ggml_work_barrier_new(int num_work_items);

/**
 * Signal that one work item has completed
 * 
 * @param barrier Barrier to signal
 */
void ggml_work_barrier_signal(ggml_work_barrier_t * barrier);

/**
 * Wait for all work items in a barrier to complete
 * 
 * @param barrier Barrier to wait on
 * @param timeout_ms Timeout in milliseconds (0 for no timeout)
 * @return true if all work completed, false if timeout
 */
bool ggml_work_barrier_wait(ggml_work_barrier_t * barrier, int timeout_ms);

/**
 * Check if barrier is completed without blocking
 * 
 * @param barrier Barrier to check
 * @return true if completed, false otherwise
 */
bool ggml_work_barrier_is_completed(ggml_work_barrier_t * barrier);

/**
 * Free a work barrier
 * 
 * @param barrier Barrier to free
 */
void ggml_work_barrier_free(ggml_work_barrier_t * barrier);

// =============================================================================
// Operation-Specific Payload Structures
// =============================================================================
// These are examples of payload structures for different operation types
// Each operation handler will define its own payload struct

/**
 * Example payload for ROPE operation
 * Split work by sequence length (tokens)
 */
typedef struct {
    struct ggml_tensor * dst;           // Output tensor
    struct ggml_tensor * src;           // Input tensor
    struct ggml_tensor * pos;           // Position tensor
    int start_token;                    // First token to process
    int end_token;                      // Last token to process (exclusive)
    float theta_base;                   // RoPE theta base parameter
    float freq_scale;                   // Frequency scaling factor
    int32_t n_ctx;                      // Context length
    int32_t n_dims;                     // Number of dimensions
} ggml_rope_task_data_t;

/**
 * Example payload for matrix multiplication
 * Split work by output rows
 */
typedef struct {
    struct ggml_tensor * dst;           // Output matrix
    struct ggml_tensor * src0;          // Left input matrix
    struct ggml_tensor * src1;          // Right input matrix
    int start_row;                      // First row to compute
    int end_row;                        // Last row to compute (exclusive)
    bool transpose_src0;                // Whether to transpose src0
    bool transpose_src1;                // Whether to transpose src1
} ggml_matmul_task_data_t;

/**
 * Example payload for element-wise operations (ADD, MUL, etc.)
 * Split work by elements
 */
typedef struct {
    struct ggml_tensor * dst;           // Output tensor
    struct ggml_tensor * src0;          // First input tensor
    struct ggml_tensor * src1;          // Second input tensor (may be NULL)
    size_t start_element;               // First element to process
    size_t end_element;                 // Last element to process (exclusive)
    enum ggml_op op_type;               // Operation type (ADD, MUL, etc.)
} ggml_elementwise_task_data_t;

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Get next unique work ID for tracking
 * Thread-safe atomic counter
 * 
 * @return Unique work ID
 */
int ggml_work_item_get_next_id(void);

/**
 * Get next unique batch ID for tracking
 * Thread-safe atomic counter
 * 
 * @return Unique batch ID
 */
int ggml_work_batch_get_next_id(void);

/**
 * Get next unique barrier ID for tracking
 * Thread-safe atomic counter
 * 
 * @return Unique barrier ID
 */
int ggml_work_barrier_get_next_id(void);

#ifdef __cplusplus
}
#endif

#endif // GGML_WORK_ITEM_H
