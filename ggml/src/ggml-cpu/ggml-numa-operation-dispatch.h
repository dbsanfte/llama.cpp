/**
 * NUMA Operation Dispatch Infrastructure
 * 
 * This header defines the intelligent dispatcher system that routes operations
 * to appropriate execution strategies while preserving existing thread synchronization.
 * 
 * Architecture:
 * - Operation Registry: Catalog of all supported operations with their characteristics
 * - Execution Strategies: Different approaches for different operation types
 * - Smart Dispatcher: Routes operations based on type, size, and system state
 * - Chunking Framework: Flexible data/work partitioning for parallelization
 */

#ifndef GGML_NUMA_OPERATION_DISPATCH_H
#define GGML_NUMA_OPERATION_DISPATCH_H

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-numa-coordinator.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct ggml_coordinator_thread;
struct ggml_numa_coordinator_manager;
struct ggml_cgraph;

//
// Main GGML Integration Function
//

// Main entry point - replaces ggml_graph_compute for NUMA-aware computation
enum ggml_status ggml_numa_graph_compute(struct ggml_cgraph * cgraph, int n_threads);

// Enhanced entry point with virtual NUMA support for testing
enum ggml_status ggml_numa_graph_compute_with_virtual(struct ggml_cgraph * cgraph, int n_threads, bool force_virtual_numa);

// Phase 1: Single-threaded fallback system for complete operation coverage
enum ggml_status ggml_numa_execute_operation_fallback(struct ggml_tensor * tensor, struct ggml_cplan * cplan);

//
// Coordinator Interface for Dispatcher
//
// This interface provides controlled access to coordinator resources
// without exposing internal coordinator implementation details

typedef struct {
    // Get coordinator threadpool for a specific NUMA node
    struct ggml_threadpool * (*get_numa_threadpool)(struct ggml_numa_coordinator_manager * manager, int numa_node);
    
    // Get thread count for a specific NUMA node  
    int (*get_numa_thread_count)(struct ggml_numa_coordinator_manager * manager, int numa_node);
    
    // Ensure adequate work buffer for a coordinator
    bool (*ensure_work_buffer)(struct ggml_numa_coordinator_manager * manager, int numa_node, size_t required_size);
    
    // Get work buffer pointer for a coordinator
    void * (*get_work_buffer)(struct ggml_numa_coordinator_manager * manager, int numa_node);
    
    // Get work buffer size for a coordinator
    size_t (*get_work_buffer_size)(struct ggml_numa_coordinator_manager * manager, int numa_node);
    
    // Submit work to a specific coordinator (for complex execution strategies)
    int (*submit_work)(struct ggml_numa_coordinator_manager * manager, struct ggml_tensor * operation, int target_numa_node);
    
    // Submit data parallel work across multiple coordinators
    int (*submit_data_parallel_work)(struct ggml_numa_coordinator_manager * manager, struct ggml_tensor * operation, 
                                   int work_group_id, const int * target_nodes, int num_target_nodes);
} ggml_numa_coordinator_interface_t;

//
// Operation Classification System
//

typedef enum {
    NUMA_EXECUTION_SINGLE_NODE,     // Execute on primary node only
    NUMA_EXECUTION_DATA_PARALLEL,   // Distribute data across nodes
    NUMA_EXECUTION_TASK_PARALLEL,   // Distribute different tasks across nodes  
    NUMA_EXECUTION_HYBRID,          // Combination of strategies
    NUMA_EXECUTION_CUSTOM           // Operation-specific strategy
} ggml_numa_execution_strategy_t;

typedef enum {
    NUMA_OP_COMPLEXITY_SIMPLE,      // O(n) operations, high parallelization potential
    NUMA_OP_COMPLEXITY_MODERATE,    // O(n log n) or simple O(n²), good parallelization
    NUMA_OP_COMPLEXITY_COMPLEX,     // Complex dependencies, limited parallelization
    NUMA_OP_COMPLEXITY_SEQUENTIAL   // Must be sequential, no parallelization
} ggml_numa_op_complexity_t;

typedef enum {
    NUMA_OP_MEMORY_BOUND,           // Performance limited by memory bandwidth
    NUMA_OP_COMPUTE_BOUND,          // Performance limited by computation
    NUMA_OP_CACHE_SENSITIVE,        // Sensitive to cache locality
    NUMA_OP_MIXED_WORKLOAD          // Mixed characteristics
} ggml_numa_op_workload_type_t;

//
// Chunking and Partitioning Framework
//

typedef struct {
    // Data dimension information
    int64_t total_elements;         // Total elements to process
    int64_t element_size;           // Size of each element in bytes
    
    // Tensor shape for multi-dimensional chunking
    int64_t ne[GGML_MAX_DIMS];      // Tensor dimensions
    int n_dims;                     // Number of dimensions
    
    // Parallelization parameters
    int numa_nodes;                 // Number of available NUMA nodes
    int threads_per_node;           // Threads available per node
    
    // Performance hints
    size_t l3_cache_size;           // L3 cache size for chunking decisions
    size_t memory_bandwidth;        // Memory bandwidth estimate
} ggml_numa_work_context_t;

typedef struct {
    // Chunk identification
    int chunk_id;                   // Unique chunk identifier
    int numa_node;                  // Target NUMA node for this chunk
    
    // Data boundaries
    int64_t start_offset;           // Starting element offset
    int64_t element_count;          // Number of elements in this chunk
    
    // Multi-dimensional chunk bounds
    int64_t dim_start[GGML_MAX_DIMS]; // Start position in each dimension
    int64_t dim_count[GGML_MAX_DIMS]; // Count in each dimension
    
    // Execution parameters
    int thread_count;               // Threads assigned to this chunk
    bool requires_synchronization;  // Whether chunk needs sync barriers
} ggml_numa_work_chunk_t;

//
// Operation Handler Interface
//

// Function pointer types for operation handlers
typedef enum ggml_status (*ggml_numa_op_analyzer_fn)(
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context,
    ggml_numa_execution_strategy_t * strategy,
    int * recommended_chunks
);

typedef enum ggml_status (*ggml_numa_op_chunker_fn)(
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context,
    int num_chunks,
    ggml_numa_work_chunk_t * chunks
);

typedef enum ggml_status (*ggml_numa_op_executor_fn)(
    struct ggml_coordinator_thread * coordinator,
    const struct ggml_tensor * operation,
    const ggml_numa_work_chunk_t * chunk
);

// Complete operation handler definition
typedef struct {
    enum ggml_op operation_type;
    
    // Operation characteristics
    ggml_numa_execution_strategy_t default_strategy;
    ggml_numa_op_complexity_t complexity;
    ggml_numa_op_workload_type_t workload_type;
    
    // Size thresholds for parallelization decisions
    int64_t min_elements_for_parallel;
    int64_t optimal_chunk_size;
    
    // Handler functions
    ggml_numa_op_analyzer_fn analyze;      // Analyze operation and recommend strategy
    ggml_numa_op_chunker_fn chunk;        // Partition work into chunks
    ggml_numa_op_executor_fn execute;     // Execute a single chunk
    
    // Performance characteristics
    float parallel_efficiency_estimate;    // Expected parallel efficiency (0.0-1.0)
    bool requires_synchronization;         // Whether operation needs cross-node sync
    bool supports_in_place;               // Whether operation supports in-place computation
} ggml_numa_operation_handler_t;

//
// Operation Registry and Dispatch System
//

// Initialize the operation dispatch system
void ggml_numa_dispatch_init(void);

// Register an operation handler
void ggml_numa_dispatch_register_handler(const ggml_numa_operation_handler_t * handler);

// Get handler for a specific operation type
const ggml_numa_operation_handler_t * ggml_numa_dispatch_get_handler(enum ggml_op operation_type);

// Main dispatch function - routes operation to appropriate execution strategy
enum ggml_status ggml_numa_dispatch_operation(
    struct ggml_numa_coordinator_manager * manager,
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context
);

//
// Smart Chunking Utilities
//

// Create work context from current system state
ggml_numa_work_context_t ggml_numa_create_work_context(
    const struct ggml_tensor * operation,
    struct ggml_numa_coordinator_manager * manager
);

// Automatic chunking based on operation characteristics
int ggml_numa_auto_chunk_operation(
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context,
    ggml_numa_work_chunk_t * chunks,
    int max_chunks
);

// Chunk validation and optimization
bool ggml_numa_validate_chunks(
    const ggml_numa_work_chunk_t * chunks,
    int num_chunks,
    const ggml_numa_work_context_t * context
);

//
// Built-in Operation Handlers
//

// Element-wise operations (ADD, MUL, DIV, etc.)
extern const ggml_numa_operation_handler_t ggml_numa_handler_elementwise;

// Matrix multiplication operations
extern const ggml_numa_operation_handler_t ggml_numa_handler_mul_mat;

// Reduction operations (SUM, MEAN, etc.)
extern const ggml_numa_operation_handler_t ggml_numa_handler_reduction;

// Normalization operations (NORM, RMS_NORM, etc.)
extern const ggml_numa_operation_handler_t ggml_numa_handler_normalization;

// Activation functions (GELU, SILU, RELU, etc.)
extern const ggml_numa_operation_handler_t ggml_numa_handler_activation;

// Complex operations (ROPE, ATTENTION, etc.) - single node by default
extern const ggml_numa_operation_handler_t ggml_numa_handler_complex;

//
// Performance Monitoring
//

typedef struct {
    // Operation statistics
    int64_t total_operations;
    int64_t parallelized_operations;
    int64_t fallback_operations;        // Phase 1: Operations executed via single-threaded fallback
    
    // Timing statistics
    int64_t total_execution_time_us;
    int64_t coordination_overhead_us;
    int64_t synchronization_time_us;
    
    // Efficiency metrics
    float average_parallel_efficiency;
    float numa_utilization;
    
    // Per-operation type statistics
    int64_t op_counts[GGML_OP_COUNT];
    int64_t op_times_us[GGML_OP_COUNT];
} ggml_numa_dispatch_stats_t;

// Get current dispatch statistics
const ggml_numa_dispatch_stats_t * ggml_numa_dispatch_get_stats(void);

// Reset statistics
void ggml_numa_dispatch_reset_stats(void);

#ifdef __cplusplus
}
#endif

#endif // GGML_NUMA_OPERATION_DISPATCH_H
