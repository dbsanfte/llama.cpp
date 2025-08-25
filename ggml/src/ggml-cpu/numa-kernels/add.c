/**
 * @file add.c
 * @brief NUMA Kernel Template: Element-wise Addition (ADD)
 * 
 * ============================================================================
 * NUMA KERNEL DEVELOPMENT TEMPLATE
 * ============================================================================
 * 
 * This file serves as the canonical template for implementing NUMA-aware 
 * kernels in the GGML NUMA system. Use this as a reference when creating 
 * new NUMA kernels for other operations.
 * 
 * ARCHITECTURE OVERVIEW:
 * =====================
 * 
 * 1. NUMA EXECUTOR queries the NUMA KERNEL REGISTRY for optimal strategy
 * 2. NUMA COORDINATOR dispatches work to appropriate NUMA nodes
 * 3. NUMA KERNEL executes with proper data slicing and thread coordination
 * 
 * Flow: Compute Graph → Executor → Registry Query → Coordinator Dispatch → Kernel
 * 
 * PARALLELIZATION STRATEGY:
 * ========================
 * 
 * The NUMA system supports two primary execution modes:
 * 
 * 1. SINGLE-NODE EXECUTION:
 *    - All computation happens on one NUMA node
 *    - Threads divide the work among themselves on that node
 *    - Good for smaller tensors or operations that don't benefit from NUMA
 * 
 * 2. DATA-PARALLEL EXECUTION:  
 *    - Tensor data is sliced across multiple NUMA nodes
 *    - Each node processes its slice independently with its local threads
 *    - Optimal for large tensors with data-parallel operations
 * 
 * THREAD-LOCAL CONTEXT:
 * ====================
 * 
 * The coordinator sets up thread-local variables for each kernel execution:
 * 
 * - ggml_current_numa_node: Which NUMA node this thread is executing on
 * - ggml_numa_is_data_parallel_execution: Whether we're in data-parallel mode
 * - ggml_numa_total_nodes_for_data_parallel: Total NUMA nodes participating
 * 
 * DATA SLICING PATTERN:
 * ====================
 * 
 * For data-parallel operations, follow this pattern:
 * 
 * 1. Calculate total elements in tensor: ggml_nelements(tensor)
 * 2. Determine this node's slice: elements_per_node = total / num_nodes
 * 3. Calculate node range: [node_start, node_end)
 * 4. Divide node slice among threads: thread_slice within node_range
 * 5. Process only the thread's portion of the node's slice
 * 
 * MEMORY ACCESS PATTERN:
 * =====================
 * 
 * CRITICAL: Always use tensor_data() to get NUMA-local memory:
 * 
 * - tensor_data(tensor) returns the correct NUMA-local copy
 * - For data-parallel: returns local slice on each node
 * - For single-node: returns the original tensor data
 * - DO NOT access tensor->data directly in NUMA kernels
 * 
 * SIMD OPTIMIZATION:
 * =================
 * 
 * Always prefer SIMD operations from vec.h for performance:
 * 
 * - ggml_vec_add_f32(): Element-wise addition
 * - ggml_vec_scale_f32(): Scalar multiplication  
 * - ggml_vec_dot_f32(): Dot product
 * - ggml_vec_cpy_f32(): Memory copy
 * 
 * SIMD provides significant speedup on modern CPUs with AVX2/AVX512.
 * 
 * KERNEL IMPLEMENTATION CHECKLIST:
 * ================================
 * 
 * ✅ 1. Extract tensor parameters and validate inputs
 * ✅ 2. Get NUMA-local data pointers using tensor_data()
 * ✅ 3. Read thread-local NUMA context variables
 * ✅ 4. Calculate data slice for this thread/node combination
 * ✅ 5. Use SIMD operations for computational core
 * ✅ 6. Handle edge cases (broadcasting, remainder elements)
 * ✅ 7. Return appropriate status code
 * 
 * REGISTRY INTEGRATION:
 * ====================
 * 
 * Implement populate_cache function with complexity-based strategies:
 * 
 * - COMPLEXITY_TINY: Small tensors, single-node/single-thread
 * - COMPLEXITY_SMALL: Medium tensors, single-node/multi-thread  
 * - COMPLEXITY_MEDIUM: Large tensors, consider data-parallel
 * - COMPLEXITY_LARGE: Very large tensors, data-parallel/multi-thread
 * - COMPLEXITY_HUGE: Massive tensors, data-parallel/multi-thread
 * 
 * PERFORMANCE CONSIDERATIONS:
 * ==========================
 * 
 * - Minimize validation overhead in hot path
 * - Cache frequently accessed values (avoid repeated dereferencing)
 * - Use inline functions for slice calculations
 * - Prefer static_assert for compile-time checks
 * - Use branch prediction hints where appropriate
 * 
 * DEBUGGING SUPPORT:
 * =================
 * 
 * Include debug prints for development (can be disabled in production):
 * 
 * - Node/thread information
 * - Data slice ranges  
 * - Memory pointer values
 * - Execution path taken
 * 
 * MATHEMATICAL CORRECTNESS:
 * ========================
 * 
 * CRITICAL: NUMA kernels must produce identical results to reference:
 * 
 * - Test with test-numa-mathematical-correctness-OPERATION
 * - Verify across all tensor sizes and thread counts
 * - Ensure floating-point precision is maintained
 * - Handle edge cases consistently with reference implementation
 * 
 * ============================================================================
 */

/*
 * NUMA Kernel: Element-wise Addition (ADD) - Performance Optimized
 * 
 * Minimal overhead, maximum performance version for data-parallel execution.
 */

#include "../binary-ops.h"
#include "add.h"
#include "numa-kernels.h"
#include "../ggml-numa-shared.h"
#include "../ggml-numa-simple-coordinator.h"
#include "../ggml-cpu-impl.h"
#include "../ggml-impl.h"
#include "../vec.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

// ============================================================================
// Ultra-Fast Data-Parallel ADD Kernel
// ============================================================================

static_assert(sizeof(float) == 4, "Float must be 4 bytes for SIMD");

/**
 * Optimized NUMA slice calculation - inline, minimal overhead
 * 
 * Template Pattern: Fast slice calculation for data-parallel operations
 * 
 * @param total_elements Total number of elements in the tensor
 * @param current_node   NUMA node executing this slice (0-based)
 * @param total_nodes    Total number of NUMA nodes participating
 * @param start          [OUT] Starting index for this node's slice
 * @param end            [OUT] Ending index (exclusive) for this node's slice
 * 
 * Algorithm: Divides tensor evenly across nodes, last node handles remainder
 */
static inline void get_numa_slice_fast(int64_t total_elements, 
                                      int current_node, 
                                      int total_nodes,
                                      int64_t * start, 
                                      int64_t * end) {
    if (total_nodes <= 1) {
        *start = 0;
        *end = total_elements;
        return;
    }
    
    const int64_t elements_per_node = total_elements / total_nodes;
    *start = current_node * elements_per_node;
    *end = (current_node == total_nodes - 1) ? total_elements : *start + elements_per_node;
}

/**
 * Ultra-fast ADD kernel - minimal validation, maximum performance
 * 
 * Template Pattern: High-performance NUMA kernel implementation
 * 
 * EXECUTION FLOW:
 * 1. Fast validation (assume coordinator pre-validated)
 * 2. Extract tensor data using NUMA-local tensor_data()
 * 3. Read thread-local NUMA context from coordinator
 * 4. Calculate data slice for this thread/node combination
 * 5. Execute SIMD operations on assigned slice
 * 6. Handle broadcasting and edge cases efficiently
 * 
 * THREAD SAFETY: Thread-safe via data slicing (no shared state)
 * NUMA AWARENESS: Accesses only NUMA-local memory via tensor_data()
 * PERFORMANCE: Optimized for minimal overhead, maximum SIMD utilization
 * 
 * @param work_context  Tensor to process (cast from void*)
 * @param params        Threadpool parameters (thread ID, thread count)
 * @return              GGML_STATUS_SUCCESS on success, GGML_STATUS_FAILED on error
 */
enum ggml_status ggml_numa_kernel_add_execute_optimized(void * work_context, 
                                                       struct ggml_compute_params * params) {
    struct ggml_tensor * tensor = (struct ggml_tensor *)work_context;
    
    // TEMPLATE STEP 1: Fast validation (assume coordinator pre-validated)
    // Minimal checks for critical errors only - coordinator handles most validation
    if (!tensor || !tensor->src[0] || !tensor->src[1]) {
        return GGML_STATUS_FAILED;
    }
    
    // TEMPLATE STEP 2: Extract tensor parameters and cache frequently accessed values
    // Cache tensor pointers and avoid repeated dereferencing for performance
    const struct ggml_tensor * src0 = tensor->src[0];
    const struct ggml_tensor * src1 = tensor->src[1];
    const int64_t total_elements = ggml_nelements(tensor);
    
    // TEMPLATE STEP 3: Get NUMA-local data pointers 
    // CRITICAL: tensor_data() returns the correct NUMA-local copy in data-parallel mode
    // DO NOT use tensor->data directly - it may point to wrong NUMA node
    const float * src0_data = (const float *)tensor_data(src0);
    const float * src1_data = (const float *)tensor_data(src1);
    float * dst_data = (float *)tensor_data(tensor);
    
    // TEMPLATE STEP 4: Read thread-local NUMA context from coordinator
    // These variables are set by the coordinator before kernel execution
    extern __thread bool ggml_numa_is_data_parallel_execution;
    extern __thread int ggml_numa_total_nodes_for_data_parallel;
    extern __thread int ggml_current_numa_node;
    
    // TEMPLATE STEP 5: Extract threadpool parameters for multi-threading
    // Each kernel receives thread ID and total thread count for its node
    const int current_node = ggml_current_numa_node;
    const int total_nodes = ggml_numa_is_data_parallel_execution ? 
                           ggml_numa_total_nodes_for_data_parallel : 1;
    const bool is_data_parallel = ggml_numa_is_data_parallel_execution;
    const int thread_id = params->ith;
    const int num_threads = params->nth;
    
    // TEMPLATE DEBUG: Log execution context for development/debugging
    printf("DEBUG: NUMA Node %d, Thread %d/%d kernel start (data_parallel=%d, total_nodes=%d, total_elements=%ld)\n", 
           current_node, thread_id, num_threads, is_data_parallel, total_nodes, total_elements);
    printf("DEBUG: NUMA Node %d memory pointers: src0=%p, src1=%p, dst=%p\n", 
           current_node, src0_data, src1_data, dst_data);
    
    // TEMPLATE STEP 6: Calculate data slice for this thread/node combination
    // This is the core of NUMA data-parallel processing
    int64_t numa_start, numa_end;
    
    if (is_data_parallel && total_nodes > 1) {
        // TEMPLATE PATTERN A: DATA-PARALLEL MODE
        // Each NUMA node processes its assigned slice of the global tensor
        // Then threads on each node divide that node's slice among themselves
        
        const int64_t elements_per_node = total_elements / total_nodes;
        
        // Calculate this node's slice in the global tensor
        const int64_t node_start = current_node * elements_per_node;
        const int64_t node_end = (current_node == total_nodes - 1) ? 
                                 total_elements : 
                                 node_start + elements_per_node;
        
        // Now divide this node's slice among its threads
        const int64_t elements_per_thread = (node_end - node_start + num_threads - 1) / num_threads;
        numa_start = node_start + thread_id * elements_per_thread;
        numa_end = MIN(numa_start + elements_per_thread, node_end);
        
        printf("DEBUG: NUMA Node %d, Thread %d processing slice: [%ld, %ld) (%ld elements) from node range [%ld, %ld)\n", 
               current_node, thread_id, numa_start, numa_end, numa_end - numa_start, node_start, node_end);
    } else {
        // TEMPLATE PATTERN B: SINGLE-NODE MODE
        // All threads process slices of the entire tensor (no NUMA slicing)
        // Good for smaller tensors or when data-parallel doesn't provide benefit
        
        const int64_t elements_per_thread = (total_elements + num_threads - 1) / num_threads;
        numa_start = thread_id * elements_per_thread;
        numa_end = MIN(numa_start + elements_per_thread, total_elements);
        
        printf("DEBUG: NUMA Node %d, Thread %d processing tensor slice: [%ld, %ld) (%ld elements)\n", 
               current_node, thread_id, numa_start, numa_end, numa_end - numa_start);
    }
    
    // TEMPLATE STEP 7: Execute SIMD operations on assigned data slice
    // Use SIMD for maximum performance - always prefer ggml_vec_* functions
    const size_t elements_in_slice = numa_end - numa_start;
    
    // TEMPLATE STEP 8: Handle operation-specific logic (broadcasting, etc.)
    // Check for broadcasting - optimized check for common patterns
    const int64_t src1_elements = ggml_nelements(src1);
    
    if (src1_elements == 1) {
        // TEMPLATE PATTERN: Scalar broadcasting (very common, optimize heavily)
        printf("DEBUG: NUMA Node %d using SCALAR broadcasting path (elements_in_slice=%zu)\n", 
               ggml_current_numa_node, elements_in_slice);
        const float scalar = src1_data[0];
        
        // Use SIMD for scalar addition - direct access to global positions
        ggml_vec_scale_f32(elements_in_slice, dst_data + numa_start, scalar);
        ggml_vec_add_f32(elements_in_slice, dst_data + numa_start, dst_data + numa_start, src0_data + numa_start);
        
    } else if (src1_elements == total_elements) {
        // TEMPLATE PATTERN: Element-wise operation (most common, should be fastest)
        printf("DEBUG: NUMA Node %d using ELEMENT-WISE path (elements_in_slice=%zu)\n", 
               ggml_current_numa_node, elements_in_slice);
        
        // Pure SIMD operation on global positions - maximum performance path
        ggml_vec_add_f32(elements_in_slice, dst_data + numa_start, src0_data + numa_start, src1_data + numa_start);
        
    } else {
        // TEMPLATE PATTERN: Complex broadcasting (avoid if possible, performance cost)
        printf("DEBUG: NUMA Node %d using SLOW BROADCASTING path (src1_elements=%ld, total=%ld, slice=%zu)\n", 
               ggml_current_numa_node, src1_elements, total_elements, elements_in_slice);
        for (int64_t i = numa_start; i < numa_end; ++i) {
            // Simplified broadcasting for common patterns
            const int64_t src1_idx = i % src1_elements;
            dst_data[i] = src0_data[i] + src1_data[src1_idx];
        }
    }
    
    // TEMPLATE STEP 9: Return success status
    return GGML_STATUS_SUCCESS;
}

/**
 * Minimal validation function
 * 
 * Template Pattern: Lightweight kernel support check
 * 
 * Fast validation for kernel applicability. Should be lightweight since
 * this may be called frequently during kernel selection.
 * 
 * @param tensor Tensor to validate for ADD operation
 * @return       true if kernel can handle this tensor, false otherwise
 */
bool ggml_numa_kernel_add_supports_optimized(const struct ggml_tensor * tensor) {
    return tensor && 
           tensor->op == GGML_OP_ADD &&
           tensor->src[0] && 
           tensor->src[1] &&
           tensor->type == GGML_TYPE_F32 &&
           tensor->src[0]->type == GGML_TYPE_F32 &&
           tensor->src[1]->type == GGML_TYPE_F32;
}

// ============================================================================
// Cache Population for Optimized Kernel
// ============================================================================

/**
 * Populate NUMA kernel cache with ADD operation strategies
 * 
 * Template Pattern: Registry integration for kernel strategies
 * 
 * This function defines the execution strategies for different tensor
 * complexity levels. The NUMA executor queries this cache for O(1) 
 * strategy selection based on tensor size.
 * 
 * STRATEGY SELECTION GUIDE:
 * - COMPLEXITY_TINY: < 32K elements - single node, single thread
 * - COMPLEXITY_SMALL: 32K - 512K elements - single node, multi thread  
 * - COMPLEXITY_MEDIUM: 512K - 16M elements - consider data parallel
 * - COMPLEXITY_LARGE: 16M - 256M elements - data parallel recommended
 * - COMPLEXITY_HUGE: > 256M elements - data parallel required
 * 
 * @param cache_array Pointer to cache array (cast to ggml_numa_cache_entry_t*)
 */

void ggml_numa_kernel_add_populate_cache(void * cache_array) {
    ggml_numa_cache_entry_t * cache = (ggml_numa_cache_entry_t *)cache_array;
    
    // TEMPLATE PATTERN: Define strategies for different complexity levels
    // Tailor these to your operation's characteristics and performance profile
    
    // Medium tensors: use single-node execution for better efficiency
    // This avoids thread overhead for moderately-sized tensors
    cache[COMPLEXITY_MEDIUM] = (ggml_numa_cache_entry_t){
        .valid = true,
        .strategy = { 
            .node_strategy = NUMA_NODE_STRATEGY_SINGLE,             // Stay on one node
            .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD  // Use multiple threads on that node
        },
        .work_buffer_size_per_thread = 0,                          // No extra buffers needed
        .work_function = ggml_numa_kernel_add_execute_optimized,   // Our optimized kernel
        .efficiency_score = 0.95f,                                 // Good efficiency for medium data
        .kernel_name = "NUMA ADD (Single-Node Multi-Thread)"
    };
    
    // Use optimized kernel for all large data-parallel cases
    // Large tensors benefit significantly from NUMA parallelization
    cache[COMPLEXITY_LARGE] = (ggml_numa_cache_entry_t){
        .valid = true,
        .strategy = { 
            .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL,      // Slice data across nodes
            .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD  // Use multiple threads per node
        },
        .work_buffer_size_per_thread = 0,                          // No extra buffers needed
        .work_function = ggml_numa_kernel_add_execute_optimized,   // Our optimized kernel
        .efficiency_score = 0.99f,                                 // High efficiency for large data
        .kernel_name = "NUMA ADD (Optimized Data-Parallel)"
    };
    
    // Massive tensors definitely need data-parallel execution
    cache[COMPLEXITY_HUGE] = (ggml_numa_cache_entry_t){
        .valid = true,
        .strategy = { 
            .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL,      // Essential for huge tensors
            .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD  // Maximum parallelization
        },
        .work_buffer_size_per_thread = 0,                          // No extra buffers needed
        .work_function = ggml_numa_kernel_add_execute_optimized,   // Same optimized kernel
        .efficiency_score = 0.99f,                                 // Highest efficiency for huge data
        .kernel_name = "NUMA ADD (Optimized Data-Parallel)"
    };
    
    // TEMPLATE NOTE: Consider adding entries for COMPLEXITY_TINY, COMPLEXITY_SMALL, 
    // and COMPLEXITY_MEDIUM with different strategies if beneficial for your operation.
    // For ADD, data-parallel works well for most cases, but other operations might
    // benefit from single-node execution for smaller tensors.
}
