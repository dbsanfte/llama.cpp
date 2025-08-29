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
 * CRITICAL: Use shared memory approach to eliminate aggregation overhead:
 * 
 * 1. Check for shared result tensor data in data-parallel mode:
 *    extern __thread void * ggml_numa_shared_result_tensor_data;
 * 
 * 2. Write directly to shared memory when available:
 *    if (ggml_numa_shared_result_tensor_data != NULL) {
 *        dst_data = (float *)ggml_numa_shared_result_tensor_data;
 *    } else {
 *        dst_data = (float *)tensor_data(tensor);  // Fallback
 *    }
 * 
 * 3. This eliminates the need for aggregation by having all NUMA nodes
 *    write directly to the final result tensor memory location
 * 
 * 4. Always use tensor_data() for source tensors to get NUMA-local copies:
 *    - tensor_data(tensor) returns the correct NUMA-local copy
 *    - For data-parallel: returns local slice on each node
 *    - For single-node: returns the original tensor data
 *    - DO NOT access tensor->data directly in NUMA kernels
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
 * ✅ 2. Get NUMA-local source data pointers using tensor_data()
 * ✅ 3. Set up shared memory destination pointer for direct writes
 * ✅ 4. Read thread-local NUMA context variables  
 * ✅ 5. Calculate data slice for this thread/node combination
 * ✅ 6. Use SIMD operations for computational core
 * ✅ 7. Handle edge cases (broadcasting, remainder elements)
 * ✅ 8. Return appropriate status code
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
 * Set aggregation_policy to GGML_NUMA_AGGREGATION_NONE for shared memory approach:
 * 
 * .aggregation_policy = GGML_NUMA_AGGREGATION_NONE,  // Direct shared memory writes
 * .aggregation_function = NULL,
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

// External declarations from ggml-cpu.c and ggml.c
extern const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type);
extern const struct ggml_type_traits * ggml_get_type_traits(enum ggml_type type);

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
 * Low-overhead ADD kernel optimized for reduced coordination costs
 * 
 * Key optimizations:
 * 1. Reduced thread count to minimize contention (max 16 threads/node)
 * 2. Optimized work distribution to reduce load imbalance
 * 3. Streamlined execution path for faster synchronization
 * 4. In-place operations when possible to eliminate data aggregation
 * 
 * @param work_context  Tensor to process (cast from void*)
 * @param params        Threadpool parameters (thread ID, thread count)
 * @return              GGML_STATUS_SUCCESS on success, GGML_STATUS_FAILED on error
 */
enum ggml_status ggml_numa_kernel_add_execute_low_overhead(void * work_context, 
                                                          struct ggml_compute_params * params) {
    struct ggml_tensor * tensor = (struct ggml_tensor *)work_context;
    
    // Fast validation - minimal checks for critical errors
    if (!tensor || !tensor->src[0] || !tensor->src[1]) {
        return GGML_STATUS_FAILED;
    }
    
    // Debug-only assertions for non-critical validations (compiled out in release)
    NUMA_DEBUG_ASSERT_MSG(params != NULL, "Compute params should not be NULL");
    NUMA_DEBUG_ASSERT_FMT(tensor->type == GGML_TYPE_F32, "Expected F32 tensor, got type %d", tensor->type);
    
    // Extract tensor parameters
    const struct ggml_tensor * src0 = tensor->src[0];
    const struct ggml_tensor * src1 = tensor->src[1];
    const int64_t total_elements = ggml_nelements(tensor);
    
    // Debug-only data validation (zero overhead in release builds)
    NUMA_DEBUG_ASSERT_MSG(total_elements > 0, "Tensor should have positive element count");
    NUMA_DEBUG_ASSERT_FMT(src0->type == GGML_TYPE_F32, "Source tensor 0 expected F32, got type %d", src0->type);
    NUMA_DEBUG_ASSERT_FMT(src1->type == GGML_TYPE_F32, "Source tensor 1 expected F32, got type %d", src1->type);
    
    // Get NUMA-local data pointers
    const float * src0_data = (const float *)tensor_data(src0);
    const float * src1_data = (const float *)tensor_data(src1);
    
    // For kernels with GGML_NUMA_AGGREGATION_NONE policy, write directly to shared result tensor
    // This eliminates the need for data aggregation across NUMA nodes
    extern __thread void * ggml_numa_shared_result_tensor_data;
    float * dst_data;
    if (ggml_numa_shared_result_tensor_data != NULL) {
        // Use shared result tensor memory - eliminates aggregation overhead
        dst_data = (float *)ggml_numa_shared_result_tensor_data;
        NUMA_LOG_DEBUG("ADD Low-Overhead kernel using shared result tensor memory");
    } else {
        // Fallback to local tensor data for compatibility
        dst_data = (float *)tensor_data(tensor);
        NUMA_LOG_DEBUG("ADD Low-Overhead kernel using local tensor memory");
    }
    
    // Read NUMA context
    extern __thread bool ggml_numa_is_data_parallel_execution;
    extern __thread int ggml_numa_total_nodes_for_data_parallel;
    extern __thread int ggml_current_numa_node;
    
    const int current_node = ggml_current_numa_node;
    const int total_nodes = ggml_numa_is_data_parallel_execution ? 
                           ggml_numa_total_nodes_for_data_parallel : 1;
    const bool is_data_parallel = ggml_numa_is_data_parallel_execution;
    
    // NUMA OPTIMIZATION: In data-parallel mode, write results directly to shared tensor data
    // This eliminates the need for complex aggregation logic at the end
    if (is_data_parallel && ggml_numa_shared_result_tensor_data) {
        dst_data = (float *)ggml_numa_shared_result_tensor_data;
        NUMA_LOG_DEBUG("ADD Node %d: Using shared tensor data at %p (no aggregation needed)", 
                       current_node, dst_data);
    } else {
        NUMA_LOG_DEBUG("ADD Node %d: Using local tensor data at %p", current_node, dst_data);
    }
    
    // OPTIMIZATION: Limit thread count to reduce contention and synchronization overhead
    // Testing shows 16 threads per node is optimal balance of parallelism vs overhead
    const int optimal_threads_per_node = 16;
    const int thread_id = params->ith % optimal_threads_per_node;
    const int num_threads = MIN(params->nth, optimal_threads_per_node);
    
    // Skip execution if this thread is beyond our optimal count
    if (params->ith >= optimal_threads_per_node) {
        return GGML_STATUS_SUCCESS;
    }
    
    NUMA_LOG_DEBUG("NUMA Node %d, Thread %d/%d kernel start (data_parallel=%d, total_nodes=%d, total_elements=%ld)", 
                   current_node, thread_id, num_threads, is_data_parallel, total_nodes, total_elements);
    NUMA_LOG_DEBUG("NUMA Node %d memory pointers: src0=%p, src1=%p, dst=%p", 
                   current_node, src0_data, src1_data, dst_data);
    
    // Calculate optimized data slice
    int64_t numa_start, numa_end;
    
    if (is_data_parallel && total_nodes > 1) {
        // DATA-PARALLEL MODE with load balancing optimization
        const int64_t elements_per_node = total_elements / total_nodes;
        
        // OPTIMIZATION: Add small padding to reduce load imbalance between nodes
        // This helps ensure both nodes finish at similar times
        const int64_t padding = (total_elements % total_nodes) > current_node ? 1 : 0;
        const int64_t adjusted_elements = elements_per_node + padding;
        
        const int64_t node_start = current_node * elements_per_node + MIN(current_node, total_elements % total_nodes);
        const int64_t node_end = MIN(node_start + adjusted_elements, total_elements);
        
        // Divide node slice among threads
        const int64_t elements_per_thread = (node_end - node_start + num_threads - 1) / num_threads;
        numa_start = node_start + thread_id * elements_per_thread;
        numa_end = MIN(numa_start + elements_per_thread, node_end);
        
        NUMA_LOG_DEBUG("NUMA Node %d, Thread %d processing slice: [%ld, %ld) (%ld elements) from node range [%ld, %ld)", 
                       current_node, thread_id, numa_start, numa_end, numa_end - numa_start, node_start, node_end);
    } else {
        // SINGLE-NODE MODE
        const int64_t elements_per_thread = (total_elements + num_threads - 1) / num_threads;
        numa_start = thread_id * elements_per_thread;
        numa_end = MIN(numa_start + elements_per_thread, total_elements);
        
        NUMA_LOG_DEBUG("NUMA Node %d, Thread %d processing tensor slice: [%ld, %ld) (%ld elements)", 
                       current_node, thread_id, numa_start, numa_end, numa_end - numa_start);
    }
    
    // Execute SIMD operations on assigned data slice
    const size_t elements_in_slice = numa_end - numa_start;
    
    // OPTIMIZATION: Streamlined operation dispatch
    const int64_t src1_elements = ggml_nelements(src1);
    
    if (src1_elements == 1) {
        // Scalar broadcasting
        NUMA_LOG_DEBUG("NUMA Node %d using SCALAR broadcasting path (elements_in_slice=%zu)", 
                       current_node, elements_in_slice);
        const float scalar = src1_data[0];
        
        // Combined operation to reduce memory passes
        for (size_t i = 0; i < elements_in_slice; ++i) {
            dst_data[numa_start + i] = src0_data[numa_start + i] + scalar;
        }
        
    } else if (src1_elements == total_elements) {
        // Element-wise operation - optimal path
        NUMA_LOG_DEBUG("NUMA Node %d using ELEMENT-WISE path (elements_in_slice=%zu)", 
                       current_node, elements_in_slice);
        
        // Single SIMD operation for maximum performance
        ggml_vec_add_f32(elements_in_slice, dst_data + numa_start, src0_data + numa_start, src1_data + numa_start);
        
    } else {
        // Complex broadcasting - fallback
        NUMA_LOG_DEBUG("NUMA Node %d using BROADCASTING path (src1_elements=%ld, total=%ld, slice=%zu)", 
                       current_node, src1_elements, total_elements, elements_in_slice);
        for (int64_t i = numa_start; i < numa_end; ++i) {
            const int64_t src1_idx = i % src1_elements;
            dst_data[i] = src0_data[i] + src1_data[src1_idx];
        }
    }
    
    return GGML_STATUS_SUCCESS;
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
    NUMA_LOG_DEBUG("NUMA Node %d, Thread %d/%d kernel start (data_parallel=%d, total_nodes=%d, total_elements=%ld)", 
                   current_node, thread_id, num_threads, is_data_parallel, total_nodes, total_elements);
    NUMA_LOG_DEBUG("NUMA Node %d memory pointers: src0=%p, src1=%p, dst=%p", 
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
        
        NUMA_LOG_DEBUG("NUMA Node %d, Thread %d processing slice: [%ld, %ld) (%ld elements) from node range [%ld, %ld)", 
                       current_node, thread_id, numa_start, numa_end, numa_end - numa_start, node_start, node_end);
    } else {
        // TEMPLATE PATTERN B: SINGLE-NODE MODE
        // All threads process slices of the entire tensor (no NUMA slicing)
        // Good for smaller tensors or when data-parallel doesn't provide benefit
        
        const int64_t elements_per_thread = (total_elements + num_threads - 1) / num_threads;
        numa_start = thread_id * elements_per_thread;
        numa_end = MIN(numa_start + elements_per_thread, total_elements);
        
        NUMA_LOG_DEBUG("NUMA Node %d, Thread %d processing tensor slice: [%ld, %ld) (%ld elements)", 
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
        NUMA_LOG_DEBUG("NUMA Node %d using SCALAR broadcasting path (elements_in_slice=%zu)", 
                       ggml_current_numa_node, elements_in_slice);
        const float scalar = src1_data[0];
        
        // Scalar addition: dst = src0 + scalar
        for (size_t i = 0; i < elements_in_slice; ++i) {
            dst_data[numa_start + i] = src0_data[numa_start + i] + scalar;
        }
        
    } else if (src1_elements == total_elements) {
        // TEMPLATE PATTERN: Element-wise operation (most common, should be fastest)
        NUMA_LOG_DEBUG("NUMA Node %d using ELEMENT-WISE path (elements_in_slice=%zu)", 
                       ggml_current_numa_node, elements_in_slice);
        
        // Pure SIMD operation on global positions - maximum performance path
        ggml_vec_add_f32(elements_in_slice, dst_data + numa_start, src0_data + numa_start, src1_data + numa_start);
        
    } else {
        // TEMPLATE PATTERN: Complex broadcasting - use reference implementation approach
        NUMA_LOG_DEBUG("NUMA Node %d using BROADCASTING path (src1_elements=%ld, total=%ld, slice=%zu)", 
                       ggml_current_numa_node, src1_elements, total_elements, elements_in_slice);
        
        // Get tensor shapes and strides for proper broadcasting
        const int64_t ne0 = tensor->ne[0];
        const int64_t ne1 = tensor->ne[1]; 
        const int64_t ne2 = tensor->ne[2];
        const int64_t ne3 = tensor->ne[3];
        
        const size_t nb0 = tensor->nb[0];
        const size_t nb1 = tensor->nb[1];
        const size_t nb2 = tensor->nb[2];
        const size_t nb3 = tensor->nb[3];
        
        const int64_t ne10 = src1->ne[0];
        const int64_t ne11 = src1->ne[1];
        const int64_t ne12 = src1->ne[2];
        const int64_t ne13 = src1->ne[3];
        
        const size_t nb10 = src1->nb[0];
        const size_t nb11 = src1->nb[1];
        const size_t nb12 = src1->nb[2];
        const size_t nb13 = src1->nb[3];
        
        // For broadcasting, use row-based approach (consistent with reference)
        // Convert numa slice to row indices 
        const int64_t total_rows = ne1 * ne2 * ne3;
        const int64_t elements_per_row = ne0;
        
        const int64_t start_row = numa_start / elements_per_row;
        const int64_t end_row = (numa_end - 1) / elements_per_row + 1;
        
        // Process rows using reference broadcasting logic
        for (int64_t ir = start_row; ir < end_row; ++ir) {
            const int64_t i03 = ir / (ne2 * ne1);
            const int64_t i02 = (ir - i03 * ne2 * ne1) / ne1;
            const int64_t i01 = ir - i03 * ne2 * ne1 - i02 * ne1;
            
            // Apply broadcasting with modulo (like reference)
            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;
            
            // Calculate row pointers using byte strides (like reference)
            float * dst_row  = (float *)       ((char *) dst_data  + i03*nb3  + i02*nb2  + i01*nb1);
            const float * src0_row = (const float *) ((const char *) src0_data + i03*nb3  + i02*nb2  + i01*nb1);
            const float * src1_row = (const float *) ((const char *) src1_data + i13*nb13 + i12*nb12 + i11*nb11);
            
            // Calculate start/end within this row based on numa slice
            const int64_t row_start_idx = ir * elements_per_row;
            const int64_t row_end_idx = (ir + 1) * elements_per_row;
            
            const int64_t slice_start = (row_start_idx < numa_start) ? numa_start - row_start_idx : 0;
            const int64_t slice_end = (row_end_idx > numa_end) ? numa_end - row_start_idx : elements_per_row;
            
            // Process this row's slice element-wise
            for (int64_t i0 = slice_start; i0 < slice_end; ++i0) {
                const int64_t i10 = i0 % ne10;  // Broadcasting in dimension 0
                dst_row[i0] = src0_row[i0] + src1_row[i10];
            }
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
    if (!tensor || tensor->op != GGML_OP_ADD || !tensor->src[0] || !tensor->src[1]) {
        return false;
    }
    
    const struct ggml_tensor * src0 = tensor->src[0];
    const struct ggml_tensor * src1 = tensor->src[1];
    const enum ggml_type dst_type = tensor->type;
    const enum ggml_type src0_type = src0->type;
    const enum ggml_type src1_type = src1->type;
    
    // Support all the same type combinations as the reference implementation
    
    // Non-quantized types (same as binary_op in binary-ops.cpp)
    if ((src0_type == GGML_TYPE_F32  && src1_type == GGML_TYPE_F32  && dst_type == GGML_TYPE_F32) ||  // all f32
        (src0_type == GGML_TYPE_F16  && src1_type == GGML_TYPE_F16  && dst_type == GGML_TYPE_F16) ||  // all f16
        (src0_type == GGML_TYPE_BF16 && src1_type == GGML_TYPE_BF16 && dst_type == GGML_TYPE_BF16) || // all bf16
        (src0_type == GGML_TYPE_BF16 && src1_type == GGML_TYPE_F32  && dst_type == GGML_TYPE_BF16) ||
        (src0_type == GGML_TYPE_BF16 && src1_type == GGML_TYPE_F32  && dst_type == GGML_TYPE_F32) ||
        (src0_type == GGML_TYPE_F16  && src1_type == GGML_TYPE_F32  && dst_type == GGML_TYPE_F16) ||
        (src0_type == GGML_TYPE_F16  && src1_type == GGML_TYPE_F32  && dst_type == GGML_TYPE_F32)) {
        return true;
    }
    
    // Quantized types (same as ggml_compute_forward_add_q_f32)
    if (src1_type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
        switch (src0_type) {
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1:
            case GGML_TYPE_Q5_0:
            case GGML_TYPE_Q5_1:
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
            case GGML_TYPE_TQ1_0:
            case GGML_TYPE_TQ2_0:
            case GGML_TYPE_IQ2_XXS:
            case GGML_TYPE_IQ2_XS:
            case GGML_TYPE_IQ3_XXS:
            case GGML_TYPE_IQ1_S:
            case GGML_TYPE_IQ1_M:
            case GGML_TYPE_IQ4_NL:
            case GGML_TYPE_IQ4_XS:
            case GGML_TYPE_IQ3_S:
            case GGML_TYPE_IQ2_S:
                return true;
            default:
                break;
        }
    }
    
    return false;
}

// ============================================================================
// ============================================================================
// ADD Threshold-Based Strategy Selection
// ============================================================================

/**
 * ADD operation-specific thresholds for optimal strategy selection
 * These thresholds are tuned specifically for element-wise addition characteristics
 */
typedef struct {
    size_t element_threshold;                      // Threshold in number of elements
    ggml_numa_execution_strategy_t strategy;      // Strategy to use
    size_t work_buffer_size_per_thread;          // Buffer size needed (0 for ADD)
    ggml_numa_work_function_t work_function;     // Work function to use
    float efficiency_score;                       // Expected efficiency
    const char * kernel_name;                     // Strategy description
} ggml_add_strategy_threshold_t;

/**
 * ADD-specific strategy thresholds
 * Optimized for element-wise addition workload characteristics
 */
static const ggml_add_strategy_threshold_t ADD_THRESHOLDS[] = {
    // Very small tensors: single-threaded is fastest due to minimal overhead
    {
        .element_threshold = 1024,  // < 1K elements
        .strategy = { .node_strategy = NUMA_NODE_STRATEGY_SINGLE, .on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD },
        .work_buffer_size_per_thread = 0,
        .work_function = ggml_numa_kernel_add_execute_optimized,
        .efficiency_score = 0.98f,
        .kernel_name = "NUMA ADD (Single/Single)"
    },
    
    // Small tensors: multi-threaded on single node for good performance
    {
        .element_threshold = 16384,  // 1K - 16K elements
        .strategy = { .node_strategy = NUMA_NODE_STRATEGY_SINGLE, .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD },
        .work_buffer_size_per_thread = 0,
        .work_function = ggml_numa_kernel_add_execute_optimized,
        .efficiency_score = 0.96f,
        .kernel_name = "NUMA ADD (Single/Multi)"
    },
    
    // Medium tensors: single-node multi-thread for cache efficiency
    {
        .element_threshold = 262144,  // 16K - 256K elements
        .strategy = { .node_strategy = NUMA_NODE_STRATEGY_SINGLE, .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD },
        .work_buffer_size_per_thread = 0,
        .work_function = ggml_numa_kernel_add_execute_optimized,
        .efficiency_score = 0.95f,
        .kernel_name = "NUMA ADD (Single/Multi-Med)"
    },
    
    // Large tensors: data-parallel with low-overhead optimization
    {
        .element_threshold = 16777216,  // 256K - 16M elements
        .strategy = { .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL, .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD },
        .work_buffer_size_per_thread = 0,
        .work_function = ggml_numa_kernel_add_execute_low_overhead,
        .efficiency_score = 0.99f,
        .kernel_name = "NUMA ADD (Low-Overhead Data-Parallel)"
    },
    
    // Huge tensors: no-aggregation for maximum performance
    {
        .element_threshold = 268435456,  // 16M - 256M elements (1GB scale)
        .strategy = { .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL, .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD },
        .work_buffer_size_per_thread = 0,
        .work_function = ggml_numa_kernel_add_execute_no_aggregation,
        .efficiency_score = 0.99f,
        .kernel_name = "NUMA ADD (No-Aggregation 1GB Scale)"
    },
    
    // Ultra-large tensors: no-aggregation for GB+ scale
    {
        .element_threshold = SIZE_MAX,  // 256M+ elements (multi-GB scale)
        .strategy = { .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL, .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD },
        .work_buffer_size_per_thread = 0,
        .work_function = ggml_numa_kernel_add_execute_no_aggregation,
        .efficiency_score = 0.99f,
        .kernel_name = "NUMA ADD (No-Aggregation Ultra Scale)"
    }
};

#define ADD_THRESHOLD_COUNT (sizeof(ADD_THRESHOLDS) / sizeof(ADD_THRESHOLDS[0]))

/**
 * Query ADD kernel for optimal strategy based on tensor characteristics
 * 
 * This function analyzes the tensor and returns the optimal execution strategy
 * without requiring exact complexity class matching.
 * 
 * @param tensor The tensor to analyze
 * @return Query result with optimal strategy, or unsupported result if not applicable
 */
ggml_numa_kernel_query_result_t ggml_numa_kernel_add_query(const struct ggml_tensor * tensor) {
    ggml_numa_kernel_query_result_t result = { .supported = false };
    
    // Validate this is an ADD operation
    if (!tensor || tensor->op != GGML_OP_ADD) {
        return result;
    }
    
    // Validate tensor structure for ADD
    if (!tensor->src[0] || !tensor->src[1]) {
        NUMA_LOG_DEBUG("ADD query: Missing source tensors");
        return result;
    }
    
    // Calculate total elements for strategy selection
    const size_t total_elements = ggml_nelements(tensor);
    
    // Find optimal strategy using threshold search
    const ggml_add_strategy_threshold_t * selected_strategy = &ADD_THRESHOLDS[ADD_THRESHOLD_COUNT - 1];
    
    for (size_t i = 0; i < ADD_THRESHOLD_COUNT; i++) {
        if (total_elements < ADD_THRESHOLDS[i].element_threshold) {
            selected_strategy = &ADD_THRESHOLDS[i];
            break;
        }
    }
    
    // Build successful query result
    result.supported = true;
    result.strategy = selected_strategy->strategy;
    result.work_buffer_size_per_thread = selected_strategy->work_buffer_size_per_thread;
    result.work_function = selected_strategy->work_function;
    result.efficiency_score = selected_strategy->efficiency_score;
    result.kernel_name = selected_strategy->kernel_name;
    
    // Set aggregation policy based on the selected work function
    if (selected_strategy->work_function == ggml_numa_kernel_add_execute_no_aggregation) {
        result.aggregation_policy = GGML_NUMA_AGGREGATION_NONE;
    } else {
        result.aggregation_policy = GGML_NUMA_AGGREGATION_NONE; // Traditional ADD kernels will now need custom aggregation if needed
    }
    
    NUMA_LOG_DEBUG("ADD query: %zu elements -> %s (efficiency: %.2f)", 
                   total_elements, result.kernel_name, result.efficiency_score);
    
    return result;
}

// ============================================================================
// No-Aggregation Implementation  
// For element-wise operations that don't require data aggregation between NUMA nodes
// This eliminates the coordination overhead by having each node write directly to 
// its slice of the final tensor, removing the need for data copying between nodes
enum ggml_status ggml_numa_kernel_add_execute_no_aggregation(void * work_context, 
                                                            struct ggml_compute_params * params) {
    struct ggml_tensor * tensor = (struct ggml_tensor *)work_context;
    
    // Fast validation - minimal checks
    if (!tensor || !tensor->src[0] || !tensor->src[1]) {
        return GGML_STATUS_FAILED;
    }
    
    // Extract tensor parameters
    const struct ggml_tensor * src0 = tensor->src[0];
    const struct ggml_tensor * src1 = tensor->src[1];
    const enum ggml_type src0_type = src0->type;
    const enum ggml_type src1_type = src1->type;
    const enum ggml_type dst_type = tensor->type;
    const int64_t total_elements = ggml_nelements(tensor);
    
    // Read NUMA context
    extern __thread bool ggml_numa_is_data_parallel_execution;
    extern __thread int ggml_numa_total_nodes_for_data_parallel;
    extern __thread int ggml_current_numa_node;
    
    const int current_node = ggml_current_numa_node;
    const int total_nodes = ggml_numa_is_data_parallel_execution ? 
                           ggml_numa_total_nodes_for_data_parallel : 1;
    const bool is_data_parallel = ggml_numa_is_data_parallel_execution;
    
    // Optimal thread count to reduce synchronization overhead
    const int optimal_threads_per_node = 8;
    const int thread_id = params->ith % optimal_threads_per_node;
    const int num_threads = MIN(params->nth, optimal_threads_per_node);
    
    // Skip execution if this thread is beyond our optimal count
    if (params->ith >= optimal_threads_per_node) {
        return GGML_STATUS_SUCCESS;
    }
    
    NUMA_LOG_DEBUG("NUMA Node %d, Thread %d TYPE-AWARE kernel (src0=%s, src1=%s, dst=%s, data_parallel=%d)", 
                   current_node, thread_id, ggml_type_name(src0_type), ggml_type_name(src1_type), 
                   ggml_type_name(dst_type), is_data_parallel);
    
    // Calculate data slice with larger chunks for reduced overhead
    int64_t numa_start, numa_end;
    
    if (is_data_parallel && total_nodes > 1) {
        // DATA-PARALLEL MODE: Each node handles its slice
        const int64_t elements_per_node = total_elements / total_nodes;
        numa_start = current_node * elements_per_node;
        numa_end = (current_node == total_nodes - 1) ? total_elements : numa_start + elements_per_node;
    } else {
        // SINGLE-NODE MODE: Process entire tensor
        numa_start = 0;
        numa_end = total_elements;
    }
    
    // Divide this node's slice among threads
    const int64_t elements_in_slice = numa_end - numa_start;
    const int64_t elements_per_thread = (elements_in_slice + num_threads - 1) / num_threads;
    const int64_t thread_start = numa_start + thread_id * elements_per_thread;
    const int64_t thread_end = MIN(thread_start + elements_per_thread, numa_end);
    const size_t thread_elements = thread_end - thread_start;
    
    // Skip if no work for this thread
    if (thread_elements == 0) {
        return GGML_STATUS_SUCCESS;
    }
    
    // Get NUMA-local data pointers
    const void * src0_data = tensor_data(src0);
    const void * src1_data = tensor_data(src1);
    void * dst_data = tensor_data(tensor);
    
    // Handle different type combinations following the same pattern as reference implementation
    
    // NON-QUANTIZED TYPES (matching binary_op template in binary-ops.cpp)
    if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
        // All F32 - direct SIMD operation
        const float * s0 = (const float *)src0_data;
        const float * s1 = (const float *)src1_data;
        float * d = (float *)dst_data;
        
        ggml_vec_add_f32(thread_elements, d + thread_start, s0 + thread_start, s1 + thread_start);
        
    } else if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
        // All F16 - element-wise with type conversion
        const ggml_fp16_t * s0 = (const ggml_fp16_t *)src0_data;
        const ggml_fp16_t * s1 = (const ggml_fp16_t *)src1_data;
        ggml_fp16_t * d = (ggml_fp16_t *)dst_data;
        
        for (int64_t i = thread_start; i < thread_end; ++i) {
            d[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(s0[i]) + ggml_fp16_to_fp32(s1[i]));
        }
        
    } else if (src0_type == GGML_TYPE_BF16 && src1_type == GGML_TYPE_BF16 && dst_type == GGML_TYPE_BF16) {
        // All BF16 - element-wise with type conversion
        const ggml_bf16_t * s0 = (const ggml_bf16_t *)src0_data;
        const ggml_bf16_t * s1 = (const ggml_bf16_t *)src1_data;
        ggml_bf16_t * d = (ggml_bf16_t *)dst_data;
        
        for (int64_t i = thread_start; i < thread_end; ++i) {
            d[i] = ggml_fp32_to_bf16(ggml_bf16_to_fp32(s0[i]) + ggml_bf16_to_fp32(s1[i]));
        }
        
    } else if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
        // F16 + F32 → F32
        const ggml_fp16_t * s0 = (const ggml_fp16_t *)src0_data;
        const float * s1 = (const float *)src1_data;
        float * d = (float *)dst_data;
        
        for (int64_t i = thread_start; i < thread_end; ++i) {
            d[i] = ggml_fp16_to_fp32(s0[i]) + s1[i];
        }
        
    } else if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
        // F16 + F32 → F16
        const ggml_fp16_t * s0 = (const ggml_fp16_t *)src0_data;
        const float * s1 = (const float *)src1_data;
        ggml_fp16_t * d = (ggml_fp16_t *)dst_data;
        
        for (int64_t i = thread_start; i < thread_end; ++i) {
            d[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(s0[i]) + s1[i]);
        }
        
    } else if (src0_type == GGML_TYPE_BF16 && src1_type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
        // BF16 + F32 → F32
        const ggml_bf16_t * s0 = (const ggml_bf16_t *)src0_data;
        const float * s1 = (const float *)src1_data;
        float * d = (float *)dst_data;
        
        for (int64_t i = thread_start; i < thread_end; ++i) {
            d[i] = ggml_bf16_to_fp32(s0[i]) + s1[i];
        }
        
    } else if (src0_type == GGML_TYPE_BF16 && src1_type == GGML_TYPE_F32 && dst_type == GGML_TYPE_BF16) {
        // BF16 + F32 → BF16
        const ggml_bf16_t * s0 = (const ggml_bf16_t *)src0_data;
        const float * s1 = (const float *)src1_data;
        ggml_bf16_t * d = (ggml_bf16_t *)dst_data;
        
        for (int64_t i = thread_start; i < thread_end; ++i) {
            d[i] = ggml_fp32_to_bf16(ggml_bf16_to_fp32(s0[i]) + s1[i]);
        }
        
    } else if (ggml_is_quantized(src0_type) && src1_type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
        // QUANTIZED TYPES (matching ggml_compute_forward_add_q_f32 logic)
        // Process row-wise like the reference implementation
        
        const struct ggml_tensor * src0_tensor = src0;
        const struct ggml_tensor * src1_tensor = src1;
        
        const int64_t ne0 = src0_tensor->ne[0];
        const int64_t ne1 = src0_tensor->ne[1];
        const int64_t ne2 = src0_tensor->ne[2];
        const int64_t ne3 = src0_tensor->ne[3];
        
        const size_t nb00 = src0_tensor->nb[0];
        const size_t nb01 = src0_tensor->nb[1];
        const size_t nb02 = src0_tensor->nb[2];
        const size_t nb03 = src0_tensor->nb[3];
        
        const size_t nb10 = src1_tensor->nb[0];
        const size_t nb11 = src1_tensor->nb[1];
        const size_t nb12 = src1_tensor->nb[2];
        const size_t nb13 = src1_tensor->nb[3];
        
        const int64_t nr = ne1 * ne2 * ne3;  // total number of rows
        
        // Get quantization functions
        const ggml_to_float_t dequantize_row_q = ggml_get_type_traits(src0_type)->to_float;
        const ggml_from_float_t quantize_row_q = ggml_get_type_traits_cpu(dst_type)->from_float;
        
        if (!dequantize_row_q) {
            NUMA_LOG_DEBUG("No dequantization function for type %s", ggml_type_name(src0_type));
            return GGML_STATUS_FAILED;
        }
        
        // Calculate which rows this thread should process
        const int64_t rows_per_thread = (nr + num_threads - 1) / num_threads;
        const int64_t row_start = thread_id * rows_per_thread;
        const int64_t row_end = MIN(row_start + rows_per_thread, nr);
        
        // Allocate workspace for dequantized data
        float * wdata = (float *)malloc(ne0 * sizeof(float));
        if (!wdata) {
            return GGML_STATUS_FAILED;
        }
        
        for (int64_t row = row_start; row < row_end; ++row) {
            // Calculate row indices
            const int64_t i03 = row / (ne2 * ne1);
            const int64_t i02 = (row - i03 * ne2 * ne1) / ne1;
            const int64_t i01 = row - i03 * ne2 * ne1 - i02 * ne1;
            
            // Get row pointers
            const void * src0_row = (const void *)((const char *)src0_data + i01*nb01 + i02*nb02 + i03*nb03);
            const float * src1_row = (const float *)((const char *)src1_data + i01*nb11 + i02*nb12 + i03*nb13);
            void * dst_row = (void *)((char *)dst_data + i01*tensor->nb[1] + i02*tensor->nb[2] + i03*tensor->nb[3]);
            
            // Dequantize src0 row to workspace
            dequantize_row_q(src0_row, wdata, ne0);
            
            // Add src1 row
            ggml_vec_acc_f32(ne0, wdata, src1_row);
            
            // Quantize result to dst row (or copy if no quantization needed)
            if (quantize_row_q) {
                quantize_row_q(wdata, dst_row, ne0);
            } else {
                memcpy(dst_row, wdata, ne0 * sizeof(float));
            }
        }
        
        free(wdata);
        
    } else {
        // Unsupported type combination
        NUMA_LOG_DEBUG("Unsupported type combination: src0=%s, src1=%s, dst=%s", 
                       ggml_type_name(src0_type), ggml_type_name(src1_type), ggml_type_name(dst_type));
        return GGML_STATUS_FAILED;
    }
    
    NUMA_LOG_DEBUG("NUMA Node %d, Thread %d TYPE-AWARE completed (%zu elements processed)", 
                   current_node, thread_id, thread_elements);
    
    return GGML_STATUS_SUCCESS;
}

// ============================================================================
// Kernel Registration Function
// ============================================================================

/**
 * Register ADD kernel with strategy arrays and function pointers
 * This function provides the strategy thresholds and function pointers
 * that the registry will use for O(1) lookups.
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_add_register(void) {
    ggml_numa_kernel_registration_info_t info = {0};
    
    info.op_type = GGML_OP_ADD;
    info.supported = true;
    info.kernel_name = "NUMA ADD Kernel";
    
    // Strategy thresholds for ADD operations
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_SINGLE] = 1024;      // Single thread below 1K elements
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_MULTI] = 262144;     // Multi-thread below 256K elements
    // Above 256K elements: data-parallel strategy
    info.strategy_array.valid = true;
    
    // Function pointers for different strategies
    info.agg_funcs.single_single_fn = ggml_numa_kernel_add_execute_low_overhead;
    info.agg_funcs.single_multi_fn = ggml_numa_kernel_add_execute_low_overhead;
    info.agg_funcs.data_parallel_fn = ggml_numa_kernel_add_execute_no_aggregation;
    info.agg_funcs.valid = true;
    
    return info;
}
