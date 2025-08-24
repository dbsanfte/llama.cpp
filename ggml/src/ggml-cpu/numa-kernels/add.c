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
 */
enum ggml_status ggml_numa_kernel_add_execute_optimized(void * work_context, 
                                                       struct ggml_compute_params * params) {
    struct ggml_tensor * tensor = (struct ggml_tensor *)work_context;
    
    // FAST PATH: Minimal validation - assume coordinator has validated inputs
    if (!tensor || !tensor->src[0] || !tensor->src[1]) {
        return GGML_STATUS_FAILED;
    }
    
    // Cache tensor pointers and avoid repeated dereferencing
    const struct ggml_tensor * src0 = tensor->src[0];
    const struct ggml_tensor * src1 = tensor->src[1];
    const int64_t total_elements = ggml_nelements(tensor);
    
    // Get data pointers directly - tensor_data() returns the correct NUMA-local copy
    const float * src0_data = (const float *)tensor_data(src0);
    const float * src1_data = (const float *)tensor_data(src1);
    float * dst_data = (float *)tensor_data(tensor);
    
    // Check if we're in data-parallel mode 
    extern __thread bool ggml_numa_is_data_parallel_execution;
    extern __thread int ggml_numa_total_nodes_for_data_parallel;
    extern __thread int ggml_current_numa_node;
    
    const int current_node = ggml_current_numa_node;
    const int total_nodes = ggml_numa_is_data_parallel_execution ? 
                           ggml_numa_total_nodes_for_data_parallel : 1;
    const bool is_data_parallel = ggml_numa_is_data_parallel_execution;
    
    // Get threadpool parameters for proper multi-threading
    const int thread_id = params->ith;
    const int num_threads = params->nth;
    
    printf("DEBUG: NUMA Node %d, Thread %d/%d kernel start (data_parallel=%d, total_nodes=%d, total_elements=%ld)\n", 
           current_node, thread_id, num_threads, is_data_parallel, total_nodes, total_elements);
    printf("DEBUG: NUMA Node %d memory pointers: src0=%p, src1=%p, dst=%p\n", 
           current_node, src0_data, src1_data, dst_data);
    
    int64_t numa_start, numa_end;
    
    if (is_data_parallel && total_nodes > 1) {
        // DATA-PARALLEL MODE: Each node processes its assigned slice
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
        // SINGLE-NODE MODE: All threads process slices of the entire tensor
        const int64_t elements_per_thread = (total_elements + num_threads - 1) / num_threads;
        numa_start = thread_id * elements_per_thread;
        numa_end = MIN(numa_start + elements_per_thread, total_elements);
        
        printf("DEBUG: NUMA Node %d, Thread %d processing tensor slice: [%ld, %ld) (%ld elements)\n", 
               current_node, thread_id, numa_start, numa_end, numa_end - numa_start);
    }
    
    const size_t elements_in_slice = numa_end - numa_start;
    
    // Check for broadcasting - optimized check
    const int64_t src1_elements = ggml_nelements(src1);
    
    if (src1_elements == 1) {
        // FASTEST PATH: Scalar broadcasting
        printf("DEBUG: NUMA Node %d using SCALAR broadcasting path (elements_in_slice=%zu)\n", 
               ggml_current_numa_node, elements_in_slice);
        const float scalar = src1_data[0];
        
        // Use SIMD for scalar addition - direct access to global positions
        ggml_vec_scale_f32(elements_in_slice, dst_data + numa_start, scalar);
        ggml_vec_add_f32(elements_in_slice, dst_data + numa_start, dst_data + numa_start, src0_data + numa_start);
        
    } else if (src1_elements == total_elements) {
        // FAST PATH: Element-wise addition (no broadcasting)
        printf("DEBUG: NUMA Node %d using ELEMENT-WISE path (elements_in_slice=%zu)\n", 
               ggml_current_numa_node, elements_in_slice);
        
        // Pure SIMD operation on global positions
        ggml_vec_add_f32(elements_in_slice, dst_data + numa_start, src0_data + numa_start, src1_data + numa_start);
        
    } else {
        // SLOW PATH: Complex broadcasting - avoid if possible
        printf("DEBUG: NUMA Node %d using SLOW BROADCASTING path (src1_elements=%ld, total=%ld, slice=%zu)\n", 
               ggml_current_numa_node, src1_elements, total_elements, elements_in_slice);
        for (int64_t i = numa_start; i < numa_end; ++i) {
            // Simplified broadcasting for common patterns
            const int64_t src1_idx = i % src1_elements;
            dst_data[i] = src0_data[i] + src1_data[src1_idx];
        }
    }
    
    return GGML_STATUS_SUCCESS;
}

/**
 * Minimal validation function
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

void ggml_numa_kernel_add_populate_cache(void * cache_array) {
    ggml_numa_cache_entry_t * cache = (ggml_numa_cache_entry_t *)cache_array;
    
    // Use optimized kernel for all large data-parallel cases
    cache[COMPLEXITY_LARGE] = (ggml_numa_cache_entry_t){
        .valid = true,
        .strategy = { 
            .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL,
            .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD 
        },
        .work_buffer_size_per_thread = 0,
        .work_function = ggml_numa_kernel_add_execute_optimized,
        .efficiency_score = 0.99f,
        .kernel_name = "NUMA ADD (Optimized Data-Parallel)"
    };
    
    cache[COMPLEXITY_HUGE] = (ggml_numa_cache_entry_t){
        .valid = true,
        .strategy = { 
            .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL,
            .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD 
        },
        .work_buffer_size_per_thread = 0,
        .work_function = ggml_numa_kernel_add_execute_optimized,
        .efficiency_score = 0.99f,
        .kernel_name = "NUMA ADD (Optimized Data-Parallel)"
    };
}
