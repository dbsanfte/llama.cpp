/*
 * NUMA Kernel: Element-wise Addition (ADD)
 * 
 * Uses pre-allocated NUMA-local mirrored data directly without migration.
 */

#include "../binary-ops.h"
#include "add.h"
#include <immintrin.h>  // For AVX2 SIMD instructions
#include "numa-kernels.h"  // For cache types
#include "../ggml-numa-shared.h"          // Shared NUMA logging and utilities
#include "../ggml-numa-simple-coordinator.h"  // For ggml_numa_get_current_node()
#include "../ggml-cpu-impl.h"
#include "../ggml-impl.h"
#include "../vec.h"  // For SIMD functions
#include "../ggml-numa-perf.h"  // Performance instrumentation
#include "../../ggml-numa-allocator.h"  // For NUMA memory assertion functions

#ifdef GGML_NUMA_MIRROR
#include <numa.h>  // For numa_num_configured_nodes()
#endif

// ============================================================================
// Static Helper Functions (Implementation Details)
// ============================================================================

/**
 * Tensor validation and type checking
 */
static bool validate_tensor_inputs(const struct ggml_tensor * tensor) {
    if (!tensor || tensor->op != GGML_OP_ADD) {
        return false;
    }
    
    const struct ggml_tensor * src0 = tensor->src[0];
    const struct ggml_tensor * src1 = tensor->src[1];
    
    if (!src0 || !src1) {
        return false;
    }
    
    // Only support f32 for now (can be extended)
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || tensor->type != GGML_TYPE_F32) {
        return false;
    }
    
    // Check compatible shapes for addition
    if (!ggml_can_repeat(src1, src0) || !ggml_are_same_shape(src0, tensor)) {
        return false;
    }
    
    return true;
}

/**
 * Calculate NUMA slice bounds using coordinator-assigned virtual node
 * For data-parallel execution, each node processes a different slice of the data
 * For single-node execution, the node processes ALL data
 */
static void calculate_numa_slice(const struct ggml_tensor * tensor, 
                                int64_t start_offset, 
                                int64_t total_elements,
                                int64_t * numa_start, 
                                int64_t * numa_end,
                                bool is_data_parallel_execution) {
    extern int ggml_numa_node_count(void);
    extern __thread int ggml_current_numa_node;
    const int total_numa_nodes = ggml_numa_node_count();
    const int current_numa_node = ggml_current_numa_node; // Use coordinator-assigned virtual node!
    
    if (total_numa_nodes <= 1 || current_numa_node < 0 || !is_data_parallel_execution) {
        // Single node, invalid node, or single-node execution mode - process ALL data
        *numa_start = start_offset;
        *numa_end = total_elements;
        printf("DEBUG: NUMA ADD SLICE: Node %d processing ALL elements [%lld,%lld) of %lld total (single-node mode)\n",
               current_numa_node, (long long)*numa_start, (long long)*numa_end, (long long)total_elements);
        return;
    }
    
    // For data-parallel execution, each NUMA node processes a different slice
    const int64_t elements_per_node = total_elements / total_numa_nodes;
    *numa_start = start_offset + current_numa_node * elements_per_node;
    
    // Last node gets any remainder elements
    if (current_numa_node == total_numa_nodes - 1) {
        *numa_end = total_elements;
    } else {
        *numa_end = *numa_start + elements_per_node;
    }
    
    printf("DEBUG: NUMA ADD SLICE: Node %d processing elements [%lld,%lld) of %lld total (data-parallel mode)\n",
           current_numa_node, (long long)*numa_start, (long long)*numa_end, (long long)total_elements);
}

// ============================================================================
// Kernel Interface Implementation
// ============================================================================

bool ggml_numa_kernel_add_supports(const struct ggml_tensor * tensor) {
    return validate_tensor_inputs(tensor);
}

// New architecture functions for kernel registry queries

ggml_numa_execution_strategy_t ggml_numa_kernel_add_get_strategy(const struct ggml_tensor * tensor) {
    ggml_numa_execution_strategy_t strategy = {
        .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL,
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD
    };
    
    // For very small tensors, use single node to avoid overhead
    const int64_t elements = ggml_nelements(tensor);
    if (elements < 8192) {  // < 32KB
        strategy.node_strategy = NUMA_NODE_STRATEGY_SINGLE;
    }
    
    return strategy;
}

size_t ggml_numa_kernel_add_get_work_buffer_size(const struct ggml_tensor * tensor) {
    // Element-wise ADD doesn't need work buffers - it operates directly on tensor data
    return 0;
}

float ggml_numa_kernel_add_get_efficiency_score(const struct ggml_tensor * tensor) {
    // Very high efficiency for element-wise operations
    return 0.98f;  
}

const char * ggml_numa_kernel_add_get_name(const struct ggml_tensor * tensor) {
    return "NUMA ADD (Direct Mirrored Data)";
}

// ============================================================================
// Main Execution Function
// ============================================================================

enum ggml_status ggml_numa_kernel_add_execute(void * work_context, 
                                              struct ggml_compute_params * params) {
    // Extract and validate inputs
    struct ggml_tensor * tensor = (struct ggml_tensor *)work_context;
    
    NUMA_ASSERT(tensor != NULL, "Tensor cannot be null");
    NUMA_ASSERT(params != NULL, "Compute params cannot be null");
    
    if (!validate_tensor_inputs(tensor)) {
        return GGML_STATUS_FAILED;
    }
    
    const struct ggml_tensor * src0 = tensor->src[0];
    const struct ggml_tensor * src1 = tensor->src[1];
    
    // Validate tensor data pointers
    if (!tensor_data(tensor) || !tensor_data(src0) || !tensor_data(src1)) {
        return GGML_STATUS_FAILED;
    }
    
    // Calculate work range for this NUMA node using coordinator-assigned virtual node
    const int64_t total_elements = ggml_nelements(tensor);
    int64_t numa_start, numa_end;
    
    // Check if we're in data-parallel execution mode
    extern __thread bool ggml_numa_is_data_parallel_execution;
    extern int ggml_numa_node_count(void);
    extern __thread int ggml_current_numa_node;
    const int current_numa_node = ggml_current_numa_node;
    
    // Use the proper slicing function
    calculate_numa_slice(tensor, 0, total_elements, &numa_start, &numa_end, ggml_numa_is_data_parallel_execution);
    
    size_t elements_in_slice = numa_end - numa_start;
    
    // Record start time for concurrency analysis
    struct timespec kernel_start_time, compute_start_time, compute_end_time;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start_time);
    
    printf("⏰ NUMA ADD KERNEL (START): Node %d starting at %ld.%09ld\n", 
           current_numa_node, kernel_start_time.tv_sec, kernel_start_time.tv_nsec);
    
    printf("🚀 NUMA ADD KERNEL (DIRECT): Processing %zu elements [%ld,%ld) on node %d\n", 
           elements_in_slice, numa_start, numa_end, current_numa_node);
    
    // DEBUG: Check what ggml_current_numa_node is set to during execution
    extern __thread int ggml_current_numa_node;
    printf("🔍 NUMA ADD KERNEL: ggml_current_numa_node=%d, current_numa_node=%d\n", 
           ggml_current_numa_node, current_numa_node);
    
    // Get NUMA-local data pointers directly (no migration needed!)
    // Extract tensor data using NUMA-aware tensor_data() function
    const float * src0_data = (const float *)tensor_data(src0);
    const float * src1_data = (const float *)tensor_data(src1);
    float * dst_data = (float *)tensor_data(tensor);
    
    printf("DEBUG: ADD kernel - node %d: src0=%p src1=%p dst=%p (elements=%zu, slice=[%zu,%zu))\n",           current_numa_node, (void*)src0_data, (void*)src1_data, (void*)dst_data,           numa_end - numa_start, numa_start, numa_end);
    
    // Show first few values of input data for debugging
    printf("DEBUG: ADD kernel - node %d: src0_slice=[%.3f, %.3f, %.3f, %.3f], src1_slice=[%.3f, %.3f, %.3f, %.3f]\n",           current_numa_node,            src0_data[numa_start], src0_data[numa_start+1], src0_data[numa_start+2], src0_data[numa_start+3],            src1_data[numa_start], src1_data[numa_start+1], src1_data[numa_start+2], src1_data[numa_start+3]);    printf("🔗 NUMA ADD KERNEL: Using mirrored data pointers: src0=%p, src1=%p, dst=%p\n", 
           (void*)src0_data, (void*)src1_data, (void*)dst_data);
    
    // ⚠️  CRITICAL NUMA DATA LOCALITY ASSERTIONS ⚠️
    // All data MUST be on the same NUMA node as assigned by coordinator
    // Cross-NUMA access is a hard failure that destroys performance
    printf("🔍 NUMA ADD KERNEL: Validating data locality on coordinator-assigned node %d\n", current_numa_node);
    
    // Assert src0 data is on the correct NUMA node
    ggml_numa_assert_allocation((void*)src0_data, current_numa_node, "ADD kernel src0 data");
    
    // Assert src1 data is on the correct NUMA node  
    ggml_numa_assert_allocation((void*)src1_data, current_numa_node, "ADD kernel src1 data");
    
    // Assert dst data is on the correct NUMA node
    ggml_numa_assert_allocation((void*)dst_data, current_numa_node, "ADD kernel dst data");
    
    // Note: Work buffers would be asserted here if this kernel used them
    // For ADD operation, no additional compute buffers are needed
    
    printf("✅ NUMA ADD KERNEL: All data validated on node %d - proceeding with computation\n", current_numa_node);
    
    // Broadcasting support: check if src1 has fewer elements
    const int64_t src1_elements = ggml_nelements(src1);
    const bool needs_broadcasting = (src1_elements < total_elements);
    
    if (needs_broadcasting) {
        printf("📡 NUMA ADD KERNEL: Broadcasting src1 (%ld elements) to dst (%ld elements)\n", 
               src1_elements, total_elements);
    }
    
    // Process this NUMA node's slice with SIMD optimization
    clock_gettime(CLOCK_MONOTONIC, &compute_start_time);
    
    printf("⚡ NUMA ADD KERNEL (COMPUTE START): Node %d beginning computation at %ld.%09ld\n", 
           current_numa_node, compute_start_time.tv_sec, compute_start_time.tv_nsec);
    
    if (needs_broadcasting) {
        // Get tensor dimensions for proper multidimensional broadcasting
        const int64_t ne00 = src0->ne[0];  // src0 dim 0 
        const int64_t ne01 = src0->ne[1];  // src0 dim 1
        const int64_t ne02 = src0->ne[2];  // src0 dim 2
        const int64_t ne03 = src0->ne[3];  // src0 dim 3
        
        const int64_t ne10 = src1->ne[0];  // src1 dim 0
        const int64_t ne11 = src1->ne[1];  // src1 dim 1  
        const int64_t ne12 = src1->ne[2];  // src1 dim 2
        const int64_t ne13 = src1->ne[3];  // src1 dim 3
        
        printf("🔍 NUMA ADD KERNEL: Broadcasting dimensions - src0[%ld,%ld,%ld,%ld], src1[%ld,%ld,%ld,%ld]\n",
               ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13);
        
        // Element-wise with proper multidimensional broadcasting
        for (int64_t i = numa_start; i < numa_end; ++i) {
            // Convert linear index i to 4D coordinates (i0, i1, i2, i3) for src0/dst
            const int64_t i03 = i / (ne02 * ne01 * ne00);
            const int64_t i02 = (i % (ne02 * ne01 * ne00)) / (ne01 * ne00);
            const int64_t i01 = (i % (ne01 * ne00)) / ne00;
            const int64_t i00 = i % ne00;
            
            // Apply broadcasting: src1 coordinates use modulo for each dimension
            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;
            const int64_t i10 = i00 % ne10;
            
            // Convert 4D coordinates back to linear indices
            const int64_t src1_idx = i13 * (ne12 * ne11 * ne10) + 
                                   i12 * (ne11 * ne10) + 
                                   i11 * ne10 + 
                                   i10;
            
            dst_data[i] = src0_data[i] + src1_data[src1_idx];
        }
    } else {
        // Direct SIMD operation (most efficient path)
        const float * src0_slice = src0_data + numa_start;
        const float * src1_slice = src1_data + numa_start;
        float * dst_slice = dst_data + numa_start;
        
        ggml_vec_add_f32(elements_in_slice, dst_slice, src0_slice, src1_slice);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &compute_end_time);
    printf("⚡ NUMA ADD KERNEL (COMPUTE END): Node %d finished computation at %ld.%09ld\n", 
           current_numa_node, compute_end_time.tv_sec, compute_end_time.tv_nsec);
    
    double compute_time_ms = (compute_end_time.tv_sec - compute_start_time.tv_sec) * 1000.0 + 
                            (compute_end_time.tv_nsec - compute_start_time.tv_nsec) / 1000000.0;
    
    printf("✅ NUMA ADD KERNEL: Completed %zu elements in %.3f ms (%.2f GB/s)\n", 
           elements_in_slice, compute_time_ms, 
           (elements_in_slice * 3 * sizeof(float)) / (compute_time_ms * 1000000.0));
    
    return GGML_STATUS_SUCCESS;
}

// ============================================================================
// Performance and Debugging Support
// ============================================================================

/**
 * Debug function to verify NUMA data locality
 */
enum ggml_status ggml_numa_kernel_add_debug_data_locality(const struct ggml_tensor * tensor) {
#ifdef GGML_NUMA_MIRROR
    if (!tensor || !validate_tensor_inputs(tensor)) {
        printf("❌ Invalid tensor for locality debug\n");
        return GGML_STATUS_FAILED;
    }
    
    const struct ggml_tensor * src0 = tensor->src[0];
    const struct ggml_tensor * src1 = tensor->src[1];
    
    extern int ggml_numa_node_count(void);
    const int node_count = ggml_numa_node_count();
    
    printf("🔍 NUMA ADD: Data locality debug for %d nodes:\n", node_count);
    
    for (int node = 0; node < node_count && node < GGML_NUMA_MAX_NODES; node++) {
        printf("   Node %d: src0=%p, src1=%p, dst=%p\n", 
               node, 
               tensor->src[0]->__data[node],
               tensor->src[1]->__data[node], 
               tensor->__data[node]);
    }
    
    return GGML_STATUS_SUCCESS;
#else
    printf("⚠️  NUMA mirroring not compiled in\n");
    return GGML_STATUS_FAILED;
#endif
}

// ============================================================================
// Cache Population for Registry
// ============================================================================

void ggml_numa_kernel_add_populate_cache(void * cache_array) {
    ggml_numa_cache_entry_t * cache = (ggml_numa_cache_entry_t *)cache_array;
    
    // TINY: < 1K elements - single node execution
    cache[COMPLEXITY_TINY] = (ggml_numa_cache_entry_t){
        .valid = true,
        .strategy = { 
            .node_strategy = NUMA_NODE_STRATEGY_SINGLE, 
            .on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD 
        },
        .work_buffer_size_per_thread = 0,
        .work_function = ggml_numa_kernel_add_execute,
        .efficiency_score = 0.95f,
        .kernel_name = "NUMA ADD (Single/Single)"
    };
    
    // SMALL: 1K - 16K elements - single node with multi-thread
    cache[COMPLEXITY_SMALL] = (ggml_numa_cache_entry_t){
        .valid = true,
        .strategy = { 
            .node_strategy = NUMA_NODE_STRATEGY_SINGLE,
            .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD 
        },
        .work_buffer_size_per_thread = 0,
        .work_function = ggml_numa_kernel_add_execute,
        .efficiency_score = 0.96f,
        .kernel_name = "NUMA ADD (Single/Multi)"
    };
    
    // MEDIUM: 16K - 256K elements - data parallel
    cache[COMPLEXITY_MEDIUM] = (ggml_numa_cache_entry_t){
        .valid = true,
        .strategy = { 
            .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL,
            .on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD 
        },
        .work_buffer_size_per_thread = 0,
        .work_function = ggml_numa_kernel_add_execute,
        .efficiency_score = 0.97f,
        .kernel_name = "NUMA ADD (Data-Parallel/Single)"
    };
    
    // LARGE: 256K - 4M elements - data parallel with multi-thread
    cache[COMPLEXITY_LARGE] = (ggml_numa_cache_entry_t){
        .valid = true,
        .strategy = { 
            .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL,
            .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD 
        },
        .work_buffer_size_per_thread = 0,
        .work_function = ggml_numa_kernel_add_execute,
        .efficiency_score = 0.98f,
        .kernel_name = "NUMA ADD (Data-Parallel/Multi)"
    };
    
    // HUGE: > 4M elements - data parallel with multi-thread
    cache[COMPLEXITY_HUGE] = (ggml_numa_cache_entry_t){
        .valid = true,
        .strategy = { 
            .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL,
            .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD 
        },
        .work_buffer_size_per_thread = 0,
        .work_function = ggml_numa_kernel_add_execute,
        .efficiency_score = 0.98f,
        .kernel_name = "NUMA ADD (Data-Parallel/Multi)"
    };
}
