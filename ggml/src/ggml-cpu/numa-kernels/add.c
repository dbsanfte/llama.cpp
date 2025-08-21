/*
 * NUMA Kernel: Element-wise Addition (ADD)
 * 
 * Simplified implementation for the new architecture.
 * Uses direct tensor access rather than complex work dispatching.
 */

#include "../binary-ops.h"
#include "add.h"
#include <immintrin.h>  // For AVX2 SIMD instructions
#include <sys/mman.h>   // For madvise() page migration
#include "numa-kernels.h"  // For cache types
#include "../ggml-numa-shared.h"          // Shared NUMA logging and utilities
#include "../ggml-numa-simple-coordinator.h"  // For ggml_numa_get_current_node()
#include "../ggml-cpu-impl.h"
#include "../ggml-impl.h"
#include "../vec.h"  // For SIMD functions
#include "../ggml-numa-perf.h"  // Performance instrumentation

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
 * Calculate NUMA data slicing parameters based on execution strategy
 */
static void calculate_numa_slice(const struct ggml_tensor * tensor, 
                                int64_t base_start, int64_t base_end,
                                int64_t * numa_start, int64_t * numa_end) {
    const ggml_numa_execution_strategy_t strategy = ggml_numa_kernel_add_get_strategy(tensor);
    
    *numa_start = base_start;
    *numa_end = base_end;
    
    if (strategy.node_strategy == NUMA_NODE_STRATEGY_DATA_PARALLEL) {
        // Split work across NUMA nodes
        const int current_numa_node = ggml_numa_get_current_node();
        const int total_numa_nodes = numa_num_configured_nodes();
        const int64_t total_iterations = base_end - base_start;
        
        printf("DEBUG ADD: Data slicing - current_node=%d, total_nodes=%d, base_range=[%ld,%ld], total_iter=%ld\n",
               current_numa_node, total_numa_nodes, base_start, base_end, total_iterations);
        
        const int64_t elements_per_node = total_iterations / total_numa_nodes;
        const int64_t numa_offset = current_numa_node * elements_per_node;
        const int64_t numa_end_offset = (current_numa_node == total_numa_nodes - 1) ? 
            total_iterations : numa_offset + elements_per_node;
        
        *numa_start = base_start + numa_offset;
        *numa_end = base_start + numa_end_offset;
        
        printf("DEBUG ADD: Node %d processing slice [%ld,%ld] (elements_per_node=%ld)\n",
               current_numa_node, *numa_start, *numa_end, elements_per_node);
    }
    // For SINGLE strategy, use full range (no change needed)
}

/**
 * Process ADD operation for contiguous src1 (optimized SIMD path)
 */
static void process_contiguous_add(const struct ggml_tensor * tensor,
                                  const struct ggml_tensor * src0, 
                                  const struct ggml_tensor * src1,
                                  int64_t i03, int64_t i02, int64_t i01,
                                  const size_t * nb, const size_t * nb0, const size_t * nb1) {
    // Calculate memory pointers for this slice
    float * dst_ptr = (float *)((char *)tensor_data(tensor) + i03*nb[3] + i02*nb[2] + i01*nb[1]);
    const float * src0_ptr = (const float *)((const char *)tensor_data(src0) + i03*nb0[3] + i02*nb0[2] + i01*nb0[1]);
    const float * src1_ptr = (const float *)((const char *)tensor_data(src1) + 
                             (i03 % src1->ne[3])*nb1[3] + (i02 % src1->ne[2])*nb1[2] + (i01 % src1->ne[1])*nb1[1]);
    
    // SIMD optimized addition with broadcasting
    const int64_t ne0 = tensor->ne[0];
    const int64_t ne10 = src1->ne[0];
    const int64_t nr0 = ne0 / ne10;
    
    for (int64_t r = 0; r < nr0; ++r) {
        ggml_vec_add_f32(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
    }
}

/**
 * Process ADD operation for non-contiguous src1 (element-wise path)
 */
static void process_noncontiguous_add(const struct ggml_tensor * tensor,
                                     const struct ggml_tensor * src0, 
                                     const struct ggml_tensor * src1,
                                     int64_t i03, int64_t i02, int64_t i01,
                                     const size_t * nb, const size_t * nb0, const size_t * nb1) {
    // Calculate base pointers for this slice
    float * dst_ptr = (float *)((char *)tensor_data(tensor) + i03*nb[3] + i02*nb[2] + i01*nb[1]);
    const float * src0_ptr = (const float *)((const char *)tensor_data(src0) + i03*nb0[3] + i02*nb0[2] + i01*nb0[1]);
    const float * src1_base = (const float *)((const char *)tensor_data(src1) + 
                              (i03 % src1->ne[3])*nb1[3] + (i02 % src1->ne[2])*nb1[2] + (i01 % src1->ne[1])*nb1[1]);
    
    // Element-wise addition with broadcasting
    const int64_t ne0 = tensor->ne[0];
    const int64_t ne10 = src1->ne[0];
    
    for (int64_t i = 0; i < ne0; ++i) {
        const int64_t i10 = i % ne10;
        const float * src1_ptr = (const float *)((const char *)src1_base + i10*nb1[0]);
        dst_ptr[i] = src0_ptr[i] + (*src1_ptr);
    }
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
        .node_strategy = NUMA_NODE_STRATEGY_SINGLE,
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
    };
    
    if (!tensor) {
        return strategy;
    }
    
    size_t tensor_size = ggml_nelements(tensor);
    
    // Use data-parallel execution for large tensors
    if (tensor_size >= 32768) {  // 32K elements threshold
        strategy.node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL;
    }
    
    return strategy;
}

size_t ggml_numa_kernel_add_get_buffer_size(const struct ggml_tensor * tensor) {
    // Element-wise ADD doesn't need significant work buffers
    // Just a small buffer for potential temporary calculations
    (void)tensor;  // Unused for ADD
    return 1024;   // 1KB per thread should be sufficient
}

ggml_numa_work_function_t ggml_numa_kernel_add_get_work_function(const struct ggml_tensor * tensor) {
    (void)tensor;  // For ADD, we use the same work function regardless of tensor
    return ggml_numa_kernel_add_work_function;
}

float ggml_numa_kernel_add_get_efficiency(const struct ggml_tensor * tensor) {
    if (!tensor) {
        return 0.0f;
    }
    
    size_t tensor_size = ggml_nelements(tensor);
    
    // ADD is highly parallel, efficiency depends mainly on data size
    if (tensor_size >= 32768) {
        return 0.95f;  // Very high efficiency for large tensors
    } else if (tensor_size >= 4096) {
        return 0.80f;  // Good efficiency for medium tensors
    } else {
        return 0.50f;  // Lower efficiency for small tensors due to overhead
    }
}

// Work function that will be called by the coordinator
enum ggml_status ggml_numa_kernel_add_work_function(void * work_context, struct ggml_compute_params * params) {
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
    
    // Extract tensor dimensions and strides
    const size_t nb[4]  = { tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3] };
    const size_t nb0[4] = { src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3] };
    const size_t nb1[4] = { src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3] };
    
    // Validate stride assumptions
    NUMA_ASSERT(nb[0] == sizeof(float), "Destination stride must match float size");
    NUMA_ASSERT(nb0[0] == sizeof(float), "Source 0 stride must match float size");
    
    // Calculate work range for element-wise operation (not row-wise!)
    // For ADD, we want to process ALL elements as a flat array for optimal SIMD
    const int64_t total_elements = ggml_nelements(tensor);
    int64_t numa_start, numa_end;
    calculate_numa_slice(tensor, 0, total_elements, &numa_start, &numa_end);
    
    printf("DEBUG ADD: Total elements=%ld, processing range [%ld,%ld]\n", 
           total_elements, numa_start, numa_end);
    
    // Check if src1 is contiguous for optimization
    const bool is_src1_contiguous = (nb1[0] == sizeof(float));
    
    if (!is_src1_contiguous) {
        // Ensure compatible shapes for non-contiguous case
        NUMA_ASSERT(ggml_are_same_shape(src0, src1), "Non-contiguous broadcast requires same shapes");
    }
    
    // Process assigned NUMA slice directly on flattened tensor data
    // For element-wise operations, work with flat array for optimal performance
    const float * src0_data = (const float *)tensor_data(src0);
    const float * src1_data = (const float *)tensor_data(src1);
    float * dst_data = (float *)tensor_data(tensor);
    
    // Process elements in this NUMA slice
    size_t elements_in_slice = numa_end - numa_start;
    const int current_numa_node = ggml_numa_get_current_node();
    
    printf("DEBUG ADD KERNEL: Processing %zu elements from %zu to %zu on NUMA node %d\n", 
           elements_in_slice, numa_start, numa_end, current_numa_node);

    printf("DEBUG ADD KERNEL: Implementing FIRST TOUCH policy for optimal NUMA locality\n");

    // Set local allocation policy for this thread to ensure new pages go to current NUMA node
    if (numa_run_on_node(current_numa_node) == 0) {
        printf("DEBUG ADD KERNEL: Set thread affinity to NUMA node %d\n", current_numa_node);
    }
    
    // Set memory allocation policy to prefer local node
    struct bitmask *local_nodes = numa_allocate_nodemask();
    numa_bitmask_setbit(local_nodes, current_numa_node);
    numa_set_membind(local_nodes);
    printf("DEBUG ADD KERNEL: Set memory binding to NUMA node %d\n", current_numa_node);
    numa_free_nodemask(local_nodes);

    // SMART PAGE MIGRATION: Check if we can safely migrate destination pages
    // Only attempt madvise on aligned memory regions that we fully control
    float* dst_ptr = (float*)dst_data + numa_start;
    const float* src0_ptr = src0_data + numa_start;
    const float* src1_ptr = src1_data + numa_start;
    
    printf("DEBUG ADD KERNEL: Smart page migration to NUMA node %d\n", current_numa_node);
    
    struct timespec migration_start, migration_end;
    clock_gettime(CLOCK_MONOTONIC, &migration_start);
    
    const size_t page_size = 4096;
    const size_t page_size_floats = page_size / sizeof(float);  // 1024 floats per page
    
    // For destination memory: attempt page migration only if we have page-aligned access
    uintptr_t dst_addr = (uintptr_t)dst_ptr;
    size_t dst_size = elements_in_slice * sizeof(float);
    
    // Check if our slice starts on a page boundary and covers complete pages
    bool can_migrate_dst = (dst_addr % page_size == 0) && (dst_size >= page_size) && (dst_size % page_size == 0);
    
    if (can_migrate_dst && dst_size <= 256 * 1024 * 1024) { // Safety limit: 256MB max
        printf("DEBUG ADD KERNEL: Attempting safe page migration for %zu bytes at %p\n", dst_size, (void*)dst_addr);
        
        if (madvise((void*)dst_addr, dst_size, MADV_DONTNEED) == 0) {
            printf("DEBUG ADD KERNEL: ✅ Destination pages unmapped successfully\n");
            
            // First-touch to reallocate on current NUMA node
            volatile float* dst_volatile = (volatile float*)dst_ptr;
            for (size_t i = 0; i < elements_in_slice; i += page_size_floats) {
                dst_volatile[i] = 0.0f; // Write to allocate page on current node
            }
            printf("DEBUG ADD KERNEL: ✅ Destination pages reallocated on node %d\n", current_numa_node);
        } else {
            printf("DEBUG ADD KERNEL: ⚠️  Page migration failed, using fallback approach\n");
            can_migrate_dst = false;
        }
    } else {
        printf("DEBUG ADD KERNEL: ⚠️  Unsafe to migrate (addr=%p, size=%zu, aligned=%s), using first-touch\n", 
               (void*)dst_addr, dst_size, (dst_addr % page_size == 0) ? "yes" : "no");
        can_migrate_dst = false;
    }
    
    // For source memory: always use safe first-touch (can't use MADV_DONTNEED as it would lose data)
    printf("DEBUG ADD KERNEL: First-touch prefaulting source pages\n");
    volatile float prefault_sum = 0.0f;  // Prevent optimization
    
    for (size_t i = 0; i < elements_in_slice; i += page_size_floats) {
        // Touch each page to ensure it's accessible and potentially migrate if kernel allows
        prefault_sum += src0_ptr[i] + src1_ptr[i];
        
        // Touch end of page too if different
        size_t page_end = (i + page_size_floats < elements_in_slice) ? i + page_size_floats : elements_in_slice;
        if (page_end - 1 > i) {
            prefault_sum += src0_ptr[page_end - 1] + src1_ptr[page_end - 1];
        }
    }
    
    // If we couldn't migrate destination pages, use first-touch
    if (!can_migrate_dst) {
        volatile float* dst_volatile = (volatile float*)dst_ptr;
        for (size_t i = 0; i < elements_in_slice; i += page_size_floats) {
            dst_volatile[i] = 0.0f; // First-touch allocation
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &migration_end);
    double migration_time_ms = (migration_end.tv_sec - migration_start.tv_sec) * 1000.0 + 
                               (migration_end.tv_nsec - migration_start.tv_nsec) / 1000000.0;
    printf("DEBUG ADD KERNEL: Memory preparation completed in %.3fms (sum=%.6f to prevent optimization)\n", 
           migration_time_ms, prefault_sum);

    printf("DEBUG ADD KERNEL: Starting SIMD-optimized ADD for %zu elements on NUMA node %d\n", 
           elements_in_slice, current_numa_node);    struct timespec time_start, time_end;
    clock_gettime(CLOCK_MONOTONIC, &time_start);
    
    // Optimized SIMD ADD implementation using AVX2
    // This is similar to what the raw test uses
    const size_t simd_width = 8; // AVX2 processes 8 floats at once
    const size_t simd_end_idx = (elements_in_slice / simd_width) * simd_width;
    
#if defined(__AVX2__)
    // Process SIMD chunks using AVX2
    for (size_t i = 0; i < simd_end_idx; i += simd_width) {
        __m256 a = _mm256_loadu_ps(&src0_ptr[i]);
        __m256 b = _mm256_loadu_ps(&src1_ptr[i]);
        __m256 result = _mm256_add_ps(a, b);
        _mm256_storeu_ps(&dst_ptr[i], result);
    }
#else
    // Fallback to scalar for the SIMD portion if AVX2 not available
    for (size_t i = 0; i < simd_end_idx; i++) {
        dst_ptr[i] = src0_ptr[i] + src1_ptr[i];
    }
#endif
    
    // Handle remainder elements
    for (size_t i = simd_end_idx; i < elements_in_slice; i++) {
        dst_ptr[i] = src0_ptr[i] + src1_ptr[i];
    }
    
    clock_gettime(CLOCK_MONOTONIC, &time_end);
    double simd_time_ms = (time_end.tv_sec - time_start.tv_sec) * 1000.0 + 
                          (time_end.tv_nsec - time_start.tv_nsec) / 1000000.0;
    printf("DEBUG ADD KERNEL: Direct SIMD time: %.3fms for %zu elements\n", simd_time_ms, elements_in_slice);
    
    printf("DEBUG ADD KERNEL: Completed NUMA node %d processing\n", current_numa_node);
    
    return GGML_STATUS_SUCCESS;
}

// Cache registration interface - kernel provides its own cache entries
void ggml_numa_kernel_add_populate_cache(void * cache_array) {
    ggml_numa_cache_entry_t * cache = (ggml_numa_cache_entry_t *)cache_array;
    // Pre-compute all ADD operation strategies across complexity classes
    for (int complexity = 0; complexity < COMPLEXITY_COUNT; complexity++) {
        ggml_numa_cache_entry_t * entry = &cache[complexity];
        
        entry->valid = true;
        entry->kernel_name = "NUMA Add";
        entry->work_function = ggml_numa_kernel_add_work_function;
        entry->work_buffer_size_per_thread = 1024; // Minimal for element-wise ops
        switch (complexity) {
            case COMPLEXITY_TINY:
            case COMPLEXITY_SMALL:
                // Small tensors: single-node execution
                entry->strategy.node_strategy = NUMA_NODE_STRATEGY_SINGLE;
                entry->strategy.on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD;
                entry->efficiency_score = 0.60f; // Lower due to overhead
                break;
                
            case COMPLEXITY_MEDIUM:
            case COMPLEXITY_LARGE:
            case COMPLEXITY_HUGE:
                // Large tensors: data-parallel across nodes
                entry->strategy.node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL;
                entry->strategy.on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD;
                entry->efficiency_score = 0.95f; // Excellent for large parallel work
                break;
                
            default:
                entry->valid = false;
                break;
        }
    }
}

enum ggml_status ggml_numa_kernel_add_execute(struct ggml_tensor * tensor, struct ggml_cplan * cplan) {
    int current_numa_node = ggml_numa_get_current_node();
    size_t tensor_size = ggml_nbytes(tensor);
    
    NUMA_PERF_START(NUMA_PERF_KERNEL_NUMA_EXEC, "ADD", "numa_add_kernel", current_numa_node, tensor_size, 1);
    
    if (!ggml_numa_kernel_add_supports(tensor)) {
        NUMA_PERF_END();
        return GGML_STATUS_FAILED;
    }
    
    const struct ggml_tensor * src0 = tensor->src[0];
    const struct ggml_tensor * src1 = tensor->src[1];
    
    const int64_t total_elements = ggml_nelements(tensor);
    
    GGML_LOG_DEBUG("ADD kernel: Processing %ld elements with %d threads\n", 
                   total_elements, cplan ? cplan->n_threads : 1);
    
    // Add timing for actual computation
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    // Direct SIMD addition across all elements
    ggml_vec_add_f32(total_elements,
        (float *)ggml_get_data(tensor),
        (const float *)ggml_get_data(src0),
        (const float *)ggml_get_data(src1));
    
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    
    double computation_time_ms = ((end_time.tv_sec - start_time.tv_sec) * 1000.0) +
                                ((end_time.tv_nsec - start_time.tv_nsec) / 1000000.0);
    
    printf("DEBUG ADD: Node %d - Computation time: %.3fms for %ld elements\n", 
           current_numa_node, computation_time_ms, total_elements);
    
    NUMA_PERF_END();
    return GGML_STATUS_SUCCESS;
}

// Legacy function for backward compatibility (different signature)
float ggml_numa_kernel_add_get_efficiency_legacy(const struct ggml_tensor * tensor, size_t tensor_size) {
    if (!ggml_numa_kernel_add_supports(tensor)) {
        return -1.0f;
    }
    
    // ADD is highly parallelizable - excellent efficiency for large tensors
    const size_t OPTIMAL_SIZE = 4096;
    
    if (tensor_size >= OPTIMAL_SIZE * 4) {
        return 0.95f;  // Excellent efficiency for large tensors
    } else if (tensor_size >= OPTIMAL_SIZE) {
        return 0.85f;  // Good efficiency for medium tensors
    } else {
        return 0.6f;   // Reduced efficiency for small tensors
    }
}
