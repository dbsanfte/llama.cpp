
#include "ggml-numa-add.h"
#include "../ggml-numa-work-shared.h"
#include "../ggml-numa-operation-dispatch.h"
#include "../ggml-numa-coordinator.h"
#include "../../ggml-impl.h"
#include "../ggml-cpu-impl.h"
#include "../../include/ggml.h"
#include "../vec.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <pthread.h>

#ifdef __linux__
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#endif

// Specialized work function for ADD operations with NUMA-aware chunking and multi-level parallelism
enum ggml_status ggml_numa_work_function_add_chunk(void * work_context, struct ggml_compute_params * params) {
    if (!work_context) {
        GGML_LOG_ERROR("ADD work function: Invalid work context\n");
        return GGML_STATUS_FAILED;
    }
    
    ggml_numa_dispatcher_work_context_t * ctx = (ggml_numa_dispatcher_work_context_t *)work_context;
    
    if (!ctx->operation) {
        GGML_LOG_ERROR("ADD work function: Operation is NULL\n");
        return GGML_STATUS_FAILED;
    }
    
    GGML_LOG_DEBUG("ADD work function: operation=%p, type=%d, elements=%ld\n", 
                   (void*)ctx->operation, ctx->operation->op, ggml_nelements(ctx->operation));
    
    // Validate ADD operation has two source tensors
    const struct ggml_tensor * src0 = ctx->operation->src[0];
    const struct ggml_tensor * src1 = ctx->operation->src[1];
    
    if (!src0 || !src1) {
        GGML_LOG_ERROR("ADD work function: Missing source tensors (src0=%p, src1=%p)\n", 
                       (void*)src0, (void*)src1);
        return GGML_STATUS_FAILED;
    }
    
    // Check tensor compatibility
    if (!ggml_are_same_shape(src0, ctx->operation)) {
        GGML_LOG_ERROR("ADD work function: src0 and destination shapes don't match\n");
        return GGML_STATUS_FAILED;
    }
    
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ADD work function: Only F32 tensors supported\n");
        return GGML_STATUS_FAILED;
    }
    
    // STRICT VALIDATION: Check input tensors for NaN/inf corruption
    const float* src0_data = (const float*)ggml_get_data(src0);
    const float* src1_data = (const float*)ggml_get_data(src1);
    
    // Check first 16 elements of src0 for corruption
    int64_t check_count = ggml_nelements(src0);
    check_count = (check_count > 16) ? 16 : check_count;
    for (int64_t i = 0; i < check_count; i++) {
        NUMA_ASSERT(isfinite(src0_data[i]), "Found NaN/inf in src0 data at index %d: %f", (int)i, (double)src0_data[i]);
    }
    
    // Check first 16 elements of src1 for corruption  
    check_count = ggml_nelements(src1);
    check_count = (check_count > 16) ? 16 : check_count;
    for (int64_t i = 0; i < check_count; i++) {
        NUMA_ASSERT(isfinite(src1_data[i]), "Found NaN/inf in src1 data at index %d: %f", (int)i, (double)src1_data[i]);
    }
    
    // Extract tensor dimensions for element-wise operations
    const int64_t ne0 = src0->ne[0];  // Elements per row
    const int64_t ne1 = src0->ne[1];  // Number of rows  
    const int64_t ne2 = src0->ne[2];  // Batch dimension 2
    const int64_t ne3 = src0->ne[3];  // Batch dimension 3
    
    const size_t nb0 = src0->nb[0];   // Element stride (should be sizeof(float))
    const size_t nb1 = src0->nb[1];   // Row stride in bytes
    const size_t nb2 = src0->nb[2];   // Batch stride 2 in bytes 
    const size_t nb3 = src0->nb[3];   // Batch stride 3 in bytes
    
    const size_t src1_nb0 = src1->nb[0];
    const size_t src1_nb1 = src1->nb[1];
    const size_t src1_nb2 = src1->nb[2];
    const size_t src1_nb3 = src1->nb[3];
    
    const size_t dst_nb0 = ctx->operation->nb[0];
    const size_t dst_nb1 = ctx->operation->nb[1];
    const size_t dst_nb2 = ctx->operation->nb[2];
    const size_t dst_nb3 = ctx->operation->nb[3];
    
    GGML_LOG_DEBUG("ADD work function: Processing tensor [%ld, %ld, %ld, %ld]\n", ne0, ne1, ne2, ne3);

    // NUMA + Thread Multi-Level Parallel ADD Implementation
    // Level 1: NUMA-level parallelism (different element ranges per NUMA node)
    // Level 2: Thread-level parallelism (subdivision within NUMA node)
    
    // Get NUMA node information from coordinator
    extern struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_get_global(int n_threads, bool force_multi_socket);
    extern int ggml_numa_coordinator_manager_get_numa_nodes(struct ggml_numa_coordinator_manager * mgr);
    
    int numa_node = ggml_numa_get_current_node();
    struct ggml_numa_coordinator_manager * mgr = ggml_numa_coordinator_manager_get_global(8, false);
    int max_numa_nodes = mgr ? ggml_numa_coordinator_manager_get_numa_nodes(mgr) : 1;
    
    // Handle fallback case where virtual node is not set
    if (numa_node < 0) {
        numa_node = 0;  // Default to node 0
    }
    if (max_numa_nodes <= 0) {
        max_numa_nodes = 1;  // At least one node
    }

    // Calculate total number of elements for distribution
    int64_t total_elements = ggml_nelements(ctx->operation);
    
    // For single-node execution (when used by single-node dispatcher), process all elements
    // For multi-node execution (when used by data-parallel dispatcher), distribute elements
    int64_t numa_start_elem, numa_end_elem;
    
    // Check if we're being called from single-node execution (max_numa_nodes == 1) 
    // or from data-parallel execution with multiple nodes
    if (max_numa_nodes == 1) {
        // Single-node execution: process entire tensor on this node
        numa_start_elem = 0;
        numa_end_elem = total_elements;
        GGML_LOG_DEBUG("ADD single-node execution: processing all %ld elements\n", total_elements);
    } else {
        // Multi-node data parallel execution: distribute elements across nodes
        int64_t elements_per_node = total_elements / max_numa_nodes;
        int64_t remainder_elements = total_elements % max_numa_nodes;
        
        // Calculate this NUMA node's element range
        numa_start_elem = numa_node * elements_per_node;
        numa_end_elem = numa_start_elem + elements_per_node;
        
        // Distribute remainder elements among first few nodes
        if (numa_node < remainder_elements) {
            numa_start_elem += numa_node;
            numa_end_elem += numa_node + 1;
        } else {
            numa_start_elem += remainder_elements;
            numa_end_elem += remainder_elements;
        }
        
        // Ensure we don't exceed tensor bounds and have valid ranges
        if (numa_end_elem > total_elements) {
            numa_end_elem = total_elements;
        }
        if (numa_start_elem < 0) {
            numa_start_elem = 0;
        }
        if (numa_end_elem <= numa_start_elem) {
            numa_end_elem = numa_start_elem + 1;  // Ensure at least one element
        }
        
        GGML_LOG_DEBUG("ADD data-parallel execution: NUMA node %d (of %d) processing elements %ld to %ld\n", 
                       numa_node, max_numa_nodes, numa_start_elem, numa_end_elem - 1);
    }

    // Thread-level parallelism within this NUMA node's assigned elements
    int64_t numa_node_elements = numa_end_elem - numa_start_elem;

    // Use existing threadpool architecture if available, otherwise fall back to single-threaded
    if (params && params->nth > 1 && params->threadpool) {
        GGML_LOG_DEBUG("ADD: Using multi-threaded execution (%d threads) on NUMA node %d (CPU: %d)\n", 
                       params->nth, numa_node, sched_getcpu());

        // Create thread data for each thread
        struct add_thread_data * thread_data = (struct add_thread_data *)calloc(params->nth, sizeof(struct add_thread_data));
        if (!thread_data) {
            GGML_LOG_ERROR("ADD: Failed to allocate thread data\n");
            return GGML_STATUS_FAILED;
        }

        // Distribute NUMA node's elements among threads
        int64_t elements_per_thread = numa_node_elements / params->nth;
        int64_t remainder_thread_elements = numa_node_elements % params->nth;

        for (int t = 0; t < params->nth; t++) {
            struct add_thread_data * td = &thread_data[t];
            
            // Copy common data
            td->src0 = src0;
            td->src1 = src1;
            td->dst = ctx->operation;
            td->ne0 = ne0; td->ne1 = ne1; td->ne2 = ne2; td->ne3 = ne3;
            td->nb0 = nb0; td->nb1 = nb1; td->nb2 = nb2; td->nb3 = nb3;
            td->src1_nb0 = src1_nb0; td->src1_nb1 = src1_nb1; td->src1_nb2 = src1_nb2; td->src1_nb3 = src1_nb3;
            td->dst_nb0 = dst_nb0; td->dst_nb1 = dst_nb1; td->dst_nb2 = dst_nb2; td->dst_nb3 = dst_nb3;
            td->thread_id = t;
            td->numa_node = numa_node;
            
            // Calculate thread-specific element range within NUMA node's elements
            int64_t thread_elem_start = t * elements_per_thread;
            int64_t thread_elem_end = thread_elem_start + elements_per_thread;
            
            // Distribute remainder elements among first few threads
            if (t < remainder_thread_elements) {
                thread_elem_start += t;
                thread_elem_end += t + 1;
            } else {
                thread_elem_start += remainder_thread_elements;
                thread_elem_end += remainder_thread_elements;
            }
            
            // Convert to absolute element indices
            td->thread_start_elem = numa_start_elem + thread_elem_start;
            td->thread_end_elem = numa_start_elem + thread_elem_end;
            
            // Bounds checking
            if (td->thread_end_elem > numa_end_elem) {
                td->thread_end_elem = numa_end_elem;
            }
            if (td->thread_start_elem >= td->thread_end_elem) {
                td->thread_end_elem = td->thread_start_elem + 1; // Ensure at least one element
            }
            
            GGML_LOG_DEBUG("ADD thread %d on NUMA %d: processing elements %ld to %ld\n", 
                           t, numa_node, td->thread_start_elem, td->thread_end_elem - 1);
        }

        // Execute using pthread directly for multi-threading within NUMA node
        pthread_t* threads = (pthread_t*)malloc(params->nth * sizeof(pthread_t));
        if (!threads) {
            free(thread_data);
            GGML_LOG_ERROR("ADD: Failed to allocate thread handles\n");
            return GGML_STATUS_FAILED;
        }
        
        // Create threads
        for (int t = 0; t < params->nth; t++) {
            int ret = pthread_create(&threads[t], NULL, add_thread_kernel, &thread_data[t]);
            if (ret != 0) {
                GGML_LOG_ERROR("ADD: Failed to create thread %d: %d\n", t, ret);
                // Clean up already created threads
                for (int i = 0; i < t; i++) {
                    pthread_join(threads[i], NULL);
                }
                free(threads);
                free(thread_data);
                return GGML_STATUS_FAILED;
            }
        }
        
        // Wait for all threads to complete
        for (int t = 0; t < params->nth; t++) {
            pthread_join(threads[t], NULL);
        }
        
        free(threads);
        free(thread_data);
        
        GGML_LOG_DEBUG("ADD: Multi-threaded execution completed on NUMA node %d\n", numa_node);
        
    } else {
        // Single-threaded execution within this NUMA node
        GGML_LOG_DEBUG("ADD: Using single-threaded execution on NUMA node %d (CPU: %d)\n", 
                       numa_node, sched_getcpu());
                       
        // Process elements in this NUMA node's assigned range
        for (int64_t i = numa_start_elem; i < numa_end_elem; i++) {
            // Convert linear index to multi-dimensional indices
            int64_t i3 = i / (ne2 * ne1 * ne0);
            int64_t i2 = (i % (ne2 * ne1 * ne0)) / (ne1 * ne0);
            int64_t i1 = (i % (ne1 * ne0)) / ne0;
            int64_t i0 = i % ne0;
            
            // Calculate memory addresses for current element
            const float * src0_ptr = (float *) ((char *) ggml_get_data(src0) + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
            const float * src1_ptr = (float *) ((char *) ggml_get_data(src1) + i3*src1_nb3 + i2*src1_nb2 + i1*src1_nb1 + i0*src1_nb0);
            float * dst_ptr = (float *) ((char *) ggml_get_data(ctx->operation) + i3*dst_nb3 + i2*dst_nb2 + i1*dst_nb1 + i0*dst_nb0);
            
            // ADD SIMD-optimized mathematical kernel: dst = src0 + src1
            // Check if we can process a contiguous chunk using SIMD
            if (i0 == 0 && ne0 > 1 && 
                nb0 == sizeof(float) && src1_nb0 == sizeof(float) && dst_nb0 == sizeof(float)) {
                // Process entire row with SIMD if contiguous
                const float * src0_row = (float *) ((char *) ggml_get_data(src0) + i3*nb3 + i2*nb2 + i1*nb1);
                const float * src1_row = (float *) ((char *) ggml_get_data(src1) + i3*src1_nb3 + i2*src1_nb2 + i1*src1_nb1);
                float * dst_row = (float *) ((char *) ggml_get_data(ctx->operation) + i3*dst_nb3 + i2*dst_nb2 + i1*dst_nb1);
                
                ggml_vec_add_f32(ne0, dst_row, src0_row, src1_row);
                
                // Skip remaining elements in this row since we processed them all
                i += (ne0 - 1); // -1 because loop will increment i
            } else {
                // Fallback to element-wise operation for non-contiguous or single elements
                *dst_ptr = *src0_ptr + *src1_ptr;
            }
        }
    }

    // Memory barrier to ensure all writes are visible before returning
    __sync_synchronize();
    
    GGML_LOG_DEBUG("Successfully executed ADD chunk work function\n");
    
    return GGML_STATUS_SUCCESS;
}

// Single-node ADD work function for processing entire tensor on one NUMA node
enum ggml_status ggml_numa_work_function_add_single(void * work_context, struct ggml_compute_params * params) {
    if (!work_context) {
        GGML_LOG_ERROR("ADD single work function: Invalid work context\n");
        return GGML_STATUS_FAILED;
    }
    
    ggml_numa_dispatcher_work_context_t * ctx = (ggml_numa_dispatcher_work_context_t *)work_context;
    
    if (!ctx->operation) {
        GGML_LOG_ERROR("ADD single work function: Operation is NULL\n");
        return GGML_STATUS_FAILED;
    }
    
    GGML_LOG_DEBUG("ADD single work function: operation=%p, type=%d, elements=%ld\n", 
                   (void*)ctx->operation, ctx->operation->op, ggml_nelements(ctx->operation));
    
    // Validate ADD operation has two source tensors
    const struct ggml_tensor * src0 = ctx->operation->src[0];
    const struct ggml_tensor * src1 = ctx->operation->src[1];
    
    if (!src0 || !src1) {
        GGML_LOG_ERROR("ADD single work function: Missing source tensors (src0=%p, src1=%p)\n", 
                       (void*)src0, (void*)src1);
        return GGML_STATUS_FAILED;
    }
    
    // Check tensor compatibility
    if (!ggml_are_same_shape(src0, ctx->operation)) {
        GGML_LOG_ERROR("ADD single work function: src0 and destination shapes don't match\n");
        return GGML_STATUS_FAILED;
    }
    
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ADD single work function: Only F32 tensors supported\n");
        return GGML_STATUS_FAILED;
    }
    
    // STRICT VALIDATION: Check input tensors for NaN/inf corruption
    const float* src0_data = (const float*)ggml_get_data(src0);
    const float* src1_data = (const float*)ggml_get_data(src1);
    
    // Check first 16 elements of src0 for corruption
    int64_t check_count = ggml_nelements(src0);
    check_count = (check_count > 16) ? 16 : check_count;
    for (int64_t i = 0; i < check_count; i++) {
        NUMA_ASSERT(isfinite(src0_data[i]), "Found NaN/inf in src0 data at index %d: %f", (int)i, (double)src0_data[i]);
    }
    
    // Check first 16 elements of src1 for corruption  
    check_count = ggml_nelements(src1);
    check_count = (check_count > 16) ? 16 : check_count;
    for (int64_t i = 0; i < check_count; i++) {
        NUMA_ASSERT(isfinite(src1_data[i]), "Found NaN/inf in src1 data at index %d: %f", (int)i, (double)src1_data[i]);
    }
    
    // Extract tensor dimensions for element-wise operations
    const int64_t ne0 = src0->ne[0];  // Elements per row
    const int64_t ne1 = src0->ne[1];  // Number of rows  
    const int64_t ne2 = src0->ne[2];  // Batch dimension 2
    const int64_t ne3 = src0->ne[3];  // Batch dimension 3
    
    const size_t nb0 = src0->nb[0];   // Element stride (should be sizeof(float))
    const size_t nb1 = src0->nb[1];   // Row stride in bytes
    const size_t nb2 = src0->nb[2];   // Batch stride 2 in bytes 
    const size_t nb3 = src0->nb[3];   // Batch stride 3 in bytes
    
    const size_t src1_nb0 = src1->nb[0];
    const size_t src1_nb1 = src1->nb[1];
    const size_t src1_nb2 = src1->nb[2];
    const size_t src1_nb3 = src1->nb[3];
    
    const size_t dst_nb0 = ctx->operation->nb[0];
    const size_t dst_nb1 = ctx->operation->nb[1];
    const size_t dst_nb2 = ctx->operation->nb[2];
    const size_t dst_nb3 = ctx->operation->nb[3];
    
    GGML_LOG_DEBUG("ADD single work function: Processing tensor [%ld, %ld, %ld, %ld]\n", ne0, ne1, ne2, ne3);

    // Single-node ADD implementation - process entire tensor
    int64_t total_elements = ggml_nelements(ctx->operation);
    
    GGML_LOG_DEBUG("ADD single-node execution: processing all %ld elements\n", total_elements);

    // Use existing threadpool architecture if available, otherwise fall back to single-threaded
    if (params && params->nth > 1 && params->threadpool) {
        GGML_LOG_DEBUG("ADD: Using multi-threaded execution (%d threads) for single-node processing (CPU: %d)\n", 
                       params->nth, sched_getcpu());

        // Create thread data for each thread
        struct add_thread_data * thread_data = (struct add_thread_data *)calloc(params->nth, sizeof(struct add_thread_data));
        if (!thread_data) {
            GGML_LOG_ERROR("ADD: Failed to allocate thread data\n");
            return GGML_STATUS_FAILED;
        }

        // Distribute entire tensor among threads (no NUMA node splitting)
        int64_t elements_per_thread = total_elements / params->nth;
        int64_t remainder_thread_elements = total_elements % params->nth;

        for (int t = 0; t < params->nth; t++) {
            struct add_thread_data * td = &thread_data[t];
            
            // Copy common data
            td->src0 = src0;
            td->src1 = src1;
            td->dst = ctx->operation;
            td->ne0 = ne0; td->ne1 = ne1; td->ne2 = ne2; td->ne3 = ne3;
            td->nb0 = nb0; td->nb1 = nb1; td->nb2 = nb2; td->nb3 = nb3;
            td->src1_nb0 = src1_nb0; td->src1_nb1 = src1_nb1; td->src1_nb2 = src1_nb2; td->src1_nb3 = src1_nb3;
            td->dst_nb0 = dst_nb0; td->dst_nb1 = dst_nb1; td->dst_nb2 = dst_nb2; td->dst_nb3 = dst_nb3;
            td->thread_id = t;
            td->numa_node = 0;  // Single NUMA node (node 0)
            
            // Calculate thread-specific element range for entire tensor
            int64_t thread_elem_start = t * elements_per_thread;
            int64_t thread_elem_end = thread_elem_start + elements_per_thread;
            
            // Distribute remainder elements among first few threads
            if (t < remainder_thread_elements) {
                thread_elem_start += t;
                thread_elem_end += t + 1;
            } else {
                thread_elem_start += remainder_thread_elements;
                thread_elem_end += remainder_thread_elements;
            }
            
            // Store absolute element indices
            td->thread_start_elem = thread_elem_start;
            td->thread_end_elem = thread_elem_end;
            
            // Bounds checking
            if (td->thread_end_elem > total_elements) {
                td->thread_end_elem = total_elements;
            }
            if (td->thread_start_elem >= td->thread_end_elem) {
                td->thread_end_elem = td->thread_start_elem + 1; // Ensure at least one element
            }
            
            GGML_LOG_DEBUG("ADD single-node thread %d: processing elements %ld to %ld\n", 
                           t, td->thread_start_elem, td->thread_end_elem - 1);
        }

        // Execute using pthread directly for multi-threading
        pthread_t* threads = (pthread_t*)malloc(params->nth * sizeof(pthread_t));
        if (!threads) {
            free(thread_data);
            GGML_LOG_ERROR("ADD: Failed to allocate thread handles\n");
            return GGML_STATUS_FAILED;
        }
        
        // Create threads
        for (int t = 0; t < params->nth; t++) {
            int ret = pthread_create(&threads[t], NULL, add_thread_kernel, &thread_data[t]);
            if (ret != 0) {
                GGML_LOG_ERROR("ADD: Failed to create thread %d: %d\n", t, ret);
                // Clean up already created threads
                for (int i = 0; i < t; i++) {
                    pthread_join(threads[i], NULL);
                }
                free(threads);
                free(thread_data);
                return GGML_STATUS_FAILED;
            }
        }
        
        // Wait for all threads to complete
        for (int t = 0; t < params->nth; t++) {
            pthread_join(threads[t], NULL);
        }
        
        free(threads);
        free(thread_data);
        
        GGML_LOG_DEBUG("ADD: Multi-threaded single-node execution completed\n");
        
    } else {
        // Single-threaded execution for entire tensor
        GGML_LOG_DEBUG("ADD: Using single-threaded execution for single-node processing (CPU: %d)\n", 
                       sched_getcpu());
                       
        // Process all elements in the tensor
        for (int64_t i = 0; i < total_elements; i++) {
            // Convert linear index to multi-dimensional indices
            int64_t i3 = i / (ne2 * ne1 * ne0);
            int64_t i2 = (i % (ne2 * ne1 * ne0)) / (ne1 * ne0);
            int64_t i1 = (i % (ne1 * ne0)) / ne0;
            int64_t i0 = i % ne0;
            
            // Calculate memory addresses for current element
            const float * src0_ptr = (float *) ((char *) ggml_get_data(src0) + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
            const float * src1_ptr = (float *) ((char *) ggml_get_data(src1) + i3*src1_nb3 + i2*src1_nb2 + i1*src1_nb1 + i0*src1_nb0);
            float * dst_ptr = (float *) ((char *) ggml_get_data(ctx->operation) + i3*dst_nb3 + i2*dst_nb2 + i1*dst_nb1 + i0*dst_nb0);
            
            // ADD SIMD-optimized mathematical kernel: dst = src0 + src1
            // Check if we can process a contiguous chunk using SIMD
            if (i0 == 0 && ne0 > 1 && 
                nb0 == sizeof(float) && src1_nb0 == sizeof(float) && dst_nb0 == sizeof(float)) {
                // Process entire row with SIMD if contiguous
                const float * src0_row = (float *) ((char *) ggml_get_data(src0) + i3*nb3 + i2*nb2 + i1*nb1);
                const float * src1_row = (float *) ((char *) ggml_get_data(src1) + i3*src1_nb3 + i2*src1_nb2 + i1*src1_nb1);
                float * dst_row = (float *) ((char *) ggml_get_data(ctx->operation) + i3*dst_nb3 + i2*dst_nb2 + i1*dst_nb1);
                
                ggml_vec_add_f32(ne0, dst_row, src0_row, src1_row);
                
                // Skip remaining elements in this row since we processed them all
                i += (ne0 - 1); // -1 because loop will increment i
            } else {
                // Fallback to element-wise operation for non-contiguous or single elements
                *dst_ptr = *src0_ptr + *src1_ptr;
            }
        }
    }

    // Memory barrier to ensure all writes are visible before returning
    __sync_synchronize();
    
    GGML_LOG_DEBUG("Successfully executed ADD single-node work function\n");
    
    return GGML_STATUS_SUCCESS;
}

// Thread kernel functions for multi-level parallelism
void* add_thread_kernel(void* data) {
    struct add_thread_data * td = (struct add_thread_data *)data;
    
    // Process elements in this thread's assigned range
    for (int64_t i = td->thread_start_elem; i < td->thread_end_elem; i++) {
        // Convert linear index to multi-dimensional indices
        int64_t i3 = i / (td->ne2 * td->ne1 * td->ne0);
        int64_t i2 = (i % (td->ne2 * td->ne1 * td->ne0)) / (td->ne1 * td->ne0);
        int64_t i1 = (i % (td->ne1 * td->ne0)) / td->ne0;
        int64_t i0 = i % td->ne0;
        
        // Calculate memory addresses for current element
        const float * src0_ptr = (float *) ((char *) ggml_get_data(td->src0) + i3*td->nb3 + i2*td->nb2 + i1*td->nb1 + i0*td->nb0);
        const float * src1_ptr = (float *) ((char *) ggml_get_data(td->src1) + i3*td->src1_nb3 + i2*td->src1_nb2 + i1*td->src1_nb1 + i0*td->src1_nb0);
        float * dst_ptr = (float *) ((char *) ggml_get_data(td->dst) + i3*td->dst_nb3 + i2*td->dst_nb2 + i1*td->dst_nb1 + i0*td->dst_nb0);
        
        // ADD SIMD-optimized mathematical kernel: dst = src0 + src1
        // Check if we can process a contiguous chunk using SIMD
        if (i0 == 0 && td->ne0 > 1 && 
            td->nb0 == sizeof(float) && td->src1_nb0 == sizeof(float) && td->dst_nb0 == sizeof(float)) {
            // Process entire row with SIMD if contiguous
            const float * src0_row = (float *) ((char *) ggml_get_data(td->src0) + i3*td->nb3 + i2*td->nb2 + i1*td->nb1);
            const float * src1_row = (float *) ((char *) ggml_get_data(td->src1) + i3*td->src1_nb3 + i2*td->src1_nb2 + i1*td->src1_nb1);
            float * dst_row = (float *) ((char *) ggml_get_data(td->dst) + i3*td->dst_nb3 + i2*td->dst_nb2 + i1*td->dst_nb1);
            
            ggml_vec_add_f32(td->ne0, dst_row, src0_row, src1_row);
            
            // Skip remaining elements in this row since we processed them all
            i += (td->ne0 - 1); // -1 because loop will increment i
        } else {
            // Fallback to element-wise operation for non-contiguous or single elements
            *dst_ptr = *src0_ptr + *src1_ptr;
        }
    }
    return NULL;
}
