
#include "ggml-numa-rms-norm.h"
#include "ggml-numa-operation-dispatch.h"
#include "ggml-numa-coordinator.h"
#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "ggml.h"
#include "vec.h"

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

// RMS_NORM chunk work function - handles row-wise normalization with data parallel execution
// This implementation extracts the mathematical kernel to avoid threading conflicts
enum ggml_status ggml_numa_work_function_rms_norm_chunk(void * work_context, struct ggml_compute_params * params) {
    if (!work_context || !params) {
        GGML_LOG_ERROR("RMS_NORM work function: Invalid parameters\n");
        return GGML_STATUS_FAILED;
    }
    
    ggml_numa_dispatcher_work_context_t * ctx = (ggml_numa_dispatcher_work_context_t *)work_context;
    
    if (!ctx->operation) {
        GGML_LOG_ERROR("RMS_NORM work function: Operation is NULL\n");
        return GGML_STATUS_FAILED;
    }
    
    GGML_LOG_DEBUG("RMS_NORM work function: operation=%p, type=%d, elements=%ld\n", 
                   (void*)ctx->operation, ctx->operation->op, ggml_nelements(ctx->operation));
    
    // Check source tensor - RMS_NORM has one input tensor
    const struct ggml_tensor * src0 = ctx->operation->src[0];
    
    if (!src0) {
        GGML_LOG_ERROR("RMS_NORM work function: Missing source tensor\n");
        return GGML_STATUS_FAILED;
    }
    
    // Validate tensor shapes and types
    if (!ggml_are_same_shape(src0, ctx->operation)) {
        GGML_LOG_ERROR("RMS_NORM work function: Source and destination shapes don't match\n");
        return GGML_STATUS_FAILED;
    }
    
    if (src0->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("RMS_NORM work function: Only F32 tensors supported\n");
        return GGML_STATUS_FAILED;
    }
    
    if (src0->nb[0] != sizeof(float)) {
        GGML_LOG_ERROR("RMS_NORM work function: Invalid tensor stride\n");
        return GGML_STATUS_FAILED;
    }
    
    // Check if we have threading parameters for multi-level parallelism
    extern enum ggml_status ggml_numa_work_function_rms_norm_chunk(void * work_context, struct ggml_compute_params * params);
    
    // This function can be called either:
    // 1. From coordinator without params (legacy single-threaded per NUMA node)
    // 2. From coordinator with params (new multi-threaded per NUMA node)
    // 3. Recursively from threadpool with params (thread subdivision)
    
    // Extract tensor dimensions (using GGML_TENSOR_UNARY_OP_LOCALS pattern)
    const int64_t ne00 = src0->ne[0];  // Elements per row
    const int64_t ne01 = src0->ne[1];  // Number of rows  
    const int64_t ne02 = src0->ne[2];  // Batch dimension 2
    const int64_t ne03 = src0->ne[3];  // Batch dimension 3
    
    const size_t nb01 = src0->nb[1];   // Row stride in bytes
    const size_t nb02 = src0->nb[2];   // Batch stride 2 in bytes
    const size_t nb03 = src0->nb[3];   // Batch stride 3 in bytes
    
    const size_t nb1 = ctx->operation->nb[1];   // Destination row stride in bytes
    const size_t nb2 = ctx->operation->nb[2];   // Destination batch stride 2 in bytes
    const size_t nb3 = ctx->operation->nb[3];   // Destination batch stride 3 in bytes
    
    // Get epsilon parameter from operation parameters
    float eps;
    memcpy(&eps, ctx->operation->op_params, sizeof(float));
    if (eps < 0.0f) {
        GGML_LOG_ERROR("RMS_NORM work function: Invalid epsilon value: %f\n", eps);
        return GGML_STATUS_FAILED;
    }

    GGML_LOG_DEBUG("RMS_NORM work function: Processing tensor [%ld, %ld, %ld, %ld] with eps=%f\n", 
                   ne00, ne01, ne02, ne03, eps);

    // NUMA + Thread Multi-Level Parallel RMS_NORM Implementation
    // Level 1: NUMA-level parallelism (different row ranges per NUMA node)
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

    // Calculate NUMA-level row distribution
    // In single-node execution, only node 0 gets work and processes entire tensor
    // In data-parallel execution, work is distributed across multiple nodes
    int64_t total_rows = ne01;
    int64_t numa_start_row, numa_end_row;
    
    // Simple heuristic: if only node 0 is being used, process entire tensor
    // Otherwise, use data-parallel distribution
    if (numa_node == 0 && max_numa_nodes > 1) {
        // For single-node strategy: process entire tensor on node 0
        // For data-parallel strategy: process node 0's slice 
        // We'll assume single-node if we're seeing consecutive calls to only node 0
        numa_start_row = 0;
        numa_end_row = total_rows;  // Process entire tensor for now
        GGML_LOG_DEBUG("RMS_NORM: NUMA node 0 processing ALL rows 0 to %ld (%ld total rows)", 
                       numa_end_row - 1, total_rows);
    } else {
        // Data-parallel mode: slice across nodes
        int64_t rows_per_node = total_rows / max_numa_nodes;
        int64_t remainder_rows = total_rows % max_numa_nodes;
        
        numa_start_row = numa_node * rows_per_node;
        numa_end_row = numa_start_row + rows_per_node;
        
        if (numa_node < remainder_rows) {
            numa_start_row += numa_node;
            numa_end_row += numa_node + 1;
        } else {
            numa_start_row += remainder_rows;
            numa_end_row += remainder_rows;
        }
        
        GGML_LOG_DEBUG("RMS_NORM: NUMA node %d processing rows %ld to %ld (%ld of %ld total)", 
                       numa_node, numa_start_row, numa_end_row - 1, numa_end_row - numa_start_row, total_rows);
    }
    
    // Ensure we don't exceed tensor bounds and have valid ranges
    if (numa_end_row > total_rows) {
        numa_end_row = total_rows;
    }
    if (numa_start_row < 0) {
        numa_start_row = 0;
    }
    if (numa_end_row <= numa_start_row) {
        numa_end_row = numa_start_row + 1;  // Ensure at least one row
    }

    // Thread-level parallelism within this NUMA node's assigned rows
    int numa_node_rows = numa_end_row - numa_start_row;
    
    GGML_LOG_DEBUG("RMS_NORM NUMA node %d (of %d): assigned rows %ld to %ld (%ld rows total)\n", 
                   numa_node, max_numa_nodes, numa_start_row, numa_end_row - 1, numa_node_rows);

    // Use existing threadpool architecture if available, otherwise fall back to single-threaded
    if (params && params->nth > 1 && params->threadpool) {
        GGML_LOG_DEBUG("RMS_NORM: Using multi-threaded execution (%d threads) on NUMA node %d (CPU: %d)\n", 
                       params->nth, numa_node, sched_getcpu());

        // Create thread data for each thread
        struct rms_norm_thread_data * thread_data = (struct rms_norm_thread_data *)calloc(params->nth, sizeof(struct rms_norm_thread_data));
        if (!thread_data) {
            GGML_LOG_ERROR("RMS_NORM: Failed to allocate thread data\n");
            return GGML_STATUS_FAILED;
        }

        // Distribute NUMA node's rows among threads
        int64_t rows_per_thread = numa_node_rows / params->nth;
        int64_t remainder_thread_rows = numa_node_rows % params->nth;

        for (int t = 0; t < params->nth; t++) {
            struct rms_norm_thread_data * td = &thread_data[t];
            
            // Copy common data
            td->src0 = src0;
            td->dst = ctx->operation;
            td->ne00 = ne00; td->ne01 = ne01; td->ne02 = ne02; td->ne03 = ne03;
            td->nb01 = nb01; td->nb02 = nb02; td->nb03 = nb03;
            td->nb1 = nb1; td->nb2 = nb2; td->nb3 = nb3;
            td->eps = eps;
            td->thread_id = t;
            td->numa_node = numa_node;
            
            // Calculate thread-specific row range within NUMA node's rows
            int64_t thread_row_start = t * rows_per_thread;
            int64_t thread_row_end = thread_row_start + rows_per_thread;
            
            // Distribute remainder rows among first few threads
            if (t < remainder_thread_rows) {
                thread_row_start += t;
                thread_row_end += t + 1;
            } else {
                thread_row_start += remainder_thread_rows;
                thread_row_end += remainder_thread_rows;
            }
            
            // Convert to absolute row indices
            td->thread_start_row = numa_start_row + thread_row_start;
            td->thread_end_row = numa_start_row + thread_row_end;
            
            // Bounds checking
            if (td->thread_end_row > numa_end_row) {
                td->thread_end_row = numa_end_row;
            }
            if (td->thread_start_row >= td->thread_end_row) {
                td->thread_end_row = td->thread_start_row + 1; // Ensure at least one row
            }
            
            GGML_LOG_DEBUG("RMS_NORM thread %d on NUMA %d: processing rows %ld to %ld\n", 
                           t, numa_node, td->thread_start_row, td->thread_end_row - 1);
        }

        // Execute using pthread directly for multi-threading within NUMA node
        pthread_t* threads = (pthread_t*)malloc(params->nth * sizeof(pthread_t));
        if (!threads) {
            free(thread_data);
            GGML_LOG_ERROR("RMS_NORM: Failed to allocate thread handles\n");
            return GGML_STATUS_FAILED;
        }
        
        // Create threads
        for (int t = 0; t < params->nth; t++) {
            int ret = pthread_create(&threads[t], NULL, rms_norm_thread_kernel, &thread_data[t]);
            if (ret != 0) {
                GGML_LOG_ERROR("RMS_NORM: Failed to create thread %d: %d\n", t, ret);
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
        
        GGML_LOG_DEBUG("RMS_NORM: Multi-threaded execution completed on NUMA node %d\n", numa_node);
        
    } else {
        // Single-threaded execution within this NUMA node
        GGML_LOG_DEBUG("RMS_NORM: Using single-threaded execution on NUMA node %d (CPU: %d)\n", 
                       numa_node, sched_getcpu());
        
        // Get raw tensor data pointers for validation
        void * src_raw_data = ggml_get_data(src0);
        void * dst_raw_data = ggml_get_data(ctx->operation);
        
        GGML_LOG_DEBUG("RMS_NORM: Data pointers - src=%p, dst=%p\n", src_raw_data, dst_raw_data);
        
        // Comprehensive bounds checking with assertions
        NUMA_ASSERT(numa_start_row >= 0);
        NUMA_ASSERT(numa_end_row > numa_start_row);
        NUMA_ASSERT(numa_end_row <= ne01);
        NUMA_ASSERT(ne00 > 0);
        NUMA_ASSERT(ne01 > 0);
        NUMA_ASSERT(ne02 > 0);
        NUMA_ASSERT(ne03 > 0);
        NUMA_ASSERT(src_raw_data != NULL);
        NUMA_ASSERT(dst_raw_data != NULL);
        
        // Validate tensor strides are sane
        NUMA_ASSERT(nb01 >= ne00 * sizeof(float));
        NUMA_ASSERT(nb02 >= ne01 * nb01);
        NUMA_ASSERT(nb03 >= ne02 * nb02);
        NUMA_ASSERT(nb1 >= ne00 * sizeof(float));
        NUMA_ASSERT(nb2 >= ne01 * nb1);
        NUMA_ASSERT(nb3 >= ne02 * nb2);
        
        // Validate epsilon parameter
        NUMA_ASSERT(eps > 0.0f);
        NUMA_ASSERT(eps < 1.0f);  // Reasonable epsilon range
        NUMA_ASSERT(isfinite(eps));
        
        GGML_LOG_DEBUG("RMS_NORM: Boundary checks passed - processing dimensions [%ld,%ld,%ld,%ld], rows %ld to %ld\n",
                       ne00, ne01, ne02, ne03, numa_start_row, numa_end_row - 1);
                       
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                for (int64_t i01 = numa_start_row; i01 < numa_end_row; i01++) {
                    // Calculate memory addresses for current row
                    const float * x = (float *) ((char *) src_raw_data + i01*nb01 + i02*nb02 + i03*nb03);
                    float * y = (float *) ((char *) dst_raw_data + i01*nb1 + i02*nb2 + i03*nb3);
                    
                    // Comprehensive memory bounds validation
                    NUMA_ASSERT(x != NULL);
                    NUMA_ASSERT(y != NULL);
                    
                    // Validate memory offsets are within tensor boundaries
                    ptrdiff_t src_offset = i01*nb01 + i02*nb02 + i03*nb03;
                    ptrdiff_t dst_offset = i01*nb1 + i02*nb2 + i03*nb3;
                    ptrdiff_t src_end_offset = src_offset + (ne00-1) * sizeof(float);
                    ptrdiff_t dst_end_offset = dst_offset + (ne00-1) * sizeof(float);
                    
                    NUMA_ASSERT(src_offset >= 0);
                    NUMA_ASSERT(dst_offset >= 0);
                    NUMA_ASSERT(src_end_offset < (ptrdiff_t)(ggml_nbytes(src0)));
                    NUMA_ASSERT(dst_end_offset < (ptrdiff_t)(ggml_nbytes(ctx->operation)));
                    
                    // Validate the row data is finite before processing  
                    for (int64_t i = 0; i < ne00; i++) {
                        NUMA_ASSERT(isfinite(x[i]), "RMS_NORM: Found NaN/inf in row data at index %d: %f", (int)i, (double)x[i]);
                    }
                    
                    // Debug first few elements of first row processed by this NUMA node
                    if (i01 == numa_start_row && i02 == 0 && i03 == 0) {
                        GGML_LOG_DEBUG("RMS_NORM NUMA %d: First row input data [0-4]: %f %f %f %f %f\n", 
                                       numa_node, x[0], x[1], x[2], x[3], x[4]);
                    }
                    
                    // RMS Normalization SIMD-optimized mathematical kernel:
                    // 1. Calculate sum of squares for the row using vector dot product
                    float sum = 0.0;
                    ggml_vec_dot_f32(ne00, &sum, 0, x, 0, x, 0, 1);
                    
                    // Validate sum is finite and reasonable
                    NUMA_ASSERT(isfinite(sum) && sum >= 0.0f);
                    
                    // 2. Calculate mean of squares
                    const float mean = sum / ne00;
                    NUMA_ASSERT(isfinite(mean) && mean >= 0.0f);
                    
                    // 3. Copy input to output and apply RMS normalization scaling
                    const float scale = 1.0f / sqrtf(mean + eps);
                    
                    // Validate scale factor
                    NUMA_ASSERT(isfinite(scale) && scale > 0.0f);
                    
                    // Debug the first row calculations for this NUMA node
                    if (i01 == numa_start_row && i02 == 0 && i03 == 0) {
                        GGML_LOG_DEBUG("RMS_NORM NUMA %d: First row - sum=%f, mean=%f, scale=%f\n", 
                                       numa_node, sum, mean, scale);
                    }
                    
                    // 4. Copy and scale using SIMD operations
                    ggml_vec_cpy_f32(ne00, y, x);
                    ggml_vec_scale_f32(ne00, y, scale);
                    
                    // Debug output for first row
                    if (i01 == numa_start_row && i02 == 0 && i03 == 0) {
                        GGML_LOG_DEBUG("RMS_NORM NUMA %d: First row output [0-4]: %f %f %f %f %f\n", 
                                       numa_node, y[0], y[1], y[2], y[3], y[4]);
                    }
                }
            }
        }
        
        GGML_LOG_DEBUG("RMS_NORM: Single-threaded processing completed for NUMA node %d\n", numa_node);
    }
    
    // Memory barrier to ensure all writes are visible before returning
    __sync_synchronize();
    
    GGML_LOG_DEBUG("Successfully executed RMS_NORM chunk work function\n");
    
    return GGML_STATUS_SUCCESS;
}


void* rms_norm_thread_kernel(void* data) {
    struct rms_norm_thread_data * td = (struct rms_norm_thread_data *)data;
    
    // Process rows across all batch dimensions in this thread's assigned range
    for (int64_t i03 = 0; i03 < td->ne03; i03++) {
        for (int64_t i02 = 0; i02 < td->ne02; i02++) {
            for (int64_t i01 = td->thread_start_row; i01 < td->thread_end_row; i01++) {
                // Calculate memory addresses for current row across all batch dimensions
                const float * src_row = (float *) ((char *) ggml_get_data(td->src0) + i01*td->nb01 + i02*td->nb02 + i03*td->nb03);
                float * dst_row = (float *) ((char *) ggml_get_data(td->dst) + i01*td->nb1 + i02*td->nb2 + i03*td->nb3);
                
                // RMS normalization SIMD-optimized: sum of squares using vector dot product
                float sum = 0.0f;
                ggml_vec_dot_f32(td->ne00, &sum, 0, src_row, 0, src_row, 0, 1);
                
                const float mean = sum / td->ne00;
                const float scale = 1.0f / sqrtf(mean + td->eps);
                
                // Check for numerical issues
                if (!isfinite(scale) || scale <= 0.0f) {
                    GGML_LOG_ERROR("RMS_NORM thread %d: numerical instability detected (scale=%f, mean=%f, eps=%f)\n", 
                                   td->thread_id, scale, mean, td->eps);
                    continue; // Skip this row
                }
                
                // Copy source to destination, then apply SIMD scaling
                ggml_vec_cpy_f32(td->ne00, dst_row, src_row);
                ggml_vec_scale_f32(td->ne00, dst_row, scale);
            }
        }
    }
    return NULL;
}