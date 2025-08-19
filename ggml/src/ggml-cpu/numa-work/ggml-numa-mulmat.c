#include "ggml-numa-mulmat.h"
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
#include <unistd.h>  // For usleep

#ifdef __linux__
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#endif

// Work function for chunking up a MUL_MAT operation
enum ggml_status ggml_numa_work_function_mul_mat_chunking(void * work_context, struct ggml_compute_params * params) {
    NUMA_ASSERT(work_context);
    NUMA_ASSERT(params);

    ggml_numa_dispatcher_work_context_t * ctx = (ggml_numa_dispatcher_work_context_t *)work_context;
    NUMA_ASSERT(ctx->operation);

    // Get dst tensor (we'll access src0 and src1 through it)
    struct ggml_tensor * dst = (struct ggml_tensor *)ctx->operation;
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    
    NUMA_THREAD_LOG_DEBUG_TENSOR(dst, "ENTERING chunking work function\n");

    NUMA_ASSERT(src0);
    NUMA_ASSERT(src1);

    // For the chunking function, we delegate to the chunk function which handles the actual work
    // The chunking function is typically called once per NUMA node, and then each call
    // to the chunk function handles the threading within that NUMA node
    
    NUMA_THREAD_LOG_DEBUG_TENSOR(dst, "Delegating to chunk function for NUMA-aware processing\n");
    
    // Simply call the chunk function which handles both NUMA and thread-level parallelism
    return ggml_numa_work_function_mul_mat_chunk(work_context, params);
}

// Work function for single-thread MUL_MAT operations
enum ggml_status ggml_numa_work_function_mul_mat_single(void * work_context, struct ggml_compute_params * params) {
    NUMA_ASSERT(work_context);
    NUMA_ASSERT(params);

    ggml_numa_dispatcher_work_context_t * ctx = (ggml_numa_dispatcher_work_context_t *)work_context;
    NUMA_ASSERT(ctx->operation);

    // Get dst tensor (we'll access src0 and src1 through it)
    struct ggml_tensor * dst = (struct ggml_tensor *)ctx->operation;
    
    NUMA_THREAD_LOG_DEBUG_TENSOR(dst, "ENTERING single-thread work function\n");
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    NUMA_ASSERT(src0);
    NUMA_ASSERT(src1);

    // Get matrix dimensions for work range calculation
    const int64_t ne01 = src0->ne[1];  // M dimension (src0 rows)
    const int64_t ne11 = src1->ne[1];  // N dimension (src1 columns)
    
    // For single-thread execution, process entire matrix
    const int64_t ir0_start = 0;
    const int64_t ir0_end = ne01;
    const int64_t ir1_start = 0;
    const int64_t ir1_end = ne11;

    // Prepare thread data for the kernel
    ggml_numa_mulmat_thread_data_t thread_data = {
        .dst = dst,
        .work_buffer = params->wdata,  // Use the work buffer provided by params
        .ir0_start = ir0_start,
        .ir0_end = ir0_end,
        .ir1_start = ir1_start,
        .ir1_end = ir1_end,
        .numa_node = ggml_numa_get_current_node(),
        .thread_id = params->ith
    };

    NUMA_THREAD_LOG_DEBUG_TENSOR(dst, "Single-thread processing full matrix: ir0=[%ld,%ld), ir1=[%ld,%ld)\n", 
                                ir0_start, ir0_end, ir1_start, ir1_end);

    // Call the mathematical kernel directly
    mul_mat_thread_kernel(&thread_data);

    NUMA_THREAD_LOG_DEBUG_TENSOR(dst, "Single-thread work function completed\n");
    return GGML_STATUS_SUCCESS;
}

// Work function for individual chunks of MUL_MAT
enum ggml_status ggml_numa_work_function_mul_mat_chunk(void * work_context, struct ggml_compute_params * params) {
    NUMA_ASSERT(work_context);
    NUMA_ASSERT(params);

    ggml_numa_dispatcher_work_context_t * ctx = (ggml_numa_dispatcher_work_context_t *)work_context;
    NUMA_ASSERT(ctx->operation);

    // Get dst tensor (we'll access src0 and src1 through it)
    struct ggml_tensor * dst = (struct ggml_tensor *)ctx->operation;
    
    NUMA_THREAD_LOG_DEBUG_TENSOR(dst, "ENTERING individual chunk work function\n");
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    NUMA_ASSERT(src0);
    NUMA_ASSERT(src1);

    // Get matrix dimensions
    const int64_t ne01 = src0->ne[1];  // M dimension (src0 rows)
    const int64_t ne11 = src1->ne[1];  // N dimension (src1 columns)
    
    // Get NUMA node and threading information for data slicing
    int numa_node = ggml_numa_get_current_node();
    if (numa_node < 0) numa_node = 0;  // Fallback
    
    extern struct ggml_numa_coordinator_manager * ggml_numa_coordinator_manager_get_global(int n_threads);
    extern int ggml_numa_coordinator_manager_get_numa_nodes(struct ggml_numa_coordinator_manager * mgr);
    
    struct ggml_numa_coordinator_manager * mgr = ggml_numa_coordinator_manager_get_global(8);
    int max_numa_nodes = mgr ? ggml_numa_coordinator_manager_get_numa_nodes(mgr) : 1;
    if (max_numa_nodes <= 0) max_numa_nodes = 1;

    // Calculate NUMA node's portion of work (data parallel across NUMA nodes)
    // For MUL_MAT, we can slice by output rows (ir1 dimension)
    const int64_t total_output_rows = ne11;
    const int64_t rows_per_numa_node = total_output_rows / max_numa_nodes;
    const int64_t numa_ir1_start = numa_node * rows_per_numa_node;
    const int64_t numa_ir1_end = (numa_node == max_numa_nodes - 1) ? total_output_rows : numa_ir1_start + rows_per_numa_node;

    // Calculate threading within NUMA node (multi-threading within each NUMA node)
    const int nth = params->nth;
    const int ith = params->ith;
    
    const int64_t numa_rows = numa_ir1_end - numa_ir1_start;
    const int64_t rows_per_thread = (numa_rows + nth - 1) / nth;  // Ceiling division
    const int64_t thread_ir1_start = numa_ir1_start + ith * rows_per_thread;
    const int64_t thread_ir1_end = (ith == nth - 1) ? numa_ir1_end : thread_ir1_start + rows_per_thread;

    // For src0 rows, process all of them (no slicing on ir0 dimension for now)
    const int64_t ir0_start = 0;
    const int64_t ir0_end = ne01;

    // Prepare thread data for the kernel
    ggml_numa_mulmat_thread_data_t thread_data = {
        .dst = dst,
        .work_buffer = params->wdata,  // Use the work buffer provided by params
        .ir0_start = ir0_start,
        .ir0_end = ir0_end,
        .ir1_start = thread_ir1_start,
        .ir1_end = thread_ir1_end,
        .numa_node = numa_node,
        .thread_id = ith
    };

    NUMA_THREAD_LOG_DEBUG_TENSOR(dst, "Chunk processing: NUMA[%d/%d], Thread[%d/%d], ir0=[%ld,%ld), ir1=[%ld,%ld)\n", 
                                numa_node, max_numa_nodes, ith, nth, ir0_start, ir0_end, thread_ir1_start, thread_ir1_end);

    // Call the mathematical kernel for this chunk
    mul_mat_thread_kernel(&thread_data);

    NUMA_THREAD_LOG_DEBUG_TENSOR(dst, "Chunk work function completed\n");
    return GGML_STATUS_SUCCESS;
}

// Kernel function for MUL_MAT mathematical computation
// Self-contained implementation based on ggml_compute_forward_mul_mat_one_chunk() logic
void* mul_mat_thread_kernel(void* data) {
    // Cast the input parameter to our thread data structure
    ggml_numa_mulmat_thread_data_t* thread_data = (ggml_numa_mulmat_thread_data_t*)data;
    NUMA_ASSERT(thread_data);
    
    // Extract tensor pointers
    struct ggml_tensor* dst = thread_data->dst;
    const struct ggml_tensor* src0 = dst->src[0];
    const struct ggml_tensor* src1 = dst->src[1];
    
    NUMA_ASSERT(src0);
    NUMA_ASSERT(src1);
    NUMA_ASSERT(dst);
    
    NUMA_THREAD_LOG_DEBUG_TENSOR(dst, "ENTERING mathematical kernel\n");
    
    // Get tensor dimensions and strides using GGML pattern
    // GGML_TENSOR_BINARY_OP_LOCALS equivalent
    const int64_t ne00 = src0->ne[0]; const int64_t ne01 = src0->ne[1]; const int64_t ne02 = src0->ne[2]; const int64_t ne03 = src0->ne[3];
    const size_t  nb00 = src0->nb[0]; const size_t  nb01 = src0->nb[1]; const size_t  nb02 = src0->nb[2]; const size_t  nb03 = src0->nb[3];
    const int64_t ne10 = src1->ne[0]; const int64_t ne11 = src1->ne[1]; const int64_t ne12 = src1->ne[2]; const int64_t ne13 = src1->ne[3];
    const size_t  nb10 = src1->nb[0]; const size_t  nb11 = src1->nb[1]; const size_t  nb12 = src1->nb[2]; const size_t  nb13 = src1->nb[3];
    const int64_t ne0  = dst->ne[0];  const int64_t ne1  = dst->ne[1];  const int64_t ne2  = dst->ne[2];  const int64_t ne3  = dst->ne[3];
    const size_t  nb0  = dst->nb[0];  const size_t  nb1  = dst->nb[1];  const size_t  nb2  = dst->nb[2];  const size_t  nb3  = dst->nb[3];
    
    // Get type traits for the source tensor
    const struct ggml_type_traits_cpu * traits = ggml_get_type_traits_cpu(src0->type);
    ggml_vec_dot_t const vec_dot = traits->vec_dot;
    enum ggml_type const vec_dot_type = traits->vec_dot_type;
    int64_t const vec_dot_num_rows = traits->nrows;
    
    // Original ggml asserts src1 is f32 when type conversion is needed
    // This is because src0 is weights (can be quantized), src1 is activations (f32)
    NUMA_ASSERT(src1->type == GGML_TYPE_F32);
    NUMA_ASSERT(dst->type == GGML_TYPE_F32);
    
    // Check tensor compatibility (same as original ggml logic)
    NUMA_ASSERT(ne0 == ne01);
    NUMA_ASSERT(ne1 == ne11);
    NUMA_ASSERT(ne2 == ne12);
    NUMA_ASSERT(ne3 == ne13);
    
    // we don't support permuted src0 or src1
    NUMA_ASSERT(nb00 == ggml_type_size(src0->type));
    NUMA_ASSERT(nb10 == ggml_type_size(src1->type));
    
    // dst cannot be transposed or permuted
    NUMA_ASSERT(nb0 == sizeof(float));
    NUMA_ASSERT(nb0 <= nb1);
    NUMA_ASSERT(nb1 <= nb2);
    NUMA_ASSERT(nb2 <= nb3);
    
    // Broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;
    
    // Check if src1 is contiguous
    const bool src1_cont = ggml_is_contiguous(src1);
    
    // Determine data source for src1 (original or converted)
    const void * wdata = (src1->type == vec_dot_type) ? tensor_data(src1) : thread_data->work_buffer;
    const size_t row_size = ggml_row_size(vec_dot_type, ne10);
    
    // If type conversion is needed and we don't have a work buffer, we can't proceed
    if (src1->type != vec_dot_type && !thread_data->work_buffer) {
        NUMA_THREAD_LOG_DEBUG_TENSOR(dst, "ERROR: Type conversion needed but no work buffer provided\n");
        return NULL;
    }
    
    // Perform type conversion if needed (same logic as ggml_compute_forward_mul_mat)
    if (src1->type != vec_dot_type && thread_data->work_buffer) {
        NUMA_THREAD_LOG_DEBUG_TENSOR(dst, "Performing type conversion from %s to %s\n", ggml_type_name(src1->type), ggml_type_name(vec_dot_type));
        
        const struct ggml_type_traits_cpu * src1_traits = ggml_get_type_traits_cpu(vec_dot_type);
        ggml_from_float_t const from_float = src1_traits->from_float;
        
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1 * ne11;
        const size_t nbw3 = nbw2 * ne12;
        
        // Convert src1 data to vec_dot_type in work buffer
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float((float *)((char *) tensor_data(src1) + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)((char *)thread_data->work_buffer + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
    }
    
    // Calculate the ranges this thread should process
    const int64_t ir0_start = thread_data->ir0_start;
    const int64_t ir0_end = thread_data->ir0_end;
    const int64_t ir1_start = thread_data->ir1_start;
    const int64_t ir1_end = thread_data->ir1_end;
    
    // If this thread has no work, return early
    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        NUMA_THREAD_LOG_DEBUG_TENSOR(dst, "No work for this thread: ir0=[%ld,%ld), ir1=[%ld,%ld)\n", 
                                    ir0_start, ir0_end, ir1_start, ir1_end);
        return NULL;
    }
    
    NUMA_THREAD_LOG_DEBUG_TENSOR(dst, "Processing ranges: ir0=[%ld,%ld), ir1=[%ld,%ld)\n", 
                                ir0_start, ir0_end, ir1_start, ir1_end);
    
    // Block tiling for cache efficiency (same as original)
    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;
    
    // Calculate stride for src1 columns
    const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;
    
    // Determine number of rows per vec_dot operation
    int64_t num_rows_per_vec_dot = vec_dot_num_rows;
    
    // Check constraints for multi-row operations (same logic as original)
    const int64_t nr0 = ir0_end - ir0_start;
    const int64_t nr1 = ir1_end - ir1_start;
    if ((nr0 % 2 != 0) || (ne11 % 2 != 0) || (nr0 % 2 != 0) || (nr1 % 2 != 0)) {
        num_rows_per_vec_dot = 1;
    }
    
    NUMA_THREAD_LOG_DEBUG_TENSOR(dst, "Using %ld rows per vec_dot operation\n", num_rows_per_vec_dot);
    
    // Main computation loop - block tiled matrix multiplication
    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
                // Calculate 3D indices for src1/dst (same pattern as original)
                const int64_t i13 = (ir1 / (ne12 * ne1));
                const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // Broadcast src0 into src1 (apply broadcast factors)
                const int64_t i03 = i13 / r3;
                const int64_t i02 = i12 / r2;

                const int64_t i1 = i11;
                const int64_t i2 = i12;
                const int64_t i3 = i13;

                // Get src0 row pointer with broadcasting
                const char * src0_row = (const char*)tensor_data(src0) + (0 + i02 * nb02 + i03 * nb03);

                // Get src1 column pointer - handle both contiguous and non-contiguous cases
                const char * src1_col = (const char*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                
                // Get destination column pointer
                float * dst_col = (float*)((char*)tensor_data(dst) + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                // Inner loop over src0 rows within the block
                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                    if (num_rows_per_vec_dot == 1) {
                        // Single row case - direct vec_dot
                        vec_dot(ne00, &dst_col[ir0], 0, src0_row + ir0*nb01, 0, src1_col, 0, 1);
                    } else {
                        // Multi-row case - process multiple rows in one vec_dot call
                        for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
                            float * dst_ptr = &dst_col[ir0 + cn * nb1 / nb0];
                            const char * src0_ptr = src0_row + (ir0 + cn) * nb01;
                            const char * src1_ptr = src1_col + cn * src1_col_stride;
                            
                            vec_dot(ne00, dst_ptr, 0, src0_ptr, 0, src1_ptr, 0, 1);
                        }
                    }
                }
            }
        }
    }
    
    NUMA_THREAD_LOG_DEBUG_TENSOR(dst, "Thread kernel completed successfully\n");
    return NULL;
}

//
// MUL_MAT-Specific Strategy Analysis and Dispatch Functions
//

// Calculate work buffer size required specifically for MUL_MAT operations
size_t ggml_numa_mul_mat_calculate_work_buffer_size(const struct ggml_tensor * operation) {
    if (!operation || operation->op != GGML_OP_MUL_MAT) {
        return 0;
    }
    
    const struct ggml_tensor * src0 = operation->src[0];
    const struct ggml_tensor * src1 = operation->src[1];
    
    if (!src0 || !src1) {
        return 0;
    }
    
    // Get the vec_dot_type the same way as ggml_compute_forward_mul_mat
    const struct ggml_type_traits_cpu * traits = ggml_get_type_traits_cpu(src0->type);
    enum ggml_type vec_dot_type = traits->vec_dot_type;
    
    // Only need work buffer if type conversion is required
    if (src1->type != vec_dot_type) {
        // Use the same calculation as ggml_compute_forward_mul_mat
        const size_t nbw1 = ggml_row_size(vec_dot_type, src1->ne[0]);  // ne10
        const size_t nbw2 = nbw1 * src1->ne[1];                       // nbw1*ne11
        const size_t nbw3 = nbw2 * src1->ne[2];                       // nbw2*ne12
        size_t work_size = src1->ne[3] * nbw3;                        // ne13*nbw3
        
        GGML_LOG_DEBUG("MUL_MAT work buffer calculation: vec_dot_type=%d, src1_type=%d, work_size=%zu\n",
                       vec_dot_type, src1->type, work_size);
        return work_size;
    }
    
    return 0; // No work buffer needed
}

// Analyze MUL_MAT operation to determine optimal execution strategy
enum ggml_status ggml_numa_mul_mat_analyze_strategy(
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context,
    ggml_numa_execution_strategy_t * strategy,
    ggml_numa_work_function_t * work_function,
    size_t * work_buffer_size) {
    
    NUMA_ASSERT(operation);
    NUMA_ASSERT(context);
    NUMA_ASSERT(strategy);
    NUMA_ASSERT(work_function);
    NUMA_ASSERT(work_buffer_size);
    
    NUMA_ASSERT(operation->op == GGML_OP_MUL_MAT);
    
    const struct ggml_tensor * src0 = operation->src[0];
    const struct ggml_tensor * src1 = operation->src[1];
    
    NUMA_ASSERT(src0);
    NUMA_ASSERT(src1);
    
    // Get matrix dimensions
    const int64_t ne01 = src0->ne[1];  // M dimension  
    const int64_t ne00 = src0->ne[0];  // K dimension
    const int64_t ne11 = src1->ne[1];  // N dimension
    
    // Calculate computational complexity
    const int64_t complexity = ne01 * ne00 * ne11;
    
    // Calculate work buffer requirements
    *work_buffer_size = ggml_numa_mul_mat_calculate_work_buffer_size(operation);
    
    // Strategy decision based on complexity and available NUMA nodes
    const int64_t COMPLEXITY_THRESHOLD = 10000000; // 10M operations
    const bool use_chunked = (context->numa_nodes > 1) && (complexity > COMPLEXITY_THRESHOLD);
    
    if (use_chunked) {
        NUMA_THREAD_LOG_DEBUG_TENSOR(operation, "strategy: chunked execution (complexity=%ld, numa_nodes=%d)\n",
                                    complexity, context->numa_nodes);
        *work_function = ggml_numa_work_function_mul_mat_chunking;
        strategy->node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL;
        strategy->on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD;
    } else {
        NUMA_THREAD_LOG_DEBUG_TENSOR(operation, "strategy: single execution (complexity=%ld)\n",
                                    complexity);
        *work_function = ggml_numa_work_function_mul_mat_single;
        strategy->node_strategy = NUMA_NODE_STRATEGY_SINGLE;
        strategy->on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD;
    }

    NUMA_THREAD_LOG_DEBUG_TENSOR(operation, "analysis: dimensions=%ldx%ld * %ldx%ld, complexity=%ld, work_buffer=%zu\n",
                                ne01, ne00, ne11, src1->ne[0], complexity, *work_buffer_size);

    return GGML_STATUS_SUCCESS;
}

// Main MUL_MAT dispatch function - handles all MUL_MAT NUMA coordination
enum ggml_status ggml_numa_mul_mat_dispatch(
    struct ggml_numa_coordinator_manager * manager,
    const struct ggml_tensor * operation,
    const ggml_numa_work_context_t * context) {

    NUMA_THREAD_LOG_DEBUG_TENSOR(operation, "dispatch: starting\n");
    
    NUMA_ASSERT(manager);
    NUMA_ASSERT(operation);
    NUMA_ASSERT(context);
    
    NUMA_ASSERT(operation->op == GGML_OP_MUL_MAT);
    
    NUMA_THREAD_LOG_DEBUG_TENSOR(operation, "dispatch: analyzing strategy for operation\n");
    
    // Analyze operation to determine optimal strategy
    ggml_numa_execution_strategy_t strategy;
    ggml_numa_work_function_t work_function;
    size_t work_buffer_size;
    
    enum ggml_status analysis_result = ggml_numa_mul_mat_analyze_strategy(
        operation, context, &strategy, &work_function, &work_buffer_size);
    
    NUMA_ASSERT(analysis_result == GGML_STATUS_SUCCESS);
    
    // Create work context for function pointer execution
    ggml_numa_dispatcher_work_context_t * work_context = ggml_numa_dispatcher_create_work_context(
        (struct ggml_tensor *)operation,
        "MUL_MAT",
        NULL,  // No additional context
        0      // No additional context size
    );

    NUMA_ASSERT(work_context);

    // Submit work using selected strategy and function
    int work_id = ggml_numa_coordinator_manager_submit_work_function(
        manager,
        work_function,                     // Strategy-selected function
        work_context,                      // Context data
        -1,                                // Auto-select NUMA node
        strategy,                          // Analyzed execution strategy
        work_buffer_size                   // Calculated buffer requirements
    );
    
    if (work_id < 0) {
        NUMA_THREAD_LOG_DEBUG_TENSOR(operation, "dispatch: failed to submit work to coordinator\n");
        ggml_numa_dispatcher_free_work_context(work_context);
        NUMA_ASSERT(work_id < 0);
    }

    NUMA_THREAD_LOG_DEBUG_TENSOR(operation, "dispatch: submitted work (ID: %d) to coordinator\n", work_id);
    
    // Wait for work completion
    NUMA_THREAD_LOG_DEBUG_TENSOR(operation, "dispatch: waiting for work completion\n");
    enum ggml_status wait_status = GGML_STATUS_FAILED;
    wait_status = ggml_numa_coordinator_manager_wait_for_completion(manager);
    NUMA_ASSERT(wait_status == GGML_STATUS_SUCCESS);
        
    // Memory barrier to ensure all coordinator work is visible
    __sync_synchronize();

    NUMA_THREAD_LOG_DEBUG_TENSOR(operation, "dispatch: completed successfully\n");
    
    return GGML_STATUS_SUCCESS;
}

// MUL_MAT operation handler
const ggml_numa_operation_handler_t ggml_numa_handler_mul_mat_enhanced = {
    .operation_type = GGML_OP_MUL_MAT,
    .default_strategy = {
        .node_strategy = NUMA_NODE_STRATEGY_SINGLE,
        .on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD
    },
    .complexity = NUMA_OP_COMPLEXITY_MODERATE,
    .workload_type = NUMA_OP_COMPUTE_BOUND,
    .min_elements_for_parallel = 100000,
    .optimal_chunk_size = 4 * 1024 * 1024,  // 4MB chunks
    .parallel_efficiency_estimate = 0.85f,
    .requires_synchronization = false,
    .supports_in_place = false,
    .analyze = NULL  // We use our own dispatch function instead
};