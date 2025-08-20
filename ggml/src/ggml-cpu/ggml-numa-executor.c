/*
 * NUMA Executor - Strategy Engine and Work Orchestration
 * 
 * This component absorbs the old dispatcher logic with a cleaner architecture:
 * - Analyzes operations and selects optimal execution strategies  
 * - Delegates to specialized kernels in numa-kernels/ directory
 * - Handles work submission to coordinator and completion synchronization
 */

#include "ggml-numa-executor.h"
#include "ggml-numa-coordinator.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"  // For ggml_compute_forward_* function declarations
#include "ops.h"       // For actual ggml_compute_forward_* declarations

#ifdef __linux__
#include <numa.h>
#include <numaif.h>
#endif

#include <stdatomic.h>
#include <stdbool.h>

// Kernel headers - each operation has its own file
#include "numa-kernels/mul_mat.h"
#include "numa-kernels/add.h"
#include "numa-kernels/rms_norm.h"
#include "numa-kernels/soft_max.h"
#include "numa-kernels/rope.h"
#include "numa-kernels/cpy.h"
#include "numa-kernels/get_rows.h"

// Missing struct definition for MUL_MAT_ID work buffer calculation
struct mmid_row_mapping {
    int32_t i1; // i1
    int32_t i2; // i2  
};

// ============================================================================
// Operation Registry - Maps operations to their kernel implementations
// ============================================================================

typedef struct {
    enum ggml_op op_type;
    bool supported;
    float base_efficiency;          // Base parallel efficiency estimate
    size_t min_elements_for_numa;   // Minimum tensor size for NUMA benefit
    
    // Kernel interface functions
    bool (*supports_operation)(const struct ggml_tensor * tensor);
    enum ggml_status (*execute_operation)(struct ggml_tensor * tensor, struct ggml_cplan * cplan);
    float (*get_efficiency)(const struct ggml_tensor * tensor, size_t tensor_size);
} ggml_numa_kernel_registry_entry_t;

static const ggml_numa_kernel_registry_entry_t numa_kernel_registry[] = {
    // Matrix multiplication - complex but high payoff (NOT IMPLEMENTED YET)
    {
        .op_type = GGML_OP_MUL_MAT,
        .supported = false,  // Mark as unsupported until implementation is complete
        .base_efficiency = 0.85f,
        .min_elements_for_numa = 8192,
        .supports_operation = NULL,  // Will fall back to CPU
        .execute_operation = NULL,
        .get_efficiency = NULL
    },
    
    // Element-wise operations - simple and highly parallel
    {
        .op_type = GGML_OP_ADD,
        .supported = true,
        .base_efficiency = 0.95f,
        .min_elements_for_numa = 4096,
        .supports_operation = ggml_numa_kernel_add_supports,
        .execute_operation = ggml_numa_kernel_add_execute,
        .get_efficiency = ggml_numa_kernel_add_get_efficiency
    },
    
    // Normalization operations - moderate complexity (NOT IMPLEMENTED YET)
    {
        .op_type = GGML_OP_RMS_NORM,
        .supported = false,  // Mark as unsupported until implementation is complete
        .base_efficiency = 0.8f,
        .min_elements_for_numa = 2048,
        .supports_operation = NULL,
        .execute_operation = NULL,
        .get_efficiency = NULL
    },
    
    // Attention operations - complex synchronization (NOT IMPLEMENTED YET)
    {
        .op_type = GGML_OP_SOFT_MAX,
        .supported = false,  // Mark as unsupported until implementation is complete
        .base_efficiency = 0.7f,
        .min_elements_for_numa = 1024,
        .supports_operation = NULL,
        .execute_operation = NULL,
        .get_efficiency = NULL
    },
    
    // Position embedding operations (NOT IMPLEMENTED YET)
    {
        .op_type = GGML_OP_ROPE,
        .supported = false,  // Mark as unsupported until implementation is complete
        .base_efficiency = 0.9f,
        .min_elements_for_numa = 2048,
        .supports_operation = NULL,
        .execute_operation = NULL,
        .get_efficiency = NULL
    },
    
    // Memory operations - bandwidth limited but simple (NOT IMPLEMENTED YET)
    {
        .op_type = GGML_OP_CPY,
        .supported = false,  // Mark as unsupported until implementation is complete
        .base_efficiency = 0.95f,
        .min_elements_for_numa = 4096,
        .supports_operation = NULL,
        .execute_operation = NULL,
        .get_efficiency = NULL
    },
    
    // Row extraction operations (NOT IMPLEMENTED YET)
    {
        .op_type = GGML_OP_GET_ROWS,
        .supported = false,  // Mark as unsupported until implementation is complete
        .base_efficiency = 0.9f,
        .min_elements_for_numa = 1024,
        .supports_operation = NULL,
        .execute_operation = NULL,
        .get_efficiency = NULL
    },
    
    // Terminator
    { .op_type = GGML_OP_COUNT, .supported = false }
};

// ============================================================================
// Registry Lookup Functions
// ============================================================================

static const ggml_numa_kernel_registry_entry_t * find_kernel_entry(enum ggml_op op) {
    for (int i = 0; numa_kernel_registry[i].op_type != GGML_OP_COUNT; i++) {
        if (numa_kernel_registry[i].op_type == op) {
            return &numa_kernel_registry[i];
        }
    }
    return NULL;
}

// ============================================================================
// Public API Implementation
// ============================================================================

enum ggml_status ggml_numa_executor_compute_graph(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan) {
    if (!cgraph || !cplan) {
        GGML_LOG_ERROR("NUMA Executor: Invalid graph or compute plan\n");
        return GGML_STATUS_FAILED;
    }
    
    // Check if NUMA dispatch is enabled
    extern bool ggml_numa_should_dispatch(void);
    if (!ggml_numa_should_dispatch()) {
        GGML_LOG_DEBUG("NUMA Executor: NUMA dispatch disabled, using standard ggml computation\n");
        return GGML_STATUS_FAILED;  // Let standard ggml handle the computation
    }
    
    GGML_LOG_INFO("🎯 NUMA Executor: Processing graph with %d nodes (original plan: %d threads, fallback will use NUMA node 0 threads)\n", 
                  cgraph->n_nodes, cplan->n_threads);
    
    // Process each operation in the graph
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if (!node) continue;
        
        GGML_LOG_DEBUG("Executing node %d: %s\n", i, ggml_op_name(node->op));
        
        enum ggml_status result = ggml_numa_executor_execute_tensor(node, cplan);
        if (result != GGML_STATUS_SUCCESS) {
            GGML_LOG_ERROR("Operation %d (%s) failed with status %d\n", i, ggml_op_name(node->op), result);
            return result;
        }
    }
    
    GGML_LOG_INFO("✅ NUMA Executor: All %d operations completed successfully\n", cgraph->n_nodes);
    return GGML_STATUS_SUCCESS;
}

enum ggml_status ggml_numa_executor_execute_tensor(struct ggml_tensor * tensor, struct ggml_cplan * cplan) {
    if (!tensor || !cplan) {
        return GGML_STATUS_FAILED;
    }
    
    // Find the kernel for this operation
    const ggml_numa_kernel_registry_entry_t * kernel = find_kernel_entry(tensor->op);
    if (!kernel || !kernel->supported || !kernel->execute_operation) {
        GGML_LOG_DEBUG("NUMA Executor: Operation %s not implemented, falling back to standard CPU\n", ggml_op_name(tensor->op));
        return ggml_numa_executor_fallback_to_cpu(tensor, cplan);
    }
    
    // Check if the specific tensor variant is supported
    if (!kernel->supports_operation || !kernel->supports_operation(tensor)) {
        GGML_LOG_DEBUG("NUMA Executor: Operation %s not supported for this tensor config, falling back to standard CPU\n", 
                      ggml_op_name(tensor->op));
        return ggml_numa_executor_fallback_to_cpu(tensor, cplan);
    }
    
    // Calculate efficiency to decide if NUMA is worthwhile
    size_t tensor_size = ggml_nelements(tensor);
    float efficiency = kernel->get_efficiency(tensor, tensor_size);
    
    if (tensor_size < kernel->min_elements_for_numa || efficiency < 0.5f) {
        GGML_LOG_DEBUG("NUMA Executor: Using single-node execution for %s (size=%zu, efficiency=%.2f)\n",
                      ggml_op_name(tensor->op), tensor_size, efficiency);
    } else {
        GGML_LOG_DEBUG("NUMA Executor: Using NUMA execution for %s (size=%zu, efficiency=%.2f)\n",
                      ggml_op_name(tensor->op), tensor_size, efficiency);
    }
    
    // Execute using the kernel
    return kernel->execute_operation(tensor, cplan);
}

bool ggml_numa_executor_supports_op(enum ggml_op op) {
    const ggml_numa_kernel_registry_entry_t * kernel = find_kernel_entry(op);
    return kernel && kernel->supported;
}

float ggml_numa_executor_get_efficiency(enum ggml_op op, size_t tensor_size) {
    const ggml_numa_kernel_registry_entry_t * kernel = find_kernel_entry(op);
    if (!kernel || !kernel->supported) {
        return -1.0f;
    }
    
    // Return base efficiency scaled by tensor size appropriateness
    if (tensor_size >= kernel->min_elements_for_numa) {
        return kernel->base_efficiency;
    } else {
        // Reduced efficiency for small tensors
        return kernel->base_efficiency * 0.3f;
    }
}

// ============================================================================
// Fallback to Standard CPU Implementation
// ============================================================================

// Forward declaration of function to control fallback flag in ggml-cpu.c
extern void ggml_numa_set_fallback_flag(bool value);
extern enum ggml_status ggml_graph_compute_impl(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);

// Calculate work buffer size needed for a specific operation (from ggml-cpu.c logic)
static size_t ggml_numa_calculate_work_size(struct ggml_tensor * tensor, int n_threads) {
    size_t work_size = 0;
    
    GGML_LOG_DEBUG("NUMA Work Buffer: Calculating for operation %s\n", ggml_op_name(tensor->op));
    
    switch (tensor->op) {
        case GGML_OP_CPY:
        case GGML_OP_DUP:
            {
                if (ggml_is_quantized(tensor->type) ||
                    // F16 -> BF16 and BF16 -> F16 copies go through intermediate F32
                    (tensor->src[0]->type == GGML_TYPE_F16  && tensor->src[1] && tensor->src[1]->type == GGML_TYPE_BF16) ||
                    (tensor->src[0]->type == GGML_TYPE_BF16 && tensor->src[1] && tensor->src[1]->type == GGML_TYPE_F16)) {
                    work_size = ggml_type_size(GGML_TYPE_F32) * tensor->ne[0] * n_threads;
                }
                GGML_LOG_DEBUG("NUMA Work Buffer: CPY/DUP calculated size: %zu bytes\n", work_size);
            } break;
        case GGML_OP_MUL_MAT:
            {
                if (tensor->src[0] && tensor->src[1]) {
                    const struct ggml_type_traits_cpu * traits = ggml_get_type_traits_cpu(tensor->src[0]->type);
                    const enum ggml_type vec_dot_type = traits->vec_dot_type;
                    
                    if (tensor->src[1]->type != vec_dot_type) {
                        work_size = ggml_row_size(vec_dot_type, ggml_nelements(tensor->src[1]));
                    }
                    GGML_LOG_DEBUG("NUMA Work Buffer: MUL_MAT src0_type=%d, src1_type=%d, vec_dot_type=%d, calculated size: %zu bytes\n", 
                                   (int)tensor->src[0]->type, (int)tensor->src[1]->type, (int)vec_dot_type, work_size);
                } else {
                    GGML_LOG_WARN("NUMA Work Buffer: MUL_MAT missing source tensors\n");
                }
            } break;
        case GGML_OP_MUL_MAT_ID:
            {
                if (tensor->src[0] && tensor->src[1] && tensor->src[2]) {
                    const struct ggml_tensor * src0 = tensor->src[0];
                    const struct ggml_tensor * src1 = tensor->src[1];
                    const struct ggml_tensor * ids = tensor->src[2];
                    const struct ggml_type_traits_cpu * traits = ggml_get_type_traits_cpu(src0->type);
                    const enum ggml_type vec_dot_type = traits->vec_dot_type;
                    const int n_as = src0->ne[2];
                    
                    // src1
                    if (src1->type != vec_dot_type) {
                        work_size += ggml_row_size(vec_dot_type, ggml_nelements(src1)) + sizeof(int64_t);
                    }
                    // matrix_row_counts + matrix_rows + atomic_current_chunk
                    work_size += n_as * sizeof(int64_t) + sizeof(int64_t);
                    work_size += n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping) + sizeof(int64_t);
                    work_size += 64*n_as + 64; // CACHE_LINE_SIZE approximation
                }
            } break;
        case GGML_OP_OUT_PROD:
            {
                if (tensor->src[0] && ggml_is_quantized(tensor->src[0]->type)) {
                    work_size = ggml_type_size(GGML_TYPE_F32) * tensor->src[0]->ne[0] * n_threads;
                }
            } break;
        default:
            // For unknown operations, use a reasonable default
            work_size = 1024 * 1024; // 1MB safety buffer
            break;
    }
    
    return work_size;
}

enum ggml_status ggml_numa_executor_fallback_to_cpu(struct ggml_tensor * tensor, struct ggml_cplan * cplan) {
    if (!tensor || !cplan) {
        return GGML_STATUS_FAILED;
    }
    
    GGML_LOG_DEBUG("NUMA Fallback: Starting fallback for operation %s\n", ggml_op_name(tensor->op));
    
    // Set flag to disable NUMA dispatch during this call (prevents infinite recursion)
    ggml_numa_set_fallback_flag(true);
    
    // Calculate work buffer size needed for this specific operation
    size_t needed_work_size = ggml_numa_calculate_work_size(tensor, cplan->n_threads);
    
    void * work_data = NULL;
    bool allocated_work_buffer = false;
    
    // Check if existing work buffer is sufficient
    if (cplan->work_data && cplan->work_size >= needed_work_size) {
        work_data = cplan->work_data;
        GGML_LOG_DEBUG("NUMA Fallback: Using existing work buffer (%zu bytes >= %zu needed)\n", 
                       cplan->work_size, needed_work_size);
    } else if (needed_work_size > 0) {
        // Allocate NUMA-aware work buffer on node 0
        #ifdef __linux__
        if (numa_available() >= 0) {
            work_data = numa_alloc_onnode(needed_work_size, 0);
            if (work_data) {
                allocated_work_buffer = true;
                GGML_LOG_DEBUG("NUMA Fallback: Allocated %zu bytes work buffer on NUMA node 0\n", needed_work_size);
            }
        }
        #endif
        
        // Fall back to regular malloc if NUMA allocation failed
        if (!work_data) {
            work_data = malloc(needed_work_size);
            if (work_data) {
                allocated_work_buffer = true;
                GGML_LOG_DEBUG("NUMA Fallback: Allocated %zu bytes work buffer with malloc\n", needed_work_size);
            }
        }
        
        if (!work_data) {
            GGML_LOG_ERROR("NUMA Fallback: Failed to allocate %zu bytes work buffer\n", needed_work_size);
            ggml_numa_set_fallback_flag(false);
            return GGML_STATUS_FAILED;
        }
    }

    // Get the dedicated fallback threadpool from the coordinator
    struct ggml_numa_coordinator_manager * coordinator = ggml_numa_coordinator_manager_get_existing();
    struct ggml_threadpool * fallback_threadpool = NULL;
    int fallback_thread_count = 1;
    
    if (coordinator) {
        fallback_threadpool = ggml_numa_coordinator_get_fallback_threadpool(coordinator);
        fallback_thread_count = ggml_numa_coordinator_get_fallback_thread_count(coordinator);
        GGML_LOG_DEBUG("NUMA Fallback: Using dedicated fallback threadpool with %d thread(s)\n", fallback_thread_count);
    } else {
        GGML_LOG_DEBUG("NUMA Fallback: No coordinator available, using single-threaded execution\n");
    }
    
    // Set up compute params for fallback execution  
    struct ggml_compute_params params = {
        .ith = 0,
        .nth = fallback_thread_count,
        .wsize = needed_work_size,
        .wdata = work_data,
        .threadpool = fallback_threadpool  // Use dedicated fallback threadpool
    };
    
    GGML_LOG_INFO("🔧 NUMA Fallback: Executing operation %s (work_size=%zu, fallback_threads=%d, params.nth=%d, threadpool=%p)\n", 
                   ggml_op_name(tensor->op), needed_work_size, fallback_thread_count, params.nth, (void*)fallback_threadpool);
    
    // CRITICAL FIX: Use ggml_graph_compute instead of ggml_compute_forward to properly activate all worker threads
    // Create a temporary single-node graph for this operation
    struct ggml_cgraph temp_graph = {0};
    temp_graph.nodes = &tensor;
    temp_graph.n_nodes = 1;
    temp_graph.size = 1;
    
    // Create a temporary compute plan for the fallback threadpool
    struct ggml_cplan temp_cplan = {
        .n_threads = fallback_thread_count,
        .threadpool = fallback_threadpool,
        .work_size = needed_work_size,
        .work_data = work_data,
        .abort_callback = NULL,
        .abort_callback_data = NULL
    };
    
    // Execute using the internal implementation that bypasses NUMA dispatcher
    enum ggml_status result = ggml_graph_compute_impl(&temp_graph, &temp_cplan);
    
    // Clear flag after computation
    ggml_numa_set_fallback_flag(false);
    
    // Check result
    if (result != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("NUMA Fallback: Graph computation failed with status %d\n", result);
        // Clean up allocated work buffer
        if (allocated_work_buffer && work_data) {
            #ifdef __linux__
            if (numa_available() >= 0) {
                numa_free(work_data, needed_work_size);
                GGML_LOG_DEBUG("NUMA Fallback: Freed NUMA work buffer\n");
            } else
            #endif
            {
                free(work_data);
                GGML_LOG_DEBUG("NUMA Fallback: Freed malloc work buffer\n");
            }
        }
        return GGML_STATUS_FAILED;
    }
    
    // Clean up allocated work buffer
    if (allocated_work_buffer && work_data) {
        #ifdef __linux__
        if (numa_available() >= 0) {
            numa_free(work_data, needed_work_size);
            GGML_LOG_DEBUG("NUMA Fallback: Freed NUMA work buffer\n");
        } else
        #endif
        {
            free(work_data);
            GGML_LOG_DEBUG("NUMA Fallback: Freed malloc work buffer\n");
        }
    }
    
    GGML_LOG_DEBUG("NUMA Fallback: Operation %s completed successfully\n", ggml_op_name(tensor->op));
    return GGML_STATUS_SUCCESS;
}
