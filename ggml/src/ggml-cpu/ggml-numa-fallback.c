#include "ggml-numa-fallback.h"
#include "ggml-cpu-impl.h"
#include "ggml-cpu.h"    // For ggml_cplan structure definition
#include "ops.h"         // For operation functions
#include "unary-ops.h"
#include "binary-ops.h"
#include "ggml-threading.h"  // For threadpool management

#include <stdatomic.h>
#include <stdlib.h>

static atomic_int_fast64_t g_fallback_operations = 0;

// Fallback system state - simplified to use original ggml threadpool
static bool g_fallback_initialized = false;

// Helper function for operations that need parameter validation
static enum ggml_status validate_tensor_operation(struct ggml_tensor * tensor, const char * op_name) {
    if (!tensor) {
        GGML_LOG_ERROR("%s: Invalid tensor\n", op_name);
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

// Calculate required work size for individual operations
static size_t ggml_numa_fallback_calc_work_size(struct ggml_tensor * tensor) {
    if (!tensor) return 0;
    
    const int n_tasks = 1;  // Single-threaded fallback
    size_t work_size = 0;
    
    switch (tensor->op) {
        case GGML_OP_MUL_MAT:
            {
                // Calculate proper work buffer size for MUL_MAT operations based on assertion requirements
                const struct ggml_tensor * src0 = tensor->src[0];
                const struct ggml_tensor * src1 = tensor->src[1];
                
                if (src0 && src1) {
                    // Get tensor dimensions (equivalent to GGML_TENSOR_BINARY_OP_LOCALS)
                    const int64_t ne10 = src1->ne[0];  // src1 first dimension
                    const int64_t ne11 = src1->ne[1];  // src1 second dimension
                    const int64_t ne12 = src1->ne[2];  // src1 third dimension
                    const int64_t ne13 = src1->ne[3];  // src1 fourth dimension
                    
                    // Get vector dot type for work buffer calculation
                    const struct ggml_type_traits_cpu * traits = ggml_get_type_traits_cpu(src0->type);
                    const enum ggml_type vec_dot_type = traits->vec_dot_type;
                    
                    // Only need work buffer if src1->type != vec_dot_type (as per ggml_compute_forward_mul_mat)
                    if (src1->type != vec_dot_type) {
                        // Calculate work buffer sizes according to ggml_compute_forward_mul_mat requirements
                        // This matches the assertion: params->wsize >= ne13*nbw3
                        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
                        const size_t nbw2 = nbw1 * ne11;
                        const size_t nbw3 = nbw2 * ne12;
                        
                        // Total work buffer size: ne13 * nbw3 (this satisfies the assertion)
                        work_size = ne13 * nbw3;
                        
                        GGML_LOG_DEBUG("MUL_MAT work buffer calculation: ne10=%ld, ne11=%ld, ne12=%ld, ne13=%ld\n", 
                                      ne10, ne11, ne12, ne13);
                        GGML_LOG_DEBUG("MUL_MAT work buffer sizes: nbw1=%zu, nbw2=%zu, nbw3=%zu, total=%zu\n", 
                                      nbw1, nbw2, nbw3, work_size);
                    } else {
                        // No work buffer needed if types match
                        work_size = 0;
                        GGML_LOG_DEBUG("MUL_MAT: src1 type matches vec_dot_type, no work buffer needed\n");
                    }
                } else {
                    GGML_LOG_WARN("MUL_MAT operation missing src tensors, using fallback work size\n");
                    work_size = ggml_row_size(GGML_TYPE_F32, 4096);  // Fallback size
                }
            } break;
        case GGML_OP_SOFT_MAX:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            {
                work_size = ggml_type_size(GGML_TYPE_F32) * tensor->ne[0] * n_tasks;
            } break;
        case GGML_OP_CPY:
        case GGML_OP_DUP:
            {
                if (ggml_is_quantized(tensor->type)) {
                    work_size = ggml_type_size(GGML_TYPE_F32) * tensor->ne[0] * n_tasks;
                }
            } break;
        case GGML_OP_ADD:
        case GGML_OP_ADD1:
            {
                if (tensor->src[0] && ggml_is_quantized(tensor->src[0]->type)) {
                    work_size = ggml_type_size(GGML_TYPE_F32) * tensor->src[0]->ne[0] * n_tasks;
                }
            } break;
        default:
            // Most operations don't need extra work space
            work_size = 0;
            break;
    }
    
    return work_size;
}

enum ggml_status ggml_numa_fallback_init(void) {
    if (g_fallback_initialized) {
        GGML_LOG_DEBUG("Fallback system already initialized\n");
        return GGML_STATUS_SUCCESS;
    }
    
    GGML_LOG_INFO("Initializing fallback system to use original ggml threadpool\n");
    g_fallback_initialized = true;
    
    GGML_LOG_INFO("Fallback system initialized successfully\n");
    return GGML_STATUS_SUCCESS;
}

void ggml_numa_fallback_cleanup(void) {
    if (!g_fallback_initialized) {
        return;
    }
    
    GGML_LOG_INFO("Cleaning up fallback system (operations executed: %zu)\n", 
                  atomic_load_explicit(&g_fallback_operations, memory_order_relaxed));
    
    g_fallback_initialized = false;
    atomic_store_explicit(&g_fallback_operations, 0, memory_order_relaxed);
    
    GGML_LOG_DEBUG("Fallback system cleanup completed\n");
}

bool ggml_numa_fallback_is_initialized(void) {
    return g_fallback_initialized;
}

enum ggml_status ggml_numa_fallback_execute(struct ggml_tensor * tensor, struct ggml_cplan * cplan) {
    if (!tensor) {
        GGML_LOG_ERROR("Invalid tensor for fallback execution\n");
        return GGML_STATUS_FAILED;
    }
    
    // Ensure fallback system is initialized
    if (!g_fallback_initialized) {
        GGML_LOG_WARN("Fallback system not initialized, initializing now\n");
        enum ggml_status init_result = ggml_numa_fallback_init();
        if (init_result != GGML_STATUS_SUCCESS) {
            GGML_LOG_ERROR("Failed to initialize fallback system\n");
            return GGML_STATUS_FAILED;
        }
    }
    
    atomic_fetch_add_explicit(&g_fallback_operations, 1, memory_order_relaxed);
    
    // Determine number of threads and threadpool to use
    int n_threads = 1;  // Default single-threaded fallback
    struct ggml_threadpool * threadpool = NULL;
    struct ggml_threadpool * allocated_threadpool = NULL;  // Track if we need to free
    
    if (cplan) {
        n_threads = cplan->n_threads > 0 ? cplan->n_threads : 1;
        threadpool = cplan->threadpool;  // Use original ggml threadpool
        GGML_LOG_DEBUG("Using original ggml threadpool with %d threads\n", n_threads);
    } else {
        GGML_LOG_DEBUG("No cplan provided, creating temporary single-threaded threadpool\n");
        // Create a temporary single-threaded threadpool when no cplan is provided
        struct ggml_threadpool_params tpp = ggml_threadpool_params_default(1);
        allocated_threadpool = ggml_threadpool_new(&tpp);
        if (!allocated_threadpool) {
            GGML_LOG_ERROR("Failed to create temporary threadpool for fallback execution\n");
            return GGML_STATUS_FAILED;
        }
        threadpool = allocated_threadpool;
        n_threads = 1;
    }
    
    // For ALL operations, use original ggml graph computation system
    GGML_LOG_DEBUG("Executing %s operation via fallback using original ggml graph computation\n", ggml_op_name(tensor->op));
    
    // Create a temporary context for this operation
    struct ggml_init_params init_params = {
        .mem_size = 1024 * 1024,  // 1MB should be enough for small graphs
        .mem_buffer = NULL,
        .no_alloc = false
    };
    
    struct ggml_context * temp_ctx = ggml_init(init_params);
    if (!temp_ctx) {
        GGML_LOG_ERROR("Failed to create temporary context for fallback execution\n");
        if (allocated_threadpool) {
            ggml_threadpool_free(allocated_threadpool);
        }
        return GGML_STATUS_FAILED;
    }
    
    // Create a simple computation graph with just this operation
    struct ggml_cgraph * cgraph = ggml_new_graph(temp_ctx);
    if (!cgraph) {
        GGML_LOG_ERROR("Failed to create temporary graph for fallback execution\n");
        ggml_free(temp_ctx);
        if (allocated_threadpool) {
            ggml_threadpool_free(allocated_threadpool);
        }
        return GGML_STATUS_FAILED;
    }
    
    // Add the operation to the graph
    ggml_build_forward_expand(cgraph, tensor);
    
    // Create a computation plan
    struct ggml_cplan plan = ggml_graph_plan(cgraph, n_threads, threadpool);
    
    // Use provided threadpool if available, otherwise use our temporary one
    if (cplan && cplan->threadpool) {
        plan.threadpool = cplan->threadpool;
        GGML_LOG_DEBUG("Using provided threadpool for original ggml computation\n");
    } else {
        plan.threadpool = threadpool;
        GGML_LOG_DEBUG("Using temporary threadpool for original ggml computation\n");
    }
    
    // Allocate work buffer if the plan requires it
    if (plan.work_size > 0) {
        plan.work_data = malloc(plan.work_size);
        if (!plan.work_data) {
            GGML_LOG_ERROR("Failed to allocate work buffer (%zu bytes) for original ggml computation\n", plan.work_size);
            ggml_free(temp_ctx);
            if (allocated_threadpool) {
                ggml_threadpool_free(allocated_threadpool);
            }
            return GGML_STATUS_FAILED;
        }
        GGML_LOG_DEBUG("Allocated work buffer: %zu bytes for original ggml computation\n", plan.work_size);
    }
    
    // Execute the graph using original ggml computation
    enum ggml_status result = ggml_graph_compute(cgraph, &plan);
    
    // Clean up
    if (plan.work_data) {
        free(plan.work_data);
        GGML_LOG_DEBUG("Freed work buffer for original ggml computation\n");
    }
    ggml_free(temp_ctx);
    
    // Clean up allocated threadpool if we created one
    if (allocated_threadpool) {
        ggml_threadpool_free(allocated_threadpool);
        GGML_LOG_DEBUG("Freed temporary threadpool\n");
    }
    
    if (result == GGML_STATUS_SUCCESS) {
        GGML_LOG_DEBUG("Original ggml computation completed successfully\n");
    } else {
        GGML_LOG_ERROR("Original ggml computation failed\n");
    }
    
    return result;
}

bool ggml_numa_fallback_is_supported(enum ggml_op op) {
    (void)op;
    return true;
}

void ggml_numa_fallback_get_stats(int64_t * fallback_count) {
    if (fallback_count) {
        *fallback_count = atomic_load_explicit(&g_fallback_operations, memory_order_relaxed);
    }
}
