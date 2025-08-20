/*
 * NUMA Kernel Registry Implementation
 * 
 * Centralized registry that acts as a database for NUMA kernels.
 * The executor queries this registry to get execution strategies,
 * buffer requirements, and work function pointers.
 */

#include "numa-kernels.h"
#include "add.h"
#include "../ggml-impl.h"

// Internal kernel registry entry structure
// This contains the internal implementation details that external code doesn't need
typedef struct {
    enum ggml_op operation;                                                           // Operation type
    bool (*supports)(const struct ggml_tensor * tensor);                            // Support check
    ggml_numa_execution_strategy_t (*get_strategy)(const struct ggml_tensor * tensor); // Strategy determination
    size_t (*get_buffer_size)(const struct ggml_tensor * tensor);                   // Buffer size calculation
    ggml_numa_work_function_t (*get_work_function)(const struct ggml_tensor * tensor); // Work function selection
    float (*get_efficiency)(const struct ggml_tensor * tensor);                     // Efficiency estimation
    const char * name;                                                               // Human-readable name
} ggml_numa_internal_kernel_entry_t;

// Global registry state
static ggml_numa_internal_kernel_entry_t * g_numa_kernels = NULL;
static size_t g_numa_kernel_count = 0;
static size_t g_numa_kernel_capacity = 0;
static bool g_numa_kernels_initialized = false;

// Forward declaration for internal registry functions
static bool register_kernel(const ggml_numa_internal_kernel_entry_t * kernel);
static const ggml_numa_internal_kernel_entry_t * find_kernel_for_operation(enum ggml_op op);

bool ggml_numa_kernels_init(void) {
    if (g_numa_kernels_initialized) {
        GGML_LOG_DEBUG("NUMA Kernels: Already initialized\n");
        return true;
    }
    
    // Initial capacity for kernel registry
    g_numa_kernel_capacity = 16;
    g_numa_kernels = (ggml_numa_internal_kernel_entry_t*)malloc(
        g_numa_kernel_capacity * sizeof(ggml_numa_internal_kernel_entry_t));
    
    if (!g_numa_kernels) {
        GGML_LOG_ERROR("NUMA Kernels: Failed to allocate kernel registry\n");
        return false;
    }
    
    g_numa_kernel_count = 0;
    
    // Register ADD kernel
    ggml_numa_internal_kernel_entry_t add_kernel = {
        .operation = GGML_OP_ADD,
        .supports = ggml_numa_kernel_add_supports,
        .get_strategy = ggml_numa_kernel_add_get_strategy,
        .get_buffer_size = ggml_numa_kernel_add_get_buffer_size,
        .get_work_function = ggml_numa_kernel_add_get_work_function,
        .get_efficiency = ggml_numa_kernel_add_get_efficiency,
        .name = "NUMA Add"
    };
    
    if (!register_kernel(&add_kernel)) {
        GGML_LOG_ERROR("NUMA Kernels: Failed to register ADD kernel\n");
        free(g_numa_kernels);
        g_numa_kernels = NULL;
        return false;
    }
    
    g_numa_kernels_initialized = true;
    GGML_LOG_DEBUG("NUMA Kernels: Successfully initialized with %zu kernels\n", g_numa_kernel_count);
    
    return true;
}

void ggml_numa_kernels_cleanup(void) {
    if (g_numa_kernels) {
        free(g_numa_kernels);
        g_numa_kernels = NULL;
    }
    
    g_numa_kernel_count = 0;
    g_numa_kernel_capacity = 0;
    g_numa_kernels_initialized = false;
    
    GGML_LOG_DEBUG("NUMA Kernels: Registry cleaned up\n");
}

// Main query interface - this is what the executor uses
ggml_numa_kernel_query_result_t ggml_numa_kernels_query(const struct ggml_tensor * tensor) {
    ggml_numa_kernel_query_result_t result = {
        .supported = false,
        .strategy = {0},
        .work_buffer_size_per_thread = 0,
        .work_function = NULL,
        .efficiency_score = 0.0f,
        .kernel_name = "None"
    };
    
    if (!tensor || !g_numa_kernels_initialized) {
        return result;
    }
    
    // Find kernel for this operation
    const ggml_numa_internal_kernel_entry_t * kernel = find_kernel_for_operation(tensor->op);
    if (!kernel) {
        GGML_LOG_DEBUG("NUMA Kernels: No kernel found for operation %s\n", ggml_op_name(tensor->op));
        return result;
    }
    
    // Check if kernel supports this specific tensor
    if (!kernel->supports(tensor)) {
        GGML_LOG_DEBUG("NUMA Kernels: %s kernel does not support this tensor configuration\n", kernel->name);
        return result;
    }
    
    // Populate result with kernel information
    result.supported = true;
    result.strategy = kernel->get_strategy(tensor);
    result.work_buffer_size_per_thread = kernel->get_buffer_size(tensor);
    result.work_function = kernel->get_work_function(tensor);
    result.efficiency_score = kernel->get_efficiency(tensor);
    result.kernel_name = kernel->name;
    
    GGML_LOG_DEBUG("NUMA Kernels: Query for %s: supported=%s, strategy=%s, buffer=%zu, efficiency=%.2f\n",
                   ggml_op_name(tensor->op),
                   result.supported ? "true" : "false",
                   (result.strategy.node_strategy == NUMA_NODE_STRATEGY_DATA_PARALLEL) ? "data-parallel" : "single-node",
                   result.work_buffer_size_per_thread,
                   result.efficiency_score);
    
    return result;
}

// Internal helper functions

static bool register_kernel(const ggml_numa_internal_kernel_entry_t * kernel) {
    if (!kernel) {
        return false;
    }
    
    // Expand capacity if needed
    if (g_numa_kernel_count >= g_numa_kernel_capacity) {
        size_t new_capacity = g_numa_kernel_capacity * 2;
        ggml_numa_internal_kernel_entry_t * new_kernels = (ggml_numa_internal_kernel_entry_t*)realloc(
            g_numa_kernels, new_capacity * sizeof(ggml_numa_internal_kernel_entry_t));
        
        if (!new_kernels) {
            GGML_LOG_ERROR("NUMA Kernels: Failed to expand kernel registry capacity\n");
            return false;
        }
        
        g_numa_kernels = new_kernels;
        g_numa_kernel_capacity = new_capacity;
    }
    
    // Add kernel to registry
    g_numa_kernels[g_numa_kernel_count] = *kernel;
    g_numa_kernel_count++;
    
    GGML_LOG_DEBUG("NUMA Kernels: Registered %s kernel for operation %s\n", 
                   kernel->name, ggml_op_name(kernel->operation));
    
    return true;
}

static const ggml_numa_internal_kernel_entry_t * find_kernel_for_operation(enum ggml_op op) {
    for (size_t i = 0; i < g_numa_kernel_count; i++) {
        if (g_numa_kernels[i].operation == op) {
            return &g_numa_kernels[i];
        }
    }
    
    return NULL;
}
