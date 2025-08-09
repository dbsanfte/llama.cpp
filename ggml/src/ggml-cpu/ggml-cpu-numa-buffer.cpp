#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include <numa.h>
#include <numaif.h>

// NUMA-aware buffer type for KV cache and other large allocations

// NUMA buffer context - stores allocation info
struct ggml_numa_buffer_context {
    void * data;
    size_t size; 
    int numa_node;
    bool is_numa_allocated;
};

// Check if NUMA is available and coordinator is active
static bool ggml_numa_buffer_should_use_numa() {
    // Only use NUMA-aware allocation if:
    // 1. NUMA is available on the system
    // 2. NUMA coordinator is active (GGML_NUMA_MIRROR enabled)
    return numa_available() != -1 && ggml_is_numa();
}

// Determine optimal NUMA node for allocation
static int ggml_numa_buffer_get_preferred_node(size_t size) {
    if (!ggml_numa_buffer_should_use_numa()) {
        return -1; // Use standard allocation
    }
    
    // For large allocations (like KV cache), use coordinator to determine placement
    // This would ideally integrate with the coordinator's strategy
    
    // For now, simple heuristic:
    // - Small allocations (< 1MB): current node
    // - Large allocations (>= 1MB): round-robin across nodes for load balancing
    
    static int next_node = 0;
    int max_node = numa_max_node();
    
    if (size < 1024 * 1024) {
        // Small allocation - use current node
        return numa_preferred();
    } else {
        // Large allocation - round-robin for load balancing
        int node = next_node;
        next_node = (next_node + 1) % (max_node + 1);
        return node;
    }
}

// NUMA-aware buffer interface implementations

static void ggml_backend_numa_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    struct ggml_numa_buffer_context * ctx = (struct ggml_numa_buffer_context *)buffer->context;
    
    if (ctx->is_numa_allocated && numa_available() != -1) {
        numa_free(ctx->data, ctx->size);
    } else {
        ggml_aligned_free(ctx->data, ctx->size);
    }
    
    free(ctx);
}

static void * ggml_backend_numa_buffer_get_base(ggml_backend_buffer_t buffer) {
    struct ggml_numa_buffer_context * ctx = (struct ggml_numa_buffer_context *)buffer->context;
    
    uintptr_t data = (uintptr_t)ctx->data;
    
    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }
    
    return (void *)data;
}

static void ggml_backend_numa_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((char *)tensor_data(tensor) + offset, value, size);
    GGML_UNUSED(buffer);
}

static void ggml_backend_numa_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy((char *)tensor_data(tensor) + offset, data, size);
    GGML_UNUSED(buffer);
}

static void ggml_backend_numa_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, (const char *)tensor_data(tensor) + offset, size);
    GGML_UNUSED(buffer);
}

static bool ggml_backend_numa_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(tensor_data(dst), tensor_data(src), ggml_nbytes(src));
        return true;
    }
    return false;
    GGML_UNUSED(buffer);
}

static void ggml_backend_numa_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    struct ggml_numa_buffer_context * ctx = (struct ggml_numa_buffer_context *)buffer->context;
    memset(ctx->data, value, ctx->size);
}

// Buffer interface struct
static const struct ggml_backend_buffer_i ggml_backend_numa_buffer_i = {
    /* .free_buffer     = */ ggml_backend_numa_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_numa_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_numa_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_numa_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_numa_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_numa_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_numa_buffer_clear,
    /* .reset           = */ NULL,
};

// Buffer type interface implementations

static const char * ggml_backend_numa_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_NUMA";
    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_numa_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    struct ggml_numa_buffer_context * ctx = (struct ggml_numa_buffer_context *)malloc(sizeof(struct ggml_numa_buffer_context));
    if (ctx == NULL) {
        return NULL;
    }
    
    ctx->size = size;
    ctx->numa_node = ggml_numa_buffer_get_preferred_node(size);
    ctx->is_numa_allocated = false;
    ctx->data = NULL;
    
    // Try NUMA-aware allocation first
    if (ctx->numa_node >= 0 && ggml_numa_buffer_should_use_numa()) {
        ctx->data = numa_alloc_onnode(size, ctx->numa_node);
        if (ctx->data != NULL) {
            ctx->is_numa_allocated = true;
            
            // Verify allocation is on correct node
            int actual_node = -1;
            if (get_mempolicy(&actual_node, NULL, 0, ctx->data, MPOL_F_NODE | MPOL_F_ADDR) == 0) {
                if (actual_node != ctx->numa_node) {
                    // Allocation ended up on wrong node, log warning but continue
                    GGML_LOG_WARN("NUMA allocation requested node %d, got node %d\n", ctx->numa_node, actual_node);
                }
            }
        }
    }
    
    // Fallback to standard aligned allocation
    if (ctx->data == NULL) {
        ctx->data = ggml_aligned_malloc(size);
        ctx->is_numa_allocated = false;
        ctx->numa_node = -1;
        
        if (ctx->data == NULL) {
            free(ctx);
            return NULL;
        }
    }
    
    // Log allocation info for debugging
    if (ctx->is_numa_allocated) {
        GGML_LOG_DEBUG("NUMA buffer allocated %zu bytes on node %d\n", size, ctx->numa_node);
    } else {
        GGML_LOG_DEBUG("Standard buffer allocated %zu bytes (NUMA not available or not beneficial)\n", size);
    }
    
    return ggml_backend_buffer_init(buft, ggml_backend_numa_buffer_i, ctx, size);
}

static size_t ggml_backend_numa_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;
    GGML_UNUSED(buft);
}

static bool ggml_backend_numa_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;
    GGML_UNUSED(buft);
}

// Buffer type interface struct
static const struct ggml_backend_buffer_type_i ggml_backend_numa_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_numa_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_numa_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_numa_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
    /* .is_host          = */ ggml_backend_numa_buffer_type_is_host,
};

// Export the NUMA-aware buffer type
ggml_backend_buffer_type_t ggml_backend_cpu_numa_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type_numa = {
        /* .iface   = */ ggml_backend_numa_buffer_type_interface,
        /* .device  = */ NULL, // TODO: should be CPU device
        /* .context = */ NULL,
    };
    
    return &ggml_backend_cpu_buffer_type_numa;
}
