#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml-numa-coordinator.h"
#include <numa.h>
#include <numaif.h>
#include <sched.h>

// NUMA-aware buffer type for KV cache and other large allocations

// Forward declarations
static int ggml_numa_buffer_get_replication_nodes(int * nodes, int max_nodes);
static int ggml_numa_buffer_get_preferred_node(size_t size);
static bool ggml_numa_buffer_should_use_numa(void);

// C linkage for coordinator function
extern "C" {
    extern int ggml_numa_coordinator_get_active_nodes(struct ggml_numa_coordinator_manager * mgr, int * nodes, int max_nodes);
}

// Cache strategy constants (should match llama_numa_cache_strategy)
#define GGML_NUMA_CACHE_STRATEGY_DISABLED 0  // No cache replication
#define GGML_NUMA_CACHE_STRATEGY_EAGER    1  // Immediate replication across all nodes
#define GGML_NUMA_CACHE_STRATEGY_LAZY     2  // On-demand replication when accessed
#define GGML_NUMA_CACHE_STRATEGY_DELTA    3  // Incremental updates only
#define GGML_NUMA_CACHE_STRATEGY_PARTIAL  4  // Replicate working set only

// NUMA buffer context - stores allocation info
struct ggml_numa_buffer_context {
    void * data;
    size_t size; 
    int numa_node;
    bool is_numa_allocated;
    
    // Cache replication support
    bool is_replicated;
    int num_replicas;
    void ** replica_data;  // Array of pointers to replica data on different nodes
    int * replica_nodes;   // Array of NUMA nodes for each replica
    
    // Strategy-specific state
    int cache_strategy;           // The cache strategy used for this buffer
    bool * replica_allocated;    // Track which replicas are actually allocated (for lazy)
    bool is_dirty;               // Track if primary data has been modified (for delta)
    uint64_t access_count;       // Access counter for working set management (for partial)
    int last_access_node;        // Last NUMA node that accessed this buffer
};

// Check if NUMA is available and coordinator is active
static bool ggml_numa_buffer_should_use_numa() {
    // Only use NUMA-aware allocation if:
    // 1. NUMA is available on the system
    // 2. NUMA coordinator is active (GGML_NUMA_MIRROR enabled)
    return numa_available() != -1 && ggml_is_numa();
}

// Check if cache replication should be used
static bool ggml_numa_buffer_should_use_replication(size_t size, int* out_strategy) {
    // Only enable replication for large allocations (likely KV caches)
    const size_t min_replication_size = 64 * 1024 * 1024; // 64MB threshold
    
    if (size < min_replication_size) {
        return false;
    }
    
    if (!ggml_numa_buffer_should_use_numa()) {
        return false;
    }
    
    // Only replicate when NUMA mirroring is enabled
    // This integrates with our runtime --numa mirror vs --numa distribute logic
#ifdef GGML_NUMA_MIRROR
    extern bool ggml_numa_should_mirror(void);
    if (!ggml_numa_should_mirror()) {
        return false; // Mirroring disabled, no buffer replication
    }
#else
    // If GGML_NUMA_MIRROR not compiled, no replication
    return false;
#endif
    
    // Check if we have multiple NUMA nodes
    int max_node = numa_max_node();
    if (max_node <= 0) {
        return false;
    }
    
    // Get the current cache strategy from GGML
    int cache_strategy = ggml_numa_get_cache_strategy();
    if (out_strategy) {
        *out_strategy = cache_strategy;
    }
    
    // Apply cache strategy logic
    switch (cache_strategy) {
        case GGML_NUMA_CACHE_STRATEGY_DISABLED:
            // No replication
            return false;
            
        case GGML_NUMA_CACHE_STRATEGY_EAGER:
            // Always replicate large allocations immediately
            return true;
            
        case GGML_NUMA_CACHE_STRATEGY_LAZY:
            // For lazy strategy, only enable replication for very large buffers
            // that are likely to be accessed frequently (like KV cache)
            return size >= 128 * 1024 * 1024; // 128MB threshold for lazy
            
        case GGML_NUMA_CACHE_STRATEGY_DELTA:
            // Delta strategy - enable replication but with reduced redundancy
            // Could be enhanced to use fewer replicas
            return size >= 256 * 1024 * 1024; // 256MB threshold for delta
            
        case GGML_NUMA_CACHE_STRATEGY_PARTIAL:
            // Partial replication - only for the largest allocations
            return size >= 512 * 1024 * 1024; // 512MB threshold for partial
            
        default:
            // Unknown strategy - fall back to conservative approach
            return false;
    }
}

// Strategy-specific allocation functions

// EAGER: Allocate replicas on all nodes immediately
static bool ggml_numa_buffer_allocate_eager(struct ggml_numa_buffer_context * ctx, size_t size) {
    const int max_nodes = 8;
    int replica_nodes[max_nodes];
    int num_nodes = ggml_numa_buffer_get_replication_nodes(replica_nodes, max_nodes);
    
    if (num_nodes <= 1) return false;
    
    ctx->num_replicas = num_nodes;
    ctx->replica_data = (void**)calloc(num_nodes, sizeof(void*));
    ctx->replica_nodes = (int*)malloc(num_nodes * sizeof(int));
    ctx->replica_allocated = (bool*)malloc(num_nodes * sizeof(bool));
    
    if (!ctx->replica_data || !ctx->replica_nodes || !ctx->replica_allocated) {
        free(ctx->replica_data); free(ctx->replica_nodes); free(ctx->replica_allocated);
        return false;
    }
    
    // Allocate on all nodes immediately
    for (int i = 0; i < num_nodes; i++) {
        ctx->replica_nodes[i] = replica_nodes[i];
        ctx->replica_data[i] = numa_alloc_onnode(size, replica_nodes[i]);
        ctx->replica_allocated[i] = (ctx->replica_data[i] != NULL);
        
        if (!ctx->replica_allocated[i]) {
            // Clean up previous allocations on failure
            for (int j = 0; j < i; j++) {
                if (ctx->replica_allocated[j]) {
                    numa_free(ctx->replica_data[j], size);
                }
            }
            free(ctx->replica_data); free(ctx->replica_nodes); free(ctx->replica_allocated);
            return false;
        }
    }
    
    ctx->data = ctx->replica_data[0]; // Primary pointer
    ctx->numa_node = ctx->replica_nodes[0];
    GGML_LOG_INFO("EAGER: allocated %zu bytes across %d nodes immediately\n", size, num_nodes);
    return true;
}

// LAZY: Allocate on preferred node, replicas created on-demand
static bool ggml_numa_buffer_allocate_lazy(struct ggml_numa_buffer_context * ctx, size_t size) {
    const int max_nodes = 8;
    int replica_nodes[max_nodes];
    int num_nodes = ggml_numa_buffer_get_replication_nodes(replica_nodes, max_nodes);
    
    if (num_nodes <= 1) return false;
    
    ctx->num_replicas = num_nodes;
    ctx->replica_data = (void**)calloc(num_nodes, sizeof(void*));
    ctx->replica_nodes = (int*)malloc(num_nodes * sizeof(int));
    ctx->replica_allocated = (bool*)calloc(num_nodes, sizeof(bool));
    
    if (!ctx->replica_data || !ctx->replica_nodes || !ctx->replica_allocated) {
        free(ctx->replica_data); free(ctx->replica_nodes); free(ctx->replica_allocated);
        return false;
    }
    
    // Set up node array but only allocate on preferred node initially
    for (int i = 0; i < num_nodes; i++) {
        ctx->replica_nodes[i] = replica_nodes[i];
        ctx->replica_data[i] = NULL;
        ctx->replica_allocated[i] = false;
    }
    
    // Find preferred node and allocate there
    int preferred_idx = 0;
    for (int i = 0; i < num_nodes; i++) {
        if (ctx->replica_nodes[i] == ctx->numa_node) {
            preferred_idx = i;
            break;
        }
    }
    
    ctx->replica_data[preferred_idx] = numa_alloc_onnode(size, ctx->replica_nodes[preferred_idx]);
    if (!ctx->replica_data[preferred_idx]) {
        free(ctx->replica_data); free(ctx->replica_nodes); free(ctx->replica_allocated);
        return false;
    }
    ctx->replica_allocated[preferred_idx] = true;
    ctx->data = ctx->replica_data[preferred_idx];
    
    GGML_LOG_INFO("LAZY: allocated %zu bytes on node %d, replicas on-demand\n", size, ctx->numa_node);
    return true;
}

// DELTA: Like eager but with dirty tracking (simplified for now)
static bool ggml_numa_buffer_allocate_delta(struct ggml_numa_buffer_context * ctx, size_t size) {
    bool result = ggml_numa_buffer_allocate_eager(ctx, size);
    if (result) {
        ctx->is_dirty = false; // Start clean
        GGML_LOG_INFO("DELTA: allocated %zu bytes with change tracking\n", size);
    }
    return result;
}

// PARTIAL: Like lazy but with access tracking
static bool ggml_numa_buffer_allocate_partial(struct ggml_numa_buffer_context * ctx, size_t size) {
    bool result = ggml_numa_buffer_allocate_lazy(ctx, size);
    if (result) {
        ctx->access_count = 0;
        ctx->last_access_node = ctx->numa_node;
        GGML_LOG_INFO("PARTIAL: allocated %zu bytes with working set tracking\n", size);
    }
    return result;
}

// On-demand replica allocation for lazy strategy
static void* ggml_numa_buffer_ensure_replica(struct ggml_numa_buffer_context * ctx, int target_node) {
    if (!ctx->is_replicated || !ctx->replica_data || !ctx->replica_nodes || !ctx->replica_allocated) {
        return ctx->data; // Fall back to primary data
    }
    
    // Find the replica index for the target node
    int replica_idx = -1;
    for (int i = 0; i < ctx->num_replicas; i++) {
        if (ctx->replica_nodes[i] == target_node) {
            replica_idx = i;
            break;
        }
    }
    
    if (replica_idx == -1) {
        return ctx->data; // Node not in replica list, use primary
    }
    
    // If replica is already allocated, return it
    if (ctx->replica_allocated[replica_idx]) {
        return ctx->replica_data[replica_idx];
    }
    
    // Allocate replica on-demand
    ctx->replica_data[replica_idx] = numa_alloc_onnode(ctx->size, target_node);
    if (ctx->replica_data[replica_idx]) {
        ctx->replica_allocated[replica_idx] = true;
        
        // Copy data from primary replica
        int primary_idx = 0; // Find first allocated replica as primary
        for (int i = 0; i < ctx->num_replicas; i++) {
            if (ctx->replica_allocated[i]) {
                primary_idx = i;
                break;
            }
        }
        
        if (ctx->replica_allocated[primary_idx]) {
            memcpy(ctx->replica_data[replica_idx], ctx->replica_data[primary_idx], ctx->size);
            GGML_LOG_INFO("LAZY: created on-demand replica on node %d (%zu bytes)\n", target_node, ctx->size);
        }
        
        return ctx->replica_data[replica_idx];
    }
    
    return ctx->data; // Allocation failed, fall back to primary
}

// Get number of NUMA nodes for replication
static int ggml_numa_buffer_get_replication_nodes(int * nodes, int max_nodes) {
    if (!numa_available()) {
        return 0;
    }
    
    // First try to get nodes from the coordinator (preferred approach)
    int count = ggml_numa_coordinator_get_active_nodes(NULL, nodes, max_nodes);
    
    if (count > 0) {
        // Successfully got nodes from coordinator - use those
        return count;
    }
    
    // Fallback: enumerate all available NUMA nodes
    int max_node = numa_max_node();
    count = 0;
    
    for (int i = 0; i <= max_node && count < max_nodes; i++) {
        if (numa_node_size(i, NULL) > 0) {  // Node has memory
            nodes[count++] = i;
        }
    }
    
    return count;
}

// Determine optimal NUMA node for allocation
static int ggml_numa_buffer_get_preferred_node(size_t size) {
    if (!ggml_numa_buffer_should_use_numa()) {
        return -1; // Use standard allocation
    }
    
    // Get the NUMA nodes that the coordinator is using
    const int max_nodes = 8;
    int coordinator_nodes[max_nodes];
    int num_coordinator_nodes = ggml_numa_coordinator_get_active_nodes(NULL, coordinator_nodes, max_nodes);
    
    if (num_coordinator_nodes > 0) {
        // Use the coordinator's nodes - this ensures buffer allocation aligns with compute
        
        if (size < 1024 * 1024) {
            // Small allocation (< 1MB): prefer current node if it's in coordinator's list
            int current_node = numa_preferred();
            if (current_node >= 0) {
                for (int i = 0; i < num_coordinator_nodes; i++) {
                    if (coordinator_nodes[i] == current_node) {
                        return current_node; // Current node is being used by coordinator
                    }
                }
            }
            
            // Current node not in coordinator list, use first coordinator node
            return coordinator_nodes[0];
        } else {
            // Large allocation (>= 1MB): round-robin across coordinator nodes for load balancing
            static int next_coordinator_idx = 0;
            int selected_idx = next_coordinator_idx % num_coordinator_nodes;
            next_coordinator_idx = (next_coordinator_idx + 1) % num_coordinator_nodes;
            
            return coordinator_nodes[selected_idx];
        }
    }
    
    // Fallback: coordinator not available, use legacy heuristic
    // For now, simple heuristic:
    // - Small allocations (< 1MB): current node
    // - Large allocations (>= 1MB): round-robin across nodes for load balancing

    GGML_LOG_WARN("NUMA cpu buffer allocation: Coordinator not enabled, falling back to legacy allocation strategy");
    static int next_node = 0;
    
    if (size < 1024 * 1024) {
        // Small allocation - use current node
        return numa_preferred();
    } else {
        // Large allocation - round-robin for load balancing
        int max_node = numa_max_node();
        int node = next_node;
        next_node = (next_node + 1) % (max_node + 1);
        return node;
    }
}

// NUMA-aware buffer interface implementations

static void ggml_backend_numa_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    struct ggml_numa_buffer_context * ctx = (struct ggml_numa_buffer_context *)buffer->context;
    
    if (ctx->is_replicated && ctx->replica_data && ctx->replica_allocated) {
        // Free replicated buffers
        for (int i = 0; i < ctx->num_replicas; i++) {
            if (ctx->replica_allocated[i] && ctx->replica_data[i]) {
                numa_free(ctx->replica_data[i], ctx->size);
            }
        }
        free(ctx->replica_data);
        free(ctx->replica_nodes);
        free(ctx->replica_allocated);
    } else if (ctx->is_numa_allocated && numa_available() != -1) {
        numa_free(ctx->data, ctx->size);
    } else {
        ggml_aligned_free(ctx->data, ctx->size);
    }
    
    free(ctx);
}

static void * ggml_backend_numa_buffer_get_base(ggml_backend_buffer_t buffer) {
    struct ggml_numa_buffer_context * ctx = (struct ggml_numa_buffer_context *)buffer->context;
    
    void * data_ptr = ctx->data;
    
    // For replicated buffers, implement strategy-specific access logic
    if (ctx->is_replicated && ctx->replica_data) {
        int current_node = numa_node_of_cpu(sched_getcpu());
        if (current_node >= 0) {
            switch (ctx->cache_strategy) {
                case GGML_NUMA_CACHE_STRATEGY_EAGER:
                    // All replicas should be available, return local one
                    for (int i = 0; i < ctx->num_replicas; i++) {
                        if (ctx->replica_nodes[i] == current_node && ctx->replica_allocated[i] && ctx->replica_data[i]) {
                            data_ptr = ctx->replica_data[i];
                            break;
                        }
                    }
                    break;
                    
                case GGML_NUMA_CACHE_STRATEGY_LAZY:
                    // Create replica on-demand if needed
                    data_ptr = ggml_numa_buffer_ensure_replica(ctx, current_node);
                    break;
                    
                case GGML_NUMA_CACHE_STRATEGY_DELTA:
                    // For now, behave like eager but could implement dirty tracking here
                    for (int i = 0; i < ctx->num_replicas; i++) {
                        if (ctx->replica_nodes[i] == current_node && ctx->replica_allocated[i] && ctx->replica_data[i]) {
                            data_ptr = ctx->replica_data[i];
                            break;
                        }
                    }
                    break;
                    
                case GGML_NUMA_CACHE_STRATEGY_PARTIAL:
                    // Track access and potentially create replica for working set
                    ctx->access_count++;
                    ctx->last_access_node = current_node;
                    data_ptr = ggml_numa_buffer_ensure_replica(ctx, current_node);
                    break;
                    
                default:
                    // Fall back to first available replica
                    for (int i = 0; i < ctx->num_replicas; i++) {
                        if (ctx->replica_allocated[i] && ctx->replica_data[i]) {
                            data_ptr = ctx->replica_data[i];
                            break;
                        }
                    }
                    break;
            }
        }
        
        // If no suitable replica found, use primary data
        if (data_ptr == ctx->data && ctx->replica_data[0] && ctx->replica_allocated[0]) {
            data_ptr = ctx->replica_data[0];
        }
    }
    
    uintptr_t data = (uintptr_t)data_ptr;
    
    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }
    
    return (void *)data;
}

static void ggml_backend_numa_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    struct ggml_numa_buffer_context * ctx = (struct ggml_numa_buffer_context *)buffer->context;
    
    // Set data in the primary tensor location
    memset((char *)tensor_data(tensor) + offset, value, size);
    
    // For replicated buffers, also update all replicas
    if (ctx->is_replicated && ctx->replica_data) {
        // Calculate tensor offset within the buffer
        void * tensor_base = tensor_data(tensor);
        void * buffer_base = ggml_backend_numa_buffer_get_base(buffer);
        ptrdiff_t tensor_offset = (char *)tensor_base - (char *)buffer_base;
        
        for (int i = 0; i < ctx->num_replicas; i++) {
            if (ctx->replica_data[i]) {
                char * replica_tensor_base = (char *)ctx->replica_data[i] + tensor_offset;
                memset(replica_tensor_base + offset, value, size);
            }
        }
    }
}

static void ggml_backend_numa_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    struct ggml_numa_buffer_context * ctx = (struct ggml_numa_buffer_context *)buffer->context;
    
    // Set data in the primary tensor location
    memcpy((char *)tensor_data(tensor) + offset, data, size);
    
    // For replicated buffers, also update all replicas
    if (ctx->is_replicated && ctx->replica_data) {
        // Calculate tensor offset within the buffer
        void * tensor_base = tensor_data(tensor);
        void * buffer_base = ggml_backend_numa_buffer_get_base(buffer);
        ptrdiff_t tensor_offset = (char *)tensor_base - (char *)buffer_base;
        
        for (int i = 0; i < ctx->num_replicas; i++) {
            if (ctx->replica_data[i]) {
                char * replica_tensor_base = (char *)ctx->replica_data[i] + tensor_offset;
                memcpy(replica_tensor_base + offset, data, size);
            }
        }
    }
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
    GGML_UNUSED(buft);
    struct ggml_numa_buffer_context * ctx = (struct ggml_numa_buffer_context *)malloc(sizeof(struct ggml_numa_buffer_context));
    if (ctx == NULL) {
        return NULL;
    }
    
    ctx->size = size;
    ctx->numa_node = ggml_numa_buffer_get_preferred_node(size);
    ctx->is_numa_allocated = false;
    ctx->is_replicated = false;
    ctx->num_replicas = 0;
    ctx->replica_data = NULL;
    ctx->replica_nodes = NULL;
    ctx->replica_allocated = NULL;
    ctx->cache_strategy = GGML_NUMA_CACHE_STRATEGY_DISABLED;
    ctx->is_dirty = false;
    ctx->access_count = 0;
    ctx->last_access_node = -1;
    ctx->data = NULL;
    
    // Check if we should use replication and get the strategy
    int cache_strategy;
    bool use_replication = ggml_numa_buffer_should_use_replication(size, &cache_strategy);
    ctx->cache_strategy = cache_strategy;
    
    if (use_replication) {
        bool replication_success = false;
        
        // Use strategy-specific allocation
        switch (cache_strategy) {
            case GGML_NUMA_CACHE_STRATEGY_EAGER:
                replication_success = ggml_numa_buffer_allocate_eager(ctx, size);
                break;
                
            case GGML_NUMA_CACHE_STRATEGY_LAZY:
                replication_success = ggml_numa_buffer_allocate_lazy(ctx, size);
                break;
                
            case GGML_NUMA_CACHE_STRATEGY_DELTA:
                replication_success = ggml_numa_buffer_allocate_delta(ctx, size);
                break;
                
            case GGML_NUMA_CACHE_STRATEGY_PARTIAL:
                replication_success = ggml_numa_buffer_allocate_partial(ctx, size);
                break;
                
            default:
                replication_success = false;
                break;
        }
        
        if (replication_success) {
            ctx->is_replicated = true;
            ctx->is_numa_allocated = true;
        } else {
            // Replication failed, will fall back to single allocation
            use_replication = false;
        }
    }
    
    // Single node allocation (either by choice or fallback)
    if (!use_replication) {
        // Try NUMA-aware allocation first
        if (ctx->numa_node >= 0 && ggml_numa_buffer_should_use_numa()) {
            ctx->data = numa_alloc_onnode(size, ctx->numa_node);
            if (ctx->data != NULL) {
                ctx->is_numa_allocated = true;
                
                // Verify allocation is on correct node
                int actual_node = -1;
                if (get_mempolicy(&actual_node, NULL, 0, ctx->data, MPOL_F_NODE | MPOL_F_ADDR) == 0) {
                    GGML_ASSERT(actual_node == ctx->numa_node);
                }
            }
        }
        
        // Fallback to standard aligned allocation
        GGML_LOG_WARN("NUMA-aware buffer allocation failed, falling back to standard allocation\n");
        if (ctx->data == NULL) {
            ctx->data = ggml_aligned_malloc(size);
            ctx->is_numa_allocated = false;
            ctx->numa_node = -1;
            
            if (ctx->data == NULL) {
                free(ctx);
                return NULL;
            }
        }
    }
    
    // Log allocation info for debugging
    if (ctx->is_replicated) {
        GGML_LOG_DEBUG("NUMA replicated buffer allocated %zu bytes across %d nodes\n", size, ctx->num_replicas);
    } else if (ctx->is_numa_allocated) {
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
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };
    
    return &ggml_backend_cpu_buffer_type_numa;
}
