
#include "ggml-numa-flash-attn-ext.h"
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

// FLASH_ATTN_EXT chunk work function - handles flash attention computation with NUMA data parallelism
// This implementation extracts the mathematical kernel to avoid threading conflicts
enum ggml_status ggml_numa_work_function_flash_attn_ext_chunk(void * work_context, struct ggml_compute_params * params) {
    if (!work_context || !params) {
        GGML_LOG_ERROR("FLASH_ATTN_EXT work function: Invalid parameters\n");
        return GGML_STATUS_FAILED;
    }
    
    ggml_numa_dispatcher_work_context_t * ctx = (ggml_numa_dispatcher_work_context_t *)work_context;
    
    if (!ctx->operation) {
        GGML_LOG_ERROR("FLASH_ATTN_EXT work function: Operation is NULL\n");
        return GGML_STATUS_FAILED;
    }
    
    GGML_LOG_DEBUG("FLASH_ATTN_EXT work function: operation=%p, type=%d, elements=%ld\n", 
                   (void*)ctx->operation, ctx->operation->op, ggml_nelements(ctx->operation));
    
    // Flash attention has 4 source tensors: q, k, v, mask
    const struct ggml_tensor * q = ctx->operation->src[0];
    const struct ggml_tensor * k = ctx->operation->src[1]; 
    const struct ggml_tensor * v = ctx->operation->src[2];
    const struct ggml_tensor * mask = ctx->operation->src[3];
    
    if (!q || !k || !v) {
        GGML_LOG_ERROR("FLASH_ATTN_EXT work function: Missing required source tensors (q=%p, k=%p, v=%p)\n", 
                       (void*)q, (void*)k, (void*)v);
        return GGML_STATUS_FAILED;
    }
    
    // Validate tensor types - flash attention typically works with F16 or F32
    if (q->type != GGML_TYPE_F32 && q->type != GGML_TYPE_F16) {
        GGML_LOG_ERROR("FLASH_ATTN_EXT work function: Unsupported Q tensor type %d\n", q->type);
        return GGML_STATUS_FAILED;
    }
    
    if (k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_F16) {
        GGML_LOG_ERROR("FLASH_ATTN_EXT work function: Unsupported K tensor type %d\n", k->type);
        return GGML_STATUS_FAILED;
    }
    
    if (v->type != GGML_TYPE_F32 && v->type != GGML_TYPE_F16) {
        GGML_LOG_ERROR("FLASH_ATTN_EXT work function: Unsupported V tensor type %d\n", v->type);
        return GGML_STATUS_FAILED;
    }
    
    // Check data pointers
    void *q_data = ggml_get_data(q);
    void *k_data = ggml_get_data(k); 
    void *v_data = ggml_get_data(v);
    void *dst_data = ggml_get_data(ctx->operation);
    
    if (!q_data || !k_data || !v_data || !dst_data) {
        GGML_LOG_ERROR("FLASH_ATTN_EXT work function: NULL tensor data (q=%p, k=%p, v=%p, dst=%p)\n",
                       q_data, k_data, v_data, dst_data);
        return GGML_STATUS_FAILED;
    }

    // NUMA-aware data slicing for Flash Attention
    // Level 1: NUMA-level parallelism (different row ranges per NUMA node)
    // Level 2: Thread-level parallelism (subdivision within NUMA node)
    
    // Get virtual NUMA node information from coordinator's thread-local storage
    extern int ggml_numa_get_current_node(void);
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
    
    // Flash attention parallelizes by Q rows (neq1*neq2*neq3)
    const int64_t neq1 = q->ne[1];
    const int64_t neq2 = q->ne[2]; 
    const int64_t neq3 = q->ne[3];
    const int nr = neq1 * neq2 * neq3;  // total rows in q
    
    // NUMA-level data slicing: divide rows among NUMA nodes
    const int numa_rows_per_node = (nr + max_numa_nodes - 1) / max_numa_nodes;
    const int numa_start_row = numa_node * numa_rows_per_node;
    const int numa_end_row = MIN(numa_start_row + numa_rows_per_node, nr);
    const int numa_node_rows = numa_end_row - numa_start_row;
    
    if (numa_node_rows <= 0) {
        GGML_LOG_DEBUG("FLASH_ATTN_EXT NUMA node %d: No rows assigned, skipping\n", numa_node);
        return GGML_STATUS_SUCCESS;
    }
    
    GGML_LOG_DEBUG("FLASH_ATTN_EXT NUMA node %d (of %d): assigned rows %d to %d (%d rows total)\n",
                   numa_node, max_numa_nodes, numa_start_row, numa_end_row - 1, numa_node_rows);
    
    // Create modified compute params for this NUMA node's row range
    // The flash attention kernel uses params->ith and params->nth for thread-level parallelization
    // within the assigned NUMA row range
    struct ggml_compute_params numa_params = *params;
    
    // Override the row calculation in the kernel by temporarily modifying tensor dimensions
    // This is a bit tricky - we need to create tensor views for the NUMA slice
    
    // For now, use the coordinator's threading parameters directly and let the kernel
    // handle the threading within our NUMA node assignment
    GGML_LOG_DEBUG("FLASH_ATTN_EXT NUMA node %d: Using coordinator threading (ith=%d, nth=%d) for %d rows\n",
                   numa_node, params->ith, params->nth, numa_node_rows);
    
    // Call the flash attention mathematical kernel with coordinator's threading parameters
    // Note: This is a simplified approach - ideally we'd create tensor slices for the exact NUMA range
    // But flash attention is complex and would need significant refactoring for proper slicing
    ggml_compute_forward_flash_attn_ext(&numa_params, q, k, v, mask, ctx->operation);
    
    // Add memory barrier to ensure all writes are visible before returning
    __sync_synchronize();
    
    GGML_LOG_DEBUG("Successfully executed FLASH_ATTN_EXT chunk work function on NUMA node %d\n", numa_node);
    
    return GGML_STATUS_SUCCESS;
}
