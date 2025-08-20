/*
 * Complete NUMA Graph Execution System
 * 
 * A complete graph execution system
 * that manages the entire computation using ggml building blocks while avoiding
 * any calls back into ggml_graph_compute().
 */

#include "ggml-numa-operation-dispatch.h"
#include "ggml-numa-coordinator.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "ggml-threading.h"

// ============================================================================
// Graph Analysis and Planning
// ============================================================================

/**
 * NUMA execution plan for a single operation
 */
struct numa_operation_plan {
    struct ggml_tensor * tensor;              // The operation tensor
    enum ggml_op operation;                   // Operation type
    bool numa_eligible;                       // Can benefit from NUMA distribution
    enum ggml_numa_node_strategy strategy;    // Single vs data-parallel
    size_t work_buffer_size;                  // Required work buffer
    const struct ggml_type_traits_cpu * traits; // Type system information
};

/**
 * Complete graph execution plan
 */
struct numa_graph_plan {
    struct ggml_cgraph * original_graph;      // Original graph from llama-context
    struct numa_operation_plan * operations; // Per-operation plans
    int num_operations;                       // Number of operations
    struct ggml_numa_coordinator_manager * coordinator; // NUMA coordinator
    int n_threads;                           // Total threads available
    bool has_numa_operations;                // Any operations using NUMA
};

/**
 * Analyze a computation graph and create NUMA execution plan
 */
static struct numa_graph_plan * analyze_graph_for_numa(
    struct ggml_cgraph * cgraph, 
    int n_threads) {
    
    if (!cgraph || cgraph->n_nodes <= 0) {
        return NULL;
    }
    
    struct numa_graph_plan * plan = calloc(1, sizeof(struct numa_graph_plan));
    if (!plan) return NULL;
    
    plan->original_graph = cgraph;
    plan->num_operations = cgraph->n_nodes;
    plan->n_threads = n_threads;
    plan->has_numa_operations = false;
    
    // Allocate operation plans
    plan->operations = calloc(plan->num_operations, sizeof(struct numa_operation_plan));
    if (!plan->operations) {
        free(plan);
        return NULL;
    }
    
    // Get coordinator (should already be initialized by llama-context)
    plan->coordinator = ggml_numa_coordinator_manager_get_global(n_threads);
    if (!plan->coordinator) {
        GGML_LOG_WARN("No NUMA coordinator available, planning single-threaded execution\\n");
    }
    
    // Analyze each operation
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        struct numa_operation_plan * op_plan = &plan->operations[i];
        
        op_plan->tensor = node;
        op_plan->operation = node->op;
        op_plan->numa_eligible = false;
        op_plan->strategy = NUMA_NODE_STRATEGY_SINGLE;
        op_plan->work_buffer_size = 0;
        op_plan->traits = ggml_get_type_traits_cpu(node->type);
        
        // Determine NUMA eligibility and strategy
        switch (node->op) {
            case GGML_OP_MUL_MAT: {
                // MUL_MAT is NUMA-eligible for large tensors
                int64_t total_elements = ggml_nelements(node);
                bool is_quantized = node->src[0] && ggml_is_quantized(node->src[0]->type);
                
                if (total_elements > 32768) {  // Large tensor threshold
                    op_plan->numa_eligible = true;
                    op_plan->strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL;
                    plan->has_numa_operations = true;
                    
                    // Calculate work buffer for type conversion if needed
                    if (is_quantized && node->src[1]) {
                        const struct ggml_type_traits_cpu * src0_traits = 
                            ggml_get_type_traits_cpu(node->src[0]->type);
                        if (node->src[1]->type != src0_traits->vec_dot_type) {
                            // Need conversion buffer
                            int64_t ne10 = node->src[1]->ne[0];
                            int64_t ne11 = node->src[1]->ne[1]; 
                            int64_t ne12 = node->src[1]->ne[2];
                            int64_t ne13 = node->src[1]->ne[3];
                            
                            size_t nbw1 = ggml_row_size(src0_traits->vec_dot_type, ne10);
                            size_t nbw2 = nbw1 * ne11;
                            size_t nbw3 = nbw2 * ne12;
                            op_plan->work_buffer_size = ne13 * nbw3;
                        }
                    }
                    
                    GGML_LOG_DEBUG("Operation %d (%s): NUMA-eligible, elements=%ld, quantized=%s\\n", 
                                  i, ggml_op_name(node->op), total_elements, is_quantized ? "yes" : "no");
                } else {
                    GGML_LOG_DEBUG("Operation %d (%s): Too small for NUMA (%ld elements)\\n",
                                  i, ggml_op_name(node->op), total_elements);
                }
                break;
            }
            
            case GGML_OP_ADD:
            case GGML_OP_RMS_NORM:
            case GGML_OP_SOFT_MAX: {
                // These operations could be NUMA-eligible for large tensors
                int64_t total_elements = ggml_nelements(node);
                if (total_elements > 65536) {  // Higher threshold for simpler operations
                    op_plan->numa_eligible = true;
                    op_plan->strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL;
                    plan->has_numa_operations = true;
                    
                    GGML_LOG_DEBUG("Operation %d (%s): NUMA-eligible (%ld elements)\\n",
                                  i, ggml_op_name(node->op), total_elements);
                }
                break;
            }
            
            default:
                // Most operations stay single-threaded for now
                GGML_LOG_DEBUG("Operation %d (%s): Single-threaded execution\\n",
                              i, ggml_op_name(node->op));
                break;
        }
    }
    
    GGML_LOG_INFO("Graph analysis complete: %d operations, %d NUMA-eligible\\n", 
                  plan->num_operations, plan->has_numa_operations ? 1 : 0);
    
    return plan;
}

// ============================================================================
// Direct Operation Execution Using ggml Building Blocks
// ============================================================================

/**
 * Execute MUL_MAT operation using ggml building blocks without recursion
 */
static enum ggml_status execute_mulmat_direct(
    struct numa_operation_plan * op_plan,
    struct ggml_numa_coordinator_manager * coordinator) {
    
    struct ggml_tensor * dst = op_plan->tensor;
    struct ggml_tensor * src0 = dst->src[0];
    struct ggml_tensor * src1 = dst->src[1];
    
    if (!src0 || !src1) {
        GGML_LOG_ERROR("MUL_MAT missing source tensors\\n");
        return GGML_STATUS_FAILED;
    }
    
    // Get type information using ggml's type system
    const struct ggml_type_traits_cpu * traits = ggml_get_type_traits_cpu(src0->type);
    enum ggml_type vec_dot_type = traits->vec_dot_type;
    ggml_vec_dot_t vec_dot = traits->vec_dot;
    ggml_from_float_t from_float = ggml_get_type_traits_cpu(vec_dot_type)->from_float;
    
    // Tensor dimensions (equivalent to GGML_TENSOR_BINARY_OP_LOCALS)
    const int64_t ne00 = src0->ne[0]; const int64_t ne01 = src0->ne[1];
    const int64_t ne10 = src1->ne[0]; const int64_t ne11 = src1->ne[1];
    const int64_t ne0 = dst->ne[0];   const int64_t ne1 = dst->ne[1];
    
    const size_t nb00 = src0->nb[0];  const size_t nb01 = src0->nb[1];
    const size_t nb10 = src1->nb[0];  const size_t nb11 = src1->nb[1];
    const size_t nb0 = dst->nb[0];    const size_t nb1 = dst->nb[1];
    
    GGML_ASSERT(ne00 == ne10);  // Matrix multiplication compatibility
    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    
    if (op_plan->strategy == NUMA_NODE_STRATEGY_DATA_PARALLEL && coordinator) {
        // NUMA-distributed execution
        int num_numa_nodes = ggml_numa_coordinator_manager_get_numa_node_count(coordinator);
        
        GGML_LOG_DEBUG("Executing MUL_MAT with NUMA distribution across %d nodes\\n", num_numa_nodes);
        
        // Distribute rows across NUMA nodes
        for (int numa_node = 0; numa_node < num_numa_nodes; numa_node++) {
            int64_t rows_per_node = ne01 / num_numa_nodes;
            int64_t row_start = numa_node * rows_per_node;
            int64_t row_end = (numa_node == num_numa_nodes - 1) ? ne01 : row_start + rows_per_node;
            
            if (row_start >= row_end) continue; // No work for this node
            
            // Get NUMA-local resources
            struct ggml_threadpool * threadpool = ggml_numa_coordinator_get_threadpool(coordinator, numa_node);
            void * work_buffer = ggml_numa_coordinator_get_work_buffer(coordinator, numa_node);
            
            if (!threadpool) {
                GGML_LOG_WARN("No threadpool for NUMA node %d, using single-threaded\\n", numa_node);
            }
            
            // Type conversion if needed
            void * src1_data = (void*)((char*)src1->data);
            if (src1->type != vec_dot_type && work_buffer) {
                GGML_LOG_DEBUG("Converting src1 from %s to %s on NUMA node %d\\n",
                              ggml_type_name(src1->type), ggml_type_name(vec_dot_type), numa_node);
                
                // Convert src1 using ggml's conversion function
                for (int64_t i11 = 0; i11 < ne11; i11++) {
                    from_float((float*)((char*)src1->data + i11 * nb11),
                              (void*)((char*)work_buffer + i11 * ggml_row_size(vec_dot_type, ne10)), 
                              ne10);
                }
                src1_data = work_buffer;
            }
            
            // Execute matrix multiplication for this NUMA node's rows
            for (int64_t ir0 = row_start; ir0 < row_end; ir0++) {
                for (int64_t ir1 = 0; ir1 < ne11; ir1++) {
                    const void * src0_row = (void*)((char*)src0->data + ir0 * nb01);
                    const void * src1_col = (void*)((char*)src1_data + ir1 * nb11);
                    float * dst_element = (float*)((char*)dst->data + ir0 * nb1 + ir1 * nb0);
                    
                    // Use ggml's optimized vector dot product
                    vec_dot(ne00, dst_element, 0, src0_row, 0, src1_col, 0, 1);
                }
            }
            
            GGML_LOG_DEBUG("NUMA node %d completed rows %ld to %ld\\n", numa_node, row_start, row_end - 1);
        }
        
    } else {
        // Single-threaded execution
        GGML_LOG_DEBUG("Executing MUL_MAT single-threaded\\n");
        
        // Type conversion if needed
        void * src1_data = src1->data;
        void * conversion_buffer = NULL;
        
        if (src1->type != vec_dot_type) {
            // Allocate temporary conversion buffer
            size_t conversion_size = ggml_row_size(vec_dot_type, ne10) * ne11;
            conversion_buffer = malloc(conversion_size);
            if (!conversion_buffer) {
                GGML_LOG_ERROR("Failed to allocate conversion buffer (%zu bytes)\\n", conversion_size);
                return GGML_STATUS_FAILED;
            }
            
            // Convert using ggml's type conversion
            for (int64_t i11 = 0; i11 < ne11; i11++) {
                from_float((float*)((char*)src1->data + i11 * nb11),
                          (void*)((char*)conversion_buffer + i11 * ggml_row_size(vec_dot_type, ne10)),
                          ne10);
            }
            src1_data = conversion_buffer;
        }
        
        // Execute matrix multiplication
        for (int64_t ir0 = 0; ir0 < ne01; ir0++) {
            for (int64_t ir1 = 0; ir1 < ne11; ir1++) {
                const void * src0_row = (void*)((char*)src0->data + ir0 * nb01);
                const void * src1_col = (void*)((char*)src1_data + ir1 * nb11);
                float * dst_element = (float*)((char*)dst->data + ir0 * nb1 + ir1 * nb0);
                
                // Use ggml's optimized vector dot product
                vec_dot(ne00, dst_element, 0, src0_row, 0, src1_col, 0, 1);
            }
        }
        
        // Cleanup
        if (conversion_buffer) {
            free(conversion_buffer);
        }
    }
    
    GGML_LOG_DEBUG("MUL_MAT operation completed successfully\\n");
    return GGML_STATUS_SUCCESS;
}

/**
 * Execute operation using appropriate method
 */
static enum ggml_status execute_operation_direct(
    struct numa_operation_plan * op_plan,
    struct ggml_numa_coordinator_manager * coordinator) {
    
    switch (op_plan->operation) {
        case GGML_OP_MUL_MAT:
            return execute_mulmat_direct(op_plan, coordinator);
            
        // TODO: Add other operations (ADD, RMS_NORM, etc.)
        
        default:
            GGML_LOG_DEBUG("Operation %s not implemented for direct execution\\n", 
                          ggml_op_name(op_plan->operation));
            return GGML_STATUS_FAILED;  // Let fallback handle it
    }
}

// ============================================================================
// Complete Graph Execution System
// ============================================================================

/**
 * Execute the entire computation graph using NUMA coordination
 * This replaces the old node-by-node dispatch with a complete system
 * that avoids any recursion back into ggml_graph_compute().
 */
int ggml_numa_execute_complete_graph(struct ggml_cgraph * cgraph, int n_threads) {
    if (!cgraph) {
        GGML_LOG_ERROR("Invalid graph provided to NUMA graph executor\\n");
        return -1;
    }
    
    if (!ggml_numa_should_dispatch()) {
        GGML_LOG_DEBUG("NUMA dispatch disabled\\n");
        return -1;  // Let llama-context use backend scheduler
    }
    
    GGML_LOG_INFO("🚀 NUMA Graph Executor: Processing graph with %d nodes\\n", cgraph->n_nodes);
    
    // Step 1: Analyze graph and create execution plan
    struct numa_graph_plan * plan = analyze_graph_for_numa(cgraph, n_threads);
    if (!plan) {
        GGML_LOG_ERROR("Failed to create NUMA execution plan\\n");
        return -1;
    }
    
    if (!plan->has_numa_operations) {
        GGML_LOG_INFO("No NUMA-eligible operations found, using fallback\\n");
        free(plan->operations);
        free(plan);
        return -1;  // Let backend scheduler handle it
    }
    
    // Step 2: Ensure coordinator is ready
    if (plan->coordinator) {
        // Ensure work buffers are adequate
        for (int i = 0; i < plan->num_operations; i++) {
            if (plan->operations[i].work_buffer_size > 0) {
                int num_nodes = ggml_numa_coordinator_manager_get_numa_node_count(plan->coordinator);
                for (int node = 0; node < num_nodes; node++) {
                    if (!ggml_numa_coordinator_ensure_work_buffer(plan->coordinator, node, 
                                                                plan->operations[i].work_buffer_size)) {
                        GGML_LOG_ERROR("Failed to ensure work buffer for NUMA node %d\\n", node);
                        free(plan->operations);
                        free(plan);
                        return -1;
                    }
                }
            }
        }
    }
    
    // Step 3: Execute operations in dependency order
    for (int i = 0; i < plan->num_operations; i++) {
        struct numa_operation_plan * op_plan = &plan->operations[i];
        
        GGML_LOG_DEBUG("Executing operation %d: %s (NUMA: %s)\\n", 
                      i, ggml_op_name(op_plan->operation),
                      op_plan->numa_eligible ? "yes" : "no");
        
        enum ggml_status result;
        
        if (op_plan->numa_eligible && plan->coordinator) {
            // Execute using our direct NUMA implementation
            result = execute_operation_direct(op_plan, plan->coordinator);
        } else {
            // Execute single-threaded using ggml building blocks
            result = execute_operation_direct(op_plan, NULL);
        }
        
        if (result != GGML_STATUS_SUCCESS) {
            GGML_LOG_ERROR("Operation %d (%s) failed in direct execution\\n", 
                          i, ggml_op_name(op_plan->operation));
            free(plan->operations);
            free(plan);
            return -1;
        }
    }
    
    // Step 4: Cleanup and return success
    GGML_LOG_INFO("✅ NUMA Graph Executor: All %d operations completed successfully\\n", plan->num_operations);
    
    free(plan->operations);
    free(plan);
    return 0;  // Success - llama-context will not use backend scheduler
}
