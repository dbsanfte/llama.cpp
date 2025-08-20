/*
 * NUMA Graph Executor - Complete Graph Computation System
 * 
 * This header declares the complete graph execution system that replaces
 * the operation-by-operation dispatch to avoid recursion issues.
 * 
 * The system analyzes entire computation graphs, creates NUMA execution plans,
 * and executes operations directly using ggml building blocks without
 * calling back into ggml_graph_compute().
 */

#ifndef GGML_NUMA_GRAPH_EXECUTOR_H
#define GGML_NUMA_GRAPH_EXECUTOR_H

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Execute a complete computation graph using NUMA coordination
 * 
 * This function provides a recursion-free implementation that:
 * 
 * 1. Analyzes the entire graph for NUMA opportunities
 * 2. Creates an execution plan with operation dependencies
 * 3. Executes operations directly using ggml building blocks
 * 4. Avoids any calls to ggml_graph_compute() to prevent recursion
 * 
 * @param cgraph The computation graph to execute
 * @param n_threads Number of threads available for computation
 * @return 0 on success (graph fully executed), 
 *         -1 on failure or if fallback should be used
 */
int ggml_numa_execute_complete_graph(struct ggml_cgraph * cgraph, int n_threads);

/**
 * Execute a complete computation graph using NUMA coordination
 * 
 * This function completely replaces the old ggml_numa_dispatch_compute_graph()
 * implementation and provides the same interface for llama-context.cpp
 * 
 * @param cgraph The computation graph to execute
 * @param n_threads Number of threads available for computation
 * @return 0 on success (graph fully executed), 
 *         -1 on failure or if fallback should be used
 */
int ggml_numa_dispatch_compute_graph(struct ggml_cgraph * cgraph, int n_threads);

#ifdef __cplusplus
}
#endif

#endif // GGML_NUMA_GRAPH_EXECUTOR_H
