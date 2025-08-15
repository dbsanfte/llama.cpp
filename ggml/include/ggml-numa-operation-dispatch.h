#pragma once

#ifndef GGML_NUMA_OPERATION_DISPATCH_H
#define GGML_NUMA_OPERATION_DISPATCH_H

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

//
// NUMA Operation Dispatcher - Public Interface
//
// This header provides the public interface for the NUMA operation dispatcher system.
// The dispatcher provides sophisticated NUMA-aware operation routing and execution.
//

struct ggml_cgraph;

/**
 * Initialize the NUMA operation dispatcher system.
 * This function is safe to call multiple times.
 */
void ggml_numa_dispatch_init(void);

/**
 * Process a computation graph through the NUMA dispatcher.
 * This is the primary interface for graph-level NUMA processing.
 * 
 * @param cgraph Computation graph to process
 * @param n_threads Number of threads to use for computation
 * @return 0 on success, -1 on failure or if NUMA mirroring is disabled
 */
int ggml_numa_dispatch_compute_graph(struct ggml_cgraph * cgraph, int n_threads);

/**
 * Check if NUMA mirroring is enabled and dispatcher should be used.
 * @return true if NUMA mirroring is enabled, false otherwise
 */
bool ggml_numa_should_mirror(void);

#ifdef __cplusplus
}
#endif

#endif // GGML_NUMA_OPERATION_DISPATCH_H
