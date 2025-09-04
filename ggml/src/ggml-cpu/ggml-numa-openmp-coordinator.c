/**
 * @file ggml-numa-openmp-coordinator.c
 * @brief OpenMP-based NUMA coordinator implementation
 * 
 * Clean OpenMP implementation that properly supports kernels expecting
 * multi-thread execution. Eliminates ggml_threadpool architectural conflicts.
 * 
 * @author David Sanftenberg
 */

#include "ggml-numa-openmp-coordinator.h"
#include "ggml-numa-shared.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"

#include <numa.h>
#include <sched.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdatomic.h>  // For atomic operations

#ifdef GGML_USE_OPENMP
#include <omp.h>
#endif

// Thread-local variables for kernels
__thread int ggml_current_numa_node = 0;
__thread bool ggml_numa_is_data_parallel_execution = false;
__thread int ggml_numa_total_nodes_for_data_parallel = 1;
__thread void * ggml_numa_shared_result_tensor_data = NULL;

// Global coordinator configuration
static ggml_numa_openmp_config_t g_openmp_config = {0};
static bool g_openmp_initialized = false;

/**
 * @brief Create a CPU mask from CPU indices
 * 
 * Helper function to create CPU masks for user-specified thread affinity.
 * Platform-specific implementation for cross-platform compatibility.
 */
ggml_numa_cpu_mask_t * ggml_numa_create_cpu_mask(const int * cpu_indices, int num_cpus) {
    if (!cpu_indices || num_cpus <= 0) {
        return NULL;
    }

    ggml_numa_cpu_mask_t * mask = malloc(sizeof(ggml_numa_cpu_mask_t));
    if (!mask) {
        return NULL;
    }

#ifdef __linux__
    cpu_set_t * cpu_set = malloc(sizeof(cpu_set_t));
    if (!cpu_set) {
        free(mask);
        return NULL;
    }

    CPU_ZERO(cpu_set);
    for (int i = 0; i < num_cpus; i++) {
        if (cpu_indices[i] >= 0 && cpu_indices[i] < CPU_SETSIZE) {
            CPU_SET(cpu_indices[i], cpu_set);
        }
    }

    mask->cpu_set_ptr = cpu_set;
    mask->max_cpus = CPU_SETSIZE;
    mask->valid = true;
#else
    // Platform not supported - create invalid mask
    mask->cpu_set_ptr = NULL;
    mask->max_cpus = 0;
    mask->valid = false;
    NUMA_LOG_DEBUG("CPU mask creation not supported on this platform\n");
#endif

    return mask;
}

/**
 * @brief Free CPU mask
 */
void ggml_numa_free_cpu_mask(ggml_numa_cpu_mask_t * mask) {
    if (!mask) {
        return;
    }

    if (mask->cpu_set_ptr) {
        free(mask->cpu_set_ptr);
    }
    free(mask);
}

/**
 * @brief Bind current thread to specific NUMA node
 * 
 * Uses numa_run_on_node() for clean NUMA binding without
 * complex CPU mask manipulation.
 */
static bool bind_thread_to_numa_node(int numa_node) {
    if (!g_openmp_config.numa_available || numa_node < 0 || numa_node >= g_openmp_config.total_numa_nodes) {
        NUMA_LOG_DEBUG("Skipping NUMA binding: node %d, available=%d, total=%d\n", 
                       numa_node, g_openmp_config.numa_available, g_openmp_config.total_numa_nodes);
        return true; // Non-NUMA systems or invalid nodes
    }

    // Use numa_run_on_node for clean binding
    if (numa_run_on_node(numa_node) != 0) {
        NUMA_LOG_DEBUG("Failed to bind thread to NUMA node %d: %s\n", numa_node, strerror(errno));
        return false;
    }

    NUMA_LOG_VERBOSE("Thread bound to NUMA node %d\n", numa_node);
    return true;
}

/**
 * @brief Initialize OpenMP coordinator
 */
bool ggml_numa_openmp_coordinator_init(void) {
    if (g_openmp_initialized) {
        return true;
    }

    // Initialize configuration
    memset(&g_openmp_config, 0, sizeof(g_openmp_config));
    
    // Check NUMA availability
    if (numa_available() == 0) {
        g_openmp_config.numa_available = true;
        g_openmp_config.total_numa_nodes = numa_max_node() + 1;
        
        NUMA_LOG_DEBUG("NUMA available: %d nodes detected\n", g_openmp_config.total_numa_nodes);
    } else {
        g_openmp_config.numa_available = false;
        g_openmp_config.total_numa_nodes = 1;
        
        NUMA_LOG_DEBUG("NUMA not available, using single node\n");
    }

    // Set threads per node based on system topology and NUMA distribution
#ifdef GGML_USE_OPENMP
    int total_cores = omp_get_max_threads();
#else
    int total_cores = 1;  // Fallback to single thread
#endif

    // Calculate threads per node with proper validation
    if (g_openmp_config.total_numa_nodes > 0) {
        g_openmp_config.threads_per_node = total_cores / g_openmp_config.total_numa_nodes;
        
        // Ensure at least 1 thread per node
        if (g_openmp_config.threads_per_node < 1) {
            g_openmp_config.threads_per_node = 1;
            NUMA_LOG_DEBUG("Warning: Insufficient cores (%d) for NUMA nodes (%d), using 1 thread per node\n",
                           total_cores, g_openmp_config.total_numa_nodes);
        }
        
        // Handle uneven distribution
        int remainder_threads = total_cores % g_openmp_config.total_numa_nodes;
        if (remainder_threads > 0) {
            NUMA_LOG_DEBUG("Note: Uneven core distribution - %d cores remain after allocating %d per node\n",
                           remainder_threads, g_openmp_config.threads_per_node);
        }
    } else {
        g_openmp_config.threads_per_node = total_cores;
    }

    NUMA_LOG_DEBUG("OpenMP coordinator initialized: %d nodes, %d threads per node\n",
                   g_openmp_config.total_numa_nodes, g_openmp_config.threads_per_node);

    g_openmp_config.initialized = true;
    g_openmp_initialized = true;
    
    return true;
}

/**
 * @brief Initialize OpenMP coordinator with user-specified CPU mask
 */
bool ggml_numa_openmp_coordinator_init_with_mask(const ggml_numa_cpu_mask_t * cpu_mask, int total_threads) {
    if (g_openmp_initialized) {
        return true;
    }

    // Initialize configuration
    memset(&g_openmp_config, 0, sizeof(g_openmp_config));
    
    // Check NUMA availability
    if (numa_available() == 0) {
        g_openmp_config.numa_available = true;
        g_openmp_config.total_numa_nodes = numa_max_node() + 1;
        
        NUMA_LOG_DEBUG("NUMA available: %d nodes detected\n", g_openmp_config.total_numa_nodes);
    } else {
        g_openmp_config.numa_available = false;
        g_openmp_config.total_numa_nodes = 1;
        
        NUMA_LOG_DEBUG("NUMA not available, using single node\n");
    }

    // Determine thread count based on user input or system detection
    int effective_threads;
    if (total_threads > 0) {
        effective_threads = total_threads;
        NUMA_LOG_DEBUG("Using user-specified thread count: %d\n", effective_threads);
    } else {
#ifdef GGML_USE_OPENMP
        effective_threads = omp_get_max_threads();
#else
        effective_threads = 1;
#endif
        NUMA_LOG_DEBUG("Auto-detected thread count: %d\n", effective_threads);
    }

    // Calculate threads per node with validation for uneven distribution
    if (g_openmp_config.total_numa_nodes > 0) {
        g_openmp_config.threads_per_node = effective_threads / g_openmp_config.total_numa_nodes;
        
        // Ensure at least 1 thread per node
        if (g_openmp_config.threads_per_node < 1) {
            g_openmp_config.threads_per_node = 1;
            NUMA_LOG_DEBUG("Warning: Insufficient threads (%d) for NUMA nodes (%d), using 1 thread per node\n",
                           effective_threads, g_openmp_config.total_numa_nodes);
        }
        
        // Handle uneven distribution
        int remainder_threads = effective_threads % g_openmp_config.total_numa_nodes;
        if (remainder_threads > 0) {
            NUMA_LOG_DEBUG("Note: Uneven thread distribution - %d threads remain after allocating %d per node\n",
                           remainder_threads, g_openmp_config.threads_per_node);
        }
    } else {
        g_openmp_config.threads_per_node = effective_threads;
    }

    // TODO: If cpu_mask is provided, use it for thread affinity
    // This would require platform-specific CPU affinity implementation
    if (cpu_mask && cpu_mask->valid) {
        NUMA_LOG_DEBUG("CPU mask provided but affinity setting not yet implemented\n");
        // Future implementation would set thread affinity based on cpu_mask
    }

    NUMA_LOG_DEBUG("OpenMP coordinator initialized with mask: %d nodes, %d threads per node\n",
                   g_openmp_config.total_numa_nodes, g_openmp_config.threads_per_node);

    g_openmp_config.initialized = true;
    g_openmp_initialized = true;
    
    return true;
}

/**
 * @brief Get coordinator configuration
 */
ggml_numa_openmp_config_t ggml_numa_openmp_coordinator_get_config(void) {
    if (!g_openmp_initialized) {
        ggml_numa_openmp_coordinator_init();
    }
    return g_openmp_config;
}

/**
 * @brief Get number of NUMA nodes from coordinator
 */
int ggml_numa_openmp_coordinator_get_num_nodes(void) {
    if (!g_openmp_initialized) {
        ggml_numa_openmp_coordinator_init();
    }
    return g_openmp_config.total_numa_nodes;
}

/**
 * @brief Execute work using single-thread strategy
 */
enum ggml_status ggml_numa_openmp_execute_single_thread(
    struct ggml_tensor * tensor,
    ggml_numa_openmp_work_fn_t work_fn,
    int target_numa_node,
    size_t work_buffer_size
) {
    NUMA_ASSERT(tensor != NULL, "Tensor cannot be null");
    NUMA_ASSERT(work_fn != NULL, "Work function cannot be null");

    if (!g_openmp_initialized) {
        ggml_numa_openmp_coordinator_init();
    }

    NUMA_LOG_DEBUG("Executing single-thread strategy on NUMA node %d (work_buffer_size=%zu)\n", target_numa_node, work_buffer_size);

    // Allocate work buffer if needed
    void * work_buffer = NULL;
    bool use_numa_alloc = false;
    if (work_buffer_size > 0) {
        work_buffer = numa_alloc_onnode(work_buffer_size, target_numa_node);
        if (work_buffer) {
            use_numa_alloc = true;
        } else {
            NUMA_LOG_DEBUG("Failed to allocate work buffer (%zu bytes) on node %d, using malloc\n", 
                          work_buffer_size, target_numa_node);
            work_buffer = malloc(work_buffer_size);
            use_numa_alloc = false;
        }
        NUMA_ASSERT(work_buffer != NULL, "Work buffer allocation failed");
    }

    // Set thread-local context for single-thread execution
    ggml_current_numa_node = target_numa_node;
    ggml_numa_is_data_parallel_execution = false;
    ggml_numa_total_nodes_for_data_parallel = 1;
    ggml_numa_shared_result_tensor_data = ggml_get_data(tensor);

    enum ggml_status result = GGML_STATUS_FAILED;

#ifdef GGML_USE_OPENMP
    // Use OpenMP single thread with NUMA binding
    #pragma omp parallel num_threads(1)
    {
        // Bind to target NUMA node
        if (bind_thread_to_numa_node(target_numa_node)) {
            // Set up compute params for single thread
            struct ggml_compute_params params = {
                .ith = 0,           // Thread index 0
                .nth = 1,           // Total threads 1
                .wdata = work_buffer,  // Work buffer allocated above
                .wsize = work_buffer_size
            };

            // Call work function
            result = work_fn(tensor, &params);
        } else {
            NUMA_LOG_DEBUG("Failed to bind to NUMA node %d, executing anyway\n", target_numa_node);
            
            struct ggml_compute_params params = {
                .ith = 0, .nth = 1, .wdata = work_buffer, .wsize = work_buffer_size
            };
            result = work_fn(tensor, &params);
        }
    }
#else
    // Fallback without OpenMP
    struct ggml_compute_params params = {
        .ith = 0, .nth = 1, .wdata = NULL, .wsize = 0
    };
    result = work_fn(tensor, &params);
#endif

    NUMA_LOG_VERBOSE("Single-thread execution completed with status %d\n", result);
    
    // Clean up work buffer
    if (work_buffer) {
        if (use_numa_alloc) {
            numa_free(work_buffer, work_buffer_size);
        } else {
            free(work_buffer);
        }
    }
    
    return result;
}

/**
 * @brief Execute work using single-node multi-thread strategy
 */
enum ggml_status ggml_numa_openmp_execute_single_node(
    struct ggml_tensor * tensor,
    ggml_numa_openmp_work_fn_t work_fn,
    int target_numa_node,
    int n_threads,
    size_t work_buffer_size
) {
    NUMA_ASSERT(tensor != NULL, "Tensor cannot be null");
    NUMA_ASSERT(work_fn != NULL, "Work function cannot be null");
    NUMA_ASSERT(n_threads > 0, "Thread count must be positive");

    if (!g_openmp_initialized) {
        ggml_numa_openmp_coordinator_init();
    }

    NUMA_LOG_DEBUG("Executing single-node strategy: %d threads on NUMA node %d (work_buffer_size=%zu)\n", n_threads, target_numa_node, work_buffer_size);

    // Allocate total work buffer for all threads
    void * work_buffer = NULL;
    bool use_numa_alloc = false;
    if (work_buffer_size > 0) {
        work_buffer = numa_alloc_onnode(work_buffer_size, target_numa_node);
        if (work_buffer) {
            use_numa_alloc = true;
        } else {
            NUMA_LOG_DEBUG("Failed to allocate work buffer (%zu bytes) on node %d, using malloc\n", 
                          work_buffer_size, target_numa_node);
            work_buffer = malloc(work_buffer_size);
            use_numa_alloc = false;
        }
        NUMA_ASSERT(work_buffer != NULL, "Work buffer allocation failed");
    }

    // Set thread-local context for single-node execution
    ggml_current_numa_node = target_numa_node;
    ggml_numa_is_data_parallel_execution = false;
    ggml_numa_total_nodes_for_data_parallel = 1;
    ggml_numa_shared_result_tensor_data = ggml_get_data(tensor);

    enum ggml_status result = GGML_STATUS_SUCCESS;

#ifdef GGML_USE_OPENMP
    // Use OpenMP parallel region with specified thread count
    #pragma omp parallel num_threads(n_threads)
    {
        // Get OpenMP thread info
        int ith = omp_get_thread_num();
        int nth = omp_get_num_threads();

        // Bind all threads to target NUMA node
        bind_thread_to_numa_node(target_numa_node);

        // Set up compute params - work buffer sharing depends on operation type
        char* thread_work_buffer = NULL;
        size_t thread_work_size = 0;
        if (work_buffer && work_buffer_size > 0) {
            // For MUL_MAT and similar operations, all threads share the same work buffer
            // For other operations, each thread gets a slice 
            // TODO: Make this configurable per operation type
            thread_work_buffer = (char*)work_buffer;  // Shared work buffer
            thread_work_size = work_buffer_size;       // Full size for each thread
        }
        
        struct ggml_compute_params params = {
            .ith = ith,                     // Unique thread index
            .nth = nth,                     // Total threads
            .wdata = thread_work_buffer,    // Shared work buffer (not sliced)
            .wsize = thread_work_size       // Full work buffer size
        };

        // Each thread calls work function with unique ith
        enum ggml_status thread_result = work_fn(tensor, &params);
        
        // Use OpenMP critical section to collect results
        #pragma omp critical
        {
            if (thread_result != GGML_STATUS_SUCCESS) {
                result = thread_result;
            }
        }
    }
#else
    // Fallback without OpenMP - single thread execution
    struct ggml_compute_params params = {
        .ith = 0, .nth = 1, .wdata = work_buffer, .wsize = work_buffer_size
    };
    result = work_fn(tensor, &params);
#endif

    NUMA_LOG_VERBOSE("Single-node execution completed with status %d\n", result);
    
    // Clean up work buffer
    if (work_buffer) {
        if (use_numa_alloc) {
            numa_free(work_buffer, work_buffer_size);
        } else {
            free(work_buffer);
        }
    }
    
    return result;
}

/**
 * @brief Execute work using data-parallel multi-node strategy
 */
enum ggml_status ggml_numa_openmp_execute_data_parallel(
    struct ggml_tensor * tensor,
    ggml_numa_openmp_work_fn_t work_fn,
    size_t work_buffer_size
) {
    NUMA_ASSERT(tensor != NULL, "Tensor cannot be null");
    NUMA_ASSERT(work_fn != NULL, "Work function cannot be null");

    if (!g_openmp_initialized) {
        ggml_numa_openmp_coordinator_init();
    }

    int total_nodes = g_openmp_config.total_numa_nodes;
    int threads_per_node = g_openmp_config.threads_per_node;
    int total_threads = total_nodes * threads_per_node;

    NUMA_LOG_DEBUG("Executing data-parallel strategy: %d nodes, %d threads per node (%d total, work_buffer_size=%zu)\n",
                   total_nodes, threads_per_node, total_threads, work_buffer_size);

    // Allocate work buffer distributed across NUMA nodes
    void * work_buffer = NULL;
    if (work_buffer_size > 0) {
        // For data-parallel, allocate total buffer and let threads access their slices
        work_buffer = malloc(work_buffer_size);  // Use malloc for simplicity across nodes
        NUMA_ASSERT(work_buffer != NULL, "Work buffer allocation failed");
    }

    // Set thread-local context for data-parallel execution
    ggml_numa_is_data_parallel_execution = true;
    ggml_numa_total_nodes_for_data_parallel = total_nodes;
    ggml_numa_shared_result_tensor_data = ggml_get_data(tensor);

    enum ggml_status result = GGML_STATUS_SUCCESS;

#ifdef GGML_USE_OPENMP
    // Use OpenMP parallel region with total thread count
    #pragma omp parallel num_threads(total_threads)
    {
        // Calculate which NUMA node this thread belongs to
        int global_thread_id = omp_get_thread_num();
        int numa_node = global_thread_id / threads_per_node;
        int local_thread_id = global_thread_id % threads_per_node;

        // Set thread-local NUMA context
        ggml_current_numa_node = numa_node;

        // Bind thread to its NUMA node
        bind_thread_to_numa_node(numa_node);

        // Set up compute params with shared work buffer
        char* thread_work_buffer = NULL;
        size_t thread_work_size = 0;
        if (work_buffer && work_buffer_size > 0) {
            // For MUL_MAT and similar operations, all threads share the same work buffer
            // For other operations, threads would get slices
            // TODO: Make this configurable per operation type  
            thread_work_buffer = (char*)work_buffer;  // Shared work buffer
            thread_work_size = work_buffer_size;       // Full size for each thread
        }
        
        struct ggml_compute_params params = {
            .ith = local_thread_id,         // Thread index within NUMA node
            .nth = threads_per_node,        // Threads per NUMA node
            .wdata = thread_work_buffer,    // Shared work buffer (not sliced)
            .wsize = thread_work_size       // Full work buffer size
        };

        // Each thread calls work function
        enum ggml_status thread_result = work_fn(tensor, &params);
        
        // Collect results
        #pragma omp critical
        {
            if (thread_result != GGML_STATUS_SUCCESS) {
                result = thread_result;
            }
        }
    }
#else
    // Fallback without OpenMP - single thread execution
    ggml_current_numa_node = 0;
    struct ggml_compute_params params = {
        .ith = 0, .nth = 1, .wdata = work_buffer, .wsize = work_buffer_size
    };
    result = work_fn(tensor, &params);
#endif

    NUMA_LOG_VERBOSE("Data-parallel execution completed with status %d\n", result);
    
    // Clean up work buffer
    if (work_buffer) {
        free(work_buffer);
    }
    
    return result;
}

/**
 * @brief Shutdown OpenMP coordinator
 */
void ggml_numa_openmp_coordinator_shutdown(void) {
    if (!g_openmp_initialized) {
        return;
    }

    NUMA_LOG_DEBUG("Shutting down OpenMP coordinator\n");

    // Reset configuration
    memset(&g_openmp_config, 0, sizeof(g_openmp_config));
    g_openmp_initialized = false;

    // OpenMP handles thread cleanup automatically
    NUMA_LOG_DEBUG("OpenMP coordinator shutdown complete\n");
}

/**
 * @brief Minimal threadpool structure for legacy ggml operations
 * 
 * Some ggml operations (like MUL_MAT) expect a valid threadpool for barrier synchronization.
 * This minimal structure provides the necessary fields without full threadpool implementation.
 */
static struct {
    // Only the essential fields needed for barrier operations
    atomic_int current_chunk;   // Required for MUL_MAT work distribution
    atomic_int n_barrier;       // Required for barrier synchronization  
    atomic_int n_barrier_passed; // Required for barrier synchronization
    bool initialized;           // Whether this fallback is initialized
} g_fallback_threadpool_minimal = {0};

/**
 * @brief Get fallback threadpool for legacy ggml operations
 * 
 * Provides a minimal threadpool structure that can handle barrier operations
 * required by some ggml functions (like ggml_compute_forward_mul_mat).
 * 
 * @return Pointer to fallback threadpool or NULL if not available
 */
struct ggml_threadpool * ggml_numa_openmp_get_fallback_threadpool(void) {
    if (!g_openmp_initialized) {
        return NULL;
    }

    // Initialize the minimal fallback on first use
    if (!g_fallback_threadpool_minimal.initialized) {
        atomic_store_explicit(&g_fallback_threadpool_minimal.current_chunk, 0, memory_order_relaxed);
        atomic_store_explicit(&g_fallback_threadpool_minimal.n_barrier, 0, memory_order_relaxed);
        atomic_store_explicit(&g_fallback_threadpool_minimal.n_barrier_passed, 0, memory_order_relaxed);
        g_fallback_threadpool_minimal.initialized = true;
        
        NUMA_LOG_DEBUG("Initialized minimal fallback threadpool for legacy operations\n");
    }

    // Cast to ggml_threadpool* - this is safe because we only access the atomic fields
    // that are at the same memory offsets in the real struct
    return (struct ggml_threadpool*)&g_fallback_threadpool_minimal;
}
