/**
 * @file ggml-numa-simple-coordinator.c
 * @brief Simple NUMA coordination system
 * 
 * Minimal NUMA coordination without complex work groups:
 * - Direct threadpool management per NUMA node
 * - Single operation execution (no graph coordination)  
 * - Proper CPU binding and thread affinity
 */

#include "ggml-numa-simple-coordinator.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "numa-kernels/numa-kernels.h"
#include "ggml-cpu.h"  // For threadpool functions
#include <stdatomic.h>  // For atomic operations
#include "ggml-numa-perf.h"  // Performance instrumentation
#include <unistd.h>    // For usleep
#include <pthread.h>   // For pthread functions
#include <time.h>      // For clock_gettime timing
#include <errno.h>     // For ETIMEDOUT

#ifdef __linux__
#include <sched.h>     // For CPU affinity functions
#endif

#ifdef GGML_NUMA_MIRROR
#include <numa.h>      // For NUMA functions
#include <numaif.h>    // For NUMA interface
#endif

#ifdef GGML_NUMA_MIRROR
#include <numa.h>
#include <sched.h>  // For sched_getcpu()
#endif

// Thread-local storage for virtual NUMA node (for simulated environments)
static __thread int g_virtual_numa_node = -1;

// Simple coordinator state with proper dispatch architecture
struct ggml_numa_simple_coordinator {
    bool initialized;
    int num_numa_nodes;
    struct ggml_threadpool * numa_threadpools[GGML_NUMA_MAX_NODES];
    int threads_per_node[GGML_NUMA_MAX_NODES];  // Track threads per node
    struct ggml_threadpool * fallback_threadpool;  // Dedicated fallback threadpool bound to NUMA node 0
    
    // Master dispatch coordination
    pthread_t dispatch_threads[GGML_NUMA_MAX_NODES];  // One dispatch thread per NUMA node
    pthread_mutex_t dispatch_mutex[GGML_NUMA_MAX_NODES];
    pthread_cond_t dispatch_cond[GGML_NUMA_MAX_NODES];
    pthread_barrier_t completion_barrier;
    
    // Work dispatch state
    _Atomic bool dispatch_work_available[GGML_NUMA_MAX_NODES];
    ggml_numa_work_function_t active_work_function;
    void * active_work_context;
    size_t active_work_size;
    _Atomic bool work_completion_status[GGML_NUMA_MAX_NODES];
    
    // NUMA-aware work buffers - one per node
    void * numa_work_buffers[GGML_NUMA_MAX_NODES];
    size_t numa_work_buffer_sizes[GGML_NUMA_MAX_NODES];
    void * fallback_work_buffer;
    size_t fallback_work_buffer_size;
};

static struct ggml_numa_simple_coordinator g_simple_coordinator = {0};

// NUMA dispatch thread worker - one per NUMA node
static void* numa_dispatch_worker(void* arg) {
    int numa_node = *(int*)arg;
    
    // Bind this dispatch thread to its NUMA node
#ifdef __linux__
    if (numa_available() >= 0) {
        struct bitmask *node_mask = numa_allocate_nodemask();
        numa_bitmask_setbit(node_mask, numa_node);
        numa_bind(node_mask);
        numa_bitmask_free(node_mask);
        
        // Also set CPU affinity to this NUMA node's CPUs
        cpu_set_t cpu_set;
        CPU_ZERO(&cpu_set);
        struct bitmask *cpu_mask = numa_allocate_cpumask();
        numa_node_to_cpus(numa_node, cpu_mask);
        for (int cpu = 0; cpu < numa_num_possible_cpus(); cpu++) {
            if (numa_bitmask_isbitset(cpu_mask, cpu)) {
                CPU_SET(cpu, &cpu_set);
            }
        }
        sched_setaffinity(0, sizeof(cpu_set), &cpu_set);
        numa_bitmask_free(cpu_mask);
    }
#endif
    
    printf("DEBUG: NUMA dispatch thread %d bound to node %d\n", numa_node, numa_node);
    
    // Main dispatch loop
    printf("DEBUG: Dispatch thread %d entering main loop (initialized=%d)\n", numa_node, g_simple_coordinator.initialized);
    while (g_simple_coordinator.initialized) {
        printf("DEBUG: Dispatch thread %d waiting for work...\n", numa_node);
        pthread_mutex_lock(&g_simple_coordinator.dispatch_mutex[numa_node]);
        
        // Wait for work to be available
        while (!atomic_load(&g_simple_coordinator.dispatch_work_available[numa_node]) && 
               g_simple_coordinator.initialized) {
            printf("DEBUG: Dispatch thread %d blocked on condition variable\n", numa_node);
            pthread_cond_wait(&g_simple_coordinator.dispatch_cond[numa_node], 
                             &g_simple_coordinator.dispatch_mutex[numa_node]);
            printf("DEBUG: Dispatch thread %d woke up from condition variable\n", numa_node);
        }
        
        pthread_mutex_unlock(&g_simple_coordinator.dispatch_mutex[numa_node]);
        
        if (!g_simple_coordinator.initialized) break;
        
        // Execute work on this NUMA node's threadpool
        if (atomic_load(&g_simple_coordinator.dispatch_work_available[numa_node])) {
            printf("DEBUG: Dispatch thread %d executing work on threadpool\n", numa_node);
            
            struct timespec work_start_time, work_end_time;
            clock_gettime(CLOCK_MONOTONIC, &work_start_time);
            
            // Set virtual NUMA node for kernel execution
            ggml_numa_set_virtual_node(numa_node);
            
            // Create work parameters for this NUMA node
            struct ggml_compute_params work_params = {
                .ith = 0,
                .nth = g_simple_coordinator.threads_per_node[numa_node],
                .wsize = g_simple_coordinator.active_work_size,
                .wdata = (g_simple_coordinator.active_work_size > 0) ? 
                         g_simple_coordinator.numa_work_buffers[numa_node] : NULL,
                .threadpool = g_simple_coordinator.numa_threadpools[numa_node]
            };
            
            // Execute the work function
            enum ggml_status result = g_simple_coordinator.active_work_function(
                g_simple_coordinator.active_work_context, &work_params);
            
            clock_gettime(CLOCK_MONOTONIC, &work_end_time);
            double work_time_ms = (work_end_time.tv_sec - work_start_time.tv_sec) * 1000.0 + 
                                  (work_end_time.tv_nsec - work_start_time.tv_nsec) / 1000000.0;
            
            // Store completion status
            atomic_store(&g_simple_coordinator.work_completion_status[numa_node], 
                        (result == GGML_STATUS_SUCCESS));
            atomic_store(&g_simple_coordinator.dispatch_work_available[numa_node], false);
            
            printf("DEBUG: Dispatch thread %d completed work with status %d (%.3fms)\n", numa_node, result, work_time_ms);
            
            // Signal completion by waiting at the barrier
            pthread_barrier_wait(&g_simple_coordinator.completion_barrier);
        }
    }
    
    printf("DEBUG: NUMA dispatch thread %d exiting\n", numa_node);
    return NULL;
}

// NUMA-aware work buffer management
static bool allocate_numa_work_buffers(size_t work_size) {
    if (work_size == 0) return true;
    
#ifdef __linux__
    if (numa_available() >= 0) {
        // Allocate work buffers on each NUMA node
        for (int node = 0; node < g_simple_coordinator.num_numa_nodes; node++) {
            if (g_simple_coordinator.numa_work_buffer_sizes[node] < work_size) {
                // Free existing buffer if too small
                if (g_simple_coordinator.numa_work_buffers[node]) {
                    numa_free(g_simple_coordinator.numa_work_buffers[node], 
                             g_simple_coordinator.numa_work_buffer_sizes[node]);
                }
                
                // Allocate new buffer on correct NUMA node
                g_simple_coordinator.numa_work_buffers[node] = numa_alloc_onnode(work_size, node);
                if (g_simple_coordinator.numa_work_buffers[node]) {
                    g_simple_coordinator.numa_work_buffer_sizes[node] = work_size;
                    printf("DEBUG: Allocated %zu bytes work buffer on NUMA node %d\n", work_size, node);
                } else {
                    printf("ERROR: Failed to allocate work buffer on NUMA node %d\n", node);
                    return false;
                }
            }
        }
        
        // Allocate fallback work buffer on NUMA node 0
        if (g_simple_coordinator.fallback_work_buffer_size < work_size) {
            if (g_simple_coordinator.fallback_work_buffer) {
                numa_free(g_simple_coordinator.fallback_work_buffer, 
                         g_simple_coordinator.fallback_work_buffer_size);
            }
            
            g_simple_coordinator.fallback_work_buffer = numa_alloc_onnode(work_size, 0);
            if (g_simple_coordinator.fallback_work_buffer) {
                g_simple_coordinator.fallback_work_buffer_size = work_size;
                printf("DEBUG: Allocated %zu bytes fallback work buffer on NUMA node 0\n", work_size);
            } else {
                printf("ERROR: Failed to allocate fallback work buffer on NUMA node 0\n");
                return false;
            }
        }
        return true;
    }
#endif
    
    // Fallback for non-NUMA systems
    return true;
}

static void free_numa_work_buffers(void) {
#ifdef __linux__
    if (numa_available() >= 0) {
        // Free NUMA node work buffers
        for (int node = 0; node < GGML_NUMA_MAX_NODES; node++) {
            if (g_simple_coordinator.numa_work_buffers[node]) {
                numa_free(g_simple_coordinator.numa_work_buffers[node], 
                         g_simple_coordinator.numa_work_buffer_sizes[node]);
                g_simple_coordinator.numa_work_buffers[node] = NULL;
                g_simple_coordinator.numa_work_buffer_sizes[node] = 0;
            }
        }
        
        // Free fallback work buffer
        if (g_simple_coordinator.fallback_work_buffer) {
            numa_free(g_simple_coordinator.fallback_work_buffer, 
                     g_simple_coordinator.fallback_work_buffer_size);
            g_simple_coordinator.fallback_work_buffer = NULL;
            g_simple_coordinator.fallback_work_buffer_size = 0;
        }
    }
#endif
}

// Helper function to create optimal CPU masks (borrowed from complex coordinator)
static void create_optimal_cpu_masks(struct ggml_threadpool_params *tpp, int num_numa_nodes) {
    if (!tpp || num_numa_nodes < 1) return;
    
    // Check if we have a custom CPU mask set
    bool has_custom_mask = false;
    int cpu_count = 0;
    
    for (int cpu = 0; cpu < GGML_MAX_N_THREADS; cpu++) {
        if (tpp->cpumask[cpu]) {
            cpu_count++;
            has_custom_mask = true;
        }
    }
    
    // If no custom mask is set, create an optimal one based on system topology
    if (!has_custom_mask) {
        // Handle auto-detection (-1) case
        int target_threads = tpp->n_threads;
        if (target_threads <= 0) {
            // Auto-detect: use all available logical CPUs
            target_threads = numa_num_configured_cpus();
            tpp->n_threads = target_threads; // Update the threadpool params
        }
        
        // Clear the mask first
        memset(tpp->cpumask, false, sizeof(tpp->cpumask));
        cpu_count = 0;
        
        // Find the actual highest CPU number on the system
        int max_cpu_found = 0;
        for (int node = 0; node < num_numa_nodes; node++) {
            struct bitmask* test_cpus = numa_allocate_cpumask();
            if (numa_node_to_cpus(node, test_cpus) == 0) {
                for (int cpu = 0; cpu < GGML_MAX_N_THREADS; cpu++) {
                    if (numa_bitmask_isbitset(test_cpus, cpu) && cpu > max_cpu_found) {
                        max_cpu_found = cpu;
                    }
                }
            }
            numa_free_cpumask(test_cpus);
        }
        int cpu_scan_limit = max_cpu_found + 1;
        
        int threads_per_node = target_threads / num_numa_nodes;
        
        for (int node = 0; node < num_numa_nodes; node++) {
            struct bitmask* node_cpus = numa_allocate_cpumask();
            if (numa_node_to_cpus(node, node_cpus) == 0) {
                int node_cpu_list[GGML_MAX_N_THREADS];
                int node_cpu_count = 0;
                
                // Collect all CPUs for this node
                for (int cpu = 0; cpu < cpu_scan_limit; cpu++) {
                    if (numa_bitmask_isbitset(node_cpus, cpu)) {
                        node_cpu_list[node_cpu_count++] = cpu;
                    }
                }
                
                // Assign threads to this node intelligently
                int assigned = 0;
                int target_threads_node = (node == num_numa_nodes - 1) ? 
                    (target_threads - (threads_per_node * (num_numa_nodes - 1))) : threads_per_node;
                
                // Sort CPUs by core topology: first physical cores, then hyperthreads
                int primary_cpus[GGML_MAX_N_THREADS];
                int hyperthread_cpus[GGML_MAX_N_THREADS];
                int primary_count = 0, hyperthread_count = 0;
                
                for (int i = 0; i < node_cpu_count; i++) {
                    int cpu = node_cpu_list[i];
                    
                    // Try to detect primary vs hyperthread by checking core_id
                    char thread_siblings_path[256];
                    snprintf(thread_siblings_path, sizeof(thread_siblings_path), 
                             "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
                    
                    FILE* siblings_file = fopen(thread_siblings_path, "r");
                    bool is_primary = true; // Default assumption
                    
                    if (siblings_file) {
                        char siblings_line[256];
                        if (fgets(siblings_line, sizeof(siblings_line), siblings_file)) {
                            // Parse thread siblings (e.g., "0,56" or "28,84")
                            // The first number in the list is typically the primary thread
                            int first_sibling = -1;
                            sscanf(siblings_line, "%d", &first_sibling);
                            is_primary = (cpu == first_sibling);
                        }
                        fclose(siblings_file);
                    }
                    
                    if (is_primary) {
                        primary_cpus[primary_count++] = cpu;
                    } else {
                        hyperthread_cpus[hyperthread_count++] = cpu;
                    }
                }
                
                // First pass: assign primary cores
                for (int i = 0; i < primary_count && assigned < target_threads_node; i++) {
                    int cpu = primary_cpus[i];
                    tpp->cpumask[cpu] = true;
                    assigned++;
                }
                
                // Second pass: assign hyperthreads if needed
                for (int i = 0; i < hyperthread_count && assigned < target_threads_node; i++) {
                    int cpu = hyperthread_cpus[i];
                    tpp->cpumask[cpu] = true;
                    assigned++;
                }
            }
            numa_free_cpumask(node_cpus);
        }
    }
}

// Initialize simple coordinator with proper CPU binding
bool ggml_numa_simple_coordinator_init(struct ggml_threadpool_params * tpp) {
    NUMA_PERF_START(NUMA_PERF_COORDINATOR_INIT, "coordinator_init", "simple_coordinator", -1, 0, tpp ? tpp->n_threads : 0);
    
    if (g_simple_coordinator.initialized) {
        NUMA_PERF_END();
        return true; // Already initialized
    }
    
    // Initialize performance measurement system
    ggml_numa_perf_init();
    ggml_numa_perf_set_enabled(true);  // Enable by default for NUMA coordinator
    
#ifdef GGML_NUMA_MIRROR
    if (numa_available() < 0) {
        return false; // NUMA not available
    }
    
    g_simple_coordinator.num_numa_nodes = numa_num_configured_nodes();
    if (g_simple_coordinator.num_numa_nodes <= 0 || g_simple_coordinator.num_numa_nodes > GGML_NUMA_MAX_NODES) {
        return false;
    }
    
    // Create optimal CPU masks using the complex coordinator logic
    struct ggml_threadpool_params optimized_tpp = *tpp;
    create_optimal_cpu_masks(&optimized_tpp, g_simple_coordinator.num_numa_nodes);
    
    // Create threadpools for each NUMA node with proper CPU filtering
    for (int node = 0; node < g_simple_coordinator.num_numa_nodes; node++) {
        // Filter the global optimized mask to only include CPUs from this NUMA node
        struct ggml_threadpool_params numa_tpp = optimized_tpp;
        
        struct bitmask *cpus = numa_allocate_cpumask();
        if (numa_node_to_cpus(node, cpus) == 0) {
            // Clear mask and filter to NUMA-local CPUs
            memset(numa_tpp.cpumask, false, sizeof(numa_tpp.cpumask));
            bool has_numa_cpus = false;
            
            for (int cpu = 0; cpu < GGML_MAX_N_THREADS; cpu++) {
                // Include CPU if it's both in the optimized mask AND on this NUMA node
                if (optimized_tpp.cpumask[cpu] && numa_bitmask_isbitset(cpus, cpu)) {
                    numa_tpp.cpumask[cpu] = true;
                    has_numa_cpus = true;
                }
            }
            
            if (!has_numa_cpus) {
                // Fallback: use original mask if filtering failed
                memcpy(numa_tpp.cpumask, optimized_tpp.cpumask, sizeof(numa_tpp.cpumask));
            }
        }
        numa_free_cpumask(cpus);
        
        // Calculate threads per node
        numa_tpp.n_threads = optimized_tpp.n_threads / g_simple_coordinator.num_numa_nodes;
        if (node == g_simple_coordinator.num_numa_nodes - 1) {
            // Last node gets remainder threads
            numa_tpp.n_threads = optimized_tpp.n_threads - (numa_tpp.n_threads * (g_simple_coordinator.num_numa_nodes - 1));
        }
        
        // Store threads per node for data-parallel execution
        g_simple_coordinator.threads_per_node[node] = numa_tpp.n_threads;
        
        numa_tpp.numa_aware = false; // Disable recursion
        numa_tpp.strict_cpu = true;  // Enable strict CPU binding
        
        g_simple_coordinator.numa_threadpools[node] = ggml_threadpool_new(&numa_tpp);
        if (!g_simple_coordinator.numa_threadpools[node]) {
            // Cleanup on failure
            for (int cleanup_node = 0; cleanup_node < node; cleanup_node++) {
                ggml_threadpool_free(g_simple_coordinator.numa_threadpools[cleanup_node]);
            }
            return false;
        }
    }
    
    // Create dedicated fallback threadpool bound to NUMA node 0
    struct ggml_threadpool_params fallback_tpp = optimized_tpp;
    
    // Configure fallback threadpool to use only NUMA node 0 CPUs
    memset(fallback_tpp.cpumask, false, sizeof(fallback_tpp.cpumask));
    struct bitmask *node0_cpus = numa_allocate_cpumask();
    if (numa_node_to_cpus(0, node0_cpus) == 0) {
        bool has_node0_cpus = false;
        for (int cpu = 0; cpu < GGML_MAX_N_THREADS; cpu++) {
            if (numa_bitmask_isbitset(node0_cpus, cpu)) {
                fallback_tpp.cpumask[cpu] = true;
                has_node0_cpus = true;
            }
        }
        
        if (!has_node0_cpus) {
            // Emergency fallback: use all available CPUs
            memcpy(fallback_tpp.cpumask, optimized_tpp.cpumask, sizeof(fallback_tpp.cpumask));
        }
    }
    numa_free_cpumask(node0_cpus);
    
    // Use NUMA node 0 thread count for fallback execution (not all threads!)
    fallback_tpp.n_threads = g_simple_coordinator.threads_per_node[0];
    fallback_tpp.numa_aware = false; // Disable NUMA recursion
    fallback_tpp.strict_cpu = true;  // Enable strict CPU binding to node 0
    
    g_simple_coordinator.fallback_threadpool = ggml_threadpool_new(&fallback_tpp);
    if (!g_simple_coordinator.fallback_threadpool) {
        // Cleanup NUMA threadpools on failure
        for (int node = 0; node < g_simple_coordinator.num_numa_nodes; node++) {
            ggml_threadpool_free(g_simple_coordinator.numa_threadpools[node]);
        }
        return false;
    }
    
    printf("DEBUG: Created dedicated fallback threadpool: %p (threads=%d, bound to NUMA node 0)\n", 
           g_simple_coordinator.fallback_threadpool, fallback_tpp.n_threads);
           
#else
    // Fallback for non-NUMA systems
    g_simple_coordinator.num_numa_nodes = 1;
    g_simple_coordinator.threads_per_node[0] = tpp->n_threads;
    g_simple_coordinator.numa_threadpools[0] = ggml_threadpool_new(tpp);
    if (!g_simple_coordinator.numa_threadpools[0]) {
        return false;
    }
    
    // For non-NUMA systems, fallback threadpool is the same as the main threadpool
    g_simple_coordinator.fallback_threadpool = g_simple_coordinator.numa_threadpools[0];
#endif

    // Initialize dispatch coordination primitives
    for (int node = 0; node < g_simple_coordinator.num_numa_nodes; node++) {
        pthread_mutex_init(&g_simple_coordinator.dispatch_mutex[node], NULL);
        pthread_cond_init(&g_simple_coordinator.dispatch_cond[node], NULL);
        atomic_init(&g_simple_coordinator.dispatch_work_available[node], false);
        atomic_init(&g_simple_coordinator.work_completion_status[node], false);
    }
    
    // Initialize completion barrier for all NUMA nodes
    pthread_barrier_init(&g_simple_coordinator.completion_barrier, NULL, 
                        g_simple_coordinator.num_numa_nodes + 1); // +1 for master thread
    
    // Set initialized flag BEFORE creating dispatch threads to avoid race condition
    g_simple_coordinator.initialized = true;
    
    // Create dispatch threads - one per NUMA node
    static int node_ids[GGML_NUMA_MAX_NODES];
    for (int node = 0; node < g_simple_coordinator.num_numa_nodes; node++) {
        node_ids[node] = node;
        int result = pthread_create(&g_simple_coordinator.dispatch_threads[node], NULL, 
                                   numa_dispatch_worker, &node_ids[node]);
        if (result != 0) {
            printf("ERROR: Failed to create dispatch thread for NUMA node %d\n", node);
            // Cleanup previously created threads
            g_simple_coordinator.initialized = false;
            for (int cleanup_node = 0; cleanup_node < node; cleanup_node++) {
                pthread_cancel(g_simple_coordinator.dispatch_threads[cleanup_node]);
                pthread_join(g_simple_coordinator.dispatch_threads[cleanup_node], NULL);
            }
            return false;
        }
    }
    
    printf("DEBUG: Created %d NUMA dispatch threads\n", g_simple_coordinator.num_numa_nodes);
    
    NUMA_PERF_END();
    return true;
}

// Cleanup simple coordinator
void ggml_numa_simple_coordinator_cleanup(void) {
    NUMA_PERF_START(NUMA_PERF_COORDINATOR_CLEANUP, "coordinator_cleanup", "simple_coordinator", -1, 0, 0);
    
    if (!g_simple_coordinator.initialized) {
        NUMA_PERF_END();
        return;
    }
    
    // Signal dispatch threads to exit
    g_simple_coordinator.initialized = false;
    
    // Wake up all dispatch threads
    for (int node = 0; node < g_simple_coordinator.num_numa_nodes; node++) {
        pthread_mutex_lock(&g_simple_coordinator.dispatch_mutex[node]);
        pthread_cond_signal(&g_simple_coordinator.dispatch_cond[node]);
        pthread_mutex_unlock(&g_simple_coordinator.dispatch_mutex[node]);
    }
    
    // Wait for dispatch threads to exit
    for (int node = 0; node < g_simple_coordinator.num_numa_nodes; node++) {
        pthread_join(g_simple_coordinator.dispatch_threads[node], NULL);
        pthread_mutex_destroy(&g_simple_coordinator.dispatch_mutex[node]);
        pthread_cond_destroy(&g_simple_coordinator.dispatch_cond[node]);
    }
    
    // Destroy completion barrier
    pthread_barrier_destroy(&g_simple_coordinator.completion_barrier);
    
    // Free work buffers
    free_numa_work_buffers();
    
    for (int node = 0; node < g_simple_coordinator.num_numa_nodes; node++) {
        if (g_simple_coordinator.numa_threadpools[node]) {
            ggml_threadpool_free(g_simple_coordinator.numa_threadpools[node]);
            g_simple_coordinator.numa_threadpools[node] = NULL;
        }
    }
    
#ifdef GGML_NUMA_MIRROR
    // Free dedicated fallback threadpool (NUMA systems only)
    if (g_simple_coordinator.fallback_threadpool && 
        g_simple_coordinator.fallback_threadpool != g_simple_coordinator.numa_threadpools[0]) {
        ggml_threadpool_free(g_simple_coordinator.fallback_threadpool);
    }
#endif
    g_simple_coordinator.fallback_threadpool = NULL;
    
    g_simple_coordinator.initialized = false;
    
    // Print performance summary before shutdown
    ggml_numa_perf_shutdown();
    NUMA_PERF_END();
}

// Execute work function on single NUMA node
enum ggml_status ggml_numa_simple_coordinator_execute_single_node(
    ggml_numa_work_function_t work_function,
    void * work_context,
    int target_node,
    size_t work_size) {
    
    if (!g_simple_coordinator.initialized) {
        return GGML_STATUS_FAILED;
    }

    if (target_node < 0 || target_node >= g_simple_coordinator.num_numa_nodes) {
        return GGML_STATUS_FAILED;
    }

    // Allocate NUMA-local work buffers if needed
    if (work_size > 0 && !allocate_numa_work_buffers(work_size)) {
        printf("ERROR: Failed to allocate NUMA work buffers\n");
        return GGML_STATUS_FAILED;
    }

    // Execute directly on target node's threadpool
    struct ggml_threadpool * threadpool = g_simple_coordinator.numa_threadpools[target_node];
    if (!threadpool) {
        return GGML_STATUS_FAILED;
    }

    printf("DEBUG: Coordinator Single-Node: Using node %d threadpool=%p, threads=%d\n", 
           target_node, threadpool, g_simple_coordinator.threads_per_node[target_node]);

    // For single-node execution, we can execute directly without coordination overhead
    // Create compute params with NUMA-local work buffer
    struct ggml_compute_params params = {
        .ith = 0,
        .nth = 1,  // Single thread for simplicity
        .wsize = work_size,
        .wdata = (work_size > 0) ? g_simple_coordinator.numa_work_buffers[target_node] : NULL,
        .threadpool = threadpool
    };    enum ggml_status result = work_function(work_context, &params);
    NUMA_PERF_END();
    return result;
}

// Execute work function across all NUMA nodes with data-parallel strategy using dispatch threads

// Removed old async task functions - using dispatch architecture instead

// Removed old async task functions - using dispatch architecture instead

// Execute work function across all NUMA nodes with data-parallel strategy using dispatch threads
enum ggml_status ggml_numa_simple_coordinator_execute_data_parallel(
    ggml_numa_work_function_t work_function,
    void * work_context,
    size_t work_size) {
    
    NUMA_PERF_START(NUMA_PERF_COORDINATOR_DISPATCH, "async_dispatch", "simple_coordinator", -1, 0, g_simple_coordinator.num_numa_nodes);
    
    if (!g_simple_coordinator.initialized) {
        NUMA_PERF_END();
        return GGML_STATUS_FAILED;
    }

    // Allocate NUMA-local work buffers if needed  
    if (work_size > 0 && !allocate_numa_work_buffers(work_size)) {
        printf("ERROR: Failed to allocate NUMA work buffers\n");
        NUMA_PERF_END();
        return GGML_STATUS_FAILED;
    }

    int num_nodes = g_simple_coordinator.num_numa_nodes;
    struct timespec coord_start_time, coord_end_time;
    clock_gettime(CLOCK_MONOTONIC, &coord_start_time);
    
    printf("DEBUG: Starting TRUE async dispatch execution across %d nodes (optimal NUMA architecture!)\n", num_nodes);
    
    // Set up work for dispatch threads
    g_simple_coordinator.active_work_function = work_function;
    g_simple_coordinator.active_work_context = work_context;
    g_simple_coordinator.active_work_size = work_size;
    
    // Signal all dispatch threads to start work IN PARALLEL
    for (int node = 0; node < num_nodes; node++) {
        pthread_mutex_lock(&g_simple_coordinator.dispatch_mutex[node]);
        atomic_store(&g_simple_coordinator.dispatch_work_available[node], true);
        pthread_cond_signal(&g_simple_coordinator.dispatch_cond[node]);
        pthread_mutex_unlock(&g_simple_coordinator.dispatch_mutex[node]);
        printf("DEBUG: Signaled dispatch thread %d to start work\n", node);
    }
    
    printf("DEBUG: Waiting for all dispatch threads to complete...\n");
    
    struct timespec barrier_start_time;
    clock_gettime(CLOCK_MONOTONIC, &barrier_start_time);
    
    // Wait at barrier for all dispatch threads to complete
    pthread_barrier_wait(&g_simple_coordinator.completion_barrier);
    
    struct timespec barrier_end_time;
    clock_gettime(CLOCK_MONOTONIC, &barrier_end_time);
    double barrier_wait_ms = (barrier_end_time.tv_sec - barrier_start_time.tv_sec) * 1000.0 + 
                             (barrier_end_time.tv_nsec - barrier_start_time.tv_nsec) / 1000000.0;
    printf("DEBUG: Barrier wait completed in %.3fms\n", barrier_wait_ms);
    
    // Collect results from all nodes
    enum ggml_status final_status = GGML_STATUS_SUCCESS;
    for (int node = 0; node < num_nodes; node++) {
        bool node_success = atomic_load(&g_simple_coordinator.work_completion_status[node]);
        printf("DEBUG: NUMA node %d completed with status %s\n", 
               node, node_success ? "SUCCESS" : "FAILED");
        if (!node_success) {
            final_status = GGML_STATUS_FAILED;
        }
    }
    
    NUMA_PERF_END();
    
    clock_gettime(CLOCK_MONOTONIC, &coord_end_time);
    double total_coord_time_ms = (coord_end_time.tv_sec - coord_start_time.tv_sec) * 1000.0 + 
                                 (coord_end_time.tv_nsec - coord_start_time.tv_nsec) / 1000000.0;
    
    printf("DEBUG: All dispatch NUMA work completed, final status: %d (total coordination time: %.3fms)\n", 
           final_status, total_coord_time_ms);
    NUMA_PERF_END();
    return final_status;
}

// Get number of available NUMA nodes
int ggml_numa_simple_coordinator_get_num_nodes(void) {
    return g_simple_coordinator.initialized ? g_simple_coordinator.num_numa_nodes : 0;
}

// Check if coordinator is initialized
bool ggml_numa_simple_coordinator_is_initialized(void) {
    return g_simple_coordinator.initialized;
}

// NUMA node detection functions (moved from complex coordinator)
void ggml_numa_set_virtual_node(int node) {
    g_virtual_numa_node = node;
}

int ggml_numa_get_current_node(void) {
    // First check if we have a virtual node set (for data-parallel execution)
    if (g_virtual_numa_node >= 0) {
        printf("DEBUG: Using virtual NUMA node %d\n", g_virtual_numa_node);
        return g_virtual_numa_node;
    }
    
#ifdef GGML_NUMA_MIRROR
    if (numa_available() >= 0) {
        // Get current CPU and determine its NUMA node
        int current_cpu = sched_getcpu();
        if (current_cpu >= 0) {
            int current_node = numa_node_of_cpu(current_cpu);
            if (current_node >= 0) {
                printf("DEBUG: Using physical NUMA node %d (CPU %d)\n", current_node, current_cpu);
                return current_node;
            }
        }
    }
#endif

    // Fallback: return node 0 if detection fails
    printf("DEBUG: Using fallback NUMA node 0\n");
    return 0;
}

// Get the dedicated fallback threadpool
struct ggml_threadpool * ggml_numa_simple_coordinator_get_fallback_threadpool(void) {
    if (!g_simple_coordinator.initialized) {
        return NULL;
    }
    return g_simple_coordinator.fallback_threadpool;
}

int ggml_numa_simple_coordinator_get_fallback_thread_count(void) {
    if (!g_simple_coordinator.initialized) {
        return 1;
    }
    return g_simple_coordinator.threads_per_node[0];
}
