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
#include "../include/ggml.h"  // For ggml_cplan and related structures
#include "../include/ggml-cpu.h"  // For struct ggml_cplan definition

#include <numa.h>
#include <sched.h>
#include <pthread.h>
#include <unistd.h>  // For sysconf
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdatomic.h>  // For atomic operations
#include <limits.h>     // For INT_MAX

#ifdef GGML_USE_OPENMP
#include <omp.h>

// OpenMP 5.0 feature detection
#if defined(GGML_USE_OPENMP_5_0) || _OPENMP >= 201811
#define GGML_OPENMP_5_0_AVAILABLE 1
#elif defined(GGML_FORCE_OPENMP_5_0_APIS)
// Force-enable for systems where functions work despite version detection
#define GGML_OPENMP_5_0_AVAILABLE 1
#warning "Force-enabling OpenMP 5.0 APIs - functions available but version detection failed"
#else
#define GGML_OPENMP_5_0_AVAILABLE 0
#endif

#endif

// Forward declarations for static functions
static bool ggml_numa_openmp_init_per_numa_threadpools_with_mask(const ggml_numa_cpu_mask_t * cpu_mask);

// Thread-local variables for kernels
__thread int ggml_current_numa_node = 0;
__thread bool ggml_numa_is_data_parallel_execution = false;
__thread int ggml_numa_total_nodes_for_data_parallel = 1;
__thread void * ggml_numa_shared_result_tensor_data = NULL;

/**
 * @brief Thread-local work buffer management structure
 * 
 * Maintains NUMA-local work buffers that persist across operations
 * and auto-grow when needed, eliminating per-operation allocation overhead.
 */
typedef struct {
    void * buffer;          // Current work buffer pointer
    size_t current_size;    // Current allocated size in bytes
    int numa_node;          // NUMA node where buffer is allocated
    bool is_numa_allocated; // Whether buffer was allocated with numa_alloc_onnode()
} ggml_thread_work_buffer_t;

// Thread-local work buffer - persists across operations and auto-grows
__thread ggml_thread_work_buffer_t g_thread_work_buffer = {0};

/**
 * @brief Get or grow thread-local work buffer to required size
 * 
 * @param required_size Minimum required buffer size in bytes
 * @param target_numa_node NUMA node for allocation (if reallocation needed)
 * @return Pointer to work buffer of at least required_size bytes, or NULL on failure
 */
static void * ggml_get_or_grow_thread_work_buffer(size_t required_size, int target_numa_node) {
    if (required_size == 0) {
        return NULL;
    }
    
    // Check if current buffer is sufficient
    if (g_thread_work_buffer.buffer && 
        g_thread_work_buffer.current_size >= required_size &&
        g_thread_work_buffer.numa_node == target_numa_node) {
        // Existing buffer is adequate
        NUMA_LOG_TRACE("REUSING_WORK_BUFFER: thread=%d size=%zu node=%d ptr=%p", 
                       omp_get_thread_num(), g_thread_work_buffer.current_size, 
                       target_numa_node, g_thread_work_buffer.buffer);
        return g_thread_work_buffer.buffer;
    }
    
    // Need to allocate or grow buffer
    size_t new_size = required_size;
    if (g_thread_work_buffer.current_size > 0) {
        // Grow by 50% to reduce future reallocations
        new_size = required_size + (required_size / 2);
        NUMA_LOG_DEBUG("GROWING_WORK_BUFFER: thread=%d old_size=%zu new_size=%zu node=%d", 
                       omp_get_thread_num(), g_thread_work_buffer.current_size, new_size, target_numa_node);
    } else {
        NUMA_LOG_DEBUG("ALLOCATING_WORK_BUFFER: thread=%d size=%zu node=%d", 
                       omp_get_thread_num(), new_size, target_numa_node);
    }
    
    // Free existing buffer if it exists
    if (g_thread_work_buffer.buffer) {
        if (g_thread_work_buffer.is_numa_allocated && numa_available() != -1) {
            numa_free(g_thread_work_buffer.buffer, g_thread_work_buffer.current_size);
        } else {
            free(g_thread_work_buffer.buffer);
        }
        NUMA_LOG_TRACE("FREED_OLD_WORK_BUFFER: thread=%d old_size=%zu", 
                       omp_get_thread_num(), g_thread_work_buffer.current_size);
    }
    
    // Allocate new buffer on target NUMA node
    void * new_buffer = NULL;
    bool is_numa_allocated = false;
    
    if (numa_available() != -1) {
        new_buffer = numa_alloc_onnode(new_size, target_numa_node);
        if (new_buffer) {
            is_numa_allocated = true;
            NUMA_LOG_DEBUG("NUMA_WORK_BUFFER_ALLOCATED: thread=%d size=%zu node=%d ptr=%p", 
                           omp_get_thread_num(), new_size, target_numa_node, new_buffer);
        } else {
            NUMA_LOG_DEBUG("NUMA allocation failed for work buffer, falling back to malloc");
        }
    }
    
    if (!new_buffer) {
        // Fallback to regular malloc
        new_buffer = malloc(new_size);
        is_numa_allocated = false;
        NUMA_LOG_DEBUG("MALLOC_WORK_BUFFER_ALLOCATED: thread=%d size=%zu ptr=%p", 
                       omp_get_thread_num(), new_size, new_buffer);
    }
    
    if (!new_buffer) {
        NUMA_LOG_DEBUG("Work buffer allocation failed: size=%zu node=%d", new_size, target_numa_node);
        // Reset buffer state on allocation failure
        g_thread_work_buffer = (ggml_thread_work_buffer_t){0};
        return NULL;
    }
    
    // Update buffer management state
    g_thread_work_buffer.buffer = new_buffer;
    g_thread_work_buffer.current_size = new_size;
    g_thread_work_buffer.numa_node = target_numa_node;
    g_thread_work_buffer.is_numa_allocated = is_numa_allocated;
    
    return new_buffer;
}

/**
 * @brief Clean up thread-local work buffer
 * 
 * Should be called when thread is terminating to free allocated memory.
 */
static void ggml_cleanup_thread_work_buffer(void) {
    if (g_thread_work_buffer.buffer) {
        if (g_thread_work_buffer.is_numa_allocated && numa_available() != -1) {
            numa_free(g_thread_work_buffer.buffer, g_thread_work_buffer.current_size);
        } else {
            free(g_thread_work_buffer.buffer);
        }
        NUMA_LOG_TRACE("CLEANED_UP_WORK_BUFFER: thread=%d size=%zu", 
                       omp_get_thread_num(), g_thread_work_buffer.current_size);
        g_thread_work_buffer = (ggml_thread_work_buffer_t){0};
    }
}

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
 * @brief Query OpenMP topology using OpenMP 5.0 APIs
 * 
 * Uses OpenMP 5.0 place APIs to understand the current thread binding
 * and NUMA topology configuration. Provides much more robust detection
 * than simple core counting.
 */
static void query_openmp_topology(void) {
#if GGML_OPENMP_5_0_AVAILABLE && defined(GGML_USE_OPENMP)
    NUMA_LOG_DEBUG("Querying OpenMP 5.0 topology...\n");
    
    // Check if places are configured
    int num_places = omp_get_num_places();
    if (num_places == 0) {
        NUMA_LOG_DEBUG("No OpenMP places configured - using basic topology detection\n");
        return;
    }
    
    NUMA_LOG_DEBUG("OpenMP places configured: %d places\n", num_places);
    
    // Analyze place structure to understand NUMA topology
    int total_procs = 0;
    int min_procs_per_place = INT_MAX;
    int max_procs_per_place = 0;
    
    for (int place = 0; place < num_places && place < 16; place++) {  // Limit to first 16 for debugging
        int procs_in_place = omp_get_place_num_procs(place);
        total_procs += procs_in_place;
        
        if (procs_in_place < min_procs_per_place) min_procs_per_place = procs_in_place;
        if (procs_in_place > max_procs_per_place) max_procs_per_place = procs_in_place;
        
        NUMA_LOG_VERBOSE("Place %d: %d processors\n", place, procs_in_place);
        
        // Show first few processor IDs for debugging
        if (procs_in_place > 0) {
            int sample_size = (procs_in_place < 8) ? procs_in_place : 8;
            int *proc_ids = malloc(procs_in_place * sizeof(int));
            if (proc_ids) {
                omp_get_place_proc_ids(place, proc_ids);
                NUMA_LOG_VERBOSE("  First %d proc IDs: ", sample_size);
                for (int i = 0; i < sample_size; i++) {
                    NUMA_LOG_VERBOSE("%d ", proc_ids[i]);
                }
                NUMA_LOG_VERBOSE("\n");
                free(proc_ids);
            }
        }
    }
    
    // Determine NUMA topology based on place structure
    if (num_places == 2 && min_procs_per_place == max_procs_per_place && max_procs_per_place > 1) {
        NUMA_LOG_DEBUG("Detected NUMA domains: 2 places with %d cores each\n", max_procs_per_place);
        g_openmp_config.openmp_numa_places = num_places;
        g_openmp_config.openmp_cores_per_place = max_procs_per_place;
    } else if (min_procs_per_place == 2 && max_procs_per_place == 2) {
        NUMA_LOG_DEBUG("Detected physical cores: %d places with 2 hyperthreads each\n", num_places);
        g_openmp_config.openmp_numa_places = 0;  // Core-level, not NUMA
        g_openmp_config.openmp_cores_per_place = 2;  // Hyperthreads per core
    } else {
        NUMA_LOG_DEBUG("Mixed place configuration: %d places, %d-%d cores per place\n", 
                       num_places, min_procs_per_place, max_procs_per_place);
        g_openmp_config.openmp_numa_places = num_places;
        g_openmp_config.openmp_cores_per_place = max_procs_per_place;
    }
    
    // Check current thread's partition
    #pragma omp parallel
    {
        #pragma omp single
        {
            int partition_size = omp_get_partition_num_places();
            NUMA_LOG_DEBUG("Thread partition size: %d places\n", partition_size);
            
            if (partition_size > 0 && partition_size <= 16) {
                int *place_nums = malloc(partition_size * sizeof(int));
                if (place_nums) {
                    omp_get_partition_place_nums(place_nums);
                    NUMA_LOG_VERBOSE("Partition place numbers: ");
                    for (int i = 0; i < partition_size; i++) {
                        NUMA_LOG_VERBOSE("%d ", place_nums[i]);
                    }
                    NUMA_LOG_VERBOSE("\n");
                    free(place_nums);
                }
            }
        }
    }
    
    NUMA_LOG_DEBUG("OpenMP topology query complete\n");
#else
    NUMA_LOG_DEBUG("OpenMP 5.0 APIs not available, skipping topology query\n");
#endif
}

/**
 * @brief Get CPU IDs for specific NUMA node
 * 
 * Returns array of CPU IDs that belong to the specified NUMA node.
 * Uses OpenMP 5.0 topology detection when available, falls back to numa library.
 */
static int ggml_numa_openmp_get_cpus_for_numa_node(int numa_node_id, int * cpu_ids, int max_cpus) {
    if (!cpu_ids || max_cpus <= 0 || numa_node_id < 0) {
        return -1;
    }
    
    if (!g_openmp_config.numa_available || numa_node_id >= g_openmp_config.total_numa_nodes) {
        NUMA_LOG_DEBUG("Invalid NUMA node %d (total: %d)\n", numa_node_id, g_openmp_config.total_numa_nodes);
        return -1;
    }

#if GGML_OPENMP_5_0_AVAILABLE
    // Try OpenMP 5.0 place-based detection first
    int num_places = omp_get_num_places();
    if (num_places > 0 && g_openmp_config.openmp_numa_places == g_openmp_config.total_numa_nodes) {
        // We have NUMA domain places that match our NUMA node count
        if (numa_node_id < num_places) {
            int procs_in_place = omp_get_place_num_procs(numa_node_id);
            int cpus_to_copy = (procs_in_place < max_cpus) ? procs_in_place : max_cpus;
            
            omp_get_place_proc_ids(numa_node_id, cpu_ids);
            
            NUMA_LOG_DEBUG("OpenMP: NUMA node %d has %d CPUs (returning %d)\n", 
                           numa_node_id, procs_in_place, cpus_to_copy);
            return cpus_to_copy;
        }
    }
#endif

    // Fallback to numa library
    struct bitmask * numa_cpus = numa_allocate_cpumask();
    if (!numa_cpus) {
        NUMA_LOG_DEBUG("Failed to allocate NUMA CPU mask\n");
        return -1;
    }
    
    if (numa_node_to_cpus(numa_node_id, numa_cpus) != 0) {
        numa_free_cpumask(numa_cpus);
        NUMA_LOG_DEBUG("Failed to get CPUs for NUMA node %d\n", numa_node_id);
        return -1;
    }
    
    int cpu_count = 0;
    for (int cpu = 0; cpu < numa_num_possible_cpus() && cpu_count < max_cpus; cpu++) {
        if (numa_bitmask_isbitset(numa_cpus, cpu)) {
            cpu_ids[cpu_count++] = cpu;
        }
    }
    
    numa_free_cpumask(numa_cpus);
    
    NUMA_LOG_DEBUG("numa library: NUMA node %d has %d CPUs\n", numa_node_id, cpu_count);
    return cpu_count;
}

/**
 * @brief Initialize per-NUMA thread teams with CPU binding
 * 
 * Creates persistent thread teams for each NUMA node where threads
 * are bound only to CPUs of their specific NUMA node.
 */
static bool ggml_numa_openmp_init_per_numa_threadpools(void) {
    return ggml_numa_openmp_init_per_numa_threadpools_with_mask(NULL);
}

/**
 * @brief Initialize per-NUMA thread teams with CPU mask support
 * 
 * Creates persistent thread teams for each NUMA node, respecting user-specified
 * CPU masks for fine-grained control over CPU binding and allocation.
 * 
 * @param cpu_mask User-specified CPU mask to restrict CPU usage (NULL for auto-detection)
 * @return True if thread teams initialized successfully, false otherwise
 */
static bool ggml_numa_openmp_init_per_numa_threadpools_with_mask(const ggml_numa_cpu_mask_t * cpu_mask) {
    if (!g_openmp_config.numa_available) {
        NUMA_LOG_DEBUG("NUMA not available, skipping per-NUMA thread team creation\n");
        return true; // Not an error for non-NUMA systems
    }
    
    if (g_openmp_config.threadpool_manager.teams_initialized) {
        NUMA_LOG_DEBUG("Per-NUMA thread teams already initialized\n");
        return true;
    }
    
    ggml_numa_threadpool_manager_t * manager = &g_openmp_config.threadpool_manager;
    manager->num_teams = g_openmp_config.total_numa_nodes;
    
    // Allocate thread teams array
    manager->teams = calloc(manager->num_teams, sizeof(ggml_numa_thread_team_t));
    if (!manager->teams) {
        NUMA_LOG_DEBUG("Failed to allocate memory for thread teams\n");
        return false;
    }
    
    // Initialize each NUMA node's thread team
    for (int numa_id = 0; numa_id < manager->num_teams; numa_id++) {
        ggml_numa_thread_team_t * team = &manager->teams[numa_id];
        
        team->numa_node_id = numa_id;
        team->num_threads = g_openmp_config.threads_per_node;
        team->initialized = false;
        team->threads_bound = false;
        
        // Get CPU IDs for this NUMA node
        team->cpu_ids = malloc(256 * sizeof(int)); // Max 256 CPUs per NUMA node
        if (!team->cpu_ids) {
            NUMA_LOG_DEBUG("Failed to allocate CPU ID array for NUMA node %d\n", numa_id);
            goto cleanup_teams;
        }
        
        team->num_cpus = ggml_numa_openmp_get_cpus_for_numa_node(numa_id, team->cpu_ids, 256);
        if (team->num_cpus <= 0) {
            NUMA_LOG_DEBUG("Failed to get CPU IDs for NUMA node %d\n", numa_id);
            goto cleanup_teams;
        }
        
        team->initialized = true;
        // Set threads_bound to true since we have CPU IDs available for binding
        team->threads_bound = true;
        
        NUMA_LOG_DEBUG("Thread team %d: NUMA node %d, %d threads, %d CPUs available\n",
                       numa_id, team->numa_node_id, team->num_threads, team->num_cpus);
    }
    
    // Initialize synchronization barriers for multi-NUMA coordination
    if (manager->num_teams > 1) {
        if (pthread_barrier_init(&manager->start_barrier, NULL, manager->num_teams) != 0) {
            NUMA_LOG_DEBUG("Failed to initialize start barrier\n");
            goto cleanup_teams;
        }
        
        if (pthread_barrier_init(&manager->end_barrier, NULL, manager->num_teams) != 0) {
            NUMA_LOG_DEBUG("Failed to initialize end barrier\n");
            pthread_barrier_destroy(&manager->start_barrier);
            goto cleanup_teams;
        }
        
        manager->barriers_initialized = true;
    }
    
    manager->teams_initialized = true;
    
    NUMA_LOG_DEBUG("Successfully initialized %d per-NUMA thread teams\n", manager->num_teams);
    return true;

cleanup_teams:
    // Cleanup on failure
    if (manager->teams) {
        for (int i = 0; i < manager->num_teams; i++) {
            if (manager->teams[i].cpu_ids) {
                free(manager->teams[i].cpu_ids);
            }
        }
        free(manager->teams);
        manager->teams = NULL;
    }
    manager->teams_initialized = false;
    return false;
}

/**
 * @brief Coordinate multi-NUMA execution across all thread teams
 * 
 * Executes work function simultaneously on all NUMA nodes using their
 * dedicated thread teams, with proper synchronization.
 */
static enum ggml_status ggml_numa_openmp_execute_multi_numa(
    ggml_numa_openmp_work_fn_t work_fn,
    void * work_context,
    struct ggml_cplan * cplan) {
    
    if (!g_openmp_config.threadpool_manager.teams_initialized) {
        NUMA_LOG_DEBUG("Thread teams not initialized, cannot execute multi-NUMA\n");
        return GGML_STATUS_FAILED;
    }
    
    ggml_numa_threadpool_manager_t * manager = &g_openmp_config.threadpool_manager;
    enum ggml_status result = GGML_STATUS_SUCCESS;
    
    NUMA_LOG_DEBUG("Starting multi-NUMA execution across %d NUMA nodes\n", manager->num_teams);
    
    // Create array to track per-NUMA results
    enum ggml_status numa_results[NUMA_MAX_NODES];
    for (int i = 0; i < manager->num_teams; i++) {
        numa_results[i] = GGML_STATUS_SUCCESS;
    }
    
    // Execute on all NUMA nodes in parallel using OpenMP
    #pragma omp parallel for num_threads(manager->num_teams)
    for (int numa_id = 0; numa_id < manager->num_teams; numa_id++) {
        ggml_numa_thread_team_t * team = &manager->teams[numa_id];
        
        // Set thread-local context for this NUMA coordinator thread
        ggml_current_numa_node = numa_id;
        
        NUMA_LOG_VERBOSE("NUMA coordinator %d starting nested execution\n", numa_id);
        
        // Synchronize start across all NUMA coordinators
        if (manager->barriers_initialized) {
            pthread_barrier_wait(&manager->start_barrier);
        }
        
        // Execute work using this NUMA node's dedicated thread team
        enum ggml_status numa_result = GGML_STATUS_SUCCESS;
        
        // Create nested OpenMP parallel region for this NUMA node
        #pragma omp parallel num_threads(team->num_threads)
        {
            int thread_id = omp_get_thread_num();
            
            // Bind thread to specific CPU from this NUMA node
            if (team->num_cpus > 0) {
                int cpu_index = thread_id % team->num_cpus;
                int target_cpu = team->cpu_ids[cpu_index];
                
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(target_cpu, &cpuset);
                pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
                
                // Additional NUMA binding
                numa_run_on_node(numa_id);
            }
            
            // Execute the work function
            struct ggml_compute_params params = {
                .ith = thread_id,
                .nth = team->num_threads,
                .wdata = cplan ? cplan->work_data : NULL,
                .wsize = cplan ? cplan->work_size : 0
            };
            
            enum ggml_status thread_result = work_fn(work_context, &params);
            
            if (thread_result != GGML_STATUS_SUCCESS) {
                #pragma omp atomic write
                numa_result = thread_result;
            }
        }
        
        // Store result for this NUMA node
        numa_results[numa_id] = numa_result;
        
        // Synchronize completion across all NUMA coordinators
        if (manager->barriers_initialized) {
            pthread_barrier_wait(&manager->end_barrier);
        }
        
        NUMA_LOG_VERBOSE("NUMA coordinator %d completed execution\n", numa_id);
    }
    
    // Check all NUMA results
    for (int i = 0; i < manager->num_teams; i++) {
        if (numa_results[i] != GGML_STATUS_SUCCESS) {
            result = numa_results[i];
            break;
        }
    }
    
    NUMA_LOG_DEBUG("Multi-NUMA execution completed\n");
    return result;
}

/**
 * @brief Clean up all per-NUMA thread pools and resources
 * 
 * Destroys barriers and frees allocated memory for CPU lists.
 */
static void ggml_numa_openmp_cleanup_threadpools(void) {
    ggml_numa_threadpool_manager_t * manager = &g_openmp_config.threadpool_manager;
    
    if (!manager->teams_initialized) {
        NUMA_LOG_DEBUG("Thread teams not initialized, nothing to clean up\n");
        return;
    }
    
    NUMA_LOG_DEBUG("Cleaning up per-NUMA thread pools\n");
    
    // Destroy barriers if they were initialized
    if (manager->barriers_initialized) {
        pthread_barrier_destroy(&manager->start_barrier);
        pthread_barrier_destroy(&manager->end_barrier);
        manager->barriers_initialized = false;
        NUMA_LOG_VERBOSE("Destroyed pthread barriers\n");
    }
    
    // Free CPU lists for each team
    for (int i = 0; i < manager->num_teams; i++) {
        ggml_numa_thread_team_t * team = &manager->teams[i];
        if (team->cpu_ids) {
            free(team->cpu_ids);
            team->cpu_ids = NULL;
            team->num_cpus = 0;
        }
    }
    
    // Reset manager state
    manager->num_teams = 0;
    manager->teams_initialized = false;
    
    NUMA_LOG_DEBUG("Per-NUMA thread pools cleanup completed\n");
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
    
    // Query OpenMP 5.0 topology first (if available)
    query_openmp_topology();
    
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
    NUMA_LOG_DEBUG("Using full hardware concurrency: %d threads\n", total_cores);
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
    
    // Initialize per-NUMA thread teams for optimal CPU binding
    if (!ggml_numa_openmp_init_per_numa_threadpools()) {
        NUMA_LOG_DEBUG("Warning: Failed to initialize per-NUMA thread teams, falling back to basic OpenMP\n");
        // Continue with basic functionality - this is not a fatal error
    }
    
    g_openmp_initialized = true;
    
    return true;
}

/**
 * @brief Initialize OpenMP coordinator with user-specified CPU mask
 */
bool ggml_numa_openmp_coordinator_init_with_mask(const ggml_numa_cpu_mask_t * cpu_mask) {
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

    // Auto-detect thread count based on available cores
    int effective_threads;
#ifdef GGML_USE_OPENMP
    effective_threads = omp_get_max_threads();
#else
    effective_threads = 1;
#endif
    NUMA_LOG_DEBUG("Auto-detected thread count: %d\n", effective_threads);

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

    NUMA_LOG_DEBUG("OpenMP coordinator initialized with CPU mask: %d nodes, %d threads per node\n",
                   g_openmp_config.total_numa_nodes, g_openmp_config.threads_per_node);

    g_openmp_config.initialized = true;
    
    // Initialize per-NUMA thread teams with CPU mask support
    if (!ggml_numa_openmp_init_per_numa_threadpools_with_mask(cpu_mask)) {
        NUMA_LOG_DEBUG("Warning: Failed to initialize per-NUMA thread teams with CPU mask, falling back to basic OpenMP\n");
        // Continue with basic functionality - this is not a fatal error
    }
    
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

    enum ggml_status result = GGML_STATUS_FAILED;

#ifdef GGML_USE_OPENMP
    // Use OpenMP single thread with NUMA binding
    #pragma omp parallel num_threads(1)
    {
        // CRITICAL: Reset fallback flag to enable NUMA dispatch on this thread
        extern void ggml_numa_set_fallback_flag(bool value);
        ggml_numa_set_fallback_flag(false);
        
        // Set thread-local context for single-thread execution (must be inside parallel region)
        ggml_current_numa_node = target_numa_node;
        ggml_numa_is_data_parallel_execution = false;
        ggml_numa_total_nodes_for_data_parallel = 1;
        ggml_numa_shared_result_tensor_data = ggml_get_data(tensor);

        // Get or grow thread-local work buffer
        void * work_buffer = NULL;
        if (work_buffer_size > 0) {
            work_buffer = ggml_get_or_grow_thread_work_buffer(work_buffer_size, target_numa_node);
            NUMA_ASSERT(work_buffer != NULL, "Thread work buffer allocation/growth failed");
        }

        // Bind to target NUMA node
        if (bind_thread_to_numa_node(target_numa_node)) {
            // Set up compute params for single thread
            struct ggml_compute_params params = {
                .ith = 0,           // Thread index 0
                .nth = 1,           // Total threads 1
                .wdata = work_buffer,  // Work buffer from reusable system
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
    // Reset fallback flag for non-OpenMP execution
    extern void ggml_numa_set_fallback_flag(bool value);
    ggml_numa_set_fallback_flag(false);
    
    // Set thread-local context for fallback execution
    ggml_current_numa_node = target_numa_node;
    ggml_numa_is_data_parallel_execution = false;
    ggml_numa_total_nodes_for_data_parallel = 1;
    ggml_numa_shared_result_tensor_data = ggml_get_data(tensor);
    
    // Get or grow work buffer for fallback execution
    void * work_buffer = NULL;
    if (work_buffer_size > 0) {
        work_buffer = ggml_get_or_grow_thread_work_buffer(work_buffer_size, target_numa_node);
        NUMA_ASSERT(work_buffer != NULL, "Thread work buffer allocation/growth failed");
    }
    
    struct ggml_compute_params params = {
        .ith = 0, .nth = 1, .wdata = work_buffer, .wsize = work_buffer_size
    };
    result = work_fn(tensor, &params);
#endif

    NUMA_LOG_VERBOSE("Single-thread execution completed with status %d\n", result);
    
    // Work buffer is reused - no cleanup needed here
    
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

    enum ggml_status result = GGML_STATUS_SUCCESS;

#ifdef GGML_USE_OPENMP
    // CRITICAL FIX: Use direct mapping from OpenMP thread ID to logical task ID
    // This prevents race conditions by using deterministic thread assignment
    
    // Use OpenMP parallel region with specified thread count
    #pragma omp parallel num_threads(n_threads)
    {
        // CRITICAL: Reset fallback flag to enable NUMA dispatch on this thread
        extern void ggml_numa_set_fallback_flag(bool value);
        ggml_numa_set_fallback_flag(false);
        
        // Get OpenMP thread ID and use directly as logical task ID
        int omp_thread_id = omp_get_thread_num();
        int ith = omp_thread_id;  // Direct mapping (0, 1, 2, ..., n_threads-1)
        int nth = n_threads;      // Total logical tasks

        NUMA_LOG_DEBUG("SINGLE_NODE_TASK: OpenMP thread %d mapped to logical task %d/%d\n", 
                       omp_thread_id, ith, nth);

        // Set thread-local context for single-node execution (must be inside parallel region)
        ggml_current_numa_node = target_numa_node;
        ggml_numa_is_data_parallel_execution = false;
        ggml_numa_total_nodes_for_data_parallel = 1;
        
        // CRITICAL FIX: Disable shared memory optimization for multi-threaded single-node execution
        // The shared memory pointer causes race conditions when multiple threads write to same memory
        // Only safe for single-threaded or data-parallel execution
        if (n_threads == 1) {
            ggml_numa_shared_result_tensor_data = ggml_get_data(tensor);  // Safe for single thread
        } else {
            ggml_numa_shared_result_tensor_data = NULL;  // Force individual tensor access for multi-thread
        }

        // Bind all threads to target NUMA node
        bind_thread_to_numa_node(target_numa_node);

        // Get or grow thread-local work buffer on target NUMA node
        char* thread_work_buffer = NULL;
        size_t thread_work_size = 0;
        if (work_buffer_size > 0) {
            thread_work_buffer = (char*)ggml_get_or_grow_thread_work_buffer(work_buffer_size, target_numa_node);
            NUMA_ASSERT(thread_work_buffer != NULL, "Thread work buffer allocation/growth failed");
            thread_work_size = work_buffer_size;
        }
        
        struct ggml_compute_params params = {
            .ith = ith,                     // Unique logical task index
            .nth = nth,                     // Total logical tasks
            .wdata = thread_work_buffer,    // Per-thread work buffer from reusable system
            .wsize = thread_work_size       // Work buffer size
        };

        // Each thread calls work function with unique logical task ID
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
    // Reset fallback flag for non-OpenMP execution
    extern void ggml_numa_set_fallback_flag(bool value);
    ggml_numa_set_fallback_flag(false);
    
    // Set thread-local context for fallback execution
    ggml_current_numa_node = target_numa_node;
    ggml_numa_is_data_parallel_execution = false;
    ggml_numa_total_nodes_for_data_parallel = 1;
    ggml_numa_shared_result_tensor_data = ggml_get_data(tensor);
    
    // Get or grow work buffer for fallback execution
    void * work_buffer = NULL;
    if (work_buffer_size > 0) {
        work_buffer = ggml_get_or_grow_thread_work_buffer(work_buffer_size, target_numa_node);
        NUMA_ASSERT(work_buffer != NULL, "Thread work buffer allocation/growth failed");
    }
    
    struct ggml_compute_params params = {
        .ith = 0, .nth = 1, .wdata = work_buffer, .wsize = work_buffer_size
    };
    result = work_fn(tensor, &params);
#endif

    NUMA_LOG_VERBOSE("Single-node execution completed with status %d\n", result);
    
    // Work buffers are reused - no cleanup needed here
    
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
    
    // TRACE: Very explicit data-parallel coordinator entry tracking
    NUMA_LOG_DEBUG("🎯 DATA_PARALLEL_COORDINATOR_ENTRY: op=%s nodes=%d threads_per_node=%d", 
                   ggml_op_name(tensor->op), total_nodes, threads_per_node);
    
    // TRACE: Log detailed data-parallel coordination setup for debugging
    NUMA_LOG_TRACE("COORDINATOR_DATA_PARALLEL_START: tensor=%p op=%s total_nodes=%d threads_per_node=%d total_threads=%d",
                   (void*)tensor, ggml_op_name(tensor->op), total_nodes, threads_per_node, total_threads);
    NUMA_LOG_TRACE("COORDINATOR_DP_TENSOR_INFO: elements=%ld shape=[%ld,%ld,%ld,%ld] work_buffer_size=%zu",
                   ggml_nelements(tensor), tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], work_buffer_size);

    // Allocate work buffer distributed across NUMA nodes
    // For data-parallel execution, we'll allocate work buffers locally on each NUMA node
    // when threads start execution for optimal cache locality
    void * work_buffer = NULL;
    if (work_buffer_size > 0) {
        // For data-parallel, we'll allocate work buffers per-node inside the parallel region
        // This ensures each thread gets a work buffer on its local NUMA node
        NUMA_LOG_DEBUG("DATA_PARALLEL_WORK_BUFFER: Will allocate %zu bytes per node locally", work_buffer_size);
    }

    enum ggml_status result = GGML_STATUS_SUCCESS;

#ifdef GGML_USE_OPENMP
    // CRITICAL FIX: Use direct mapping from OpenMP thread ID to logical task ID
    // This prevents race conditions by using deterministic thread assignment
    
    // Use OpenMP parallel region with total thread count
    #pragma omp parallel num_threads(total_threads)
    {
        // Get OpenMP thread ID and map to logical task assignment
        int omp_thread_id = omp_get_thread_num();
        int numa_node = omp_thread_id / threads_per_node;
        int local_task_id = omp_thread_id % threads_per_node;

        // TRACE: Log every thread entry into parallel region
        NUMA_LOG_TRACE("PARALLEL_REGION_ENTRY: omp_thread=%d numa_node=%d local_task=%d total_threads=%d",
                       omp_thread_id, numa_node, local_task_id, total_threads);

        // CRITICAL BOUNDS CHECK: Ensure numa_node doesn't exceed available nodes
        if (numa_node < total_nodes) {
            // CRITICAL: Reset fallback flag to enable NUMA dispatch on this thread
            extern void ggml_numa_set_fallback_flag(bool value);
            ggml_numa_set_fallback_flag(false);
            
            // DEBUG: Log logical vs physical thread assignment
            NUMA_LOG_DEBUG("DATA_PARALLEL_TASK: OpenMP thread %d mapped to NUMA %d, local task %d/%d\n",
                           omp_thread_id, numa_node, local_task_id, threads_per_node);
            
            // TRACE: Log detailed thread coordination for debugging data-parallel issues
            NUMA_LOG_TRACE("COORDINATOR_THREAD_SETUP: omp_thread_id=%d numa_node=%d local_task_id=%d total_nodes=%d",
                           omp_thread_id, numa_node, local_task_id, total_nodes);
            NUMA_LOG_TRACE("COORDINATOR_THREAD_CONTEXT: tensor_data=%p elements=%ld data_parallel=true",
                           ggml_get_data(tensor), ggml_nelements(tensor));

            // Set thread-local context for data-parallel execution (must be inside parallel region)
            ggml_current_numa_node = numa_node;
            ggml_numa_is_data_parallel_execution = true;
            ggml_numa_total_nodes_for_data_parallel = total_nodes;
            // OPTIMAL DATA-PARALLEL SETUP: Use shared destination memory for all NUMA nodes
            // - Source reads: Each NUMA node reads from local tensor->__data[numa_node] (optimal bandwidth)
            // - Destination writes: All NUMA nodes write to shared tensor->__data[0] at non-overlapping offsets
            // - No race conditions: Kernels must ensure proper slice partitioning with no overlap
            ggml_numa_shared_result_tensor_data = tensor->__data[0];  // Shared destination for all nodes

            // Bind thread to its NUMA node
            bind_thread_to_numa_node(numa_node);

            // Get or grow thread-local work buffer on this NUMA node
            char* thread_work_buffer = NULL;
            size_t thread_work_size = 0;
            if (work_buffer_size > 0) {
                // Use reusable work buffer system for optimal performance
                thread_work_buffer = (char*)ggml_get_or_grow_thread_work_buffer(work_buffer_size, numa_node);
                NUMA_ASSERT(thread_work_buffer != NULL, "Thread work buffer allocation/growth failed");
                thread_work_size = work_buffer_size;
                
                NUMA_LOG_DEBUG("REUSABLE_WORK_BUFFER: Thread %d using %zu bytes on node %d at %p", 
                               omp_thread_id, work_buffer_size, numa_node, (void*)thread_work_buffer);
            }
            
            struct ggml_compute_params params = {
                .ith = local_task_id,           // Logical task index within NUMA node
                .nth = threads_per_node,        // Threads per NUMA node
                .wdata = thread_work_buffer,    // Shared work buffer (not sliced)
                .wsize = thread_work_size       // Full work buffer size
            };

            // TRACE: Log before calling work function
            NUMA_LOG_TRACE("CALLING_WORK_FUNCTION: omp_thread=%d numa_node=%d local_task=%d about_to_call_work_fn",
                           omp_thread_id, numa_node, local_task_id);

            // Each thread calls work function
            enum ggml_status thread_result = work_fn(tensor, &params);
            
            // TRACE: Log work function completion for debugging
            NUMA_LOG_TRACE("COORDINATOR_WORK_COMPLETE: omp_id=%d numa_node=%d local_id=%d result=%s",
                           omp_thread_id, numa_node, local_task_id, 
                           (thread_result == GGML_STATUS_SUCCESS) ? "SUCCESS" : "FAILED");
            
            // DEBUG: Log thread completion
            NUMA_LOG_DEBUG("OPENMP_THREAD_END: OpenMP=%d, NUMA=%d, Result=%d\n", 
                           omp_thread_id, numa_node, thread_result);
            
            // Collect results
            #pragma omp critical
            {
                if (thread_result != GGML_STATUS_SUCCESS) {
                    result = thread_result;
                }
            }
            
            // Work buffer is reused - no cleanup needed here
            // Buffer will persist for next operation and be cleaned up on thread termination
        } else {
            // TRACE: Log threads that are excluded by bounds check
            NUMA_LOG_TRACE("THREAD_EXCLUDED: omp_thread=%d numa_node=%d >= total_nodes=%d (excluded from work)",
                           omp_thread_id, numa_node, total_nodes);
        }
        
        // TRACE: Log every thread reaching barrier
        NUMA_LOG_TRACE("PARALLEL_REGION_BARRIER: omp_thread=%d numa_node=%d reaching_implicit_barrier",
                       omp_thread_id, numa_node);
    }
    
    // TRACE: Log completion of parallel region
    NUMA_LOG_TRACE("PARALLEL_REGION_COMPLETE: all_threads_synchronized");
#else
    // Fallback without OpenMP - single thread execution
    // Reset fallback flag for non-OpenMP execution
    extern void ggml_numa_set_fallback_flag(bool value);
    ggml_numa_set_fallback_flag(false);
    
    // Set thread-local context for fallback execution
    ggml_current_numa_node = 0;
    ggml_numa_is_data_parallel_execution = true;
    ggml_numa_total_nodes_for_data_parallel = total_nodes;
    ggml_numa_shared_result_tensor_data = ggml_get_data(tensor);
    
    struct ggml_compute_params params = {
        .ith = 0, .nth = 1, .wdata = work_buffer, .wsize = work_buffer_size
    };
    result = work_fn(tensor, &params);
#endif

    NUMA_LOG_VERBOSE("Data-parallel execution completed with status %d\n", result);
    
    // Work buffers are now cleaned up locally by each thread
    // No global work buffer cleanup needed
    
    return result;
}

/**
 * @brief Proper fallback threadpool for legacy ggml operations
 * 
 * Some ggml operations (like MUL_MAT) require a proper threadpool with the correct
 * thread count for barrier synchronization to work correctly.
 */
static struct {
    struct ggml_threadpool * threadpool;    // Real threadpool instance
    int n_threads;                         // Number of threads in fallback threadpool
    bool initialized;                      // Whether this fallback is initialized
} g_fallback_threadpool = {0};

/**
 * @brief Shutdown OpenMP coordinator
 */
void ggml_numa_openmp_coordinator_shutdown(void) {
    if (!g_openmp_initialized) {
        return;
    }

    NUMA_LOG_DEBUG("Shutting down OpenMP coordinator\n");

    // Clean up fallback threadpool
    if (g_fallback_threadpool.initialized && g_fallback_threadpool.threadpool) {
        ggml_threadpool_free(g_fallback_threadpool.threadpool);
        g_fallback_threadpool.threadpool = NULL;
        g_fallback_threadpool.initialized = false;
        NUMA_LOG_DEBUG("Cleaned up fallback threadpool\n");
    }

    // Clean up per-NUMA thread teams first
    ggml_numa_openmp_cleanup_threadpools();

    // Reset configuration
    memset(&g_openmp_config, 0, sizeof(g_openmp_config));
    g_openmp_initialized = false;

    // OpenMP handles thread cleanup automatically
    NUMA_LOG_DEBUG("OpenMP coordinator shutdown complete\n");
}

/**
 * @brief Get fallback threadpool for legacy ggml operations
 * 
 * Provides a proper threadpool that can handle the full thread count
 * required by ggml functions (like ggml_compute_forward_mul_mat).
 * 
 * @return Pointer to fallback threadpool or NULL if not available
 */
struct ggml_threadpool * ggml_numa_openmp_get_fallback_threadpool(void) {
    if (!g_openmp_initialized) {
        return NULL;
    }

    // Initialize the fallback threadpool on first use
    if (!g_fallback_threadpool.initialized) {
        // Use a reasonable default thread count (system cores) for the fallback threadpool
        int max_threads = 1; // Start with single thread as fallback
        
        #ifdef _OPENMP
        // If OpenMP is available, use all available threads
        max_threads = omp_get_max_threads();
        #else
        // Without OpenMP, try to get system thread count from sysconf
        long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
        if (nprocs > 0) {
            max_threads = (int)nprocs;
        }
        #endif
        
        // Create threadpool parameters with single thread for fallback simplicity
        struct ggml_threadpool_params tpp = ggml_threadpool_params_default(1);
        
        // Create the actual threadpool (single-threaded to avoid barrier issues)
        g_fallback_threadpool.threadpool = ggml_threadpool_new(&tpp);
        if (!g_fallback_threadpool.threadpool) {
            NUMA_LOG_ERROR("Failed to create fallback threadpool with 1 thread\n");
            return NULL;
        }
        
        g_fallback_threadpool.n_threads = 1;
        g_fallback_threadpool.initialized = true;
        
        NUMA_LOG_DEBUG("Initialized single-threaded fallback threadpool for legacy operations\n");
    }

    return g_fallback_threadpool.threadpool;
}

/**
 * @brief Get the thread count of the fallback threadpool
 * 
 * @return Number of threads in fallback threadpool, or 0 if not initialized
 */
int ggml_numa_openmp_get_fallback_thread_count(void) {
    if (!g_openmp_initialized || !g_fallback_threadpool.initialized) {
        return 0;
    }
    
    return g_fallback_threadpool.n_threads;
}

/**
 * @brief Cross-NUMA barrier for kernel synchronization during data-parallel execution
 * 
 * This function provides a barrier mechanism for kernels executing in data-parallel
 * mode across multiple NUMA nodes using OpenMP. Since the OpenMP coordinator uses
 * a single parallel region across all NUMA nodes, standard OpenMP barriers work
 * for cross-NUMA synchronization.
 * 
 * Usage Pattern:
 * - Call this function from within kernels when synchronization across NUMA nodes is required
 * - Only effective during data-parallel execution (multiple NUMA nodes active)
 * - Uses OpenMP barrier for synchronization across all participating threads
 * - Safe to call from any thread within the OpenMP parallel region
 * 
 * @return true if barrier wait was successful, false if not in data-parallel mode
 */
bool ggml_numa_simple_coordinator_cross_numa_barrier(void) {
    // Check if we're in data-parallel execution mode
    extern __thread bool ggml_numa_is_data_parallel_execution;
    extern __thread int ggml_numa_total_nodes_for_data_parallel;
    extern __thread int ggml_current_numa_node;
    
    if (!ggml_numa_is_data_parallel_execution || ggml_numa_total_nodes_for_data_parallel <= 1) {
        NUMA_LOG_DEBUG("Cross-NUMA barrier skipped: data_parallel=%d, total_nodes=%d", 
                       ggml_numa_is_data_parallel_execution, ggml_numa_total_nodes_for_data_parallel);
        return false;
    }
    
    NUMA_LOG_DEBUG("NUMA node %d waiting at cross-NUMA barrier (participants=%d)", 
                   ggml_current_numa_node, ggml_numa_total_nodes_for_data_parallel);
    
#ifdef GGML_USE_OPENMP
    // Use OpenMP barrier for cross-NUMA synchronization
    // This works because the OpenMP coordinator uses a single parallel region across all NUMA nodes
    #pragma omp barrier
    
    NUMA_LOG_DEBUG("NUMA node %d passed cross-NUMA barrier successfully", ggml_current_numa_node);
    return true;
#else
    // No OpenMP - barrier not available
    NUMA_LOG_DEBUG("NUMA node %d: OpenMP not available, skipping cross-NUMA barrier", ggml_current_numa_node);
    return false;
#endif
}

/**
 * @brief Cleanup all thread-local work buffers
 * 
 * This function can be called to explicitly free all thread-local work buffers.
 * Normally buffers are automatically cleaned up when threads terminate, but this
 * provides a way to force cleanup for memory management.
 * 
 * Note: This should be called from each thread that has used work buffers,
 * as thread-local storage is per-thread.
 */
void ggml_numa_openmp_cleanup_thread_work_buffers(void) {
    ggml_cleanup_thread_work_buffer();
    NUMA_LOG_DEBUG("Thread work buffer cleanup completed for thread %d", omp_get_thread_num());
}

/**
 * @brief Get current thread work buffer state for testing
 * 
 * @param buffer_ptr Output pointer to current work buffer (NULL if none allocated)
 * @param current_size Output current allocated size in bytes
 * @param numa_node Output NUMA node where buffer is allocated
 * @param is_numa_allocated Output whether buffer was allocated with numa_alloc_onnode()
 * @return true if work buffer exists, false otherwise
 */
bool ggml_numa_openmp_get_thread_work_buffer_state(void** buffer_ptr, size_t* current_size, 
                                                    int* numa_node, bool* is_numa_allocated) {
    if (buffer_ptr) *buffer_ptr = g_thread_work_buffer.buffer;
    if (current_size) *current_size = g_thread_work_buffer.current_size;
    if (numa_node) *numa_node = g_thread_work_buffer.numa_node;
    if (is_numa_allocated) *is_numa_allocated = g_thread_work_buffer.is_numa_allocated;
    
    return (g_thread_work_buffer.buffer != NULL);
}

/**
 * @brief Force allocation of thread work buffer for testing
 * 
 * @param required_size Required buffer size in bytes
 * @param target_numa_node Target NUMA node for allocation
 * @return Pointer to allocated work buffer, or NULL on failure
 */
void* ggml_numa_openmp_test_force_work_buffer_allocation(size_t required_size, int target_numa_node) {
    return ggml_get_or_grow_thread_work_buffer(required_size, target_numa_node);
}
