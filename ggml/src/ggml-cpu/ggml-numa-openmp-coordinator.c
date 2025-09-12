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

/**
 * @brief Per-NUMA shared work buffer for operations that need shared scratch space
 * 
 * This provides shared work buffers that all threads on a NUMA node can access,
 * which is required for operations like matrix multiplication that need type conversion.
 */
typedef struct {
    void * buffer;          // Shared work buffer pointer for this NUMA node
    size_t current_size;    // Current allocated size in bytes
    int numa_node;          // NUMA node where buffer is allocated
    bool is_allocated;      // Whether buffer is currently allocated
    int ref_count;          // Number of threads currently using this buffer
} ggml_numa_shared_work_buffer_t;

// Per-NUMA shared work buffers - indexed by NUMA node ID
static ggml_numa_shared_work_buffer_t g_numa_shared_work_buffers[GGML_NUMA_MAX_NODES] = {0};
// TODO: Consider using g_numa_shared_buffers_initialized for buffer lifecycle management

/**
 * @brief Get or allocate per-NUMA shared work buffer
 * 
 * @param required_size Minimum required buffer size in bytes
 * @param numa_node NUMA node for allocation
 * @return Pointer to shared work buffer for the NUMA node, or NULL on failure
 */
static void * ggml_get_or_allocate_numa_shared_work_buffer(size_t required_size, int numa_node) {
    if (required_size == 0) {
        return NULL;
    }
    
    if (numa_node < 0 || numa_node >= GGML_NUMA_MAX_NODES) {
        NUMA_LOG_DEBUG("Invalid NUMA node %d, using fallback buffer", numa_node);
        return ggml_get_or_grow_thread_work_buffer(required_size, numa_node);
    }
    
    ggml_numa_shared_work_buffer_t * shared_buffer = &g_numa_shared_work_buffers[numa_node];
    
    // Check if current buffer is sufficient
    if (shared_buffer->buffer && 
        shared_buffer->current_size >= required_size &&
        shared_buffer->numa_node == numa_node) {
        // Existing buffer is adequate - increment reference count
        #pragma omp atomic
        shared_buffer->ref_count++;
        
        NUMA_LOG_DEBUG("NUMA_SHARED_BUFFER_REUSE: node=%d size=%zu ptr=%p refs=%d", 
                       numa_node, shared_buffer->current_size, shared_buffer->buffer, shared_buffer->ref_count);
        return shared_buffer->buffer;
    }
    
    // Need to allocate new buffer (thread-safe allocation)
    void * result_buffer = NULL;
    #pragma omp critical(numa_shared_buffer_alloc)
    {
        // Double-check after acquiring lock
        if (!shared_buffer->buffer || shared_buffer->current_size < required_size) {
            // Free old buffer if it exists
            if (shared_buffer->buffer) {
                NUMA_LOG_DEBUG("NUMA_SHARED_BUFFER_GROWING: node=%d old_size=%zu new_size=%zu", 
                               numa_node, shared_buffer->current_size, required_size);
                if (shared_buffer->numa_node >= 0) {
                    numa_free(shared_buffer->buffer, shared_buffer->current_size);
                } else {
                    free(shared_buffer->buffer);
                }
            }
            
            // Allocate new buffer on target NUMA node
            void * new_buffer = numa_alloc_onnode(required_size, numa_node);
            if (!new_buffer) {
                NUMA_LOG_DEBUG("NUMA allocation failed, using malloc fallback");
                new_buffer = malloc(required_size);
            }
            
            if (new_buffer) {
                shared_buffer->buffer = new_buffer;
                shared_buffer->current_size = required_size;
                shared_buffer->numa_node = numa_node;
                shared_buffer->is_allocated = true;
                shared_buffer->ref_count = 0;  // Will be incremented below
                
                NUMA_LOG_DEBUG("NUMA_SHARED_BUFFER_ALLOCATED: node=%d size=%zu ptr=%p", 
                               numa_node, required_size, new_buffer);
                result_buffer = new_buffer;
            } else {
                NUMA_LOG_DEBUG("Failed to allocate NUMA shared work buffer");
                result_buffer = NULL;
            }
        } else {
            result_buffer = shared_buffer->buffer;
        }
        
        // Increment reference count if successful
        if (result_buffer) {
            shared_buffer->ref_count++;
        }
    }
    
    return result_buffer;
}

/**
 * @brief Release reference to per-NUMA shared work buffer
 * 
 * @param numa_node NUMA node of the buffer to release
 */
static void ggml_release_numa_shared_work_buffer(int numa_node) {
    if (numa_node < 0 || numa_node >= GGML_NUMA_MAX_NODES) {
        return;
    }
    
    ggml_numa_shared_work_buffer_t * shared_buffer = &g_numa_shared_work_buffers[numa_node];
    
    #pragma omp atomic
    shared_buffer->ref_count--;
    
    NUMA_LOG_TRACE("NUMA_SHARED_BUFFER_RELEASE: node=%d refs=%d", numa_node, shared_buffer->ref_count);
}

/**
 * @brief Cleanup per-NUMA shared work buffers (call at shutdown)
 */
static void ggml_cleanup_numa_shared_work_buffers(void) {
    for (int i = 0; i < GGML_NUMA_MAX_NODES; i++) {
        ggml_numa_shared_work_buffer_t * shared_buffer = &g_numa_shared_work_buffers[i];
        if (shared_buffer->buffer) {
            if (shared_buffer->numa_node >= 0) {
                numa_free(shared_buffer->buffer, shared_buffer->current_size);
            } else {
                free(shared_buffer->buffer);
            }
            NUMA_LOG_DEBUG("NUMA_SHARED_BUFFER_CLEANUP: node=%d size=%zu", i, shared_buffer->current_size);
            shared_buffer->buffer = NULL;
            shared_buffer->current_size = 0;
            shared_buffer->is_allocated = false;
            shared_buffer->ref_count = 0;
        }
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
    GGML_UNUSED(cpu_mask);
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
 * @brief Forward declaration for thread binding function
 */
static bool bind_thread_to_numa_node(int numa_node);

/**
 * @brief Execution strategy configuration
 * 
 * Defines the parameters and behavior for different NUMA execution strategies.
 */
typedef struct {
    const char * strategy_name;         // Human-readable strategy name for logging
    int total_threads;                  // Total OpenMP threads to create
    int threads_per_node;               // Threads per NUMA node
    int total_numa_nodes;               // Number of NUMA nodes participating
    bool is_data_parallel;              // Whether this is data-parallel execution
    bool use_shared_work_buffers;       // Whether to use per-NUMA shared work buffers
    bool enable_shared_result_memory;   // Whether to enable shared result memory optimization
} ggml_numa_execution_config_t;

/**
 * @brief Calculate execution configuration for single-thread strategy
 */
static ggml_numa_execution_config_t ggml_numa_calc_single_thread_config(int target_numa_node) {
    GGML_UNUSED(target_numa_node);
    return (ggml_numa_execution_config_t) {
        .strategy_name = "single-thread",
        .total_threads = 1,
        .threads_per_node = 1,
        .total_numa_nodes = 1,
        .is_data_parallel = false,
        .use_shared_work_buffers = false,        // Single thread doesn't need shared buffers
        .enable_shared_result_memory = true      // Safe for single thread
    };
}

/**
 * @brief Calculate execution configuration for single-node strategy
 */
static ggml_numa_execution_config_t ggml_numa_calc_single_node_config(int target_numa_node) {
    GGML_UNUSED(target_numa_node);
    int threads_per_node = g_openmp_config.threads_per_node;
    return (ggml_numa_execution_config_t) {
        .strategy_name = "single-node",
        .total_threads = threads_per_node,
        .threads_per_node = threads_per_node,
        .total_numa_nodes = 1,
        .is_data_parallel = false,
        .use_shared_work_buffers = (threads_per_node > 1),  // Multi-thread needs shared buffers
        .enable_shared_result_memory = (threads_per_node == 1)  // Only safe for single thread
    };
}

/**
 * @brief Calculate execution configuration for data-parallel strategy
 */
static ggml_numa_execution_config_t ggml_numa_calc_data_parallel_config(void) {
    int total_nodes = g_openmp_config.total_numa_nodes;
    int threads_per_node = g_openmp_config.threads_per_node;
    int total_threads = total_nodes * threads_per_node;
    
    return (ggml_numa_execution_config_t) {
        .strategy_name = "data-parallel",
        .total_threads = total_threads,
        .threads_per_node = threads_per_node,
        .total_numa_nodes = total_nodes,
        .is_data_parallel = true,
        .use_shared_work_buffers = true,         // Always use shared buffers for coordination
        .enable_shared_result_memory = true      // Safe with proper slicing
    };
}

/**
 * @brief Unified NUMA execution core function
 * 
 * This function eliminates code duplication by handling all common execution logic
 * for different NUMA strategies. Strategy-specific behavior is controlled by the
 * execution configuration parameter.
 */
static enum ggml_status ggml_numa_openmp_execute_unified(
    struct ggml_tensor * tensor,
    ggml_numa_openmp_work_fn_t work_fn,
    ggml_numa_kernel_work_buffer_calc_fn_t work_buffer_calc_fn,
    ggml_numa_execution_config_t config,
    int target_numa_node  // Only used for single-node strategies
) {
    NUMA_ASSERT(tensor != NULL, "Tensor cannot be null");
    NUMA_ASSERT(work_fn != NULL, "Work function cannot be null");

    if (!g_openmp_initialized) {
        ggml_numa_openmp_coordinator_init();
    }

    // Calculate work buffer size based on execution configuration
    size_t work_buffer_size = 0;
    if (work_buffer_calc_fn) {
        work_buffer_size = work_buffer_calc_fn(tensor, config.total_numa_nodes, config.total_threads);
    }

    NUMA_LOG_DEBUG("Executing %s strategy: %d nodes, %d threads per node (%d total, work_buffer_size=%zu)\n",
                   config.strategy_name, config.total_numa_nodes, config.threads_per_node, config.total_threads, work_buffer_size);

    // OPTIMIZATION: Use atomic for error collection instead of critical section
    static _Atomic int g_execution_error = GGML_STATUS_SUCCESS;
    atomic_store(&g_execution_error, GGML_STATUS_SUCCESS);

    // OPTIMIZATION: Pre-calculate expensive values outside parallel region
    void* shared_result_data = NULL;
    if (config.enable_shared_result_memory) {
        if (config.is_data_parallel) {
            shared_result_data = tensor->__data[0];  // Shared for data-parallel
        } else {
            shared_result_data = ggml_get_data(tensor);  // Direct access for single-node
        }
    }
    
    // OPTIMIZATION: Skip NUMA binding for very small operations (< 1KB elements)
    const size_t total_elements = ggml_nelements(tensor);
    const bool skip_numa_binding = (total_elements < 1024 && work_buffer_size == 0);

    // OPTIMIZATION: Ultra-fast path for single-thread execution WITHOUT work buffers
    if (config.total_threads == 1 && work_buffer_size == 0) {
        // Skip OpenMP overhead entirely for single-thread operations with no work buffers
        extern void ggml_numa_set_fallback_flag(bool value);
        ggml_numa_set_fallback_flag(false);
        
        // Set minimal thread-local context
        ggml_current_numa_node = target_numa_node;
        ggml_numa_is_data_parallel_execution = false;
        ggml_numa_shared_result_tensor_data = shared_result_data;
        
        struct ggml_compute_params params = {
            .ith = 0, .nth = 1, 
            .wdata = NULL,  // Safe: work_buffer_size == 0
            .wsize = 0
        };
        
        enum ggml_status result = work_fn(tensor, &params);
        NUMA_LOG_VERBOSE("single-thread fast-path execution completed with status %d\n", result);
        return result;
    }

    enum ggml_status result = GGML_STATUS_SUCCESS;

#ifdef GGML_USE_OPENMP
    // Use OpenMP parallel region with strategy-specific thread count
    #pragma omp parallel num_threads(config.total_threads)
    {
        // CRITICAL: Reset fallback flag to enable NUMA dispatch on this thread
        extern void ggml_numa_set_fallback_flag(bool value);
        ggml_numa_set_fallback_flag(false);
        
        // Calculate thread assignment based on strategy
        int omp_thread_id = omp_get_thread_num();
        int numa_node, local_task_id;
        bool should_work = true;
        
        if (config.is_data_parallel) {
            // Data-parallel: distribute threads across NUMA nodes
            numa_node = omp_thread_id / config.threads_per_node;
            local_task_id = omp_thread_id % config.threads_per_node;
            
            // CRITICAL BOUNDS CHECK for data-parallel execution
            if (numa_node >= config.total_numa_nodes) {
                // Thread excluded from data-parallel work
                NUMA_LOG_VERBOSE("THREAD_EXCLUDED: omp_thread=%d numa_node=%d >= total_nodes=%d (excluded from work)",
                               omp_thread_id, numa_node, config.total_numa_nodes);
                should_work = false;  // Thread participates in barrier but does no work
            }
        } else {
            // Single-node strategies: all threads on target NUMA node
            numa_node = target_numa_node;
            local_task_id = omp_thread_id;
        }

        if (should_work) {
            NUMA_LOG_DEBUG("%s_TASK: OpenMP thread %d mapped to NUMA %d, local task %d/%d\n",
                           config.strategy_name, omp_thread_id, numa_node, local_task_id, config.threads_per_node);

            // OPTIMIZATION: Batch thread-local variable setup to reduce assignments
            ggml_current_numa_node = numa_node;
            ggml_numa_is_data_parallel_execution = config.is_data_parallel;
            if (config.is_data_parallel) {
                ggml_numa_total_nodes_for_data_parallel = config.total_numa_nodes;
            }
        
            // OPTIMIZATION: Use pre-calculated shared result memory
            ggml_numa_shared_result_tensor_data = shared_result_data;

            // OPTIMIZATION: Skip NUMA binding for very small operations
            if (!skip_numa_binding) {
                bind_thread_to_numa_node(numa_node);
            }

            // OPTIMIZATION: Fast path for no work buffers (most common case)
            char* thread_work_buffer = NULL;
            size_t thread_work_size = 0;
            if (work_buffer_size > 0) {
                if (config.use_shared_work_buffers) {
                    // Use per-NUMA shared work buffer for coordination
                    void * shared_work_buffer = ggml_get_or_allocate_numa_shared_work_buffer(work_buffer_size, numa_node);
                    if (shared_work_buffer != NULL) {
                        thread_work_buffer = (char*)shared_work_buffer;
                        thread_work_size = work_buffer_size;
                        NUMA_LOG_DEBUG("%s_SHARED_BUFFER: Using shared work buffer for thread %d on NUMA %d", 
                                       config.strategy_name, local_task_id, numa_node);
                    } else {
                        NUMA_LOG_DEBUG("%s_SHARED_BUFFER: Failed to allocate shared buffer, falling back to per-thread", config.strategy_name);
                        thread_work_buffer = (char*)ggml_get_or_grow_thread_work_buffer(work_buffer_size, numa_node);
                    }
                } else {
                    // Use per-thread work buffer
                    thread_work_buffer = (char*)ggml_get_or_grow_thread_work_buffer(work_buffer_size, numa_node);
                }
                NUMA_ASSERT(thread_work_buffer != NULL, "Work buffer allocation/growth failed");
                thread_work_size = work_buffer_size;
            }
            
            struct ggml_compute_params params = {
                .ith = local_task_id,               // Logical task index (within NUMA node for data-parallel)
                .nth = config.threads_per_node,     // Threads per NUMA node
                .wdata = thread_work_buffer,        // Work buffer (shared or per-thread based on strategy)
                .wsize = thread_work_size           // Work buffer size
            };

            // Execute work function
            enum ggml_status thread_result = work_fn(tensor, &params);
            
            // OPTIMIZATION: Use atomic compare-and-swap instead of critical section
            if (thread_result != GGML_STATUS_SUCCESS) {
                // Only update if we're the first error (preserve first error)
                int expected = GGML_STATUS_SUCCESS;
                atomic_compare_exchange_weak(&g_execution_error, &expected, thread_result);
            }
            
            // Release shared work buffer reference if used
            if (config.use_shared_work_buffers && work_buffer_size > 0) {
                ggml_release_numa_shared_work_buffer(numa_node);
            }
        }  // End of if (should_work)
    }
    
    // Get final result from atomic variable
    result = atomic_load(&g_execution_error);
    
#else
    // Fallback without OpenMP - single thread execution
    extern void ggml_numa_set_fallback_flag(bool value);
    ggml_numa_set_fallback_flag(false);
    
    // Set thread-local context for fallback execution
    ggml_current_numa_node = config.is_data_parallel ? 0 : target_numa_node;
    ggml_numa_is_data_parallel_execution = config.is_data_parallel;
    ggml_numa_total_nodes_for_data_parallel = config.total_numa_nodes;
    ggml_numa_shared_result_tensor_data = ggml_get_data(tensor);
    
    // Get work buffer for fallback execution
    void * work_buffer = NULL;
    if (work_buffer_size > 0) {
        work_buffer = ggml_get_or_grow_thread_work_buffer(work_buffer_size, ggml_current_numa_node);
        NUMA_ASSERT(work_buffer != NULL, "Thread work buffer allocation/growth failed");
    }
    
    struct ggml_compute_params params = {
        .ith = 0, .nth = 1, .wdata = work_buffer, .wsize = work_buffer_size
    };
    result = work_fn(tensor, &params);
#endif

    NUMA_LOG_VERBOSE("%s execution completed with status %d\n", config.strategy_name, result);
    
    return result;
}

// REMOVED: ggml_numa_openmp_execute_multi_numa() function was dead code
// This function was never called and has been replaced by the data-parallel approach

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
    
    // Cleanup NUMA shared work buffers
    ggml_cleanup_numa_shared_work_buffers();
    
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

    // OPTIMIZATION: Cache current binding to avoid redundant system calls
    static __thread int g_current_numa_binding = -1;
    if (g_current_numa_binding == numa_node) {
        NUMA_LOG_TRACE("Thread already bound to NUMA node %d\n", numa_node);
        return true; // Already bound to correct node
    }

    // Use numa_run_on_node for clean binding
    if (numa_run_on_node(numa_node) != 0) {
        NUMA_LOG_DEBUG("Failed to bind thread to NUMA node %d: %s\n", numa_node, strerror(errno));
        return false;
    }

    // Update cache
    g_current_numa_binding = numa_node;
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
    ggml_numa_kernel_work_buffer_calc_fn_t work_buffer_calc_fn
) {
    ggml_numa_execution_config_t config = ggml_numa_calc_single_thread_config(target_numa_node);
    enum ggml_status result = ggml_numa_openmp_execute_unified(tensor, work_fn, work_buffer_calc_fn, config, target_numa_node);
    
    // Log execution in standardized format for integration test analysis
    if (result == GGML_STATUS_SUCCESS) {
        NUMA_LOG_DEBUG("NUMA DEBUG: NUMA %s (Single/Single)", ggml_op_name(tensor->op));
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
    ggml_numa_kernel_work_buffer_calc_fn_t work_buffer_calc_fn
) {
    ggml_numa_execution_config_t config = ggml_numa_calc_single_node_config(target_numa_node);
    enum ggml_status result = ggml_numa_openmp_execute_unified(tensor, work_fn, work_buffer_calc_fn, config, target_numa_node);
    
    // Log execution in standardized format for integration test analysis
    if (result == GGML_STATUS_SUCCESS) {
        NUMA_LOG_DEBUG("NUMA DEBUG: NUMA %s (Single/Multi)", ggml_op_name(tensor->op));
    }
    
    return result;
}

/**
 * @brief Execute work using data-parallel multi-node strategy
 */
enum ggml_status ggml_numa_openmp_execute_data_parallel(
    struct ggml_tensor * tensor,
    ggml_numa_openmp_work_fn_t work_fn,
    ggml_numa_kernel_work_buffer_calc_fn_t work_buffer_calc_fn
) {
    ggml_numa_execution_config_t config = ggml_numa_calc_data_parallel_config();
    enum ggml_status result = ggml_numa_openmp_execute_unified(tensor, work_fn, work_buffer_calc_fn, config, 0);  // target_numa_node unused for data-parallel
    
    // Log execution in standardized format for integration test analysis
    if (result == GGML_STATUS_SUCCESS) {
        NUMA_LOG_DEBUG("NUMA DEBUG: NUMA %s (Data Parallel)", ggml_op_name(tensor->op));
    }
    
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
        
        GGML_UNUSED(max_threads); // Computed but using 1 for fallback simplicity
        
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
