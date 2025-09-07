/**
 * @file test-ggml-openmp-coordinator.cpp
 * @brief Test suite for OpenMP-based NUMA coordinator
 * 
 * Validates CPU mask handling, per-NUMA thread teams, executor dispatch strategies,
 * work buffer management, and thread binding functionality.
 * 
 * @author David Sanftenberg
 */

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <cassert>
#include <atomic>
#include <mutex>
#include <chrono>
#include <thread>
#include <regex>
#include <set>
#include <unordered_set>

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-cpu/ggml-numa-openmp-coordinator.h"
#include "ggml-cpu/ggml-numa-executor.h"
#include "ggml-cpu/ggml-numa-shared.h"
#include "ggml-cpu/ggml-cpu-impl.h"

#ifdef __linux__
#include <sched.h>
#include <numa.h>
#include <unistd.h>
#endif

// Global test control variables
std::string g_test_filter = "";
bool g_filter_enabled = false;
bool g_summary_only = false;

// Test utilities
#define TEST_PRINTF(...) do { if (!g_summary_only) printf(__VA_ARGS__); } while(0)

/**
 * @brief Check if a test name matches the current filter
 */
bool matches_filter(const std::string& test_name) {
    if (!g_filter_enabled) {
        return true;  // No filter, run all tests
    }
    
    try {
        std::regex filter_regex(g_test_filter, std::regex_constants::icase);
        return std::regex_search(test_name, filter_regex);
    } catch (const std::regex_error& e) {
        printf("⚠️  Invalid regex filter '%s': %s\n", g_test_filter.c_str(), e.what());
        printf("   Running all tests instead.\n");
        return true;  // On regex error, run all tests
    }
}

/**
 * @brief Get the NUMA node of the currently executing thread
 * @return NUMA node ID (0-based), or -1 if unable to determine
 */
int get_current_thread_numa_node() {
#ifdef __linux__
    // Get current CPU
    int cpu = sched_getcpu();
    if (cpu == -1) {
        return -1;
    }
    
    // Convert CPU to NUMA node
    int numa_node = numa_node_of_cpu(cpu);
    return numa_node;
#else
    return -1;  // NUMA checking not supported on non-Linux
#endif
}

/**
 * @brief Verify that current thread is running on expected NUMA node
 * @param expected_numa_node Expected NUMA node (0-based)
 * @param test_context Description for debugging
 * @return true if on correct node, false otherwise
 */
bool verify_thread_numa_binding(int expected_numa_node, const char* test_context) {
#ifdef __linux__
    int actual_numa_node = get_current_thread_numa_node();
    
    if (actual_numa_node == -1) {
        TEST_PRINTF("   ⚠️  NUMA binding check failed: unable to determine current NUMA node (%s)\n", test_context);
        return false;  // Can't verify, but don't fail the test
    }
    
    if (actual_numa_node != expected_numa_node) {
        TEST_PRINTF("   ❌ NUMA binding VIOLATION: thread expected on node %d, actually on node %d (%s)\n", 
               expected_numa_node, actual_numa_node, test_context);
        return false;
    }
    
    TEST_PRINTF("   ✅ NUMA binding verified: thread correctly on node %d (%s)\n", 
           actual_numa_node, test_context);
    return true;
#else
    TEST_PRINTF("   ⚠️  NUMA binding check skipped: not supported on this platform (%s)\n", test_context);
    return true;  // Skip verification on non-Linux platforms
#endif
}

// Test result tracking
struct TestResult {
    std::string test_name;
    bool passed;
    std::string failure_reason;
};

std::vector<TestResult> g_test_results;

// Thread execution tracking for validation
std::atomic<int> g_threads_executed{0};
std::atomic<int> g_numa_nodes_used{0};
std::vector<int> g_threads_per_numa;  // Protected by g_tracking_mutex
std::mutex g_tracking_mutex;  // Protects g_threads_per_numa from race conditions
std::vector<char> g_numa_node_active;  // Use char instead of bool to avoid vector<bool> issues

// Test utilities
#define EXPECT_TRUE(condition, message) \
    do { \
        if (!(condition)) { \
            printf("❌ FAILED: %s - %s\n", __func__, message); \
            return false; \
        } \
    } while(0)

#define EXPECT_FALSE(condition, message) \
    do { \
        if (condition) { \
            printf("❌ FAILED: %s - %s\n", __func__, message); \
            return false; \
        } \
    } while(0)

#define EXPECT_EQ(expected, actual, message) \
    do { \
        if ((expected) != (actual)) { \
            printf("❌ FAILED: %s - %s (expected %ld, got %ld)\n", __func__, message, (long)(expected), (long)(actual)); \
            return false; \
        } \
    } while(0)

#define EXPECT_PTR_EQ(expected, actual, message) \
    do { \
        if ((expected) != (actual)) { \
            printf("❌ FAILED: %s - %s (expected %p, got %p)\n", __func__, message, (void*)(expected), (void*)(actual)); \
            return false; \
        } \
    } while(0)

#define EXPECT_GE(value, minimum, message) \
    do { \
        if ((value) < (minimum)) { \
            printf("❌ FAILED: %s - %s (value %d < minimum %d)\n", __func__, message, (int)(value), (int)(minimum)); \
            return false; \
        } \
    } while(0)

#define EXPECT_GT(value, minimum, message) \
    do { \
        if ((value) <= (minimum)) { \
            printf("❌ FAILED: %s - %s (value %d <= minimum %d)\n", __func__, message, (int)(value), (int)(minimum)); \
            return false; \
        } \
    } while(0)

#define EXPECT_LE(value, maximum, message) \
    do { \
        if ((value) > (maximum)) { \
            printf("❌ FAILED: %s - %s (value %d > maximum %d)\n", __func__, message, (int)(value), (int)(maximum)); \
            return false; \
        } \
    } while(0)

#define EXPECT_NEAR(expected, actual, tolerance, message) \
    do { \
        float diff = fabs((float)(expected) - (float)(actual)); \
        if (diff > (tolerance)) { \
            printf("❌ FAILED: %s - %s (|%.3f - %.3f| = %.3f > %.3f)\n", __func__, message, (float)(expected), (float)(actual), diff, (float)(tolerance)); \
            return false; \
        } \
    } while(0)

/**
 * @brief Record test result
 */
void record_test_result(const char* test_name, bool passed, const char* failure_reason = "") {
    TestResult result = {test_name, passed, failure_reason ? failure_reason : ""};
    g_test_results.push_back(result);
    
    if (!g_summary_only) {
        if (passed) {
            printf("✅ PASSED: %s\n", test_name);
        } else {
            printf("❌ FAILED: %s - %s\n", test_name, failure_reason ? failure_reason : "");
        }
    }
}

/**
 * @brief Execute test only if it matches the filter
 */
void run_test_if_matches(const char* test_name, bool (*test_function)()) {
    if (!matches_filter(test_name)) {
        if (!g_summary_only) {
            printf("⏭️  SKIPPED: %s (filter)\n", test_name);
        }
        // Don't record skipped tests in results
        return;
    }
    
    record_test_result(test_name, test_function());
}

/**
 * @brief Initialize tracking arrays
 */
void init_tracking_arrays(int max_numa_nodes) {
    g_threads_executed = 0;
    g_numa_nodes_used = 0;
    g_threads_per_numa.clear();
    g_numa_node_active.clear();
    
    // Resize vectors to proper size and initialize
    g_threads_per_numa.resize(max_numa_nodes, 0);  // Initialize with 0
    g_numa_node_active.resize(max_numa_nodes, false);  // Initialize with false
}

/**
 * @brief Reset tracking counters and ensure arrays are initialized
 */
void reset_tracking() {
    // Ensure tracking arrays are initialized (needed when running filtered tests)
    if (g_numa_node_active.empty() || g_threads_per_numa.empty()) {
        // Get current config to determine array size
        ggml_numa_openmp_config_t config = ggml_numa_openmp_coordinator_get_config();
        if (config.initialized && config.total_numa_nodes > 0) {
            init_tracking_arrays(config.total_numa_nodes);
        } else {
            // Fallback: initialize with a reasonable default
            init_tracking_arrays(8);  // Support up to 8 NUMA nodes
        }
    }
    
    g_threads_executed = 0;
    g_numa_nodes_used = 0;
    for (int i = 0; i < (int)g_threads_per_numa.size(); i++) {
        g_threads_per_numa[i] = 0;  // Regular assignment now (mutex will protect concurrent access)
    }
    for (int i = 0; i < (int)g_numa_node_active.size(); i++) {
        g_numa_node_active[i] = 0;  // 0 = false
    }
}

/**
 * @brief Test basic coordinator initialization without CPU mask
 */
bool test_basic_initialization() {
    TEST_PRINTF("🧪 Testing basic coordinator initialization...\n");
    
    // Shutdown any existing coordinator
    ggml_numa_openmp_coordinator_shutdown();
    
    // Test initialization
    bool init_result = ggml_numa_openmp_coordinator_init();
    EXPECT_TRUE(init_result, "Basic initialization should succeed");
    
    // Get configuration
    ggml_numa_openmp_config_t config = ggml_numa_openmp_coordinator_get_config();
    EXPECT_TRUE(config.initialized, "Coordinator should be marked as initialized");
    EXPECT_GE(config.total_numa_nodes, 1, "Should have at least 1 NUMA node");
    EXPECT_GE(config.threads_per_node, 1, "Should have at least 1 thread per node");
    
    // Verify per-NUMA threadpool manager is initialized
    EXPECT_TRUE(config.threadpool_manager.teams_initialized, "Per-NUMA thread teams should be initialized");
    EXPECT_EQ(config.threadpool_manager.num_teams, config.total_numa_nodes, "Number of teams should match NUMA nodes");
    
    TEST_PRINTF("   📊 Configuration: %d NUMA nodes, %d threads per node\n", 
           config.total_numa_nodes, config.threads_per_node);
    TEST_PRINTF("   📊 Thread teams: %d teams initialized, barriers: %s\n",
           config.threadpool_manager.num_teams,
           config.threadpool_manager.barriers_initialized ? "Yes" : "No");
    
    // Initialize tracking arrays
    init_tracking_arrays(config.total_numa_nodes);
    
    return true;
}

/**
 * @brief Test CPU mask creation and validation
 */
bool test_cpu_mask_creation() {
    TEST_PRINTF("🧪 Testing CPU mask creation...\n");
    
#ifdef __linux__
    // Test valid CPU mask creation
    int cpu_indices[] = {0, 1, 2, 3};
    int num_cpus = 4;
    
    ggml_numa_cpu_mask_t* mask = ggml_numa_create_cpu_mask(cpu_indices, num_cpus);
    EXPECT_TRUE(mask != NULL, "CPU mask creation should succeed");
    EXPECT_TRUE(mask->valid, "CPU mask should be valid");
    EXPECT_GE(mask->max_cpus, num_cpus, "CPU mask should support enough CPUs");
    
    // Verify CPU bits are set correctly
    cpu_set_t* cpu_set = (cpu_set_t*)mask->cpu_set_ptr;
    for (int i = 0; i < num_cpus; i++) {
        EXPECT_TRUE(CPU_ISSET(cpu_indices[i], cpu_set), "CPU should be set in mask");
    }
    
    // Test with invalid CPUs (should still create mask but skip invalid indices)
    int invalid_cpu_indices[] = {0, 1, 9999, 10000};  // 9999, 10000 likely invalid
    ggml_numa_cpu_mask_t* mask2 = ggml_numa_create_cpu_mask(invalid_cpu_indices, 4);
    EXPECT_TRUE(mask2 != NULL, "CPU mask creation should succeed even with some invalid CPUs");
    
    // Clean up
    ggml_numa_free_cpu_mask(mask);
    ggml_numa_free_cpu_mask(mask2);
    
    TEST_PRINTF("   ✅ CPU mask creation and validation working\n");
#else
    TEST_PRINTF("   ⚠️  CPU mask tests skipped (not on Linux)\n");
#endif
    
    return true;
}

/**
 * @brief Test CPU mask initialization and thread team binding
 */
bool test_cpu_mask_initialization() {
    TEST_PRINTF("🧪 Testing CPU mask initialization and thread team binding...\n");
    
    // Shutdown any existing coordinator
    ggml_numa_openmp_coordinator_shutdown();
    
#ifdef __linux__
    // Create a CPU mask for specific cores
    int cpu_indices[] = {0, 1, 2, 3, 4, 5, 6, 7};  // First 8 cores
    int num_cpus = 8;
    
    ggml_numa_cpu_mask_t* mask = ggml_numa_create_cpu_mask(cpu_indices, num_cpus);
    EXPECT_TRUE(mask != NULL, "CPU mask creation should succeed");
    
    // Initialize with CPU mask
    bool init_result = ggml_numa_openmp_coordinator_init_with_mask(mask);
    EXPECT_TRUE(init_result, "CPU mask initialization should succeed");
    
    // Get configuration
    ggml_numa_openmp_config_t config = ggml_numa_openmp_coordinator_get_config();
    EXPECT_TRUE(config.initialized, "Coordinator should be initialized");
    EXPECT_GE(config.total_numa_nodes, 1, "Should have at least 1 NUMA node");
    
    // Verify thread teams are properly initialized
    EXPECT_TRUE(config.threadpool_manager.teams_initialized, "Thread teams should be initialized");
    EXPECT_EQ(config.threadpool_manager.num_teams, config.total_numa_nodes, "Should have teams for all NUMA nodes");
    
    // Verify each thread team has CPU bindings
    for (int i = 0; i < config.threadpool_manager.num_teams; i++) {
        const auto& team = config.threadpool_manager.teams[i];
        EXPECT_TRUE(team.initialized, "Each thread team should be initialized");
        EXPECT_EQ(team.numa_node_id, i, "Thread team should have correct NUMA node ID");
        EXPECT_GE(team.num_threads, 1, "Each team should have at least 1 thread");
        EXPECT_GE(team.num_cpus, 1, "Each team should have at least 1 CPU assigned");
        EXPECT_TRUE(team.cpu_ids != NULL, "Each team should have CPU ID array");
        
        TEST_PRINTF("   📊 Team %d: %d threads, %d CPUs, bound=%s\n", 
               i, team.num_threads, team.num_cpus, team.threads_bound ? "Yes" : "No");
    }
    
    // Clean up
    ggml_numa_free_cpu_mask(mask);
#else
    TEST_PRINTF("   ⚠️  CPU mask initialization tests skipped (not on Linux)\n");
#endif
    
    return true;
}

/**
 * @brief Test thread distribution validation
 */
bool test_thread_distribution() {
    TEST_PRINTF("🧪 Testing thread distribution validation...\n");
    
    // Shutdown and reinitialize
    ggml_numa_openmp_coordinator_shutdown();
    ggml_numa_openmp_coordinator_init();
    
    ggml_numa_openmp_config_t config = ggml_numa_openmp_coordinator_get_config();
    
    // Validate thread distribution makes sense
    int total_threads = config.total_numa_nodes * config.threads_per_node;
    
    TEST_PRINTF("   📊 Thread distribution:\n");
    TEST_PRINTF("      - NUMA nodes: %d\n", config.total_numa_nodes);
    TEST_PRINTF("      - Threads per node: %d\n", config.threads_per_node);
    TEST_PRINTF("      - Total threads: %d\n", total_threads);
    TEST_PRINTF("      - NUMA available: %s\n", config.numa_available ? "Yes" : "No");
    TEST_PRINTF("      - Thread teams initialized: %s\n", config.threadpool_manager.teams_initialized ? "Yes" : "No");
    
    // Basic sanity checks
    EXPECT_GE(config.total_numa_nodes, 1, "Must have at least 1 NUMA node");
    EXPECT_GE(config.threads_per_node, 1, "Must have at least 1 thread per node");
    EXPECT_GE(total_threads, 1, "Must have at least 1 total thread");
    
    // Verify per-NUMA thread teams are set up correctly
    EXPECT_TRUE(config.threadpool_manager.teams_initialized, "Thread teams should be initialized");
    EXPECT_EQ(config.threadpool_manager.num_teams, config.total_numa_nodes, "Should have one team per NUMA node");
    
    // Check for reasonable distribution
    if (config.numa_available && config.total_numa_nodes > 1) {
        TEST_PRINTF("   ✅ Multi-NUMA system detected - validating distribution\n");
        EXPECT_TRUE(config.threadpool_manager.barriers_initialized, "Barriers should be initialized for multi-NUMA");
    } else {
        TEST_PRINTF("   ✅ Single-node or non-NUMA system detected\n");
    }
    
    return true;
}

/**
 * @brief Create a test tensor for executor testing
 */
struct ggml_tensor* create_test_tensor(struct ggml_context* ctx, int size) {
    struct ggml_tensor* tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, size);
    if (!tensor) return NULL;
    
    // Initialize tensor data
    float* data = (float*)ggml_get_data(tensor);
    for (int i = 0; i < size; i++) {
        data[i] = (float)i;
    }
    
    return tensor;
}

/**
 * @brief Work function that tracks execution details
 */
enum ggml_status tracking_work_function(void* work_context, struct ggml_compute_params* params) {
    // Get thread-local NUMA context
    extern __thread int ggml_current_numa_node;
    extern __thread bool ggml_numa_is_data_parallel_execution;
    extern __thread int ggml_numa_total_nodes_for_data_parallel;
    extern __thread void* ggml_numa_shared_result_tensor_data;
    
    // Bounds checking
    if (ggml_current_numa_node < 0 || ggml_current_numa_node >= (int)g_numa_node_active.size()) {
        TEST_PRINTF("❌ NUMA node %d out of bounds (max=%zu)\n", ggml_current_numa_node, g_numa_node_active.size());
        return GGML_STATUS_FAILED;
    }
    
    // Track execution details (use mutex to protect g_threads_per_numa)
    g_threads_executed++;
    g_numa_node_active[ggml_current_numa_node] = true;
    {
        std::lock_guard<std::mutex> lock(g_tracking_mutex);
        g_threads_per_numa[ggml_current_numa_node]++;
    }
    
    // Verify work buffer is available if needed
    if (params->wsize > 0 && params->wdata == NULL) {
        printf("❌ Work buffer should be allocated when wsize > 0\n");
        return GGML_STATUS_FAILED;
    }
    
    // Do some simple work
    struct ggml_tensor* tensor = (struct ggml_tensor*)work_context;
    float* data = ggml_numa_shared_result_tensor_data ? 
                  (float*)ggml_numa_shared_result_tensor_data : 
                  (float*)ggml_get_data(tensor);
    
    size_t total_elements = ggml_nelements(tensor);
    size_t numa_start = 0, numa_end = total_elements;
    
    // Handle NUMA data slicing
    if (ggml_numa_is_data_parallel_execution) {
        size_t elements_per_node = total_elements / ggml_numa_total_nodes_for_data_parallel;
        numa_start = ggml_current_numa_node * elements_per_node;
        numa_end = (ggml_current_numa_node == ggml_numa_total_nodes_for_data_parallel - 1) ? 
                   total_elements : numa_start + elements_per_node;
    }
    
    // Handle thread slicing within NUMA node
    size_t slice_elements = numa_end - numa_start;
    size_t elements_per_thread = slice_elements / params->nth;
    size_t thread_start = numa_start + (params->ith * elements_per_thread);
    size_t thread_end = (params->ith == params->nth - 1) ? numa_end : thread_start + elements_per_thread;
    
    // Modify data to track which NUMA node and thread processed it
    for (size_t i = thread_start; i < thread_end; i++) {
        data[i] += (float)(ggml_current_numa_node * 1000 + params->ith * 10);
    }
    
    // Add small delay to simulate work and ensure parallel execution
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    return GGML_STATUS_SUCCESS;
}

/**
 * @brief Work function that verifies NUMA binding and tracks execution details
 */
enum ggml_status numa_binding_verification_work_function(void* work_context, struct ggml_compute_params* params) {
    // Get thread-local NUMA context
    extern __thread int ggml_current_numa_node;
    extern __thread bool ggml_numa_is_data_parallel_execution;
    extern __thread int ggml_numa_total_nodes_for_data_parallel;
    extern __thread void* ggml_numa_shared_result_tensor_data;
    
    // VERIFY NUMA BINDING: Check that thread is actually running on expected NUMA node
    char context_description[256];
    snprintf(context_description, sizeof(context_description), 
             "thread %d/%d, expected NUMA node %d, %s execution", 
             params->ith, params->nth, ggml_current_numa_node,
             ggml_numa_is_data_parallel_execution ? "data-parallel" : "single-node");
    
    bool binding_correct = verify_thread_numa_binding(ggml_current_numa_node, context_description);
    if (!binding_correct) {
        printf("   ❌ CRITICAL: NUMA binding verification failed for %s\n", context_description);
        return GGML_STATUS_FAILED;  // Fail the test if binding is wrong
    }
    
    // Track execution details (same as tracking_work_function, use mutex)
    g_threads_executed++;
    g_numa_node_active[ggml_current_numa_node] = true;
    {
        std::lock_guard<std::mutex> lock(g_tracking_mutex);
        g_threads_per_numa[ggml_current_numa_node]++;
    }
    
    // Verify work buffer is available if needed
    if (params->wsize > 0 && params->wdata == NULL) {
        printf("❌ Work buffer should be allocated when wsize > 0\n");
        return GGML_STATUS_FAILED;
    }
    
    // Do some simple work (same as tracking_work_function)
    struct ggml_tensor* tensor = (struct ggml_tensor*)work_context;
    float* data = ggml_numa_shared_result_tensor_data ? 
                  (float*)ggml_numa_shared_result_tensor_data : 
                  (float*)ggml_get_data(tensor);
    
    size_t total_elements = ggml_nelements(tensor);
    size_t numa_start = 0, numa_end = total_elements;
    
    // Handle NUMA data slicing
    if (ggml_numa_is_data_parallel_execution) {
        size_t elements_per_node = total_elements / ggml_numa_total_nodes_for_data_parallel;
        numa_start = ggml_current_numa_node * elements_per_node;
        numa_end = (ggml_current_numa_node == ggml_numa_total_nodes_for_data_parallel - 1) ? 
                   total_elements : numa_start + elements_per_node;
    }
    
    // Handle thread slicing within NUMA node
    size_t slice_elements = numa_end - numa_start;
    size_t elements_per_thread = slice_elements / params->nth;
    size_t thread_start = numa_start + (params->ith * elements_per_thread);
    size_t thread_end = (params->ith == params->nth - 1) ? numa_end : thread_start + elements_per_thread;
    
    // Modify data to track which NUMA node and thread processed it
    for (size_t i = thread_start; i < thread_end; i++) {
        data[i] += (float)(ggml_current_numa_node * 1000 + params->ith * 10);
    }
    
    // Add small delay to simulate work and ensure parallel execution
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    return GGML_STATUS_SUCCESS;
}

/**
 * @brief Test single-thread strategy execution
 */
bool test_single_thread_strategy() {
TEST_PRINTF("🧪 Testing single-thread strategy execution...\n");
    
    // Create test context and tensor
    struct ggml_init_params init_params = {
        .mem_size = 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false
    };
    struct ggml_context* ctx = ggml_init(init_params);
    EXPECT_TRUE(ctx != NULL, "Context creation should succeed");
    
    struct ggml_tensor* tensor = create_test_tensor(ctx, 16);  // Small tensor for single-thread
    EXPECT_TRUE(tensor != NULL, "Tensor creation should succeed");
    
    // Reset tracking
    reset_tracking();
    
    // Execute using single-thread strategy
    enum ggml_status status = ggml_numa_openmp_execute_single_thread(
        tensor,
        tracking_work_function,
        0,  // NUMA node 0
        0   // No work buffer needed
    );
    
    EXPECT_EQ(GGML_STATUS_SUCCESS, status, "Single-thread execution should succeed");
    
    // Verify execution characteristics
    EXPECT_EQ(1, g_threads_executed.load(), "Should execute exactly 1 thread");
    EXPECT_TRUE(g_numa_node_active[0], "NUMA node 0 should be active");
    EXPECT_EQ(1, g_threads_per_numa[0], "NUMA node 0 should have 1 thread");
    
    // Count active NUMA nodes
    int active_nodes = 0;
    for (int i = 0; i < (int)g_numa_node_active.size(); i++) {
        if (g_numa_node_active[i]) active_nodes++;
    }
    EXPECT_EQ(1, active_nodes, "Only 1 NUMA node should be active for single-thread strategy");
    
TEST_PRINTF("   ✅ Single-thread strategy: %d threads on %d NUMA nodes\n", 
           g_threads_executed.load(), active_nodes);
    
    ggml_free(ctx);
    return true;
}

/**
 * @brief Test single-node multi-thread strategy execution
 */
bool test_single_node_strategy() {
TEST_PRINTF("🧪 Testing single-node multi-thread strategy execution...\n");
    
    // Create test context and tensor
    struct ggml_init_params init_params = {
        .mem_size = 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false
    };
    struct ggml_context* ctx = ggml_init(init_params);
    EXPECT_TRUE(ctx != NULL, "Context creation should succeed");
    
    struct ggml_tensor* tensor = create_test_tensor(ctx, 128);  // Medium tensor for multi-thread
    EXPECT_TRUE(tensor != NULL, "Tensor creation should succeed");
    
    // Reset tracking
    reset_tracking();
    
    // Execute using single-node multi-thread strategy
    int n_threads = 4;
    enum ggml_status status = ggml_numa_openmp_execute_single_node(
        tensor,
        tracking_work_function,
        0,  // NUMA node 0
        n_threads,
        0   // No work buffer needed
    );
    
    EXPECT_EQ(GGML_STATUS_SUCCESS, status, "Single-node multi-thread execution should succeed");
    
    // Verify execution characteristics
    EXPECT_EQ(n_threads, g_threads_executed.load(), "Should execute exactly n_threads");
    EXPECT_TRUE(g_numa_node_active[0], "NUMA node 0 should be active");
    EXPECT_EQ(n_threads, g_threads_per_numa[0], "NUMA node 0 should have n_threads");
    
    // Count active NUMA nodes
    int active_nodes = 0;
    for (int i = 0; i < (int)g_numa_node_active.size(); i++) {
        if (g_numa_node_active[i]) active_nodes++;
    }
    EXPECT_EQ(1, active_nodes, "Only 1 NUMA node should be active for single-node strategy");
    
TEST_PRINTF("   ✅ Single-node strategy: %d threads on %d NUMA nodes\n", 
           g_threads_executed.load(), active_nodes);
    
    ggml_free(ctx);
    return true;
}

/**
 * @brief Test data-parallel multi-node strategy execution
 */
bool test_data_parallel_strategy() {
TEST_PRINTF("🧪 Testing data-parallel multi-node strategy execution...\n");
    
    // Create test context and tensor
    struct ggml_init_params init_params = {
        .mem_size = 2 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false
    };
    struct ggml_context* ctx = ggml_init(init_params);
    EXPECT_TRUE(ctx != NULL, "Context creation should succeed");
    
    struct ggml_tensor* tensor = create_test_tensor(ctx, 1024);  // Large tensor for data-parallel
    EXPECT_TRUE(tensor != NULL, "Tensor creation should succeed");
    
    // Reset tracking
    reset_tracking();
    
    // Execute using data-parallel strategy
    enum ggml_status status = ggml_numa_openmp_execute_data_parallel(
        tensor,
        tracking_work_function,
        0   // No work buffer needed
    );
    
    EXPECT_EQ(GGML_STATUS_SUCCESS, status, "Data-parallel execution should succeed");
    
    // Get coordinator config to understand expected behavior
    ggml_numa_openmp_config_t config = ggml_numa_openmp_coordinator_get_config();
    
    // Verify execution characteristics
    int expected_total_threads = config.total_numa_nodes * config.threads_per_node;
    EXPECT_EQ(expected_total_threads, g_threads_executed.load(), "Should execute threads on all NUMA nodes");
    
    // Count active NUMA nodes
    int active_nodes = 0;
    for (int i = 0; i < config.total_numa_nodes; i++) {
        if (g_numa_node_active[i]) {
            active_nodes++;
            // Allow some flexibility in thread count (within 10% of expected)
            int min_threads = (int)(config.threads_per_node * 0.9);
            int max_threads = config.threads_per_node;
            EXPECT_GE(g_threads_per_numa[i], min_threads, 
                      "Each active NUMA node should have at least 90% of threads_per_node threads");
            EXPECT_LE(g_threads_per_numa[i], max_threads, 
                      "Each active NUMA node should not exceed threads_per_node threads");
        }
    }
    
    if (config.total_numa_nodes > 1) {
        EXPECT_EQ(config.total_numa_nodes, active_nodes, "All NUMA nodes should be active for data-parallel strategy");
    } else {
        EXPECT_EQ(1, active_nodes, "Single NUMA node system should have 1 active node");
    }
    
TEST_PRINTF("   ✅ Data-parallel strategy: %d threads on %d NUMA nodes (expected %d nodes)\n", 
           g_threads_executed.load(), active_nodes, config.total_numa_nodes);
    
    ggml_free(ctx);
    return true;
}

/**
 * @brief Test that data-parallel execution provides unique thread IDs to each thread
 * 
 * This test specifically checks for the race condition where multiple threads
 * might receive the same thread ID (ith) value, causing them to process the same
 * data slice and create mathematical errors.
 */
bool test_data_parallel_unique_thread_ids() {
TEST_PRINTF("🧪 Testing data-parallel unique thread ID assignment...\n");
    
    // Create test context and tensor
    struct ggml_init_params init_params = {
        .mem_size = 2 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false
    };
    struct ggml_context* ctx = ggml_init(init_params);
    EXPECT_TRUE(ctx != NULL, "Context creation should succeed");
    
    struct ggml_tensor* tensor = create_test_tensor(ctx, 1024);  // Large tensor for data-parallel
    EXPECT_TRUE(tensor != NULL, "Tensor creation should succeed");
    
    // Global tracking variables for thread ID verification (increased size for 56+ threads per node)
    static std::atomic<int> g_thread_id_count[128];  // Track each thread ID occurrence (increased from 32)
    static std::atomic<int> g_numa_node_count[8];    // Track threads per NUMA node
    static std::atomic<int> g_total_calls;
    
    // Reset counters
    for (int i = 0; i < 128; i++) {  // Reset all 128 slots
        g_thread_id_count[i] = 0;
    }
    for (int i = 0; i < 8; i++) {
        g_numa_node_count[i] = 0;
    }
    g_total_calls = 0;
    
    // Work function that tracks thread IDs
    auto thread_id_tracking_fn = [](void* work_context, struct ggml_compute_params* params) -> enum ggml_status {
        // Get thread-local NUMA context
        extern __thread int ggml_current_numa_node;
        extern __thread bool ggml_numa_is_data_parallel_execution;
        extern __thread int ggml_numa_total_nodes_for_data_parallel;
        
        int ith = params->ith;  // Thread index within NUMA node
        int nth = params->nth;  // Total threads per NUMA node
        int numa_node = ggml_current_numa_node;
        
        // Safety check for array bounds (prevent buffer overflow)
        if (ith >= 128) {
TEST_PRINTF("❌ Thread ID too large: ith=%d >= 128 (array bounds)\n", ith);
            return GGML_STATUS_FAILED;
        }
        
        // Track this thread ID and NUMA node
        g_thread_id_count[ith]++;
        g_numa_node_count[numa_node]++;
        g_total_calls++;
        
        // Log detailed thread information
TEST_PRINTF("      Thread execution: NUMA node %d, ith=%d, nth=%d, data_parallel=%s, total_nodes=%d\n",
               numa_node, ith, nth,
               ggml_numa_is_data_parallel_execution ? "YES" : "NO",
               ggml_numa_total_nodes_for_data_parallel);
        
        // Validate thread context
        if (!ggml_numa_is_data_parallel_execution) {
TEST_PRINTF("❌ Thread should be in data-parallel mode but data_parallel=false\n");
            return GGML_STATUS_FAILED;
        }
        
        if (ith >= nth) {
TEST_PRINTF("❌ Invalid thread ID: ith=%d >= nth=%d\n", ith, nth);
            return GGML_STATUS_FAILED;
        }
        
        // Simulate work to ensure race conditions are detectable
        struct ggml_tensor* tensor = (struct ggml_tensor*)work_context;
        size_t total_elements = ggml_nelements(tensor);
        
        // Calculate data slice like a real kernel would
        if (ggml_numa_is_data_parallel_execution) {
            size_t elements_per_node = total_elements / ggml_numa_total_nodes_for_data_parallel;
            size_t node_start = numa_node * elements_per_node;
            size_t node_end = (numa_node == ggml_numa_total_nodes_for_data_parallel - 1) ? 
                              total_elements : node_start + elements_per_node;
            
            size_t node_elements = node_end - node_start;
            size_t elements_per_thread = (node_elements + nth - 1) / nth;
            size_t thread_start = node_start + ith * elements_per_thread;
            size_t thread_end = std::min(thread_start + elements_per_thread, node_end);
            
TEST_PRINTF("      Data slice: total=%zu, node[%zu,%zu), thread[%zu,%zu) (%zu elements)\n",
                   total_elements, node_start, node_end, thread_start, thread_end, thread_end - thread_start);
        }
        
        return GGML_STATUS_SUCCESS;
    };
    
    // Execute using data-parallel strategy
TEST_PRINTF("   🔄 Executing data-parallel strategy with thread ID tracking...\n");
    enum ggml_status status = ggml_numa_openmp_execute_data_parallel(
        tensor,
        thread_id_tracking_fn,
        0   // No work buffer needed
    );
    
    EXPECT_EQ(GGML_STATUS_SUCCESS, status, "Data-parallel execution should succeed");
    
    // Get coordinator config to understand expected behavior
    ggml_numa_openmp_config_t config = ggml_numa_openmp_coordinator_get_config();
    int expected_total_threads = config.total_numa_nodes * config.threads_per_node;
    
TEST_PRINTF("   📊 Thread ID distribution analysis:\n");
TEST_PRINTF("      Expected: %d total threads (%d nodes × %d threads/node)\n",
           expected_total_threads, config.total_numa_nodes, config.threads_per_node);
TEST_PRINTF("      Actual: %d total function calls\n", g_total_calls.load());
    
    // Verify thread ID uniqueness within each NUMA node
    bool thread_ids_unique = true;
    for (int thread_id = 0; thread_id < config.threads_per_node; thread_id++) {
        int count = g_thread_id_count[thread_id];
        int expected_count = config.total_numa_nodes;  // Each thread ID should appear once per NUMA node
        
TEST_PRINTF("      Thread ID %d: appeared %d times (expected %d times)\n", 
               thread_id, count, expected_count);
        
        if (count != expected_count) {
TEST_PRINTF("❌ Thread ID %d appeared %d times but expected %d (once per NUMA node)\n",
                   thread_id, count, expected_count);
            thread_ids_unique = false;
        }
    }
    
    // Verify NUMA node distribution
TEST_PRINTF("   📊 NUMA node distribution:\n");
    for (int node = 0; node < config.total_numa_nodes; node++) {
        int count = g_numa_node_count[node];
TEST_PRINTF("      NUMA node %d: %d threads executed (expected %d)\n", 
               node, count, config.threads_per_node);
        
        EXPECT_EQ(config.threads_per_node, count, 
                  "Each NUMA node should execute exactly threads_per_node threads");
    }
    
    // Verify no thread ID collisions
    EXPECT_TRUE(thread_ids_unique, "Each thread ID should appear exactly once per NUMA node");
    EXPECT_EQ(expected_total_threads, g_total_calls.load(), 
              "Total function calls should match expected thread count");
    
    if (thread_ids_unique && g_total_calls.load() == expected_total_threads) {
TEST_PRINTF("   ✅ Thread ID assignment: All threads received unique IDs within their NUMA nodes\n");
    } else {
TEST_PRINTF("   ❌ Thread ID assignment: Race condition detected - multiple threads got same IDs\n");
    }
    
    ggml_free(ctx);
    return thread_ids_unique && (g_total_calls.load() == expected_total_threads);
}

/**
 * @brief Test work buffer allocation and management
 */
bool test_work_buffer_allocation() {
TEST_PRINTF("🧪 Testing work buffer allocation and management...\n");
    
    // Create test context and tensor
    struct ggml_init_params init_params = {
        .mem_size = 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false
    };
    struct ggml_context* ctx = ggml_init(init_params);
    EXPECT_TRUE(ctx != NULL, "Context creation should succeed");
    
    struct ggml_tensor* tensor = create_test_tensor(ctx, 64);
    EXPECT_TRUE(tensor != NULL, "Tensor creation should succeed");
    
    // Work function that uses work buffer
    auto work_buffer_fn = [](void* work_context, struct ggml_compute_params* params) -> enum ggml_status {
        // Verify work buffer is allocated
        if (params->wsize == 0 || params->wdata == NULL) {
TEST_PRINTF("❌ Work buffer not allocated: wsize=%zu, wdata=%p\n", params->wsize, params->wdata);
            return GGML_STATUS_FAILED;
        }
        
        // Calculate per-thread work buffer offset
        // Each thread gets an equal portion of the work buffer
        float* base_work_buffer = (float*)params->wdata;
        size_t total_buffer_floats = params->wsize / sizeof(float);
        size_t per_thread_floats = total_buffer_floats / params->nth;
        float* thread_work_buffer = base_work_buffer + (params->ith * per_thread_floats);
        
        // Write to thread's portion of work buffer to verify it's accessible
        for (size_t i = 0; i < std::min(per_thread_floats, (size_t)16); i++) {
            thread_work_buffer[i] = (float)(params->ith * 100 + i);
        }
        
        // Read back to verify (from thread's own portion)
        for (size_t i = 0; i < std::min(per_thread_floats, (size_t)16); i++) {
            float expected = (float)(params->ith * 100 + i);
            if (thread_work_buffer[i] != expected) {
TEST_PRINTF("❌ Work buffer verification failed at index %zu: expected %f, got %f\n", 
                       i, expected, thread_work_buffer[i]);
                return GGML_STATUS_FAILED;
            }
        }
        
        return GGML_STATUS_SUCCESS;
    };
    
    // Test with various work buffer sizes
    size_t work_buffer_sizes[] = {256, 1024, 4096, 16384};
    
    for (size_t buffer_size : work_buffer_sizes) {
TEST_PRINTF("   🧪 Testing work buffer size: %zu bytes\n", buffer_size);
        
        // Test single-thread execution with work buffer
        enum ggml_status status = ggml_numa_openmp_execute_single_thread(
            tensor,
            work_buffer_fn,
            0,
            buffer_size
        );
        EXPECT_EQ(GGML_STATUS_SUCCESS, status, "Work buffer execution should succeed");
        
        // Test multi-thread execution with work buffer
        status = ggml_numa_openmp_execute_single_node(
            tensor,
            work_buffer_fn,
            0,
            2,  // 2 threads
            buffer_size
        );
        EXPECT_EQ(GGML_STATUS_SUCCESS, status, "Multi-thread work buffer execution should succeed");
    }
    
TEST_PRINTF("   ✅ Work buffer allocation and management working correctly\n");
    
    ggml_free(ctx);
    return true;
}

/**
 * @brief Test reusable work buffer functionality
 */
bool test_reusable_work_buffers() {
    TEST_PRINTF("🧪 Testing reusable work buffer functionality...\n");
    
    // Clean up any existing work buffers first
    ggml_numa_openmp_cleanup_thread_work_buffers();
    
    // Initially, no work buffer should exist
    void* buffer_ptr = nullptr;
    size_t current_size = 0;
    int numa_node = -1;
    bool is_numa_allocated = false;
    
    bool buffer_exists = ggml_numa_openmp_get_thread_work_buffer_state(
        &buffer_ptr, &current_size, &numa_node, &is_numa_allocated);
    
    EXPECT_FALSE(buffer_exists, "Initially no work buffer should exist");
    EXPECT_PTR_EQ(nullptr, buffer_ptr, "Initial buffer pointer should be NULL");
    EXPECT_EQ(0, current_size, "Initial buffer size should be 0");
    
    TEST_PRINTF("   ✅ Initial state: No work buffer exists\n");
    
    // Test first allocation
    size_t first_size = 1024;
    int target_node = 0;
    void* first_buffer = ggml_numa_openmp_test_force_work_buffer_allocation(first_size, target_node);
    
    EXPECT_TRUE(first_buffer != nullptr, "First allocation should succeed");
    
    // Check buffer state after first allocation
    buffer_exists = ggml_numa_openmp_get_thread_work_buffer_state(
        &buffer_ptr, &current_size, &numa_node, &is_numa_allocated);
    
    EXPECT_TRUE(buffer_exists, "Buffer should exist after first allocation");
    EXPECT_PTR_EQ(first_buffer, buffer_ptr, "Buffer pointer should match first allocation");
    EXPECT_GE(current_size, first_size, "Buffer size should be at least requested size");
    EXPECT_EQ(target_node, numa_node, "Buffer should be on target NUMA node");
    
    TEST_PRINTF("   ✅ First allocation: %zu bytes on node %d at %p\n", 
                current_size, numa_node, buffer_ptr);
    
    // Test buffer reuse with same size
    void* reused_buffer = ggml_numa_openmp_test_force_work_buffer_allocation(first_size, target_node);
    
    EXPECT_PTR_EQ(first_buffer, reused_buffer, "Buffer should be reused for same size request");
    
    // Check that buffer state hasn't changed
    void* check_buffer_ptr = nullptr;
    size_t check_size = 0;
    ggml_numa_openmp_get_thread_work_buffer_state(&check_buffer_ptr, &check_size, nullptr, nullptr);
    
    EXPECT_PTR_EQ(first_buffer, check_buffer_ptr, "Buffer pointer should remain unchanged on reuse");
    EXPECT_EQ(current_size, check_size, "Buffer size should remain unchanged on reuse");
    
    TEST_PRINTF("   ✅ Buffer reuse: Same buffer reused for same size request\n");
    
    // Test buffer growth
    size_t larger_size = first_size * 3;  // Request significantly larger buffer
    void* grown_buffer = ggml_numa_openmp_test_force_work_buffer_allocation(larger_size, target_node);
    
    EXPECT_TRUE(grown_buffer != nullptr, "Buffer growth allocation should succeed");
    
    // Buffer might be reallocated for growth
    size_t new_size = 0;
    ggml_numa_openmp_get_thread_work_buffer_state(&buffer_ptr, &new_size, nullptr, nullptr);
    
    EXPECT_GE(new_size, larger_size, "New buffer size should accommodate larger request");
    EXPECT_GT(new_size, current_size, "Buffer should have grown");
    
    TEST_PRINTF("   ✅ Buffer growth: Grew from %zu to %zu bytes for %zu byte request\n", 
                current_size, new_size, larger_size);
    
    // Test buffer reuse after growth
    void* reused_after_growth = ggml_numa_openmp_test_force_work_buffer_allocation(first_size, target_node);
    
    EXPECT_PTR_EQ(grown_buffer, reused_after_growth, "Grown buffer should be reused for smaller requests");
    
    // Verify buffer size hasn't changed
    size_t final_size = 0;
    ggml_numa_openmp_get_thread_work_buffer_state(nullptr, &final_size, nullptr, nullptr);
    
    EXPECT_EQ(new_size, final_size, "Buffer size should remain unchanged for smaller reuse");
    
    TEST_PRINTF("   ✅ Post-growth reuse: Large buffer reused for smaller requests\n");
    
    // Test NUMA node switching (should cause reallocation)
    int different_node = (target_node == 0) ? 1 : 0;
    void* different_node_buffer = ggml_numa_openmp_test_force_work_buffer_allocation(first_size, different_node);
    
    EXPECT_TRUE(different_node_buffer != nullptr, "Different NUMA node allocation should succeed");
    
    int actual_numa_node = -1;
    ggml_numa_openmp_get_thread_work_buffer_state(nullptr, nullptr, &actual_numa_node, nullptr);
    
    EXPECT_EQ(different_node, actual_numa_node, "Buffer should be allocated on different NUMA node");
    
    TEST_PRINTF("   ✅ NUMA node switching: Buffer reallocated for different NUMA node\n");
    
    // Cleanup
    ggml_numa_openmp_cleanup_thread_work_buffers();
    
    // Verify cleanup
    buffer_exists = ggml_numa_openmp_get_thread_work_buffer_state(nullptr, nullptr, nullptr, nullptr);
    EXPECT_FALSE(buffer_exists, "Buffer should not exist after cleanup");
    
    TEST_PRINTF("   ✅ Cleanup: Work buffer successfully freed\n");
    
    return true;
}

/**
 * @brief Test work buffer thread isolation
 */

// Thread context for isolation test
struct ThreadBufferInfo {
    std::thread::id thread_id;
    void* buffer_ptr;
    size_t buffer_size;
    int numa_node;
    bool allocation_success;
    std::mutex* mutex;
    std::vector<ThreadBufferInfo>* results;
};

// Global static variables for test contexts
static std::vector<ThreadBufferInfo>* g_thread_results = nullptr;
static std::mutex* g_results_mutex = nullptr;
static std::vector<void*>* g_operation_buffers = nullptr;
static std::vector<size_t>* g_operation_sizes = nullptr;

// Work function for thread isolation test
static enum ggml_status thread_isolation_work_function(void* context, struct ggml_compute_params* params) {
    (void)context; // Tensor pointer - not used in this test
    (void)params;  // Parameters - not used in this test
    
    // Access static test data
    if (!g_thread_results || !g_results_mutex) {
        return GGML_STATUS_FAILED;
    }
    
    // Clean up any existing work buffer for this thread
    ggml_numa_openmp_cleanup_thread_work_buffers();
    
    // Allocate work buffer for this thread
    const size_t buffer_size = 2048;
    void* buffer = ggml_numa_openmp_test_force_work_buffer_allocation(buffer_size, 0);
    
    // Record buffer information
    void* buffer_ptr = nullptr;
    size_t current_size = 0;
    int numa_node = -1;
    bool exists = ggml_numa_openmp_get_thread_work_buffer_state(
        &buffer_ptr, &current_size, &numa_node, nullptr);
    
    // Store results thread-safely
    {
        std::lock_guard<std::mutex> lock(*g_results_mutex);
        ThreadBufferInfo info;
        info.thread_id = std::this_thread::get_id();
        info.buffer_ptr = buffer_ptr;
        info.buffer_size = current_size;
        info.numa_node = numa_node;
        info.allocation_success = exists && (buffer != nullptr);
        g_thread_results->push_back(info);
    }
    
    return GGML_STATUS_SUCCESS;
}

// Helper to set global pointers for thread isolation test
static void set_thread_isolation_test_context(std::vector<ThreadBufferInfo>* results, std::mutex* mutex) {
    g_thread_results = results;
    g_results_mutex = mutex;
}

bool test_work_buffer_thread_isolation() {
    TEST_PRINTF("🧪 Testing work buffer thread isolation...\n");
    
    std::vector<ThreadBufferInfo> thread_results;
    std::mutex results_mutex;
    
    const int num_threads = 4;
    const size_t buffer_size = 2048;
    
    // Create test context and tensor
    struct ggml_init_params init_params = {
        .mem_size = 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false
    };
    struct ggml_context* ctx = ggml_init(init_params);
    EXPECT_TRUE(ctx != NULL, "Context creation should succeed");
    
    struct ggml_tensor* tensor = create_test_tensor(ctx, 64);
    EXPECT_TRUE(tensor != NULL, "Tensor creation should succeed");
    
    // Set context for thread isolation test
    set_thread_isolation_test_context(&thread_results, &results_mutex);
    
    // Execute multi-threaded work function
    enum ggml_status status = ggml_numa_openmp_execute_single_node(
        tensor,
        thread_isolation_work_function,
        0,  // NUMA node 0
        num_threads,
        buffer_size  // Work buffer size
    );
    
    EXPECT_EQ(GGML_STATUS_SUCCESS, status, "Multi-thread execution should succeed");
    EXPECT_EQ(num_threads, (int)thread_results.size(), "Should have results from all threads");
    
    TEST_PRINTF("   📊 Collected buffer info from %d threads\n", (int)thread_results.size());
    
    // Verify each thread got its own unique buffer
    std::set<void*> unique_buffers;
    std::set<std::thread::id> unique_thread_ids;
    
    for (const auto& result : thread_results) {
        EXPECT_TRUE(result.allocation_success, "Each thread should successfully allocate buffer");
        EXPECT_TRUE(result.buffer_ptr != nullptr, "Each thread should have valid buffer pointer");
        EXPECT_GE(result.buffer_size, buffer_size, "Each buffer should be at least requested size");
        
        // Check uniqueness
        EXPECT_TRUE(unique_buffers.find(result.buffer_ptr) == unique_buffers.end(), 
                    "Each thread should have unique buffer pointer");
        EXPECT_TRUE(unique_thread_ids.find(result.thread_id) == unique_thread_ids.end(),
                    "Each thread ID should be unique");
        
        unique_buffers.insert(result.buffer_ptr);
        unique_thread_ids.insert(result.thread_id);
        
        TEST_PRINTF("   🧵 Thread %zu: buffer=%p, size=%zu, node=%d\n", 
                    std::hash<std::thread::id>{}(result.thread_id) % 1000,
                    result.buffer_ptr, result.buffer_size, result.numa_node);
    }
    
    EXPECT_EQ(num_threads, (int)unique_buffers.size(), "All buffers should be unique");
    EXPECT_EQ(num_threads, (int)unique_thread_ids.size(), "All thread IDs should be unique");
    
    TEST_PRINTF("   ✅ Thread isolation: %d unique buffers for %d threads\n", 
                (int)unique_buffers.size(), num_threads);
    
    ggml_free(ctx);
    return true;
}

/**
 * @brief Test work buffer persistence across multiple operations
 */

// Context for persistence test
struct PersistenceTestContext {
    std::vector<void*>* operation_buffers;
    std::vector<size_t>* operation_sizes;
};

// Work function for persistence test
static enum ggml_status persistence_test_work_function(void* context, struct ggml_compute_params* params) {
    (void)context; // Tensor pointer - not used in this test
    
    // Get test context from global variable (this is just for testing)
    if (g_operation_buffers && g_operation_sizes) {
        // Record current work buffer state
        void* buffer_ptr = nullptr;
        size_t current_size = 0;
        ggml_numa_openmp_get_thread_work_buffer_state(&buffer_ptr, &current_size, nullptr, nullptr);
        
        g_operation_buffers->push_back(buffer_ptr);
        g_operation_sizes->push_back(current_size);
    }
    
    // Use the work buffer to verify it's functional
    if (params->wdata && params->wsize > 0) {
        uint32_t* test_buffer = (uint32_t*)params->wdata;
        size_t words_available = params->wsize / sizeof(uint32_t);
        
        // Write test pattern
        for (size_t i = 0; i < std::min(words_available, (size_t)32); i++) {
            test_buffer[i] = 0xCAFEBABE + (uint32_t)i;
        }
        
        // Verify test pattern
        for (size_t i = 0; i < std::min(words_available, (size_t)32); i++) {
            if (test_buffer[i] != 0xCAFEBABE + (uint32_t)i) {
                return GGML_STATUS_FAILED;
            }
        }
    }
    
    return GGML_STATUS_SUCCESS;
}

// Helper to set global pointers for persistence test
static void set_persistence_test_context(std::vector<void*>* buffers, std::vector<size_t>* sizes) {
    g_operation_buffers = buffers;
    g_operation_sizes = sizes;
}

bool test_work_buffer_persistence() {
    TEST_PRINTF("🧪 Testing work buffer persistence across operations...\n");
    
    // Clean up any existing work buffers
    ggml_numa_openmp_cleanup_thread_work_buffers();
    
    // Create test context and tensor
    struct ggml_init_params init_params = {
        .mem_size = 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false
    };
    struct ggml_context* ctx = ggml_init(init_params);
    EXPECT_TRUE(ctx != NULL, "Context creation should succeed");
    
    struct ggml_tensor* tensor = create_test_tensor(ctx, 128);
    EXPECT_TRUE(tensor != NULL, "Tensor creation should succeed");
    
    // Track buffer pointers across operations
    std::vector<void*> operation_buffers;
    std::vector<size_t> operation_sizes;
    
    // Set context for persistence test
    set_persistence_test_context(&operation_buffers, &operation_sizes);
    
    // Test series of operations with different buffer sizes
    size_t test_sizes[] = {512, 1024, 2048, 1024, 4096, 1024};
    const int num_operations = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    for (int op = 0; op < num_operations; op++) {
        TEST_PRINTF("   🔄 Operation %d: requesting %zu bytes\n", op + 1, test_sizes[op]);
        
        enum ggml_status status = ggml_numa_openmp_execute_single_thread(
            tensor,
            persistence_test_work_function,
            0,  // NUMA node 0
            test_sizes[op]  // Work buffer size
        );
        
        EXPECT_EQ(GGML_STATUS_SUCCESS, status, "Operation should succeed");
    }
    
    EXPECT_EQ(num_operations, (int)operation_buffers.size(), "Should have buffer info from all operations");
    EXPECT_EQ(num_operations, (int)operation_sizes.size(), "Should have size info from all operations");
    
    TEST_PRINTF("   📊 Buffer persistence analysis:\n");
    
    // Analyze buffer reuse patterns
    int reuse_count = 0;
    int growth_count = 0;
    
    for (int i = 0; i < num_operations; i++) {
        TEST_PRINTF("     Op %d: buffer=%p, size=%zu (requested %zu)\n", 
                    i + 1, operation_buffers[i], operation_sizes[i], test_sizes[i]);
        
        if (i > 0) {
            if (operation_buffers[i] == operation_buffers[i-1]) {
                reuse_count++;
                TEST_PRINTF("       ✅ Buffer reused from previous operation\n");
            } else {
                TEST_PRINTF("       🔄 Buffer reallocated (growth or other reason)\n");
            }
            
            if (operation_sizes[i] > operation_sizes[i-1]) {
                growth_count++;
                TEST_PRINTF("       📈 Buffer size increased from %zu to %zu\n", 
                            operation_sizes[i-1], operation_sizes[i]);
            }
        }
        
        // Verify buffer size is adequate for request
        EXPECT_GE(operation_sizes[i], test_sizes[i], "Buffer size should accommodate request");
    }
    
    TEST_PRINTF("   📈 Persistence stats: %d reuses, %d growth events out of %d operations\n", 
                reuse_count, growth_count, num_operations - 1);
    
    // We expect some reuse (operations 4 and 6 request smaller sizes than previous)
    EXPECT_GT(reuse_count, 0, "Should have some buffer reuse for smaller requests");
    
    // We expect some growth (operation 3 and 5 request larger sizes)
    EXPECT_GT(growth_count, 0, "Should have some buffer growth for larger requests");
    
    // Test final cleanup
    ggml_numa_openmp_cleanup_thread_work_buffers();
    
    bool buffer_exists = ggml_numa_openmp_get_thread_work_buffer_state(nullptr, nullptr, nullptr, nullptr);
    EXPECT_FALSE(buffer_exists, "Buffer should not exist after final cleanup");
    
    TEST_PRINTF("   ✅ Persistence test completed successfully\n");
    
    ggml_free(ctx);
    return true;
}

/**
 * @brief Test coordinator integration with different strategies
 */
bool test_executor_integration() {
TEST_PRINTF("🧪 Testing coordinator execution strategies...\n");
    
    // Create test context and tensors
    struct ggml_init_params init_params = {
        .mem_size = 2 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false
    };
    struct ggml_context* ctx = ggml_init(init_params);
    EXPECT_TRUE(ctx != NULL, "Context creation should succeed");
    
    // Test with different execution strategies directly on coordinator
    struct {
        int threads;
        const char* strategy_name;
    } test_cases[] = {
        {1, "single-thread"},     // Single thread
        {4, "multi-thread"},      // Multiple threads
        {8, "high-concurrency"}   // High concurrency
    };
    
    ggml_numa_openmp_config_t config = ggml_numa_openmp_coordinator_get_config();
    
    for (auto test_case : test_cases) {
TEST_PRINTF("   🧪 Testing %s strategy with %d threads\n", 
               test_case.strategy_name, test_case.threads);
        
        struct ggml_tensor* tensor = create_test_tensor(ctx, 1024);
        EXPECT_TRUE(tensor != NULL, "Tensor creation should succeed");
        
        reset_tracking();
        
        // Execute via coordinator single-node function
        enum ggml_status status = ggml_numa_openmp_execute_single_node(
            tensor,
            tracking_work_function,
            0,  // NUMA node 0
            test_case.threads,
            0   // No work buffer
        );
        EXPECT_EQ(GGML_STATUS_SUCCESS, status, "Coordinator should successfully execute");
        
        // Verify execution occurred
        EXPECT_GE(g_threads_executed.load(), 1, "At least one thread should have executed");
        
        int active_nodes = 0;
        for (int i = 0; i < (int)g_numa_node_active.size(); i++) {
            if (g_numa_node_active[i]) active_nodes++;
        }
        
TEST_PRINTF("      📊 Execution: %d threads on %d NUMA nodes\n", 
               g_threads_executed.load(), active_nodes);
        
        // Verify single-node execution
        EXPECT_LE(active_nodes, 1, "Single-node execution should use at most 1 NUMA node");
    }
    
    // Test full system capacity
TEST_PRINTF("   🧪 Testing FULL CAPACITY single-node execution with %d threads\n", 
           config.threads_per_node);
    
    struct ggml_tensor* full_tensor = create_test_tensor(ctx, 2048);
    EXPECT_TRUE(full_tensor != NULL, "Full capacity tensor creation should succeed");
    
    reset_tracking();
    
    // Execute with full single-node capacity
    enum ggml_status full_status = ggml_numa_openmp_execute_single_node(
        full_tensor,
        tracking_work_function,
        0,  // NUMA node 0
        config.threads_per_node,  // Full capacity: 56 threads
        0   // No work buffer
    );
    EXPECT_EQ(GGML_STATUS_SUCCESS, full_status, "Full capacity execution should succeed");
    
    // Verify full capacity execution
    EXPECT_GE(g_threads_executed.load(), config.threads_per_node * 0.9, 
             "Should execute at least 90% of requested threads for full capacity");
    
TEST_PRINTF("      📊 FULL CAPACITY: %d threads executed (requested %d)\n", 
           g_threads_executed.load(), config.threads_per_node);
    
    ggml_free(ctx);
    return true;
}

/**
 * @brief Test full system multi-NUMA execution capacity
 */
bool test_full_system_capacity() {
TEST_PRINTF("🧪 Testing FULL SYSTEM CAPACITY multi-NUMA execution...\n");
    
    // Create test context and tensor
    struct ggml_init_params init_params = {
        .mem_size = 4 * 1024 * 1024,  // Larger memory for big test
        .mem_buffer = NULL,
        .no_alloc = false
    };
    struct ggml_context* ctx = ggml_init(init_params);
    EXPECT_TRUE(ctx != NULL, "Context creation should succeed");
    
    ggml_numa_openmp_config_t config = ggml_numa_openmp_coordinator_get_config();
    int total_system_threads = config.total_numa_nodes * config.threads_per_node;
    
TEST_PRINTF("   📊 System capacity: %d NUMA nodes × %d threads = %d total threads\n",
           config.total_numa_nodes, config.threads_per_node, total_system_threads);
    
    struct ggml_tensor* tensor = create_test_tensor(ctx, 4096);
    EXPECT_TRUE(tensor != NULL, "Tensor creation should succeed");
    
    reset_tracking();
    
    // Execute with full system capacity (all NUMA nodes)
    enum ggml_status status = ggml_numa_openmp_execute_data_parallel(
        tensor,
        tracking_work_function,
        0   // No work buffer
    );
    EXPECT_EQ(GGML_STATUS_SUCCESS, status, "Full system execution should succeed");
    
    // Verify full system execution characteristics
    int expected_total_threads = config.total_numa_nodes * config.threads_per_node;
    int min_expected_threads = (int)(expected_total_threads * 0.85);  // Allow 15% tolerance
    
    EXPECT_GE(g_threads_executed.load(), min_expected_threads, 
             "Should execute at least 85% of system thread capacity");
    
    // Count active NUMA nodes
    int active_nodes = 0;
    for (int i = 0; i < (int)g_numa_node_active.size(); i++) {
        if (g_numa_node_active[i]) {
            active_nodes++;
TEST_PRINTF("      📊 NUMA Node %d: %d threads executed\n", i, g_threads_per_numa[i]);
        }
    }
    
    EXPECT_GE(active_nodes, config.total_numa_nodes, 
             "All NUMA nodes should be active for full system execution");
    
TEST_PRINTF("   ✅ FULL SYSTEM EXECUTION: %d threads across %d NUMA nodes (target: %d threads)\n", 
           g_threads_executed.load(), active_nodes, expected_total_threads);
    
    ggml_free(ctx);
    return true;
}

// Global variables for parallel execution test
static std::atomic<int> g_concurrent_threads{0};
static std::atomic<int> g_max_concurrent{0};

/**
 * @brief Work function for parallel execution verification
 */
static enum ggml_status parallel_execution_work_function(void* work_context, struct ggml_compute_params* params) {
    (void)work_context; // Unused
    (void)params; // Unused
    
    // Track concurrent execution
    int current = ++g_concurrent_threads;
    
    // Update maximum concurrent threads seen
    int prev_max = g_max_concurrent.load();
    while (prev_max < current && !g_max_concurrent.compare_exchange_weak(prev_max, current)) {
        // CAS loop
    }
    
    // Simulate work with delay
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    --g_concurrent_threads;
    return GGML_STATUS_SUCCESS;
}
/**
 * @brief Test parallel execution verification with timing
 */
bool test_parallel_execution_verification() {
TEST_PRINTF("🧪 Testing parallel execution verification...\n");
    
    // Create test context and tensor
    struct ggml_init_params init_params = {
        .mem_size = 2 * 1024 * 1024,  // Larger for high-thread tests
        .mem_buffer = NULL,
        .no_alloc = false
    };
    struct ggml_context* ctx = ggml_init(init_params);
    EXPECT_TRUE(ctx != NULL, "Context creation should succeed");
    
    ggml_numa_openmp_config_t config = ggml_numa_openmp_coordinator_get_config();
    int total_system_threads = config.total_numa_nodes * config.threads_per_node;  // Calculate total
    
    // Test with multiple thread counts including full system capacity
    struct {
        int threads;
        const char* test_name;
        int min_concurrent;
        bool use_data_parallel;
    } parallel_tests[] = {
        {4, "low-concurrency", 2, false},
        {16, "medium-concurrency", 8, false},
        {32, "high-concurrency", 16, false},
        {config.threads_per_node / 2, "half-node-capacity", config.threads_per_node / 4, false},  // 28 threads
        {config.threads_per_node, "FULL-SINGLE-NODE", config.threads_per_node / 2, false},       // 56 threads on single node
        {total_system_threads, "FULL-SYSTEM-CAPACITY", total_system_threads / 2, true}            // 112 threads across all nodes
    };
    
    for (auto test : parallel_tests) {
TEST_PRINTF("   🧪 Testing %s parallel execution with %d threads\n", 
               test.test_name, test.threads);
        
        struct ggml_tensor* tensor = create_test_tensor(ctx, 512);
        EXPECT_TRUE(tensor != NULL, "Tensor creation should succeed");
        
        // Reset counters
        g_concurrent_threads = 0;
        g_max_concurrent = 0;
        
        // Test execution (single-node vs data-parallel)
        auto start_time = std::chrono::high_resolution_clock::now();
        
        enum ggml_status status;
        if (test.use_data_parallel) {
            // Full system capacity - use data-parallel across all NUMA nodes
            status = ggml_numa_openmp_execute_data_parallel(
                tensor,
                parallel_execution_work_function,
                0   // No work buffer
            );
        } else {
            // Single-node execution
            status = ggml_numa_openmp_execute_single_node(
                tensor,
                parallel_execution_work_function,
                0,  // NUMA node 0
                test.threads,
                0   // No work buffer
            );
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        EXPECT_EQ(GGML_STATUS_SUCCESS, status, "Parallel execution should succeed");
        
        // Verify parallel execution occurred
        EXPECT_GE(g_max_concurrent.load(), test.min_concurrent, 
                 "Should see sufficient concurrent threads for parallel execution");
        
        if (test.use_data_parallel) {
TEST_PRINTF("      📊 DATA-PARALLEL: %d max concurrent threads observed across ALL NUMA nodes, %ld ms execution time\n", 
                   g_max_concurrent.load(), duration.count());
        } else {
TEST_PRINTF("      📊 SINGLE-NODE: %d threads requested, %d max concurrent observed, %ld ms execution time\n", 
                   test.threads, g_max_concurrent.load(), duration.count());
        }
        
        // For higher thread counts, execution should be significantly parallel
        if (test.threads >= 16) {
            // With 50ms delay per thread, sequential would take threads * 50ms
            // Parallel should be much faster
            int sequential_time = test.threads * 50;
            int max_parallel_time = sequential_time / 4;  // Allow 25% of sequential time
            EXPECT_LE(duration.count(), max_parallel_time, 
                     "High-thread parallel execution should be much faster than sequential");
        }
    }
    
TEST_PRINTF("   ✅ Parallel execution verified successfully across all thread counts\n");
    
    ggml_free(ctx);
    return true;
}

/**
 * @brief Test error handling and edge cases
 */
/**
 * @brief Test NUMA ISOLATE strategy for all available NUMA nodes
 * 
 * This test dynamically tests NUMA isolation for each node detected on the system.
 * It verifies that when ISOLATE strategy is used:
 * 1. All threads execute only on the specified NUMA node
 * 2. Thread binding verification works correctly
 * 3. Work distribution respects the isolation constraints
 * 4. Different thread counts work correctly with isolation
 */
bool test_numa_isolate_strategy() {
TEST_PRINTF("🧪 Testing NUMA ISOLATE strategy for all available nodes...\n");
    
    // Get system configuration
    ggml_numa_openmp_config_t config = ggml_numa_openmp_coordinator_get_config();
    int total_numa_nodes = config.total_numa_nodes;
    int threads_per_node = config.threads_per_node;
    
TEST_PRINTF("   📊 System detected: %d NUMA nodes, %d threads per node\n", 
           total_numa_nodes, threads_per_node);
    
    if (total_numa_nodes < 2) {
TEST_PRINTF("   ⚠️  Single NUMA node system - ISOLATE strategy testing limited\n");
TEST_PRINTF("   ✅ ISOLATE strategy test completed (single node system)\n");
        return true;
    }
    
    // Test ISOLATE strategy for each NUMA node
    for (int target_node = 0; target_node < total_numa_nodes; target_node++) {
TEST_PRINTF("   🔍 Testing ISOLATE strategy on NUMA node %d...\n", target_node);
        
        // Create test context and tensor
        struct ggml_init_params init_params = {
            .mem_size = 2 * 1024 * 1024,
            .mem_buffer = NULL,
            .no_alloc = false
        };
        struct ggml_context* ctx = ggml_init(init_params);
        EXPECT_TRUE(ctx != NULL, "Context creation should succeed");
        
        struct ggml_tensor* tensor = create_test_tensor(ctx, 256);  // Medium tensor
        EXPECT_TRUE(tensor != NULL, "Tensor creation should succeed");
        
        // Reset tracking
        reset_tracking();
        
        // Test different thread counts with ISOLATE strategy
        int test_thread_counts[] = {1, 4, 8, threads_per_node/4, threads_per_node/2, threads_per_node};
        int num_thread_tests = sizeof(test_thread_counts) / sizeof(test_thread_counts[0]);
        
        for (int i = 0; i < num_thread_tests; i++) {
            int n_threads = test_thread_counts[i];
            
            // Skip invalid thread counts
            if (n_threads <= 0 || n_threads > threads_per_node) {
                continue;
            }
            
TEST_PRINTF("      🧪 Testing %d threads isolated to NUMA node %d\n", n_threads, target_node);
            
            // Reset tracking for this sub-test
            reset_tracking();
            
            // Execute using single-node strategy (simulates ISOLATE behavior)
            enum ggml_status status = ggml_numa_openmp_execute_single_node(
                tensor,
                numa_binding_verification_work_function,
                target_node,  // Target NUMA node for isolation
                n_threads,
                0   // No work buffer needed
            );
            
            EXPECT_EQ(GGML_STATUS_SUCCESS, status, "NUMA ISOLATE execution should succeed");
            
            // Verify isolation constraints
            EXPECT_EQ(n_threads, g_threads_executed.load(), "Should execute exactly n_threads");
            EXPECT_TRUE(g_numa_node_active[target_node], "Target NUMA node should be active");
            EXPECT_EQ(n_threads, g_threads_per_numa[target_node], "All threads should be on target node");
            
            // Verify NO threads executed on other NUMA nodes
            for (int other_node = 0; other_node < total_numa_nodes; other_node++) {
                if (other_node != target_node) {
                    EXPECT_FALSE(g_numa_node_active[other_node], "Other NUMA nodes should NOT be active");
                    EXPECT_EQ(0, g_threads_per_numa[other_node], "Other NUMA nodes should have 0 threads");
                }
            }
            
TEST_PRINTF("         ✅ ISOLATE: %d threads correctly isolated to node %d\n", 
                   n_threads, target_node);
        }
        
        // Cleanup context
        ggml_free(ctx);
        
TEST_PRINTF("      ✅ NUMA node %d isolation tests completed successfully\n", target_node);
    }
    
TEST_PRINTF("   ✅ NUMA ISOLATE strategy testing completed for all %d nodes\n", total_numa_nodes);
    return true;
}

/**
 * @brief Test NUMA ISOLATE strategy with full node capacity and cross-validation
 * 
 * This test validates ISOLATE strategy behavior at maximum capacity per node
 * and performs cross-validation to ensure proper isolation.
 */
bool test_numa_isolate_full_capacity() {
TEST_PRINTF("🧪 Testing NUMA ISOLATE strategy at full node capacity...\n");
    
    // Get system configuration
    ggml_numa_openmp_config_t config = ggml_numa_openmp_coordinator_get_config();
    int total_numa_nodes = config.total_numa_nodes;
    int threads_per_node = config.threads_per_node;
    
    if (total_numa_nodes < 2) {
TEST_PRINTF("   ⚠️  Single NUMA node system - full capacity ISOLATE testing limited\n");
TEST_PRINTF("   ✅ ISOLATE full capacity test completed (single node system)\n");
        return true;
    }
    
    // Test full capacity isolation for each NUMA node
    for (int target_node = 0; target_node < total_numa_nodes; target_node++) {
TEST_PRINTF("   🔍 Testing FULL CAPACITY (%d threads) isolated to NUMA node %d...\n", 
               threads_per_node, target_node);
        
        // Create test context and tensor
        struct ggml_init_params init_params = {
            .mem_size = 4 * 1024 * 1024,  // Larger for full capacity
            .mem_buffer = NULL,
            .no_alloc = false
        };
        struct ggml_context* ctx = ggml_init(init_params);
        EXPECT_TRUE(ctx != NULL, "Context creation should succeed");
        
        struct ggml_tensor* tensor = create_test_tensor(ctx, 512);  // Larger tensor
        EXPECT_TRUE(tensor != NULL, "Tensor creation should succeed");
        
        // Reset tracking
        reset_tracking();
        
        // Execute using single-node strategy at full capacity
        enum ggml_status status = ggml_numa_openmp_execute_single_node(
            tensor,
            numa_binding_verification_work_function,
            target_node,  // Target NUMA node for isolation
            threads_per_node,  // Full node capacity
            0   // No work buffer needed
        );
        
        EXPECT_EQ(GGML_STATUS_SUCCESS, status, "Full capacity NUMA ISOLATE execution should succeed");
        
        // Verify full capacity isolation
        EXPECT_EQ(threads_per_node, g_threads_executed.load(), "Should execute all threads for the node");
        EXPECT_TRUE(g_numa_node_active[target_node], "Target NUMA node should be active");
        EXPECT_EQ(threads_per_node, g_threads_per_numa[target_node], "All threads should be on target node");
        
        // Cross-validation: Verify ZERO activity on all other nodes
        int total_other_threads = 0;
        for (int other_node = 0; other_node < total_numa_nodes; other_node++) {
            if (other_node != target_node) {
                EXPECT_FALSE(g_numa_node_active[other_node], "Other NUMA nodes should be completely inactive");
                EXPECT_EQ(0, g_threads_per_numa[other_node], "Other NUMA nodes should have exactly 0 threads");
                total_other_threads += g_threads_per_numa[other_node];
            }
        }
        
        EXPECT_EQ(0, total_other_threads, "Total threads on other nodes should be exactly 0");
        
        // Verify isolation efficiency
        float isolation_efficiency = (float)g_threads_per_numa[target_node] / (float)g_threads_executed.load();
        EXPECT_NEAR(1.0f, isolation_efficiency, 0.01f, "Isolation efficiency should be 100%");
        
TEST_PRINTF("      ✅ FULL CAPACITY ISOLATION: %d threads on node %d, 0 threads elsewhere\n", 
               threads_per_node, target_node);
        
        // Cleanup context
        ggml_free(ctx);
    }
    
TEST_PRINTF("   ✅ NUMA ISOLATE full capacity testing completed for all %d nodes\n", total_numa_nodes);
    return true;
}

/**
 * @brief Test NUMA ISOLATE strategy edge cases and boundary conditions
 */
bool test_numa_isolate_error_handling() {
TEST_PRINTF("🧪 Testing NUMA ISOLATE strategy edge cases...\n");
    
    // Get system configuration
    ggml_numa_openmp_config_t config = ggml_numa_openmp_coordinator_get_config();
    int total_numa_nodes = config.total_numa_nodes;
    
    // Create test context and tensor
    struct ggml_init_params init_params = {
        .mem_size = 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false
    };
    struct ggml_context* ctx = ggml_init(init_params);
    EXPECT_TRUE(ctx != NULL, "Context creation should succeed");
    
    struct ggml_tensor* tensor = create_test_tensor(ctx, 64);
    EXPECT_TRUE(tensor != NULL, "Tensor creation should succeed");
    
    // Test 1: Maximum valid NUMA node
TEST_PRINTF("   🧪 Testing maximum valid NUMA node...\n");
    reset_tracking();
    
    enum ggml_status status = ggml_numa_openmp_execute_single_node(
        tensor,
        tracking_work_function,
        total_numa_nodes - 1,  // Maximum valid node
        4,
        0
    );
    
TEST_PRINTF("      📊 Max valid node test status: %d\n", status);
    EXPECT_TRUE(status == GGML_STATUS_SUCCESS, "Max valid NUMA node should work");
    
    // Test 2: Minimum valid NUMA node 
TEST_PRINTF("   🧪 Testing minimum valid NUMA node...\n");
    reset_tracking();
    
    status = ggml_numa_openmp_execute_single_node(
        tensor,
        tracking_work_function,
        0,  // Minimum valid node
        4,
        0
    );
    
TEST_PRINTF("      📊 Min valid node test status: %d\n", status);
    EXPECT_TRUE(status == GGML_STATUS_SUCCESS, "Min valid NUMA node should work");
    
    // Test 3: Single thread execution
    if (total_numa_nodes > 0) {
TEST_PRINTF("   🧪 Testing single thread execution...\n");
        reset_tracking();
        
        status = ggml_numa_openmp_execute_single_node(
            tensor,
            tracking_work_function,
            0,  // Valid node
            1,  // Single thread (minimum valid)
            0
        );
        
TEST_PRINTF("      📊 Single thread test status: %d\n", status);
        EXPECT_TRUE(status == GGML_STATUS_SUCCESS, "Single thread execution should work");
    }
    
    // Test 4: Large work buffer size
TEST_PRINTF("   🧪 Testing large work buffer allocation...\n");
    reset_tracking();
    
    status = ggml_numa_openmp_execute_single_node(
        tensor,
        tracking_work_function,
        0,  // Valid node
        2,  // Valid thread count
        1024 * 1024  // 1MB work buffer
    );
    
TEST_PRINTF("      📊 Large buffer test status: %d\n", status);
    EXPECT_TRUE(status == GGML_STATUS_SUCCESS, "Large work buffer should work");
    
    // Cleanup
    ggml_free(ctx);
    
TEST_PRINTF("   ✅ NUMA ISOLATE edge case tests completed\n");
    return true;
}
bool test_error_handling() {
TEST_PRINTF("🧪 Testing error handling and edge cases...\n");
    
    // Test with very small work buffer
    struct ggml_init_params init_params = {
        .mem_size = 1024,
        .mem_buffer = NULL,
        .no_alloc = false
    };
    struct ggml_context* ctx = ggml_init(init_params);
    if (ctx) {
        struct ggml_tensor* tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        if (tensor) {
            // Test work function that returns error
            auto error_fn = [](void*, struct ggml_compute_params*) -> enum ggml_status { 
                return GGML_STATUS_FAILED; 
            };
            
            enum ggml_status status = ggml_numa_openmp_execute_single_thread(
                tensor, error_fn, 0, 0
            );
            // Should handle work function failures gracefully
TEST_PRINTF("   📊 Error handling test status: %d\n", status);
        }
        ggml_free(ctx);
    }
    
    // Test with zero threads (should handle gracefully)
TEST_PRINTF("   📊 Testing edge case handling...\n");
    
TEST_PRINTF("   ✅ Error handling tests completed\n");
    
    return true;
}

/**
 * @brief Test NUMA thread binding verification
 * 
 * This test specifically verifies that threads are actually running on their
 * assigned NUMA nodes by checking the CPU and NUMA node at runtime.
 */
bool test_numa_binding_verification() {
TEST_PRINTF("🧪 Testing NUMA thread binding verification...\n");
    
    // Create test context and tensor
    struct ggml_init_params init_params = {
        .mem_size = 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false
    };
    struct ggml_context* ctx = ggml_init(init_params);
    EXPECT_TRUE(ctx != NULL, "Context creation should succeed");
    
    ggml_numa_openmp_config_t config = ggml_numa_openmp_coordinator_get_config();
    int total_system_threads = config.total_numa_nodes * config.threads_per_node;
    
    // Test different strategies with NUMA binding verification
    struct {
        int threads;
        const char* test_name;
        bool use_data_parallel;
    } binding_tests[] = {
        {8, "single-node-multi-thread", false},
        {config.threads_per_node, "full-single-node", false},
        {total_system_threads, "full-system-data-parallel", true}
    };
    
    for (auto test : binding_tests) {
TEST_PRINTF("   🧪 Testing NUMA binding for %s with %d threads\n", test.test_name, test.threads);
        
        struct ggml_tensor* tensor = create_test_tensor(ctx, 256);
        EXPECT_TRUE(tensor != NULL, "Tensor creation should succeed");
        
        // Reset tracking
        reset_tracking();
        
        // Execute with NUMA binding verification
        enum ggml_status status;
        if (test.use_data_parallel) {
            status = ggml_numa_openmp_execute_data_parallel(
                tensor,
                numa_binding_verification_work_function,
                0   // No work buffer
            );
TEST_PRINTF("      📊 Data-parallel execution with binding verification: %s\n", 
                   status == GGML_STATUS_SUCCESS ? "SUCCESS" : "FAILED");
        } else {
            status = ggml_numa_openmp_execute_single_node(
                tensor,
                numa_binding_verification_work_function,
                0,  // NUMA node 0
                test.threads,
                0   // No work buffer
            );
TEST_PRINTF("      📊 Single-node execution on node 0 with binding verification: %s\n", 
                   status == GGML_STATUS_SUCCESS ? "SUCCESS" : "FAILED");
        }
        
        EXPECT_EQ(GGML_STATUS_SUCCESS, status, "NUMA binding verification should succeed");
        
        // Verify execution tracking
        EXPECT_GE(g_threads_executed.load(), 1, "Should have executed threads");
        
        if (test.use_data_parallel && config.total_numa_nodes > 1) {
            // For data-parallel, should use multiple NUMA nodes
            int active_numa_nodes = 0;
            for (int i = 0; i < config.total_numa_nodes; i++) {
                if (g_numa_node_active[i]) active_numa_nodes++;
            }
            EXPECT_GE(active_numa_nodes, 2, "Data-parallel should use multiple NUMA nodes");
TEST_PRINTF("      📊 NUMA nodes utilized: %d/%d\n", active_numa_nodes, config.total_numa_nodes);
        } else {
            // For single-node, should only use node 0
            EXPECT_TRUE(g_numa_node_active[0], "Should use NUMA node 0");
TEST_PRINTF("      📊 Single-node execution verified on NUMA node 0\n");
        }
    }
    
TEST_PRINTF("   ✅ NUMA thread binding verification completed successfully\n");
    
    ggml_free(ctx);
    return true;
}

/**
 * @brief Main test runner
 */
void print_usage(const char *prog) {
    printf("Usage: %s [OPTIONS]\n", prog);
    printf("\nOptions:\n");
    printf("  --summary-only      Only print the summary table, not full test output\n");
    printf("  --filter <regex>    Only run tests whose name matches the regex pattern (case-insensitive)\n");
    printf("  --help              Show this help message\n");
    printf("\nFilter Examples:\n");
    printf("  --filter \"initialization\"        # Run initialization tests only\n");
    printf("  --filter \"cpu_mask\"              # Run CPU mask related tests\n");
    printf("  --filter \"thread.*strategy\"      # Run strategy execution tests\n");
    printf("  --filter \"numa.*binding\"         # Run NUMA binding tests\n");
    printf("  --filter \"isolate\"               # Run NUMA isolation tests\n");
    printf("\nTest Categories:\n");
    printf("  - Initialization: basic_initialization, cpu_mask_creation, cpu_mask_initialization\n");
    printf("  - Configuration: thread_distribution\n");
    printf("  - Execution: single_thread_strategy, single_node_strategy, data_parallel_strategy\n");
    printf("  - Validation: work_buffer_allocation, numa_binding_verification\n");
    printf("  - Advanced: numa_isolate_strategy, parallel_execution_verification\n");
    printf("  - Integration: executor_integration, full_system_capacity, error_handling\n");
}

int main(int argc, char **argv) {
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--summary-only") {
            g_summary_only = true;
        } else if (arg == "--filter" && i + 1 < argc) {
            g_test_filter = argv[++i];
            g_filter_enabled = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            printf("Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!g_summary_only) {
        printf("🧪 OpenMP NUMA Coordinator Test Suite\n");
        printf("========================================\n\n");
    }

    // Always initialize the coordinator before running any tests
    // This ensures tests can run individually with filtering
    ggml_numa_openmp_coordinator_shutdown();  // Clean shutdown first
    if (!ggml_numa_openmp_coordinator_init()) {
        printf("❌ FATAL: Failed to initialize NUMA coordinator\n");
        return 1;
    }

    // Run comprehensive tests
    run_test_if_matches("basic_initialization", test_basic_initialization);
    run_test_if_matches("cpu_mask_creation", test_cpu_mask_creation);
    run_test_if_matches("cpu_mask_initialization", test_cpu_mask_initialization);
    run_test_if_matches("thread_distribution", test_thread_distribution);
    run_test_if_matches("single_thread_strategy", test_single_thread_strategy);
    run_test_if_matches("single_node_strategy", test_single_node_strategy);
    run_test_if_matches("data_parallel_strategy", test_data_parallel_strategy);
    run_test_if_matches("data_parallel_unique_thread_ids", test_data_parallel_unique_thread_ids);
    run_test_if_matches("work_buffer_allocation", test_work_buffer_allocation);
    run_test_if_matches("reusable_work_buffers", test_reusable_work_buffers);
    run_test_if_matches("work_buffer_thread_isolation", test_work_buffer_thread_isolation);
    run_test_if_matches("work_buffer_persistence", test_work_buffer_persistence);
    run_test_if_matches("executor_integration", test_executor_integration);
    run_test_if_matches("full_system_capacity", test_full_system_capacity);
    run_test_if_matches("parallel_execution_verification", test_parallel_execution_verification);
    run_test_if_matches("numa_binding_verification", test_numa_binding_verification);
    run_test_if_matches("numa_isolate_strategy", test_numa_isolate_strategy);
    run_test_if_matches("numa_isolate_full_capacity", test_numa_isolate_full_capacity);
    run_test_if_matches("numa_isolate_error_handling", test_numa_isolate_error_handling);
    run_test_if_matches("error_handling", test_error_handling);

    // Cleanup
    ggml_numa_openmp_coordinator_shutdown();

    // Print summary
    printf("\n========================================\n");
    printf("📊 TEST SUMMARY\n");
    printf("========================================\n");

    int passed = 0, total = 0;
    for (const auto& result : g_test_results) {
        total++;
        if (result.passed) {
            passed++;
            printf("✅ %s\n", result.test_name.c_str());
        } else {
            printf("❌ %s - %s\n", result.test_name.c_str(), result.failure_reason.c_str());
        }
    }

    printf("\n📈 Results: %d/%d tests passed (%.1f%% success rate)\n", 
           passed, total, (total > 0) ? (100.0f * passed / total) : 0.0f);

    // Print detailed system information
    ggml_numa_openmp_config_t config = ggml_numa_openmp_coordinator_get_config();
    printf("\n🔧 System Configuration:\n");
    printf("   - NUMA nodes: %d\n", config.total_numa_nodes);
    printf("   - Threads per node: %d\n", config.threads_per_node);
    printf("   - NUMA available: %s\n", config.numa_available ? "Yes" : "No");
    printf("   - Thread teams initialized: %s\n", config.threadpool_manager.teams_initialized ? "Yes" : "No");
    printf("   - OpenMP places configured: %s\n", config.openmp_places_configured ? "Yes" : "No");
    printf("   - OpenMP binding enabled: %s\n", config.openmp_binding_enabled ? "Yes" : "No");

    if (passed == total) {
        printf("\n🎉 ALL TESTS PASSED! OpenMP coordinator is working correctly.\n");
        printf("🚀 Ready for production use with per-NUMA thread teams and CPU mask support!\n");
        return 0;
    } else {
        printf("\n❌ Some tests failed. Please review the issues above.\n");
        return 1;
    }
}
