/**
 * @file test-ggml-openmp-coordinator-clean.cpp
 * @brief Clean test suite for OpenMP-based NUMA coordinator
 * 
 * Tests the current functionality of the refactored NUMA coordinator:
 * - Basic initialization and configuration
 * - Three execution strategies (single-thread, single-node, data-parallel)
 * - Work buffer management
 * - Integration with tensor operations
 * 
 * @author David Sanftenberg
 */

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <atomic>
#include <chrono>
#include <thread>

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-cpu/ggml-numa-openmp-coordinator.h"
#include "ggml-cpu/ggml-numa-executor.h"
#include "ggml-cpu/ggml-numa-shared.h"

#ifdef __linux__
#include <numa.h>
#include <unistd.h>
#endif

// Test result tracking
struct TestResult {
    std::string test_name;
    bool passed;
    std::string failure_reason;
};

std::vector<TestResult> g_test_results;
bool g_summary_only = false;

#define TEST_PRINTF(...) do { if (!g_summary_only) printf(__VA_ARGS__); } while(0)

/**
 * @brief Record test result
 */
void record_test_result(const std::string& test_name, bool passed, const std::string& failure_reason = "") {
    g_test_results.push_back({test_name, passed, failure_reason});
    if (passed) {
        TEST_PRINTF("   ✅ %s PASSED\n", test_name.c_str());
    } else {
        TEST_PRINTF("   ❌ %s FAILED: %s\n", test_name.c_str(), failure_reason.c_str());
    }
}

/**
 * @brief Test basic coordinator initialization
 */
bool test_basic_initialization() {
    TEST_PRINTF("🧪 Testing basic coordinator initialization...\n");
    
    // Shutdown any existing coordinator first
    ggml_numa_openmp_coordinator_shutdown();
    
    // Test initialization
    bool init_result = ggml_numa_openmp_coordinator_init();
    if (!init_result) {
        record_test_result("basic_initialization", false, "Initialization failed");
        return false;
    }
    
    // Get configuration
    ggml_numa_openmp_config_t config = ggml_numa_openmp_coordinator_get_config();
    
    // Basic validation
    if (!config.initialized) {
        record_test_result("basic_initialization", false, "Coordinator not marked as initialized");
        return false;
    }
    
    if (config.total_numa_nodes < 1) {
        record_test_result("basic_initialization", false, "Invalid NUMA node count");
        return false;
    }
    
    if (config.threads_per_node < 1) {
        record_test_result("basic_initialization", false, "Invalid threads per node count");
        return false;
    }
    
    TEST_PRINTF("   📊 Configuration: %d NUMA nodes, %d threads per node\n", 
           config.total_numa_nodes, config.threads_per_node);
    TEST_PRINTF("   📊 NUMA available: %s\n", config.numa_available ? "Yes" : "No");
    
    record_test_result("basic_initialization", true);
    return true;
}

/**
 * @brief Simple work function for testing
 */
enum ggml_status test_work_function(void * work_context, struct ggml_compute_params * params) {
    struct ggml_tensor * tensor = (struct ggml_tensor *)work_context;
    
    // Simple validation
    if (!tensor || !params) {
        return GGML_STATUS_FAILED;
    }
    
    // Mark that this thread did work (for validation)
    static std::atomic<int> work_counter{0};
    work_counter++;
    
    TEST_PRINTF("     Thread %d/%d completed work (total completions: %d)\n", 
           params->ith, params->nth, work_counter.load());
    
    return GGML_STATUS_SUCCESS;
}

/**
 * @brief Test single-thread execution strategy
 */
bool test_single_thread_strategy() {
    TEST_PRINTF("🧪 Testing single-thread strategy...\n");
    
    // Create a simple tensor for testing
    struct ggml_context * ctx = nullptr;
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    ctx = ggml_init(params);
    
    if (!ctx) {
        record_test_result("single_thread_strategy", false, "Failed to create ggml context");
        return false;
    }
    
    struct ggml_tensor * tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 16);
    if (!tensor) {
        ggml_free(ctx);
        record_test_result("single_thread_strategy", false, "Failed to create tensor");
        return false;
    }
    
    // Test execution on node 0
    enum ggml_status result = ggml_numa_openmp_execute_single_thread(
        tensor, test_work_function, 0, nullptr);
    
    ggml_free(ctx);
    
    if (result != GGML_STATUS_SUCCESS) {
        record_test_result("single_thread_strategy", false, "Single-thread execution failed");
        return false;
    }
    
    record_test_result("single_thread_strategy", true);
    return true;
}

/**
 * @brief Test single-node multi-thread strategy
 */
bool test_single_node_strategy() {
    TEST_PRINTF("🧪 Testing single-node strategy...\n");
    
    // Create a simple tensor for testing
    struct ggml_context * ctx = nullptr;
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    ctx = ggml_init(params);
    
    if (!ctx) {
        record_test_result("single_node_strategy", false, "Failed to create ggml context");
        return false;
    }
    
    struct ggml_tensor * tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 32);
    if (!tensor) {
        ggml_free(ctx);
        record_test_result("single_node_strategy", false, "Failed to create tensor");
        return false;
    }
    
    // Test execution on node 0
    enum ggml_status result = ggml_numa_openmp_execute_single_node(
        tensor, test_work_function, 0, nullptr);
    
    ggml_free(ctx);
    
    if (result != GGML_STATUS_SUCCESS) {
        record_test_result("single_node_strategy", false, "Single-node execution failed");
        return false;
    }
    
    record_test_result("single_node_strategy", true);
    return true;
}

/**
 * @brief Test data-parallel multi-node strategy
 */
bool test_data_parallel_strategy() {
    TEST_PRINTF("🧪 Testing data-parallel strategy...\n");
    
    // Create a simple tensor for testing
    struct ggml_context * ctx = nullptr;
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    ctx = ggml_init(params);
    
    if (!ctx) {
        record_test_result("data_parallel_strategy", false, "Failed to create ggml context");
        return false;
    }
    
    struct ggml_tensor * tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    if (!tensor) {
        ggml_free(ctx);
        record_test_result("data_parallel_strategy", false, "Failed to create tensor");
        return false;
    }
    
    // Test data-parallel execution
    enum ggml_status result = ggml_numa_openmp_execute_data_parallel(
        tensor, test_work_function, nullptr);
    
    ggml_free(ctx);
    
    if (result != GGML_STATUS_SUCCESS) {
        record_test_result("data_parallel_strategy", false, "Data-parallel execution failed");
        return false;
    }
    
    record_test_result("data_parallel_strategy", true);
    return true;
}

/**
 * @brief Test work buffer calculation function
 */
size_t test_work_buffer_calc(const struct ggml_tensor * tensor, int total_numa_nodes, int total_threads) {
    // Simple calculation: 1KB per thread for testing
    size_t per_thread_buffer = 1024;
    return per_thread_buffer * total_threads;
}

/**
 * @brief Test work buffer allocation
 */
bool test_work_buffer_allocation() {
    TEST_PRINTF("🧪 Testing work buffer allocation...\n");
    
    // Create a simple tensor for testing
    struct ggml_context * ctx = nullptr;
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    ctx = ggml_init(params);
    
    if (!ctx) {
        record_test_result("work_buffer_allocation", false, "Failed to create ggml context");
        return false;
    }
    
    struct ggml_tensor * tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 32);
    if (!tensor) {
        ggml_free(ctx);
        record_test_result("work_buffer_allocation", false, "Failed to create tensor");
        return false;
    }
    
    // Test execution with work buffer calculation
    enum ggml_status result = ggml_numa_openmp_execute_single_node(
        tensor, test_work_function, 0, test_work_buffer_calc);
    
    ggml_free(ctx);
    
    if (result != GGML_STATUS_SUCCESS) {
        record_test_result("work_buffer_allocation", false, "Work buffer execution failed");
        return false;
    }
    
    record_test_result("work_buffer_allocation", true);
    return true;
}

/**
 * @brief Test configuration access
 */
bool test_configuration_access() {
    TEST_PRINTF("🧪 Testing configuration access...\n");
    
    ggml_numa_openmp_config_t config = ggml_numa_openmp_coordinator_get_config();
    
    // Basic validation that we can access configuration
    if (config.total_numa_nodes < 1 || config.threads_per_node < 1) {
        record_test_result("configuration_access", false, "Invalid configuration values");
        return false;
    }
    
    int num_nodes = ggml_numa_openmp_coordinator_get_num_nodes();
    if (num_nodes != config.total_numa_nodes) {
        record_test_result("configuration_access", false, "Inconsistent NUMA node count");
        return false;
    }
    
    TEST_PRINTF("   📊 NUMA nodes: %d\n", config.total_numa_nodes);
    TEST_PRINTF("   📊 Threads per node: %d\n", config.threads_per_node);
    TEST_PRINTF("   📊 NUMA available: %s\n", config.numa_available ? "Yes" : "No");
    TEST_PRINTF("   📊 Initialized: %s\n", config.initialized ? "Yes" : "No");
    
    record_test_result("configuration_access", true);
    return true;
}

/**
 * @brief Main test runner
 */
void run_test(const std::string& test_name, bool (*test_func)()) {
    TEST_PRINTF("\n🔍 Running test: %s\n", test_name.c_str());
    
    try {
        bool success = test_func();
        if (!success && g_test_results.empty()) {
            // If test function didn't record a result, record a generic failure
            record_test_result(test_name, false, "Test function returned false");
        }
    } catch (const std::exception& e) {
        record_test_result(test_name, false, std::string("Exception: ") + e.what());
    } catch (...) {
        record_test_result(test_name, false, "Unknown exception");
    }
}

/**
 * @brief Print usage information
 */
void print_usage(const char *prog) {
    printf("Usage: %s [OPTIONS]\n", prog);
    printf("\nOptions:\n");
    printf("  --summary-only      Only print the summary, not detailed test output\n");
    printf("  --help              Show this help message\n");
    printf("\nThis test validates the refactored NUMA OpenMP coordinator functionality.\n");
}

/**
 * @brief Main entry point
 */
int main(int argc, char **argv) {
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--summary-only") {
            g_summary_only = true;
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
        printf("🧪 Clean OpenMP NUMA Coordinator Test Suite\n");
        printf("===========================================\n");
    }

    // Initialize coordinator before running tests
    if (!ggml_numa_openmp_coordinator_init()) {
        printf("❌ FATAL: Failed to initialize NUMA coordinator\n");
        return 1;
    }

    // Run tests
    run_test("basic_initialization", test_basic_initialization);
    run_test("configuration_access", test_configuration_access);
    run_test("single_thread_strategy", test_single_thread_strategy);
    run_test("single_node_strategy", test_single_node_strategy);
    run_test("data_parallel_strategy", test_data_parallel_strategy);
    run_test("work_buffer_allocation", test_work_buffer_allocation);

    // Cleanup
    ggml_numa_openmp_coordinator_shutdown();

    // Print summary
    printf("\n===========================================\n");
    printf("📊 TEST SUMMARY\n");
    printf("===========================================\n");

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

    if (passed == total) {
        printf("\n🎉 ALL TESTS PASSED! Refactored OpenMP coordinator is working correctly.\n");
        return 0;
    } else {
        printf("\n❌ Some tests failed. Please review the issues above.\n");
        return 1;
    }
}
