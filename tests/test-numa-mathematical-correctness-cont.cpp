/**
 * Mathematical Correctness Test for NUMA CONT Kernel
 * Validates tensor contiguity operation with comprehensive dimension coverage
 * 
 * CONT Operation: Makes tensors contiguous by copying data to contiguous memory layout
 * Mathematical Requirements: Exact element preservation during data copying
 * NUMA Implementation: Data-parallel copying with shared memory optimization
 */

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-numa-executor.h"
#include "ops.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <random>
#include <thread>
#include <chrono>

// Debug control - follows GGML_NUMA_DEBUG environment variable
static bool debug_enabled = false;

static void init_debug_control() {
    const char* debug_env = getenv("GGML_NUMA_DEBUG");
    debug_enabled = debug_env && (atoi(debug_env) > 0);
}

#define DEBUG_PRINT(...) do { if (debug_enabled) fprintf(stderr, __VA_ARGS__); } while(0)

// Test statistics tracking
struct test_stats_t {
    int tests_run = 0;
    int tests_passed = 0;
    int tests_failed = 0;
    std::vector<std::string> failure_details;
    
    void record_pass() { tests_run++; tests_passed++; }
    void record_fail(const std::string& detail) { 
        tests_run++; tests_failed++; 
        failure_details.push_back(detail);
    }
    
    void print_summary() {
        printf("\n=== CONT Mathematical Correctness Test Results ===\n");
        printf("Tests Run: %d\n", tests_run);
        printf("Passed: %d\n", tests_passed);
        printf("Failed: %d\n", tests_failed);
        if (tests_failed > 0) {
            printf("\nFailure Details:\n");
            for (const auto& detail : failure_details) {
                printf("  - %s\n", detail.c_str());
            }
        }
        printf("Overall Result: %s\n", (tests_failed == 0) ? "PASS" : "FAIL");
    }
};

static test_stats_t g_test_stats;

/**
 * Create a non-contiguous tensor for testing CONT operation
 * Uses permutation to create a tensor that requires contiguity correction
 */
static struct ggml_tensor* create_non_contiguous_tensor(
    struct ggml_context* ctx, 
    ggml_type type,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
    const char* test_name
) {
    // Create original tensor
    struct ggml_tensor* original = ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);
    
    // Fill with test data
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
    
    if (type == GGML_TYPE_F32) {
        float* data = (float*)ggml_get_data(original);
        for (int64_t i = 0; i < ggml_nelements(original); i++) {
            data[i] = dis(gen);
        }
    } else if (type == GGML_TYPE_F16) {
        ggml_fp16_t* data = (ggml_fp16_t*)ggml_get_data(original);
        for (int64_t i = 0; i < ggml_nelements(original); i++) {
            data[i] = ggml_fp32_to_fp16(dis(gen));
        }
    }
    
    // Create a permutation to make tensor non-contiguous
    // Only permute if we have multiple dimensions
    if (ne1 > 1 && ne0 > 1) {
        struct ggml_tensor* permuted = ggml_permute(ctx, original, 1, 0, 2, 3);
        DEBUG_PRINT("Created non-contiguous tensor for %s: [%ld,%ld,%ld,%ld] -> permuted\n", 
                   test_name, (long)ne0, (long)ne1, (long)ne2, (long)ne3);
        return permuted;
    } else {
        // For 1D tensors or cases where permutation doesn't help,
        // just return original (it's already contiguous, but CONT should handle it)
        DEBUG_PRINT("Created tensor for %s: [%ld,%ld,%ld,%ld] (already contiguous)\n", 
                   test_name, (long)ne0, (long)ne1, (long)ne2, (long)ne3);
        return original;
    }
}

/**
 * Test mathematical correctness of CONT operation
 * Validates that CONT preserves all elements during contiguity operation
 */
static void test_cont_mathematical_correctness(
    ggml_type type,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
    int n_threads,
    const char* test_name
) {
    DEBUG_PRINT("\n--- Testing CONT: %s ---\n", test_name);
    DEBUG_PRINT("Dimensions: [%ld, %ld, %ld, %ld], Threads: %d, Type: %d\n", 
               (long)ne0, (long)ne1, (long)ne2, (long)ne3, n_threads, type);
    
    // Create contexts
    size_t ctx_size = ggml_tensor_overhead() * 10 + ggml_type_size(type) * ne0 * ne1 * ne2 * ne3 * 4;
    struct ggml_init_params params = { ctx_size, NULL, false };
    struct ggml_context* ctx = ggml_init(params);
    
    if (!ctx) {
        g_test_stats.record_fail(std::string(test_name) + ": Failed to create context");
        return;
    }
    
    // Create non-contiguous input tensor
    struct ggml_tensor* input = create_non_contiguous_tensor(ctx, type, ne0, ne1, ne2, ne3, test_name);
    
    // Create CONT operation
    struct ggml_tensor* result = ggml_cont(ctx, input);
    
    // Set up NUMA computation parameters
    struct ggml_compute_params numa_params;
    numa_params.ith = 0;
    numa_params.nth = n_threads;
    numa_params.wsize = 0;
    numa_params.wdata = nullptr;
    numa_params.threadpool = nullptr;
    
    // Create minimal compute plan for single tensor execution
    struct ggml_cplan cplan = {};
    cplan.work_size = 0;
    cplan.work_data = nullptr;
    cplan.n_threads = n_threads;
    cplan.threadpool = nullptr;
    cplan.abort_callback = nullptr;
    cplan.abort_callback_data = nullptr;
    
    // Execute with NUMA executor architecture
    enum ggml_status dispatch_result = ggml_numa_executor_execute_tensor(result, &cplan);
    
    if (dispatch_result != GGML_STATUS_SUCCESS) {
        g_test_stats.record_fail(std::string(test_name) + ": NUMA dispatch failed: " + std::to_string(dispatch_result));
        ggml_free(ctx);
        return;
    }
    
    // Compute reference result (single-threaded, no NUMA)
    struct ggml_context* ref_ctx = ggml_init(params);
    struct ggml_tensor* ref_input = create_non_contiguous_tensor(ref_ctx, type, ne0, ne1, ne2, ne3, "reference");
    struct ggml_tensor* ref_result = ggml_cont(ref_ctx, ref_input);
    
    struct ggml_compute_params ref_params;
    ref_params.ith = 0;
    ref_params.nth = 1;
    ref_params.wsize = 0;
    ref_params.wdata = nullptr;
    ref_params.threadpool = nullptr;
    
    // Execute reference computation (ensure NUMA is disabled)
    setenv("GGML_NUMA_STRATEGY", "disabled", 1);
    ggml_compute_forward_dup(&ref_params, ref_result);
    unsetenv("GGML_NUMA_STRATEGY");
    
    // Compare results
    bool test_passed = true;
    std::string failure_reason;
    
    size_t total_elements = ggml_nelements(result);
    DEBUG_PRINT("Comparing %zu elements\n", total_elements);
    
    if (type == GGML_TYPE_F32) {
        const float* numa_data = (const float*)ggml_get_data(result);
        const float* ref_data = (const float*)ggml_get_data(ref_result);
        
        for (size_t i = 0; i < total_elements; i++) {
            float numa_val = numa_data[i];
            float ref_val = ref_data[i];
            
            // CONT should preserve exact values (no mathematical operations)
            if (numa_val != ref_val) {
                test_passed = false;
                failure_reason = "Element mismatch at index " + std::to_string(i) + 
                               ": NUMA=" + std::to_string(numa_val) + 
                               ", Reference=" + std::to_string(ref_val);
                break;
            }
        }
    } else if (type == GGML_TYPE_F16) {
        const ggml_fp16_t* numa_data = (const ggml_fp16_t*)ggml_get_data(result);
        const ggml_fp16_t* ref_data = (const ggml_fp16_t*)ggml_get_data(ref_result);
        
        for (size_t i = 0; i < total_elements; i++) {
            float numa_val = ggml_fp16_to_fp32(numa_data[i]);
            float ref_val = ggml_fp16_to_fp32(ref_data[i]);
            
            // CONT should preserve exact values
            if (numa_val != ref_val) {
                test_passed = false;
                failure_reason = "Element mismatch at index " + std::to_string(i) + 
                               ": NUMA=" + std::to_string(numa_val) + 
                               ", Reference=" + std::to_string(ref_val);
                break;
            }
        }
    }
    
    // Check contiguity property
    if (test_passed) {
        bool is_contiguous = ggml_is_contiguous(result);
        if (!is_contiguous) {
            test_passed = false;
            failure_reason = "Result tensor is not contiguous after CONT operation";
        } else {
            DEBUG_PRINT("Contiguity verified: result tensor is contiguous\n");
        }
    }
    
    // Record test result
    if (test_passed) {
        g_test_stats.record_pass();
        DEBUG_PRINT("Test PASSED: %s\n", test_name);
    } else {
        g_test_stats.record_fail(std::string(test_name) + ": " + failure_reason);
        DEBUG_PRINT("Test FAILED: %s - %s\n", test_name, failure_reason.c_str());
    }
    
    // Cleanup
    ggml_free(ref_ctx);
    ggml_free(ctx);
}

/**
 * Comprehensive test suite covering multiple dimensions and thread counts
 */
static void run_comprehensive_tests() {
    printf("Running comprehensive CONT mathematical correctness tests...\n");
    
    // Test configurations: dimensions and thread counts
    struct test_config {
        int64_t ne0, ne1, ne2, ne3;
        const char* name;
    };
    
    // Multi-dimensional test cases covering various complexity levels
    std::vector<test_config> test_configs = {
        // 1D tensors (edge cases)
        {1024, 1, 1, 1, "1D_Small"},
        {65536, 1, 1, 1, "1D_Medium"},
        {1048576, 1, 1, 1, "1D_Large"},
        
        // 2D tensors (matrix-like)
        {32, 32, 1, 1, "2D_Square_Small"},
        {128, 128, 1, 1, "2D_Square_Medium"},
        {256, 512, 1, 1, "2D_Rectangular"},
        {1024, 768, 1, 1, "2D_Large"},
        
        // 3D tensors (typical for transformers)
        {64, 64, 16, 1, "3D_Medium"},
        {128, 256, 32, 1, "3D_Large"},
        {512, 128, 8, 1, "3D_Wide"},
        
        // 4D tensors (batch processing)
        {32, 32, 8, 4, "4D_Small_Batch"},
        {64, 128, 16, 8, "4D_Medium_Batch"},
        {128, 256, 12, 4, "4D_Large_Batch"},
        
        // Extreme cases
        {2048, 2048, 1, 1, "2D_Very_Large"},
        {4096, 1024, 1, 1, "2D_Ultra_Large"},
    };
    
    std::vector<int> thread_counts = {1, 2, 4, 6, 8};
    std::vector<ggml_type> types = {GGML_TYPE_F32, GGML_TYPE_F16};
    
    // Run tests for each configuration
    for (const auto& config : test_configs) {
        for (int threads : thread_counts) {
            for (ggml_type type : types) {
                std::string test_name = std::string(config.name) + "_T" + std::to_string(threads) + 
                                      "_" + (type == GGML_TYPE_F32 ? "F32" : "F16");
                test_cont_mathematical_correctness(
                    type, config.ne0, config.ne1, config.ne2, config.ne3, 
                    threads, test_name.c_str()
                );
            }
        }
    }
}

/**
 * Performance stress test with larger tensors
 */
static void run_performance_stress_tests() {
    printf("\nRunning CONT performance stress tests...\n");
    
    struct stress_config {
        int64_t ne0, ne1, ne2, ne3;
        const char* name;
    };
    
    std::vector<stress_config> stress_configs = {
        // GB-scale tensors for stress testing
        {8192, 8192, 1, 1, "Stress_64MB"},
        {11585, 11585, 1, 1, "Stress_512MB"},  // ~512MB tensor
        {16384, 8192, 1, 1, "Stress_1GB"},     // ~1GB tensor
    };
    
    for (const auto& config : stress_configs) {
        // Test with maximum threads for stress testing
        int max_threads = std::thread::hardware_concurrency();
        test_cont_mathematical_correctness(
            GGML_TYPE_F32, config.ne0, config.ne1, config.ne2, config.ne3,
            max_threads, config.name
        );
    }
}

/**
 * Regression tests for known edge cases
 */
static void run_regression_tests() {
    printf("\nRunning CONT regression tests...\n");
    
    // Test edge cases that could cause issues
    test_cont_mathematical_correctness(GGML_TYPE_F32, 1, 1, 1, 1, 1, "Regression_Single_Element");
    test_cont_mathematical_correctness(GGML_TYPE_F32, 3, 1, 1, 1, 2, "Regression_Odd_Size");
    test_cont_mathematical_correctness(GGML_TYPE_F32, 7, 11, 1, 1, 3, "Regression_Prime_Dimensions");
    test_cont_mathematical_correctness(GGML_TYPE_F16, 4097, 1, 1, 1, 4, "Regression_Large_Prime_F16");
    
    // Test already contiguous tensors (should be no-op but still work)
    test_cont_mathematical_correctness(GGML_TYPE_F32, 1024, 1, 1, 1, 2, "Regression_Already_Contiguous");
}

int main() {
    init_debug_control();
    
    printf("NUMA CONT Kernel Mathematical Correctness Test\n");
    printf("==============================================\n");
    printf("Debug output: %s\n", debug_enabled ? "ENABLED" : "DISABLED");
    printf("Hardware threads: %u\n", std::thread::hardware_concurrency());
    
    // Initialize GGML
    ggml_time_init();
    
    // Run test suites
    run_comprehensive_tests();
    run_performance_stress_tests();
    run_regression_tests();
    
    // Print final results
    g_test_stats.print_summary();
    
    // Return appropriate exit code
    return (g_test_stats.tests_failed == 0) ? 0 : 1;
}
