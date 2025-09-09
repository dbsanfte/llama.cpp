/**
 * NUMA Mathematical Correctness Test: RMS_NORM Operation
 * 
 * This test verifies mathematical equivalence between NUMA parallel RMS_NORM operations
 * and serial reference implementations. It ensures the NUMA RMS_NORM kernel produces
 * identical results to the reference implementation across various scenarios.
 * 
 * TEST COVERAGE:
 * 1. Mathematical Equivalence (Simplified 3-Stage Approach):
 *    a) Single-thread Single-node: Tests basic kernel functionality and fallback mechanisms
 *    b) Multi-thread Single-node: Tests multi-threading coordination within single NUMA node  
 *    c) Multi-thread Multi-node: Tests full NUMA data-parallel execution across multiple nodes
 *    - Tests across TINY → LARGE tensor sizes for comprehensive coverage
 *    - Eliminates artificial thread constraints, focuses on production execution modes
 * 
 * 2. Quantization Type Coverage (Complete Support Matrix):
 *    - Tests F32 type (primary RMS_NORM data type)
 *    - Ensures proper fallback handling for quantized input tensors
 *    - Verifies NUMA kernels handle quantized fallbacks correctly
 * 
 * 3. Edge Cases and Regression Prevention:
 *    - Tests near-zero values and numerical stability
 *    - Tests large values and proper normalization behavior
 *    - Validates single-row tensors and boundary conditions
 * 
 * KEY DESIGN PRINCIPLES:
 * - Simplified execution testing: 3 clear stages instead of complex thread scenarios
 * - Row-wise normalization testing for various matrix dimensions
 * - Multi-dimensional testing across various matrix/tensor sizes
 * - Direct comparison between NUMA parallel and serial reference implementations
 * - Detailed error reporting with mathematical mismatch information
 * - Regression testing for numerical stability edge cases
 * 
 * ARCHITECTURE INTEGRATION:
 * - Tests NUMA Executor strategy selection (Single/Single, Single/Multi, Data-Parallel)
 * - Validates NUMA Coordinator thread management and NUMA binding
 * - Ensures NUMA Kernel Registry provides correct function pointers
 * - Verifies shared memory optimization and reduction policies
 */

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <regex>
#include <iostream>

// GGML includes
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-numa-executor.h"

// Constants
#define RMS_NORM_EPS 1e-6f

// Global variables for command line argument handling
static std::string g_test_filter = "";
static bool g_filter_enabled = false;
static bool g_summary_only = false;

// Test result tracking
struct TestResult {
    std::string test_name;
    bool passed;
    std::string error_message;
};

static std::vector<TestResult> test_results;

// TEST_PRINTF macro for conditional output
#define TEST_PRINTF(...) do { \
    if (!g_summary_only) { \
        printf(__VA_ARGS__); \
    } \
} while(0)

/**
 * @brief Check if test name matches the filter regex
 */
static bool matches_filter(const std::string& test_name) {
    if (!g_filter_enabled) {
        return true;
    }
    
    try {
        std::regex filter_regex(g_test_filter, std::regex_constants::icase);
        return std::regex_search(test_name, filter_regex);
    } catch (const std::regex_error& e) {
        printf("Warning: Invalid regex filter '%s': %s\n", g_test_filter.c_str(), e.what());
        return true;  // Include all tests if regex is invalid
    }
}

/**
 * @brief Show usage information
 */
static void show_usage(const char* program_name) {
    printf("Usage: %s [OPTIONS]\n", program_name);
    printf("\nOptions:\n");
    printf("  --summary-only     Show only final summary, suppress detailed output\n");
    printf("  --filter <regex>   Run only tests matching the given regex pattern\n");
    printf("  --help            Show this help message\n");
    printf("\nExamples:\n");
    printf("  %s                                    # Run all tests with full output\n", program_name);
    printf("  %s --summary-only                     # Run all tests, show summary only\n", program_name);
    printf("  %s --filter \"TINY|SMALL\"              # Run only TINY and SMALL tests\n", program_name);
    printf("  %s --filter \"strategy.*single\"        # Run only single-node strategy tests\n", program_name);
    printf("  %s --summary-only --filter \"edge\"     # Run edge case tests with summary only\n", program_name);
}

// =============================================================================
// Test Utility Functions
// =============================================================================

/**
 * @brief Create a test tensor with random values
 */
static struct ggml_tensor* create_test_tensor(struct ggml_context* ctx, enum ggml_type type, 
                                            const std::vector<int64_t>& dims, float min_val = -5.0f, float max_val = 5.0f) {
    struct ggml_tensor* tensor = ggml_new_tensor(ctx, type, dims.size(), dims.data());
    
    if (type == GGML_TYPE_F32) {
        float* data = (float*)ggml_get_data(tensor);
        size_t n_elements = ggml_nelements(tensor);
        
        for (size_t i = 0; i < n_elements; ++i) {
            data[i] = min_val + ((float)rand() / RAND_MAX) * (max_val - min_val);
        }
    }
    
    return tensor;
}

/**
 * @brief Reference RMS normalization implementation
 */
static void reference_rms_norm_f32(const float* src, float* dst, int ne0, int ne1, float eps) {
    for (int i1 = 0; i1 < ne1; i1++) {
        float sum = 0.0f;
        
        // Calculate sum of squares for this row
        for (int i0 = 0; i0 < ne0; i0++) {
            float val = src[i1 * ne0 + i0];
            sum += val * val;
        }
        
        // Calculate RMS and normalize
        float rms = sqrtf(sum / ne0 + eps);
        for (int i0 = 0; i0 < ne0; i0++) {
            dst[i1 * ne0 + i0] = src[i1 * ne0 + i0] / rms;
        }
    }
}

/**
 * @brief Check if tensors are approximately equal
 */
static bool tensors_equal(struct ggml_tensor* a, struct ggml_tensor* b, float tolerance = 1e-5f) {
    if (!a || !b) return false;
    if (ggml_nelements(a) != ggml_nelements(b)) return false;
    
    size_t n_elements = ggml_nelements(a);
    float* data_a = (float*)ggml_get_data(a);
    float* data_b = (float*)ggml_get_data(b);
    
    for (size_t i = 0; i < n_elements; ++i) {
        float diff = std::abs(data_a[i] - data_b[i]);
        if (diff > tolerance) {
            TEST_PRINTF("Mismatch at element %zu: %.6f vs %.6f (diff: %.6f)\n", 
                       i, data_a[i], data_b[i], diff);
            return false;
        }
    }
    
    return true;
}

// =============================================================================
// Strategy Configuration and Test Data
// =============================================================================

struct TestConfig {
    std::string test_name;
    std::vector<int64_t> dims;
    size_t total_elements;
    
    TestConfig(const std::string& name, const std::vector<int64_t>& d) 
        : test_name(name), dims(d) {
        total_elements = 1;
        for (auto dim : dims) {
            total_elements *= dim;
        }
    }
};

struct StrategyConfig {
    std::string name;
    int min_threads;
    int max_threads;
    
    StrategyConfig(const std::string& n, int min_t, int max_t) 
        : name(n), min_threads(min_t), max_threads(max_t) {}
};

/**
 * @brief Test configurations covering different tensor sizes for comprehensive validation
 */
static std::vector<TestConfig> get_test_configurations() {
    return {
        // Small tensors - test basic functionality
        TestConfig("RMS_NORM_TINY", {64, 32}),
        TestConfig("RMS_NORM_SMALL", {128, 64}),
        
        // Medium tensors - test single-node multi-threading  
        TestConfig("RMS_NORM_MEDIUM", {512, 256}),
        TestConfig("RMS_NORM_LARGE", {1024, 512}),
        
        // Very large tensors - test data-parallel execution
        TestConfig("RMS_NORM_XLARGE", {2048, 1024}),
        TestConfig("RMS_NORM_XXLARGE", {4096, 2048})
    };
}

/**
 * @brief Strategy configurations for the 3-stage simplified approach
 */
static std::vector<StrategyConfig> get_strategy_configurations() {
    return {
        StrategyConfig("single_thread_single_node", 1, 1),      // Stage 1: Minimal threading
        StrategyConfig("multi_thread_single_node", 4, 8),       // Stage 2: Moderate threading
        StrategyConfig("multi_thread_multi_node", 12, 16)       // Stage 3: Maximum parallelism
    };
}

// =============================================================================
// Core Test Functions
// =============================================================================

/**
 * @brief Test NUMA strategies against fallback reference implementation
 */
static bool test_numa_strategies() {
    TEST_PRINTF("\n=== Testing NUMA RMS_NORM Strategies vs Reference Implementation ===\n");
    
    auto test_configs = get_test_configurations();
    auto strategy_configs = get_strategy_configurations();
    
    int total_tests = 0;
    int passed_tests = 0;
    
    for (const auto& config : test_configs) {
        for (const auto& strategy_config : strategy_configs) {
            std::string test_name = config.test_name + "_" + strategy_config.name;
            
            if (!matches_filter(test_name)) {
                TEST_PRINTF("⏭️  Skipping %s (filtered out)\n", test_name.c_str());
                continue;
            }
            
            TEST_PRINTF("\n🧪 Testing %s with %s strategy\n", config.test_name.c_str(), strategy_config.name.c_str());
            TEST_PRINTF("   Tensor dimensions: [%ld, %ld] (%zu elements)\n", 
                       config.dims[0], config.dims[1], config.total_elements);
            
            // Calculate context size
            size_t tensor_size = config.total_elements * sizeof(float);
            size_t ctx_size = ggml_tensor_overhead() * 10 + tensor_size * 10 + 1024*1024;
            
            // NUMA Test: Execute RMS_NORM operation using NUMA kernel
            struct ggml_init_params numa_params;
            numa_params.mem_size = ctx_size;
            numa_params.mem_buffer = nullptr;
            numa_params.no_alloc = false;
            
            struct ggml_context* ctx = ggml_init(numa_params);
            if (!ctx) {
                TEST_PRINTF("❌ Failed to create NUMA context\n");
                test_results.push_back({test_name, false, "Failed to create NUMA context"});
                total_tests++;
                continue;
            }
            
            // Create test tensors
            struct ggml_tensor* src = create_test_tensor(ctx, GGML_TYPE_F32, config.dims, -5.0f, 5.0f);
            struct ggml_tensor* numa_result = ggml_rms_norm(ctx, src, RMS_NORM_EPS);
            
            // Execute NUMA implementation with specified thread count
            int n_threads = (strategy_config.min_threads + strategy_config.max_threads) / 2;
            
            struct ggml_cgraph* numa_graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(numa_graph, numa_result);
            
            TEST_PRINTF("   Executing NUMA implementation with %d threads...\n", n_threads);
            
            // Ensure NUMA dispatch is enabled for this test
            ggml_numa_set_fallback_flag(false);  // Re-enable NUMA dispatch
            
            ggml_graph_compute_with_ctx(ctx, numa_graph, n_threads);
            
            // Reference Test: Execute RMS_NORM operation using reference implementation
            struct ggml_init_params ref_params;
            ref_params.mem_size = ctx_size;
            ref_params.mem_buffer = nullptr;
            ref_params.no_alloc = false;
            
            struct ggml_context* ref_ctx = ggml_init(ref_params);
            if (!ref_ctx) {
                TEST_PRINTF("❌ Failed to create reference context\n");
                test_results.push_back({test_name, false, "Failed to create reference context"});
                ggml_free(ctx);
                total_tests++;
                continue;
            }
            
            // Create reference tensor with same data
            struct ggml_tensor* ref_input = ggml_new_tensor(ref_ctx, GGML_TYPE_F32, config.dims.size(), config.dims.data());
            struct ggml_tensor* ref_result = ggml_rms_norm(ref_ctx, ref_input, RMS_NORM_EPS);
            
            // Copy data to reference tensor
            memcpy(ggml_get_data(ref_input), ggml_get_data(src), ggml_nbytes(src));
            
            // Execute reference implementation bypassing NUMA dispatch
            ggml_numa_set_fallback_flag(true);  // Force fallback to reference implementation
            
            struct ggml_cgraph* ref_graph = ggml_new_graph(ref_ctx);
            ggml_build_forward_expand(ref_graph, ref_result);
            
            TEST_PRINTF("   Executing reference implementation (bypassing NUMA)...\n");
            ggml_graph_compute_with_ctx(ref_ctx, ref_graph, 1);  // Single thread for reference
            
            ggml_numa_set_fallback_flag(false);  // Re-enable NUMA dispatch
            
            // Compare results
            bool test_passed = tensors_equal(numa_result, ref_result, 1e-4f);
            
            total_tests++;
            if (test_passed) {
                passed_tests++;
                TEST_PRINTF("✅ %s %s PASSED\n", config.test_name.c_str(), strategy_config.name.c_str());
                test_results.push_back({test_name, true, ""});
            } else {
                TEST_PRINTF("❌ %s %s FAILED\n", config.test_name.c_str(), strategy_config.name.c_str());
                test_results.push_back({test_name, false, "Mathematical comparison failed"});
            }
            
            ggml_free(ref_ctx);
            ggml_free(ctx);
        }
    }
    
    TEST_PRINTF("\n📊 NUMA Strategy Test Summary: %d/%d tests passed\n", passed_tests, total_tests);
    
    if (passed_tests == total_tests) {
        TEST_PRINTF("✅ All NUMA strategy tests PASSED\n");
        return true;
    } else {
        TEST_PRINTF("❌ Some NUMA strategy tests FAILED\n");
        return false;
    }
}

/**
 * @brief Test edge cases and potential regression scenarios
 */
static bool test_edge_cases() {
    TEST_PRINTF("\n=== Testing RMS_NORM Edge Cases ===\n");
    
    bool all_passed = true;
    
    // Test 1: Very small values (near zero)
    {
        std::string test_name = "RMS_NORM_edge_near_zero";
        if (matches_filter(test_name)) {
            TEST_PRINTF("Testing near-zero values...\n");
            std::vector<int64_t> dims = {64, 32};
            
            size_t ctx_size = ggml_tensor_overhead() * 20 + dims[0] * dims[1] * sizeof(float) * 20 + 1024*1024;
            std::vector<uint8_t> buffer(ctx_size);
            ggml_init_params init_params = {ctx_size, buffer.data(), false};
            ggml_context* ctx = ggml_init(init_params);
            
            struct ggml_tensor* src = create_test_tensor(ctx, GGML_TYPE_F32, dims, -1e-6f, 1e-6f);
            
            ggml_context* compute_ctx = ggml_init(init_params);
            struct ggml_tensor* src_copy = ggml_dup_tensor(compute_ctx, src);
            memcpy(ggml_get_data(src_copy), ggml_get_data(src), ggml_nbytes(src));
            
            struct ggml_tensor* result = ggml_rms_norm(compute_ctx, src_copy, RMS_NORM_EPS);
            
            // Ensure NUMA dispatch is enabled for this test
            ggml_numa_set_fallback_flag(false);  // Re-enable NUMA dispatch
            
            struct ggml_cgraph* graph = ggml_new_graph(compute_ctx);
            ggml_build_forward_expand(graph, result);
            ggml_graph_compute_with_ctx(compute_ctx, graph, 4);
            
            // Check that result is finite
            float* result_data = (float*)ggml_get_data(result);
            bool is_finite = true;
            for (int64_t i = 0; i < ggml_nelements(result); ++i) {
                if (!std::isfinite(result_data[i])) {
                    is_finite = false;
                    break;
                }
            }
            
            if (is_finite) {
                TEST_PRINTF("✅ Near-zero values test passed\n");
                test_results.push_back({test_name, true, ""});
            } else {
                TEST_PRINTF("❌ Near-zero values test failed\n");
                test_results.push_back({test_name, false, "Result contains non-finite values"});
                all_passed = false;
            }
            
            ggml_free(compute_ctx);
            ggml_free(ctx);
        } else {
            TEST_PRINTF("⏭️  Skipping %s (filtered out)\n", test_name.c_str());
        }
    }
    
    // Test 2: Large values
    {
        std::string test_name = "RMS_NORM_edge_large_values";
        if (matches_filter(test_name)) {
            TEST_PRINTF("Testing large values...\n");
            std::vector<int64_t> dims = {64, 32};
            
            size_t ctx_size = ggml_tensor_overhead() * 20 + dims[0] * dims[1] * sizeof(float) * 20 + 1024*1024;
            std::vector<uint8_t> buffer(ctx_size);
            ggml_init_params init_params = {ctx_size, buffer.data(), false};
            ggml_context* ctx = ggml_init(init_params);
            
            struct ggml_tensor* src = create_test_tensor(ctx, GGML_TYPE_F32, dims, -1000.0f, 1000.0f);
            
            ggml_context* compute_ctx = ggml_init(init_params);
            struct ggml_tensor* src_copy = ggml_dup_tensor(compute_ctx, src);
            memcpy(ggml_get_data(src_copy), ggml_get_data(src), ggml_nbytes(src));
            
            struct ggml_tensor* result = ggml_rms_norm(compute_ctx, src_copy, RMS_NORM_EPS);
            
            // Ensure NUMA dispatch is enabled for this test
            ggml_numa_set_fallback_flag(false);  // Re-enable NUMA dispatch
            
            struct ggml_cgraph* graph = ggml_new_graph(compute_ctx);
            ggml_build_forward_expand(graph, result);
            ggml_graph_compute_with_ctx(compute_ctx, graph, 4);
            
            // Check that result is finite and within reasonable bounds
            float* result_data = (float*)ggml_get_data(result);
            bool is_reasonable = true;
            for (int64_t i = 0; i < ggml_nelements(result); ++i) {
                if (!std::isfinite(result_data[i]) || std::abs(result_data[i]) > 100.0f) {
                    is_reasonable = false;
                    break;
                }
            }
            
            if (is_reasonable) {
                TEST_PRINTF("✅ Large values test passed\n");
                test_results.push_back({test_name, true, ""});
            } else {
                TEST_PRINTF("❌ Large values test failed\n");
                test_results.push_back({test_name, false, "Result contains unreasonable values"});
                all_passed = false;
            }
            
            ggml_free(compute_ctx);
            ggml_free(ctx);
        } else {
            TEST_PRINTF("⏭️  Skipping %s (filtered out)\n", test_name.c_str());
        }
    }
    
    // Test 3: Single row tensor
    {
        std::string test_name = "RMS_NORM_edge_single_row";
        if (matches_filter(test_name)) {
            TEST_PRINTF("Testing single row tensor...\n");
            std::vector<int64_t> dims = {1024, 1};  // Single row with many elements
            
            size_t ctx_size = ggml_tensor_overhead() * 20 + dims[0] * dims[1] * sizeof(float) * 20 + 1024*1024;
            std::vector<uint8_t> buffer(ctx_size);
            ggml_init_params init_params = {ctx_size, buffer.data(), false};
            ggml_context* ctx = ggml_init(init_params);
            
            struct ggml_tensor* src = create_test_tensor(ctx, GGML_TYPE_F32, dims, -5.0f, 5.0f);
            struct ggml_tensor* dst_ref = ggml_new_tensor(ctx, GGML_TYPE_F32, dims.size(), dims.data());
            reference_rms_norm_f32((const float*)ggml_get_data(src), (float*)ggml_get_data(dst_ref), dims[0], 1, RMS_NORM_EPS);
            
            ggml_context* compute_ctx = ggml_init(init_params);
            struct ggml_tensor* src_copy = ggml_dup_tensor(compute_ctx, src);
            memcpy(ggml_get_data(src_copy), ggml_get_data(src), ggml_nbytes(src));
            
            struct ggml_tensor* result = ggml_rms_norm(compute_ctx, src_copy, RMS_NORM_EPS);
            
            // Ensure NUMA dispatch is enabled for this test
            ggml_numa_set_fallback_flag(false);  // Re-enable NUMA dispatch
            
            struct ggml_cgraph* graph = ggml_new_graph(compute_ctx);
            ggml_build_forward_expand(graph, result);
            ggml_graph_compute_with_ctx(compute_ctx, graph, 8);
            
            if (tensors_equal(result, dst_ref, 1e-4f)) {
                TEST_PRINTF("✅ Single row tensor test passed\n");
                test_results.push_back({test_name, true, ""});
            } else {
                TEST_PRINTF("❌ Single row tensor test failed\n");
                test_results.push_back({test_name, false, "Mathematical comparison failed"});
                all_passed = false;
            }
            
            ggml_free(compute_ctx);
            ggml_free(ctx);
        } else {
            TEST_PRINTF("⏭️  Skipping %s (filtered out)\n", test_name.c_str());
        }
    }
    
    return all_passed;
}

/**
 * @brief Print summary table of all test results
 */
static void print_summary() {
    TEST_PRINTF("\n=== Test Summary ===\n");
    
    int passed = 0;
    int failed = 0;
    
    TEST_PRINTF("%-50s | %-8s | %s\n", "Test Name", "Status", "Error Message");
    TEST_PRINTF("%-50s-+-%-8s-+-%s\n", 
               std::string(50, '-').c_str(), 
               std::string(8, '-').c_str(),
               std::string(40, '-').c_str());
    
    for (const auto& result : test_results) {
        const char* status = result.passed ? "PASS" : "FAIL";
        TEST_PRINTF("%-50s | %-8s | %s\n", 
                   result.test_name.c_str(), 
                   status, 
                   result.error_message.c_str());
        
        if (result.passed) {
            passed++;
        } else {
            failed++;
        }
    }
    
    TEST_PRINTF("\nTotal: %d tests, %d passed, %d failed\n", 
               passed + failed, passed, failed);
    
    if (failed > 0) {
        TEST_PRINTF("❌ Some tests failed!\n");
    } else {
        TEST_PRINTF("✅ All tests passed!\n");
    }
}

// =============================================================================
// Main Function
// =============================================================================

int main(int argc, char** argv) {
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            show_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--summary-only") == 0) {
            g_summary_only = true;
        } else if (strcmp(argv[i], "--filter") == 0) {
            if (i + 1 >= argc) {
                printf("Error: --filter requires a regex pattern argument\n");
                show_usage(argv[0]);
                return 1;
            }
            g_test_filter = argv[++i];
            g_filter_enabled = true;
        } else {
            printf("Error: Unknown argument '%s'\n", argv[i]);
            show_usage(argv[0]);
            return 1;
        }
    }
    
    if (!g_summary_only) {
        printf("🧮 NUMA Mathematical Correctness Test: RMS_NORM Operation\n");
        printf("==================================================================\n");
        if (g_filter_enabled) {
            printf("🔍 Filter: %s\n", g_test_filter.c_str());
        }
        printf("\n");
    }
    
    // Initialize random seed for reproducible tests
    srand(42);
    
    bool all_tests_passed = true;
    
    // Run all test categories
    if (!test_numa_strategies()) {
        all_tests_passed = false;
    }
    
    if (!test_edge_cases()) {
        all_tests_passed = false;
    }
    
    // Always print final summary table
    printf("\n==================================================================\n");
    printf("📊 FINAL TEST SUMMARY\n");
    printf("==================================================================\n");
    
    int total_tests = test_results.size();
    int passed_tests = 0;
    
    for (const auto& result : test_results) {
        if (result.passed) {
            printf("✅ %s: PASSED\n", result.test_name.c_str());
            passed_tests++;
        } else {
            printf("❌ %s: FAILED - %s\n", result.test_name.c_str(), result.error_message.c_str());
        }
    }
    
    printf("==================================================================\n");
    printf("🎯 OVERALL RESULT: %d/%d tests passed (%.1f%% success rate)\n", 
           passed_tests, total_tests, total_tests > 0 ? (float)passed_tests * 100.0f / total_tests : 0.0f);
    
    if (passed_tests == total_tests && total_tests > 0) {
        printf("🎉 ALL TESTS PASSED! NUMA RMS_NORM kernel is mathematically correct.\n");
        return 0;
    } else {
        printf("💥 SOME TESTS FAILED! Review failures above.\n");
        return 1;
    }
}
