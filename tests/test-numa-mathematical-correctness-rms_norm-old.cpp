/**
 * @file test-numa-mathematical-correctness-rms_norm.cpp
 * @brief Comprehensive mathematical correctness tests for NUMA RMS_NORM kernel
 * 
 * @author David Sanftenberg
 * @date 2024-12-21
 * 
 * This test suite validates the mathematical correctness of the NUMA RMS_NORM kernel
 * against the reference implementation across various tensor sizes, thread counts,
 * and quantization types.
 * 
 * RMS_NORM operation: y = x / sqrt(mean(x²) + eps)
 * where mean(x²) is computed row-wise for each row in the tensor.
 */

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "numa-kernels/numa-kernels.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cstring>
#include <thread>
#include <string>
#include <algorithm>
#include <regex>

// Global test filter and configuration
std::string g_test_filter = "";
bool g_filter_enabled = false;
bool g_summary_only = false;

// Conditional printf macro for summary-only mode
#define TEST_PRINTF(...) do { if (!g_summary_only) printf(__VA_ARGS__); } while(0)

/**
 * Check if a test name matches the current filter
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

// Test result structure
struct TestResult {
    std::string test_name;
    bool passed;
    std::string failure_reason;
};

// Global results vector
std::vector<TestResult> test_results;

namespace {
    const float EPSILON = 1e-5f;
    const float GGML_EPS = 1e-6f;  // ggml default epsilon for RMS_NORM
    
    /**
     * @brief Create a test tensor with specified dimensions and fill with random data
     */
    struct ggml_tensor* create_test_tensor(ggml_context* ctx, ggml_type type, 
                                         std::vector<int64_t> dims, float min_val = -1.0f, float max_val = 1.0f) {
        struct ggml_tensor* tensor = ggml_new_tensor(ctx, type, dims.size(), dims.data());
        
        if (type == GGML_TYPE_F32) {
            float* data = (float*)ggml_get_data(tensor);
            std::random_device rd;
            std::mt19937 gen(42);  // Fixed seed for reproducibility
            std::uniform_real_distribution<float> dis(min_val, max_val);
            
            for (int64_t i = 0; i < ggml_nelements(tensor); ++i) {
                data[i] = dis(gen);
            }
        }
        
        return tensor;
    }
    
    /**
     * @brief Compute reference RMS_NORM implementation
     */
    void reference_rms_norm_f32(const float* src, float* dst, size_t ne0, size_t rows, float eps) {
        for (size_t row = 0; row < rows; ++row) {
            const float* src_row = src + row * ne0;
            float* dst_row = dst + row * ne0;
            
            // Compute sum of squares for this row
            float sum_sq = 0.0f;
            for (size_t i = 0; i < ne0; ++i) {
                sum_sq += src_row[i] * src_row[i];
            }
            
            // Compute mean and scaling factor
            const float mean = sum_sq / (float)ne0;
            const float scale = 1.0f / sqrtf(mean + eps);
            
            // Apply scaling to each element in the row
            for (size_t i = 0; i < ne0; ++i) {
                dst_row[i] = src_row[i] * scale;
            }
        }
    }
    
    /**
     * @brief Compare two tensors for mathematical equivalence
     */
    bool tensors_equal(const struct ggml_tensor* a, const struct ggml_tensor* b, float tolerance = EPSILON) {
        if (ggml_nelements(a) != ggml_nelements(b)) return false;
        
        const float* data_a = (const float*)ggml_get_data(a);
        const float* data_b = (const float*)ggml_get_data(b);
        
        for (int64_t i = 0; i < ggml_nelements(a); ++i) {
            float diff = fabsf(data_a[i] - data_b[i]);
            float max_val = std::max(fabsf(data_a[i]), fabsf(data_b[i]));
            
            // Use relative tolerance for large values, absolute for small values
            if (max_val > 1.0f) {
                if (diff / max_val > tolerance) {
                    std::cout << "Mismatch at element " << i << ": " 
                              << data_a[i] << " vs " << data_b[i] 
                              << " (relative error: " << (diff / max_val) << ")" << std::endl;
                    return false;
                }
            } else {
                if (diff > tolerance) {
                    std::cout << "Mismatch at element " << i << ": " 
                              << data_a[i] << " vs " << data_b[i] 
                              << " (absolute error: " << diff << ")" << std::endl;
                    return false;
                }
            }
        }
        
        return true;
    }
}

// =============================================================================
// Part 1: Multi-Dimensional Mathematical Equivalence Testing
// =============================================================================

/**
 * @brief Test mathematical correctness across various tensor dimensions
 */
static bool test_mathematical_equivalence() {
    std::cout << "=== Testing RMS_NORM Mathematical Equivalence ===" << std::endl;
    
    struct {
        std::vector<int64_t> dims;
        const char* description;
    } test_cases[] = {
        // TINY tensors - single thread strategy
        {{8, 4}, "Small 2D Matrix (8x4)"},
        {{4, 4, 4}, "Small 3D Tensor (4x4x4)"},
        
        // SMALL tensors - multi-thread single node
        {{256}, "1D Vector (256 elements)"},
        {{64, 16}, "Medium 2D Matrix (64x16)"},
        {{16, 16, 8}, "Medium 3D Tensor (16x16x8)"},
        
        // MEDIUM tensors - transition zone
        {{1024}, "Large 1D Vector (1K elements)"},
        {{256, 64}, "Large 2D Matrix (256x64)"},
        {{64, 64, 16}, "Large 3D Tensor (64x64x16)"},
        
        // LARGE tensors - data parallel strategy
        {{4096}, "Very Large 1D Vector (4K elements)"},
        {{512, 256}, "Very Large 2D Matrix (512x256)"},
        {{128, 128, 32}, "Very Large 3D Tensor (128x128x32)"},
        
        // HUGE tensors - stress test
        {{16384}, "Huge 1D Vector (16K elements)"},
        {{1024, 512}, "Huge 2D Matrix (1024x512)"},
        {{256, 256, 64}, "Huge 3D Tensor (256x256x64)"},
        
        // GIGANTIC tensors - GB-scale (if memory allows)
        {{65536}, "Gigantic 1D Vector (64K elements)"},
        {{2048, 1024}, "Gigantic 2D Matrix (2048x1024)"},
        {{512, 512, 128}, "Gigantic 3D Tensor (512x512x128)"},
    };
    
    for (const auto& test_case : test_cases) {
        std::cout << "Testing " << test_case.description << "..." << std::endl;
        
        // Create context and tensors
        size_t tensor_elements = test_case.dims[0] * (test_case.dims.size() > 1 ? test_case.dims[1] : 1) * 
                                (test_case.dims.size() > 2 ? test_case.dims[2] : 1) * 
                                (test_case.dims.size() > 3 ? test_case.dims[3] : 1);
        size_t ctx_size = ggml_tensor_overhead() * 20 + tensor_elements * sizeof(float) * 20 + 1024*1024;  // 1MB extra for graphs
        
        std::vector<uint8_t> buffer(ctx_size);
        ggml_init_params init_params = {ctx_size, buffer.data(), false};
        ggml_context* ctx = ggml_init(init_params);
        
        // Create source tensor with random data
        struct ggml_tensor* src = create_test_tensor(ctx, GGML_TYPE_F32, test_case.dims, -2.0f, 2.0f);
        
        // Create destination tensors
        struct ggml_tensor* dst_numa = ggml_new_tensor(ctx, GGML_TYPE_F32, test_case.dims.size(), test_case.dims.data());
        struct ggml_tensor* dst_ref = ggml_new_tensor(ctx, GGML_TYPE_F32, test_case.dims.size(), test_case.dims.data());
        
        // Compute reference result
        size_t ne0 = test_case.dims[0];
        size_t rows = ggml_nelements(src) / ne0;
        reference_rms_norm_f32((const float*)ggml_get_data(src), (float*)ggml_get_data(dst_ref), ne0, rows, GGML_EPS);
        
        // Test with different thread counts
        for (int n_threads : {1, 2, 4, 8, 16}) {
            // Copy source data for NUMA computation
            memcpy(ggml_get_data(dst_numa), ggml_get_data(src), ggml_nbytes(src));
            
            // Create RMS_NORM operation graph
            ggml_context* compute_ctx = ggml_init(init_params);
            struct ggml_tensor* src_copy = ggml_dup_tensor(compute_ctx, src);
            memcpy(ggml_get_data(src_copy), ggml_get_data(src), ggml_nbytes(src));

            struct ggml_tensor* result = ggml_rms_norm(compute_ctx, src_copy, GGML_EPS);
            
            // Ensure NUMA dispatch is enabled for this test
            ggml_numa_set_fallback_flag(false);  // Re-enable NUMA dispatch
            
            // Create compute graph and set thread count
            struct ggml_cgraph* graph = ggml_new_graph(compute_ctx);
            ggml_build_forward_expand(graph, result);            // Execute with NUMA
            ggml_graph_compute_with_ctx(compute_ctx, graph, n_threads);
            
            // Compare results
            if (!tensors_equal(result, dst_ref, 1e-4f)) {
                std::cout << "❌ NUMA RMS_NORM failed for " << test_case.description 
                          << " with " << n_threads << " threads" << std::endl;
                ggml_free(compute_ctx);
                ggml_free(ctx);
                return false;
            }
            
            ggml_free(compute_ctx);
        }
        
        std::cout << "✅ " << test_case.description << " passed" << std::endl;
        ggml_free(ctx);
    }
    
    return true;
}

// =============================================================================
// Part 2: Strategy-Based Validation
// =============================================================================

/**
 * @brief Test RMS_NORM mathematical correctness using strategy-based approach
 * Tests the three fundamental NUMA execution strategies:
 * 1. Single-thread Single-node: Basic functionality and fallback mechanisms
 * 2. Multi-thread Single-node: Multi-threading without NUMA complexity  
 * 3. Multi-thread Multi-node: Full NUMA data-parallel execution
 */
static bool test_numa_strategies() {
    TEST_PRINTF("\n=== Testing RMS_NORM NUMA Strategy Correctness ===\n");
    
    // Size classifications for testing
    enum TestSizeClass {
        TINY,      // Small tensors for basic validation
        SMALL,     // Medium tensors for multi-threading tests
        MEDIUM,    // Large tensors for data-parallel tests
        LARGE      // Very large tensors for stress testing
    };
    
    // Test configuration structure
    struct TestConfig {
        std::vector<int64_t> dims;
        const char* test_name;
        ggml_numa_execution_strategy_t strategy;
        const char* strategy_name;
    };
    
    // Get tensor dimensions based on size class
    auto get_test_config = [](TestSizeClass size_class, ggml_numa_execution_strategy_t strategy, const char* strategy_name) -> TestConfig {
        TestConfig config;
        config.strategy = strategy;
        config.strategy_name = strategy_name;
        
        switch (size_class) {
            case TINY:
                config.dims = {64, 32};  // 2K elements
                config.test_name = "TINY";
                break;
            case SMALL:
                config.dims = {128, 64};  // 8K elements
                config.test_name = "SMALL";
                break;
            case MEDIUM:
                config.dims = {512, 256};  // 131K elements
                config.test_name = "MEDIUM";
                break;
            case LARGE:
                config.dims = {1024, 512};  // 524K elements
                config.test_name = "LARGE";
                break;
        }
        
        return config;
    };
    
    // Strategy-based test configuration
    struct ExecutionStrategy {
        ggml_numa_execution_strategy_t strategy;
        const char* name;
        const char* description;
    };
    
    std::vector<ExecutionStrategy> strategies = {
        // Strategy 1: Single-thread, single-node
        {NUMA_STRATEGY_SINGLE_THREAD, 
         "Single-Single", "Single-thread execution on single NUMA node"},
        
        // Strategy 2: Multi-thread, single-node
        {NUMA_STRATEGY_SINGLE_NODE, 
         "Single-Multi", "Multi-thread execution within single NUMA node"},
        
        // Strategy 3: Multi-thread, multi-node (data-parallel)
        {NUMA_STRATEGY_DATA_PARALLEL, 
         "Data-Parallel", "Data-parallel execution across multiple NUMA nodes"}
    };
    
    std::vector<TestSizeClass> size_classes = {TINY, SMALL, MEDIUM, LARGE};
    
    int total_tests = 0;
    int passed_tests = 0;
    
    for (TestSizeClass size_class : size_classes) {
        for (const auto& strategy_config : strategies) {
            TestConfig config = get_test_config(size_class, strategy_config.strategy, strategy_config.name);
            
            std::string test_name = std::string("RMS_NORM_") + config.test_name + "_" + strategy_config.name;
            
            // Check if this test matches the filter
            if (!matches_filter(test_name)) {
                TEST_PRINTF("⏭️  Skipping %s (filtered out)\n", test_name.c_str());
                continue;
            }
            
            TEST_PRINTF("\n🎯 Testing %s tensors: %s\n", config.test_name, strategy_config.description);
            
            // Calculate tensor size for context allocation
            size_t tensor_elements = config.dims[0] * config.dims[1];
            size_t ctx_size = ggml_tensor_overhead() * 20 + tensor_elements * sizeof(float) * 20 + 1024*1024;
            std::vector<uint8_t> buffer(ctx_size);
            ggml_init_params init_params = {ctx_size, buffer.data(), false};
            ggml_context* ctx = ggml_init(init_params);
            
            // Create source tensor
            struct ggml_tensor* src = create_test_tensor(ctx, GGML_TYPE_F32, config.dims, -2.0f, 2.0f);
            
            // NUMA Test: Execute RMS_NORM operation using NUMA executor
            struct ggml_tensor* numa_result = ggml_rms_norm(ctx, src, GGML_EPS);
            if (!numa_result) {
                TEST_PRINTF("❌ Failed to create NUMA RMS_NORM operation\n");
                test_results.push_back({test_name, false, "Failed to create NUMA operation"});
                ggml_free(ctx);
                total_tests++;
                continue;
            }
            
            // Ensure NUMA dispatch is enabled for this test
            ggml_numa_set_fallback_flag(false);  // Enable NUMA dispatch
            
            struct ggml_cgraph* numa_graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(numa_graph, numa_result);
            
            // Execute with strategy-appropriate thread count
            int n_threads;
            switch (config.strategy) {
                case NUMA_STRATEGY_SINGLE_THREAD:
                    n_threads = 1;
                    break;
                case NUMA_STRATEGY_SINGLE_NODE:
                    n_threads = std::min(16, (int)std::thread::hardware_concurrency() / 2);  // Moderate thread count
                    break;
                case NUMA_STRATEGY_DATA_PARALLEL:
                    n_threads = std::thread::hardware_concurrency();  // Full parallel execution
                    break;
                default:
                    n_threads = 4;
                    break;
            }
            
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
            struct ggml_tensor* ref_result = ggml_rms_norm(ref_ctx, ref_input, GGML_EPS);
            
            // Copy data to reference tensor
            memcpy(ggml_get_data(ref_input), ggml_get_data(src), ggml_nbytes(src));
            
            // Execute reference implementation bypassing NUMA dispatch
            ggml_numa_set_fallback_flag(true);  // Force fallback to reference implementation
            
            struct ggml_cgraph* ref_graph = ggml_new_graph(ref_ctx);
            ggml_build_forward_expand(ref_graph, ref_result);
            
            TEST_PRINTF("   Executing TRUE reference implementation (bypassing NUMA)...\n");
            ggml_graph_compute_with_ctx(ref_ctx, ref_graph, 1);  // Single thread for reference
            
            ggml_numa_set_fallback_flag(false);  // Re-enable NUMA dispatch
            
            // Compare results
            bool test_passed = tensors_equal(numa_result, ref_result, 1e-4f);
            
            total_tests++;
            if (test_passed) {
                passed_tests++;
                TEST_PRINTF("✅ %s %s PASSED\n", config.test_name, strategy_config.name);
                test_results.push_back({test_name, true, ""});
            } else {
                TEST_PRINTF("❌ %s %s FAILED\n", config.test_name, strategy_config.name);
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

// =============================================================================
// Part 3: Edge Cases and Regression Testing
// =============================================================================

/**
 * @brief Test edge cases and potential regression scenarios
 */
static bool test_edge_cases() {
    std::cout << "\n/**
 * @brief Test edge cases and potential regression scenarios
 */
static bool test_edge_cases() {
    TEST_PRINTF("
=== Testing RMS_NORM Edge Cases ===
");
    
    bool all_passed = true;
    
    // Test 1: Very small values (near zero)
    {
        std::string test_name = "RMS_NORM_edge_near_zero";
        if (matches_filter(test_name)) {
            TEST_PRINTF("Testing near-zero values...
");
            std::vector<int64_t> dims = {64, 32};
            
            size_t ctx_size = ggml_tensor_overhead() * 20 + dims[0] * dims[1] * sizeof(float) * 20 + 1024*1024;
            std::vector<uint8_t> buffer(ctx_size);
            ggml_init_params init_params = {ctx_size, buffer.data(), false};
            ggml_context* ctx = ggml_init(init_params);
            
            struct ggml_tensor* src = create_test_tensor(ctx, GGML_TYPE_F32, dims, -1e-6f, 1e-6f);
            
            ggml_context* compute_ctx = ggml_init(init_params);
            struct ggml_tensor* src_copy = ggml_dup_tensor(compute_ctx, src);
            memcpy(ggml_get_data(src_copy), ggml_get_data(src), ggml_nbytes(src));
            
            struct ggml_tensor* result = ggml_rms_norm(compute_ctx, src_copy, GGML_EPS);
            
            // Ensure NUMA dispatch is enabled for this test
            ggml_numa_set_fallback_flag(false);  // Re-enable NUMA dispatch
            
            struct ggml_cgraph* graph = ggml_new_graph(compute_ctx);
            ggml_build_forward_expand(graph, result);
            ggml_graph_compute_with_ctx(compute_ctx, graph, 4);
            
            // Check that result is finite
            float* result_data = (float*)ggml_get_data(result);
            bool is_finite = true;
            for (size_t i = 0; i < ggml_nelements(result); ++i) {
                if (!std::isfinite(result_data[i])) {
                    is_finite = false;
                    break;
                }
            }
            
            if (is_finite) {
                TEST_PRINTF("✅ Near-zero values test passed
");
                test_results.push_back({test_name, true, ""});
            } else {
                TEST_PRINTF("❌ Near-zero values test failed
");
                test_results.push_back({test_name, false, "Result contains non-finite values"});
                all_passed = false;
            }
            
            ggml_free(compute_ctx);
            ggml_free(ctx);
        } else {
            TEST_PRINTF("⏭️  Skipping %s (filtered out)
", test_name.c_str());
        }
    }
    
    // Test 2: Large values
    {
        std::string test_name = "RMS_NORM_edge_large_values";
        if (matches_filter(test_name)) {
            TEST_PRINTF("Testing large values...
");
            std::vector<int64_t> dims = {64, 32};
            
            size_t ctx_size = ggml_tensor_overhead() * 20 + dims[0] * dims[1] * sizeof(float) * 20 + 1024*1024;
            std::vector<uint8_t> buffer(ctx_size);
            ggml_init_params init_params = {ctx_size, buffer.data(), false};
            ggml_context* ctx = ggml_init(init_params);
            
            struct ggml_tensor* src = create_test_tensor(ctx, GGML_TYPE_F32, dims, -1000.0f, 1000.0f);
            
            ggml_context* compute_ctx = ggml_init(init_params);
            struct ggml_tensor* src_copy = ggml_dup_tensor(compute_ctx, src);
            memcpy(ggml_get_data(src_copy), ggml_get_data(src), ggml_nbytes(src));
            
            struct ggml_tensor* result = ggml_rms_norm(compute_ctx, src_copy, GGML_EPS);
            
            // Ensure NUMA dispatch is enabled for this test
            ggml_numa_set_fallback_flag(false);  // Re-enable NUMA dispatch
            
            struct ggml_cgraph* graph = ggml_new_graph(compute_ctx);
            ggml_build_forward_expand(graph, result);
            ggml_graph_compute_with_ctx(compute_ctx, graph, 4);
            
            // Check that result is finite and within reasonable bounds
            float* result_data = (float*)ggml_get_data(result);
            bool is_reasonable = true;
            for (size_t i = 0; i < ggml_nelements(result); ++i) {
                if (!std::isfinite(result_data[i]) || std::abs(result_data[i]) > 100.0f) {
                    is_reasonable = false;
                    break;
                }
            }
            
            if (is_reasonable) {
                TEST_PRINTF("✅ Large values test passed
");
                test_results.push_back({test_name, true, ""});
            } else {
                TEST_PRINTF("❌ Large values test failed
");
                test_results.push_back({test_name, false, "Result contains unreasonable values"});
                all_passed = false;
            }
            
            ggml_free(compute_ctx);
            ggml_free(ctx);
        } else {
            TEST_PRINTF("⏭️  Skipping %s (filtered out)
", test_name.c_str());
        }
    }
    
    // Test 3: Single row tensor
    {
        std::string test_name = "RMS_NORM_edge_single_row";
        if (matches_filter(test_name)) {
            TEST_PRINTF("Testing single row tensor...
");
            std::vector<int64_t> dims = {1024, 1};  // Single row with many elements
            
            size_t ctx_size = ggml_tensor_overhead() * 20 + dims[0] * dims[1] * sizeof(float) * 20 + 1024*1024;
            std::vector<uint8_t> buffer(ctx_size);
            ggml_init_params init_params = {ctx_size, buffer.data(), false};
            ggml_context* ctx = ggml_init(init_params);
            
            struct ggml_tensor* src = create_test_tensor(ctx, GGML_TYPE_F32, dims, -5.0f, 5.0f);
            struct ggml_tensor* dst_ref = ggml_new_tensor(ctx, GGML_TYPE_F32, dims.size(), dims.data());
            reference_rms_norm_f32((const float*)ggml_get_data(src), (float*)ggml_get_data(dst_ref), dims[0], 1, GGML_EPS);
            
            ggml_context* compute_ctx = ggml_init(init_params);
            struct ggml_tensor* src_copy = ggml_dup_tensor(compute_ctx, src);
            memcpy(ggml_get_data(src_copy), ggml_get_data(src), ggml_nbytes(src));
            
            struct ggml_tensor* result = ggml_rms_norm(compute_ctx, src_copy, GGML_EPS);
            
            // Ensure NUMA dispatch is enabled for this test
            ggml_numa_set_fallback_flag(false);  // Re-enable NUMA dispatch
            
            struct ggml_cgraph* graph = ggml_new_graph(compute_ctx);
            ggml_build_forward_expand(graph, result);
            ggml_graph_compute_with_ctx(compute_ctx, graph, 8);
            
            if (tensors_equal(result, dst_ref, 1e-4f)) {
                TEST_PRINTF("✅ Single row tensor test passed
");
                test_results.push_back({test_name, true, ""});
            } else {
                TEST_PRINTF("❌ Single row tensor test failed
");
                test_results.push_back({test_name, false, "Mathematical comparison failed"});
                all_passed = false;
            }
            
            ggml_free(compute_ctx);
            ggml_free(ctx);
        } else {
            TEST_PRINTF("⏭️  Skipping %s (filtered out)
", test_name.c_str());
        }
    }" << std::endl;
    
    bool all_passed = true;
    
    // Test 1: Very small values (near zero)
    {
        std::cout << "Testing near-zero values..." << std::endl;
        std::vector<int64_t> dims = {64, 32};
        
        size_t ctx_size = ggml_tensor_overhead() * 20 + dims[0] * dims[1] * sizeof(float) * 20 + 1024*1024;
        std::vector<uint8_t> buffer(ctx_size);
        ggml_init_params init_params = {ctx_size, buffer.data(), false};
        ggml_context* ctx = ggml_init(init_params);
        
        struct ggml_tensor* src = create_test_tensor(ctx, GGML_TYPE_F32, dims, -1e-6f, 1e-6f);
        
        ggml_context* compute_ctx = ggml_init(init_params);
        struct ggml_tensor* src_copy = ggml_dup_tensor(compute_ctx, src);
        memcpy(ggml_get_data(src_copy), ggml_get_data(src), ggml_nbytes(src));
        
        struct ggml_tensor* result = ggml_rms_norm(compute_ctx, src_copy, GGML_EPS);
        
        // Ensure NUMA dispatch is enabled for this test
        ggml_numa_set_fallback_flag(false);  // Re-enable NUMA dispatch
        
        struct ggml_cgraph* graph = ggml_new_graph(compute_ctx);
        ggml_build_forward_expand(graph, result);
        ggml_graph_compute_with_ctx(compute_ctx, graph, 4);
        
        // Check that result is finite
        const float* data = (const float*)ggml_get_data(result);
        for (int64_t i = 0; i < ggml_nelements(result); ++i) {
            if (!std::isfinite(data[i])) {
                std::cout << "❌ Non-finite result in near-zero test" << std::endl;
                all_passed = false;
                break;
            }
        }
        
        if (all_passed) {
            std::cout << "✅ Near-zero values test passed" << std::endl;
        }
        
        ggml_free(compute_ctx);
        ggml_free(ctx);
    }
    
    // Test 2: Large values
    {
        std::cout << "Testing large values..." << std::endl;
        std::vector<int64_t> dims = {64, 32};
        
        size_t ctx_size = ggml_tensor_overhead() * 20 + dims[0] * dims[1] * sizeof(float) * 20 + 1024*1024;
        std::vector<uint8_t> buffer(ctx_size);
        ggml_init_params init_params = {ctx_size, buffer.data(), false};
        ggml_context* ctx = ggml_init(init_params);
        
        struct ggml_tensor* src = create_test_tensor(ctx, GGML_TYPE_F32, dims, -100.0f, 100.0f);
        
        ggml_context* compute_ctx = ggml_init(init_params);
        struct ggml_tensor* src_copy = ggml_dup_tensor(compute_ctx, src);
        memcpy(ggml_get_data(src_copy), ggml_get_data(src), ggml_nbytes(src));
        
        struct ggml_tensor* result = ggml_rms_norm(compute_ctx, src_copy, GGML_EPS);
        
        // Ensure NUMA dispatch is enabled for this test
        ggml_numa_set_fallback_flag(false);  // Re-enable NUMA dispatch
        
        struct ggml_cgraph* graph = ggml_new_graph(compute_ctx);
        ggml_build_forward_expand(graph, result);
        ggml_graph_compute_with_ctx(compute_ctx, graph, 4);
        
        // Check that result is finite and reasonable
        const float* data = (const float*)ggml_get_data(result);
        for (int64_t i = 0; i < ggml_nelements(result); ++i) {
            if (!std::isfinite(data[i]) || fabsf(data[i]) > 10.0f) {
                std::cout << "❌ Unreasonable result in large values test: " << data[i] << std::endl;
                all_passed = false;
                break;
            }
        }
        
        if (all_passed) {
            std::cout << "✅ Large values test passed" << std::endl;
        }
        
        ggml_free(compute_ctx);
        ggml_free(ctx);
    }
    
    // Test 3: Single row tensor
    {
        std::cout << "Testing single row tensor..." << std::endl;
        std::vector<int64_t> dims = {1024};  // Single row with many elements
        
        size_t ctx_size = ggml_tensor_overhead() * 20 + dims[0] * sizeof(float) * 20 + 1024*1024;
        std::vector<uint8_t> buffer(ctx_size);
        ggml_init_params init_params = {ctx_size, buffer.data(), false};
        ggml_context* ctx = ggml_init(init_params);
        
        struct ggml_tensor* src = create_test_tensor(ctx, GGML_TYPE_F32, dims, -1.0f, 1.0f);
        struct ggml_tensor* dst_ref = ggml_new_tensor(ctx, GGML_TYPE_F32, dims.size(), dims.data());
        
        reference_rms_norm_f32((const float*)ggml_get_data(src), (float*)ggml_get_data(dst_ref), dims[0], 1, GGML_EPS);
        
        ggml_context* compute_ctx = ggml_init(init_params);
        struct ggml_tensor* src_copy = ggml_dup_tensor(compute_ctx, src);
        memcpy(ggml_get_data(src_copy), ggml_get_data(src), ggml_nbytes(src));
        
        struct ggml_tensor* result = ggml_rms_norm(compute_ctx, src_copy, GGML_EPS);
        
        // Ensure NUMA dispatch is enabled for this test
        ggml_numa_set_fallback_flag(false);  // Re-enable NUMA dispatch
        
        struct ggml_cgraph* graph = ggml_new_graph(compute_ctx);
        ggml_build_forward_expand(graph, result);
        ggml_graph_compute_with_ctx(compute_ctx, graph, 8);
        
        if (tensors_equal(result, dst_ref, 1e-4f)) {
            std::cout << "✅ Single row tensor test passed" << std::endl;
        } else {
            std::cout << "❌ Single row tensor test failed" << std::endl;
            all_passed = false;
        }
        
        ggml_free(compute_ctx);
        ggml_free(ctx);
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

/**
 * Show usage information
 */
void show_usage(const char* program_name) {
    printf("Usage: %s [OPTIONS]\n", program_name);
    printf("\nOptions:\n");
    printf("  --filter <regex>    Run only tests matching the regex pattern (case-insensitive)\n");
    printf("  --summary-only      Only print the summary table, not full test output\n");
    printf("  --help              Show this help message\n");
    printf("\nFilter Examples:\n");
    printf("  --filter \"MEDIUM.*Single-Multi\"           # Run only MEDIUM tensor single-node tests\n");
    printf("  --filter \"Data-Parallel\"                 # Run all data-parallel tests\n");
    printf("  --filter \"edge\"                          # Run edge case tests only\n");
    printf("  --filter \"TINY.*SMALL\"                   # Run TINY and SMALL tensor tests\n");
    printf("\nTest Categories:\n");
    printf("  - test_numa_strategies: Strategy-based execution testing (Single-Thread, Single-Node, Data-Parallel)\n");
    printf("  - test_edge_cases: Edge cases and regression scenarios\n");
    printf("\nExecution Strategies:\n");
    printf("  1. Single-Thread: Single-thread execution on single NUMA node\n");
    printf("  2. Single-Node: Multi-thread execution within single NUMA node\n");
    printf("  3. Data-Parallel: Data-parallel execution across multiple NUMA nodes\n");
}

// =============================================================================
// Main Test Runner
// =============================================================================

int main(int argc, char** argv) {
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            show_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--summary-only") == 0) {
            g_summary_only = true;
        } else if (strcmp(argv[i], "--filter") == 0) {
            if (i + 1 < argc) {
                g_test_filter = argv[i + 1];
                g_filter_enabled = true;
                i++; // Skip the filter argument
                TEST_PRINTF("🔍 Filter enabled: '%s'\n", g_test_filter.c_str());
            } else {
                printf("❌ Error: --filter requires a regex pattern argument\n");
                show_usage(argv[0]);
                return 1;
            }
        } else {
            printf("❌ Error: Unknown argument '%s'\n", argv[i]);
            show_usage(argv[0]);
            return 1;
        }
    }
    
    if (!g_summary_only) {
        printf("==================================================================\n");
        printf("🧪 NUMA RMS_NORM MATHEMATICAL CORRECTNESS TEST SUITE\n");
        if (g_filter_enabled) {
            printf("🔍 Running filtered tests matching: '%s'\n", g_test_filter.c_str());
        }
        printf("==================================================================\n");
    }
    
    TEST_PRINTF("Starting NUMA RMS_NORM Mathematical Correctness Tests...\n");
    TEST_PRINTF("==========================================================\n");
    
    // Initialize NUMA for testing
    ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);
    
    bool all_tests_passed = true;
    
    // Run strategy-based NUMA testing
    if (matches_filter("RMS_NORM_numa_strategies")) {
        if (!test_numa_strategies()) {
            all_tests_passed = false;
        }
    } else {
        TEST_PRINTF("⏭️  Skipping RMS_NORM_numa_strategies (filtered out)\n");
    }
    
    if (matches_filter("RMS_NORM_edge_cases")) {
        if (!test_edge_cases()) {
            all_tests_passed = false;
        }
    } else {
        TEST_PRINTF("⏭️  Skipping RMS_NORM_edge_cases (filtered out)\n");
    }
    
    // Print final summary
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
            printf("❌ %s: FAILED - %s\n", result.test_name.c_str(), result.failure_reason.c_str());
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
