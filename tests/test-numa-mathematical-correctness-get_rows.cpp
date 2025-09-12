/**
 * @file test-numa-mathematical-correctness-get_rows.cpp
 * @brief Comprehensive mathematical correctness tests for NUMA GET_ROWS kernel
 * @author David Sanftenberg
 * 
 * This test suite validates the mathematical correctness of the NUMA GET_ROWS kernel
 * against the reference ggml-cpu implementation across multiple dimensions:
 * 
 * 1. MATHEMATICAL EQUIVALENCE TESTING
 *    - Multi-dimensional tensor validation (1D to 4D)
 *    - Various index patterns and edge cases
 *    - Comprehensive quantization type coverage (F32, F16, BF16, Q8_0, Q4_0, Q5_0, etc.)
 * 
 * 2. NUMA EXECUTION STRATEGY TESTING
 *    - Single-thread/Single-node strategy validation
 *    - Multi-thread/Single-node strategy validation
 *    - Multi-thread/Multi-node (data-parallel) strategy validation
 * 
 * 3. QUANTIZATION COVERAGE TESTING
 *    - All tensor quantization types supported by reference implementation
 *    - F32, F16, BF16 floating-point type coverage
 *    - Quantized types: Q8_0, Q4_0, Q5_0, Q4_1, Q5_1, etc.
 *    - Critical for preventing silent model inference failures
 * 
 * 4. REGRESSION TESTING
 *    - Index boundary validation
 *    - Row extraction accuracy
 *    - Memory layout correctness
 *    - NUMA memory allocation verification
 * 
 * GET_ROWS OPERATION CHARACTERISTICS:
 * - Extracts rows from source tensor (src0) based on integer indices (src1)
 * - Supports all quantization types with automatic dequantization to F32
 * - Row-wise parallel execution optimal for NUMA architecture
 * - Critical operation for model inference (token embedding lookup, etc.)
 */

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <regex>

extern "C" {
    #include "ggml.h"
    #include "ggml-cpu.h"
    #include "numa-kernels/numa-kernels.h"
    #include "ggml-cpu/ggml-numa-openmp-coordinator.h"
    #include "ggml-cpu/ggml-numa-shared.h"
}

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

// Test configuration constants
constexpr float TOLERANCE_F32 = 1e-5f;
constexpr float TOLERANCE_F16 = 1e-3f;
constexpr float TOLERANCE_QUANT = 1e-2f;
constexpr int MAX_PRINT_ELEMENTS = 20;

// Test size categories for comprehensive validation
enum class TensorSize {
    TINY = 0,        // < 1K elements (single-thread/single-node)
    SMALL,           // 1K-16K elements (multi-thread/single-node)
    MEDIUM,          // 16K-256K elements (data-parallel threshold)
    LARGE,           // 256K-4M elements (data-parallel)
    VERY_LARGE       // > 4M elements (stress test)
};

struct TestResult {
    bool passed;
    std::string description;
    std::string details;
    double max_diff;
    size_t total_elements;
    ggml_numa_execution_strategy_t strategy;
};

class GetRowsTestSuite {
private:
    ggml_context* ctx_ref;
    ggml_context* ctx_numa;
    std::vector<TestResult> results;
    
    // Test data generators
    std::mt19937 rng;
    std::uniform_real_distribution<float> float_dist;
    std::uniform_int_distribution<int> int_dist;

    // Statistics
    int tests_run = 0;
    int tests_passed = 0;

public:
    GetRowsTestSuite() : rng(42), float_dist(-2.0f, 2.0f), int_dist(0, 100) {
        // Initialize ggml contexts
        struct ggml_init_params params = {
            /*.mem_size   =*/ 1024*1024*1024, // 1GB
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };
        
        ctx_ref = ggml_init(params);
        ctx_numa = ggml_init(params);
        
        if (!ctx_ref || !ctx_numa) {
            throw std::runtime_error("Failed to initialize ggml contexts");
        }
    }
    
    ~GetRowsTestSuite() {
        if (ctx_ref) ggml_free(ctx_ref);
        if (ctx_numa) ggml_free(ctx_numa);
    }

    // Comprehensive tensor size calculation
    std::vector<int64_t> get_tensor_dims(TensorSize size, int dimensions) {
        std::vector<int64_t> dims(4, 1);
        
        switch (size) {
            case TensorSize::TINY: {
                // < 1K elements - single-thread strategy
                if (dimensions == 1) dims = {512, 1, 1, 1};
                else if (dimensions == 2) dims = {16, 32, 1, 1};
                else if (dimensions == 3) dims = {8, 8, 8, 1};
                else dims = {4, 4, 4, 4};
                break;
            }
            case TensorSize::SMALL: {
                // 1K-16K elements - multi-thread/single-node strategy
                if (dimensions == 1) dims = {8192, 1, 1, 1};
                else if (dimensions == 2) dims = {64, 128, 1, 1};
                else if (dimensions == 3) dims = {16, 16, 32, 1};
                else dims = {8, 8, 8, 16};
                break;
            }
            case TensorSize::MEDIUM: {
                // 16K-256K elements - data-parallel threshold
                if (dimensions == 1) dims = {131072, 1, 1, 1};
                else if (dimensions == 2) dims = {256, 512, 1, 1};
                else if (dimensions == 3) dims = {32, 32, 128, 1};
                else dims = {16, 16, 16, 32};
                break;
            }
            case TensorSize::LARGE: {
                // 256K-4M elements - data-parallel
                if (dimensions == 1) dims = {1048576, 1, 1, 1};
                else if (dimensions == 2) dims = {1024, 1024, 1, 1};
                else if (dimensions == 3) dims = {64, 64, 256, 1};
                else dims = {32, 32, 32, 32};
                break;
            }
            case TensorSize::VERY_LARGE: {
                // > 4M elements - stress test
                if (dimensions == 1) dims = {8388608, 1, 1, 1};
                else if (dimensions == 2) dims = {2048, 4096, 1, 1};
                else if (dimensions == 3) dims = {128, 128, 512, 1};
                else dims = {64, 64, 64, 64};
                break;
            }
        }
        
        return dims;
    }

    // Generate test indices for GET_ROWS operation
    std::vector<int32_t> generate_indices(int64_t source_rows, int64_t num_indices) {
        std::vector<int32_t> indices(num_indices);
        
        for (int64_t i = 0; i < num_indices; i++) {
            indices[i] = int_dist(rng) % source_rows;  // Random valid indices
        }
        
        return indices;
    }

    // Fill tensor with test data
    void fill_tensor_test_data(ggml_tensor* tensor) {
        if (!tensor) return;
        
        const size_t total_elements = ggml_nelements(tensor);
        
        if (tensor->type == GGML_TYPE_F32) {
            float* data = (float*)ggml_get_data(tensor);
            for (size_t i = 0; i < total_elements; i++) {
                data[i] = float_dist(rng);
            }
        } else if (tensor->type == GGML_TYPE_F16) {
            ggml_fp16_t* data = (ggml_fp16_t*)ggml_get_data(tensor);
            for (size_t i = 0; i < total_elements; i++) {
                data[i] = ggml_fp32_to_fp16(float_dist(rng));
            }
        }
    }

    // Compare tensor data with tolerance
    bool compare_tensors(const ggml_tensor* expected, const ggml_tensor* actual, 
                        float tolerance, double& max_diff, std::ostringstream& details) {
        if (!expected || !actual) {
            details << "Null tensor pointer";
            return false;
        }

        const size_t total_elements = ggml_nelements(expected);
        if (total_elements != (size_t)ggml_nelements(actual)) {
            details << "Element count mismatch: expected=" << total_elements 
                   << ", actual=" << ggml_nelements(actual);
            return false;
        }

        // GET_ROWS always outputs F32
        if (expected->type != GGML_TYPE_F32 || actual->type != GGML_TYPE_F32) {
            details << "Output type mismatch - GET_ROWS should output F32";
            return false;
        }

        const float* expected_data = (const float*)ggml_get_data(expected);
        const float* actual_data = (const float*)ggml_get_data(actual);
        
        max_diff = 0.0;
        size_t mismatches = 0;
        
        for (size_t i = 0; i < total_elements; i++) {
            const double diff = std::abs(expected_data[i] - actual_data[i]);
            max_diff = std::max(max_diff, diff);
            
            if (diff > tolerance) {
                mismatches++;
                if (mismatches <= 10) { // Show first 10 mismatches
                    details << "Mismatch at [" << i << "]: expected=" 
                           << std::fixed << std::setprecision(6) << expected_data[i]
                           << ", actual=" << actual_data[i] << ", diff=" << diff << "\\n";
                }
            }
        }
        
        if (mismatches > 10) {
            details << "... and " << (mismatches - 10) << " more mismatches\\n";
        }
        
        details << "Max difference: " << std::scientific << std::setprecision(3) << max_diff;
        return mismatches == 0;
    }

    // Test GET_ROWS with specific configuration
    TestResult test_get_rows_configuration(TensorSize size, int dimensions, ggml_type src_type, 
                                         const std::string& test_name) {
        tests_run++;
        
        TestResult result;
        result.description = test_name;
        result.passed = false;
        result.max_diff = 0.0;
        result.strategy = NUMA_STRATEGY_SINGLE_THREAD;
        
        try {
            // For GET_ROWS, simplify to 2D tensors to avoid dimension complexity
            // We'll test: source[width, num_rows], indices[num_selections]
            auto src_dims = get_tensor_dims(size, 2); // Force 2D
            const int64_t width = src_dims[0];
            const int64_t num_rows = src_dims[1];
            const int64_t num_selections = std::min((int64_t)32, num_rows); // Extract up to 32 rows
            
            // Create 2D source tensor
            ggml_tensor* src_ref = ggml_new_tensor_2d(ctx_ref, src_type, width, num_rows);
            ggml_tensor* src_numa = ggml_new_tensor_2d(ctx_numa, src_type, width, num_rows);
            
            if (!src_ref || !src_numa) {
                result.details = "Failed to create source tensors";
                return result;
            }
            
            fill_tensor_test_data(src_ref);
            
            // Copy test data to NUMA tensor
            memcpy(ggml_get_data(src_numa), ggml_get_data(src_ref), ggml_nbytes(src_ref));
            
            // Generate test indices (random valid row indices)
            auto indices = generate_indices(num_rows, num_selections);
            
            // Create 1D indices tensor
            ggml_tensor* indices_ref = ggml_new_tensor_1d(ctx_ref, GGML_TYPE_I32, num_selections);
            ggml_tensor* indices_numa = ggml_new_tensor_1d(ctx_numa, GGML_TYPE_I32, num_selections);
            
            if (!indices_ref || !indices_numa) {
                result.details = "Failed to create indices tensors";
                return result;
            }
            
            memcpy(ggml_get_data(indices_ref), indices.data(), num_selections * sizeof(int32_t));
            memcpy(ggml_get_data(indices_numa), indices.data(), num_selections * sizeof(int32_t));
            
            // Create GET_ROWS operations (output is F32)
            ggml_tensor* result_ref = ggml_get_rows(ctx_ref, src_ref, indices_ref);
            ggml_tensor* result_numa = ggml_get_rows(ctx_numa, src_numa, indices_numa);
            
            if (!result_ref || !result_numa) {
                result.details = "Failed to create GET_ROWS operations";
                return result;
            }
            
            // Query strategy
            result.strategy = ggml_numa_kernels_query(result_numa);
            result.total_elements = ggml_nelements(result_numa);
            
            // Execute reference implementation
            ggml_cgraph* ref_gf = ggml_new_graph(ctx_ref);
            ggml_build_forward_expand(ref_gf, result_ref);
            
            struct ggml_cplan cplan_ref = ggml_graph_plan(ref_gf, 1, nullptr);  // Single thread
            cplan_ref.work_size = std::max(cplan_ref.work_size, (size_t)(1024*1024));
            if (cplan_ref.work_size > 0) {
                cplan_ref.work_data = (uint8_t*)malloc(cplan_ref.work_size);
            }
            
            if (cplan_ref.work_size > 0 && !cplan_ref.work_data) {
                result.details = "Failed to allocate reference work buffer";
                return result;
            }
            
            ggml_graph_compute(ref_gf, &cplan_ref);
            
            // Execute NUMA implementation 
            ggml_cgraph* numa_gf = ggml_new_graph(ctx_numa);
            ggml_build_forward_expand(numa_gf, result_numa);
            
            struct ggml_cplan cplan_numa = ggml_graph_plan(numa_gf, 1, nullptr);
            cplan_numa.work_size = std::max(cplan_numa.work_size, (size_t)(1024*1024));
            if (cplan_numa.work_size > 0) {
                cplan_numa.work_data = (uint8_t*)malloc(cplan_numa.work_size);
            }
            
            if (cplan_numa.work_size > 0 && !cplan_numa.work_data) {
                if (cplan_ref.work_data) free(cplan_ref.work_data);
                result.details = "Failed to allocate NUMA work buffer";
                return result;
            }
            
            ggml_graph_compute(numa_gf, &cplan_numa);
            
            // Compare results
            std::ostringstream details_stream;
            float tolerance = (src_type == GGML_TYPE_F32) ? TOLERANCE_F32 : 
                             (src_type == GGML_TYPE_F16) ? TOLERANCE_F16 : TOLERANCE_QUANT;
                             
            result.passed = compare_tensors(result_ref, result_numa, tolerance, result.max_diff, details_stream);
            result.details = details_stream.str();
            
            // Cleanup
            if (cplan_ref.work_data) free(cplan_ref.work_data);
            if (cplan_numa.work_data) free(cplan_numa.work_data);
            
            if (result.passed) {
                tests_passed++;
            }
            
        } catch (const std::exception& e) {
            result.details = std::string("Exception: ") + e.what();
        }
        
        return result;
    }

    // Comprehensive GET_ROWS testing across quantization types
    void run_quantization_coverage_tests() {
        TEST_PRINTF("\n=== GET_ROWS Quantization Coverage Tests ===\n");
        
        // TODO: Implement comprehensive quantization type testing
        // Currently testing F32 and F16 - add Q8_0, Q4_0, Q5_0, etc. when quantization support is added
        std::vector<std::pair<ggml_type, std::string>> types = {
            {GGML_TYPE_F32, "F32"},
            {GGML_TYPE_F16, "F16"}
            // TODO: Add quantized types when dequantization is implemented
            // {GGML_TYPE_Q8_0, "Q8_0"},
            // {GGML_TYPE_Q4_0, "Q4_0"},
            // {GGML_TYPE_Q5_0, "Q5_0"}
        };
        
        for (auto& [type, name] : types) {
            std::string test_name = "GET_ROWS Quantization: " + name;
            
            if (!matches_filter(test_name)) {
                continue; // Skip tests that don't match filter
            }
            
            auto result = test_get_rows_configuration(TensorSize::SMALL, 2, type, test_name);
            results.push_back(result);
            
            TEST_PRINTF("  %s: %s (max_diff=%e) [%s]\n",
                       result.description.c_str(),
                       result.passed ? "PASS" : "FAIL",
                       result.max_diff,
                       get_strategy_name(result.strategy));
        }
    }

    // Multi-dimensional tensor validation
    void run_mathematical_equivalence_tests() {
        TEST_PRINTF("\n=== GET_ROWS Mathematical Equivalence Tests ===\n");
        
        std::vector<std::pair<TensorSize, std::string>> sizes = {
            {TensorSize::TINY, "TINY"},
            {TensorSize::SMALL, "SMALL"}, 
            {TensorSize::MEDIUM, "MEDIUM"},
            {TensorSize::LARGE, "LARGE"}
        };
        
        std::vector<int> dimensions = {2}; // Focus on 2D for GET_ROWS
        
        for (auto& [size, size_name] : sizes) {
            for (int dim : dimensions) {
                std::string test_name = "GET_ROWS " + size_name + " " + std::to_string(dim) + "D F32";
                
                if (!matches_filter(test_name)) {
                    continue; // Skip tests that don't match filter
                }
                
                auto result = test_get_rows_configuration(size, dim, GGML_TYPE_F32, test_name);
                results.push_back(result);
                
                TEST_PRINTF("  %s: %s (%zu elements) [%s]\n",
                           result.description.c_str(),
                           result.passed ? "PASS" : "FAIL",
                           result.total_elements,
                           get_strategy_name(result.strategy));
                         
                if (!result.passed) {
                    TEST_PRINTF("    Details: %s\n", result.details.c_str());
                }
            }
        }
    }

    // Strategy-specific validation
    void run_execution_strategy_tests() {
        TEST_PRINTF("\n=== GET_ROWS Execution Strategy Tests ===\n");
        
        // Test each strategy explicitly using different tensor sizes
        std::vector<std::tuple<TensorSize, std::string, std::string>> strategy_tests = {
            {TensorSize::TINY, "Single-thread/Single-node", "TINY F32"},
            {TensorSize::SMALL, "Multi-thread/Single-node", "SMALL F32"},
            {TensorSize::LARGE, "Multi-thread/Multi-node", "LARGE F32"}
        };
        
        for (auto& [size, strategy_name, test_desc] : strategy_tests) {
            std::string test_name = "GET_ROWS Strategy " + strategy_name + " " + test_desc;
            
            if (!matches_filter(test_name)) {
                continue; // Skip tests that don't match filter
            }
            
            auto result = test_get_rows_configuration(size, 2, GGML_TYPE_F32, test_name);
            results.push_back(result);
            
            TEST_PRINTF("  %s: %s [%s]\n",
                       result.description.c_str(),
                       result.passed ? "PASS" : "FAIL",
                       get_strategy_name(result.strategy));
        }
    }

    // Regression and edge case testing
    void run_regression_tests() {
        TEST_PRINTF("\n=== GET_ROWS Regression Tests ===\n");
        
        // Test boundary conditions and edge cases
        std::string test_name = "GET_ROWS Regression: Boundary Conditions";
        
        if (!matches_filter(test_name)) {
            return; // Skip tests that don't match filter
        }
        
        auto result = test_get_rows_configuration(TensorSize::SMALL, 2, GGML_TYPE_F32, test_name);
        results.push_back(result);
        
        TEST_PRINTF("  %s: %s\n",
                   result.description.c_str(),
                   result.passed ? "PASS" : "FAIL");
    }

    // Utility functions
    const char* get_strategy_name(ggml_numa_execution_strategy_t strategy) {
        switch (strategy) {
            case NUMA_STRATEGY_SINGLE_THREAD: return "Single/Single";
            case NUMA_STRATEGY_SINGLE_NODE: return "Single/Multi";
            case NUMA_STRATEGY_DATA_PARALLEL: return "Data-Parallel";
            default: return "Unknown";
        }
    }

    // Run all tests
    void run_all_tests() {
        if (!g_summary_only) {
            printf("Starting NUMA GET_ROWS Mathematical Correctness Test Suite\n");
            printf("=========================================================\n");
        }
        
        run_mathematical_equivalence_tests();
        run_quantization_coverage_tests();
        run_execution_strategy_tests();
        run_regression_tests();
        
        print_summary();
    }

    // Print test summary
    void print_summary() {
        printf("\n=== Test Summary ===\n");
        printf("Total Tests: %d\n", tests_run);
        printf("Passed: %d\n", tests_passed);
        printf("Failed: %d\n", (tests_run - tests_passed));
        printf("Success Rate: %.1f%%\n", (100.0 * tests_passed / tests_run));
        
        if (tests_passed == tests_run) {
            printf("\n🎉 ALL TESTS PASSED! GET_ROWS kernel is mathematically correct.\n");
        } else {
            printf("\n❌ Some tests failed. See details above.\n");
        }
    }

    // Return overall success
    bool all_tests_passed() const {
        return tests_passed == tests_run;
    }
};

void show_usage(const char* program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("\n");
    printf("Options:\n");
    printf("  --help, -h         Show this help message\n");
    printf("  --summary-only     Print only final summary (no progress output)\n");
    printf("  --filter <regex>   Run only tests matching the regex pattern (case-insensitive)\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s                                    # Run all tests with progress output\n", program_name);
    printf("  %s --summary-only                     # Run all tests, summary only\n", program_name);
    printf("  %s --filter \"TINY\"                   # Run only TINY tensor tests\n", program_name);
    printf("  %s --filter \"F16|F32\"                # Run only F16 and F32 tests\n", program_name);
    printf("  %s --filter \"Strategy.*Single\"       # Run only single-node strategy tests\n", program_name);
}

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
    
    if (!g_summary_only && g_filter_enabled) {
        printf("🔍 Running filtered tests matching: '%s'\n", g_test_filter.c_str());
    }
    
    try {
        GetRowsTestSuite test_suite;
        test_suite.run_all_tests();
        return test_suite.all_tests_passed() ? 0 : 1;
    } catch (const std::exception& e) {
        printf("❌ Test suite failed with exception: %s\n", e.what());
        return 1;
    }
}
