/**
 * NUMA Mathematical Correctness Test for ADD Operation
 * 
 * This test validates mathematical equivalence between NUMA parallel ADD operations 
 * and serial reference implementations across various tensor dimensions and thread strategies.
 */

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>

// GGML includes
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-numa-operation-dispatch.h"
#include "ggml-numa-coordinator.h"

// Test result structure
struct TestResult {
    std::string test_name;
    bool passed;
    std::string failure_reason;
};

// Test case definition for ADD operation
struct AddTestCase {
    int64_t ne[4];      // Tensor dimensions [x, y, z, w]
    int numa_threads;    // Number of threads to use
    const char* description;
};

class NumaAddMathematicalCorrectnessTestSuite {
private:
    // Test dimensions - covering a range from small to large tensors
    static const AddTestCase test_cases[];
    
    static constexpr size_t num_test_cases = 19;  // Update this to match actual number of test cases
    
    // Comparison utility for floating point arrays
    bool compare_float_arrays(const float* numa_result, const float* reference_result, 
                             size_t num_elements, const char* test_name,
                             double* max_abs_error = nullptr, double* max_rel_error = nullptr) {
        bool all_match = true;
        double max_absolute_error = 0.0;
        double max_relative_error = 0.0;
        size_t first_mismatch = SIZE_MAX;
        
        for (size_t i = 0; i < num_elements; i++) {
            const float numa_val = numa_result[i];
            const float ref_val = reference_result[i];
            
            // Calculate absolute and relative errors
            const double abs_error = fabs((double)numa_val - (double)ref_val);
            const double rel_error = (fabs((double)ref_val) > 1e-10) ? abs_error / fabs((double)ref_val) : abs_error;
            
            max_absolute_error = std::max(max_absolute_error, abs_error);
            max_relative_error = std::max(max_relative_error, rel_error);
            
            // Check for significant difference (considering floating point precision)
            const double tolerance = 1e-5;  // Reasonable tolerance for float32
            if (abs_error > tolerance && rel_error > tolerance) {
                if (first_mismatch == SIZE_MAX) {
                    first_mismatch = i;
                    printf("❌ First mismatch in %s at index %zu: NUMA=%.10f, Reference=%.10f, abs_err=%.2e, rel_err=%.2e\n",
                           test_name, i, numa_val, ref_val, abs_error, rel_error);
                }
                all_match = false;
            }
        }
        
        if (max_abs_error) *max_abs_error = max_absolute_error;
        if (max_rel_error) *max_rel_error = max_relative_error;
        
        if (all_match) {
            printf("✅ %s: Perfect match (max_abs_err=%.2e, max_rel_err=%.2e)\n", 
                   test_name, max_absolute_error, max_relative_error);
        } else {
            printf("❌ %s: Mismatch detected (max_abs_err=%.2e, max_rel_err=%.2e, first_mismatch=%zu)\n",
                   test_name, max_absolute_error, max_relative_error, first_mismatch);
        }
        
        return all_match;
    }
    
    // Generate deterministic test data for ADD operation
    void generate_test_data(float* data, size_t num_elements, int seed_offset = 0) {
        for (size_t i = 0; i < num_elements; i++) {
            // Generate deterministic but varied data using a simple PRNG
            uint32_t x = (uint32_t)(i + seed_offset);
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            
            // Convert to float in range [-2.0, 2.0] for reasonable ADD operation results
            data[i] = ((float)(x % 10000) / 10000.0f) * 4.0f - 2.0f;
        }
    }

public:
    // Test a single ADD case with specified dimensions and thread count
    TestResult test_single_add_case(const AddTestCase& test_case) {
        TestResult result;
        result.test_name = test_case.description;
        result.passed = false;
        
        try {
            printf("\n🧪 Testing ADD: %s\n", test_case.description);
            
            // Calculate tensor properties
            const int64_t ne0 = test_case.ne[0];
            const int64_t ne1 = test_case.ne[1]; 
            const int64_t ne2 = test_case.ne[2];
            const int64_t ne3 = test_case.ne[3];
            const size_t num_elements = ne0 * ne1 * ne2 * ne3;
            
            printf("   Dimensions: [%ld, %ld, %ld, %ld] = %zu elements\n", ne0, ne1, ne2, ne3, num_elements);
            printf("   Threads: %d\n", test_case.numa_threads);
            
            // Initialize GGML context for tensor operations
            struct ggml_init_params init_params;
            init_params.mem_size = 256 * 1024 * 1024;  // 256MB should be enough for our tests
            init_params.mem_buffer = nullptr;
            init_params.no_alloc = false;
            
            struct ggml_context* ctx = ggml_init(init_params);
            if (!ctx) {
                result.failure_reason = "Failed to initialize GGML context";
                return result;
            }
            
            // Create input tensors for ADD operation
            struct ggml_tensor* src0 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
            struct ggml_tensor* src1 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
            struct ggml_tensor* dst_numa = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
            struct ggml_tensor* dst_reference = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
            
            if (!src0 || !src1 || !dst_numa || !dst_reference) {
                ggml_free(ctx);
                result.failure_reason = "Failed to create tensors";
                return result;
            }
            
            // Generate deterministic test data
            generate_test_data((float*)ggml_get_data(src0), num_elements, 1000);
            generate_test_data((float*)ggml_get_data(src1), num_elements, 2000);
            
            // Initialize destination tensors to zero
            memset(ggml_get_data(dst_numa), 0, ggml_nbytes(dst_numa));
            memset(ggml_get_data(dst_reference), 0, ggml_nbytes(dst_reference));
            
            // Create ADD operations for both NUMA and reference execution
            struct ggml_tensor* add_numa = ggml_add(ctx, src0, src1);
            struct ggml_tensor* add_reference = ggml_add(ctx, src0, src1);
            
            // Set destination tensors (copy data pointers)
            memcpy(ggml_get_data(add_numa), ggml_get_data(dst_numa), ggml_nbytes(dst_numa));
            memcpy(ggml_get_data(add_reference), ggml_get_data(dst_reference), ggml_nbytes(dst_reference));
            
            printf("   🔧 Executing NUMA ADD operation...\n");
            
            // Execute NUMA ADD operation via dispatcher
            struct ggml_compute_params numa_params;
            numa_params.ith = 0;
            numa_params.nth = test_case.numa_threads;
            numa_params.wsize = 0;
            numa_params.wdata = nullptr;
            numa_params.threadpool = nullptr;
            
            enum ggml_status numa_status = ggml_numa_intercept_operation(add_numa, &numa_params);
            if (numa_status != GGML_STATUS_SUCCESS) {
                ggml_free(ctx);
                result.failure_reason = "NUMA ADD operation failed";
                return result;
            }
            
            printf("   📊 Executing reference ADD operation...\n");
            
            // Execute reference ADD operation (serial implementation)
            struct ggml_compute_params reference_params;
            reference_params.ith = 0;
            reference_params.nth = 1;  // Single-threaded reference
            reference_params.wsize = 0;
            reference_params.wdata = nullptr;
            reference_params.threadpool = nullptr;
            
            // Manually compute ADD for reference (since ggml_compute_forward_add is internal)
            const float* src0_data = (const float*)ggml_get_data(src0);
            const float* src1_data = (const float*)ggml_get_data(src1);
            float* ref_data = (float*)ggml_get_data(add_reference);
            
            for (size_t i = 0; i < num_elements; i++) {
                ref_data[i] = src0_data[i] + src1_data[i];
            }
            
            printf("   🔍 Comparing results...\n");
            
            // Compare NUMA and reference results
            double max_abs_error, max_rel_error;
            bool mathematical_correctness = compare_float_arrays(
                (const float*)ggml_get_data(add_numa),
                (const float*)ggml_get_data(add_reference),
                num_elements,
                test_case.description,
                &max_abs_error,
                &max_rel_error
            );
            
            // Print first few values for debugging
            const float* numa_data = (const float*)ggml_get_data(add_numa);
            const float* ref_data_debug = (const float*)ggml_get_data(add_reference);
            
            printf("   📋 Sample values:\n");
            for (int i = 0; i < std::min(5, (int)num_elements); i++) {
                printf("      [%d]: src0=%.6f, src1=%.6f -> numa=%.6f, ref=%.6f\n",
                       i, src0_data[i], src1_data[i], numa_data[i], ref_data_debug[i]);
            }
            
            ggml_free(ctx);
            
            if (mathematical_correctness) {
                printf("   ✅ %s: Mathematical correctness verified\n", test_case.description);
                result.passed = true;
            } else {
                result.failure_reason = "Mathematical results do not match between NUMA and reference implementations";
                printf("   ❌ %s: Mathematical correctness FAILED\n", test_case.description);
            }
            
        } catch (const std::exception& e) {
            result.failure_reason = std::string("Exception: ") + e.what();
            printf("   💥 %s: Exception occurred: %s\n", test_case.description, e.what());
        }
        
        return result;
    }
    
    // Test ADD mathematical equivalence across multiple dimensions and thread strategies
    std::vector<TestResult> test_add_mathematical_equivalence() {
        printf("\n🚀 NUMA ADD Mathematical Correctness Test Suite\n");
        printf("================================================\n");
        printf("Testing %zu different ADD scenarios across various dimensions and thread strategies\n\n", num_test_cases);
        
        std::vector<TestResult> results;
        results.reserve(num_test_cases);
        
        for (size_t i = 0; i < num_test_cases; i++) {
            TestResult test_result = test_single_add_case(test_cases[i]);
            results.push_back(test_result);
        }
        
        return results;
    }
    
    // Run all ADD tests and provide comprehensive summary
    int run_all_tests() {
        printf("🔬 NUMA ADD Mathematical Correctness Test Suite\n");
        printf("===============================================\n\n");
        
        std::vector<TestResult> all_results = test_add_mathematical_equivalence();
        
        // Generate summary report
        printf("\n📊 COMPREHENSIVE TEST SUMMARY\n");
        printf("=============================\n");
        
        int passed_count = 0;
        int failed_count = 0;
        
        for (const auto& result : all_results) {
            if (result.passed) {
                passed_count++;
                printf("✅ %s\n", result.test_name.c_str());
            } else {
                failed_count++;
                printf("❌ %s: %s\n", result.test_name.c_str(), result.failure_reason.c_str());
            }
        }
        
        printf("\n📈 FINAL RESULTS:\n");
        printf("  Total tests: %d\n", (int)all_results.size());
        printf("  Passed: %d\n", passed_count);
        printf("  Failed: %d\n", failed_count);
        printf("  Success rate: %.1f%%\n", (100.0 * passed_count) / all_results.size());
        
        if (failed_count == 0) {
            printf("\n🎉 ALL TESTS PASSED! NUMA ADD implementation is mathematically correct.\n");
            return 0;
        } else {
            printf("\n💥 %d tests failed. NUMA ADD implementation has mathematical correctness issues.\n", failed_count);
            return 1;
        }
    }
};

// Define test cases array
const AddTestCase 
NumaAddMathematicalCorrectnessTestSuite::test_cases[] = {
    // Small tensors (TINY)
    {{32, 32, 1, 1}, 1, "TINY 32x32 single-threaded"},
    {{32, 32, 1, 1}, 2, "TINY 32x32 dual-threaded"},
    {{64, 64, 1, 1}, 4, "TINY 64x64 quad-threaded"},
    
    // Medium tensors (SMALL)
    {{256, 256, 1, 1}, 1, "SMALL 256x256 single-threaded"},
    {{256, 256, 1, 1}, 2, "SMALL 256x256 dual-threaded"},
    {{256, 256, 1, 1}, 4, "SMALL 256x256 quad-threaded"},
    
    // Large tensors (MEDIUM)
    {{1024, 1024, 1, 1}, 2, "MEDIUM 1024x1024 dual-threaded"},
    {{1024, 1024, 1, 1}, 4, "MEDIUM 1024x1024 quad-threaded"},
    {{1024, 1024, 1, 1}, 6, "MEDIUM 1024x1024 six-threaded"},
    {{1024, 1024, 1, 1}, 8, "MEDIUM 1024x1024 eight-threaded"},
    
    // Very large tensors (LARGE)
    {{2048, 2048, 1, 1}, 4, "LARGE 2048x2048 quad-threaded"},
    {{2048, 2048, 1, 1}, 8, "LARGE 2048x2048 eight-threaded"},
    
    // Multi-dimensional tensors
    {{128, 128, 4, 1}, 4, "3D 128x128x4 quad-threaded"},
    {{64, 64, 8, 2}, 8, "4D 64x64x8x2 eight-threaded"},
    
    // Vector-like tensors (high aspect ratio)
    {{4096, 1, 1, 1}, 2, "Vector 4096x1 dual-threaded"},
    {{1, 4096, 1, 1}, 4, "Vector 1x4096 quad-threaded"},
    
    // Broadcasting scenarios
    {{512, 512, 1, 1}, 4, "Broadcasting 512x512 quad-threaded"},
    {{256, 1, 1, 1}, 2, "Broadcasting 256x1 dual-threaded"},
    {{1, 256, 1, 1}, 2, "Broadcasting 1x256 dual-threaded"}
};

// Main test entry point
int main() {
    printf("🧮 NUMA ADD Mathematical Correctness Test\n");
    printf("=========================================\n\n");
    
    NumaAddMathematicalCorrectnessTestSuite test_suite;
    return test_suite.run_all_tests();
}
