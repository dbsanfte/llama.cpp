/**
 * NUMA Mathematical Correctness Test: Matrix Multiplication (MUL_MAT)
 * 
 * This test validates that NUMA-parallel matrix multiplication produces
 * mathematically identical results to the reference implem        if (results_match) {
            printf("      ✅ %s: MUL_MAT results match\n", size_label);
        } else {
            printf("      ❌ %s: MUL_MAT results do not match\n", size_label);
        }
        
        ggml_free(ref_ctx);  // Clean up reference context
        ggml_free(ctx);
        return results_match;.
 * 
 * Test Strategy:
 * - Multi-dimensional matrices from tiny to gigantic scales
 * - Various thread counts to test coordinator execution
 * - Direct comparison between NUMA parallel and serial reference
 * - Comprehensive error reporting for any mismatches
 */

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <random>

// GGML includes
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-numa-executor.h"
#include "ggml-numa-simple-coordinator.h"
#include "ggml-cpu/binary-ops.h"

// Forward declarations
extern "C" void ggml_compute_forward_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst);

// F16 dot product test declarations
extern "C" {
    void ggml_numa_vec_dot_f16_custom(int n, float* s, size_t s_off,
                                     const void* x, size_t x_off,
                                     const void* y, size_t y_off, int nrc);
    void ggml_vec_dot_f16(int n, float * s, size_t bs, ggml_fp16_t * x, size_t bx, ggml_fp16_t * y, size_t by, int nrc);
}

// Test result structure
struct TestResult {
    std::string test_name;
    bool passed;
    std::string failure_reason;
};

class NumaMathematicalCorrectnessTestSuite {
private:
    std::vector<TestResult> results;
    
    // Utility function to compare float arrays with detailed error reporting
    bool compare_float_arrays(const float* numa_data, const float* ref_data, int count, const char* operation_name) {
        bool all_match = true;
        int error_count = 0;
        double max_abs_error = 0.0;
        double max_rel_error = 0.0;
        
        for (int i = 0; i < count; i++) {
            double numa_val = numa_data[i];
            double ref_val = ref_data[i];
            double abs_error = fabs(numa_val - ref_val);
            double rel_error = ref_val != 0.0 ? abs_error / fabs(ref_val) : 0.0;
            
            max_abs_error = fmax(max_abs_error, abs_error);
            max_rel_error = fmax(max_rel_error, rel_error);
            
            // Use reasonable tolerance for matrix multiplication (accumulated floating-point errors)
            if (abs_error > 1e-4 && rel_error > 1e-4) {
                if (error_count < 5) { // Show first 5 errors for debugging
                    printf("      ❌ %s Element[%d]: NUMA=%.8f, Reference=%.8f, AbsErr=%.2e, RelErr=%.2e\n",
                           operation_name, i, numa_val, ref_val, abs_error, rel_error);
                }
                error_count++;
                all_match = false;
            }
        }
        
        if (!all_match) {
            printf("    Total errors: %d/%d, MaxAbsErr=%.2e, MaxRelErr=%.2e\n", 
                   error_count, count, max_abs_error, max_rel_error);
        }
        
        return all_match;
    }
    
    // F16 dot product utility functions
    float reference_dot_product_f32(const std::vector<ggml_fp16_t>& x, const std::vector<ggml_fp16_t>& y) {
        float sum = 0.0f;
        for (size_t i = 0; i < x.size(); i++) {
            float x_f32 = ggml_fp16_to_fp32(x[i]);
            float y_f32 = ggml_fp16_to_fp32(y[i]);
            sum += x_f32 * y_f32;
        }
        return sum;
    }
    
    bool within_tolerance(float a, float b, float tol) {
        float abs_diff = std::abs(a - b);
        float max_val = std::max(std::abs(a), std::abs(b));
        
        // Use relative tolerance for large values, absolute for small
        if (max_val > 1.0f) {
            return abs_diff <= tol * max_val;
        } else {
            return abs_diff <= tol;
        }
    }
    
    void generate_f16_test_vectors(std::vector<ggml_fp16_t>& x, std::vector<ggml_fp16_t>& y, int n, int pattern, std::mt19937& rng) {
        x.resize(n);
        y.resize(n);
        
        switch (pattern) {
            case 0: { // Sequential pattern
                for (int i = 0; i < n; i++) {
                    x[i] = ggml_fp32_to_fp16((i + 1) * 0.1f);
                    y[i] = ggml_fp32_to_fp16((i + 1) * 0.05f);
                }
                break;
            }
                
            case 1: { // Alternating pattern  
                for (int i = 0; i < n; i++) {
                    x[i] = ggml_fp32_to_fp16((i % 2 == 0) ? 1.0f : -1.0f);
                    y[i] = ggml_fp32_to_fp16((i % 3 == 0) ? 2.0f : 0.5f);
                }
                break;
            }
                
            case 2: { // Random pattern
                std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
                for (int i = 0; i < n; i++) {
                    x[i] = ggml_fp32_to_fp16(dist(rng));
                    y[i] = ggml_fp32_to_fp16(dist(rng));
                }
                break;
            }
                
            case 3: { // Edge cases
                for (int i = 0; i < n; i++) {
                    float x_val = (i == 0) ? 0.0f : ((i == n-1) ? 1.0f : 0.1f);
                    float y_val = (i == 0) ? 1.0f : ((i == n-1) ? 0.0f : 0.2f);
                    x[i] = ggml_fp32_to_fp16(x_val);
                    y[i] = ggml_fp32_to_fp16(y_val);
                }
                break;
            }
        }
    }
    
    // Test single F16 dot product case
    bool test_single_f16_dot_product_case(int n, const char* description) {
        printf("      🧮 Testing F16 Dot Product %s (length=%d)\n", description, n);
        
        std::mt19937 rng(12345);
        const float tolerance = 1e-3f;  // F16 precision tolerance
        bool all_patterns_passed = true;
        const char* pattern_names[] = {"Sequential", "Alternating", "Random", "EdgeCases"};
        
        // Test multiple data patterns
        for (int pattern = 0; pattern < 4; pattern++) {
            std::vector<ggml_fp16_t> x, y;
            generate_f16_test_vectors(x, y, n, pattern, rng);
            
            // Test our custom implementation
            float custom_result = 0.0f;
            ggml_numa_vec_dot_f16_custom(n, &custom_result, 0, x.data(), 0, y.data(), 0, 1);
            
            // Test GGML reference implementation
            float ggml_result = 0.0f;
            ggml_vec_dot_f16(n, &ggml_result, 0, x.data(), 0, y.data(), 0, 1);
            
            // Test with our F32 reference for absolute truth
            float f32_reference = reference_dot_product_f32(x, y);
            
            // Check against F32 reference (which should be most accurate)
            float error = std::abs(custom_result - f32_reference);
            bool passed = within_tolerance(custom_result, f32_reference, tolerance);
            
            if (passed) {
                printf("        ✅ %s_%s: Results match (Custom=%.6f, F32Ref=%.6f, GGML=%.6f, Err=%.2e)\n", 
                       description, pattern_names[pattern], custom_result, f32_reference, ggml_result, error);
            } else {
                printf("        ❌ %s_%s: Results differ (Custom=%.6f, F32Ref=%.6f, GGML=%.6f, Err=%.2e)\n", 
                       description, pattern_names[pattern], custom_result, f32_reference, ggml_result, error);
                all_patterns_passed = false;
            }
        }
        
        return all_patterns_passed;
    }
    
    // Test a single MUL_MAT case with specific dimensions and thread count
    bool test_single_MUL_MAT_case(int k, int m, int n, int num_threads, const char* size_label) {
        printf("    🧮 Testing %s: MUL_MAT with dimensions [%d×%d] × [%d×%d] (threads=%d)\n", 
               size_label, k, m, k, n, num_threads);
        
        // Create test context with sufficient memory for larger tensors
        struct ggml_init_params params;
        size_t total_elements = k * m + k * n + m * n; // A + B + C matrices
        params.mem_size = std::max((size_t)(512 * 1024 * 1024), total_elements * sizeof(float) * 4); // Scale memory with tensor size
        params.mem_buffer = nullptr;
        params.no_alloc = false;
        
        struct ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            printf("      ❌ Failed to create ggml context\n");
            return false;
        }
        
        // Create matrices A [k×m] and B [k×n] following GGML convention
        // GGML performs A * B^T effectively: [k,m] * [k,n] => [m,n]
        // Both matrices share the same width k (constraint: t0->ne[0] == t1->ne[0])
        struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);  // [k, m]
        struct ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);  // [k, n]
        
        if (!A || !B) {
            printf("      ❌ Failed to create input tensors\n");
            ggml_free(ctx);
            return false;
        }
        
        // Initialize matrices with deterministic values for reproducible tests
        float * A_data = (float *)ggml_get_data(A);
        float * B_data = (float *)ggml_get_data(B);
        
        // Fill A with pattern: A[col][row] = (col * m + row + 1) * 0.1
        for (int row = 0; row < m; row++) {
            for (int col = 0; col < k; col++) {
                A_data[col * m + row] = (col * m + row + 1) * 0.1f;
            }
        }
        
        // Fill B with pattern: B[col][row] = (col * n + row + 1) * 0.05  
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < k; col++) {
                B_data[col * n + row] = (col * n + row + 1) * 0.05f;
            }
        }
        
        // Create result tensor C = A × B (effectively A * B^T) => [m×n]
        struct ggml_tensor * C = ggml_mul_mat(ctx, A, B);
        if (!C) {
            printf("      ❌ Failed to create result tensor\n");
            ggml_free(ctx);
            return false;
        }
        
        // ==================================================================
        // Test 1: NUMA execution
        // ==================================================================
        
        // Enable NUMA mode for coordinator testing  
        printf("      📊 Executing with NUMA (threads=%d)...\n", num_threads);
        
        // Execute with NUMA executor directly (like ADD test does)
        struct ggml_cplan numa_plan = {};
        numa_plan.work_size = 0;
        numa_plan.work_data = nullptr;
        numa_plan.n_threads = num_threads;
        numa_plan.threadpool = nullptr;
        numa_plan.abort_callback = nullptr;
        numa_plan.abort_callback_data = nullptr;
        
        // Execute with new executor architecture
        enum ggml_status numa_status = ggml_numa_executor_execute_tensor(C, &numa_plan);
        if (numa_status != GGML_STATUS_SUCCESS) {
            printf("      ❌ NUMA execution failed with status %d\n", numa_status);
            ggml_free(ctx);
            return false;
        }
        
        // Copy NUMA result
        const int result_size = m * n;  // Result is [m×n]
        std::vector<float> numa_result(result_size);
        memcpy(numa_result.data(), ggml_get_data(C), result_size * sizeof(float));
        
        // ==================================================================
        // Test 2: Reference implementation (CPU fallback)
        // ==================================================================
        
        // Reset result tensor
        memset(ggml_get_data(C), 0, result_size * sizeof(float));
        
        printf("      📊 Executing with CPU reference (threads=%d)...\n", num_threads);
        
        // Create a separate context for reference computation to avoid NUMA interference
        struct ggml_init_params ref_init_params;
        ref_init_params.mem_size = std::max((size_t)(512 * 1024 * 1024), total_elements * sizeof(float) * 8);  // Extra space
        ref_init_params.mem_buffer = nullptr;
        ref_init_params.no_alloc = false;
        
        struct ggml_context* ref_ctx = ggml_init(ref_init_params);
        if (!ref_ctx) {
            printf("Failed to create reference context\n");
            ggml_free(ctx);
            return false;
        }
        
        // Create reference tensors (identical layout to NUMA tensors)
        struct ggml_tensor* ref_A = ggml_new_tensor_4d(ref_ctx, GGML_TYPE_F32, k, m, 1, 1);  // [k, m, 1, 1] 
        struct ggml_tensor* ref_B = ggml_new_tensor_4d(ref_ctx, GGML_TYPE_F32, k, n, 1, 1);  // [k, n, 1, 1]
        struct ggml_tensor* ref_C = ggml_mul_mat(ref_ctx, ref_A, ref_B);
        
        // Copy input data to reference tensors
        memcpy(ggml_get_data(ref_A), ggml_get_data(A), ggml_nbytes(A));
        memcpy(ggml_get_data(ref_B), ggml_get_data(B), ggml_nbytes(B));
        
        // Build and compute reference graph using standard GGML (no NUMA)
        struct ggml_cgraph* ref_cgraph = ggml_new_graph(ref_ctx);
        ggml_build_forward_expand(ref_cgraph, ref_C);
        
        // Execute using standard CPU computation
        ggml_graph_compute_with_ctx(ref_ctx, ref_cgraph, num_threads);
        
        // Copy reference result
        std::vector<float> ref_result(result_size);
        memcpy(ref_result.data(), ggml_get_data(ref_C), result_size * sizeof(float));
        
        // ==================================================================
        // Test 3: Compare results
        // ==================================================================
        
        printf("      🔍 Comparing results (%d elements)...\n", result_size);
        bool results_match = compare_float_arrays(numa_result.data(), ref_result.data(), result_size, "MUL_MAT");
        
        if (results_match) {
            printf("      ✅ %s: MUL_MAT results match perfectly\n", size_label);
        } else {
            printf("      ❌ %s: MUL_MAT results do not match\n", size_label);
        }
        
        ggml_free(ctx);
        return results_match;
    }
    
public:
    // Test MUL_MAT mathematical equivalence across multiple dimensions and thread counts
    bool test_MUL_MAT_mathematical_equivalence() {
        printf("  🧮 Testing MUL_MAT Mathematical Equivalence\n");
        
        // Test dimensions: {k, m, n} where result is [m×n] = [k×m] * [k×n]
        std::vector<std::tuple<int, int, int, std::string>> test_cases = {
            // Tiny matrices
            {4, 4, 4, "TINY_4x4x4"},
            {6, 8, 10, "TINY_6x8x10"},
            {8, 12, 16, "TINY_8x12x16"},
            
            // Small matrices  
            {16, 32, 48, "SMALL_16x32x48"},
            {32, 16, 24, "SMALL_32x16x24"},
            {24, 48, 32, "SMALL_24x48x32"},
            
            // Medium matrices
            {64, 128, 96, "MEDIUM_64x128x96"},
            {128, 64, 96, "MEDIUM_128x64x96"},
            {96, 192, 128, "MEDIUM_96x192x128"},
            
            // Large matrices 
            {256, 512, 384, "LARGE_256x512x384"},
            {512, 256, 384, "LARGE_512x256x384"},
            {384, 768, 512, "LARGE_384x768x512"}
        };
        
        // Test different thread counts
        std::vector<int> thread_counts = {1, 2, 4, 6, 8};
        
        bool all_passed = true;
        
        for (const auto& test_case : test_cases) {
            int k = std::get<0>(test_case);
            int m = std::get<1>(test_case);
            int n = std::get<2>(test_case);
            std::string label = std::get<3>(test_case);
            
            for (int num_threads : thread_counts) {
                std::string test_name = "MUL_MAT_" + label + "_threads" + std::to_string(num_threads);
                
                try {
                    bool passed = test_single_MUL_MAT_case(k, m, n, num_threads, label.c_str());
                    
                    results.push_back({
                        test_name,
                        passed,
                        passed ? "" : "Mathematical mismatch between NUMA and reference"
                    });
                    
                    if (!passed) {
                        all_passed = false;
                        printf("    ❌ FAILED: %s\n", test_name.c_str());
                    } else {
                        printf("    ✅ PASSED: %s\n", test_name.c_str());
                    }
                    
                } catch (const std::exception& e) {
                    printf("    💥 EXCEPTION in %s: %s\n", test_name.c_str(), e.what());
                    results.push_back({test_name, false, std::string("Exception: ") + e.what()});
                    all_passed = false;
                }
            }
        }
        
        return all_passed;
    }
    
    // Test F16 dot product mathematical equivalence
    bool test_f16_dot_product_mathematical_equivalence() {
        printf("  🧮 Testing F16 Dot Product Mathematical Equivalence\n");
        
        // Test different vector sizes
        struct {
            int size;
            const char* name;
        } test_sizes[] = {
            {4, "TINY_4"},
            {16, "SMALL_16"}, 
            {64, "MEDIUM_64"},
            {256, "LARGE_256"},
            {1024, "HUGE_1024"},
            {4096, "GIGANTIC_4096"}
        };
        
        bool all_passed = true;
        
        for (auto& test : test_sizes) {
            std::string test_name = "F16_DOT_PRODUCT_" + std::string(test.name);
            
            try {
                bool passed = test_single_f16_dot_product_case(test.size, test.name);
                
                results.push_back({
                    test_name,
                    passed,
                    passed ? "" : "Mathematical mismatch in F16 dot product"
                });
                
                if (!passed) {
                    all_passed = false;
                    printf("    ❌ FAILED: %s\n", test_name.c_str());
                } else {
                    printf("    ✅ PASSED: %s\n", test_name.c_str());
                }
                
            } catch (const std::exception& e) {
                printf("    💥 EXCEPTION in %s: %s\n", test_name.c_str(), e.what());
                results.push_back({test_name, false, std::string("Exception: ") + e.what()});
                all_passed = false;
            }
        }
        
        return all_passed;
    }
    
    // Run all MUL_MAT tests and provide summary
    bool run_all_tests() {
        printf("🧪 NUMA Mathematical Correctness Test Suite: MUL_MAT & F16 DOT PRODUCT\n");
        printf("================================================================\n");
        
        bool all_passed = true;
        
        if (!test_MUL_MAT_mathematical_equivalence()) {
            all_passed = false;
        }
        
        if (!test_f16_dot_product_mathematical_equivalence()) {
            all_passed = false;
        }
        
        // Print summary
        printf("\n📊 Test Summary\n");
        printf("===============\n");
        
        int passed_count = 0;
        int failed_count = 0;
        
        for (const auto& result : results) {
            if (result.passed) {
                passed_count++;
                printf("✅ %s\n", result.test_name.c_str());
            } else {
                failed_count++;
                printf("❌ %s: %s\n", result.test_name.c_str(), result.failure_reason.c_str());
            }
        }
        
        printf("\n🎯 Final Results\n");
        printf("================\n");
        printf("Passed: %d\n", passed_count);
        printf("Failed: %d\n", failed_count);
        printf("Total:  %d\n", passed_count + failed_count);
        
        if (all_passed) {
            printf("🎉 ALL TESTS PASSED! MUL_MAT & F16 DOT PRODUCT NUMA implementations are mathematically correct.\n");
        } else {
            printf("💥 SOME TESTS FAILED! Check the MUL_MAT & F16 DOT PRODUCT NUMA implementations.\n");
        }
        
        return all_passed;
    }
};

int main() {
    try {
        NumaMathematicalCorrectnessTestSuite test_suite;
        bool success = test_suite.run_all_tests();
        return success ? 0 : 1;
    } catch (const std::exception& e) {
        printf("💥 Fatal error: %s\n", e.what());
        return 1;
    }
}
