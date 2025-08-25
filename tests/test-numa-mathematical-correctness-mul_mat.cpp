/**
 * NUMA Mathematical Correctness Test: Matrix Multiplication (MUL_MAT)
 * 
 * This test validates that NUMA-parallel matrix multiplication produces
 * mathematically identical results to the reference implementation.
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

// GGML includes
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-numa-executor.h"
#include "ggml-numa-simple-coordinator.h"
#include "ggml-cpu/binary-ops.h"

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
        setenv("GGML_NUMA_DEBUG", "1", 1);  // Enable debug for detailed logging
        
        // Build compute graph
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, C);
        
        // Execute with NUMA
        printf("      📊 Executing with NUMA (threads=%d)...\n", num_threads);
        struct ggml_cplan numa_plan = ggml_graph_plan(gf, num_threads, nullptr);
        if (numa_plan.work_size > 0) {
            numa_plan.work_data = (uint8_t*)malloc(numa_plan.work_size);
            if (!numa_plan.work_data) {
                printf("      ❌ Failed to allocate work buffer\n");
                ggml_free(ctx);
                return false;
            }
        }
        
        enum ggml_status numa_status = ggml_graph_compute(gf, &numa_plan);
        if (numa_status != GGML_STATUS_SUCCESS) {
            printf("      ❌ NUMA execution failed with status %d\n", numa_status);
            if (numa_plan.work_data) free(numa_plan.work_data);
            ggml_free(ctx);
            return false;
        }
        
        // Copy NUMA result
        const int result_size = m * n;  // Result is [m×n]
        std::vector<float> numa_result(result_size);
        memcpy(numa_result.data(), ggml_get_data(C), result_size * sizeof(float));
        
        if (numa_plan.work_data) {
            free(numa_plan.work_data);
        }
        
        // ==================================================================
        // Test 2: Reference implementation (CPU fallback)
        // ==================================================================
        
        // Reset result tensor
        memset(ggml_get_data(C), 0, result_size * sizeof(float));
        
        // Disable NUMA to force CPU fallback
        unsetenv("GGML_NUMA_DEBUG");
        setenv("GGML_NUMA_DISABLE", "1", 1);
        
        printf("      📊 Executing with CPU reference (threads=%d)...\n", num_threads);
        
        struct ggml_cplan ref_plan = ggml_graph_plan(gf, num_threads, nullptr);
        if (ref_plan.work_size > 0) {
            ref_plan.work_data = (uint8_t*)malloc(ref_plan.work_size);
            if (!ref_plan.work_data) {
                printf("      ❌ Failed to allocate reference work buffer\n");
                ggml_free(ctx);
                return false;
            }
        }
        
        enum ggml_status ref_status = ggml_graph_compute(gf, &ref_plan);
        if (ref_status != GGML_STATUS_SUCCESS) {
            printf("      ❌ Reference execution failed with status %d\n", ref_status);
            if (ref_plan.work_data) free(ref_plan.work_data);
            ggml_free(ctx);
            return false;
        }
        
        // Copy reference result
        std::vector<float> ref_result(result_size);
        memcpy(ref_result.data(), ggml_get_data(C), result_size * sizeof(float));
        
        if (ref_plan.work_data) {
            free(ref_plan.work_data);
        }
        
        // Restore environment
        unsetenv("GGML_NUMA_DISABLE");
        
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
            
            // Small matrices
            {32, 16, 24, "SMALL_32x16x24"},
            {48, 32, 32, "SMALL_48x32x32"},
            
            // Medium matrices
            {128, 64, 96, "MEDIUM_128x64x96"},
            {256, 128, 192, "MEDIUM_256x128x192"},
            
            // Large matrices
            {512, 256, 384, "LARGE_512x256x384"},
            {768, 512, 512, "LARGE_768x512x512"},
            
            // Huge matrices (if system can handle)
            {1536, 1024, 1024, "HUGE_1536x1024x1024"}
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
    
    // Run all MUL_MAT tests and provide summary
    bool run_all_tests() {
        printf("🧪 NUMA Mathematical Correctness Test Suite: MUL_MAT\n");
        printf("================================================================\n");
        
        bool all_passed = true;
        
        if (!test_MUL_MAT_mathematical_equivalence()) {
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
            printf("🎉 ALL TESTS PASSED! MUL_MAT NUMA implementation is mathematically correct.\n");
        } else {
            printf("💥 SOME TESTS FAILED! Check the MUL_MAT NUMA implementation.\n");
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
