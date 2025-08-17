/**
 * NUMA GLU Mathematical Correctness Test
 * 
 * This test validates mathematical equivalence between NUMA parallel GLU operations 
 * and serial reference implementations across all GLU variants.
 * 
 * Test Coverage:
 * - All GLU variants: REGLU, GEGLU, SwiGLU, GEGLU_ERF, GEGLU_QUICK
 * - Multiple tensor dimensions: TINY, SMALL, MEDIUM, LARGE
 * - Multiple thread strategies: 1, 2, 4, 6, 8 threads
 * - Comprehensive error reporting with detailed mismatch information
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
#include "ggml-cpu/ops.h"

// Test result structure
struct TestResult {
    std::string test_name;
    bool passed;
    std::string failure_reason;
};

class NumaGluMathematicalCorrectnessTestSuite {
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
            
            // Use strict tolerance for mathematical equivalence
            if (abs_error > 1e-6 && rel_error > 1e-6) {
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
    
    // Test a single TEMPLATE_OPERATION case with specific dimensions and thread count
    // TODO: Implement this method for your specific operation
    bool test_single_glu_case(int dim1, int dim2, int dim3, int glu_variant, int num_threads, const char* size_label) {
        const char* glu_names[] = {"REGLU", "GEGLU", "SwiGLU", "GEGLU_ERF", "GEGLU_QUICK"};
        printf("    🧮 Testing %s: %s with dimensions [%d,%d,%d] (threads=%d)\n", 
               size_label, glu_names[glu_variant], dim1, dim2, dim3, num_threads);
        
        // Create test context with sufficient memory for GLU operations
        struct ggml_init_params params = {0};
        params.mem_size = std::max((size_t)(512 * 1024 * 1024), (size_t)(dim1 * dim2 * dim3) * sizeof(float) * 8);
        params.mem_buffer = nullptr;
        params.no_alloc = false;
        
        struct ggml_context* numa_ctx = ggml_init(params);
        struct ggml_context* ref_ctx = ggml_init(params);
        if (!numa_ctx || !ref_ctx) {
            printf("      ❌ Failed to create test contexts for %s\n", size_label);
            if (numa_ctx) ggml_free(numa_ctx);
            if (ref_ctx) ggml_free(ref_ctx);
            return false;
        }
        
        bool case_passed = false;
        
        // Create input tensors for GLU operation
        struct ggml_tensor* input_numa = ggml_new_tensor_3d(numa_ctx, GGML_TYPE_F32, dim1, dim2, dim3);
        struct ggml_tensor* input_ref = ggml_new_tensor_3d(ref_ctx, GGML_TYPE_F32, dim1, dim2, dim3);
        
        if (!input_numa || !input_ref) {
            printf("      ❌ Failed to create input tensors for %s\n", size_label);
        } else {
            // Fill tensors with deterministic test data
            int total_elements = dim1 * dim2 * dim3;
            float* numa_data = (float*)ggml_get_data(input_numa);
            float* ref_data = (float*)ggml_get_data(input_ref);
            
            for (int i = 0; i < total_elements; i++) {
                float val = 0.1f + (i % 100) * 0.01f; // Values between 0.1 and 1.09
                numa_data[i] = val;
                ref_data[i] = val;
            }
            
            // Map glu_variant to ggml_glu_op
            enum ggml_glu_op glu_op;
            switch (glu_variant) {
                case 0: glu_op = GGML_GLU_OP_REGLU; break;
                case 1: glu_op = GGML_GLU_OP_GEGLU; break;
                case 2: glu_op = GGML_GLU_OP_SWIGLU; break;
                case 3: glu_op = GGML_GLU_OP_GEGLU_ERF; break;
                case 4: glu_op = GGML_GLU_OP_GEGLU_QUICK; break;
                default: 
                    printf("      ❌ Unknown GLU variant: %d\n", glu_variant);
                    ggml_free(numa_ctx);
                    ggml_free(ref_ctx);
                    return false;
            }
            
            // Create GLU operations
            struct ggml_tensor* numa_result = ggml_glu(numa_ctx, input_numa, glu_op, false);
            struct ggml_tensor* ref_result = ggml_glu(ref_ctx, input_ref, glu_op, false);
            
            if (!numa_result || !ref_result) {
                printf("      ❌ Failed to create GLU operations for %s\n", size_label);
            } else {
                // Execute via NUMA intercept
                struct ggml_compute_params numa_params = {0};
                numa_params.ith = 0;
                numa_params.nth = num_threads;
                numa_params.wsize = 0;
                numa_params.wdata = nullptr;
                numa_params.threadpool = nullptr;
                
                enum ggml_status dispatch_result = ggml_numa_intercept_operation(numa_result, &numa_params);
                
                if (dispatch_result != GGML_STATUS_SUCCESS) {
                    printf("      ❌ NUMA dispatcher execution failed for %s (status=%d, threads=%d)\n", 
                           size_label, dispatch_result, num_threads);
                } else {
                    // Execute reference computation (serial)
                    struct ggml_compute_params ref_params = {0};
                    ref_params.ith = 0;
                    ref_params.nth = 1;
                    ref_params.wsize = 0;
                    ref_params.wdata = nullptr;
                    ref_params.threadpool = nullptr;
                    
                    ggml_compute_forward_glu(&ref_params, ref_result);
                    
                    // Compare results
                    float* numa_output = (float*)ggml_get_data(numa_result);
                    float* ref_output = (float*)ggml_get_data(ref_result);
                    int result_elements = ggml_nelements(numa_result);
                    
                    case_passed = compare_float_arrays(numa_output, ref_output, result_elements, glu_names[glu_variant]);
                    
                    if (case_passed) {
                        printf("      ✅ %s (%d threads): MATHEMATICALLY EQUIVALENT\n", size_label, num_threads);
                    } else {
                        printf("      ❌ %s (%d threads): MATHEMATICAL MISMATCH\n", size_label, num_threads);
                    }
                }
            }
        }
        
        ggml_free(numa_ctx);
        ggml_free(ref_ctx);
        
        return case_passed;
        printf("      ⚠️  TEMPLATE METHOD - IMPLEMENT FOR YOUR OPERATION\n");
        return true; // Placeholder - replace with actual test result
    }

    void test_glu_mathematical_equivalence() {
        printf("--- Test: TEMPLATE_OPERATION Mathematical Equivalence (Multi-Dimensional) ---\n");
        printf("Testing NUMA parallel TEMPLATE_OPERATION vs serial reference implementation...\n");
        printf("Testing across various tensor sizes with different coordinator execution strategies\n\n");
        
        bool overall_test_passed = true;
        const char* failure_reason = nullptr;
        
        // Define test cases for GLU operations across different tensor sizes and variants
        struct {
            int dim1, dim2, dim3;
            int glu_variant;  // 0=REGLU, 1=GEGLU, 2=SWIGLU, 3=GEGLU_ERF, 4=GEGLU_QUICK
            const char* label;
        } test_cases[] = {
            {64, 64, 1, 0, "REGLU-TINY"},           // REGLU with small tensors
            {64, 64, 1, 1, "GEGLU-TINY"},           // GEGLU with small tensors
            {128, 128, 1, 2, "SWIGLU-SMALL"},       // SwiGLU with medium tensors
            {256, 256, 1, 3, "GEGLU_ERF-MEDIUM"},   // GEGLU_ERF with larger tensors
            {512, 512, 1, 4, "GEGLU_QUICK-LARGE"},  // GEGLU_QUICK with large tensors
            {128, 256, 2, 0, "REGLU-RECT"},         // Rectangular tensors
            {256, 128, 2, 1, "GEGLU-RECT"},         // Different aspect ratio
            {64, 64, 4, 2, "SWIGLU-4D"}             // Multi-dimensional tensor
        };
        
        // Define coordinator execution strategies (various thread counts)
        int thread_strategies[] = {1, 2, 4, 6, 8};
        int num_strategies = sizeof(thread_strategies) / sizeof(thread_strategies[0]);
        int num_test_cases = sizeof(test_cases) / sizeof(test_cases[0]);
        
        printf("  🎯 Testing %d tensor dimensions with %d thread strategies (%d total test combinations)\n\n", 
               num_test_cases, num_strategies, num_test_cases * num_strategies);
        
        int total_tests = 0;
        int passed_tests = 0;
        
        // Test each tensor dimension with each thread strategy
        for (int case_idx = 0; case_idx < num_test_cases; case_idx++) {
            printf("  📏 Testing %s tensors (%dx%dx%d):\n", 
                   test_cases[case_idx].label, 
                   test_cases[case_idx].dim1, 
                   test_cases[case_idx].dim2, 
                   test_cases[case_idx].dim3);
            
            for (int strategy_idx = 0; strategy_idx < num_strategies; strategy_idx++) {
                int num_threads = thread_strategies[strategy_idx];
                
                bool case_passed = test_single_glu_case(
                    test_cases[case_idx].dim1, 
                    test_cases[case_idx].dim2, 
                    test_cases[case_idx].dim3,
                    test_cases[case_idx].glu_variant,
                    num_threads,
                    test_cases[case_idx].label
                );                total_tests++;
                if (case_passed) {
                    passed_tests++;
                } else {
                    overall_test_passed = false;
                    if (!failure_reason) {
                        failure_reason = "Mathematical mismatch detected in multi-dimensional testing";
                    }
                }
            }
            printf("\n");
        }
        
        // Print summary for this test
        printf("  📊 GLU Multi-Dimensional Test Summary:\n");
        printf("    Total test combinations: %d\n", total_tests);
        printf("    Passed: %d\n", passed_tests);
        printf("    Failed: %d\n", total_tests - passed_tests);
        
        if (overall_test_passed) {
            printf("✅ GLU mathematical equivalence (multi-dimensional): VERIFIED\n");
            printf("  🎉 All tensor dimensions and thread strategies produce mathematically equivalent results!\n\n");
        } else {
            printf("❌ GLU mathematical equivalence (multi-dimensional): FAILED - %s\n", failure_reason ? failure_reason : "Unknown error");
            printf("  ⚠️  Mathematical mismatches detected across different dimensions or thread strategies\n\n");
        }
        
        results.push_back({"glu_mathematical_equivalence", overall_test_passed, failure_reason ? failure_reason : "No error details"});
    }

public:
    bool run_all_tests() {
        printf("🧪 NUMA Mathematical Correctness Test Suite - GLU\n");
        printf("================================================================================\n");
        printf("🔧 Testing mathematical correctness with function pointer architecture\n");
        printf("Comparing NUMA parallel execution against serial reference implementation\n");
        printf("📊 Testing all GLU variants: REGLU, GEGLU, SwiGLU, GEGLU_ERF, GEGLU_QUICK\n\n");
        printf("================================================================================\n\n");
        
        // Run mathematical correctness tests
        test_glu_mathematical_equivalence();
        
        print_summary();
        
        // Check if any tests failed
        bool all_passed = true;
        for (const auto& result : results) {
            if (!result.passed) {
                all_passed = false;
                break;
            }
        }
        
        return all_passed;
    }

private:
    void print_summary() {
        printf("\n================================================================================\n");
        printf("                    Mathematical Correctness Test Results\n");
        printf("================================================================================\n");
        
        int passed_count = 0;
        for (const auto& result : results) {
            const char* status = result.passed ? "✅" : "❌";
            printf("%s %s: %s", status, result.test_name.c_str(), 
                   result.passed ? "PASSED" : "FAILED");
            
            if (!result.passed && !result.failure_reason.empty()) {
                printf(" - %s", result.failure_reason.c_str());
            }
            printf("\n");
            
            if (result.passed) passed_count++;
        }
        
        printf("------------------------------------------------------------------------\n");
        printf("Total: %d/%zu tests passed", passed_count, results.size());
        
        if (passed_count == (int)results.size()) {
            printf(" 🎉 All tests passed!\n");
        } else {
            printf(" 💥 Some tests failed.\n");
        }
        
        printf("================================================================================\n");
        
        if (passed_count != (int)results.size()) {
            printf("❌ NUMA Mathematical Correctness: FAILURES DETECTED\n\n");
            printf("⚠️  Mathematical mismatch between NUMA parallel and serial execution detected\n");
        } else {
            printf("✅ NUMA Mathematical Correctness: ALL TESTS PASSED\n\n");
            printf("🎯 NUMA parallel execution produces mathematically equivalent results\n");
        }
        
        printf("🧪 Mathematical correctness testing completed!\n\n");
    }
};

// Main function - entry point for the test
int main() {
    printf("🌟 Initializing NUMA system for mathematical correctness testing...\n");
    
    // Initialize the NUMA coordinator system
    struct ggml_numa_coordinator_manager* mgr = ggml_numa_coordinator_manager_get_global(8, false);
    if (!mgr) {
        printf("❌ Failed to initialize NUMA coordinator manager\n");
        return 1;
    }
    
    printf("✅ NUMA system initialized successfully\n\n");
    
    NumaGluMathematicalCorrectnessTestSuite test_suite;
    bool all_passed = test_suite.run_all_tests();
    
    return all_passed ? 0 : 1;
}

/**
 * IMPLEMENTATION CHECKLIST:
 * 
 * When adapting this template for a new operation:
 * 
 * 1. ✅ Replace all instances of "TEMPLATE_OPERATION" with your operation name
 * 2. ✅ Update test dimensions in test_cases[] array to match your operation's needs
 * 3. ✅ Implement test_single_OPERATION_case() with:
 *    - Appropriate tensor creation for your operation
 *    - Deterministic test data generation
 *    - NUMA operation execution via ggml_numa_intercept_operation
 *    - Reference implementation (serial computation or mathematical kernel)
 *    - Result comparison using compare_float_arrays()
 * 4. ✅ Update CMakeLists.txt to include your new test file
 * 5. ✅ Test your implementation with: ./tests/run-numa-tests.sh
 * 6. ✅ Verify all test combinations pass before considering complete
 * 
 * REFERENCE IMPLEMENTATIONS:
 * - See test-numa-mathematical-correctness.cpp for working MUL_MAT example
 * - Mathematical kernels available in ggml/src/ggml-cpu/ggml-cpu-impl.h
 * - Operation dispatch examples in ggml/src/ggml-cpu/ggml-numa-operation-dispatch.c
 * 
 * TESTING PRINCIPLES:
 * - Each operation should be tested across multiple dimensions
 * - Multiple thread strategies should be tested to verify coordinator behavior
 * - Mathematical equivalence should be exact (within floating-point tolerance)
 * - Tests should be deterministic and reproducible
 * - Comprehensive error reporting should help debug any failures
 */
