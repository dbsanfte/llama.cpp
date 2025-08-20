/**
 * NUMA Mathematical Correctness Test Template
 * 
 * This template provides a comprehensive framework for testing mathematical equivalence
 * between NUMA parallel operations and serial reference implementations.
 * 
 * USAGE INSTRUCTIONS:
 * 1. Copy this template to create a new test file (e.g., test-numa-mathematical-correctness-OPERATION.cpp)
 * 2. Replace TEMPLATE_OPERATION with your operation name (e.g., MUL_MAT, ADD, etc.)
 * 3. Implement test_single_OPERATION_case() for your specific operation
 * 4. Define appropriate test dimensions and thread strategies for your operation
 * 5. Update the operation-specific logic in test_OPERATION_mathematical_equivalence()
 * 6. Add your new test file to CMakeLists.txt in the tests directory
 * 
 * KEY DESIGN PRINCIPLES:
 * - Multi-dimensional testing across various matrix/tensor sizes
 * - Multiple thread strategies to test coordinator execution
 * - Direct comparison between NUMA parallel and serial reference implementations
 * - Comprehensive error reporting with detailed mismatch information
 * - Modular design for easy extension to new operations
 * 
 * TEMPLATE STRUCTURE:
 * - TestResult: Simple structure for tracking test outcomes
 * - NumaMathematicalCorrectnessTestSuite: Main test class with helper methods
 * - test_single_OPERATION_case(): Tests one specific case with given parameters
 * - test_OPERATION_mathematical_equivalence(): Runs comprehensive multi-dimensional testing
 * - run_all_tests(): Entry point that orchestrates all tests and provides summary
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
    
    // Test a single ADD case with specific dimensions and thread count
    bool test_single_ADD_case(int dim1, int dim2, int dim3, int num_threads, const char* size_label) {
        printf("    🧮 Testing %s: ADD with dimensions [%d,%d,%d] (threads=%d)\n", 
               size_label, dim1, dim2, dim3, num_threads);
        
        // Create test context with sufficient memory for larger tensors
        struct ggml_init_params params;
        params.mem_size = 0;
        params.mem_buffer = nullptr;
        params.no_alloc = false;
        params.mem_size = std::max((size_t)(512 * 1024 * 1024), (size_t)(dim1 * dim2 * dim3) * sizeof(float) * 8); // Scale memory with tensor size
        params.mem_buffer = nullptr;
        params.no_alloc = false;
        
        struct ggml_context* test_ctx = ggml_init(params);
        if (!test_ctx) {
            printf("      ❌ Failed to create test context for %s\n", size_label);
            return false;
        }
        
        bool case_passed = false;
        
        // Create input tensors for ADD operation (binary operation: A + B = C)
        struct ggml_tensor* input_a = ggml_new_tensor_3d(test_ctx, GGML_TYPE_F32, dim1, dim2, dim3);
        struct ggml_tensor* input_b = ggml_new_tensor_3d(test_ctx, GGML_TYPE_F32, dim1, dim2, dim3);
        
        if (!input_a || !input_b) {
            printf("      ❌ Failed to create input tensors for %s\n", size_label);
            ggml_free(test_ctx);
            return false;
        }
        
        // Fill tensors with deterministic test data
        float* a_data = (float*)ggml_get_data(input_a);
        float* b_data = (float*)ggml_get_data(input_b);
        int total_elements = ggml_nelements(input_a);
        
        for (int i = 0; i < total_elements; i++) {
            a_data[i] = 0.1f + (i % 37) * 0.01f; // Deterministic test pattern
            b_data[i] = 0.05f + (i % 23) * 0.015f; // Different pattern for B
        }
        
        // Create ADD operation
        struct ggml_tensor* numa_result = ggml_add(test_ctx, input_a, input_b);
        
        if (!numa_result) {
            printf("      ❌ Failed to create ADD operation for %s\n", size_label);
            ggml_free(test_ctx);
            return false;
        }
        
        // Execute via NUMA intercept
        struct ggml_compute_params numa_params;
        numa_params.ith = 0;
        numa_params.nth = num_threads;
        numa_params.wsize = 0;
        numa_params.wdata = nullptr;
        numa_params.threadpool = nullptr;
        enum ggml_status dispatch_result = ggml_numa_intercept_operation(numa_result, &numa_params);
        
        if (dispatch_result != GGML_STATUS_SUCCESS) {
            printf("      ❌ NUMA dispatch failed for %s: %d\n", size_label, dispatch_result);
            ggml_free(test_ctx);
            return false;
        }
        
        // Create reference computation using serial execution
        struct ggml_tensor* ref_result = ggml_add(test_ctx, input_a, input_b);
        struct ggml_compute_params ref_params;
        ref_params.ith = 0;
        ref_params.nth = 1;
        ref_params.wsize = 0;
        ref_params.wdata = nullptr;
        ref_params.threadpool = nullptr;
        ggml_compute_forward_add_non_quantized(&ref_params, ref_result);
        
        // Compare results
        float* numa_data = (float*)ggml_get_data(numa_result);
        float* ref_data = (float*)ggml_get_data(ref_result);
        case_passed = compare_float_arrays(numa_data, ref_data, total_elements, "ADD");
        
        if (case_passed) {
            printf("      ✅ %s ADD case passed (threads=%d)\n", size_label, num_threads);
        } else {
            printf("      ❌ %s ADD case failed (threads=%d)\n", size_label, num_threads);
        }
        
        ggml_free(test_ctx);
        return case_passed;
    }

    void test_ADD_mathematical_equivalence() {
        printf("--- Test: ADD Mathematical Equivalence (Multi-Dimensional) ---\n");
        printf("Testing NUMA parallel ADD vs serial reference implementation...\n");
        printf("Testing across various tensor sizes with different coordinator execution strategies\n\n");
        
        bool overall_test_passed = true;
        const char* failure_reason = nullptr;
        
        // Define test dimensions appropriate for ADD operation (element-wise binary operation)
        struct {
            int dim1, dim2, dim3;
            const char* label;
        } test_cases[] = {
            {8, 8, 4, "TINY"},           // Small tensors for basic verification
            {32, 32, 16, "SMALL"},       // Medium tensors
            {128, 64, 32, "MEDIUM"},     // Larger tensors
            {256, 128, 64, "LARGE"}      // Large tensors for stress testing
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
                
                bool case_passed = test_single_ADD_case(
                    test_cases[case_idx].dim1, 
                    test_cases[case_idx].dim2, 
                    test_cases[case_idx].dim3, 
                    num_threads,
                    test_cases[case_idx].label
                );
                
                total_tests++;
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
        printf("  📊 ADD Multi-Dimensional Test Summary:\n");
        printf("    Total test combinations: %d\n", total_tests);
        printf("    Passed: %d\n", passed_tests);
        printf("    Failed: %d\n", total_tests - passed_tests);
        
        if (overall_test_passed) {
            printf("✅ ADD mathematical equivalence (multi-dimensional): VERIFIED\n");
            printf("  🎉 All tensor dimensions and thread strategies produce mathematically equivalent results!\n\n");
        } else {
            printf("❌ ADD mathematical equivalence (multi-dimensional): FAILED - %s\n", failure_reason);
            printf("  ⚠️  Mathematical mismatches detected across different dimensions or thread strategies\n\n");
        }
        
        results.push_back({"ADD_mathematical_equivalence", overall_test_passed, failure_reason ? failure_reason : ""});
    }

public:
    bool run_all_tests() {
        printf("🧪 NUMA Mathematical Correctness Test Suite - ADD\n");
        printf("================================================================================\n");
        printf("🔧 Testing mathematical correctness with function pointer architecture\n");
        printf("Comparing NUMA parallel execution against serial reference implementation\n");
        printf("================================================================================\n\n");
        
        // Run mathematical correctness tests
        test_ADD_mathematical_equivalence();
        
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
int main(int argc, char** argv) {
    // Initialize NUMA with MIRROR strategy for testing
    ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);
    
    // Check for --summary-only flag
    bool summary_only = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--summary-only") == 0) {
            summary_only = true;
            break;
        }
    }
    
    // Redirect stdout to /dev/null if summary-only mode (but keep final results)
    FILE* original_stdout = nullptr;
    if (summary_only) {
        original_stdout = stdout;
        stdout = fopen("/dev/null", "w");
        if (!stdout) {
            stdout = original_stdout;
            summary_only = false; // Fall back if redirection fails
        }
    }
    
    printf("🌟 Initializing NUMA system for mathematical correctness testing...\n");
    
    // Initialize the NUMA coordinator system
    struct ggml_numa_coordinator_manager* mgr = ggml_numa_coordinator_manager_get_global(8); // Enable NUMA MIRROR mode
    if (!mgr) {
        if (summary_only) {
            fclose(stdout);
            stdout = original_stdout;
        }
        printf("❌ Failed to initialize NUMA coordinator manager\n");
        return 1;
    }
    
    printf("✅ NUMA system initialized successfully\n\n");
    
    NumaMathematicalCorrectnessTestSuite test_suite;
    bool all_passed = test_suite.run_all_tests();
    
    // Restore stdout for final results
    if (summary_only) {
        fclose(stdout);
        stdout = original_stdout;
    }
    
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
