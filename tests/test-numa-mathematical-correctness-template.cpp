/**
 * NUMA Mathematical Correctness Test Template
 * 
 * This template provides a comprehensive framework for testing mathematical equivalence
 * between NUMA parallel operations and serial reference implementations.
 * 
 * USAGE INSTRUCTIONS:
 * 1. Copy this template to create a new test file (e.g., test-numa-mathematical-correctness-OPERATION.cpp)
 * 2. Replace TEMPLATE_OPERATION with your operation name (e.g., MUL_MAT, RMS_NORM, etc.)
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
#include "ggml-numa-executor.h"
#include "ggml-numa-simple-coordinator.h"  // For NUMA functions
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
    
    // Test a single TEMPLATE_OPERATION case with specific dimensions and thread count
    // TODO: Implement this method for your specific operation
    bool test_single_TEMPLATE_OPERATION_case(int dim1, int dim2, int dim3, int num_threads, const char* size_label) {
        printf("    🧮 Testing %s: TEMPLATE_OPERATION with dimensions [%d,%d,%d] (threads=%d)\n", 
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
        
        // TODO: Create input tensors appropriate for your operation
        // Examples for different operation types:
        
        // For binary operations (like ADD, MUL, SUB):
        // struct ggml_tensor* input_a = ggml_new_tensor_3d(test_ctx, GGML_TYPE_F32, dim1, dim2, dim3);
        // struct ggml_tensor* input_b = ggml_new_tensor_3d(test_ctx, GGML_TYPE_F32, dim1, dim2, dim3);
        
        // For unary operations (like RMS_NORM, GELU):
        // struct ggml_tensor* input_a = ggml_new_tensor_3d(test_ctx, GGML_TYPE_F32, dim1, dim2, dim3);
        
        // For matrix operations (like MUL_MAT):
        // struct ggml_tensor* input_a = ggml_new_tensor_2d(test_ctx, GGML_TYPE_F32, dim1, dim2);
        // struct ggml_tensor* input_b = ggml_new_tensor_2d(test_ctx, GGML_TYPE_F32, dim2, dim3);
        
        // PLACEHOLDER: Create example tensors (replace with your operation's requirements)
        struct ggml_tensor* input_a = ggml_new_tensor_3d(test_ctx, GGML_TYPE_F32, dim1, dim2, dim3);
        struct ggml_tensor* input_b = ggml_new_tensor_3d(test_ctx, GGML_TYPE_F32, dim1, dim2, dim3);
        
        if (!input_a || !input_b) {
            printf("      ❌ Failed to create input tensors for %s\n", size_label);
            ggml_free(test_ctx);
            return false;
        }
        
        // TODO: Fill tensors with deterministic test data appropriate for your operation
        float* a_data = (float*)ggml_get_data(input_a);
        float* b_data = (float*)ggml_get_data(input_b);
        int total_elements = ggml_nelements(input_a);
        
        for (int i = 0; i < total_elements; i++) {
            a_data[i] = 0.1f + (i % 37) * 0.01f; // Deterministic test pattern
            // TODO: For unary operations, you might not need b_data
            if (input_b) {
                b_data[i] = 0.05f + (i % 23) * 0.015f; // Different pattern for B
            }
        }
        
        // CRITICAL: Re-initialize NUMA mirroring after filling data to ensure all NUMA copies have correct data
        // The initial NUMA mirroring during tensor creation copied uninitialized memory (zeros)
        // We need to re-mirror with the actual test data we just wrote
        tensor_set_data_numa_mirror(input_a, a_data);
        if (input_b) {
            tensor_set_data_numa_mirror(input_b, b_data);
        }
        
        // TODO: Create TEMPLATE_OPERATION operation
        // Examples for different operations:
        // struct ggml_tensor* numa_result = ggml_add(test_ctx, input_a, input_b);       // Binary
        // struct ggml_tensor* numa_result = ggml_rms_norm(test_ctx, input_a, 1e-5f);   // Unary with params
        // struct ggml_tensor* numa_result = ggml_mul_mat(test_ctx, input_a, input_b);  // Matrix multiplication
        
        // PLACEHOLDER: Create example operation (replace with your operation)
        struct ggml_tensor* numa_result = ggml_add(test_ctx, input_a, input_b);
        
        if (!numa_result) {
            printf("      ❌ Failed to create TEMPLATE_OPERATION operation for %s\n", size_label);
            ggml_free(test_ctx);
            return false;
        }
        
        // Execute via NUMA executor
        struct ggml_compute_params numa_params;
        numa_params.ith = 0;
        numa_params.nth = num_threads;
        numa_params.wsize = 0;
        numa_params.wdata = nullptr;
        numa_params.threadpool = nullptr;
        
        // Create minimal compute plan for single tensor execution
        struct ggml_cplan cplan = {};
        cplan.work_size = 0;
        cplan.work_data = nullptr;
        cplan.n_threads = num_threads;
        cplan.threadpool = nullptr;
        cplan.abort_callback = nullptr;
        cplan.abort_callback_data = nullptr;
        
        // Execute with new executor architecture
        enum ggml_status dispatch_result = ggml_numa_executor_execute_tensor(numa_result, &cplan);
        
        if (dispatch_result != GGML_STATUS_SUCCESS) {
            printf("      ❌ NUMA dispatch failed for %s: %d\n", size_label, dispatch_result);
            ggml_free(test_ctx);
            return false;
        }
        
        // TODO: Create reference computation using serial execution
        // This should use the appropriate reference implementation for your operation
        // Examples:
        
        // For operations with existing reference functions:
        // struct ggml_tensor* ref_result = ggml_add(test_ctx, input_a, input_b);
        // struct ggml_compute_params ref_params = {0, 1, 0, nullptr, nullptr};
        // ggml_compute_forward_add_non_quantized(&ref_params, ref_result);
        
        // For operations needing direct kernel calls:
        // ggml_compute_forward_rms_norm_f32(&ref_params, ref_result);
        
        // PLACEHOLDER: Create reference computation (replace with your operation's reference)
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
        case_passed = compare_float_arrays(numa_data, ref_data, total_elements, "TEMPLATE_OPERATION");
        
        if (case_passed) {
            printf("      ✅ %s TEMPLATE_OPERATION case passed (threads=%d)\n", size_label, num_threads);
        } else {
            printf("      ❌ %s TEMPLATE_OPERATION case failed (threads=%d)\n", size_label, num_threads);
        }
        
        ggml_free(test_ctx);
        return case_passed;
    }

    void test_TEMPLATE_OPERATION_mathematical_equivalence() {
        printf("--- Test: TEMPLATE_OPERATION Mathematical Equivalence (Multi-Dimensional) ---\n");
        printf("Testing NUMA parallel TEMPLATE_OPERATION vs serial reference implementation...\n");
        printf("Testing across various tensor sizes with different coordinator execution strategies\n\n");
        
        bool overall_test_passed = true;
        const char* failure_reason = nullptr;
        
        // TODO: Define test dimensions appropriate for your operation
        // Example for matrix operations:
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
                
                bool case_passed = test_single_TEMPLATE_OPERATION_case(
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
        printf("  📊 TEMPLATE_OPERATION Multi-Dimensional Test Summary:\n");
        printf("    Total test combinations: %d\n", total_tests);
        printf("    Passed: %d\n", passed_tests);
        printf("    Failed: %d\n", total_tests - passed_tests);
        
        if (overall_test_passed) {
            printf("✅ TEMPLATE_OPERATION mathematical equivalence (multi-dimensional): VERIFIED\n");
            printf("  🎉 All tensor dimensions and thread strategies produce mathematically equivalent results!\n\n");
        } else {
            printf("❌ TEMPLATE_OPERATION mathematical equivalence (multi-dimensional): FAILED - %s\n", failure_reason);
            printf("  ⚠️  Mathematical mismatches detected across different dimensions or thread strategies\n\n");
        }
        
        results.push_back({"TEMPLATE_OPERATION_mathematical_equivalence", overall_test_passed, failure_reason});
    }

public:
    bool run_all_tests() {
        printf("🧪 NUMA Mathematical Correctness Test Suite - TEMPLATE_OPERATION\n");
        printf("================================================================================\n");
        printf("🔧 Testing mathematical correctness with function pointer architecture\n");
        printf("Comparing NUMA parallel execution against serial reference implementation\n");
        printf("================================================================================\n\n");
        
        // Run mathematical correctness tests
        test_TEMPLATE_OPERATION_mathematical_equivalence();
        
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
    
    // Initialize NUMA system before tests with auto-detection
    if (ggml_numa_is_available()) {
        printf("🔧 Initializing NUMA mirroring mode...\n");
        
        // Initialize with auto-detection pattern used in ADD test
        int numa_initialization_result = ggml_numa_auto_init_mirror();
        
        if (numa_initialization_result != 0) {
            printf("⚠️  NUMA auto-initialization failed with code: %d, proceeding with default configuration\n", numa_initialization_result);
            // Continue with tests as NUMA will fall back to CPU implementation
        } else {
            printf("✅ NUMA mirroring mode initialized successfully\n");
        }
    } else {
        printf("⚠️  NUMA not available on this system, tests will use CPU-only execution\n");
    }
    
    printf("✅ NUMA system initialized successfully\n\n");
    
    NumaMathematicalCorrectnessTestSuite test_suite;
    bool all_passed = test_suite.run_all_tests();
    
    return all_passed ? 0 : 1;
}

/**
 * IMPLEMENTATION CHECKLIST:
 * 
 * When adapting this template for a new operation:
 * 
 * 1. ✅ Replace all instances of "TEMPLATE_OPERATION" with your operation name (e.g., "GLU", "RMS_NORM")
 * 2. ✅ Update test dimensions in test_cases[] array to match your operation's needs:
 *    - Binary ops (ADD, MUL): Use matching dimensions for both inputs
 *    - Unary ops (RMS_NORM, GELU): Single input tensor dimensions
 *    - Matrix ops (MUL_MAT): Consider [M, K] x [K, N] → [M, N] patterns
 * 3. ✅ Implement test_single_OPERATION_case() with:
 *    - Appropriate tensor creation (ggml_new_tensor_*) for your operation
 *    - Deterministic test data generation (avoid random data for reproducibility)
 *    - NUMA mirroring setup using tensor_set_data_numa_mirror() after data filling
 *    - NUMA operation execution via ggml_numa_executor_execute_tensor()
 *    - Reference implementation using appropriate serial computation
 *    - Result comparison using compare_float_arrays()
 * 4. ✅ Update CMakeLists.txt to include your new test file:
 *    - Add executable definition
 *    - Link against required libraries (ggml-cpu, common, etc.)
 *    - Include in test target dependencies
 * 5. ✅ Add your test to tests/run-numa-tests.sh script
 * 6. ✅ Test your implementation thoroughly:
 *    - Run individual test: cmake --build build --target test-numa-mathematical-correctness-YOUR_OPERATION
 *    - Run full test suite: ./tests/run-numa-tests.sh
 *    - Verify all test combinations pass before considering complete
 * 
 * REFERENCE IMPLEMENTATIONS:
 * - See test-numa-mathematical-correctness-add.cpp for binary operation example
 * - See test-numa-mathematical-correctness-rms-norm.cpp for unary operation example
 * - Mathematical kernels available in ggml/src/ggml-cpu/ggml-cpu.c
 * - NUMA kernels available in ggml/src/ggml-cpu/numa-kernels/
 * 
 * TESTING PATTERNS:
 * - TINY: Single-node, single-thread execution
 * - SMALL: Single-node, multi-thread execution  
 * - MEDIUM: Multi-node, single-thread per node
 * - LARGE: Multi-node, multi-thread execution
 * - HUGE: Full NUMA parallelization with optimal chunking
 */
 * - Operation dispatch examples in ggml/src/ggml-cpu/ggml-numa-operation-dispatch.c
 * 
 * TESTING PRINCIPLES:
 * - Each operation should be tested across multiple dimensions
 * - Multiple thread strategies should be tested to verify coordinator behavior
 * - Mathematical equivalence should be exact (within floating-point tolerance)
 * - Tests should be deterministic and reproducible
 * - Comprehensive error reporting should help debug any failures
 */
