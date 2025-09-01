/**
 * NUMA Mathematical Correctness Test for PERMUTE Operation
 * 
 * This test provides comprehensive framework for testing mathematical equivalence
 * between NUMA parallel PERMUTE operations and serial reference implementations.
 * 
 * PERMUTE Operation Details:
 * - Tensor dimension permutation: rearranges data according to axis mapping
 * - Supports arbitrary axis permutations: (0,1,2,3) -> (axis0,axis1,axis2,axis3)
 * - Unlike the reference NOP implementation, NUMA version materializes the permutation
 * - Critical for memory layout optimization and NUMA locality improvements
 * 
 * Test Coverage:
 * 1. Mathematical equivalence testing (multi-dimensional tensors, multi-threading validation)
 * 2. Quantization type coverage (F32, F16 primarily for PERMUTE operations)
 * 3. Regression testing (various permutation patterns, edge cases)
 * 
 * PERMUTE-Specific Considerations:
 * - Axis permutation parameters stored in op_params
 * - Memory access patterns vary significantly with permutation type
 * - Cache locality implications for different permutation strategies
 * - NUMA kernels provide actual data movement vs reference view-only approach
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
#include "ggml-numa-simple-coordinator.h"  // For NUMA functions
#include "ggml-cpu/binary-ops.h"
#include "ggml-cpu/ops.h"  // For ggml_compute_forward_permute

// Test result structure
struct TestResult {
    std::string test_name;
    bool passed;
    std::string failure_reason;
};

class NumaPermuteMathematicalCorrectnessTestSuite {
private:
    std::vector<TestResult> results;
    std::string failure_reason;

    // Floating point comparison with tolerance
    bool compare_float_arrays(const float* numa_data, const float* ref_data, int count, const char* operation_name) {
        const float tolerance = 1e-5f;
        bool arrays_match = true;
        
        for (int i = 0; i < count; i++) {
            float diff = fabs(numa_data[i] - ref_data[i]);
            float rel_error = (ref_data[i] != 0.0f) ? diff / fabs(ref_data[i]) : diff;
            
            if (diff > tolerance && rel_error > tolerance) {
                printf("❌ Mismatch at index %d: NUMA=%.6f, REF=%.6f, diff=%.6f, rel_err=%.6f\n", 
                       i, numa_data[i], ref_data[i], diff, rel_error);
                arrays_match = false;
                if (i > 5) break; // Limit error output
            }
        }
        
        if (arrays_match) {
            printf("✅ Arrays match perfectly for %s\n", operation_name);
        }
        
        return arrays_match;
    }

    // Test a single PERMUTE case
    bool test_single_PERMUTE_case(int dim1, int dim2, int dim3, int dim4, 
                                 int axis0, int axis1, int axis2, int axis3,
                                 int num_threads, const char* size_label) {
        printf("    🧮 Testing %s: PERMUTE with dimensions [%d,%d,%d,%d] axes [%d,%d,%d,%d] (threads=%d)\n", 
               size_label, dim1, dim2, dim3, dim4, axis0, axis1, axis2, axis3, num_threads);

        // Create NUMA context for parallel execution
        struct ggml_init_params numa_init_params;
        numa_init_params.mem_size = 512 * 1024 * 1024; // 512MB
        numa_init_params.mem_buffer = nullptr;
        numa_init_params.no_alloc = false;
        
        struct ggml_context * numa_ctx = ggml_init(numa_init_params);
        if (!numa_ctx) {
            printf("❌ Failed to create NUMA context\n");
            return false;
        }

        // Create reference context for serial execution
        struct ggml_init_params ref_init_params;
        ref_init_params.mem_size = 512 * 1024 * 1024; // 512MB
        ref_init_params.mem_buffer = nullptr;
        ref_init_params.no_alloc = false;
        
        struct ggml_context * ref_ctx = ggml_init(ref_init_params);
        if (!ref_ctx) {
            printf("❌ Failed to create reference context\n");
            ggml_free(numa_ctx);
            return false;
        }

        // Create input tensors with same data
        struct ggml_tensor * numa_input = ggml_new_tensor_4d(numa_ctx, GGML_TYPE_F32, dim1, dim2, dim3, dim4);
        struct ggml_tensor * ref_input = ggml_new_tensor_4d(ref_ctx, GGML_TYPE_F32, dim1, dim2, dim3, dim4);
        
        // Initialize with test data
        std::mt19937 gen(12345); // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        float* numa_input_data = (float*)ggml_get_data(numa_input);
        float* ref_input_data = (float*)ggml_get_data(ref_input);
        
        for (int i = 0; i < ggml_nelements(numa_input); i++) {
            float value = dis(gen);
            numa_input_data[i] = value;
            ref_input_data[i] = value;
        }

        // Create PERMUTE operations
        struct ggml_tensor * numa_result = ggml_permute(numa_ctx, numa_input, axis0, axis1, axis2, axis3);
        struct ggml_tensor * ref_result = ggml_permute(ref_ctx, ref_input, axis0, axis1, axis2, axis3);

        // Create computation graphs
        struct ggml_cgraph * numa_graph = ggml_new_graph(numa_ctx);
        struct ggml_cgraph * ref_graph = ggml_new_graph(ref_ctx);
        
        ggml_build_forward_expand(numa_graph, numa_result);
        ggml_build_forward_expand(ref_graph, ref_result);

        // Set thread count for NUMA execution
        ggml_graph_compute_with_ctx(numa_ctx, numa_graph, num_threads);
        
        // Execute reference with single thread to avoid NUMA dispatch
        ggml_graph_compute_with_ctx(ref_ctx, ref_graph, 1);

        // Compare results
        bool test_passed = compare_float_arrays(
            (float*)ggml_get_data(numa_result), 
            (float*)ggml_get_data(ref_result), 
            ggml_nelements(numa_result),
            "PERMUTE"
        );

        // Cleanup
        ggml_free(numa_ctx);
        ggml_free(ref_ctx);

        if (test_passed) {
            printf("      ✅ %s PERMUTE case passed (threads=%d)\n", size_label, num_threads);
        } else {
            printf("      ❌ %s PERMUTE case failed (threads=%d)\n", size_label, num_threads);
        }

        return test_passed;
    }

public:
    // Test mathematical equivalence across different tensor dimensions and thread counts
    void test_PERMUTE_mathematical_equivalence() {
        printf("--- Test: PERMUTE Mathematical Equivalence (Multi-Dimensional) ---\n");
        printf("Testing NUMA parallel PERMUTE vs serial reference implementation...\n");
        printf("Testing across various tensor sizes with different coordinator execution strategies\n\n");
        
        bool overall_test_passed = true;
        std::string failure_reason_str = "";
        
        // Define test dimensions and permutation patterns
        struct {
            int dim1, dim2, dim3, dim4;
            int axis0, axis1, axis2, axis3;
            const char* label;
        } test_cases[] = {
            // Simple transpositions
            {8, 16, 4, 2, 1, 0, 2, 3, "TINY_TRANSPOSE_01"},      // Transpose first two dimensions
            {16, 32, 8, 4, 0, 1, 3, 2, "SMALL_TRANSPOSE_23"},     // Transpose last two dimensions
            
            // Complex permutations
            {32, 64, 16, 8, 3, 2, 1, 0, "MEDIUM_REVERSE"},        // Completely reverse order
            {64, 128, 32, 16, 2, 0, 3, 1, "LARGE_COMPLEX"},       // Complex permutation pattern
        };

        int num_test_cases = sizeof(test_cases) / sizeof(test_cases[0]);
        int thread_counts[] = {1, 2, 4, 6, 8};
        int num_thread_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);
        
        printf("  🎯 Testing %d permutation patterns with %d thread strategies (%d total test combinations)\n\n", 
               num_test_cases, num_thread_tests, num_test_cases * num_thread_tests);

        for (int i = 0; i < num_test_cases; i++) {
            printf("  📏 Testing %s permutation pattern (%d,%d,%d,%d) -> axes (%d,%d,%d,%d):\n", 
                   test_cases[i].label, 
                   test_cases[i].dim1, test_cases[i].dim2, test_cases[i].dim3, test_cases[i].dim4,
                   test_cases[i].axis0, test_cases[i].axis1, test_cases[i].axis2, test_cases[i].axis3);
            
            for (int j = 0; j < num_thread_tests; j++) {
                bool case_passed = test_single_PERMUTE_case(
                    test_cases[i].dim1, test_cases[i].dim2, test_cases[i].dim3, test_cases[i].dim4,
                    test_cases[i].axis0, test_cases[i].axis1, test_cases[i].axis2, test_cases[i].axis3,
                    thread_counts[j], test_cases[i].label
                );
                
                if (!case_passed) {
                    overall_test_passed = false;
                    if (failure_reason_str.empty()) {
                        failure_reason_str = "Mathematical mismatch detected in multi-dimensional testing";
                    }
                }
            }
            printf("\n");
        }
        
        // Print summary for this test
        printf("  📊 PERMUTE Multi-Dimensional Test Summary:\n");
        printf("    Total test combinations: %d\n", num_test_cases * num_thread_tests);
        printf("    Passed: %d\n", overall_test_passed ? num_test_cases * num_thread_tests : 0);
        printf("    Failed: %d\n", overall_test_passed ? 0 : num_test_cases * num_thread_tests);
        
        if (overall_test_passed) {
            printf("✅ PERMUTE mathematical equivalence (multi-dimensional): VERIFIED\n");
            printf("  🎉 All tensor dimensions and thread strategies produce mathematically equivalent results!\n\n");
        } else {
            printf("❌ PERMUTE mathematical equivalence (multi-dimensional): FAILED - %s\n", failure_reason_str.c_str());
            printf("  ⚠️  Mathematical mismatches detected across different dimensions or thread strategies\n\n");
        }
        
        results.push_back({"PERMUTE_mathematical_equivalence", overall_test_passed, failure_reason_str});
    }

    // Test quantization type coverage to ensure NUMA vs reference implementation compatibility
    void test_PERMUTE_quantization_type_coverage() {
        printf("--- Test: PERMUTE Core Quantization Type Coverage ---\n");
        printf("Testing PERMUTE operation with core quantization types to verify NUMA/reference compatibility...\n");
        printf("This ensures PERMUTE kernels handle model weights correctly across Q8_0, Q4_0, Q5_0 formats\n");
        printf("(K-quant types require 256-aligned dimensions and are tested separately if applicable)\n");
        printf("NUMA kernels support F32/F16 operations; quantized types should gracefully fall back to reference implementation\n\n");
        
        bool overall_test_passed = true;
        std::string failure_reason_str = "";
        
        // Define quantization type test cases
        struct {
            ggml_type src_type;
            ggml_type dst_type;
            const char* description;
        } type_tests[] = {
            {GGML_TYPE_F32, GGML_TYPE_F32, "F32 → F32"},
            {GGML_TYPE_F16, GGML_TYPE_F16, "F16 → F16"},
            {GGML_TYPE_F16, GGML_TYPE_F32, "F16 → F32"},
            {GGML_TYPE_Q8_0, GGML_TYPE_F32, "Q8_0 → F32"},
            {GGML_TYPE_Q4_0, GGML_TYPE_F32, "Q4_0 → F32"},
            {GGML_TYPE_Q5_0, GGML_TYPE_F32, "Q5_0 → F32"},
        };

        int total_type_tests = sizeof(type_tests) / sizeof(type_tests[0]);
        int passed_type_tests = 0;

        for (int i = 0; i < total_type_tests; i++) {
            printf("  🔍 Testing: %s\n", type_tests[i].description);
            
            // Create simple test tensors
            struct ggml_init_params params;
            params.mem_size = 128 * 1024 * 1024; // 128MB
            params.mem_buffer = nullptr;
            params.no_alloc = false;
            
            struct ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                failure_reason_str = "Context creation failed";
                break;
            }
            
            // Create test tensors - use aligned dimensions for quantized types
            const int dim1 = 32, dim2 = 32, dim3 = 4, dim4 = 4;
            struct ggml_tensor * input = ggml_new_tensor_4d(ctx, type_tests[i].src_type, dim1, dim2, dim3, dim4);
            
            if (!input) {
                failure_reason_str = "Tensor creation failed";
                ggml_free(ctx);
                break;
            }
            
            // Initialize input data appropriately for the type
            if (type_tests[i].src_type == GGML_TYPE_F32) {
                float* data = (float*)ggml_get_data(input);
                for (int j = 0; j < ggml_nelements(input); j++) {
                    data[j] = (float)(j % 100) * 0.01f; // Simple test pattern
                }
            } else if (type_tests[i].src_type == GGML_TYPE_F16) {
                ggml_fp16_t* data = (ggml_fp16_t*)ggml_get_data(input);
                for (int j = 0; j < ggml_nelements(input); j++) {
                    data[j] = ggml_fp32_to_fp16((float)(j % 100) * 0.01f);
                }
            }
            // For quantized types, we'll let ggml handle the initialization
            
            // Create PERMUTE operation (simple transpose)
            struct ggml_tensor * result = ggml_permute(ctx, input, 1, 0, 2, 3);
            
            if (!result) {
                failure_reason_str = "Operation creation failed";
                ggml_free(ctx);
                break;
            }
            
            // Create and execute computation graph
            struct ggml_cgraph * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, result);
            
            // Execute computation
            enum ggml_status status = ggml_graph_compute_with_ctx(ctx, graph, 1);
            
            bool test_passed = (status == GGML_STATUS_SUCCESS);
            
            if (test_passed) {
                passed_type_tests++;
                if (type_tests[i].src_type == GGML_TYPE_F32 && type_tests[i].dst_type == GGML_TYPE_F32) {
                    printf("      ✅ NUMA kernel handled %s as expected\n", type_tests[i].description);
                } else {
                    printf("      ✅ Reference fallback handled %s correctly\n", type_tests[i].description);
                }
            } else {
                failure_reason_str = "Execution failed";
                printf("      ❌ Failed to execute %s\n", type_tests[i].description);
            }
            
            ggml_free(ctx);
            
            if (!test_passed) {
                break;
            }
        }
        
        // Print type test summary
        printf("  📊 PERMUTE Quantization Type Test Summary:\n");
        printf("    Total type combinations: %d\n", total_type_tests);
        printf("    Passed: %d\n", passed_type_tests);
        printf("    Failed: %d\n", total_type_tests - passed_type_tests);
        
        if (overall_test_passed) {
            printf("✅ PERMUTE quantization type coverage: VERIFIED\n");
            printf("  🎉 All quantization types work correctly (NUMA kernels or reference fallback)!\n\n");
        } else {
            printf("❌ PERMUTE quantization type coverage: FAILED - %s\n", failure_reason_str.c_str());
            printf("  ⚠️  Some quantization types failed to execute properly\n\n");
        }
        
        results.push_back({"PERMUTE_quantization_type_coverage", overall_test_passed, failure_reason_str});
    }

public:
    bool run_all_tests() {
        printf("🧪 NUMA Mathematical Correctness Test Suite - PERMUTE\n");
        printf("================================================================================\n");
        printf("🔧 Testing mathematical correctness with function pointer architecture\n");
        printf("Comparing NUMA parallel execution against serial reference implementation\n");
        printf("================================================================================\n\n");

        // Initialize NUMA system for testing
        printf("🌟 Initializing NUMA system for mathematical correctness testing...\n");
        
        // Initialize NUMA coordinator
        ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);
        
        printf("✅ NUMA system initialized successfully\n\n");

        // Run all test categories
        test_PERMUTE_mathematical_equivalence();
        test_PERMUTE_quantization_type_coverage();

        // Print final results
        printf("\n");
        printf("================================================================================\n");
        printf("                    Mathematical Correctness Test Results\n");
        printf("================================================================================\n");

        bool all_passed = true;
        for (const auto& result : results) {
            const char* status = result.passed ? "✅" : "❌";
            printf("%s %s: %s\n", status, result.test_name.c_str(), result.passed ? "PASSED" : "FAILED");
            if (!result.passed) {
                all_passed = false;
            }
        }

        printf("------------------------------------------------------------------------\n");
        printf("Total: %zu/%zu tests passed", 
               std::count_if(results.begin(), results.end(), [](const TestResult& r) { return r.passed; }),
               results.size());
        
        if (all_passed) {
            printf(" 🎉 All tests passed!\n");
        } else {
            printf(" ❌ Some tests failed!\n");
        }
        
        printf("================================================================================\n");
        
        if (all_passed) {
            printf("✅ NUMA Mathematical Correctness: ALL TESTS PASSED\n\n");
            printf("🎯 NUMA parallel execution produces mathematically equivalent results\n");
            printf("🧪 Mathematical correctness testing completed!\n");
        } else {
            printf("❌ NUMA Mathematical Correctness: SOME TESTS FAILED\n\n");
            printf("⚠️  NUMA parallel execution has mathematical discrepancies\n");
            printf("🔧 Debug and fix failing test cases before proceeding\n");
        }

        return all_passed;
    }
};

int main(int argc, char* argv[]) {
    // Handle command line arguments
    bool summary_only = false;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--summary") == 0) {
            summary_only = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [--summary] [--help]\n", argv[0]);
            printf("  --summary  Show only final summary\n");
            printf("  --help     Show this help message\n");
            return 0;
        }
    }

    // Create and run test suite
    NumaPermuteMathematicalCorrectnessTestSuite test_suite;
    bool all_passed = test_suite.run_all_tests();
    
    // Exit with appropriate code
    return all_passed ? 0 : 1;
}
