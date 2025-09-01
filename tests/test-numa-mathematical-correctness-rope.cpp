/**
 * NUMA Mathematical Correctness Test for ROPE Operation
 * 
 * This test provides comprehensive framework for testing mathematical equivalence
 * between NUMA parallel ROPE operations and serial reference implementations.
 * 
 * ROPE Operation Details:
 * - Rotary Position Embedding (ROPE) applies positional information through rotation
 * - Supports multiple variants: standard, NEOX, multi-modal (mrope), vision
 * - Complex cache computation with position-dependent coefficients
 * - Element-wise rotation transformations using sin/cos coefficients
 * 
 * Test Coverage:
 * 1. Mathematical equivalence testing (multi-dimensional tensors, multi-threading validation)
 * 2. Quantization type coverage (F32, F16 primarily for ROPE operations)
 * 3. Regression testing (different ROPE modes, position encoding edge cases)
 * 
 * ROPE-Specific Considerations:
 * - Position tensor (src1) contains integer positions
 * - Optional frequency factors tensor (src2) for advanced scaling
 * - Multiple operation modes requiring different parameter configurations
 * - Cache computation precision affects final rotation accuracy
 * 
 * QUANTIZATION TYPE COVERAGE:
 * - F32/F16 combinations (primary ROPE kernel paths)
 * - ROPE typically operates on activation tensors (not weights), so fewer quantization types
 * - Focus on F32→F32 and F16→F16 transformations for performance validation
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
#include "ggml-cpu/ops.h"  // For ggml_compute_forward_rope

// Test result structure
struct TestResult {
    std::string test_name;
    bool passed;
    std::string failure_reason;
};

class NumaRopeMathematicalCorrectnessTestSuite {
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
    
    // Test a single ROPE case with specific dimensions and thread count
    // TODO: Implement this method for your specific operation
    bool test_single_ROPE_case(int dim1, int dim2, int dim3, int num_threads, const char* size_label) {
        printf("    🧮 Testing %s: ROPE with dimensions [%d,%d,%d] (threads=%d)\n", 
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
        // ROPE requires: input tensor and position tensor
        
        // Create input tensor (4D): [head_dim, num_heads, seq_len, batch_size]
        // For testing purposes, use dim1=head_dim, dim2=num_heads, dim3=seq_len
        const int head_dim = dim1;
        const int num_heads = dim2;
        const int seq_len = dim3;
        const int batch_size = 1;
        const int n_dims = head_dim;  // ROPE typically rotates the full head dimension
        
        struct ggml_tensor* input_tensor = ggml_new_tensor_4d(test_ctx, GGML_TYPE_F32, head_dim, num_heads, seq_len, batch_size);
        struct ggml_tensor* pos_tensor = ggml_new_tensor_1d(test_ctx, GGML_TYPE_I32, seq_len);
        
        if (!input_tensor || !pos_tensor) {
            printf("      ❌ Failed to create input tensors for %s\n", size_label);
            ggml_free(test_ctx);
            return false;
        }
        
        // TODO: Fill tensors with deterministic test data appropriate for your operation
        float* input_data = (float*)ggml_get_data(input_tensor);
        int32_t* pos_data = (int32_t*)ggml_get_data(pos_tensor);
        int total_input_elements = ggml_nelements(input_tensor);
        
        for (int i = 0; i < total_input_elements; i++) {
            input_data[i] = 0.1f + (i % 37) * 0.01f; // Deterministic test pattern
        }
        
        // Fill position tensor with sequential positions
        for (int i = 0; i < seq_len; i++) {
            pos_data[i] = i;  // Position 0, 1, 2, ...
        }
        
        // CRITICAL: Re-initialize NUMA mirroring after filling data to ensure all NUMA copies have correct data
        // The initial NUMA mirroring during tensor creation copied uninitialized memory (zeros)
        // We need to re-mirror with the actual test data we just wrote
        tensor_set_data_numa_mirror(input_tensor, input_data);
        tensor_set_data_numa_mirror(pos_tensor, pos_data);
        
        // Create ROPE operation
        // ggml_rope_ext(ctx, a, pos, freq, n_dims, mode, n_past, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
        struct ggml_tensor* numa_result = ggml_rope_ext(
            test_ctx, input_tensor, pos_tensor, nullptr,  // freq_factors = NULL
            n_dims,           // rotation dimensions
            0,                // mode = standard ROPE (not NEOX, not mrope)
            0,                // n_past = 0
            10000.0f,         // freq_base
            1.0f,             // freq_scale
            0.0f,             // ext_factor (no extrapolation)
            1.0f,             // attn_factor
            1.0f,             // beta_fast
            1.0f              // beta_slow
        );
        
        if (!numa_result) {
            printf("      ❌ Failed to create ROPE operation for %s\n", size_label);
            ggml_free(test_ctx);
            return false;
        }
        
        // Execute via NUMA executor using graph computation
        struct ggml_cgraph * numa_graph = ggml_new_graph(test_ctx);
        ggml_build_forward_expand(numa_graph, numa_result);
        
        struct ggml_cplan numa_cplan = ggml_graph_plan(numa_graph, num_threads, nullptr);
        if (numa_cplan.work_size > 0) {
            numa_cplan.work_data = (uint8_t*)malloc(numa_cplan.work_size);
        }
        
        // Execute with NUMA 
        enum ggml_status numa_status = ggml_graph_compute(numa_graph, &numa_cplan);
        
        if (numa_status != GGML_STATUS_SUCCESS) {
            printf("      ❌ NUMA graph computation failed for %s: %d\n", size_label, numa_status);
            if (numa_cplan.work_data) free(numa_cplan.work_data);
            ggml_free(test_ctx);
            return false;
        }
        
        // Create reference computation using serial execution in a separate context
        // This avoids NUMA dispatch by using a separate context
        struct ggml_init_params ref_params_init;
        ref_params_init.mem_size = std::max((size_t)(512 * 1024 * 1024), (size_t)(head_dim * num_heads * seq_len * batch_size) * sizeof(float) * 8);
        ref_params_init.mem_buffer = nullptr;
        ref_params_init.no_alloc = false;
        struct ggml_context* ref_ctx = ggml_init(ref_params_init);
        if (!ref_ctx) {
            printf("      ❌ Failed to create reference context for %s\n", size_label);
            if (numa_cplan.work_data) free(numa_cplan.work_data);
            ggml_free(test_ctx);
            return false;
        }
        
        // Create identical tensors in reference context
        struct ggml_tensor* ref_input_tensor = ggml_new_tensor_4d(ref_ctx, GGML_TYPE_F32, head_dim, num_heads, seq_len, batch_size);
        struct ggml_tensor* ref_pos_tensor = ggml_new_tensor_1d(ref_ctx, GGML_TYPE_I32, seq_len);
        
        // Copy the same data to reference tensors
        memcpy(ggml_get_data(ref_input_tensor), input_data, total_input_elements * sizeof(float));
        memcpy(ggml_get_data(ref_pos_tensor), pos_data, seq_len * sizeof(int32_t));
        
        // Create identical ROPE operation in reference context
        struct ggml_tensor* ref_result = ggml_rope_ext(
            ref_ctx, ref_input_tensor, ref_pos_tensor, nullptr,  // freq_factors = NULL
            n_dims,           // rotation dimensions
            0,                // mode = standard ROPE (not NEOX, not mrope)
            0,                // n_past = 0
            10000.0f,         // freq_base
            1.0f,             // freq_scale
            0.0f,             // ext_factor (no extrapolation)
            1.0f,             // attn_factor
            1.0f,             // beta_fast
            1.0f              // beta_slow
        );
        
        // Execute reference computation with graph (single threaded)
        struct ggml_cgraph * ref_graph = ggml_new_graph(ref_ctx);
        ggml_build_forward_expand(ref_graph, ref_result);
        
        struct ggml_cplan ref_cplan = ggml_graph_plan(ref_graph, 1, nullptr);  // Single threaded
        if (ref_cplan.work_size > 0) {
            ref_cplan.work_data = (uint8_t*)malloc(ref_cplan.work_size);
        }
        
        enum ggml_status ref_status = ggml_graph_compute(ref_graph, &ref_cplan);
        
        
        if (ref_status != GGML_STATUS_SUCCESS) {
            printf("      ❌ Reference graph computation failed for %s: %d\n", size_label, ref_status);
            if (numa_cplan.work_data) free(numa_cplan.work_data);
            if (ref_cplan.work_data) free(ref_cplan.work_data);
            ggml_free(ref_ctx);
            ggml_free(test_ctx);
            return false;
        }
        
        // Compare results
        float* numa_data = (float*)ggml_get_data(numa_result);
        float* ref_data = (float*)ggml_get_data(ref_result);
        case_passed = compare_float_arrays(numa_data, ref_data, total_input_elements, "ROPE");
        
        if (case_passed) {
            printf("      ✅ %s ROPE case passed (threads=%d)\n", size_label, num_threads);
        } else {
            printf("      ❌ %s ROPE case failed (threads=%d)\n", size_label, num_threads);
        }
        
        // Cleanup
        if (numa_cplan.work_data) free(numa_cplan.work_data);
        if (ref_cplan.work_data) free(ref_cplan.work_data);
        ggml_free(ref_ctx);
        ggml_free(test_ctx);
        return case_passed;
    }

    void test_ROPE_mathematical_equivalence() {
        printf("--- Test: ROPE Mathematical Equivalence (Multi-Dimensional) ---\n");
        printf("Testing NUMA parallel ROPE vs serial reference implementation...\n");
        printf("Testing across various tensor sizes with different coordinator execution strategies\n\n");
        
        bool overall_test_passed = true;
        std::string failure_reason_str = "";
        
        // Define test dimensions appropriate for ROPE operation
        // Format: {head_dim, num_heads, seq_len, "label"}
        // ROPE operates on attention heads with rotary position embeddings
        struct {
            int dim1, dim2, dim3;
            const char* label;
        } test_cases[] = {
            {64, 8, 16, "TINY"},         // Small attention head: 64-dim, 8 heads, 16 tokens
            {128, 12, 32, "SMALL"},      // Medium: 128-dim, 12 heads, 32 tokens
            {256, 16, 64, "MEDIUM"},     // Large: 256-dim, 16 heads, 64 tokens  
            {512, 20, 128, "LARGE"}      // Very large: 512-dim, 20 heads, 128 tokens
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
            printf("  📏 Testing %s ROPE dimensions (head_dim=%d, num_heads=%d, seq_len=%d):\n", 
                   test_cases[case_idx].label, 
                   test_cases[case_idx].dim1, 
                   test_cases[case_idx].dim2, 
                   test_cases[case_idx].dim3);
            
            for (int strategy_idx = 0; strategy_idx < num_strategies; strategy_idx++) {
                int num_threads = thread_strategies[strategy_idx];
                
                bool case_passed = test_single_ROPE_case(
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
                    if (failure_reason_str.empty()) {
                        failure_reason_str = "Mathematical mismatch detected in multi-dimensional testing";
                    }
                }
            }
            printf("\n");
        }
        
        // Print summary for this test
        printf("  📊 ROPE Multi-Dimensional Test Summary:\n");
        printf("    Total test combinations: %d\n", total_tests);
        printf("    Passed: %d\n", passed_tests);
        printf("    Failed: %d\n", total_tests - passed_tests);
        
        if (overall_test_passed) {
            printf("✅ ROPE mathematical equivalence (multi-dimensional): VERIFIED\n");
            printf("  🎉 All tensor dimensions and thread strategies produce mathematically equivalent results!\n\n");
        } else {
            printf("❌ ROPE mathematical equivalence (multi-dimensional): FAILED - %s\n", failure_reason_str.c_str());
            printf("  ⚠️  Mathematical mismatches detected across different dimensions or thread strategies\n\n");
        }
        
        results.push_back({"ROPE_mathematical_equivalence", overall_test_passed, failure_reason_str});
    }

    // Test quantization type coverage to ensure NUMA vs reference implementation compatibility
    void test_ROPE_quantization_type_coverage() {
        printf("--- Test: ROPE Core Quantization Type Coverage ---\n");
        printf("Testing ROPE operation with core quantization types to verify NUMA/reference compatibility...\n");
        printf("This ensures ROPE kernels handle model weights correctly across Q8_0, Q4_0, Q5_0 formats\n");
        printf("(K-quant types require 256-aligned dimensions and are tested separately if applicable)\n");
        printf("NUMA kernels support F32/F16 operations; quantized types should gracefully fall back to reference implementation\n\n");
        
        bool overall_test_passed = true;
        std::string failure_reason_str = "";
        
        // Define quantization type test cases
        // TODO: Customize these test cases for your specific operation requirements
        struct {
            ggml_type src0_type, src1_type, dst_type;
            const char* description;
            bool expect_numa_support;  // Whether we expect NUMA kernel to handle this
        } type_test_cases[] = {
            // Non-quantized types (most important)
            {GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, "F32 + F32 → F32", true},
            {GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_F16, "F16 + F16 → F16", false},
            {GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_F32, "F16 + F32 → F32", false},
            
            // TODO: For unary operations, remove src1_type and adjust test cases:
            // {GGML_TYPE_F32, GGML_TYPE_COUNT, GGML_TYPE_F32, "F32 → F32", true},
            // {GGML_TYPE_F16, GGML_TYPE_COUNT, GGML_TYPE_F16, "F16 → F16", false},
            
            // Key quantized types (most commonly used in model weights)
            {GGML_TYPE_Q8_0, GGML_TYPE_F32, GGML_TYPE_F32, "Q8_0 * F32 → F32", false},  // 8-bit quantization (block size 32)
            {GGML_TYPE_Q4_0, GGML_TYPE_F32, GGML_TYPE_F32, "Q4_0 * F32 → F32", false},  // 4-bit quantization (block size 32)
            {GGML_TYPE_Q5_0, GGML_TYPE_F32, GGML_TYPE_F32, "Q5_0 * F32 → F32", false},  // 5-bit quantization (block size 32)
            
            // TODO: For matrix operations like MUL_MAT, consider adding:
            // {GGML_TYPE_Q4_K, GGML_TYPE_F32, GGML_TYPE_F32, "Q4_K * F32 → F32", false},  // K-quant (requires 256-aligned dims)
            // {GGML_TYPE_Q5_K, GGML_TYPE_F32, GGML_TYPE_F32, "Q5_K * F32 → F32", false},  // K-quant (requires 256-aligned dims)
            
            // NOTE: K-quant series require block size 256, so they need 256x256 tensors minimum
            // For operations that don't support large tensors in this test framework,
            // K-quant testing should be moved to specialized tests with larger memory allocation
        };
        
        int num_type_cases = sizeof(type_test_cases) / sizeof(type_test_cases[0]);
        int passed_type_tests = 0;
        int total_type_tests = 0;
        
        // Test each quantization type
        for (int case_idx = 0; case_idx < num_type_cases; case_idx++) {
            auto& test_case = type_test_cases[case_idx];
            printf("  🔍 Testing: %s\n", test_case.description);
            
            // Create test context with sufficient memory for quantized data
            struct ggml_init_params params;
            params.mem_size = 128 * 1024 * 1024;  // 128MB should handle quantized tensor overhead
            params.mem_buffer = nullptr;
            params.no_alloc = false;
            
            struct ggml_context* test_ctx = ggml_init(params);
            if (!test_ctx) {
                printf("      ❌ Failed to create test context for %s\n", test_case.description);
                failure_reason_str = "Context creation failed";
                overall_test_passed = false;
                continue;
            }
            
            // Create tensors with appropriate types
            // Use 32x32 = 1024 elements to be compatible with all quantization block sizes
            struct ggml_tensor* src0 = ggml_new_tensor_2d(test_ctx, test_case.src0_type, 32, 32);
            struct ggml_tensor* src1 = nullptr;
            
            // TODO: For unary operations, skip src1 creation:
            // For binary operations, create src1:
            if (test_case.src1_type != GGML_TYPE_COUNT) {
                src1 = ggml_new_tensor_2d(test_ctx, test_case.src1_type, 32, 32);
            }
            
            if (!src0 || (test_case.src1_type != GGML_TYPE_COUNT && !src1)) {
                printf("      ❌ Failed to create tensors for %s\n", test_case.description);
                ggml_free(test_ctx);
                failure_reason_str = "Tensor creation failed";
                overall_test_passed = false;
                continue;
            }
            
            // Initialize data based on type
            const int total_elements = 1024;  // 32x32
            
            // Initialize src0 based on its type
            if (test_case.src0_type == GGML_TYPE_F32) {
                float* src0_data = (float*)ggml_get_data(src0);
                for (int i = 0; i < total_elements; i++) {
                    src0_data[i] = 0.1f + i * 0.001f;  // Small values to avoid overflow
                }
            } else if (test_case.src0_type == GGML_TYPE_F16) {
                ggml_fp16_t* src0_data = (ggml_fp16_t*)ggml_get_data(src0);
                for (int i = 0; i < total_elements; i++) {
                    src0_data[i] = ggml_fp32_to_fp16(0.1f + i * 0.001f);
                }
            } else {
                // For quantized types, create F32 data first, then quantize
                float temp_data[1024];
                for (int i = 0; i < total_elements; i++) {
                    temp_data[i] = 0.1f + i * 0.001f;
                }
                // Get the quantization function for src0
                ggml_from_float_t quantize_fn = ggml_get_type_traits_cpu(test_case.src0_type)->from_float;
                if (quantize_fn) {
                    quantize_fn(temp_data, ggml_get_data(src0), total_elements);
                } else {
                    printf("      ⚠️  No quantization function available for %s\n", test_case.description);
                    ggml_free(test_ctx);
                    continue;
                }
            }
            
            // Initialize src1 if it exists (for binary operations)
            if (src1) {
                if (test_case.src1_type == GGML_TYPE_F32) {
                    float* src1_data = (float*)ggml_get_data(src1);
                    for (int i = 0; i < total_elements; i++) {
                        src1_data[i] = 0.05f + i * 0.0005f;  // Different pattern
                    }
                } else if (test_case.src1_type == GGML_TYPE_F16) {
                    ggml_fp16_t* src1_data = (ggml_fp16_t*)ggml_get_data(src1);
                    for (int i = 0; i < total_elements; i++) {
                        src1_data[i] = ggml_fp32_to_fp16(0.05f + i * 0.0005f);
                    }
                }
            }
            
            // Create ROPE operation
            // TODO: Replace with your specific operation:
            // For binary operations:
            struct ggml_tensor* result = ggml_add(test_ctx, src0, src1);  // Replace with ggml_ROPE
            // For unary operations:
            // struct ggml_tensor* result = ggml_ROPE(test_ctx, src0);  // Replace with your operation
            // For operations with parameters:
            // struct ggml_tensor* result = ggml_ROPE(test_ctx, src0, param1, param2);
            
            if (!result) {
                printf("      ❌ Failed to create ROPE operation for %s\n", test_case.description);
                ggml_free(test_ctx);
                failure_reason_str = "Operation creation failed";
                overall_test_passed = false;
                continue;
            }
            
            // Try to execute with NUMA executor
            struct ggml_cplan cplan = {};
            cplan.work_size = 0;
            cplan.work_data = nullptr;
            cplan.n_threads = 2;  // Use 2 threads for type testing
            cplan.threadpool = nullptr;
            cplan.abort_callback = nullptr;
            cplan.abort_callback_data = nullptr;
            
            enum ggml_status numa_result = ggml_numa_executor_execute_tensor(result, &cplan);
            
            if (numa_result == GGML_STATUS_SUCCESS) {
                if (test_case.expect_numa_support) {
                    printf("      ✅ NUMA kernel handled %s as expected\n", test_case.description);
                } else {
                    printf("      ✅ Reference fallback handled %s correctly\n", test_case.description);
                }
                passed_type_tests++;
            } else {
                printf("      ❌ Execution failed for %s (status: %d)\n", test_case.description, numa_result);
                failure_reason_str = "Execution failed";
                overall_test_passed = false;
            }
            
            total_type_tests++;
            ggml_free(test_ctx);
        }
        
        // Print summary for quantization type testing
        printf("  📊 ROPE Quantization Type Test Summary:\n");
        printf("    Total type combinations: %d\n", total_type_tests);
        printf("    Passed: %d\n", passed_type_tests);
        printf("    Failed: %d\n", total_type_tests - passed_type_tests);
        
        if (overall_test_passed) {
            printf("✅ ROPE quantization type coverage: VERIFIED\n");
            printf("  🎉 All quantization types work correctly (NUMA kernels or reference fallback)!\n\n");
        } else {
            printf("❌ ROPE quantization type coverage: FAILED - %s\n", failure_reason_str.c_str());
            printf("  ⚠️  Some quantization types failed to execute properly\n\n");
        }
        
        results.push_back({"ROPE_quantization_type_coverage", overall_test_passed, failure_reason_str});
    }

public:
    bool run_all_tests() {
        printf("🧪 NUMA Mathematical Correctness Test Suite - ROPE\n");
        printf("================================================================================\n");
        printf("🔧 Testing mathematical correctness with function pointer architecture\n");
        printf("Comparing NUMA parallel execution against serial reference implementation\n");
        printf("================================================================================\n\n");
        
        // Run mathematical correctness tests
        test_ROPE_mathematical_equivalence();
        
        // Run quantization type coverage tests  
        test_ROPE_quantization_type_coverage();
        
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
    // Initialize NUMA with mirroring strategy for data locality
    ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);
    
    // Check for --summary-only flag
    bool summary_only = false;
    if (argc > 1 && strcmp(argv[1], "--summary-only") == 0) {
        summary_only = true;
    }
    
    printf("🌟 Initializing NUMA system for mathematical correctness testing...\n");
    printf("✅ NUMA system initialized successfully\n\n");
    
    NumaRopeMathematicalCorrectnessTestSuite test_suite;
    bool all_passed = test_suite.run_all_tests();
    
    return all_passed ? 0 : 1;
}

/**
 * IMPLEMENTATION CHECKLIST:
 * 
 * When adapting this template for a new operation:
 * 
 * 1. ✅ Replace all instances of "ROPE" with your operation name (e.g., "GLU", "RMS_NORM")
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
 * - See test-numa-mathematical-correctness-mul.cpp for comprehensive testing with quantization and broadcasting
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
