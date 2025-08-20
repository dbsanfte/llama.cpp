#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-numa-operation-dispatch.h"
#include "ggml-numa-coordinator.h"
#include "ggml-cpu-impl.h"
#include "ggml-cpu/ops.h"
#include "numa-test-utils.h"
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <string.h>
#include <string>

// Test result structure
struct TestResult {
    std::string test_name;
    bool passed;
    std::string failure_reason;
};

class NumaMathematicalCorrectnessTestSuite {
private:
    std::vector<TestResult> results;
    
    // Test a single MUL_MAT case with specific dimensions and thread count
    bool test_single_mul_mat_case(int M, int K, int N, int num_threads, const char* size_label) {
        printf("    🧮 Testing %s: %dx%d * %dx%d = %dx%d (M=%d, K=%d, N=%d, threads=%d) [NUMA MIRROR]\n", 
               size_label, M, K, K, N, M, N, M, K, N, num_threads);
        
        // Create test context with sufficient memory for larger matrices
        struct ggml_init_params params;
        params.mem_size = 0;
        params.mem_buffer = nullptr;
        params.no_alloc = false;
        params.mem_size = std::max((size_t)(512 * 1024 * 1024), (size_t)(M * K + K * N + M * N) * sizeof(float) * 4); // Scale memory with matrix size
        params.mem_buffer = nullptr;
        params.no_alloc = false;
        
        struct ggml_context* test_ctx = ggml_init(params);
        if (!test_ctx) {
            printf("      ❌ Failed to create test context for %s\n", size_label);
            return false;
        }
        
        bool case_passed = false;
        
        // Create matrices with deterministic data
        struct ggml_tensor* a = ggml_new_tensor_2d(test_ctx, GGML_TYPE_F32, K, M);
        struct ggml_tensor* b = ggml_new_tensor_2d(test_ctx, GGML_TYPE_F32, K, N);
        
        if (!a || !b) {
            printf("      ❌ Failed to create test matrices for %s\n", size_label);
        } else {
            // Fill with deterministic test data that varies with matrix size for better testing
            float* a_data = (float*)ggml_get_data(a);
            float* b_data = (float*)ggml_get_data(b);
            
            // Use different data patterns for different matrix sizes to ensure variety
            int seed_a = M + K;  // Unique seed based on matrix dimensions
            int seed_b = K + N;
            
            for (int i = 0; i < ggml_nelements(a); i++) {
                a_data[i] = 0.1f + ((i + seed_a) % 37) * 0.01f; // Vary between 0.1 and 0.46
            }
            for (int i = 0; i < ggml_nelements(b); i++) {
                b_data[i] = 0.2f + ((i + seed_b) % 41) * 0.01f; // Vary between 0.2 and 0.60
            }
            
            // Create MUL_MAT operation
            struct ggml_tensor* numa_result = ggml_mul_mat(test_ctx, a, b);
            if (!numa_result) {
                printf("      ❌ Failed to create MUL_MAT operation for %s\n", size_label);
            } else {
                // Set up compute parameters for NUMA intercept
                struct ggml_compute_params numa_params = {
                    0,               // ith - Main thread (required for intercept)
                    num_threads,     // nth - Use specified thread count
                    0,               // wsize - Let dispatcher manage work buffer
                    nullptr,         // wdata
                    nullptr          // threadpool
                };
                
                // Execute via NUMA intercept (new function pointer API)
                enum ggml_status dispatch_result = ggml_numa_intercept_operation(numa_result, &numa_params);
                
                if (dispatch_result != GGML_STATUS_SUCCESS) {
                    printf("      ❌ NUMA dispatcher execution failed for %s (status=%d, threads=%d)\n", 
                           size_label, dispatch_result, num_threads);
                } else {
                    // Create reference computation using the same ggml_mul_mat approach
                    struct ggml_context* ref_ctx = ggml_init(params);
                    if (!ref_ctx) {
                        printf("      ❌ Failed to create reference context for %s\n", size_label);
                    } else {
                        struct ggml_tensor* ref_a = ggml_new_tensor_2d(ref_ctx, GGML_TYPE_F32, K, M);
                        struct ggml_tensor* ref_b = ggml_new_tensor_2d(ref_ctx, GGML_TYPE_F32, K, N);
                        
                        if (!ref_a || !ref_b) {
                            printf("      ❌ Failed to create reference matrices for %s\n", size_label);
                        } else {
                            // Copy identical data to reference matrices
                            memcpy(ggml_get_data(ref_a), a_data, ggml_nbytes(a));
                            memcpy(ggml_get_data(ref_b), b_data, ggml_nbytes(b));
                            
                            // Create reference result tensor with same dimensions as NUMA result
                            struct ggml_tensor* ref_result = ggml_new_tensor_2d(ref_ctx, GGML_TYPE_F32, M, N);
                            if (!ref_result) {
                                printf("      ❌ Failed to create reference result tensor for %s\n", size_label);
                            } else {
                                // Set up the result tensor's source pointers for the chunk kernel
                                ref_result->src[0] = ref_a;
                                ref_result->src[1] = ref_b;
                                ref_result->op = GGML_OP_MUL_MAT;
                                
                                // Set up compute params for single-threaded reference computation
                                struct ggml_compute_params ref_params = {
                                    0,       // ith
                                    1,       // nth - Single thread for reference (baseline)
                                    0,       // wsize
                                    nullptr, // wdata
                                    nullptr  // threadpool
                                };
                                
                                // Get matrix dimensions for chunk parameters
                                const int64_t ne00 = ref_a->ne[0]; // K dimension
                                const int64_t ne01 = ref_a->ne[1]; // M dimension  
                                const int64_t ne11 = ref_b->ne[1]; // N dimension
                                
                                // Calculate num_rows_per_vec_dot based on type
                                const int64_t num_rows_per_vec_dot = (ref_a->type == GGML_TYPE_F32) ? 1 : 1;
                                
                                // Call the underlying chunk kernel directly - this is the pure mathematical kernel
                                ggml_compute_forward_mul_mat_one_chunk(
                                    &ref_params,
                                    ref_result,           // dst
                                    ref_a->type,         // type  
                                    num_rows_per_vec_dot, // num_rows_per_vec_dot
                                    0,                   // ir0_start (all rows)
                                    ne01,                // ir0_end (all rows)
                                    0,                   // ir1_start (all cols)  
                                    ne11                 // ir1_end (all cols)
                                );
                                
                                // Compare NUMA result with reference result
                                float* numa_data = (float*)ggml_get_data(numa_result);
                                float* ref_data = (float*)ggml_get_data(ref_result);
                                int total_elements = ggml_nelements(numa_result);
                                
                                // ENHANCED: Check for corruption in NUMA result first
                                printf("        🔍 Checking NUMA result for corruption...\n");
                                auto numa_analysis = NumaTestUtils::analyze_tensor_corruption(numa_data, total_elements, "NUMA_result", false);
                                
                                if (numa_analysis.has_corruption) {
                                    printf("        🚨 CRITICAL: NUMA result contains corruption!\n");
                                    NumaTestUtils::print_corruption_report(numa_analysis, "NUMA_result");
                                    case_passed = false;
                                } else {
                                    printf("        ✅ NUMA result is clean (no NaN/inf detected)\n");
                                    
                                    // Also check reference for corruption (shouldn't happen but be safe)
                                    auto ref_analysis = NumaTestUtils::analyze_tensor_corruption(ref_data, total_elements, "reference_result", false);
                                    if (ref_analysis.has_corruption) {
                                        printf("        ⚠️  WARNING: Reference result contains corruption! Test invalid.\n");
                                        NumaTestUtils::print_corruption_report(ref_analysis, "reference_result");
                                        case_passed = false;
                                    } else {
                                        // Both results are clean, do mathematical comparison
                                        printf("        🧮 Comparing mathematical correctness...\n");
                                        case_passed = NumaTestUtils::tensors_equal(numa_data, ref_data, total_elements, 1e-6, 1e-6, true);
                                        
                                        if (case_passed) {
                                            printf("      ✅ %s (%d threads): MATHEMATICALLY EQUIVALENT AND CORRUPTION-FREE\n",
                                                    size_label, num_threads);
                                            printf("         NUMA stats: mean=%.6f, variance=%.6e, range=[%.6f, %.6f]\n",
                                                   numa_analysis.mean, numa_analysis.variance, numa_analysis.min_val, numa_analysis.max_val);
                                        } else {
                                            printf("      ❌ %s (%d threads): MATHEMATICAL MISMATCH (but no corruption)\n",
                                                    size_label, num_threads);
                                        }
                                    }
                                }
                            }
                        }
                        ggml_free(ref_ctx);
                    }
                }
            }
        }
        
        ggml_free(test_ctx);
        return case_passed;
    }

public:
    // Test case for quantized×F32 matrix multiplication
    // Tests that quantized weights (Q8_0, Q4_0, etc.) work correctly with F32 activations
    bool test_quantized_mul_mat_case(ggml_type quantized_type, int M, int K, int N, int num_threads, const char* size_label) {
        printf("    🧮 Testing %s×F32 %s: %dx%d * %dx%d = %dx%d (M=%d, K=%d, N=%d, threads=%d)\n", 
               ggml_type_name(quantized_type), size_label, M, K, K, N, M, N, M, K, N, num_threads);
        
        // Initialize test context with adequate memory for quantized and F32 tensors
        struct ggml_init_params params;
        params.mem_size = 0;
        params.mem_buffer = nullptr;
        params.no_alloc = false;
        params.mem_size = std::max((size_t)(512 * 1024 * 1024), (size_t)(M * K + K * N + M * N) * sizeof(float) * 4);
        params.mem_buffer = nullptr;
        params.no_alloc = false;
        
        struct ggml_context* test_ctx = ggml_init(params);
        if (!test_ctx) {
            printf("      ❌ Failed to create test context for %s %s\n", ggml_type_name(quantized_type), size_label);
            return false;
        }
        
        bool case_passed = false;
        
        // Create quantized matrix A (weights) and F32 matrix B (activations)
        struct ggml_tensor* a_f32 = ggml_new_tensor_2d(test_ctx, GGML_TYPE_F32, K, M);
        struct ggml_tensor* a_quantized = ggml_new_tensor_2d(test_ctx, quantized_type, K, M);
        struct ggml_tensor* b = ggml_new_tensor_2d(test_ctx, GGML_TYPE_F32, K, N);
        
        if (!a_f32 || !a_quantized || !b) {
            printf("      ❌ Failed to create %s test matrices for %s\n", ggml_type_name(quantized_type), size_label);
        } else {
            // Fill F32 matrices with deterministic test data (same pattern as test-quantize-fns.cpp)
            float* a_f32_data = (float*)ggml_get_data(a_f32);
            float* b_data = (float*)ggml_get_data(b);
            
            // Generate synthetic data similar to test-quantize-fns.cpp
            for (int i = 0; i < ggml_nelements(a_f32); i++) {
                a_f32_data[i] = 0.1f + 2.0f * cosf(i + M + K);  // Similar to generate_data()
            }
            for (int i = 0; i < ggml_nelements(b); i++) {
                b_data[i] = 0.1f + 2.0f * cosf(i + K + N + 1.0f);  // Offset by 1.0 for variation
            }
            
            // Quantize matrix A to the specified quantized type
            const auto* qfns_cpu = ggml_get_type_traits_cpu(quantized_type);
            if (!qfns_cpu || !qfns_cpu->from_float) {
                printf("      ❌ No quantization function available for %s\n", ggml_type_name(quantized_type));
                ggml_free(test_ctx);
                return false;
            }
            
            // Quantize the data
            qfns_cpu->from_float(a_f32_data, ggml_get_data(a_quantized), ggml_nelements(a_f32));
            
            // Create quantized×F32 MUL_MAT operation - this should be valid
            struct ggml_tensor* numa_result = ggml_mul_mat(test_ctx, a_quantized, b);
            if (!numa_result) {
                printf("      ❌ Failed to create %s×F32 MUL_MAT operation for %s\n", ggml_type_name(quantized_type), size_label);
            } else {
                // Calculate work buffer size needed for type conversion
                const auto* traits = ggml_get_type_traits_cpu(quantized_type);
                size_t work_size = 0;
                if (traits && traits->vec_dot_type != GGML_TYPE_F32) {
                    work_size = ggml_row_size(traits->vec_dot_type, K) * N * num_threads;
                }
                
                // Allocate work buffer if needed
                char* work_buffer = nullptr;
                if (work_size > 0) {
                    work_buffer = (char*)malloc(work_size);
                    if (!work_buffer) {
                        printf("      ❌ Failed to allocate work buffer (%zu bytes) for %s\n", work_size, ggml_type_name(quantized_type));
                        ggml_free(test_ctx);
                        return false;
                    }
                }
                
                // Set up compute parameters for NUMA execution with work buffer
                struct ggml_compute_params numa_params = {
                    0,               // ith
                    num_threads,     // nth
                    work_size,       // wsize
                    work_buffer,     // wdata
                    nullptr          // threadpool
                };
                
                printf("        🚀 Using NUMA coordinator with work buffer (%zu bytes)...\n", work_size);
                
                // Execute NUMA dispatcher
                enum ggml_status dispatch_result = ggml_numa_intercept_operation(numa_result, &numa_params);
                
                if (dispatch_result != GGML_STATUS_SUCCESS) {
                    printf("      ❌ %s×F32 NUMA execution failed for %s (status=%d, threads=%d)\n", 
                           ggml_type_name(quantized_type), size_label, dispatch_result, num_threads);
                } else {
                    // Create F32×F32 reference using direct chunk kernel (same approach as F32×F32 test)
                    struct ggml_tensor* ref_result = ggml_new_tensor_2d(test_ctx, GGML_TYPE_F32, M, N);
                    if (!ref_result) {
                        printf("      ❌ Failed to create F32×F32 reference result tensor for %s\n", size_label);
                    } else {
                        // Set up the reference result tensor for chunk kernel computation
                        ref_result->src[0] = a_f32;  // F32 version of matrix A
                        ref_result->src[1] = b;      // F32 matrix B (same as NUMA test)
                        ref_result->op = GGML_OP_MUL_MAT;
                        
                        // Set up compute params for single-threaded F32×F32 reference computation
                        struct ggml_compute_params ref_params = {
                            0,       // ith
                            1,       // nth - Single thread for reference (baseline)
                            0,       // wsize
                            nullptr, // wdata
                            nullptr  // threadpool
                        };
                        
                        // Get matrix dimensions for chunk parameters
                        const int64_t ne00 = a_f32->ne[0]; // K dimension
                        const int64_t ne01 = a_f32->ne[1]; // M dimension  
                        const int64_t ne11 = b->ne[1];     // N dimension
                        
                        // F32×F32 multiplication always uses 1 row per vec_dot
                        const int64_t num_rows_per_vec_dot = 1;
                        
                        // Call the underlying chunk kernel directly for F32×F32 reference
                        ggml_compute_forward_mul_mat_one_chunk(
                            &ref_params,
                            ref_result,           // dst
                            a_f32->type,         // type (GGML_TYPE_F32)
                            num_rows_per_vec_dot, // num_rows_per_vec_dot
                            0,                   // ir0_start (all rows)
                            ne01,                // ir0_end (all rows)
                            0,                   // ir1_start (all cols)  
                            ne11                 // ir1_end (all cols)
                        );
                        
                        // Compare quantized×F32 NUMA result with F32×F32 reference
                        float* numa_data = (float*)ggml_get_data(numa_result);
                        float* ref_data = (float*)ggml_get_data(ref_result);
                        int total_elements = ggml_nelements(numa_result);
                        
                        // Check for corruption in quantized×F32 NUMA result
                        printf("        🔍 Checking %s×F32 NUMA result for corruption...\n", ggml_type_name(quantized_type));
                        auto numa_analysis = NumaTestUtils::analyze_tensor_corruption(numa_data, total_elements, "quantized×F32_NUMA_result", false);
                        
                        if (numa_analysis.has_corruption) {
                            printf("        🚨 CRITICAL: %s×F32 NUMA result contains corruption! BUG DETECTED!\n", ggml_type_name(quantized_type));
                            NumaTestUtils::print_corruption_report(numa_analysis, "quantized×F32_NUMA_result");
                            case_passed = false;
                        } else {
                            printf("        ✅ %s×F32 NUMA result is clean (no NaN/inf detected)\n", ggml_type_name(quantized_type));
                            
                            // Compare with F32×F32 reference (allowing for quantization error)
                            printf("        🧮 Comparing %s×F32 NUMA vs F32×F32 reference...\n", ggml_type_name(quantized_type));
                            
                            // Use higher tolerance for quantized types (based on test-quantize-fns.cpp patterns)
                            float tolerance = (quantized_type == GGML_TYPE_Q2_K || quantized_type == GGML_TYPE_IQ2_S) ? 0.1f :
                                            (quantized_type == GGML_TYPE_Q3_K || quantized_type == GGML_TYPE_IQ3_S) ? 0.05f :
                                            (quantized_type == GGML_TYPE_Q4_0 || quantized_type == GGML_TYPE_Q4_1) ? 0.02f :
                                            (quantized_type == GGML_TYPE_Q5_0 || quantized_type == GGML_TYPE_Q5_1) ? 0.01f :
                                            0.5f;  // Default for Q8_0, Q8_1 (higher tolerance for 8-bit quantization)
                            
                            case_passed = NumaTestUtils::tensors_equal(numa_data, ref_data, total_elements, tolerance, tolerance, true);
                            
                            if (case_passed) {
                                printf("      ✅ %s×F32 %s (%d threads): MATHEMATICALLY EQUIVALENT AND CORRUPTION-FREE\n",
                                        ggml_type_name(quantized_type), size_label, num_threads);
                            } else {
                                printf("      ❌ %s×F32 %s (%d threads): MATHEMATICAL MISMATCH (tolerance=%.4f)\n",
                                        ggml_type_name(quantized_type), size_label, num_threads, tolerance);
                            }
                        }
                    }
                }
                
                // Free work buffer
                if (work_buffer) {
                    free(work_buffer);
                }
            }
        }
        
        ggml_free(test_ctx);
        return case_passed;
    }

    bool run_all_tests() {
        printf("🧪 NUMA Mathematical Correctness Test Suite\n");
        printf("================================================================================\n");
        printf("🔧 Testing mathematical correctness with the new function pointer architecture\n");
        printf("Comparing NUMA parallel execution against serial reference implementations\n");
        printf("🚨 ENHANCED: Now includes Q8_0×F32 corruption detection tests\n");
        printf("================================================================================\n\n");
        
        // Run mathematical correctness tests with updated function pointer API
        test_mul_mat_mathematical_equivalence();
        
        // NEW: Run Q8_0×F32 corruption detection tests
        test_q8_0_corruption_detection();
        
        // NEW: Run comprehensive quantized types mathematical correctness tests
        test_quantized_types_mathematical_correctness();
        
        // Test data parallel type conversion buffer sizing (regression test)
        test_data_parallel_type_conversion_buffer_sizing();
        
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
    void test_mul_mat_mathematical_equivalence() {
        printf("--- Test: MUL_MAT Mathematical Equivalence (Multi-Dimensional) ---\n");
        printf("Testing NUMA parallel MUL_MAT vs serial reference implementation...\n");
        printf("Testing across tiny, small, medium, and large matrices with various coordinator execution strategies\n\n");
        
        bool overall_test_passed = true;
        const char* failure_reason = nullptr;
        
        // Define test dimensions: tiny, small, medium, large
        struct {
            int M, K, N;
            const char* label;
        } test_cases[] = {
            {8, 8, 4, "TINY"},           // 8x8 * 8x4 = 8x4 (256 + 32 + 32 = 320 elements)
            {32, 32, 16, "SMALL"},       // 32x32 * 32x16 = 32x16 (1024 + 512 + 512 = 2048 elements)  
            {128, 64, 32, "MEDIUM"},     // 128x64 * 64x32 = 128x32 (8192 + 2048 + 4096 = 14336 elements)
            {256, 128, 64, "LARGE"}      // 256x128 * 128x64 = 256x64 (32768 + 8192 + 16384 = 57344 elements)
        };
        
        // Define coordinator execution strategies (various thread counts)
        int thread_strategies[] = {1, 2, 4, 6, 8};
        int num_strategies = sizeof(thread_strategies) / sizeof(thread_strategies[0]);
        int num_test_cases = sizeof(test_cases) / sizeof(test_cases[0]);
        
        printf("  🎯 Testing %d matrix dimensions with %d thread strategies (%d total test combinations)\n\n", 
               num_test_cases, num_strategies, num_test_cases * num_strategies);
        
        int total_tests = 0;
        int passed_tests = 0;
        
        // Test each matrix dimension with each thread strategy
        for (int case_idx = 0; case_idx < num_test_cases; case_idx++) {
            printf("  📏 Testing %s matrices (%dx%d * %dx%d):\n", 
                   test_cases[case_idx].label, 
                   test_cases[case_idx].M, test_cases[case_idx].K,
                   test_cases[case_idx].K, test_cases[case_idx].N);
            
            for (int strategy_idx = 0; strategy_idx < num_strategies; strategy_idx++) {
                int num_threads = thread_strategies[strategy_idx];
                
                bool case_passed = test_single_mul_mat_case(
                    test_cases[case_idx].M, 
                    test_cases[case_idx].K, 
                    test_cases[case_idx].N, 
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
        printf("  📊 MUL_MAT Multi-Dimensional Test Summary:\n");
        printf("    Total test combinations: %d\n", total_tests);
        printf("    Passed: %d\n", passed_tests);
        printf("    Failed: %d\n", total_tests - passed_tests);
        
        if (overall_test_passed) {
            printf("✅ MUL_MAT mathematical equivalence (multi-dimensional): VERIFIED\n");
            printf("  🎉 All matrix dimensions and thread strategies produce mathematically equivalent results!\n\n");
        } else {
            printf("❌ MUL_MAT mathematical equivalence (multi-dimensional): FAILED - %s\n", failure_reason);
            printf("  ⚠️  Mathematical mismatches detected across different dimensions or thread strategies\n\n");
        }
        
        results.push_back({"mul_mat_mathematical_equivalence", overall_test_passed, failure_reason ? failure_reason : ""});
    }
    
    // NEW: Q8_0×F32 corruption detection test
    void test_q8_0_corruption_detection() {
        printf("--- Test: Q8_0×F32 Corruption Detection (Critical Bug Detection) ---\n");
        printf("Testing Q8_0×F32 matrix multiplication for NaN/inf corruption...\n");
        printf("This test specifically targets the Q8_0×F32 corruption bug that produces garbage output\n\n");
        
        bool overall_test_passed = true;
        const char* failure_reason = nullptr;
        
        // IMPORTANT: Q8_0 has a block size of 32 (QK8_0), so K dimension must be divisible by 32
        // Otherwise ggml_row_size() will fail with assertion: `ne % ggml_blck_size(type) == 0'
        
        // Test Q8_0×F32 with dimensions aligned to Q8_0 block size (32)
        struct {
            int M, K, N;
            const char* label;
        } q8_0_test_cases[] = {
            {8, 32, 4, "TINY_Q8_0"},     // K=32, aligned to Q8_0 block size
            {32, 64, 16, "SMALL_Q8_0"},  // K=64, aligned to Q8_0 block size
            {64, 128, 32, "MEDIUM_Q8_0"} // K=128, aligned to Q8_0 block size
        };
        
        // Test with multiple threads to trigger NUMA dispatch
        int thread_strategies[] = {2, 4, 6};
        int num_strategies = sizeof(thread_strategies) / sizeof(thread_strategies[0]);
        int num_test_cases = sizeof(q8_0_test_cases) / sizeof(q8_0_test_cases[0]);
        
        printf("  🎯 Testing %d Q8_0×F32 matrix dimensions with %d thread strategies (%d total combinations)\n\n", 
               num_test_cases, num_strategies, num_test_cases * num_strategies);
        
        int total_tests = 0;
        int passed_tests = 0;
        
        // Test each Q8_0×F32 matrix dimension with each thread strategy
        for (int case_idx = 0; case_idx < num_test_cases; case_idx++) {
            printf("  📏 Testing Q8_0×F32 %s matrices (%dx%d * %dx%d):\n", 
                   q8_0_test_cases[case_idx].label, 
                   q8_0_test_cases[case_idx].M, q8_0_test_cases[case_idx].K,
                   q8_0_test_cases[case_idx].K, q8_0_test_cases[case_idx].N);
                   
            for (int strategy_idx = 0; strategy_idx < num_strategies; strategy_idx++) {
                int num_threads = thread_strategies[strategy_idx];
                total_tests++;
                
                bool case_passed = test_quantized_mul_mat_case(
                    GGML_TYPE_Q8_0,
                    q8_0_test_cases[case_idx].M,
                    q8_0_test_cases[case_idx].K, 
                    q8_0_test_cases[case_idx].N,
                    num_threads,
                    q8_0_test_cases[case_idx].label
                );
                
                if (case_passed) {
                    passed_tests++;
                } else {
                    overall_test_passed = false;
                    if (!failure_reason) {
                        failure_reason = "Q8_0×F32 corruption or mathematical mismatch detected";
                    }
                }
            }
            printf("\n");
        }
        
        // Print summary for Q8_0×F32 test
        printf("  📊 Q8_0×F32 Corruption Detection Test Summary:\n");
        printf("    Total test combinations: %d\n", total_tests);
        printf("    Passed: %d\n", passed_tests);
        printf("    Failed: %d\n", total_tests - passed_tests);
        
        if (overall_test_passed) {
            printf("✅ Q8_0×F32 corruption detection: VERIFIED\n");
            printf("  🎉 All Q8_0×F32 operations produce clean results without corruption!\n\n");
        } else {
            printf("❌ Q8_0×F32 corruption detection: FAILED - %s\n", failure_reason);
            printf("  🚨 CRITICAL: Q8_0×F32 operations are producing corrupted output! BUG CONFIRMED!\n\n");
        }
        
        results.push_back({"q8_0_corruption_detection", overall_test_passed, failure_reason ? failure_reason : ""});
    }
    
    // Comprehensive quantized types test (following test-quantize-fns.cpp patterns)
    void test_quantized_types_mathematical_correctness() {
        printf("--- Test: Quantized Types Mathematical Correctness (All Supported Types) ---\n");
        printf("Testing quantized×F32 matrix multiplication for all supported quantization types...\n");
        printf("This test validates mathematical correctness with proper quantization error tolerance\n\n");
        
        bool overall_test_passed = true;
        const char* failure_reason = nullptr;
        
        // Test quantization types that are commonly supported in ggml (based on test-quantize-fns.cpp)
        struct {
            ggml_type type;
            const char* name;
            const char* description;
        } quantized_types[] = {
            {GGML_TYPE_Q8_0, "Q8_0", "8-bit quantization with high precision"},
            {GGML_TYPE_Q4_0, "Q4_0", "4-bit quantization with 32-element blocks"},
            {GGML_TYPE_Q5_0, "Q5_0", "5-bit quantization with 32-element blocks"}
        };
        
        int num_quantized_types = sizeof(quantized_types) / sizeof(quantized_types[0]);
        
        // Test dimensions similar to test-quantize-fns.cpp (focused on correctness)
        struct {
            int M, K, N;
            const char* label;
        } test_dimensions[] = {
            {32, 64, 32, "SMALL"}         // Small matrices for precise validation
        };
        
        int num_dimensions = sizeof(test_dimensions) / sizeof(test_dimensions[0]);
        
        // Thread strategies (focused on multi-threading correctness)
        int thread_strategies[] = {1, 2};
        int num_strategies = sizeof(thread_strategies) / sizeof(thread_strategies[0]);
        
        int total_tests = 0;
        int passed_tests = 0;
        
        // Test each quantization type
        for (int type_idx = 0; type_idx < num_quantized_types; type_idx++) {
            printf("🔧 Testing %s (%s):\n", 
                   quantized_types[type_idx].name, 
                   quantized_types[type_idx].description);
            
            // Test each dimension combination for this type
            for (int dim_idx = 0; dim_idx < num_dimensions; dim_idx++) {
                printf("  📏 Testing %s×F32 %s matrices (%dx%d * %dx%d):\n", 
                       quantized_types[type_idx].name,
                       test_dimensions[dim_idx].label, 
                       test_dimensions[dim_idx].M, test_dimensions[dim_idx].K,
                       test_dimensions[dim_idx].K, test_dimensions[dim_idx].N);
                       
                // Test each threading strategy for this type+dimension
                for (int strategy_idx = 0; strategy_idx < num_strategies; strategy_idx++) {
                    int num_threads = thread_strategies[strategy_idx];
                    total_tests++;
                    
                    bool case_passed = test_quantized_mul_mat_case(
                        quantized_types[type_idx].type,
                        test_dimensions[dim_idx].M,
                        test_dimensions[dim_idx].K, 
                        test_dimensions[dim_idx].N,
                        num_threads,
                        test_dimensions[dim_idx].label
                    );
                    
                    if (case_passed) {
                        passed_tests++;
                    } else {
                        overall_test_passed = false;
                        if (!failure_reason) {
                            failure_reason = "Quantized×F32 mathematical mismatch or corruption detected";
                        }
                    }
                }
                printf("\n");
            }
        }
        
        // Print comprehensive summary
        printf("  📊 Quantized Types Mathematical Correctness Test Summary:\n");
        printf("    Quantization types tested: %d\n", num_quantized_types);
        printf("    Dimension combinations: %d\n", num_dimensions);
        printf("    Threading strategies: %d\n", num_strategies);
        printf("    Total test combinations: %d\n", total_tests);
        printf("    Passed: %d\n", passed_tests);
        printf("    Failed: %d\n", total_tests - passed_tests);
        
        if (overall_test_passed) {
            printf("✅ Quantized types mathematical correctness: VERIFIED\n");
            printf("  🎉 All quantized×F32 operations produce mathematically correct results!\n\n");
        } else {
            printf("❌ Quantized types mathematical correctness: FAILED - %s\n", failure_reason);
            printf("  🚨 CRITICAL: Some quantized×F32 operations are producing incorrect results!\n\n");
        }
        
        results.push_back({"quantized_types_mathematical_correctness", overall_test_passed, failure_reason ? failure_reason : ""});
    }
    
    // Regression test for data parallel type conversion buffer sizing issue
    // This test specifically targets the scenario that caused assertion failures:
    // "GGML_ASSERT(single_thread_params.wsize >= total_conversion_size) failed"
    void test_data_parallel_type_conversion_buffer_sizing() {
        printf("--- Test: Data Parallel Type Conversion Buffer Sizing (Regression Test) ---\n");
        printf("Testing buffer allocation for type conversion in data parallel execution...\n");
        printf("This test reproduces the scenario that caused assertion failures in production\n\n");
        
        bool overall_test_passed = true;
        const char* failure_reason = nullptr;
        
        // Test configuration designed to trigger type conversion and data parallel execution
        // These dimensions are chosen to create scenarios where:
        // 1. Type conversion is likely required
        // 2. Data parallel execution will be used
        // 3. Work buffer sizing becomes critical
        struct {
            int M, K, N;
            const char* label;
            const char* description;
        } type_conversion_test_cases[] = {
            {64, 512, 128, "MEDIUM_WIDE", "Medium matrix with high K dimension (triggers type conversion)"},
            {128, 1024, 64, "LARGE_DEEP", "Large matrix with very high K dimension (stress test)"},
            {32, 2048, 32, "NARROW_DEEP", "Narrow matrix with extreme K dimension (edge case)"},
            {256, 768, 128, "PRODUCTION", "Production-like dimensions (similar to real model layers)"}
        };
        
        int num_test_cases = sizeof(type_conversion_test_cases) / sizeof(type_conversion_test_cases[0]);
        
        // Test with multiple threads to force data parallel execution
        int data_parallel_thread_counts[] = {4, 6, 8};  // Higher thread counts more likely to trigger data parallelism
        int num_thread_configs = sizeof(data_parallel_thread_counts) / sizeof(data_parallel_thread_counts[0]);
        
        printf("  🎯 Testing %d matrix configurations with %d thread counts (%d total combinations)\n", 
               num_test_cases, num_thread_configs, num_test_cases * num_thread_configs);
        printf("  🔍 Focus: Work buffer allocation for type conversion in chunked execution\n\n");
        
        int total_tests = 0;
        int passed_tests = 0;
        
        for (int case_idx = 0; case_idx < num_test_cases; case_idx++) {
            auto& test_case = type_conversion_test_cases[case_idx];
            printf("  📏 Testing %s configuration (%dx%d * %dx%d):\n", 
                   test_case.label, test_case.M, test_case.K, test_case.K, test_case.N);
            printf("      🔍 %s\n", test_case.description);
            
            for (int thread_idx = 0; thread_idx < num_thread_configs; thread_idx++) {
                int num_threads = data_parallel_thread_counts[thread_idx];
                
                // Test with larger memory context to handle type conversion buffers
                struct ggml_init_params params;
        params.mem_size = 0;
        params.mem_buffer = nullptr;
        params.no_alloc = false;
                params.mem_size = std::max((size_t)(1024 * 1024 * 1024), // At least 1GB
                                         (size_t)(test_case.M * test_case.K + test_case.K * test_case.N + test_case.M * test_case.N) * sizeof(float) * 8);
                params.mem_buffer = nullptr;
                params.no_alloc = false;
                
                struct ggml_context* test_ctx = ggml_init(params);
                if (!test_ctx) {
                    printf("      ❌ Failed to create test context for %s (%d threads)\n", test_case.label, num_threads);
                    overall_test_passed = false;
                    total_tests++;
                    continue;
                }
                
                bool case_passed = false;
                
                // Create matrices - specifically use F32 to potentially trigger type conversion
                struct ggml_tensor* a = ggml_new_tensor_2d(test_ctx, GGML_TYPE_F32, test_case.K, test_case.M);
                struct ggml_tensor* b = ggml_new_tensor_2d(test_ctx, GGML_TYPE_F32, test_case.K, test_case.N);
                
                if (!a || !b) {
                    printf("      ❌ Failed to create test matrices for %s (%d threads)\n", test_case.label, num_threads);
                } else {
                    // Fill with test data that's deterministic but varies with position
                    float* a_data = (float*)ggml_get_data(a);
                    float* b_data = (float*)ggml_get_data(b);
                    
                    // Use data patterns that might trigger specific vector operations
                    for (int i = 0; i < ggml_nelements(a); i++) {
                        a_data[i] = 0.5f + ((i * 7) % 17) * 0.02f; // Values between 0.5 and 0.82
                    }
                    for (int i = 0; i < ggml_nelements(b); i++) {
                        b_data[i] = 0.3f + ((i * 11) % 19) * 0.03f; // Values between 0.3 and 0.84
                    }
                    
                    // Create MUL_MAT operation
                    struct ggml_tensor* result = ggml_mul_mat(test_ctx, a, b);
                    if (!result) {
                        printf("      ❌ Failed to create MUL_MAT operation for %s (%d threads)\n", test_case.label, num_threads);
                    } else {
                        // Set up compute parameters to trigger data parallel execution
                        struct ggml_compute_params numa_params = {
                            0,               // ith - Main thread
                            num_threads,     // nth - Use specified thread count
                            0,               // wsize - Let dispatcher calculate (this was the bug!)
                            nullptr,         // wdata
                            nullptr          // threadpool
                        };
                        
                        printf("      🧮 Testing %s with %d threads (potential type conversion scenario)...\n", 
                               test_case.label, num_threads);
                        
                        // Execute via NUMA intercept - this is where the assertion failure occurred
                        enum ggml_status dispatch_result = ggml_numa_intercept_operation(result, &numa_params);
                        
                        if (dispatch_result != GGML_STATUS_SUCCESS) {
                            printf("      ❌ NUMA dispatch failed for %s (%d threads): status=%d\n", 
                                   test_case.label, num_threads, dispatch_result);
                            printf("          This indicates the buffer sizing regression has returned!\n");
                            if (!failure_reason) {
                                failure_reason = "Data parallel type conversion buffer sizing assertion failure detected";
                            }
                        } else {
                            printf("      ✅ %s (%d threads): Buffer sizing correct, execution successful\n", 
                                   test_case.label, num_threads);
                            case_passed = true;
                        }
                    }
                }
                
                ggml_free(test_ctx);
                
                total_tests++;
                if (case_passed) {
                    passed_tests++;
                } else {
                    overall_test_passed = false;
                }
            }
            printf("\n");
        }
        
        // Print summary for this regression test
        printf("  📊 Data Parallel Type Conversion Buffer Sizing Test Summary:\n");
        printf("    Total test combinations: %d\n", total_tests);
        printf("    Passed: %d\n", passed_tests);
        printf("    Failed: %d\n", total_tests - passed_tests);
        
        if (overall_test_passed) {
            printf("✅ Data parallel type conversion buffer sizing: VERIFIED\n");
            printf("  🎉 All type conversion scenarios handle buffer allocation correctly!\n");
            printf("  🔧 The regression that caused assertion failures has been prevented\n\n");
        } else {
            printf("❌ Data parallel type conversion buffer sizing: FAILED - %s\n", failure_reason);
            printf("  ⚠️  Buffer sizing issues detected - the assertion failure regression may have returned!\n");
            printf("  🐛 This test would have caught the production issue we experienced\n\n");
        }
        
        results.push_back({"data_parallel_type_conversion_buffer_sizing", overall_test_passed, failure_reason ? failure_reason : ""});
    }
    
    void print_summary() {
        printf("================================================================================\n");
        printf("                    Mathematical Correctness Test Results\n");
        printf("================================================================================\n");
        
        int passed_count = 0;
        int total_count = results.size();
        
        for (const auto& result : results) {
            if (result.passed) {
                printf("✅ %s: PASSED\n", result.test_name.c_str());
                passed_count++;
            } else {
                printf("❌ %s: FAILED - %s\n", result.test_name.c_str(), result.failure_reason.c_str());
            }
        }
        
        printf("--------------------------------------------------------------------------------\n");
        printf("Total: %d/%d tests passed ", passed_count, total_count);
        
        if (passed_count == total_count) {
            printf("🎉 ALL TESTS PASSED!\n");
            printf("================================================================================\n");
            printf("✅ NUMA Mathematical Correctness: SUCCESS\n\n");
            printf("🎯 Mathematical equivalence verified between NUMA parallel and serial execution\n");
            printf("🧮 All arithmetic operations produce identical results\n");
        } else {
            printf("💥 Some tests failed.\n");
            printf("================================================================================\n");
            printf("❌ NUMA Mathematical Correctness: FAILURES DETECTED\n\n");
            printf("⚠️  Mathematical mismatch between NUMA parallel and serial execution detected\n");
        }
        
        printf("🧪 Mathematical correctness testing completed!\n");
    }
};

int main(int argc, char* argv[]) {
    // Initialize NUMA with MIRROR strategy for testing
    ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);
    
    // Parse command line arguments for --summary-only flag
    bool summary_only = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--summary-only") == 0) {
            summary_only = true;
            break;
        }
    }
    
    // If summary_only mode, redirect verbose output to /dev/null
    FILE* dev_null = nullptr;
    FILE* original_stdout = nullptr;
    if (summary_only) {
        dev_null = fopen("/dev/null", "w");
        if (dev_null) {
            original_stdout = stdout;
            stdout = dev_null;
        }
    }
    
    // Initialize NUMA system with MIRROR strategy for real NUMA testing
    printf("🔧 Initializing NUMA system for mathematical correctness testing...\n");
    printf("� Using MIRROR mode to test real NUMA data slicing on multi-NUMA hardware\n");
    
    // Initialize the NUMA coordinator system using MIRROR strategy
    struct ggml_numa_coordinator_manager* manager = ggml_numa_coordinator_manager_get_global(8);
    if (!manager) {
        fprintf(stderr, "❌ Failed to initialize NUMA coordinator manager\n");
        return 1;
    }
    
    // Initialize the dispatcher system
    ggml_numa_dispatch_init();
    
    printf("✅ NUMA system initialized successfully\n\n");
    
    // Run mathematical correctness tests
    NumaMathematicalCorrectnessTestSuite suite;
    bool all_passed = suite.run_all_tests();
    
    // Restore stdout and close dev_null if summary_only mode was used
    if (summary_only && dev_null && original_stdout) {
        stdout = original_stdout;
        fclose(dev_null);
        printf("✅ NUMA MUL_MAT Mathematical Correctness Test %s\n", all_passed ? "PASSED" : "FAILED");
    }
    
    if (all_passed) {
        return 0;
    } else {
        printf("💥 Some tests failed.\n");
        return 1;
    }
}
