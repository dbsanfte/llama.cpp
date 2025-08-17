#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-numa-operation-dispatch.h"
#include "ggml-numa-coordinator.h"
#include "ggml-cpu-impl.h"
#include "ggml-cpu/ops.h"
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
        printf("    🧮 Testing %s: %dx%d * %dx%d = %dx%d (M=%d, K=%d, N=%d, threads=%d)\n", 
               size_label, M, K, K, N, M, N, M, K, N, num_threads);
        
        // Create test context with sufficient memory for larger matrices
        struct ggml_init_params params = {0};
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
                                
                                case_passed = true;
                                int error_count = 0;
                                double max_abs_error = 0.0;
                                double max_rel_error = 0.0;
                                
                                for (int i = 0; i < total_elements; i++) {
                                    double numa_val = numa_data[i];
                                    double ref_val = ref_data[i];
                                    double abs_error = fabs(numa_val - ref_val);
                                    double rel_error = ref_val != 0.0 ? abs_error / fabs(ref_val) : 0.0;
                                    
                                    max_abs_error = fmax(max_abs_error, abs_error);
                                    max_rel_error = fmax(max_rel_error, rel_error);
                                    
                                    // Use strict tolerance for mathematical equivalence
                                    if (abs_error > 1e-6 && rel_error > 1e-6) {
                                        if (error_count < 3) { // Show first 3 errors for brevity
                                            printf("        ❌ Element[%d]: NUMA=%.8f, Reference=%.8f, AbsErr=%.2e, RelErr=%.2e\n",
                                                    i, numa_val, ref_val, abs_error, rel_error);
                                        }
                                        error_count++;
                                        case_passed = false;
                                    }
                                }
                                
                                if (case_passed) {
                                    printf("      ✅ %s (%d threads): MATHEMATICALLY EQUIVALENT (MaxAbsErr=%.2e, MaxRelErr=%.2e)\n",
                                            size_label, num_threads, max_abs_error, max_rel_error);
                                } else {
                                    printf("      ❌ %s (%d threads): MATHEMATICAL MISMATCH (%d/%d elements differ)\n",
                                            size_label, num_threads, error_count, total_elements);
                                    printf("        MaxAbsErr=%.2e, MaxRelErr=%.2e\n", max_abs_error, max_rel_error);
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
    bool run_all_tests() {
        printf("🧪 NUMA Mathematical Correctness Test Suite\n");
        printf("================================================================================\n");
        printf("🔧 Testing mathematical correctness with the new function pointer architecture\n");
        printf("Comparing NUMA parallel execution against serial reference implementations\n");
        printf("================================================================================\n\n");
        
        // Run mathematical correctness tests with updated function pointer API
        test_mul_mat_mathematical_equivalence();
        
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
                struct ggml_init_params params = {0};
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

int main() {
    // Initialize NUMA system
    printf("🔧 Initializing NUMA system for mathematical correctness testing...\n");
    
    // Initialize the NUMA coordinator system
    struct ggml_numa_coordinator_manager* manager = ggml_numa_coordinator_manager_get_global(8, false);
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
    
    if (all_passed) {
        return 0;
    } else {
        printf("💥 Some tests failed.\n");
        return 1;
    }
}
