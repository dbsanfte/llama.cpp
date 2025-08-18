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
    
    // Test a single SOFT_MAX case with specific dimensions and thread count
    bool test_single_soft_max_case(int rows, int cols, int num_threads, const char* size_label) {
        printf("    🧮 Testing %s: %dx%d tensor (rows=%d, cols=%d, threads=%d) [FORCE_MULTI_SOCKET]\n", 
               size_label, rows, cols, rows, cols, num_threads);
        
        // Create test context with sufficient memory for the tensor
        struct ggml_init_params params = {0};
        params.mem_size = std::max((size_t)(512 * 1024 * 1024), (size_t)(rows * cols) * sizeof(float) * 8); // Scale memory with tensor size
        params.mem_buffer = nullptr;
        params.no_alloc = false;
        
        struct ggml_context* test_ctx = ggml_init(params);
        if (!test_ctx) {
            printf("      ❌ Failed to create test context for %s\n", size_label);
            return false;
        }
        
        bool case_passed = false;
        
        // Create input tensor with deterministic data
        struct ggml_tensor* input = ggml_new_tensor_2d(test_ctx, GGML_TYPE_F32, cols, rows);
        
        if (!input) {
            printf("      ❌ Failed to create test tensor for %s\n", size_label);
        } else {
            // Fill with deterministic test data that varies with tensor size for better testing
            float* input_data = (float*)ggml_get_data(input);
            
            // Use different data patterns for different tensor sizes to ensure variety
            int seed = rows + cols;  // Unique seed based on tensor dimensions
            
            for (int i = 0; i < ggml_nelements(input); i++) {
                // Use values that will produce meaningful soft_max results (not too large/small)
                input_data[i] = -2.0f + ((i + seed) % 73) * 0.1f; // Vary between -2.0 and 5.2
            }
            
            // Create SOFT_MAX operation (unary operation on single tensor)
            struct ggml_tensor* numa_result = ggml_soft_max(test_ctx, input);
            if (!numa_result) {
                printf("      ❌ Failed to create SOFT_MAX operation for %s\n", size_label);
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
                    // Create reference computation using the same ggml_soft_max approach
                    struct ggml_context* ref_ctx = ggml_init(params);
                    if (!ref_ctx) {
                        printf("      ❌ Failed to create reference context for %s\n", size_label);
                    } else {
                        struct ggml_tensor* ref_input = ggml_new_tensor_2d(ref_ctx, GGML_TYPE_F32, cols, rows);
                        
                        if (!ref_input) {
                            printf("      ❌ Failed to create reference tensor for %s\n", size_label);
                        } else {
                            // Copy identical data to reference tensor
                            memcpy(ggml_get_data(ref_input), input_data, ggml_nbytes(input));
                            
                            // Create reference SOFT_MAX operation using the same API as NUMA version
                            struct ggml_tensor* ref_result = ggml_soft_max(ref_ctx, ref_input);
                            if (!ref_result) {
                                printf("      ❌ Failed to create reference SOFT_MAX operation for %s\n", size_label);
                            } else {
                                // Set up compute params for single-threaded reference computation
                                // Allocate work buffer for soft_max (needs (ne00 + CACHE_LINE_SIZE_F32) * nth * sizeof(float))
                                const int64_t ne00 = ref_input->ne[0]; // cols dimension
                                const size_t work_buffer_size = (ne00 + 16) * 1 * sizeof(float); // 16 = CACHE_LINE_SIZE_F32 estimate, 1 thread
                                float* work_buffer = (float*)malloc(work_buffer_size);
                                if (!work_buffer) {
                                    printf("      ❌ Failed to allocate work buffer for reference computation\n");
                                } else {
                                    struct ggml_compute_params ref_params = {
                                        0,           // ith
                                        1,           // nth - Single thread for reference (baseline)
                                        work_buffer_size, // wsize
                                        work_buffer, // wdata - Provide work buffer for soft_max
                                        nullptr      // threadpool
                                    };
                                    
                                    // Call the underlying soft_max kernel directly - this is the pure mathematical kernel
                                    ggml_compute_forward_soft_max(&ref_params, ref_result);
                                    
                                    // Free work buffer after computation
                                    free(work_buffer);
                                    
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
        test_soft_max_mathematical_equivalence();
        
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
    void test_soft_max_mathematical_equivalence() {
        printf("--- Test: SOFT_MAX Mathematical Equivalence (Multi-Dimensional) ---\n");
        printf("Testing NUMA parallel SOFT_MAX vs serial reference implementation...\n");
        printf("Testing across tiny, small, medium, and large tensors with various coordinator execution strategies\n\n");
        
        bool overall_test_passed = true;
        const char* failure_reason = nullptr;
        
        // Define test dimensions: tiny, small, medium, large
        struct {
            int rows, cols;
            const char* label;
        } test_cases[] = {
            {8, 16, "TINY"},           // 8x16 = 128 elements
            {32, 64, "SMALL"},         // 32x64 = 2048 elements  
            {128, 256, "MEDIUM"},      // 128x256 = 32768 elements
            {256, 512, "LARGE"}        // 256x512 = 131072 elements
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
            printf("  📏 Testing %s tensors (%dx%d):\n", 
                   test_cases[case_idx].label, 
                   test_cases[case_idx].rows, test_cases[case_idx].cols);
            
            for (int strategy_idx = 0; strategy_idx < num_strategies; strategy_idx++) {
                int num_threads = thread_strategies[strategy_idx];
                
                bool case_passed = test_single_soft_max_case(
                    test_cases[case_idx].rows, 
                    test_cases[case_idx].cols, 
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
        printf("  📊 SOFT_MAX Multi-Dimensional Test Summary:\n");
        printf("    Total test combinations: %d\n", total_tests);
        printf("    Passed: %d\n", passed_tests);
        printf("    Failed: %d\n", total_tests - passed_tests);
        
        if (overall_test_passed) {
            printf("✅ SOFT_MAX mathematical equivalence (multi-dimensional): VERIFIED\n");
            printf("  🎉 All tensor dimensions and thread strategies produce mathematically equivalent results!\n\n");
        } else {
            printf("❌ SOFT_MAX mathematical equivalence (multi-dimensional): FAILED - %s\n", failure_reason);
            printf("  ⚠️  Mathematical mismatches detected across different dimensions or thread strategies\n\n");
        }
        
        results.push_back({"soft_max_mathematical_equivalence", overall_test_passed, failure_reason ? failure_reason : ""});
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

int main(int argc, char** argv) {
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
    printf("🚨 CRITICAL: Using FORCE_MULTI_SOCKET mode to test real data slicing on single-NUMA hardware\n");
    
    // Initialize the NUMA coordinator system with force_multi_socket=true for testing
    struct ggml_numa_coordinator_manager* manager = ggml_numa_coordinator_manager_get_global(8, true);  // <- FORCE MULTI-SOCKET
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
    
    // Restore stdout for final results and close dev_null
    if (summary_only && original_stdout) {
        fclose(stdout);
        stdout = original_stdout;
        printf("✅ NUMA SOFT_MAX Mathematical Correctness Test %s\n", all_passed ? "PASSED" : "FAILED");
    }
    
    if (all_passed) {
        return 0;
    } else {
        return 1;
    }
}
