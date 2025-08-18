#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-numa-operation-dispatch.h"
#include "ggml-numa-coordinator.h"
#include "ggml-cpu-impl.h"
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

// Tensor type combination for testing
struct TensorTypePair {
    ggml_type src0_type;
    ggml_type src1_type;
    const char* description;
    bool should_work; // Expected to work or known to be problematic
};

class NumaTensorCorruptionDetectionSuite {
private:
    std::vector<TestResult> results;
    
    // Define tensor type combinations to test
    std::vector<TensorTypePair> get_test_combinations() {
        return {
            {GGML_TYPE_F32,  GGML_TYPE_F32,  "F32×F32 (baseline)",     true},
            {GGML_TYPE_F16,  GGML_TYPE_F32,  "F16×F32 (common)",       true},
            {GGML_TYPE_Q8_0, GGML_TYPE_F32,  "Q8_0×F32 (CHECK_CORRUPTION)", true}, // CHECK: Real corruption issue we need to catch
            {GGML_TYPE_Q4_0, GGML_TYPE_F32,  "Q4_0×F32 (quantized)",   true},
            {GGML_TYPE_Q4_1, GGML_TYPE_F32,  "Q4_1×F32 (quantized)",   true}
        };
    }
    
    // Test MUL_MAT operation with specific tensor types using the dispatcher
    bool test_mul_mat_tensor_types(const TensorTypePair& types, int M, int K, int N) {
        printf("    🔍 Testing MUL_MAT: %s, dims=%dx%d*%dx%d\n", types.description, M, K, K, N);
        
        // Create context with sufficient memory
        struct ggml_init_params params = {};
        params.mem_size = 256 * 1024 * 1024; // 256MB should be enough for test tensors
        params.mem_buffer = nullptr;
        params.no_alloc = false;
        
        struct ggml_context* ctx = ggml_init(params);
        if (!ctx) {
            printf("        ❌ Failed to create context for %s\n", types.description);
            return false;
        }
        
        bool test_passed = false;
        
        try {
            // Create tensors with specified types
            struct ggml_tensor* src0 = ggml_new_tensor_2d(ctx, types.src0_type, K, M);
            struct ggml_tensor* src1 = ggml_new_tensor_2d(ctx, types.src1_type, K, N);
            struct ggml_tensor* dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, M, N);
            
            if (!src0 || !src1 || !dst) {
                printf("        ❌ Failed to create tensors for %s\n", types.description);
                goto cleanup;
            }
            
            // Initialize input tensors with deterministic data that should produce non-zero results
            if (types.src0_type == GGML_TYPE_F32) {
                NumaTestUtils::init_f32_test_data((float*)ggml_get_data(src0), ggml_nelements(src0), 1.0f, 0.01f);
            } else {
                // For quantized types, create F32 data first, then quantize
                std::vector<float> temp_data(ggml_nelements(src0));
                NumaTestUtils::init_f32_test_data(temp_data.data(), temp_data.size(), 1.0f, 0.01f);
                
                // Use the correct quantize function signature
                size_t quantized_size = ggml_row_size(types.src0_type, K);
                ggml_quantize_chunk(types.src0_type, temp_data.data(), ggml_get_data(src0), 
                                   0, M, K, nullptr);
            }
            
            // Initialize src1 (always F32 in our test cases) with values that ensure non-zero products
            NumaTestUtils::init_f32_test_data((float*)ggml_get_data(src1), ggml_nelements(src1), 0.5f, 0.02f);
            
            // Set up the operation
            dst->src[0] = src0;
            dst->src[1] = src1;
            dst->op = GGML_OP_MUL_MAT;
            
            // Try to execute through NUMA dispatcher
            printf("        🚀 Executing %s through NUMA dispatcher...\n", types.description);
            
            // Get the global manager (this should already be initialized)
            struct ggml_numa_coordinator_manager * mgr = ggml_numa_coordinator_manager_get_global(4, true);
            if (!mgr) {
                printf("        ❌ Failed to get NUMA coordinator manager\n");
                goto cleanup;
            }
            
            // Execute using the dispatcher (this is how the real system works)
            ggml_numa_work_context_t work_ctx = {};
            work_ctx.total_elements = ggml_nelements(dst);
            work_ctx.element_size = sizeof(float);
            work_ctx.n_dims = GGML_MAX_DIMS; // Use maximum for simplicity
            for (int i = 0; i < GGML_MAX_DIMS; i++) {
                work_ctx.ne[i] = dst->ne[i];
            }
            work_ctx.numa_nodes = 2;
            work_ctx.threads_per_node = 2;
            
            ggml_status status = ggml_numa_dispatch_operation(mgr, dst, &work_ctx);
            
            // Analyze results
            float* result_data = (float*)ggml_get_data(dst);
            size_t num_elements = ggml_nelements(dst);
            
            auto analysis = NumaTestUtils::analyze_tensor_corruption(result_data, num_elements, "MUL_MAT result");
            
            // Report results
            if (status == GGML_STATUS_SUCCESS) {
                if (!analysis.has_corruption) {
                    printf("        ✅ %s: Operation succeeded, no corruption detected\n", types.description);
                    printf("           Stats: mean=%.6f, variance=%.6f, elements=%zu\n", 
                           analysis.mean, analysis.variance, num_elements);
                    test_passed = true;
                } else {
                    printf("        🚨 %s: CORRUPTION DETECTED!\n", types.description);
                    NumaTestUtils::print_corruption_report(analysis, "MUL_MAT result");
                    printf("        ❌ This is a REAL mathematical correctness failure!\n");
                    test_passed = false; // Always fail on corruption
                }
            } else {
                printf("        ❌ %s: Operation failed with status %d\n", types.description, (int)status);
                test_passed = false; // Always fail on operation failure
            }
            
        } catch (...) {
            printf("        💥 %s: Exception during test execution\n", types.description);
            test_passed = false;
        }
        
cleanup:
        if (ctx) {
            ggml_free(ctx);
        }
        
        return test_passed;
    }
    
public:
    // Run comprehensive tensor type corruption detection tests
    void run_comprehensive_corruption_tests() {
        printf("\n🔬 NUMA Tensor Corruption Detection Test Suite\n");
        printf("================================================\n");
        printf("Testing various tensor type combinations for corruption detection\n\n");
        
        auto combinations = get_test_combinations();
        int passed = 0;
        int total = 0;
        
        // Test different matrix sizes
        std::vector<std::tuple<int, int, int, const char*>> test_sizes = {
            {16, 32, 16, "small"},
            {64, 128, 64, "medium"}
        };
        
        for (const auto& size_tuple : test_sizes) {
            int M, K, N;
            const char* size_label;
            std::tie(M, K, N, size_label) = size_tuple;
            
            printf("📏 Testing %s matrices (%dx%d * %dx%d)\n", size_label, M, K, K, N);
            printf("─────────────────────────────────────────\n");
            
            for (const auto& combo : combinations) {
                total++;
                std::string test_name = std::string(combo.description) + "_" + size_label;
                
                bool result = test_mul_mat_tensor_types(combo, M, K, N);
                
                TestResult test_result;
                test_result.test_name = test_name;
                test_result.passed = result;
                if (!result) {
                    test_result.failure_reason = "Corruption detected or unexpected failure";
                }
                results.push_back(test_result);
                
                if (result) {
                    passed++;
                }
                
                printf("\n");
            }
        }
        
        printf("\n📊 Tensor Corruption Detection Summary\n");
        printf("=====================================\n");
        printf("Total tests: %d\n", total);
        printf("Passed: %d\n", passed);
        printf("Failed: %d\n", total - passed);
        
        // Report specific failures
        if (passed < total) {
            printf("\n🚨 CRITICAL FAILURES DETECTED:\n");
            for (const auto& result : results) {
                if (!result.passed) {
                    printf("  ❌ %s: %s\n", result.test_name.c_str(), result.failure_reason.c_str());
                }
            }
            printf("\n💥 These are REAL mathematical correctness issues that must be fixed!\n");
        }
    }
    
    // Get overall test success
    bool all_tests_passed() const {
        for (const auto& result : results) {
            if (!result.passed) {
                return false;
            }
        }
        return true;
    }
};

int main() {
    printf("🧪 NUMA Tensor Corruption Detection Test\n");
    printf("========================================\n");
    printf("This test validates tensor operations across different type combinations\n");
    printf("and detects corruption issues like NaN/inf generation.\n");
    printf("🚨 ANY CORRUPTION DETECTED WILL CAUSE TEST FAILURE!\n");
    
    // Initialize NUMA coordinator for testing (use the global manager approach)
    printf("\n🔧 Initializing NUMA coordinator for testing...\n");
    
    // Get global manager (this will create it if it doesn't exist)
    struct ggml_numa_coordinator_manager * mgr = ggml_numa_coordinator_manager_get_global(4, true);
    if (!mgr) {
        printf("❌ Failed to initialize NUMA coordinator manager\n");
        return 1;
    }
    
    printf("✅ NUMA coordinator manager initialized\n");
    
    // Run the comprehensive corruption detection tests
    NumaTensorCorruptionDetectionSuite test_suite;
    test_suite.run_comprehensive_corruption_tests();
    
    // Return appropriate exit code
    bool success = test_suite.all_tests_passed();
    if (success) {
        printf("\n✅ Overall test result: PASSED\n");
        printf("🎉 All tensor operations completed without corruption!\n");
    } else {
        printf("\n❌ Overall test result: FAILED\n");
        printf("💥 CRITICAL: Mathematical correctness issues detected!\n");
        printf("🔧 These failures indicate real bugs that must be fixed.\n");
    }
    
    return success ? 0 : 1;
}
