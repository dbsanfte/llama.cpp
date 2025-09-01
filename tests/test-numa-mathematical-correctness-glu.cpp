#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

// Include implementation headers for ggml_compute_params
#include "../ggml/src/ggml-cpu/ggml-cpu-impl.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <thread>

// Import the NUMA executor function (needed for direct NUMA testing)
extern "C" {
    enum ggml_status ggml_numa_executor_execute_tensor(struct ggml_tensor * tensor, struct ggml_cplan * cplan);
    void ggml_compute_forward_glu(const struct ggml_compute_params * params, struct ggml_tensor * dst);
}

// Test configuration
static const float TOLERANCE = 1e-5f;

// Helper function: Fill tensor with random data
void fill_random_f32(struct ggml_tensor * tensor, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    float* data = (float*)ggml_get_data(tensor);
    int64_t total_elements = ggml_nelements(tensor);
    
    for (int64_t i = 0; i < total_elements; i++) {
        data[i] = dist(rng);
    }
}

// Helper function: Compare float arrays
bool compare_float_arrays(const float* numa_data, const float* ref_data, size_t count, const char* op_name) {
    const float tolerance = TOLERANCE;
    size_t failed_comparisons = 0;
    
    for (size_t i = 0; i < count; i++) {
        float diff = fabsf(numa_data[i] - ref_data[i]);
        if (diff > tolerance) {
            failed_comparisons++;
            if (failed_comparisons <= 5) { // Only show first 5 failures
                printf("      🔍 Mismatch at [%zu]: NUMA=%.6f, REF=%.6f, diff=%.6f (tol=%.6f)\n", 
                       i, numa_data[i], ref_data[i], diff, tolerance);
            }
        }
    }
    
    if (failed_comparisons > 0) {
        printf("      ❌ %s: %zu/%zu elements failed (%.2f%% accuracy)\n", 
               op_name, failed_comparisons, count, 
               100.0 * (count - failed_comparisons) / count);
        return false;
    }
    
    return true;
}

// Test GLU operation for specific size and thread count
bool test_single_GLU_case(int dim1, int dim2, int dim3, int num_threads, const char* size_label, enum ggml_glu_op glu_op) {
    // Ensure the last dimension is even for GLU (splits into two halves)
    if (dim1 % 2 != 0) dim1++;
    
    int64_t total_elements = (int64_t)dim1 * dim2 * dim3;
    int64_t output_elements = total_elements / 2; // GLU outputs half the input size
    
    printf("        Testing %s GLU: %dx%dx%d (%ld elements) with %d threads\n", 
           ggml_glu_op_name(glu_op), dim1, dim2, dim3, total_elements, num_threads);
    
    // Create context with adequate memory
    struct ggml_init_params params = {};
    params.no_alloc = false;
    params.mem_size = std::max((size_t)(512 * 1024 * 1024), (size_t)(dim1 * dim2 * dim3) * sizeof(float) * 8);
    params.mem_buffer = nullptr;
    
    struct ggml_context* test_ctx = ggml_init(params);
    if (!test_ctx) {
        printf("      ❌ Failed to create test context for %s\n", size_label);
        return false;
    }
    
    bool case_passed = false;
    
    // Create input tensor for GLU operation (single input with even last dimension)
    struct ggml_tensor* input = ggml_new_tensor_3d(test_ctx, GGML_TYPE_F32, dim1, dim2, dim3);
    
    if (!input) {
        printf("      ❌ Failed to create input tensor for %s\n", size_label);
        ggml_free(test_ctx);
        return false;
    }
    
    // Fill with random data
    std::mt19937 rng(42); // Fixed seed for reproducibility
    fill_random_f32(input, rng);
    
    // Create NUMA computation tensor  
    struct ggml_tensor* numa_result = ggml_glu(test_ctx, input, glu_op, false);
    
    // Setup NUMA execution
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
    
    // Create reference computation using serial execution
    struct ggml_tensor* ref_result = ggml_glu(test_ctx, input, glu_op, false);
    struct ggml_compute_params ref_params;
    ref_params.ith = 0;
    ref_params.nth = 1;
    ref_params.wsize = 0;
    ref_params.wdata = nullptr;
    ref_params.threadpool = nullptr;
    
    ggml_compute_forward_glu(&ref_params, ref_result);
    
    // Compare results
    float* numa_data = (float*)ggml_get_data(numa_result);
    float* ref_data = (float*)ggml_get_data(ref_result);
    case_passed = compare_float_arrays(numa_data, ref_data, output_elements, "GLU");
    
    if (case_passed) {
        printf("      ✅ %s GLU case passed (threads=%d)\n", size_label, num_threads);
    } else {
        printf("      ❌ %s GLU case failed (threads=%d)\n", size_label, num_threads);
    }
    
    ggml_free(test_ctx);
    return case_passed;
}

void test_GLU_mathematical_equivalence() {
    printf("--- Test: GLU Mathematical Equivalence (Multi-Dimensional) ---\n");
    printf("Testing NUMA parallel GLU vs serial reference implementation...\n");
    printf("Testing across various tensor sizes with different coordinator execution strategies\n\n");
    
    bool overall_test_passed = true;
    const char* failure_reason = nullptr;
    
    // Define test dimensions appropriate for GLU operation (single input operation)
    struct {
        int dim1, dim2, dim3;
        const char* label;
    } test_cases[] = {
        {16, 8, 4, "TINY"},           // Small tensors for basic verification - 512 elements (256 output)
        {128, 64, 8, "SMALL"},        // Medium tensors - 65,536 elements (32,768 output)
        {256, 64, 32, "MEDIUM"},      // Larger tensors - 524,288 elements (262,144 output)
        {512, 128, 64, "LARGE"}       // Large tensors for stress testing - 4,194,304 elements (2,097,152 output)
    };
    
    // GLU operation variants to test
    enum ggml_glu_op glu_ops[] = {
        GGML_GLU_OP_REGLU,
        GGML_GLU_OP_SWIGLU,
        GGML_GLU_OP_GEGLU
    };
    
    // Define coordinator execution strategies (various thread counts)
    int thread_strategies[] = {1, 2, 4, 6, 8};
    int num_strategies = sizeof(thread_strategies) / sizeof(thread_strategies[0]);
    int num_test_cases = sizeof(test_cases) / sizeof(test_cases[0]);
    int num_glu_ops = sizeof(glu_ops) / sizeof(glu_ops[0]);
    
    printf("  🎯 Testing %d GLU operations with %d tensor dimensions and %d thread strategies (%d total test combinations)\n\n", 
           num_glu_ops, num_test_cases, num_strategies, num_glu_ops * num_test_cases * num_strategies);
    
    int total_tests = 0;
    int passed_tests = 0;
    
    // Test each GLU operation variant
    for (int op_idx = 0; op_idx < num_glu_ops; op_idx++) {
        enum ggml_glu_op current_op = glu_ops[op_idx];
        printf("  🧠 Testing %s operation:\n", ggml_glu_op_name(current_op));
        
        // Test each tensor dimension with each thread strategy
        for (int case_idx = 0; case_idx < num_test_cases; case_idx++) {
            printf("    📏 Testing %s tensors (%dx%dx%d):\n", 
                   test_cases[case_idx].label, test_cases[case_idx].dim1, 
                   test_cases[case_idx].dim2, test_cases[case_idx].dim3);
            
            for (int strategy_idx = 0; strategy_idx < num_strategies; strategy_idx++) {
                total_tests++;
                bool test_passed = test_single_GLU_case(
                    test_cases[case_idx].dim1, test_cases[case_idx].dim2, test_cases[case_idx].dim3,
                    thread_strategies[strategy_idx], test_cases[case_idx].label, current_op
                );
                
                if (test_passed) {
                    passed_tests++;
                } else {
                    overall_test_passed = false;
                    if (!failure_reason) {
                        failure_reason = "GLU mathematical equivalence test failed";
                    }
                }
            }
            printf("\n");
        }
        printf("\n");
    }
    
    printf("  📊 GLU Mathematical Equivalence Results:\n");
    printf("     Total tests: %d\n", total_tests);
    printf("     Passed: %d\n", passed_tests);
    printf("     Failed: %d\n", total_tests - passed_tests);
    printf("     Success rate: %.1f%%\n\n", 100.0 * passed_tests / total_tests);
    
    if (overall_test_passed) {
        printf("  🎉 GLU Mathematical Equivalence: ALL TESTS PASSED\n");
    } else {
        printf("  ❌ GLU Mathematical Equivalence: TESTS FAILED (%s)\n", failure_reason);
    }
}

int main() {
    printf("🧪 NUMA GLU Mathematical Correctness Test\n");
    printf("==========================================\n\n");

    test_GLU_mathematical_equivalence();

    printf("\n🏁 GLU Test Complete!\n");
    
    return 0;
}
