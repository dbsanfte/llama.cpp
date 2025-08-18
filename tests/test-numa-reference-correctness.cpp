#include "numa-test-utils.h"
#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-backend.h>
#include <ggml-quants.h>
#include <ggml-numa-operation-dispatch.h>
#include <ggml-numa-coordinator.h>
#include <ggml-cpu-impl.h>
#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <cstring>

namespace ReferenceOperations {
    
    // Reference matrix multiplication using standard C math
    // Computes C = A × B where A is MxK, B is KxN, C is MxN
    void reference_mat_mul_f32(
        const float* A, int M, int K,
        const float* B, int N,
        float* C
    ) {
        // Initialize output to zero
        for (int i = 0; i < M * N; i++) {
            C[i] = 0.0f;
        }
        
        // Standard matrix multiplication: C[i][j] = sum(A[i][k] * B[k][j])
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }
    
    // Reference quantized to F32 matrix multiplication
    // First dequantizes A, then performs F32 multiplication
    void reference_mat_mul_q8_0_f32(
        const void* A_quantized, int M, int K,
        const float* B, int N,
        float* C
    ) {
        // Dequantize A to F32
        std::vector<float> A_f32(M * K);
        
        // Dequantize row by row using ggml's dequantization function
        const block_q8_0* blocks = (const block_q8_0*)A_quantized;
        int blocks_per_row = K / QK8_0;  // QK8_0 is the block size for Q8_0
        
        for (int row = 0; row < M; row++) {
            const block_q8_0* row_blocks = blocks + row * blocks_per_row;
            float* row_output = A_f32.data() + row * K;
            dequantize_row_q8_0(row_blocks, row_output, K);
        }
        
        // Now perform standard F32 matrix multiplication
        reference_mat_mul_f32(A_f32.data(), M, K, B, N, C);
    }
}

struct TestCase {
    int M, K, N;
    ggml_type src0_type;
    ggml_type src1_type;
    const char* description;
};

class NumaReferenceCorrectnessTest {
private:
    ggml_context* ctx;
    std::vector<TestCase> test_cases;
    bool verbose;
    
public:
    NumaReferenceCorrectnessTest(bool verbose = false) : ctx(nullptr), verbose(verbose) {
        // Initialize test cases with various tensor types and sizes
        test_cases = {
            {16, 32, 24, GGML_TYPE_F32, GGML_TYPE_F32, "Small F32×F32"},
            {64, 128, 96, GGML_TYPE_F32, GGML_TYPE_F32, "Medium F32×F32"},
            {16, 32, 24, GGML_TYPE_Q8_0, GGML_TYPE_F32, "Small Q8_0×F32"},
            {64, 128, 96, GGML_TYPE_Q8_0, GGML_TYPE_F32, "Medium Q8_0×F32"},
            {32, 64, 48, GGML_TYPE_Q4_0, GGML_TYPE_F32, "Medium Q4_0×F32"},
        };
        
        // Initialize ggml context
        struct ggml_init_params params;
        params.mem_size = 128 * 1024 * 1024; // 128MB
        params.mem_buffer = nullptr;
        params.no_alloc = false;
        ctx = ggml_init(params);
        assert(ctx != nullptr);
    }
    
    ~NumaReferenceCorrectnessTest() {
        if (ctx) {
            ggml_free(ctx);
        }
    }
    
    bool compare_results(const float* reference, const float* numa_result, int size, 
                        const char* test_name, float tolerance = 1e-4f) {
        bool all_match = true;
        float max_diff = 0.0f;
        float avg_diff = 0.0f;
        int mismatch_count = 0;
        
        for (int i = 0; i < size; i++) {
            float diff = std::abs(reference[i] - numa_result[i]);
            avg_diff += diff;
            max_diff = std::max(max_diff, diff);
            
            if (diff > tolerance) {
                all_match = false;
                mismatch_count++;
                
                if (verbose && mismatch_count <= 10) { // Show first 10 mismatches
                    std::cout << "  Mismatch at [" << i << "]: reference=" << reference[i] 
                              << ", numa=" << numa_result[i] << ", diff=" << diff << std::endl;
                }
            }
        }
        
        avg_diff /= size;
        
        std::cout << test_name << ": ";
        if (all_match) {
            std::cout << "✅ PASS (max_diff=" << std::fixed << std::setprecision(8) << max_diff 
                      << ", avg_diff=" << avg_diff << ")" << std::endl;
        } else {
            std::cout << "❌ FAIL (mismatches=" << mismatch_count << "/" << size 
                      << ", max_diff=" << max_diff << ", avg_diff=" << avg_diff << ")" << std::endl;
        }
        
        return all_match;
    }
    
    bool test_matrix_multiplication(const TestCase& test_case) {
        if (verbose) {
            std::cout << "\n--- Testing " << test_case.description << " ---" << std::endl;
            std::cout << "Matrix dimensions: " << test_case.M << "×" << test_case.K 
                      << " × " << test_case.K << "×" << test_case.N << std::endl;
        }
        
        // Create tensors - for mul_mat: src0 is MxK, src1 is KxN
        // Note: ggml tensors are stored as [width, height, ...] = [K, M] for src0 and [N, K] for src1
        ggml_tensor* src0 = ggml_new_tensor_2d(ctx, test_case.src0_type, test_case.K, test_case.M);
        ggml_tensor* src1 = ggml_new_tensor_2d(ctx, test_case.src1_type, test_case.K, test_case.N);
        
        // Generate identical input data for both reference and NUMA tests
        std::vector<float> src0_f32_data(test_case.M * test_case.K);
        std::vector<float> src1_f32_data(test_case.K * test_case.N);
        
        // Initialize with deterministic data that should produce meaningful results
        NumaTestUtils::init_f32_test_data(src0_f32_data.data(), src0_f32_data.size(), 1.0f, 0.01f);
        NumaTestUtils::init_f32_test_data(src1_f32_data.data(), src1_f32_data.size(), 0.5f, 0.02f);
        
        // Copy src1 data (always F32)
        memcpy(ggml_get_data(src1), src1_f32_data.data(), src1_f32_data.size() * sizeof(float));
        
        // Handle src0 data (may need quantization)
        if (test_case.src0_type == GGML_TYPE_F32) {
            memcpy(ggml_get_data(src0), src0_f32_data.data(), src0_f32_data.size() * sizeof(float));
        } else {
            // Quantize the F32 data
            ggml_quantize_chunk(test_case.src0_type, src0_f32_data.data(), ggml_get_data(src0), 
                               0, test_case.M, test_case.K, nullptr);
        }
        
        // Compute reference result using standard C math
        std::vector<float> reference_result(test_case.M * test_case.N);
        
        if (test_case.src0_type == GGML_TYPE_F32) {
            ReferenceOperations::reference_mat_mul_f32(
                src0_f32_data.data(), test_case.M, test_case.K,
                src1_f32_data.data(), test_case.N,
                reference_result.data()
            );
        } else {
            // For quantized types, use reference dequantization + multiplication
            if (test_case.src0_type == GGML_TYPE_Q8_0) {
                ReferenceOperations::reference_mat_mul_q8_0_f32(
                    ggml_get_data(src0), test_case.M, test_case.K,
                    src1_f32_data.data(), test_case.N,
                    reference_result.data()
                );
            } else if (test_case.src0_type == GGML_TYPE_Q4_0) {
                // For Q4_0, use general dequantization approach
                std::vector<float> src0_dequantized(test_case.M * test_case.K);
                const struct ggml_type_traits* traits = ggml_get_type_traits(GGML_TYPE_Q4_0);
                
                // Dequantize Q4_0 data
                const block_q4_0* blocks = (const block_q4_0*)ggml_get_data(src0);
                int blocks_per_row = test_case.K / QK4_0;
                
                for (int row = 0; row < test_case.M; row++) {
                    const block_q4_0* row_blocks = blocks + row * blocks_per_row;
                    float* row_output = src0_dequantized.data() + row * test_case.K;
                    dequantize_row_q4_0(row_blocks, row_output, test_case.K);
                }
                
                // Now perform F32 matrix multiplication
                ReferenceOperations::reference_mat_mul_f32(
                    src0_dequantized.data(), test_case.M, test_case.K,
                    src1_f32_data.data(), test_case.N,
                    reference_result.data()
                );
            } else {
                std::cout << "❌ ERROR: Quantization type " << test_case.src0_type 
                          << " not implemented in reference" << std::endl;
                return false;
            }
        }
        
        // Compute NUMA result using NUMA intercept
        ggml_tensor* mul_result = ggml_mul_mat(ctx, src0, src1);
        
        // Use NUMA-aware computation with multiple threads
        struct ggml_compute_params numa_params = {
            0,               // ith - Main thread
            4,               // nth - Use 4 threads to test NUMA coordination
            0,               // wsize
            nullptr,         // wdata
            nullptr          // threadpool
        };
        
        // Execute via NUMA intercept to trigger NUMA dispatch
        enum ggml_status dispatch_result = ggml_numa_intercept_operation(mul_result, &numa_params);
        
        if (dispatch_result != GGML_STATUS_SUCCESS) {
            std::cout << "❌ ERROR: NUMA dispatcher execution failed for " 
                      << test_case.description << " (status=" << dispatch_result << ")" << std::endl;
            return false;
        }
        
        // Compare results
        float* numa_result = (float*)ggml_get_data(mul_result);
        bool success = compare_results(reference_result.data(), numa_result, 
                                     test_case.M * test_case.N, test_case.description);
        
        return success;
    }
    
    bool run_all_tests() {
        std::cout << "🧮 NUMA Reference Correctness Test" << std::endl;
        std::cout << "Comparing NUMA operations against known-good reference implementations" << std::endl;
        std::cout << "========================================" << std::endl;
        
        bool all_passed = true;
        int passed = 0;
        int total = test_cases.size();
        
        for (const auto& test_case : test_cases) {
            bool result = test_matrix_multiplication(test_case);
            if (result) {
                passed++;
            } else {
                all_passed = false;
            }
        }
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "Results: " << passed << "/" << total << " tests passed" << std::endl;
        
        if (all_passed) {
            std::cout << "🎉 All reference correctness tests PASSED!" << std::endl;
            std::cout << "NUMA operations produce mathematically correct results." << std::endl;
        } else {
            std::cout << "❌ Some tests FAILED!" << std::endl;
            std::cout << "NUMA operations produce incorrect mathematical results." << std::endl;
            std::cout << "This indicates bugs in the NUMA operation implementations." << std::endl;
        }
        
        return all_passed;
    }
};

int main(int argc, char** argv) {
    bool verbose = false;
    if (argc > 1 && std::string(argv[1]) == "--verbose") {
        verbose = true;
    }
    
    NumaReferenceCorrectnessTest test(verbose);
    bool success = test.run_all_tests();
    
    return success ? 0 : 1;
}
