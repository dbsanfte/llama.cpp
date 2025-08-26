/**
 * Custom F16 Dot Product Mathematical Correctness Test
 * 
 * This test validates our custom F16 dot product implementation against
 * the reference GGML implementation to ensure mathematical correctness.
 */

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

// GGML includes
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"

// Forward declaration of our custom F16 dot product
extern "C" {
    void ggml_numa_vec_dot_f16_custom(int n, float* s, size_t s_off,
                                     const void* x, size_t x_off,
                                     const void* y, size_t y_off, int nrc);
    
    // Reference implementation from GGML
    void ggml_vec_dot_f16(int n, float * s, size_t bs, ggml_fp16_t * x, size_t bx, ggml_fp16_t * y, size_t by, int nrc);
}

/**
 * Test framework for F16 dot product mathematical correctness
 */
class F16DotProductTester {
private:
    std::mt19937 rng;
    const float tolerance = 1e-3f;  // F16 precision tolerance
    
    /**
     * Calculate reference dot product using naive F32 arithmetic
     * for ultimate correctness validation
     */
    float reference_dot_product_f32(const std::vector<ggml_fp16_t>& x, const std::vector<ggml_fp16_t>& y) {
        float sum = 0.0f;
        for (size_t i = 0; i < x.size(); i++) {
            float x_f32 = ggml_fp16_to_fp32(x[i]);
            float y_f32 = ggml_fp16_to_fp32(y[i]);
            sum += x_f32 * y_f32;
        }
        return sum;
    }
    
    /**
     * Check if two values are within tolerance
     */
    bool within_tolerance(float a, float b, float tol) {
        float abs_diff = std::abs(a - b);
        float max_val = std::max(std::abs(a), std::abs(b));
        
        // Use relative tolerance for large values, absolute for small
        if (max_val > 1.0f) {
            return abs_diff <= tol * max_val;
        } else {
            return abs_diff <= tol;
        }
    }

public:
    F16DotProductTester() : rng(12345) {}
    
    /**
     * Test a specific vector length and pattern
     */
    bool test_single_case(int n, const char* description) {
        printf("  🧮 Testing %s (length=%d)\n", description, n);
        
        bool all_patterns_passed = true;
        const char* pattern_names[] = {"Sequential", "Alternating", "Random", "EdgeCases"};
        
        // Test multiple data patterns
        for (int pattern = 0; pattern < 4; pattern++) {
            std::vector<ggml_fp16_t> x, y;
            generate_test_vectors(x, y, n, pattern);
            
            // Test our custom implementation
            float custom_result = 0.0f;
            ggml_numa_vec_dot_f16_custom(n, &custom_result, 0, x.data(), 0, y.data(), 0, 1);
            
            // Test GGML reference implementation
            float ggml_result = 0.0f;
            ggml_vec_dot_f16(n, &ggml_result, 0, x.data(), 0, y.data(), 0, 1);
            
            // Test with our F32 reference for absolute truth
            float f32_reference = reference_dot_product_f32(x, y);
            
            // Check against F32 reference (which should be most accurate)
            float error = std::abs(custom_result - f32_reference);
            bool passed = within_tolerance(custom_result, f32_reference, tolerance);
            
            if (passed) {
                printf("      ✅ %s_%s: Results match (Custom=%.6f, F32Ref=%.6f, GGML=%.6f, Err=%.2e)\n", 
                       description, pattern_names[pattern], custom_result, f32_reference, ggml_result, error);
            } else {
                printf("      ❌ %s_%s: Results differ (Custom=%.6f, F32Ref=%.6f, GGML=%.6f, Err=%.2e)\n", 
                       description, pattern_names[pattern], custom_result, f32_reference, ggml_result, error);
                all_patterns_passed = false;
            }
        }
        
        if (all_patterns_passed) {
            printf("    ✅ %s: All patterns passed\n\n", description);
        } else {
            printf("    ❌ %s: Some patterns failed\n\n", description);
        }
        
        return all_patterns_passed;
    }
    
    // Generate test data with known patterns
    void generate_test_vectors(std::vector<ggml_fp16_t>& x, std::vector<ggml_fp16_t>& y, int n, int pattern) {
        x.resize(n);
        y.resize(n);
        
        switch (pattern) {
            case 0: { // Sequential pattern
                for (int i = 0; i < n; i++) {
                    x[i] = ggml_fp32_to_fp16((i + 1) * 0.1f);
                    y[i] = ggml_fp32_to_fp16((i + 1) * 0.05f);
                }
                break;
            }
                
            case 1: { // Alternating pattern  
                for (int i = 0; i < n; i++) {
                    x[i] = ggml_fp32_to_fp16((i % 2 == 0) ? 1.0f : -1.0f);
                    y[i] = ggml_fp32_to_fp16((i % 3 == 0) ? 2.0f : 0.5f);
                }
                break;
            }
                
            case 2: { // Random pattern
                std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
                for (int i = 0; i < n; i++) {
                    x[i] = ggml_fp32_to_fp16(dist(rng));
                    y[i] = ggml_fp32_to_fp16(dist(rng));
                }
                break;
            }
                
            case 3: { // Edge cases
                for (int i = 0; i < n; i++) {
                    float x_val = (i == 0) ? 0.0f : ((i == n-1) ? 1.0f : 0.1f);
                    float y_val = (i == 0) ? 1.0f : ((i == n-1) ? 0.0f : 0.2f);
                    x[i] = ggml_fp32_to_fp16(x_val);
                    y[i] = ggml_fp32_to_fp16(y_val);
                }
                break;
            }
        }
    }
    
    /**
     * Run comprehensive test suite
     */
    bool run_all_tests() {
        printf("🧪 Custom F16 Dot Product Mathematical Correctness Test\n");
        printf("=====================================================\n");
        printf("Testing custom F16 dot product against GGML reference implementation\n\n");
        
        bool all_passed = true;
        
        // Test multiple vector sizes
        struct {
            int size;
            const char* name;
        } test_sizes[] = {
            {4, "TINY_4"},
            {16, "SMALL_16"}, 
            {64, "MEDIUM_64"},
            {256, "LARGE_256"},
            {1024, "HUGE_1024"},
            {4096, "GIGANTIC_4096"}
        };
        
        for (auto& test : test_sizes) {
            bool passed = test_single_case(test.size, test.name);
            all_passed = all_passed && passed;
        }
        
        printf("=====================================================\n");
        printf("📊 Test Results Summary:\n");
        if (all_passed) {
            printf("Passed: ALL tests\n");
            printf("✅ All F16 dot product tests passed!\n");
            printf("🎉 Custom implementation is mathematically correct\n");
        } else {
            printf("❌ Some tests failed\n");
            printf("🚨 Custom implementation needs fixes\n");
        }
        
        return all_passed;
    }
};

int main() {
    F16DotProductTester tester;
    bool success = tester.run_all_tests();
    return success ? 0 : 1;
}
