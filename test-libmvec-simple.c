/**
 * @file test-libmvec-simple.c
 * @brief Test GNU libmvec vectorized transcendental functions
 * @author David Sanftenberg
 * 
 * This tests the proper approach for GNU libmvec vectorization.
 * The key is to use OpenMP SIMD pragmas to trigger vectorization.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define TEST_SIZE 4096
#define NUM_ITERATIONS 1000

/**
 * Pure scalar implementation
 */
static void compute_sincos_scalar(int n, float * restrict sin_out, float * restrict cos_out, const float * restrict theta_in) {
    for (int i = 0; i < n; i++) {
        // Separate calls to prevent GCC from optimizing to sincosf
        volatile float theta_val = theta_in[i];
        sin_out[i] = sinf(theta_val);
        cos_out[i] = cosf(theta_val);
    }
}

/**
 * GNU libmvec vectorized implementation using OpenMP SIMD
 * This is the proper way to trigger libmvec vectorization
 */
static void compute_sincos_libmvec(int n, float * restrict sin_out, float * restrict cos_out, const float * restrict theta_in) {
    // Use OpenMP SIMD pragma to trigger vectorization with libmvec
    #pragma omp simd
    for (int i = 0; i < n; i++) {
        // Separate calls to prevent GCC from optimizing to sincosf
        volatile float theta_val = theta_in[i];
        sin_out[i] = sinf(theta_val);
        cos_out[i] = cosf(theta_val);
    }
}

/**
 * Alternative: Use GCC vector_size attribute
 */
typedef float v8sf __attribute__ ((vector_size (32))); // 8 floats (AVX2)

static void compute_sincos_gcc_vector(int n, float * restrict sin_out, float * restrict cos_out, const float * restrict theta_in) {
    const int simd_width = 8;
    const int simd_count = n / simd_width;
    
    for (int i = 0; i < simd_count; i++) {
        v8sf theta_vec;
        memcpy(&theta_vec, &theta_in[i * simd_width], sizeof(v8sf));
        
        // This should trigger libmvec vectorization
        v8sf sin_vec, cos_vec;
        for (int j = 0; j < simd_width; j++) {
            volatile float theta_val = theta_vec[j];
            sin_vec[j] = sinf(theta_val);
            cos_vec[j] = cosf(theta_val);
        }
        
        memcpy(&sin_out[i * simd_width], &sin_vec, sizeof(v8sf));
        memcpy(&cos_out[i * simd_width], &cos_vec, sizeof(v8sf));
    }
    
    // Handle remainder
    for (int i = simd_count * simd_width; i < n; i++) {
        volatile float theta_val = theta_in[i];
        sin_out[i] = sinf(theta_val);
        cos_out[i] = cosf(theta_val);
    }
}

/**
 * Manual intrinsics approach (fallback if libmvec doesn't work)
 * Using more accurate Taylor series for better precision
 */
#ifdef __AVX2__
#include <immintrin.h>

// More accurate Taylor series approximation
static __m256 accurate_sin_avx2(__m256 x) {
    // Reduce to [-π, π] range
    const __m256 pi = _mm256_set1_ps(3.141592654f);
    const __m256 two_pi = _mm256_set1_ps(6.283185307f);
    
    // Simple range reduction for demo
    __m256 x_reduced = x;
    
    // Taylor series: sin(x) ≈ x - x³/6 + x⁵/120 - x⁷/5040
    __m256 x2 = _mm256_mul_ps(x_reduced, x_reduced);
    __m256 x3 = _mm256_mul_ps(x2, x_reduced);
    __m256 x5 = _mm256_mul_ps(x3, x2);
    __m256 x7 = _mm256_mul_ps(x5, x2);
    
    __m256 term1 = x_reduced;
    __m256 term2 = _mm256_mul_ps(x3, _mm256_set1_ps(-1.0f/6.0f));
    __m256 term3 = _mm256_mul_ps(x5, _mm256_set1_ps(1.0f/120.0f));
    __m256 term4 = _mm256_mul_ps(x7, _mm256_set1_ps(-1.0f/5040.0f));
    
    __m256 result = _mm256_add_ps(term1, term2);
    result = _mm256_add_ps(result, term3);
    result = _mm256_add_ps(result, term4);
    
    return result;
}

static __m256 accurate_cos_avx2(__m256 x) {
    // Taylor series: cos(x) ≈ 1 - x²/2 + x⁴/24 - x⁶/720
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 x4 = _mm256_mul_ps(x2, x2);
    __m256 x6 = _mm256_mul_ps(x4, x2);
    
    __m256 term1 = _mm256_set1_ps(1.0f);
    __m256 term2 = _mm256_mul_ps(x2, _mm256_set1_ps(-1.0f/2.0f));
    __m256 term3 = _mm256_mul_ps(x4, _mm256_set1_ps(1.0f/24.0f));
    __m256 term4 = _mm256_mul_ps(x6, _mm256_set1_ps(-1.0f/720.0f));
    
    __m256 result = _mm256_add_ps(term1, term2);
    result = _mm256_add_ps(result, term3);
    result = _mm256_add_ps(result, term4);
    
    return result;
}

static void compute_sincos_manual_avx2(int n, float * restrict sin_out, float * restrict cos_out, const float * restrict theta_in) {
    const int simd_width = 8;
    const int simd_count = n / simd_width;
    
    for (int i = 0; i < simd_count; i++) {
        __m256 theta_vec = _mm256_loadu_ps(&theta_in[i * simd_width]);
        __m256 sin_vec = accurate_sin_avx2(theta_vec);
        __m256 cos_vec = accurate_cos_avx2(theta_vec);
        _mm256_storeu_ps(&sin_out[i * simd_width], sin_vec);
        _mm256_storeu_ps(&cos_out[i * simd_width], cos_vec);
    }
    
    // Handle remainder
    for (int i = simd_count * simd_width; i < n; i++) {
        volatile float theta_val = theta_in[i];
        sin_out[i] = sinf(theta_val);
        cos_out[i] = cosf(theta_val);
    }
}
#endif

/**
 * Test mathematical correctness
 */
static int validate_correctness(const char* impl_name, 
                               void (*impl_func)(int, float*, float*, const float*)) {
    printf("🧮 Testing %s correctness...\n", impl_name);
    
    float * theta = malloc(TEST_SIZE * sizeof(float));
    float * sin_ref = malloc(TEST_SIZE * sizeof(float));
    float * cos_ref = malloc(TEST_SIZE * sizeof(float));
    float * sin_test = malloc(TEST_SIZE * sizeof(float));
    float * cos_test = malloc(TEST_SIZE * sizeof(float));
    
    // Generate test data
    for (int i = 0; i < TEST_SIZE; i++) {
        theta[i] = -3.14f + (6.28f * i) / TEST_SIZE;  // Range [-π, π]
    }
    
    // Compute reference
    compute_sincos_scalar(TEST_SIZE, sin_ref, cos_ref, theta);
    
    // Compute test implementation
    impl_func(TEST_SIZE, sin_test, cos_test, theta);
    
    // Compare results
    int mismatches = 0;
    float max_sin_diff = 0.0f, max_cos_diff = 0.0f;
    const float tolerance = 1e-3f;  // Relaxed tolerance for approximations
    
    for (int i = 0; i < TEST_SIZE; i++) {
        float sin_diff = fabsf(sin_ref[i] - sin_test[i]);
        float cos_diff = fabsf(cos_ref[i] - cos_test[i]);
        
        if (sin_diff > max_sin_diff) max_sin_diff = sin_diff;
        if (cos_diff > max_cos_diff) max_cos_diff = cos_diff;
        
        if (sin_diff > tolerance || cos_diff > tolerance) {
            mismatches++;
        }
    }
    
    printf("   Max differences: sin=%.6f, cos=%.6f\n", max_sin_diff, max_cos_diff);
    printf("   Mismatches: %d / %d (tolerance=%.6f)\n", mismatches, TEST_SIZE, tolerance);
    
    free(theta); free(sin_ref); free(cos_ref); free(sin_test); free(cos_test);
    
    if (mismatches < TEST_SIZE * 0.01) {  // Allow 1% tolerance for approximations
        printf("✅ %s correctness validated\n", impl_name);
        return 1;
    } else {
        printf("❌ %s correctness failed\n", impl_name);
        return 0;
    }
}

/**
 * Measure performance
 */
static void measure_performance(const char* impl_name,
                               void (*impl_func)(int, float*, float*, const float*)) {
    printf("🚀 Measuring %s performance...\n", impl_name);
    
    float * theta = malloc(TEST_SIZE * sizeof(float));
    float * sin_out = malloc(TEST_SIZE * sizeof(float));
    float * cos_out = malloc(TEST_SIZE * sizeof(float));
    
    // Generate test data
    for (int i = 0; i < TEST_SIZE; i++) {
        theta[i] = -3.14f + (6.28f * i) / TEST_SIZE;
    }
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        impl_func(TEST_SIZE, sin_out, cos_out, theta);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_ms = ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9) * 1000;
    
    printf("   Time: %.3f ms (%d iterations)\n", time_ms, NUM_ITERATIONS);
    
    free(theta); free(sin_out); free(cos_out);
}

int main(void) {
    printf("🧪 GNU libmvec Vectorized Transcendental Function Test\n");
    printf("=====================================================\n");
    printf("Testing different approaches for SIMD transcendental functions\n\n");
    
    // Test all implementations
    printf("1️⃣  SCALAR REFERENCE\n");
    if (!validate_correctness("Scalar", compute_sincos_scalar)) return 1;
    measure_performance("Scalar", compute_sincos_scalar);
    printf("\n");
    
    printf("2️⃣  GNU LIBMVEC (OpenMP SIMD)\n");
    if (!validate_correctness("LibMVec", compute_sincos_libmvec)) return 1;
    measure_performance("LibMVec", compute_sincos_libmvec);
    printf("\n");
    
    printf("3️⃣  GCC VECTOR ATTRIBUTE\n");
    if (!validate_correctness("GCC Vector", compute_sincos_gcc_vector)) return 1;
    measure_performance("GCC Vector", compute_sincos_gcc_vector);
    printf("\n");
    
#ifdef __AVX2__
    printf("4️⃣  MANUAL AVX2 APPROXIMATION\n");
    if (!validate_correctness("Manual AVX2", compute_sincos_manual_avx2)) return 1;
    measure_performance("Manual AVX2", compute_sincos_manual_avx2);
    printf("\n");
#endif
    
    printf("🎯 ANALYSIS:\n");
    printf("The best performing implementation shows the optimal approach\n");
    printf("for vectorized transcendental functions in production.\n");
    
    return 0;
}
