/**
 * @file test-numa-libmvec-performance.c
 * @brief Test GNU libmvec vectorization performance vs scalar for transcendental functions
 * @author David Sanftenberg
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define ARRAY_SIZE 1000000
#define NUM_ITERATIONS 10

// Function pointers for testing both scalar and vectorized versions
typedef void (*transcendental_func_t)(int n, float* y, const float* x);

// Forward declarations for NUMA vector functions
void ggml_vec_numa_init(void);
void ggml_vec_sin_f32_numa(int n, float* y, const float* x);
void ggml_vec_cos_f32_numa(int n, float* y, const float* x);

// Scalar reference implementations (for comparison)
static void scalar_sin(int n, float* y, const float* x) {
    for (int i = 0; i < n; i++) {
        y[i] = sinf(x[i]);
    }
}

static void scalar_cos(int n, float* y, const float* x) {
    for (int i = 0; i < n; i++) {
        y[i] = cosf(x[i]);
    }
}

// Timing utilities
static double get_time_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double measure_function_performance(transcendental_func_t func, const float* input, float* output, int size) {
    double start_time = get_time_seconds();
    
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        func(size, output, input);
    }
    
    double end_time = get_time_seconds();
    return (end_time - start_time) / NUM_ITERATIONS;  // Average time per call
}

int main(int argc, char* argv[]) {
    printf("=== GNU libmvec Vectorization Performance Test ===\n\n");
    
    // Initialize NUMA vector system
    ggml_vec_numa_init();
    
    // Allocate test arrays
    float* input = (float*)malloc(ARRAY_SIZE * sizeof(float));
    float* output_scalar = (float*)malloc(ARRAY_SIZE * sizeof(float));
    float* output_vectorized = (float*)malloc(ARRAY_SIZE * sizeof(float));
    
    if (!input || !output_scalar || !output_vectorized) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    // Initialize input with random values in [0, 2π] for trigonometric functions
    srand(42);
    for (int i = 0; i < ARRAY_SIZE; i++) {
        input[i] = ((float)rand() / RAND_MAX) * 2.0f * M_PI;
    }
    
    printf("Array size: %d elements\n", ARRAY_SIZE);
    printf("Iterations per test: %d\n", NUM_ITERATIONS);
    printf("Testing with GNU libmvec vectorization...\n\n");
    
    // Test SIN function
    printf("=== SIN Performance Test ===\n");
    
    double scalar_sin_time = measure_function_performance(scalar_sin, input, output_scalar, ARRAY_SIZE);
    double vectorized_sin_time = measure_function_performance(ggml_vec_sin_f32_numa, input, output_vectorized, ARRAY_SIZE);
    
    printf("Scalar sin():         %.6f seconds\n", scalar_sin_time);
    printf("NUMA vectorized sin:  %.6f seconds\n", vectorized_sin_time);
    
    if (vectorized_sin_time > 0) {
        double speedup = scalar_sin_time / vectorized_sin_time;
        printf("Speedup: %.2fx %s\n", speedup, speedup > 1.0 ? "(FASTER)" : "(SLOWER)");
    }
    
    // Verify correctness (first few elements)
    printf("Correctness check (first 5 elements):\n");
    for (int i = 0; i < 5; i++) {
        float scalar_result = output_scalar[i];
        float vectorized_result = output_vectorized[i];
        float error = fabsf(scalar_result - vectorized_result);
        printf("  [%d] scalar=%.6f, vectorized=%.6f, error=%.2e\n", 
               i, scalar_result, vectorized_result, error);
    }
    printf("\n");
    
    // Test COS function
    printf("=== COS Performance Test ===\n");
    
    double scalar_cos_time = measure_function_performance(scalar_cos, input, output_scalar, ARRAY_SIZE);
    double vectorized_cos_time = measure_function_performance(ggml_vec_cos_f32_numa, input, output_vectorized, ARRAY_SIZE);
    
    printf("Scalar cos():         %.6f seconds\n", scalar_cos_time);
    printf("NUMA vectorized cos:  %.6f seconds\n", vectorized_cos_time);
    
    if (vectorized_cos_time > 0) {
        double speedup = scalar_cos_time / vectorized_cos_time;
        printf("Speedup: %.2fx %s\n", speedup, speedup > 1.0 ? "(FASTER)" : "(SLOWER)");
    }
    
    // Verify correctness (first few elements)
    printf("Correctness check (first 5 elements):\n");
    for (int i = 0; i < 5; i++) {
        float scalar_result = output_scalar[i];
        float vectorized_result = output_vectorized[i];
        float error = fabsf(scalar_result - vectorized_result);
        printf("  [%d] scalar=%.6f, vectorized=%.6f, error=%.2e\n", 
               i, scalar_result, vectorized_result, error);
    }
    printf("\n");
    
    // Summary
    printf("=== Summary ===\n");
    if (scalar_sin_time > 0 && vectorized_sin_time > 0) {
        double sin_speedup = scalar_sin_time / vectorized_sin_time;
        double cos_speedup = scalar_cos_time / vectorized_cos_time;
        double avg_speedup = (sin_speedup + cos_speedup) / 2.0;
        
        printf("Average speedup: %.2fx\n", avg_speedup);
        if (avg_speedup > 2.0) {
            printf("✅ SUCCESS: Significant speedup achieved with GNU libmvec!\n");
        } else if (avg_speedup > 1.2) {
            printf("✅ GOOD: Moderate speedup achieved.\n");
        } else if (avg_speedup < 0.8) {
            printf("❌ REGRESSION: Vectorized version is slower (fake SIMD problem?).\n");
        } else {
            printf("⚠️  NEUTRAL: Similar performance to scalar.\n");
        }
    }
    
    // Cleanup
    free(input);
    free(output_scalar);
    free(output_vectorized);
    
    return 0;
}
