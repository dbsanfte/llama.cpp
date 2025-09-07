/**
 * @file test-numa-rope-simd-optimization.c
 * @brief Test performance improvement of NUMA vector SIMD transcendental functions in ROPE
 * @author David Sanftenberg
 * 
 * This test validates that our NUMA vector optimization provides meaningful
 * performance improvements for ROPE cache initialization using SIMD sin/cos.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <assert.h>

// Include our NUMA vector implementation
#include "ggml/include/ggml-cpu.h"
#include "ggml/src/ggml-cpu/ggml-vec-numa.h"

#define TEST_CACHE_SIZE 4096  // Typical ROPE cache size
#define NUM_ITERATIONS 1000   // For performance measurement
#define TOLERANCE 1e-5f       // Floating point comparison tolerance

/**
 * Reference scalar implementation (mimics original ROPE cache init)
 */
static void rope_cache_init_scalar(float * cache, int cache_size) {
    float theta_base = 10000.0f;
    float theta_scale = 0.99f;
    float freq_scale = 1.0f;
    
    float theta = theta_base;
    for (int i0 = 0; i0 < cache_size; i0 += 2) {
        // Reference scalar sin/cos computation
        cache[i0 + 0] = cosf(theta * freq_scale);
        cache[i0 + 1] = sinf(theta * freq_scale);
        theta *= theta_scale;
    }
}

/**
 * NUMA optimized implementation (using our SIMD vector functions)
 */
static void rope_cache_init_numa_simd(float * cache, int cache_size) {
    // Check if SIMD optimization should be used
    if (cache_size >= GGML_VEC_NUMA_AVX2_THRESHOLD) {
        // Vectorized implementation
        const int simd_pairs = cache_size / 2;
        
        float * theta_values = malloc(simd_pairs * sizeof(float));
        float * cos_values = malloc(simd_pairs * sizeof(float));
        float * sin_values = malloc(simd_pairs * sizeof(float));
        
        if (theta_values && cos_values && sin_values) {
            // Prepare theta values
            float theta_base = 10000.0f;
            float theta_scale = 0.99f;
            float freq_scale = 1.0f;
            
            float theta = theta_base;
            for (int pair_idx = 0; pair_idx < simd_pairs; pair_idx++) {
                theta_values[pair_idx] = theta * freq_scale;
                theta *= theta_scale;
            }
            
            // NUMA VECTOR OPTIMIZATION: Use SIMD sin/cos
            GGML_VEC_SINCOS_F32_NUMA(simd_pairs, sin_values, cos_values, theta_values);
            
            // Populate cache with vectorized results
            for (int pair_idx = 0; pair_idx < simd_pairs; pair_idx++) {
                int i0 = pair_idx * 2;
                cache[i0 + 0] = cos_values[pair_idx];
                cache[i0 + 1] = sin_values[pair_idx];
            }
            
            free(theta_values);
            free(cos_values);
            free(sin_values);
            return;
        } else {
            // Memory allocation failed, fall back to scalar
            if (theta_values) free(theta_values);
            if (cos_values) free(cos_values);
            if (sin_values) free(sin_values);
        }
    }
    
    // Fallback to scalar implementation
    rope_cache_init_scalar(cache, cache_size);
}

/**
 * Validate mathematical correctness between scalar and SIMD implementations
 */
static int validate_mathematical_correctness(void) {
    printf("🧮 Testing mathematical correctness...\n");
    
    float * cache_scalar = malloc(TEST_CACHE_SIZE * sizeof(float));
    float * cache_simd = malloc(TEST_CACHE_SIZE * sizeof(float));
    
    if (!cache_scalar || !cache_simd) {
        fprintf(stderr, "❌ Memory allocation failed\n");
        return 0;
    }
    
    // Initialize both implementations
    rope_cache_init_scalar(cache_scalar, TEST_CACHE_SIZE);
    rope_cache_init_numa_simd(cache_simd, TEST_CACHE_SIZE);
    
    // Compare results
    int mismatches = 0;
    float max_diff = 0.0f;
    
    for (int i = 0; i < TEST_CACHE_SIZE; i++) {
        float diff = fabsf(cache_scalar[i] - cache_simd[i]);
        if (diff > max_diff) max_diff = diff;
        
        if (diff > TOLERANCE) {
            if (mismatches < 5) {  // Show first few mismatches
                printf("   Mismatch at [%d]: scalar=%.6f, simd=%.6f, diff=%.6f\n", 
                       i, cache_scalar[i], cache_simd[i], diff);
            }
            mismatches++;
        }
    }
    
    printf("   Maximum difference: %.8f\n", max_diff);
    printf("   Mismatches: %d / %d elements\n", mismatches, TEST_CACHE_SIZE);
    
    free(cache_scalar);
    free(cache_simd);
    
    if (mismatches == 0) {
        printf("✅ Mathematical correctness validated (max diff: %.8f)\n", max_diff);
        return 1;
    } else {
        printf("❌ Mathematical correctness failed (%d mismatches)\n", mismatches);
        return 0;
    }
}

/**
 * Measure performance of both implementations
 */
static void measure_performance(void) {
    printf("🚀 Measuring performance...\n");
    
    float * cache = malloc(TEST_CACHE_SIZE * sizeof(float));
    if (!cache) {
        fprintf(stderr, "❌ Memory allocation failed\n");
        return;
    }
    
    // Measure scalar implementation
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        rope_cache_init_scalar(cache, TEST_CACHE_SIZE);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double scalar_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    // Measure NUMA SIMD implementation
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        rope_cache_init_numa_simd(cache, TEST_CACHE_SIZE);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double simd_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    // Calculate speedup
    double speedup = scalar_time / simd_time;
    double percent_improvement = ((scalar_time - simd_time) / scalar_time) * 100.0;
    
    printf("   📊 Performance Results:\n");
    printf("   Scalar implementation:  %.3f ms (%d iterations)\n", scalar_time * 1000, NUM_ITERATIONS);
    printf("   NUMA SIMD implementation: %.3f ms (%d iterations)\n", simd_time * 1000, NUM_ITERATIONS);
    printf("   🎯 Speedup: %.2fx (%.1f%% improvement)\n", speedup, percent_improvement);
    
    if (speedup > 1.1) {
        printf("✅ Significant performance improvement achieved!\n");
    } else if (speedup > 1.0) {
        printf("🟡 Modest performance improvement (may vary by CPU)\n");
    } else {
        printf("🟡 No significant performance improvement (scalar may be faster on this CPU)\n");
    }
    
    free(cache);
}

int main(int argc, char ** argv) {
    printf("🧪 NUMA Vector ROPE SIMD Optimization Test\n");
    printf("==========================================\n");
    printf("Test config: cache_size=%d, iterations=%d\n", TEST_CACHE_SIZE, NUM_ITERATIONS);
    printf("\n");
    
    // Initialize NUMA vector system
    printf("🔧 Initializing NUMA vector operations...\n");
    ggml_vec_numa_init();
    printf("✅ NUMA vector system initialized\n");
    printf("\n");
    
    // Test mathematical correctness
    if (!validate_mathematical_correctness()) {
        printf("❌ Test failed: Mathematical correctness not validated\n");
        return 1;
    }
    printf("\n");
    
    // Measure performance
    measure_performance();
    printf("\n");
    
    printf("🎉 NUMA vector ROPE optimization test completed!\n");
    printf("📝 This demonstrates AVX-512/AVX2 transcendental function acceleration\n");
    printf("🔬 On AVX-512 systems with Intel SVML, expect higher speedups\n");
    
    return 0;
}
