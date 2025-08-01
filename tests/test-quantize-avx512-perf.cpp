// Performance benchmark specifically for AVX-512 optimized quantization functions

#include "ggml.h"
#include "ggml-cpu.h"

#undef NDEBUG
#include <algorithm>
#include <assert.h>
#include <functional>
#include <math.h>
#include <memory>
#include <stdio.h>
#include <string>
#include <vector>
#include <chrono>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#define MAX_ALIGNMENT 64
#define QK 32
#define WARMUP 5
#define ITERATIONS 10

#define TEST_SIZE_SMALL   32*64      // 2KB  - fits in L1 cache
#define TEST_SIZE_MEDIUM  32*512     // 16KB - fits in L2 cache  
#define TEST_SIZE_LARGE   32*4096    // 128KB- fits in L3 cache
#define TEST_SIZE_XLARGE  32*32768   // 1MB  - main memory

// List of quantization types that have AVX-512 optimizations
static const ggml_type AVX512_OPTIMIZED_TYPES[] = {
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q4_K,
    GGML_TYPE_IQ2_XXS,
    GGML_TYPE_IQ3_XXS,
    GGML_TYPE_IQ4_XS,
};

static const char* AVX512_TYPE_NAMES[] = {
    "q4_0 (AVX-512)",
    "q5_0 (AVX-512)",
    "q8_0 (AVX-512)",
    "q4_K (AVX-512)",
    "iq2_xxs (AVX-512)",
    "iq3_xxs (AVX-512)",
    "iq4_xs (AVX-512)",
};

#if defined(__x86_64__) || defined(__i386__)

#include <x86intrin.h>
inline int64_t cpu_cycles() {
// Rough way to detect new-ish CPUs
#ifdef __POPCNT__
    unsigned int dummy;
    return __rdtscp(&dummy);
#else
    return __rdtsc();
#endif
}

#else

#define cpu_cycles() 0

#endif

// Generate synthetic data
static void generate_data(float offset, size_t n, float * dst) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.1 + 2*cosf(i + offset);
    }
}

static float gigabytes_per_second(size_t bytes, int64_t usecs) {
    return bytes / (float) usecs * 1000000 / (1024*1024*1024);
}

static void * align_with_offset(void * ptr, int offset) {
    size_t dummy_size = MAX_ALIGNMENT * 4;
    return (char *) std::align(MAX_ALIGNMENT, MAX_ALIGNMENT, ptr, dummy_size) + offset;
}

static void benchmark_function(const char* test_name, size_t size, size_t q_size, int64_t iterations, const std::function<float(void)> & func) {
    int64_t min_time_us = INT64_MAX;
    int64_t total_time_us = 0;
    int64_t min_time_cycles = INT64_MAX;
    int64_t total_time_cycles = 0;

    // Warmup
    for (int i = 0; i < WARMUP; i++) {
        func();
    }

    // Benchmark
    for (int i = 0; i < iterations; i++) {
        const auto start_time = std::chrono::high_resolution_clock::now();
        const int64_t start_cycles = cpu_cycles();

        func();

        const int64_t end_cycles = cpu_cycles();
        const auto end_time = std::chrono::high_resolution_clock::now();
        
        const int64_t elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

        total_time_cycles += end_cycles - start_cycles;
        min_time_cycles = std::min(min_time_cycles, end_cycles - start_cycles);
        total_time_us += elapsed_us;
        min_time_us = std::min(min_time_us, elapsed_us);
    }

    printf("    %s\n", test_name);
    printf("      size: %zu values (%.2f MB)\n", size, 4*size/(float)(1024*1024));
    printf("      min cycles/%d vals   : %9.2f\n",  QK, QK * min_time_cycles / (float) size);
    printf("      avg cycles/%d vals   : %9.2f\n",  QK, QK * total_time_cycles / (float) (size * iterations));
    printf("      min time (μs)        : %9lld\n",  (long long)min_time_us);
    printf("      avg time (μs)        : %9.2f\n",  total_time_us / (float)iterations);
    printf("      float32 throughput   : %9.2f GB/s\n",  gigabytes_per_second(4 * size * iterations, total_time_us));
    printf("      quantized throughput : %9.2f GB/s\n",  gigabytes_per_second(q_size * iterations, total_time_us));
    printf("\n");
}

static void benchmark_quantization_type(ggml_type type, const char* type_name, size_t test_size) {
    const auto * qfns = ggml_get_type_traits(type);
    const auto * qfns_cpu = ggml_get_type_traits_cpu(type);
    
    if (!qfns_cpu->from_float || !qfns->to_float || !qfns_cpu->vec_dot) {
        printf("  %s: SKIPPED (missing required functions)\n\n", type_name);
        return;
    }

    printf("  %s:\n", type_name);

    // Ensure size is aligned to block size
    const size_t block_size = qfns->blck_size;
    size_t aligned_size = ((test_size + block_size - 1) / block_size) * block_size;
    
    // For K-type quantizations, ensure we use proper sizes
    if (type == GGML_TYPE_Q4_K || type == GGML_TYPE_IQ2_XXS || 
        type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ4_XS) {
        aligned_size = ((aligned_size + 255) / 256) * 256; // QK_K = 256
    }

    // Allocate aligned memory
    std::vector<uint8_t> test_data1_v(aligned_size*4 + MAX_ALIGNMENT*2);
    std::vector<uint8_t> test_data2_v(aligned_size*4 + MAX_ALIGNMENT*2);
    std::vector<uint8_t> test_q1_v   (aligned_size*4 + MAX_ALIGNMENT*2);
    std::vector<uint8_t> test_q2_v   (aligned_size*4 + MAX_ALIGNMENT*2);

    float * test_data1 = (float *) align_with_offset(test_data1_v.data(), 0);
    float * test_data2 = (float *) align_with_offset(test_data2_v.data(), 0);
    float * test_q1    = (float *) align_with_offset(test_q1_v.data(),    0);
    float * test_q2    = (float *) align_with_offset(test_q2_v.data(),    0);

    generate_data(0, aligned_size, test_data1);
    generate_data(1, aligned_size, test_data2);

    ggml_quantize_init(type);

    size_t quantized_size = ggml_row_size(type, aligned_size);

    // Benchmark quantization
    auto quantize_fn = [&](void) -> float {
        qfns_cpu->from_float(test_data1, test_q1, aligned_size);
        return test_q1[0];
    };
    benchmark_function("quantize", aligned_size, quantized_size, ITERATIONS, quantize_fn);

    // Benchmark dequantization  
    qfns_cpu->from_float(test_data1, test_q1, aligned_size);
    auto dequantize_fn = [&](void) -> float {
        qfns->to_float(test_q1, test_data1, aligned_size);
        return test_data1[0];
    };
    benchmark_function("dequantize", aligned_size, quantized_size, ITERATIONS, dequantize_fn);

    // Benchmark dot product
    qfns_cpu->from_float(test_data1, test_q1, aligned_size);
    const auto * vdot = ggml_get_type_traits_cpu(qfns_cpu->vec_dot_type);
    if (vdot && vdot->from_float) {
        vdot->from_float(test_data2, test_q2, aligned_size);
        auto dot_fn = [&](void) -> float {
            float result;
            qfns_cpu->vec_dot(aligned_size, &result, 0, test_q1, 0, test_q2, 0, 1);
            return result;
        };
        benchmark_function("vec_dot", aligned_size, quantized_size, ITERATIONS, dot_fn);
    }
}

int main(int argc, char * argv[]) {
    std::vector<size_t> test_sizes = {TEST_SIZE_SMALL, TEST_SIZE_MEDIUM, TEST_SIZE_LARGE};
    bool test_xlarge = false;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-xlarge") {
            test_xlarge = true;
        } else if (arg == "-h" || arg == "--help") {
            printf("AVX-512 Quantization Performance Benchmark\n");
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  -xlarge    Include extra large test size (1MB)\n");
            printf("  -h, --help Show this help message\n");
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    if (test_xlarge) {
        test_sizes.push_back(TEST_SIZE_XLARGE);
    }

    // Initialize GGML
    struct ggml_init_params ggml_params = {
        /* .mem_size   = */ 1*1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true,
    };
    struct ggml_context * ctx = ggml_init(ggml_params);

    printf("AVX-512 Quantization Performance Benchmark\n");
    printf("==========================================\n\n");

    // Display CPU features
    printf("CPU Features:\n");
    printf("-------------\n");
#if defined(__AVX512F__)
    printf("  AVX-512F: ENABLED\n");
#else
    printf("  AVX-512F: DISABLED\n");
#endif

#if defined(__AVX512VNNI__)
    printf("  AVX-512VNNI: ENABLED\n");
#else
    printf("  AVX-512VNNI: DISABLED\n");
#endif

#if defined(__AVX512VL__)
    printf("  AVX-512VL: ENABLED\n");
#else
    printf("  AVX-512VL: DISABLED\n");
#endif

#if defined(__AVX2__)
    printf("  AVX2: ENABLED\n");
#else
    printf("  AVX2: DISABLED\n");
#endif

    printf("\n");

    constexpr size_t num_types = sizeof(AVX512_OPTIMIZED_TYPES) / sizeof(AVX512_OPTIMIZED_TYPES[0]);

    for (size_t test_size : test_sizes) {
        printf("Test Size: %zu values (%.2f MB)\n", test_size, 4*test_size/(float)(1024*1024));
        printf("====================================\n");

        for (size_t i = 0; i < num_types; i++) {
            benchmark_quantization_type(AVX512_OPTIMIZED_TYPES[i], AVX512_TYPE_NAMES[i], test_size);
        }
        
        printf("\n");
    }

    ggml_free(ctx);
    return 0;
}
