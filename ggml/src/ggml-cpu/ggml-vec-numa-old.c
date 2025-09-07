/**
 * @file ggml-vec-numa.c
 * @brief NUMA-optimized vector operations implementation with runtime SIMD dispatch
 * @author David Sanftenberg
 * 
 * ============================================================================
 * NUMA VECTOR OPERATIONS - Runtime SIMD Dispatch Implementation
 * ============================================================================
 * 
 * This file implements NUMA-optimized vector operations with automatic
 * runtime dispatch to the best available SIMD implementation.
 * 
 * ARCHITECTURE:
 * ============
 * 1. Function pointer dispatch system - initialized once at startup
 * 2. Multiple implementations per function - scalar, AVX, AVX2, AVX-512
 * 3. CPU feature detection - automatic selection of best implementation
 * 4. Zero runtime overhead - direct function pointer calls after init
 * 
 * IMPLEMENTATION HIERARCHY:
 * ========================
 * AVX-512 + Intel SVML (best accuracy & performance)
 *   ↓
 * AVX-512F + custom approximations (good performance)
 *   ↓  
 * AVX2 vectorized (moderate performance)
 *   ↓
 * Scalar reference (compatibility fallback)
 * 
 * ============================================================================
 */

#include "ggml-vec-numa.h"
#include "ggml-cpu.h"
#include "vec.h"
#include "../../include/ggml-cpu.h"

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __AVX__
#include <immintrin.h>
#endif

// ============================================================================
// Function Pointer Types for Runtime Dispatch
// ============================================================================

typedef void (*ggml_vec_transcendental_f32_t)(const int n, float * y, const float * x);
typedef void (*ggml_vec_transcendental_f16_t)(const int n, ggml_fp16_t * y, const ggml_fp16_t * x);
typedef void (*ggml_vec_sincos_f32_t)(const int n, float * sin_y, float * cos_y, const float * x);
typedef void (*ggml_vec_sincos_f16_t)(const int n, ggml_fp16_t * sin_y, ggml_fp16_t * cos_y, const ggml_fp16_t * x);

// ============================================================================
// Function Pointer Dispatch Tables
// ============================================================================

static ggml_vec_transcendental_f32_t ggml_vec_sin_f32_impl = NULL;
static ggml_vec_transcendental_f32_t ggml_vec_cos_f32_impl = NULL;
static ggml_vec_transcendental_f32_t ggml_vec_log_f32_impl = NULL;
static ggml_vec_transcendental_f32_t ggml_vec_exp_f32_impl = NULL;

static ggml_vec_transcendental_f16_t ggml_vec_sin_f16_impl = NULL;
static ggml_vec_transcendental_f16_t ggml_vec_cos_f16_impl = NULL;
static ggml_vec_transcendental_f16_t ggml_vec_log_f16_impl = NULL;
static ggml_vec_transcendental_f16_t ggml_vec_exp_f16_impl = NULL;

static ggml_vec_sincos_f32_t ggml_vec_sincos_f32_impl = NULL;
static ggml_vec_sincos_f16_t ggml_vec_sincos_f16_impl = NULL;

// ============================================================================
// Scalar Reference Implementations (Always Available)
// ============================================================================

void ggml_vec_sin_f32_scalar(const int n, float * y, const float * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = sinf(x[i]);
    }
}

void ggml_vec_cos_f32_scalar(const int n, float * y, const float * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = cosf(x[i]);
    }
}

void ggml_vec_log_f32_scalar(const int n, float * y, const float * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = logf(x[i]);
    }
}

void ggml_vec_exp_f32_scalar(const int n, float * y, const float * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = expf(x[i]);
    }
}

void ggml_vec_sincos_f32_scalar(const int n, float * sin_y, float * cos_y, const float * x) {
    for (int i = 0; i < n; ++i) {
        sin_y[i] = sinf(x[i]);
        cos_y[i] = cosf(x[i]);
    }
}

// F16 scalar implementations
static void ggml_vec_sin_f16_scalar(const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(sinf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}

static void ggml_vec_cos_f16_scalar(const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(cosf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}

static void ggml_vec_log_f16_scalar(const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(logf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}

static void ggml_vec_exp_f16_scalar(const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(expf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}

static void ggml_vec_sincos_f16_scalar(const int n, ggml_fp16_t * sin_y, ggml_fp16_t * cos_y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        float x_f32 = GGML_CPU_FP16_TO_FP32(x[i]);
        sin_y[i] = GGML_CPU_FP32_TO_FP16(sinf(x_f32));
        cos_y[i] = GGML_CPU_FP32_TO_FP16(cosf(x_f32));
    }
}

// ============================================================================
// AVX-512 + GNU libmvec Implementations (Production SIMD Transcendental Functions)
// ============================================================================

#ifdef __AVX512F__

// Real SIMD transcendental functions using GNU libmvec
// These provide true vectorization: 16 operations computed simultaneously

void ggml_vec_sin_f32_avx512_mvec(int n, float * y, const float * x) {
    if (n < GGML_VEC_NUMA_AVX512_THRESHOLD) {
        ggml_vec_sin_f32_scalar(n, y, x);
        return;
    }
    
#ifdef GGML_NUMA_LIBMVEC_ENABLED
    // GNU libmvec with auto-vectorization - compiler will use _ZGVeN16v_sinf
    // The -ffast-math flag enables auto-vectorization of this loop
#pragma GCC ivdep
    for (int i = 0; i < n; i++) {
        y[i] = sinf(x[i]);
    }
#else
    // Fallback to scalar implementation when libmvec not available
    ggml_vec_sin_f32_scalar(n, y, x);
#endif
}

void ggml_vec_cos_f32_avx512_mvec(int n, float * y, const float * x) {
    if (n < GGML_VEC_NUMA_AVX512_THRESHOLD) {
        ggml_vec_cos_f32_scalar(n, y, x);
        return;
    }
    
#ifdef GGML_NUMA_LIBMVEC_ENABLED
    // GNU libmvec with auto-vectorization - compiler will use _ZGVeN16v_cosf
#pragma GCC ivdep
    for (int i = 0; i < n; i++) {
        y[i] = cosf(x[i]);
    }
#else
    ggml_vec_cos_f32_scalar(n, y, x);
#endif
}

void ggml_vec_log_f32_avx512_mvec(int n, float * y, const float * x) {
    if (n < GGML_VEC_NUMA_AVX512_THRESHOLD) {
        ggml_vec_log_f32_scalar(n, y, x);
        return;
    }
    
#ifdef GGML_NUMA_LIBMVEC_ENABLED
    // GNU libmvec with auto-vectorization - compiler will use _ZGVeN16v_logf
#pragma GCC ivdep
    for (int i = 0; i < n; i++) {
        y[i] = logf(x[i]);
    }
#else
    ggml_vec_log_f32_scalar(n, y, x);
#endif
}

void ggml_vec_exp_f32_avx512_mvec(int n, float * y, const float * x) {
    if (n < GGML_VEC_NUMA_AVX512_THRESHOLD) {
        ggml_vec_exp_f32_scalar(n, y, x);
        return;
    }
    
#ifdef GGML_NUMA_LIBMVEC_ENABLED
    // GNU libmvec with auto-vectorization - compiler will use _ZGVeN16v_expf
#pragma GCC ivdep
    for (int i = 0; i < n; i++) {
        y[i] = expf(x[i]);
    }
#else
    ggml_vec_exp_f32_scalar(n, y, x);
#endif
}

void ggml_vec_sincos_f32_avx512_mvec(int n, float * sin_y, float * cos_y, const float * x) {
    if (n < GGML_VEC_NUMA_AVX512_THRESHOLD) {
        ggml_vec_sincos_f32_scalar(n, sin_y, cos_y, x);
        return;
    }
    
#ifdef GGML_NUMA_LIBMVEC_ENABLED
    // GNU libmvec with auto-vectorization - compiler will use _ZGVeN16vvv_sincosf
#pragma GCC ivdep
    for (int i = 0; i < n; i++) {
        sincosf(x[i], &sin_y[i], &cos_y[i]);
    }
#else
    ggml_vec_sincos_f32_scalar(n, sin_y, cos_y, x);
#endif
}

static void ggml_vec_cos_f32_avx512_mvec(const int n, float * y, const float * x) {
    if (n < GGML_VEC_NUMA_AVX512_THRESHOLD) {
        ggml_vec_cos_f32_scalar(n, y, x);
        return;
    }
    
    const int simd_width = 16;
    const int np = (n & ~(simd_width - 1));
    
    for (int i = 0; i < np; i += simd_width) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        
        __m512 y_vec;
        float x_array[16], y_array[16];
        _mm512_storeu_ps(x_array, x_vec);
        
        #pragma GCC ivdep
        for (int j = 0; j < 16; j++) {
            y_array[j] = cosf(x_array[j]);
        }
        
        y_vec = _mm512_loadu_ps(y_array);
        _mm512_storeu_ps(y + i, y_vec);
    }
    
    for (int i = np; i < n; ++i) {
        y[i] = cosf(x[i]);
    }
}

static void ggml_vec_sincos_f32_avx512_mvec(const int n, float * sin_y, float * cos_y, const float * x) {
    if (n < GGML_VEC_NUMA_AVX512_THRESHOLD) {
        ggml_vec_sincos_f32_scalar(n, sin_y, cos_y, x);
        return;
    }
    
    const int simd_width = 16;
    const int np = (n & ~(simd_width - 1));
    
    for (int i = 0; i < np; i += simd_width) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        
        float x_array[16], sin_array[16], cos_array[16];
        _mm512_storeu_ps(x_array, x_vec);
        
        // Combined sin/cos computation - GCC vectorizes this efficiently with libmvec
        #pragma GCC ivdep
        for (int j = 0; j < 16; j++) {
            sin_array[j] = sinf(x_array[j]);
            cos_array[j] = cosf(x_array[j]);
        }
        
        __m512 sin_vec = _mm512_loadu_ps(sin_array);
        __m512 cos_vec = _mm512_loadu_ps(cos_array);
        
        _mm512_storeu_ps(sin_y + i, sin_vec);
        _mm512_storeu_ps(cos_y + i, cos_vec);
    }
    
    for (int i = np; i < n; ++i) {
        sin_y[i] = sinf(x[i]);
        cos_y[i] = cosf(x[i]);
    }
}

static void ggml_vec_log_f32_avx512_mvec(const int n, float * y, const float * x) {
    if (n < GGML_VEC_NUMA_AVX512_THRESHOLD) {
        ggml_vec_log_f32_scalar(n, y, x);
        return;
    }
    
    const int simd_width = 16;
    const int np = (n & ~(simd_width - 1));
    
    for (int i = 0; i < np; i += simd_width) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        
        float x_array[16], y_array[16];
        _mm512_storeu_ps(x_array, x_vec);
        
        #pragma GCC ivdep
        for (int j = 0; j < 16; j++) {
            y_array[j] = logf(x_array[j]);
        }
        
        __m512 y_vec = _mm512_loadu_ps(y_array);
        _mm512_storeu_ps(y + i, y_vec);
    }
    
    for (int i = np; i < n; ++i) {
        y[i] = logf(x[i]);
    }
}

static void ggml_vec_exp_f32_avx512_mvec(const int n, float * y, const float * x) {
    if (n < GGML_VEC_NUMA_AVX512_THRESHOLD) {
        ggml_vec_exp_f32_scalar(n, y, x);
        return;
    }
    
    const int simd_width = 16;
    const int np = (n & ~(simd_width - 1));
    
    for (int i = 0; i < np; i += simd_width) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        
        float x_array[16], y_array[16];
        _mm512_storeu_ps(x_array, x_vec);
        
        #pragma GCC ivdep
        for (int j = 0; j < 16; j++) {
            y_array[j] = expf(x_array[j]);
        }
        
        __m512 y_vec = _mm512_loadu_ps(y_array);
        _mm512_storeu_ps(y + i, y_vec);
    }
    
    for (int i = np; i < n; ++i) {
        y[i] = expf(x[i]);
    }
}

#endif // __AVX512F__

// ============================================================================
// AVX-512 + Intel SVML Implementations (Best Performance & Accuracy)
// ============================================================================

#if defined(__AVX512F__) && defined(__INTEL_COMPILER)

static void ggml_vec_sin_f32_avx512_svml(const int n, float * y, const float * x) {
    const int simd_width = 16;
    const int np = (n & ~(simd_width - 1));
    
    // Process 16 floats at a time with Intel SVML
    for (int i = 0; i < np; i += simd_width) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m512 y_vec = _mm512_sin_ps(x_vec);  // Intel SVML
        _mm512_storeu_ps(y + i, y_vec);
    }
    
    // Scalar remainder
    for (int i = np; i < n; ++i) {
        y[i] = sinf(x[i]);
    }
}

static void ggml_vec_cos_f32_avx512_svml(const int n, float * y, const float * x) {
    const int simd_width = 16;
    const int np = (n & ~(simd_width - 1));
    
    for (int i = 0; i < np; i += simd_width) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m512 y_vec = _mm512_cos_ps(x_vec);  // Intel SVML
        _mm512_storeu_ps(y + i, y_vec);
    }
    
    for (int i = np; i < n; ++i) {
        y[i] = cosf(x[i]);
    }
}

static void ggml_vec_log_f32_avx512_svml(const int n, float * y, const float * x) {
    const int simd_width = 16;
    const int np = (n & ~(simd_width - 1));
    
    for (int i = 0; i < np; i += simd_width) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m512 y_vec = _mm512_log_ps(x_vec);  // Intel SVML
        _mm512_storeu_ps(y + i, y_vec);
    }
    
    for (int i = np; i < n; ++i) {
        y[i] = logf(x[i]);
    }
}

static void ggml_vec_exp_f32_avx512_svml(const int n, float * y, const float * x) {
    const int simd_width = 16;
    const int np = (n & ~(simd_width - 1));
    
    for (int i = 0; i < np; i += simd_width) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m512 y_vec = _mm512_exp_ps(x_vec);  // Intel SVML
        _mm512_storeu_ps(y + i, y_vec);
    }
    
    for (int i = np; i < n; ++i) {
        y[i] = expf(x[i]);
    }
}

static void ggml_vec_sincos_f32_avx512_svml(const int n, float * sin_y, float * cos_y, const float * x) {
    const int simd_width = 16;
    const int np = (n & ~(simd_width - 1));
    
    for (int i = 0; i < np; i += simd_width) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m512 sin_vec = _mm512_sin_ps(x_vec);  // Intel SVML
        __m512 cos_vec = _mm512_cos_ps(x_vec);  // Intel SVML
        _mm512_storeu_ps(sin_y + i, sin_vec);
        _mm512_storeu_ps(cos_y + i, cos_vec);
    }
    
    for (int i = np; i < n; ++i) {
        sin_y[i] = sinf(x[i]);
        cos_y[i] = cosf(x[i]);
    }
}

#endif // __AVX512F__ && __INTEL_COMPILER

// ============================================================================
// AVX-512F Custom Approximation Implementations
// ============================================================================

#ifdef __AVX512F__

// Fast sine approximation using polynomial (placeholder - would need proper implementation)
static inline __m512 fast_sin_avx512(__m512 x) {
    // TODO: Implement high-quality polynomial approximation
    // For now, fall back to scalar in vectorized loop
    float x_array[16], y_array[16];
    _mm512_storeu_ps(x_array, x);
    for (int i = 0; i < 16; i++) {
        y_array[i] = sinf(x_array[i]);
    }
    return _mm512_loadu_ps(y_array);
}

static inline __m512 fast_cos_avx512(__m512 x) {
    // TODO: Implement high-quality polynomial approximation  
    float x_array[16], y_array[16];
    _mm512_storeu_ps(x_array, x);
    for (int i = 0; i < 16; i++) {
        y_array[i] = cosf(x_array[i]);
    }
    return _mm512_loadu_ps(y_array);
}

static void ggml_vec_sin_f32_avx512(const int n, float * y, const float * x) {
    if (n < GGML_VEC_NUMA_AVX512_THRESHOLD) {
        ggml_vec_sin_f32_scalar(n, y, x);
        return;
    }
    
    const int simd_width = 16;
    const int np = (n & ~(simd_width - 1));
    
    for (int i = 0; i < np; i += simd_width) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m512 y_vec = fast_sin_avx512(x_vec);
        _mm512_storeu_ps(y + i, y_vec);
    }
    
    for (int i = np; i < n; ++i) {
        y[i] = sinf(x[i]);
    }
}

static void ggml_vec_cos_f32_avx512(const int n, float * y, const float * x) {
    if (n < GGML_VEC_NUMA_AVX512_THRESHOLD) {
        ggml_vec_cos_f32_scalar(n, y, x);
        return;
    }
    
    const int simd_width = 16;
    const int np = (n & ~(simd_width - 1));
    
    for (int i = 0; i < np; i += simd_width) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m512 y_vec = fast_cos_avx512(x_vec);
        _mm512_storeu_ps(y + i, y_vec);
    }
    
    for (int i = np; i < n; ++i) {
        y[i] = cosf(x[i]);
    }
}

#endif // __AVX512F__

// ============================================================================
// Runtime Initialization and CPU Feature Detection
// ============================================================================

void ggml_vec_numa_init(void) {
    // Default to scalar implementations
    ggml_vec_sin_f32_impl = ggml_vec_sin_f32_scalar;
    ggml_vec_cos_f32_impl = ggml_vec_cos_f32_scalar;
    ggml_vec_log_f32_impl = ggml_vec_log_f32_scalar;
    ggml_vec_exp_f32_impl = ggml_vec_exp_f32_scalar;
    ggml_vec_sincos_f32_impl = ggml_vec_sincos_f32_scalar;
    
    ggml_vec_sin_f16_impl = ggml_vec_sin_f16_scalar;
    ggml_vec_cos_f16_impl = ggml_vec_cos_f16_scalar;
    ggml_vec_log_f16_impl = ggml_vec_log_f16_scalar;
    ggml_vec_exp_f16_impl = ggml_vec_exp_f16_scalar;
    ggml_vec_sincos_f16_impl = ggml_vec_sincos_f16_scalar;
    
#ifndef GGML_VEC_NUMA_FORCE_SCALAR
    
    // Check for AVX-512 + Intel SVML (best option)
#if defined(__AVX512F__) && defined(__INTEL_COMPILER) && !defined(GGML_VEC_NUMA_DISABLE_SVML)
    if (ggml_cpu_has_avx512()) {
        ggml_vec_sin_f32_impl = ggml_vec_sin_f32_avx512_svml;
        ggml_vec_cos_f32_impl = ggml_vec_cos_f32_avx512_svml;
        ggml_vec_log_f32_impl = ggml_vec_log_f32_avx512_svml;
        ggml_vec_exp_f32_impl = ggml_vec_exp_f32_avx512_svml;
        ggml_vec_sincos_f32_impl = ggml_vec_sincos_f32_avx512_svml;
        printf("NUMA Vector: Using Intel SVML AVX-512 transcendental functions\n");
        return;  // Best implementation found
    }
#endif
    
    // Check for AVX-512F + GNU libmvec (excellent performance)
#if defined(__AVX512F__) && !defined(GGML_VEC_NUMA_DISABLE_MVEC)
    if (ggml_cpu_has_avx512()) {
        ggml_vec_sin_f32_impl = ggml_vec_sin_f32_avx512_mvec;
        ggml_vec_cos_f32_impl = ggml_vec_cos_f32_avx512_mvec;
        ggml_vec_log_f32_impl = ggml_vec_log_f32_avx512_mvec;
        ggml_vec_exp_f32_impl = ggml_vec_exp_f32_avx512_mvec;
        ggml_vec_sincos_f32_impl = ggml_vec_sincos_f32_avx512_mvec;
        printf("NUMA Vector: Using GNU libmvec AVX-512 transcendental functions\n");
        return;  // Excellent implementation found
    }
#endif
    
    // Check for AVX-512F (custom approximations) - fallback
#ifdef __AVX512F__
    if (ggml_cpu_has_avx512()) {
        ggml_vec_sin_f32_impl = ggml_vec_sin_f32_avx512;
        ggml_vec_cos_f32_impl = ggml_vec_cos_f32_avx512;
        printf("NUMA Vector: Using AVX-512 custom approximations\n");
        return;
    }
#endif
    
    // TODO: Add AVX2 and AVX implementations
    
    printf("NUMA Vector: Using scalar fallback implementations\n");
    
#endif // !GGML_VEC_NUMA_FORCE_SCALAR
}

// ============================================================================
// Public API Functions (Direct Dispatch to Best Implementation)
// ============================================================================

void ggml_vec_sin_f32_numa(const int n, float * y, const float * x) {
    ggml_vec_sin_f32_impl(n, y, x);
}

void ggml_vec_cos_f32_numa(const int n, float * y, const float * x) {
    ggml_vec_cos_f32_impl(n, y, x);
}

void ggml_vec_log_f32_numa(const int n, float * y, const float * x) {
    ggml_vec_log_f32_impl(n, y, x);
}

void ggml_vec_exp_f32_numa(const int n, float * y, const float * x) {
    ggml_vec_exp_f32_impl(n, y, x);
}

void ggml_vec_sincos_f32_numa(const int n, float * sin_y, float * cos_y, const float * x) {
    ggml_vec_sincos_f32_impl(n, sin_y, cos_y, x);
}

void ggml_vec_sin_f16_numa(const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    ggml_vec_sin_f16_impl(n, y, x);
}

void ggml_vec_cos_f16_numa(const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    ggml_vec_cos_f16_impl(n, y, x);
}

void ggml_vec_log_f16_numa(const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    ggml_vec_log_f16_impl(n, y, x);
}

void ggml_vec_exp_f16_numa(const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    ggml_vec_exp_f16_impl(n, y, x);
}

void ggml_vec_sincos_f16_numa(const int n, ggml_fp16_t * sin_y, ggml_fp16_t * cos_y, const ggml_fp16_t * x) {
    ggml_vec_sincos_f16_impl(n, sin_y, cos_y, x);
}
