/**
 * @file ggml-vec-numa.h
 * @brief NUMA-optimized vector operations with automatic SIMD dispatch
 * @author David Sanftenberg
 * 
 * ============================================================================
 * NUMA VECTOR OPERATIONS - SIMD-Optimized Transcendental Functions
 * ============================================================================
 * 
 * This header provides NUMA-optimized versions of vector operations with
 * automatic runtime dispatch to the best available SIMD implementation.
 * 
 * DESIGN PRINCIPLES:
 * =================
 * 1. Zero impact on reference implementations - all functions have _NUMA suffix
 * 2. Automatic runtime dispatch - no conditional code in kernels
 * 3. Fallback compatibility - always works, optimizes when possible
 * 4. Clean macro interface - looks like regular function calls
 * 
 * SUPPORTED OPTIMIZATIONS:
 * =======================
 * - AVX-512 + Intel SVML: Hardware-optimized transcendental functions
 * - AVX-512F: Custom polynomial approximations for trigonometric functions
 * - AVX2/AVX: Fallback SIMD implementations
 * - Scalar: Reference implementation fallback
 * 
 * USAGE:
 * =====
 * Instead of: ggml_vec_cos_f32(n, y, x);
 * Use:        GGML_VEC_COS_F32_NUMA(n, y, x);
 * 
 * The macro automatically selects the best implementation at runtime.
 * 
 * ============================================================================
 */

#ifndef GGML_VEC_NUMA_H
#define GGML_VEC_NUMA_H

#include "ggml-cpu.h"
#include "vec.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// NUMA Vector Function Declarations
// ============================================================================

/**
 * NUMA-optimized transcendental vector functions
 * These functions automatically dispatch to the best available SIMD implementation
 */

// Trigonometric functions
void ggml_vec_sin_f32_numa(const int n, float * y, const float * x);
void ggml_vec_cos_f32_numa(const int n, float * y, const float * x);
void ggml_vec_sin_f16_numa(const int n, ggml_fp16_t * y, const ggml_fp16_t * x);
void ggml_vec_cos_f16_numa(const int n, ggml_fp16_t * y, const ggml_fp16_t * x);

// Logarithmic functions  
void ggml_vec_log_f32_numa(const int n, float * y, const float * x);
void ggml_vec_exp_f32_numa(const int n, float * y, const float * x);
void ggml_vec_log_f16_numa(const int n, ggml_fp16_t * y, const ggml_fp16_t * x);
void ggml_vec_exp_f16_numa(const int n, ggml_fp16_t * y, const ggml_fp16_t * x);

// Combined sin/cos functions for ROPE optimization
void ggml_vec_sincos_f32_numa(const int n, float * sin_y, float * cos_y, const float * x);
void ggml_vec_sincos_f16_numa(const int n, ggml_fp16_t * sin_y, ggml_fp16_t * cos_y, const ggml_fp16_t * x);

// ============================================================================
// IMPLEMENTATION FUNCTION DECLARATIONS (Internal)
// ============================================================================

// Scalar fallback implementations
void ggml_vec_sin_f32_scalar(int n, float * y, const float * x);
void ggml_vec_cos_f32_scalar(int n, float * y, const float * x);
void ggml_vec_log_f32_scalar(int n, float * y, const float * x);
void ggml_vec_exp_f32_scalar(int n, float * y, const float * x);
void ggml_vec_sincos_f32_scalar(int n, float * sin_y, float * cos_y, const float * x);

// AVX-512 custom approximations
void ggml_vec_sin_f32_avx512(int n, float * y, const float * x);
void ggml_vec_cos_f32_avx512(int n, float * y, const float * x);

// GNU libmvec implementations (excellent performance with GCC)
void ggml_vec_sin_f32_avx512_mvec(int n, float * y, const float * x);
void ggml_vec_cos_f32_avx512_mvec(int n, float * y, const float * x);
void ggml_vec_log_f32_avx512_mvec(int n, float * y, const float * x);
void ggml_vec_exp_f32_avx512_mvec(int n, float * y, const float * x);
void ggml_vec_sincos_f32_avx512_mvec(int n, float * sin_y, float * cos_y, const float * x);

// Intel SVML implementations (best performance with Intel compiler)
void ggml_vec_sin_f32_avx512_svml(int n, float * y, const float * x);
void ggml_vec_cos_f32_avx512_svml(int n, float * y, const float * x);
void ggml_vec_log_f32_avx512_svml(int n, float * y, const float * x);
void ggml_vec_exp_f32_avx512_svml(int n, float * y, const float * x);
void ggml_vec_sincos_f32_avx512_svml(int n, float * sin_y, float * cos_y, const float * x);

// Function pointer types for runtime dispatch
typedef void (*ggml_vec_transcendental_f32_t)(int n, float * y, const float * x);
typedef void (*ggml_vec_sincos_f32_t)(int n, float * sin_y, float * cos_y, const float * x);
typedef void (*ggml_vec_transcendental_f16_t)(int n, ggml_fp16_t * y, const ggml_fp16_t * x);
typedef void (*ggml_vec_sincos_f16_t)(int n, ggml_fp16_t * sin_y, ggml_fp16_t * cos_y, const ggml_fp16_t * x);

// Global function pointers (set during ggml_vec_numa_init)
extern ggml_vec_transcendental_f32_t ggml_vec_sin_f32_impl;
extern ggml_vec_transcendental_f32_t ggml_vec_cos_f32_impl;
extern ggml_vec_transcendental_f32_t ggml_vec_log_f32_impl;
extern ggml_vec_transcendental_f32_t ggml_vec_exp_f32_impl;
extern ggml_vec_sincos_f32_t ggml_vec_sincos_f32_impl;

// ============================================================================
// NUMA Vector Function Initialization
// ============================================================================

/**
 * Initialize NUMA vector function pointers based on CPU capabilities
 * This is called automatically during ggml initialization
 */
void ggml_vec_numa_init(void);

// ============================================================================
// CLEAN MACRO INTERFACE - No conditional code in kernels
// ============================================================================

/**
 * NUMA-optimized vector operation macros
 * These macros provide a clean interface that automatically uses the best
 * available implementation without polluting kernel code with conditionals.
 */

#define GGML_VEC_SIN_F32_NUMA(n, y, x)       ggml_vec_sin_f32_numa(n, y, x)
#define GGML_VEC_COS_F32_NUMA(n, y, x)       ggml_vec_cos_f32_numa(n, y, x)
#define GGML_VEC_LOG_F32_NUMA(n, y, x)       ggml_vec_log_f32_numa(n, y, x)
#define GGML_VEC_EXP_F32_NUMA(n, y, x)       ggml_vec_exp_f32_numa(n, y, x)

#define GGML_VEC_SIN_F16_NUMA(n, y, x)       ggml_vec_sin_f16_numa(n, y, x)
#define GGML_VEC_COS_F16_NUMA(n, y, x)       ggml_vec_cos_f16_numa(n, y, x)
#define GGML_VEC_LOG_F16_NUMA(n, y, x)       ggml_vec_log_f16_numa(n, y, x)
#define GGML_VEC_EXP_F16_NUMA(n, y, x)       ggml_vec_exp_f16_numa(n, y, x)

// Special combined operations for ROPE optimization
#define GGML_VEC_SINCOS_F32_NUMA(n, sin_y, cos_y, x)  ggml_vec_sincos_f32_numa(n, sin_y, cos_y, x)
#define GGML_VEC_SINCOS_F16_NUMA(n, sin_y, cos_y, x)  ggml_vec_sincos_f16_numa(n, sin_y, cos_y, x)

// ============================================================================
// IMPLEMENTATION SELECTION THRESHOLDS
// ============================================================================

/**
 * Minimum element counts for different SIMD implementations
 * Below these thresholds, scalar implementations may be faster due to overhead
 */
#define GGML_VEC_NUMA_AVX512_THRESHOLD    32  // Minimum elements for AVX-512 
#define GGML_VEC_NUMA_AVX2_THRESHOLD      16  // Minimum elements for AVX2
#define GGML_VEC_NUMA_AVX_THRESHOLD        8  // Minimum elements for AVX

// ============================================================================
// PERFORMANCE TUNING MACROS
// ============================================================================

/**
 * Control NUMA vector optimization behavior
 * These can be defined to customize behavior for specific use cases
 */

// Force specific implementations (for testing/debugging)
// #define GGML_VEC_NUMA_FORCE_SCALAR     1  // Force scalar implementations
// #define GGML_VEC_NUMA_FORCE_AVX512     1  // Force AVX-512 (if available)
// #define GGML_VEC_NUMA_DISABLE_SVML     1  // Disable Intel SVML usage

// Accuracy vs speed trade-offs for approximations
#ifndef GGML_VEC_NUMA_TRANSCENDENTAL_PRECISION
#define GGML_VEC_NUMA_TRANSCENDENTAL_PRECISION  3  // ULP precision target (1-4)
#endif

#ifdef __cplusplus
}
#endif

#endif // GGML_VEC_NUMA_H
