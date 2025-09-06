/**
 * @file rope.c
 * @brief NUMA ROPE (Rotary Position Embedding) Kernel Implementation
 * @author David Sanftenberg
 * 
 * ============================================================================
 * NUMA KERNEL: ROPE (Rotary Position Embedding) - Complete Implementation
 * ============================================================================
 * 
 * This implementation provides comprehensive ROPE kernel functionality with:
 * - All ROPE variants: Standard, NEOX, Vision, Multi-modal (mrope)
 * - Forward and backward pass support
 * - All quantization types supported by reference (F32, F16)
 * - NUMA-aware data-parallel execution
 * - YaRN (Yarn) algorithm support for extended context
 * 
 * OPERATION CHARACTERISTICS:
 * ========================
 * - Complex 4D tensor operations with position-dependent rotations
 * - Uses pre-computed cosine/sine cache for efficiency
 * - Thread-wise data parallelization across sequence elements
 * - Complex indexing patterns for different ROPE variants
 * 
 * IMPLEMENTATION STRATEGY:
 * =======================
 * 1. Type-based dispatch following reference ops.cpp exactly
 * 2. Row-based parallelization for optimal NUMA performance
 * 3. Pre-computed cache system for cosine/sine values
 * 4. Support for all ROPE variants with proper indexing
 * 5. Comprehensive error handling and validation
 * 
 * ROPE VARIANTS:
 * =============
 * - Standard ROPE: Basic rotary position embedding
 * - NEOX ROPE: Half-dimension rotation variant
 * - Vision ROPE: 2D spatial position embedding
 * - Multi-modal ROPE: Multiple position embeddings for multi-modal models
 * 
 * ============================================================================
 */

#include "rope.h"
#include "numa-kernels.h"
#include "../ggml-numa-shared.h"
#include "../ggml-numa-openmp-coordinator.h"
#include "../ggml-cpu-impl.h"
#include "../ggml-impl.h"

// Cache line padding for performance
#define CACHE_LINE_SIZE_F32 (64/sizeof(float))

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

// ============================================================================
// Helper Functions for ROPE Cache Computation
// ============================================================================

/**
 * Rope yarn ramp function for YaRN algorithm
 */
static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

/**
 * YaRN algorithm implementation - FIXED to match reference exactly
 * Based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
 * MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
 */
static void rope_yarn(
        const float theta_extrap, const float freq_scale, const float corr_dims[2],
        const int64_t i0, const float ext_factor, float mscale,
        float * cos_theta, float * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

/**
 * Initialize ROPE cache for standard/NEOX variants - FIXED to match reference implementation exactly
 */
static void ggml_rope_cache_init(
        const float theta_base, const float freq_scale, const float * freq_factors,
        const float corr_dims[2], const int64_t ne0, const float ext_factor, const float mscale,
        float * cache, const float sin_sign, const float theta_scale) {
    
    // FIXED: Use reference implementation logic exactly
    float theta = theta_base;
    for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0/2] : 1.0f;
        rope_yarn(
            theta/ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        theta *= theta_scale;
    }
}

/**
 * Initialize multi-modal ROPE cache (mrope) - FIXED to match reference exactly
 */
static void ggml_mrope_cache_init(
        const float theta_base_t, const float theta_base_h, const float theta_base_w, const float theta_base_e, 
        const int sections[4], const bool indep_sects,
        const float freq_scale, const float * freq_factors,
        const float corr_dims[2], const int64_t ne0, const float ext_factor, const float mscale,
        float * cache, const float sin_sign, const float theta_scale) {
    
    // FIXED: Use reference implementation logic exactly
    float theta_t = theta_base_t;
    float theta_h = theta_base_h;
    float theta_w = theta_base_w;
    float theta_e = theta_base_e;
    int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    int sec_w = sections[1] + sections[0];
    int sec_e = sections[2] + sec_w;
    
    for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0/2] : 1.0f;

        int sector = (i0 / 2) % sect_dims;
        if (indep_sects) {
            // compute theta independently for each dim sections
            if (sector == 0) {
                theta_t = theta_base_t;
            }
            else if (sector == sections[0]) {
                theta_h = theta_base_h;
            }
            else if (sector == sec_w) {
                theta_w = theta_base_w;
            }
            else if (sector == sec_e) {
                theta_e = theta_base_e;
            }
        }

        float theta_base = 0.0f;
        if (sector < sections[0]) {
            theta_base = theta_t;
        } else if (sector >= sections[0] && sector < sec_w) {
            theta_base = theta_h;
        } else if (sector >= sec_w && sector < sec_e) {
            theta_base = theta_w;
        } else if (sector >= sec_e) {
            theta_base = theta_e;
        }

        rope_yarn(
            theta_base/ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        if (indep_sects) {
            if (sector < sections[0]) {
                theta_t *= theta_scale;
            } else if (sector >= sections[0] && sector < sec_w) {
                theta_h *= theta_scale;
            } else if (sector >= sec_w && sector < sec_e) {
                theta_w *= theta_scale;
            } else if (sector >= sec_e) {
                theta_e *= theta_scale;
            }
        } else {
            theta_t *= theta_scale;
            theta_w *= theta_scale;
            theta_h *= theta_scale;
            theta_e *= theta_scale;
        }
    }
}

// ============================================================================
// ROPE Kernel Implementation for F32
// ============================================================================

/**
 * High-performance ROPE kernel for F32 type tensors
 */
static enum ggml_status ggml_numa_kernel_rope_f32_execute(void * work_context, 
                                                          struct ggml_compute_params * params,
                                                          const bool forward) {
    struct ggml_tensor * dst = (struct ggml_tensor *)work_context;
    
    // Validate inputs
    NUMA_ASSERT(dst != NULL, "Destination tensor cannot be null");
    NUMA_ASSERT(dst->src[0] != NULL, "Source tensor 0 cannot be null");
    NUMA_ASSERT(dst->src[1] != NULL, "Source tensor 1 (positions) cannot be null");
    NUMA_ASSERT(params != NULL, "Compute params cannot be null");
    
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * src2 = dst->src[2];
    
    // Extract ROPE parameters from op_params
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    int sections[4];
    
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];
    
    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
    memcpy(&sections,    (int32_t *) dst->op_params + 11, sizeof(int)*4);
    
    // Get tensor dimensions
    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t ne3 = dst->ne[3];
    
    const size_t nb00 = dst->src[0]->nb[0];
    const size_t nb01 = dst->src[0]->nb[1];
    const size_t nb02 = dst->src[0]->nb[2];
    const size_t nb03 = dst->src[0]->nb[3];
    
    const size_t nb0 = dst->nb[0];
    const size_t nb1 = dst->nb[1];
    const size_t nb2 = dst->nb[2];
    const size_t nb3 = dst->nb[3];
    
    GGML_ASSERT(nb00 == sizeof(float));
    GGML_ASSERT(n_dims <= ne0);
    GGML_ASSERT(n_dims % 2 == 0);
    
    // Strategy logging - only log once per operation (thread 0 of NUMA node 0)
    if (params->ith == 0 && ggml_current_numa_node == 0) {
        if (ggml_numa_is_data_parallel_execution) {
            NUMA_LOG_STRATEGY_DATA_PARALLEL("ROPE");
        } else {
            NUMA_LOG_STRATEGY_SINGLE_MULTI("ROPE");
        }
    }
    
    // Calculate threading parameters FIRST
    const int ith = params->ith;
    const int nth = params->nth;
    
    const int nr = ggml_nrows(dst);
    
    // ========================================
    // CLEAN NUMA WORK DISTRIBUTION FOR ROPE
    // ========================================
    
    // Step 1: Decide work division basis - ROPE slices by sequences (i2 dimension)
    const int64_t total_sequences = ne2;
    
    // Step 2: Divide work by NUMA nodes first
    int numa_start_seq = 0, numa_end_seq = total_sequences;
    
    if (ggml_numa_is_data_parallel_execution) {
        const int64_t seqs_per_node = total_sequences / ggml_numa_total_nodes_for_data_parallel;
        numa_start_seq = ggml_current_numa_node * seqs_per_node;
        numa_end_seq = (ggml_current_numa_node == ggml_numa_total_nodes_for_data_parallel - 1) ?
                       total_sequences : numa_start_seq + seqs_per_node;
    }
    
    const int numa_sequences = numa_end_seq - numa_start_seq;
    
    // Step 3: Divide remaining work among threads within this NUMA node
    const int64_t seqs_per_thread = (numa_sequences + nth - 1) / nth;  // Ceiling division
    const int thread_start_seq = ith * seqs_per_thread;
    const int thread_end_seq = MIN(thread_start_seq + seqs_per_thread, numa_sequences);
    
    // Convert to global sequence indices
    const int i2_start = numa_start_seq + thread_start_seq;
    const int i2_end = numa_start_seq + thread_end_seq;
    
    // Step 4: Check if this thread has work
    const bool has_work = (thread_start_seq < numa_sequences && i2_start < i2_end);
    
    // Debug logging (first thread per NUMA node only)
    if (ith == 0) {
        NUMA_LOG_DEBUG("ROPE WORK DISTRIBUTION: total_sequences=%lld, NUMA node %d: numa_start_seq=%d numa_end_seq=%d numa_sequences=%d", 
                       (long long)total_sequences, ggml_current_numa_node, numa_start_seq, numa_end_seq, numa_sequences);
        NUMA_LOG_DEBUG("ROPE WORK DISTRIBUTION: is_data_parallel=%s total_nodes=%d seqs_per_node=%lld", 
                       ggml_numa_is_data_parallel_execution ? "YES" : "NO", 
                       ggml_numa_total_nodes_for_data_parallel,
                       ggml_numa_is_data_parallel_execution ? (long long)(total_sequences / ggml_numa_total_nodes_for_data_parallel) : 0LL);
        NUMA_LOG_DEBUG("ROPE THREAD DISTRIBUTION: thread %d/%d local_range=[%d,%d) global_range=[%d,%d) has_work=%s", 
                       ith, nth, thread_start_seq, thread_end_seq, i2_start, i2_end, has_work ? "YES" : "NO");
    }
    if (ith == 0) {
        NUMA_LOG_DEBUG("ROPE WORK DISTRIBUTION: NUMA node %d gets sequences [%d,%d) (%d sequences of %lld total)",
                       ggml_current_numa_node, numa_start_seq, numa_end_seq, numa_sequences, (long long)total_sequences);
    }
    
    // Skip threads with no work - they go directly to barrier
    if (!has_work) {
        NUMA_LOG_TRACE("Thread %d/%d (NUMA node %d) has no work: thread_seqs=[%d,%d), global_seqs=[%d,%d) - going to barrier",
                       ith, nth, ggml_current_numa_node, thread_start_seq, thread_end_seq, i2_start, i2_end);
        goto barrier_wait;
    }
    
    // ONLY access tensor data AFTER validating thread has work
    // Use shared memory optimization for data-parallel execution
    float * dst_base = ggml_numa_shared_result_tensor_data ? 
                       (float *)ggml_numa_shared_result_tensor_data : 
                       (float *)tensor_data(dst);
    const float * src0_base = (const float *)tensor_data(src0);
    
    // TRACE: Log memory base addresses for every thread
    NUMA_LOG_TRACE("ROPE TRACE: Thread %d (NUMA %d) memory setup: dst_base=%p, src0_base=%p, shared=%s, tensor_shape=[%lld,%lld,%lld,%lld]", 
                   ith, ggml_current_numa_node, (void*)dst_base, (void*)src0_base,
                   ggml_numa_shared_result_tensor_data ? "YES" : "NO",
                   (long long)ne0, (long long)ne1, (long long)ne2, (long long)ne3);
    
    // Debug the dst_base pointer (first thread per NUMA node only)
    if (ith == 0) {
        NUMA_LOG_DEBUG("Thread %d (NUMA node %d) processing sequences [%d,%d), dst_base=%p, shared=%s", 
                       ith, ggml_current_numa_node, i2_start, i2_end, (void*)dst_base,
                       ggml_numa_shared_result_tensor_data ? "YES" : "NO");
    }
    
    // ROPE variant flags
    const float theta_scale = powf(freq_base, -2.0f/n_dims);
    
    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
    
    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
    const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;
    
    if (is_mrope) {
        GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
    }
    
    if (is_vision) {
        GGML_ASSERT(n_dims == ne0/2);
    }
    
    // Frequency factors for extended context
    const float * freq_factors = NULL;
    if (src2 != NULL) {
        GGML_ASSERT(src2->type == GGML_TYPE_F32);
        GGML_ASSERT(src2->ne[0] >= n_dims / 2);
        freq_factors = (const float *) tensor_data(src2);
    }
    
    // Sin sign for forward/backward pass
    const float sin_sign = forward ? 1.0f : -1.0f;
    
    const int32_t * pos = (const int32_t *) tensor_data(src1);
    
    // Allocate and initialize cache per thread
    float * cache = (float *) params->wdata + (ne0 + CACHE_LINE_SIZE_F32) * ith;
    memset(cache, 0, ne0 * sizeof(float));
    
    // ROPE main computation loop - Step 4: Threads with work do calculations
    for (int64_t i3 = 0; i3 < ne3; i3++) { // batch
        for (int64_t i2 = i2_start; i2 < i2_end; i2++) { // sequences - ONLY process assigned sequences
            
            // TRACE: Log sequence processing start for every thread
            NUMA_LOG_TRACE("ROPE TRACE: Thread %d (NUMA %d) starting sequence i2=%lld", 
                           ith, ggml_current_numa_node, (long long)i2);
            
            // Debug ROPE variant detection
            if (ith == 0 && i2 == i2_start) {
                NUMA_LOG_DEBUG("ROPE VARIANT: mode=%d is_neox=%s is_mrope=%s is_vision=%s (NUMA node %d, thread %d)", 
                              mode, is_neox ? "true" : "false", is_mrope ? "true" : "false", is_vision ? "true" : "false",
                              ggml_current_numa_node, ith);
            }
            
            // Initialize cache for this sequence position
            if (!is_mrope) {
                const int64_t p = pos[i2];
                
                // TRACE: Log position access for every thread
                NUMA_LOG_TRACE("ROPE TRACE: Thread %d (NUMA %d) accessing pos[%lld]=%lld", 
                               ith, ggml_current_numa_node, (long long)i2, (long long)p);
                
                ggml_rope_cache_init((float)p, freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
            } else {
                const int64_t p_t = pos[i2];
                const int64_t p_h = pos[i2 + ne2];
                const int64_t p_w = pos[i2 + ne2 * 2];
                const int64_t p_e = pos[i2 + ne2 * 3];
                ggml_mrope_cache_init(
                    (float)p_t, (float)p_h, (float)p_w, (float)p_e, sections, is_vision,
                    freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
            }
            
            // Process ALL attention heads for this sequence (no slicing within sequence)
            for (int64_t i1 = 0; i1 < ne1; i1++) { // attention heads
                
                // TRACE: Log head processing for every thread
                NUMA_LOG_TRACE("ROPE TRACE: Thread %d (NUMA %d) processing head i1=%lld for sequence i2=%lld", 
                               ith, ggml_current_numa_node, (long long)i1, (long long)i2);
                
                // Apply rotation based on ROPE variant
                if (is_neox || is_mrope) {
                    if (is_vision) {
                        // Vision ROPE with NEOX layout
                        for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                            const int64_t ic = i0/2;
                            
                            const float cos_theta = cache[i0 + 0];
                            const float sin_theta = cache[i0 + 1];
                            
                            const float * const src = (float *)((char *) src0_base + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
                            float * dst_data  = (float *)((char *) dst_base + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);
                            
                            const float x0 = src[0];
                            const float x1 = src[n_dims];
                            
                            dst_data[0]      = x0*cos_theta - x1*sin_theta;
                            dst_data[n_dims] = x0*sin_theta + x1*cos_theta;
                        }
                    } else {
                        // NEOX ROPE (half-dimension pairs)
                        if (i2 == i2_start) {
                            NUMA_LOG_DEBUG("NEOX ROPE: Starting processing on NUMA node %d, thread %d, sequence range [%d,%d)", 
                                          ggml_current_numa_node, ith, i2_start, i2_end);
                        }
                        
                        for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                            const float cos_theta = cache[i0 + 0];
                            const float sin_theta = cache[i0 + 1];
                            
                            // NEOX ROPE: Elements at i0 and i0+n_dims/2, NOT using ic index
                            const float * const src = (float *)((char *) src0_base + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                            float * dst_data  = (float *)((char *) dst_base + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);
                            
                            // TRACE: Log detailed memory access patterns
                            NUMA_LOG_TRACE("ROPE TRACE: Thread %d (NUMA %d) NEOX element i0=%lld: src_addr=%p, dst_addr=%p, cos=%f, sin=%f", 
                                           ith, ggml_current_numa_node, (long long)i0, (void*)src, (void*)dst_data, cos_theta, sin_theta);
                            
                            const float x0 = src[0];
                            const float x1 = src[n_dims/2];
                            
                            // TRACE: Log input values and calculated output values
                            NUMA_LOG_TRACE("ROPE TRACE: Thread %d (NUMA %d) NEOX computation: x0=%f (from src[0]), x1=%f (from src[%d]), result0=%f, result1=%f", 
                                           ith, ggml_current_numa_node, x0, x1, (int)(n_dims/2), 
                                           x0*cos_theta - x1*sin_theta, x0*sin_theta + x1*cos_theta);
                            
                            // Debug output for specific elements that are failing
                            if (i3 == 0 && ((i2 >= 0 && i2 <= 3) || (i2 >= 12 && i2 <= 15)) && i1 == 0 && i0 == 0) {
                                size_t linear_idx = i3*(ne2*ne1*ne0) + i2*(ne1*ne0) + i1*ne0 + i0;
                                NUMA_LOG_DEBUG("NEOX ROPE ELEMENT: seq=%d head=%d elem=%d thread=%d NUMA=%d linear_idx=%zu x0=%f x1=%f cos=%f sin=%f -> dst[0]=%f dst[n_dims/2]=%f dst_addr=%p n_dims=%d", 
                                              (int)i2, (int)i1, (int)i0, ith, ggml_current_numa_node, linear_idx, x0, x1, cos_theta, sin_theta,
                                              x0*cos_theta - x1*sin_theta, x0*sin_theta + x1*cos_theta, (void*)dst_data, (int)n_dims);
                            }
                            
                            // Perform the actual writes and log them
                            float result0 = x0*cos_theta - x1*sin_theta;
                            float result1 = x0*sin_theta + x1*cos_theta;
                            
                            dst_data[0]        = result0;
                            dst_data[n_dims/2] = result1;
                            
                            // TRACE: Log the actual memory writes with destination addresses
                            NUMA_LOG_TRACE("ROPE TRACE: Thread %d (NUMA %d) NEOX writes: dst_data[0]=%f written to %p, dst_data[%d]=%f written to %p", 
                                           ith, ggml_current_numa_node, result0, (void*)&dst_data[0], 
                                           (int)(n_dims/2), result1, (void*)&dst_data[n_dims/2]);
                        }
                    }
                } else {
                    // Standard ROPE (adjacent pairs)
                    if (i2 == i2_start) {
                        NUMA_LOG_DEBUG("STANDARD ROPE: Starting processing on NUMA node %d, thread %d, sequence range [%d,%d)", 
                                      ggml_current_numa_node, ith, i2_start, i2_end);
                    }
                    
                    for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                        const float cos_theta = cache[i0 + 0];
                        const float sin_theta = cache[i0 + 1];
                        
                        const float * const src = (float *)((char *) src0_base + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                        float * dst_data  = (float *)((char *) dst_base + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);
                        
                        const float x0 = src[0];
                        const float x1 = src[1];
                        
                        // Debug output for specific elements that are failing
                        if (i3 == 0 && i2 == 12 && i1 == 0 && i0 == 0) {
                            size_t linear_idx = i3*(ne2*ne1*ne0) + i2*(ne1*ne0) + i1*ne0 + i0;
                            NUMA_LOG_DEBUG("STANDARD ROPE ELEMENT: seq=%d head=%d elem=%d thread=%d linear_idx=%zu x0=%f x1=%f cos=%f sin=%f -> dst[0]=%f dst[1]=%f dst_addr=%p", 
                                          (int)i2, (int)i1, (int)i0, ith, linear_idx, x0, x1, cos_theta, sin_theta,
                                          x0*cos_theta - x1*sin_theta, x0*sin_theta + x1*cos_theta, (void*)dst_data);
                        }
                        
                        const float result0 = x0*cos_theta - x1*sin_theta;
                        const float result1 = x0*sin_theta + x1*cos_theta;
                        
                        dst_data[0] = result0;
                        dst_data[1] = result1;
                        
                        // TRACE: Log every single computation for detailed analysis
                        if (i2 >= i2_start && i2 < i2_end) {
                            NUMA_LOG_TRACE("ROPE TRACE: Thread %d (NUMA %d) STANDARD computation: seq=%d head=%d elem=%d inputs(x0=%f,x1=%f) cache(cos=%f,sin=%f) outputs(result0=%f,result1=%f)", 
                                           ith, ggml_current_numa_node, (int)i2, (int)i1, (int)i0, x0, x1, cos_theta, sin_theta, result0, result1);
                            
                            // TRACE: Log the actual memory writes with destination addresses
                            NUMA_LOG_TRACE("ROPE TRACE: Thread %d (NUMA %d) STANDARD writes: dst_data[0]=%f written to %p, dst_data[1]=%f written to %p", 
                                           ith, ggml_current_numa_node, result0, (void*)&dst_data[0], result1, (void*)&dst_data[1]);
                        }
                    }
                }
                
                // Handle remaining dimensions for vision ROPE
                if (is_vision) {
                    for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
                        const int64_t ic = i0/2;
                        
                        const float cos_theta = cache[i0 + 0];
                        const float sin_theta = cache[i0 + 1];
                        
                        const float * const src = (float *)((char *) src0_base + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
                        float * dst_data  = (float *)((char *) dst_base + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);
                        
                        const float x0 = src[0];
                        const float x1 = src[n_dims];
                        
                        dst_data[0]      = x0*cos_theta - x1*sin_theta;
                        dst_data[n_dims] = x0*sin_theta + x1*cos_theta;
                    }
                } else {
                    // Copy unrotated dimensions
                    for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
                        const float * const src = (float *)((char *) src0_base + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                        float * dst_data  = (float *)((char *) dst_base + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);
                        
                        dst_data[0] = src[0];
                        dst_data[1] = src[1];
                    }
                }
            }
        }
    }
    
    // Log completion for threads that did work
    NUMA_LOG_TRACE("Processed sequences %d-%d on NUMA node %d, thread %d/%d", 
                   i2_start, i2_end, ggml_current_numa_node, ith, nth);

barrier_wait:
    ; // Empty statement to satisfy C syntax requirements - labels must be followed by statements
    // CRITICAL: Synchronization barrier to ensure all writes complete before operation finishes
    if (ggml_numa_is_data_parallel_execution || nth > 1) {
        #ifdef GGML_USE_OPENMP
        #pragma omp barrier
        #endif
        NUMA_LOG_DEBUG("ROPE F32: Thread %d/%d (NUMA node %d) completed operation using OpenMP barrier", 
                      ith, nth, ggml_current_numa_node);
    } else {
        NUMA_LOG_DEBUG("ROPE F32: Single thread execution, skipping operation completion barrier");
    }
    
    return GGML_STATUS_SUCCESS;
}

// ============================================================================
// ROPE Kernel Implementation for F16
// ============================================================================

/**
 * ROPE kernel for F16 type tensors (similar to F32 but with type conversion)
 */
static enum ggml_status ggml_numa_kernel_rope_f16_execute(void * work_context, 
                                                          struct ggml_compute_params * params,
                                                          const bool forward) {
    struct ggml_tensor * dst = (struct ggml_tensor *)work_context;
    
    // Validate inputs
    NUMA_ASSERT(dst != NULL, "Destination tensor cannot be null");
    NUMA_ASSERT(dst->src[0] != NULL, "Source tensor 0 cannot be null");
    NUMA_ASSERT(dst->src[1] != NULL, "Source tensor 1 (positions) cannot be null");
    NUMA_ASSERT(params != NULL, "Compute params cannot be null");
    
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * src2 = dst->src[2];
    
    // Extract ROPE parameters from op_params (same as F32)
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    int sections[4];
    
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];
    
    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
    memcpy(&sections,    (int32_t *) dst->op_params + 11, sizeof(int)*4);
    
    // Get tensor dimensions
    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t ne3 = dst->ne[3];
    
    const size_t nb00 = dst->src[0]->nb[0];
    const size_t nb01 = dst->src[0]->nb[1];
    const size_t nb02 = dst->src[0]->nb[2];
    const size_t nb03 = dst->src[0]->nb[3];
    
    const size_t nb0 = dst->nb[0];
    const size_t nb1 = dst->nb[1];
    const size_t nb2 = dst->nb[2];
    const size_t nb3 = dst->nb[3];
    
    GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));
    GGML_ASSERT(n_dims <= ne0);
    GGML_ASSERT(n_dims % 2 == 0);
    
    // Get NUMA execution context
    
    // Strategy logging - only log once per operation (thread 0 of NUMA node 0)
    if (params->ith == 0 && ggml_current_numa_node == 0) {
        if (ggml_numa_is_data_parallel_execution) {
            NUMA_LOG_STRATEGY_DATA_PARALLEL("ROPE");
        } else {
            NUMA_LOG_STRATEGY_SINGLE_MULTI("ROPE");
        }
    }
    
    // Calculate threading parameters FIRST
    const int ith = params->ith;
    const int nth = params->nth;
    
    const int nr = ggml_nrows(dst);
    
    // Calculate NUMA data slice for data-parallel execution
    int numa_start_row = 0, numa_end_row = nr;
    
    if (ggml_numa_is_data_parallel_execution) {
        int rows_per_node = nr / ggml_numa_total_nodes_for_data_parallel;
        numa_start_row = ggml_current_numa_node * rows_per_node;
        numa_end_row = (ggml_current_numa_node == ggml_numa_total_nodes_for_data_parallel - 1) ? 
                       nr : numa_start_row + rows_per_node;
    }
    
    // Calculate thread slice within NUMA slice
    int numa_rows = numa_end_row - numa_start_row;
    int rows_per_thread = (numa_rows + nth - 1) / nth;
    int ir0 = numa_start_row + (ith * rows_per_thread);
    int ir1 = MIN(ir0 + rows_per_thread, numa_end_row);
    
    // CRITICAL FIX: Skip threads with no work (when more threads than rows)
    if (ir0 >= numa_end_row) {
        NUMA_LOG_TRACE("Thread %d/%d (NUMA node %d) has no work: ir0=%d >= numa_end_row=%d", 
                       ith, nth, ggml_current_numa_node, ir0, numa_end_row);
        goto barrier_wait;  // Skip to barrier to maintain synchronization
    }
    
    // ONLY access tensor data AFTER validating thread has work
    // Use shared memory optimization for data-parallel execution
    ggml_fp16_t * dst_base = ggml_numa_shared_result_tensor_data ? 
                             (ggml_fp16_t *)ggml_numa_shared_result_tensor_data : 
                             (ggml_fp16_t *)tensor_data(dst);
    const ggml_fp16_t * src0_base = (const ggml_fp16_t *)tensor_data(src0);
    
    // ROPE variant flags
    const float theta_scale = powf(freq_base, -2.0f/n_dims);
    
    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
    
    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
    const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;
    
    if (is_mrope) {
        GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
    }
    
    if (is_vision) {
        GGML_ASSERT(n_dims == ne0/2);
    }
    
    // Frequency factors for extended context
    const float * freq_factors = NULL;
    if (src2 != NULL) {
        GGML_ASSERT(src2->type == GGML_TYPE_F32);
        GGML_ASSERT(src2->ne[0] >= n_dims / 2);
        freq_factors = (const float *) tensor_data(src2);
    }
    
    // Sin sign for forward/backward pass
    const float sin_sign = forward ? 1.0f : -1.0f;
    
    const int32_t * pos = (const int32_t *) tensor_data(src1);
    
    // Allocate cache per thread
    float * cache = (float *) params->wdata + (ne0 + CACHE_LINE_SIZE_F32) * ith;
    
    // CRITICAL FIX: Zero-initialize cache to prevent non-deterministic behavior
    memset(cache, 0, ne0 * sizeof(float));
    
    // Process tensor slices
    // FIXED: Initialize ir to match the global row counting pattern from reference implementation
    int ir = 0;
    for (int64_t i3 = 0; i3 < ne3; i3++) { // batch
        for (int64_t i2 = 0; i2 < ne2; i2++) { // sequence length
            
            // Initialize cache for this sequence position
            if (!is_mrope) {
                const int64_t p = pos[i2];
                ggml_rope_cache_init((float)p, freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
            } else {
                const int64_t p_t = pos[i2];
                const int64_t p_h = pos[i2 + ne2];
                const int64_t p_w = pos[i2 + ne2 * 2];
                const int64_t p_e = pos[i2 + ne2 * 3];
                ggml_mrope_cache_init(
                    (float)p_t, (float)p_h, (float)p_w, (float)p_e, sections, is_vision,
                    freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
            }
            
            for (int64_t i1 = 0; i1 < ne1; i1++) { // attention heads
                if (ir++ < ir0) continue;
                if (ir > ir1) break;
                
                // Apply rotation based on ROPE variant (with F16 conversions)
                if (is_neox || is_mrope) {
                    if (is_vision) {
                        // Vision ROPE with NEOX layout
                        for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                            const int64_t ic = i0/2;
                            
                            const float cos_theta = cache[i0 + 0];
                            const float sin_theta = cache[i0 + 1];
                            
                            const ggml_fp16_t * const src = (ggml_fp16_t *)((char *) src0_base + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
                            ggml_fp16_t * dst_data  = (ggml_fp16_t *)((char *) dst_base + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);
                            
                            const float x0 = GGML_FP16_TO_FP32(src[0]);
                            const float x1 = GGML_FP16_TO_FP32(src[n_dims]);
                            
                            dst_data[0]      = GGML_FP32_TO_FP16(x0*cos_theta - x1*sin_theta);
                            dst_data[n_dims] = GGML_FP32_TO_FP16(x0*sin_theta + x1*cos_theta);
                        }
                    } else {
                        // NEOX ROPE (half-dimension pairs)
                        for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                            const int64_t ic = i0/2;
                            
                            const float cos_theta = cache[i0 + 0];
                            const float sin_theta = cache[i0 + 1];
                            
                            const ggml_fp16_t * const src = (ggml_fp16_t *)((char *) src0_base + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
                            ggml_fp16_t * dst_data  = (ggml_fp16_t *)((char *) dst_base + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);
                            
                            const float x0 = GGML_FP16_TO_FP32(src[0]);
                            const float x1 = GGML_FP16_TO_FP32(src[n_dims/2]);
                            
                            dst_data[0]        = GGML_FP32_TO_FP16(x0*cos_theta - x1*sin_theta);
                            dst_data[n_dims/2] = GGML_FP32_TO_FP16(x0*sin_theta + x1*cos_theta);
                        }
                    }
                } else {
                    // Standard ROPE (adjacent pairs)
                    for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                        const float cos_theta = cache[i0 + 0];
                        const float sin_theta = cache[i0 + 1];
                        
                        const ggml_fp16_t * const src = (ggml_fp16_t *)((char *) src0_base + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                        ggml_fp16_t * dst_data  = (ggml_fp16_t *)((char *) dst_base + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);
                        
                        const float x0 = GGML_FP16_TO_FP32(src[0]);
                        const float x1 = GGML_FP16_TO_FP32(src[1]);
                        
                        dst_data[0] = GGML_FP32_TO_FP16(x0*cos_theta - x1*sin_theta);
                        dst_data[1] = GGML_FP32_TO_FP16(x0*sin_theta + x1*cos_theta);
                    }
                }
                
                // Handle remaining dimensions for vision ROPE
                if (is_vision) {
                    for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
                        const int64_t ic = i0/2;
                        
                        const float cos_theta = cache[i0 + 0];
                        const float sin_theta = cache[i0 + 1];
                        
                        const ggml_fp16_t * const src = (ggml_fp16_t *)((char *) src0_base + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
                        ggml_fp16_t * dst_data  = (ggml_fp16_t *)((char *) dst_base + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);
                        
                        const float x0 = GGML_FP16_TO_FP32(src[0]);
                        const float x1 = GGML_FP16_TO_FP32(src[n_dims]);
                        
                        dst_data[0]      = GGML_FP32_TO_FP16(x0*cos_theta - x1*sin_theta);
                        dst_data[n_dims] = GGML_FP32_TO_FP16(x0*sin_theta + x1*cos_theta);
                    }
                } else {
                    // Copy unrotated dimensions
                    for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
                        const ggml_fp16_t * const src = (ggml_fp16_t *)((char *) src0_base + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                        ggml_fp16_t * dst_data  = (ggml_fp16_t *)((char *) dst_base + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);
                        
                        dst_data[0] = src[0];
                        dst_data[1] = src[1];
                    }
                }
            }
        }
    }
    
    // TODO: Update F16 function to use sequence-based slicing like F32
    NUMA_LOG_TRACE("Processed F16 tensor on NUMA node %d, thread %d/%d", 
                   ggml_current_numa_node, ith, nth);
    
barrier_wait:
    ; // Empty statement to satisfy C syntax requirements - labels must be followed by statements
    // CRITICAL: Synchronization barrier to ensure all writes complete before operation finishes
    if (ggml_numa_is_data_parallel_execution || nth > 1) {
        #ifdef GGML_USE_OPENMP
        #pragma omp barrier
        #endif
        NUMA_LOG_DEBUG("ROPE F16: Thread %d/%d (NUMA node %d) completed operation using OpenMP barrier", 
                      ith, nth, ggml_current_numa_node);
    } else {
        NUMA_LOG_DEBUG("ROPE F16: Single thread execution, skipping operation completion barrier");
    }
    
    return GGML_STATUS_SUCCESS;
}

// ============================================================================
// Main ROPE Kernel Execute Function
// ============================================================================

/**
 * Main ROPE kernel execution function with type dispatch
 */
enum ggml_status ggml_numa_kernel_rope_execute(void * work_context, struct ggml_compute_params * params) {
    struct ggml_tensor * tensor = (struct ggml_tensor *)work_context;
    
    // Validate inputs
    NUMA_ASSERT(tensor != NULL, "Tensor cannot be null");
    NUMA_ASSERT(tensor->src[0] != NULL, "Source tensor 0 cannot be null");
    NUMA_ASSERT(tensor->src[1] != NULL, "Source tensor 1 (positions) cannot be null");
    NUMA_ASSERT(params != NULL, "Compute params cannot be null");
    
    const struct ggml_tensor * src0 = tensor->src[0];
    
    // Dispatch based on tensor type
    switch (src0->type) {
        case GGML_TYPE_F32:
            return ggml_numa_kernel_rope_f32_execute(work_context, params, true);
            
        case GGML_TYPE_F16:
            return ggml_numa_kernel_rope_f16_execute(work_context, params, true);
            
        default:
            NUMA_LOG_ERROR("Unsupported tensor type for ROPE: %d", src0->type);
            return GGML_STATUS_FAILED;
    }
}

// ============================================================================
// ROPE Kernel Registration Functions
// ============================================================================

/**
 * Query function for ROPE kernel strategy selection
 */
ggml_numa_kernel_query_result_t ggml_numa_kernel_rope_query(const struct ggml_tensor * tensor) {
    ggml_numa_kernel_query_result_t result = {0};
    
    if (!tensor || !tensor->src[0]) {
        result.supported = false;
        return result;
    }
    
    // Get cache entry for this operation
    const ggml_numa_kernel_cache_entry_t * cache_entry = ggml_numa_lookup_kernel_direct(GGML_OP_ROPE);
    if (!cache_entry || !cache_entry->supported) {
        result.supported = false;
        return result;
    }
    
    // Calculate total elements for strategy selection
    size_t total_elements = ggml_nelements(tensor);
    
    // Use shared macro for unified strategy selection
    ggml_numa_execution_strategy_t selected_strategy;
    NUMA_SELECT_STRATEGY_FROM_CACHE(cache_entry, total_elements, selected_strategy);
    
    // Calculate work buffer size needed for ROPE cache
    // ROPE needs cache space: (ne0 + CACHE_LINE_SIZE_F32) * sizeof(float) per thread
    const size_t ne0 = tensor->ne[0];
    const size_t work_buffer_size = (ne0 + CACHE_LINE_SIZE_F32) * sizeof(float);
    
    // Set result with proper work function selection
    result.supported = true;
    result.strategy = selected_strategy;
    
    // Select work function based on strategy - all strategies now supported
    if (selected_strategy.on_node_strategy == NUMA_ON_NODE_STRATEGY_SINGLE_THREAD) {
        result.work_function = cache_entry->work_funcs.single_single_fn;
    } else if (selected_strategy.node_strategy == NUMA_NODE_STRATEGY_SINGLE) {
        result.work_function = cache_entry->work_funcs.single_multi_fn;
    } else {
        // Data-parallel mode now properly supported with shared destination memory
        result.work_function = cache_entry->work_funcs.data_parallel_fn;
    }
    
    result.aggregation_function = NULL; // ROPE doesn't need aggregation
    result.aggregation_policy = GGML_NUMA_AGGREGATION_NONE;
    result.work_buffer_size_per_thread = work_buffer_size; // Proper work buffer size for ROPE
    result.efficiency_score = 0.9f; // High efficiency for ROPE operations
    result.kernel_name = "NUMA ROPE Kernel";
    result.aggregation_user_data = NULL;
    
    NUMA_LOG_DEBUG("ROPE strategy selected: %d for %zu elements, work_buffer_size=%zu bytes", 
                   selected_strategy, total_elements, work_buffer_size);
    
    return result;
}

/**
 * Calculate work buffer size for ROPE operation
 * @param tensor - The tensor being processed
 * @param total_numa_nodes - Total NUMA nodes participating 
 * @param total_threads - Total threads participating across all nodes
 * @return Per-thread work buffer size in bytes
 */
size_t ggml_numa_kernel_rope_work_buffer_calc(const struct ggml_tensor * tensor, int total_numa_nodes, int total_threads) {
    if (!tensor) {
        return 0;
    }
    
    // Calculate work buffer size for ROPE cache
    const int64_t ne0 = tensor->ne[0];
    
    // Work buffer needs space for pre-computed cosine/sine cache per thread
    // Each thread needs cache for ne0 elements + cache line alignment
    size_t cache_size_per_thread = (ne0 + CACHE_LINE_SIZE_F32) * sizeof(float);
    
    // Total work buffer needs space for ALL threads that will execute on this node
    // The executor will call this function and the coordinator will use max threads per node
    size_t total_work_buffer_size = cache_size_per_thread * total_threads;
    
    NUMA_LOG_TRACE("ROPE work buffer: ne0=%lld, cache_size_per_thread=%zu bytes, total_threads=%d, total_size=%zu bytes", 
                   (long long)ne0, cache_size_per_thread, total_threads, total_work_buffer_size);
    
    return total_work_buffer_size;
}

/**
 * Register ROPE kernel with NUMA strategy array and work functions
 */
ggml_numa_kernel_registration_info_t ggml_numa_kernel_rope_register(void) {
    ggml_numa_kernel_registration_info_t info = {0};
    
    info.op_type = GGML_OP_ROPE;
    info.supported = true;
    info.kernel_name = "NUMA ROPE Kernel";
    
    // Strategy thresholds for ROPE operations
    // ROPE is now compatible with data-parallel execution after coordinator fix
    // Each NUMA node processes non-overlapping sequences with shared destination memory
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_SINGLE] = 128;     // Single thread below 1K elements
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_MULTI] = 1024;     // Multi-thread below 16K elements
    // Above 16K elements: data-parallel strategy with proper shared destination
    info.strategy_array.valid = true;
    
    // All strategies supported with proper coordinator setup
    info.work_funcs.single_single_fn = ggml_numa_kernel_rope_execute;
    info.work_funcs.single_multi_fn = ggml_numa_kernel_rope_execute;
    info.work_funcs.data_parallel_fn = ggml_numa_kernel_rope_execute;  // Re-enabled after coordinator fix
    info.work_funcs.valid = true;
    
    // Query function pointer for direct dispatch
    info.query_fn = (void*)ggml_numa_kernel_rope_query;
    
    // Work buffer calculation function
    info.work_buffer_calc_fn = (void*)ggml_numa_kernel_rope_work_buffer_calc;
    
    // ROPE doesn't need aggregation functions
    info.agg_funcs.single_single_fn = NULL;
    info.agg_funcs.single_multi_fn = NULL;
    info.agg_funcs.data_parallel_fn = NULL;
    info.agg_funcs.valid = false;
    
    return info;
}
