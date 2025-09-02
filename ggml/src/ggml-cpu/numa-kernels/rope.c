/**
 * @file rope.c
 * @brief NUMA ROPE (Rotary Position Embedding) Kernel Implementation
 * 
 * ============================================================================
 * NUMA KERNEL TEMPLATE: COMPLEX OPERATIONS (ROPE)
 * ============================================================================
 * 
 * This file implements NUMA kernels for ROPE operations based on the complex
 * operations template. ROPE requires sophisticated parallelization due to:
 * - Multiple operation modes (standard, NEOX, mrope, vision)
 * - Complex parameter handling and cache computation
 * - Multi-dimensional data access patterns
 * - Forward and backward pass support
 * 
 * MATHEMATICAL OPERATION (ROPE):
 * =============================
 * 
 * ROPE applies rotary position embeddings to input tensors using:
 * - Rotation matrices computed from position and frequency parameters
 * - Element-wise rotation transformations: x' = x*cos(θ) - y*sin(θ), y' = x*sin(θ) + y*cos(θ)
 * - Support for multiple variants (standard, NEOX, multi-modal, vision)
 * 
 * NUMA PARALLELIZATION STRATEGY:
 * ==============================
 * 
 * ROPE operations are parallelized by distributing attention heads (ne1) and
 * batch dimensions (ne2, ne3) across NUMA nodes:
 * - Each NUMA node processes a slice of the batch dimensions
 * - Within each node, threads process different attention heads
 * - Cache computation is done per-sequence to maintain correctness
 * - Position and frequency parameters are shared across all nodes
 */

#include "rope.h"
#include "../ggml-impl.h"
#include "../ggml-quants.h"
#include "../ggml-cpu-impl.h"
#include "../ops.h"
#include <math.h>

// ============================================================================
// ROPE Support Functions (Copied from ops.cpp)
// ============================================================================

static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / MAX(0.001f, high - low);
    return 1 - MIN(1, MAX(0, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
static void rope_yarn(
    float theta_extrap, float freq_scale, float corr_dims[2], int64_t i0, float ext_factor, float mscale,
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

static void ggml_rope_cache_init(
     int64_t pos, float freq_scale, const float * freq_factors, float corr_dims[2], int64_t ne0, float ext_factor, float attn_factor,
     float * cache, float sin_sign, float theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    float theta = pos;
    for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0/2] : 1.0f;
        rope_yarn(
            theta/ff, freq_scale, corr_dims, i0, ext_factor, attn_factor, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        theta *= theta_scale;
    }
}

static void ggml_mrope_cache_init(
     int64_t p_t, int64_t p_h, int64_t p_w, int64_t p_e, int sections[4], bool is_vision,
     float freq_scale, const float * freq_factors, float corr_dims[2], int64_t ne0, float ext_factor, float attn_factor,
     float * cache, float sin_sign, float theta_scale) {
     
    // Convert positions to base theta values
    float theta_base_t = p_t;
    float theta_base_h = p_h;
    float theta_base_w = p_w;
    float theta_base_e = p_e;
    
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    float theta_t = theta_base_t;
    float theta_h = theta_base_h;
    float theta_w = theta_base_w;
    float theta_e = theta_base_e;  // extra position id for vision encoder
    int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    int sec_w = sections[1] + sections[0];
    int sec_e = sections[2] + sec_w;
    NUMA_ASSERT(sect_dims <= ne0, "Section dimensions must not exceed ne0");

    for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0/2] : 1.0f;

        int sector = (i0 / 2) % sect_dims;
        bool indep_sects = is_vision; // For vision, compute theta independently

        if (indep_sects) {
            // compute theta independently for each dim sections
            // (i.e. reset corresponding theta when `i0` go from one section to another)
            if (sector == 0) {
                theta_t = theta_base_t;
            }
            else if (sector == sections[0]) {
                theta_h = theta_base_h;;
            }
            else if (sector == sec_w) {
                theta_w = theta_base_w;
            }
            else if (sector == sec_e) {
                theta_e = theta_base_e;
            }
        }

        float theta = theta_t;
        if (sector >= sections[0] && sector < sec_w) {
            theta = theta_h;
        }
        else if (sector >= sec_w && sector < sec_w + sections[2]) {
            theta = theta_w;
        }
        else if (sector >= sec_w + sections[2]) {
            theta = theta_e;
        }

        rope_yarn(
            theta/ff, freq_scale, corr_dims, i0, ext_factor, attn_factor, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        theta_t *= theta_scale;
        theta_w *= theta_scale;
        theta_h *= theta_scale;
        theta_e *= theta_scale;
    }
}

// ============================================================================
// NUMA ROPE Kernel Implementation
// ============================================================================

/**
 * NUMA-aware ROPE kernel execution function
 * Handles all ROPE variants with optimal NUMA parallelization
 */
enum ggml_status ggml_numa_kernel_rope_execute(void * work_context, struct ggml_compute_params * params) {
    struct ggml_tensor * dst = (struct ggml_tensor *)work_context;
    
    NUMA_ASSERT(dst != NULL, "ROPE destination tensor cannot be null");
    NUMA_ASSERT(dst->src[0] != NULL, "ROPE source tensor cannot be null");
    NUMA_ASSERT(dst->src[1] != NULL, "ROPE position tensor cannot be null");
    NUMA_ASSERT(params != NULL, "Compute params cannot be null");
    
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * src2 = dst->src[2]; // freq_factors (optional)
    
    // Get NUMA execution context from thread-local variables
    extern __thread int ggml_current_numa_node;
    extern __thread int ggml_numa_total_nodes_for_data_parallel;
    extern __thread bool ggml_numa_is_data_parallel_execution;
    extern __thread void * ggml_numa_shared_result_tensor_data;
    
    NUMA_LOG_TRACE("ROPE kernel executing on NUMA node %d/%d (data_parallel=%s)",
                   ggml_current_numa_node, ggml_numa_total_nodes_for_data_parallel, 
                   ggml_numa_is_data_parallel_execution ? "true" : "false");
    
    // Extract ROPE operation parameters
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];
    
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    int sections[4];
    
    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
    memcpy(&sections,    (int32_t *) dst->op_params + 11, sizeof(int)*4);
    
    // Determine ROPE variant
    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
    const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;
    const bool forward = true; // NUMA kernels handle forward pass; backward handled separately
    
    // Validate ROPE constraints
    if (is_mrope) {
        NUMA_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0, 
                   "MROPE requires at least one section to be > 0");
    }
    
    if (is_vision) {
        NUMA_ASSERT(n_dims == src0->ne[0]/2, "Vision ROPE requires n_dims == ne0/2");
    }
    
    // Extract tensor dimensions
    const int64_t ne0 = dst->ne[0];  // head dimensions
    const int64_t ne1 = dst->ne[1];  // attention heads
    const int64_t ne2 = dst->ne[2];  // sequence length
    const int64_t ne3 = dst->ne[3];  // batch size
    
    const size_t nb0 = dst->nb[0];
    const size_t nb1 = dst->nb[1];
    const size_t nb2 = dst->nb[2];
    const size_t nb3 = dst->nb[3];
    
    const size_t nb00 = src0->nb[0];
    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];
    
    NUMA_ASSERT(n_dims <= ne0, "n_dims must be <= ne0");
    NUMA_ASSERT(n_dims % 2 == 0, "n_dims must be even");
    NUMA_ASSERT(nb00 == sizeof(float), "Source tensor must be float32");
    
    // Setup data pointers with shared memory optimization
    const float * src0_data = (const float *) tensor_data(src0);
    const int32_t * pos_data = (const int32_t *) tensor_data(src1);
    
    float * dst_data;
    if (ggml_numa_shared_result_tensor_data != NULL) {
        // Use shared result tensor memory for direct writes
        dst_data = (float *) ggml_numa_shared_result_tensor_data;
        NUMA_LOG_TRACE("Using shared result tensor memory for ROPE output");
    } else {
        // Fallback to local tensor data
        dst_data = (float *) tensor_data(dst);
    }
    
    // Optional frequency factors
    const float * freq_factors = NULL;
    if (src2 != NULL) {
        NUMA_ASSERT(src2->type == GGML_TYPE_F32, "Frequency factors must be float32");
        NUMA_ASSERT(src2->ne[0] >= n_dims / 2, "Insufficient frequency factors");
        freq_factors = (const float *) tensor_data(src2);
    }
    
    // Calculate correlation dimensions for YaRN
    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
    
    // NUMA parallelization: distribute batch dimensions across nodes
    const int64_t total_batch_size = ne1 * ne2 * ne3;  // heads * seq_len * batch
    int64_t batch_start = 0, batch_end = total_batch_size;
    
    if (ggml_numa_is_data_parallel_execution && ggml_numa_total_nodes_for_data_parallel > 1) {
        const int64_t batch_per_node = total_batch_size / ggml_numa_total_nodes_for_data_parallel;
        batch_start = ggml_current_numa_node * batch_per_node;
        batch_end = (ggml_current_numa_node == ggml_numa_total_nodes_for_data_parallel - 1) ? 
                   total_batch_size : batch_start + batch_per_node;
        
        NUMA_LOG_TRACE("ROPE NUMA slice: processing batch range [%ld, %ld) of %ld total",
                      batch_start, batch_end, total_batch_size);
    }
    
    // Compute backward process sign
    const float sin_sign = forward ? 1.0f : -1.0f;
    const float theta_scale = powf(freq_base, -2.0f/n_dims);
    
    // Thread allocation within this NUMA node
    const int ith = params->ith;
    const int nth = params->nth;
    
    // Calculate thread's work slice within the NUMA node's batch range
    const int64_t numa_batch_size = batch_end - batch_start;
    const int64_t batches_per_thread = (numa_batch_size + nth - 1) / nth;
    const int64_t thread_batch_start = batch_start + ith * batches_per_thread;
    const int64_t thread_batch_end = MIN(thread_batch_start + batches_per_thread, batch_end);
    
    NUMA_LOG_TRACE("ROPE thread %d/%d processing batch range [%ld, %ld)",
                  ith, nth, thread_batch_start, thread_batch_end);
    
    // Allocate thread-local cache for rotation coefficients
    float * cache = (float *) params->wdata + (ne0 + CACHE_LINE_SIZE_F32) * ith;
    
    // Process assigned batch range
    int64_t current_batch = 0;
    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                // Check if this batch element is in our thread's range
                if (current_batch < thread_batch_start) {
                    current_batch++;
                    continue;
                }
                if (current_batch >= thread_batch_end) {
                    goto batch_loop_end;
                }
                
                // Initialize cache for this sequence position
                if (!is_mrope) {
                    const int64_t p = pos_data[i2];
                    ggml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, 
                                       ext_factor, attn_factor, cache, sin_sign, theta_scale);
                } else {
                    const int64_t p_t = pos_data[i2];
                    const int64_t p_h = pos_data[i2 + ne2];
                    const int64_t p_w = pos_data[i2 + ne2 * 2];
                    const int64_t p_e = pos_data[i2 + ne2 * 3];
                    ggml_mrope_cache_init(p_t, p_h, p_w, p_e, sections, is_vision,
                                        freq_scale, freq_factors, corr_dims, ne0,
                                        ext_factor, attn_factor, cache, sin_sign, theta_scale);
                }
                
                // Apply ROPE transformation to this attention head
                if (is_neox || is_mrope) {
                    if (is_vision) {
                        // Vision ROPE variant
                        for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                            const int64_t ic = i0/2;
                            
                            const float cos_theta = cache[i0 + 0];
                            const float sin_theta = cache[i0 + 1];
                            
                            const float * const src = (const float *)((char *) src0_data + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
                            float * dst_ptr = (float *)((char *) dst_data + i3*nb3 + i2*nb2 + i1*nb1 + ic*nb0);
                            
                            const float x0 = src[0];
                            const float x1 = src[n_dims];
                            
                            dst_ptr[0]      = x0*cos_theta - x1*sin_theta;
                            dst_ptr[n_dims] = x0*sin_theta + x1*cos_theta;
                        }
                    } else {
                        // Standard NEOX/MROPE variant
                        for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                            const int64_t ic = i0/2;
                            
                            const float cos_theta = cache[i0 + 0];
                            const float sin_theta = cache[i0 + 1];
                            
                            const float * const src = (const float *)((char *) src0_data + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
                            float * dst_ptr = (float *)((char *) dst_data + i3*nb3 + i2*nb2 + i1*nb1 + ic*nb0);
                            
                            const float x0 = src[0];
                            const float x1 = src[n_dims/2];
                            
                            dst_ptr[0]        = x0*cos_theta - x1*sin_theta;
                            dst_ptr[n_dims/2] = x0*sin_theta + x1*cos_theta;
                        }
                    }
                } else {
                    // Standard ROPE variant
                    for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                        const float cos_theta = cache[i0 + 0];
                        const float sin_theta = cache[i0 + 1];
                        
                        const float * const src = (const float *)((char *) src0_data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                        float * dst_ptr = (float *)((char *) dst_data + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
                        
                        const float x0 = src[0];
                        const float x1 = src[1];
                        
                        dst_ptr[0] = x0*cos_theta - x1*sin_theta;
                        dst_ptr[1] = x0*sin_theta + x1*cos_theta;
                    }
                }
                
                // Handle remaining channels for vision mode
                if (is_vision) {
                    for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
                        const int64_t ic = i0/2;
                        
                        const float cos_theta = cache[i0 + 0];
                        const float sin_theta = cache[i0 + 1];
                        
                        const float * const src = (const float *)((char *) src0_data + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
                        float * dst_ptr = (float *)((char *) dst_data + i3*nb3 + i2*nb2 + i1*nb1 + ic*nb0);
                        
                        const float x0 = src[0];
                        const float x1 = src[n_dims];
                        
                        dst_ptr[0]      = x0*cos_theta - x1*sin_theta;
                        dst_ptr[n_dims] = x0*sin_theta + x1*cos_theta;
                    }
                } else {
                    // Copy remaining channels without rotation
                    for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
                        const float * const src = (const float *)((char *) src0_data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                        float * dst_ptr = (float *)((char *) dst_data + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
                        
                        dst_ptr[0] = src[0];
                        dst_ptr[1] = src[1];
                    }
                }
                
                current_batch++;
            }
        }
    }
    
batch_loop_end:
    NUMA_LOG_TRACE("ROPE kernel completed processing %ld batch elements on NUMA node %d",
                  current_batch - thread_batch_start, ggml_current_numa_node);
    
    return GGML_STATUS_SUCCESS;
}

// ============================================================================
// NUMA ROPE Kernel Registration
//============================================================================
// NUMA ROPE Kernel Query and Registration
// ============================================================================

/**
 * Query ROPE kernel for optimal execution strategy based on tensor characteristics.
 * ROPE operations involve complex trigonometric computations with cache operations,
 * so they benefit from NUMA parallelization even with smaller tensors.
 */
ggml_numa_kernel_query_result_t ggml_numa_kernel_rope_query(const struct ggml_tensor * tensor) {
    ggml_numa_kernel_query_result_t result = { .supported = false };
    
    // Validate this is a ROPE operation
    if (!tensor || tensor->op != GGML_OP_ROPE) {
        return result;
    }
    
    // ROPE operations require at least one source tensor
    if (!tensor->src[0]) {
        NUMA_LOG_DEBUG("ROPE query: Missing source tensor");
        return result;
    }
    
    // Calculate tensor dimensions for complexity assessment
    const int64_t ne0 = tensor->src[0]->ne[0];  // sequence length
    const int64_t ne1 = tensor->src[0]->ne[1];  // number of heads
    const int64_t ne2 = tensor->src[0]->ne[2];  // batch size
    const size_t total_elements = (size_t)ne0 * ne1 * ne2;
    
    // Check if this kernel is actually registered and supported
    if (!ggml_numa_is_kernel_supported(GGML_OP_ROPE)) {
        NUMA_LOG_DEBUG("ROPE kernel not supported - registration disabled");
        result.supported = false;
        return result;
    }
    
    // ROPE operations are compute-intensive with trigonometric calculations
    // Strategy selection based on element count thresholds
    
    result.supported = true;
    // ROPE requires work buffer for cache: (ne0 + CACHE_LINE_SIZE_F32) * sizeof(float) per thread
    result.work_buffer_size_per_thread = (ne0 + CACHE_LINE_SIZE_F32) * sizeof(float);
    result.work_function = ggml_numa_kernel_rope_execute;
    result.kernel_name = "NUMA ROPE Kernel";
    result.aggregation_policy = GGML_NUMA_AGGREGATION_NONE;  // Independent batch processing, no aggregation
    result.aggregation_function = NULL;
    result.aggregation_user_data = NULL;
    
    // Select strategy based on element count thresholds (ROPE benefits from parallelization early)
    ggml_numa_execution_strategy_t strategy = {0};
    if (total_elements < 512) {  // 512 elements - single thread
        strategy.node_strategy = NUMA_NODE_STRATEGY_SINGLE;
        strategy.on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD;
        result.efficiency_score = 0.90f;  // Good efficiency for small tensors
    } else if (total_elements < 16384) {  // 16K elements - multi-thread single node
        strategy.node_strategy = NUMA_NODE_STRATEGY_SINGLE;
        strategy.on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD;
        result.efficiency_score = 0.87f;  // Good efficiency for medium tensors
    } else {  // Above 16K elements - data-parallel strategy across NUMA nodes
        strategy.node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL;
        strategy.on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD;
        result.efficiency_score = 0.85f;  // Excellent for large tensors with complex computations
    }
    
    result.strategy = strategy;
    
    // Apply force strategy override if set
    ggml_numa_apply_kernel_force_strategy(&result, "ROPE", 
                                          ggml_numa_kernel_rope_execute, 
                                          ggml_numa_kernel_rope_execute,
                                          ggml_numa_kernel_rope_execute);
    
    NUMA_LOG_TRACE("ROPE query: elements=%zu, strategy=node:%d/thread:%d, efficiency=%.2f", 
                   total_elements, strategy.node_strategy, strategy.on_node_strategy, result.efficiency_score);
    
    return result;
}

ggml_numa_kernel_registration_info_t ggml_numa_kernel_rope_register(void) {
    ggml_numa_kernel_registration_info_t info = {0};
    
    info.op_type = GGML_OP_ROPE;
    info.supported = true;
    info.kernel_name = "NUMA ROPE Kernel";
    
    // Strategy thresholds for ROPE operations
    // ROPE benefits from parallelization even with smaller tensors due to complex cache operations
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_SINGLE] = 512;      // Single thread below 512 elements
    info.strategy_array.thresholds[NUMA_STRATEGY_IDX_SINGLE_MULTI] = 16384;     // Multi-thread below 16K elements
    // Above 16K elements: data-parallel strategy across NUMA nodes
    info.strategy_array.valid = true;
    
    // Function pointers for different execution strategies
    info.work_funcs.single_single_fn = ggml_numa_kernel_rope_execute;
    info.work_funcs.single_multi_fn = ggml_numa_kernel_rope_execute;
    info.work_funcs.data_parallel_fn = ggml_numa_kernel_rope_execute;
    info.work_funcs.valid = true;
    
    // ROPE does not require aggregation since each node processes independent batch slices
    info.agg_funcs.single_single_fn = NULL;
    info.agg_funcs.single_multi_fn = NULL;
    info.agg_funcs.data_parallel_fn = NULL;
    info.agg_funcs.valid = false;
    
    // ROPE is a computational operation, not a no-op
    info.is_noop = false;
    
    return info;
}
