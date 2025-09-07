# ROPE Kernel Refactoring Plan

## 🎯 Objective
Eliminate ~80% of code duplication between F32 and F16 ROPE implementations while maintaining performance, type safety, and debugging capability.

## 📊 Current State Analysis
- **Total Lines**: ~1000 lines in rope.c
- **Duplicated Code**: ~600-700 lines (60-70%)
- **Major Duplication Areas**:
  1. Parameter extraction (~60 lines)
  2. Thread work distribution (~40 lines)
  3. Cache setup (~30 lines)
  4. Core computation loops (~200+ lines)
  5. Copy logic for non-rotated elements (~40 lines)

## 🏗️ Refactoring Strategy

### Phase 1: Common Data Structures and Helper Functions

#### 1.1 Parameter Structure
```c
typedef struct {
    // ROPE parameters
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    int sections[4];
    int n_dims, mode, n_ctx_orig;
    
    // Tensor dimensions
    int64_t ne0, ne1, ne2, ne3;
    
    // Source strides
    size_t src_nb0, src_nb1, src_nb2, src_nb3;
    
    // Destination strides
    size_t dst_nb0, dst_nb1, dst_nb2, dst_nb3;
    
    // ROPE variant flags
    bool is_mrope, is_vision, is_neox;
    
    // Sign for forward/backward pass
    float sin_sign;
} numa_rope_params_t;
```

#### 1.2 Thread Work Structure
```c
typedef struct {
    int ir0, ir1;        // Row range for this thread
    int i2_start, i2_end; // Sequence range
    bool has_work;       // Whether thread has any work
    int total_rows;      // Total rows to process
} numa_rope_thread_work_t;
```

#### 1.3 Helper Functions
```c
// Extract all ROPE parameters from tensor
static numa_rope_params_t extract_rope_params(const struct ggml_tensor* dst);

// Calculate thread work distribution
static numa_rope_thread_work_t calculate_rope_thread_work(
    const numa_rope_params_t* params, int ith, int nth);

// Setup cache for specific sequence position
static void setup_rope_cache_for_position(
    float* cache, const numa_rope_params_t* params,
    int64_t i1, int64_t i2, int64_t i3,
    const int32_t* pos, const float* freq_factors);
```

### Phase 2: Type-Generic Core Computation Macros

#### 2.1 Standard ROPE Computation
```c
#define ROPE_STANDARD_COMPUTE_CORE(TYPE, SRC_PTR, DST_PTR, TO_F32, FROM_F32, \
                                   cos_theta, sin_theta, trace_prefix) do { \
    const float x0 = TO_F32((SRC_PTR)[0]); \
    const float x1 = TO_F32((SRC_PTR)[1]); \
    \
    const float result0 = x0 * (cos_theta) - x1 * (sin_theta); \
    const float result1 = x0 * (sin_theta) + x1 * (cos_theta); \
    \
    (DST_PTR)[0] = FROM_F32(result0); \
    (DST_PTR)[1] = FROM_F32(result1); \
    \
    NUMA_LOG_TRACE("%s: x0=%f x1=%f cos=%f sin=%f -> r0=%f r1=%f", \
                   trace_prefix, (double)x0, (double)x1, (double)(cos_theta), \
                   (double)(sin_theta), (double)result0, (double)result1); \
} while(0)
```

#### 2.2 NEOX ROPE Computation
```c
#define ROPE_NEOX_COMPUTE_CORE(TYPE, SRC_PTR, DST_PTR, TO_F32, FROM_F32, \
                               n_dims, cos_theta, sin_theta, trace_prefix) do { \
    const float x0 = TO_F32((SRC_PTR)[0]); \
    const float x1 = TO_F32((SRC_PTR)[(n_dims)/2]); \
    \
    const float result0 = x0 * (cos_theta) - x1 * (sin_theta); \
    const float result1 = x0 * (sin_theta) + x1 * (cos_theta); \
    \
    (DST_PTR)[0] = FROM_F32(result0); \
    (DST_PTR)[(n_dims)/2] = FROM_F32(result1); \
    \
    NUMA_LOG_TRACE("%s NEOX: x0=%f x1=%f cos=%f sin=%f -> r0=%f r1=%f", \
                   trace_prefix, (double)x0, (double)x1, (double)(cos_theta), \
                   (double)(sin_theta), (double)result0, (double)result1); \
} while(0)
```

#### 2.3 Copy Non-Rotated Elements
```c
#define ROPE_COPY_NONROTATED(TYPE, SRC_PTR, DST_PTR) do { \
    (DST_PTR)[0] = (SRC_PTR)[0]; \
    (DST_PTR)[1] = (SRC_PTR)[1]; \
} while(0)
```

### Phase 3: Main Loop Template

#### 3.1 Unified Computation Loop
```c
#define ROPE_MAIN_COMPUTATION_LOOP(TYPE, SRC_BASE, DST_BASE, TO_F32, FROM_F32, ELEMENT_SIZE) do { \
    /* Process assigned rows */ \
    for (int ir = work.ir0; ir < work.ir1; ir++) { \
        /* Convert row index to 4D coordinates */ \
        const int64_t i3 = ir / (params->ne2 * params->ne1); \
        const int64_t remainder = ir % (params->ne2 * params->ne1); \
        const int64_t i2 = remainder / params->ne1; \
        const int64_t i1 = remainder % params->ne1; \
        \
        /* Set up cache for this position */ \
        setup_rope_cache_for_position(cache, params, i1, i2, i3, pos, freq_factors); \
        \
        /* Vision ROPE processing */ \
        if (params->is_vision) { \
            ROPE_VISION_COMPUTE_LOOP(TYPE, SRC_BASE, DST_BASE, TO_F32, FROM_F32, i3, i2, i1); \
        } \
        /* NEOX ROPE processing */ \
        else if (params->is_neox) { \
            ROPE_NEOX_COMPUTE_LOOP(TYPE, SRC_BASE, DST_BASE, TO_F32, FROM_F32, i3, i2, i1); \
        } \
        /* Standard ROPE processing */ \
        else { \
            ROPE_STANDARD_COMPUTE_LOOP(TYPE, SRC_BASE, DST_BASE, TO_F32, FROM_F32, i3, i2, i1); \
        } \
        \
        /* Copy non-rotated elements */ \
        ROPE_COPY_REMAINING_ELEMENTS(TYPE, SRC_BASE, DST_BASE, i3, i2, i1); \
    } \
} while(0)
```

### Phase 4: Unified Implementation Function

#### 4.1 Single Internal Implementation
```c
static enum ggml_status ggml_numa_kernel_rope_execute_internal(
    void * work_context, 
    struct ggml_compute_params * params,
    enum ggml_type tensor_type) {
    
    // Common validation and setup
    struct ggml_tensor * dst = (struct ggml_tensor *)work_context;
    NUMA_ASSERT(dst != NULL, "Destination tensor cannot be null");
    // ... other validations
    
    // Extract parameters (once, shared)
    numa_rope_params_t rope_params = extract_rope_params(dst);
    
    // Calculate thread work (once, shared)
    numa_rope_thread_work_t work = calculate_rope_thread_work(&rope_params, params->ith, params->nth);
    
    if (!work.has_work) {
        return GGML_STATUS_SUCCESS;
    }
    
    // Get shared memory pointers
    void* src_base = tensor_data(dst->src[0]);
    void* dst_base = ggml_numa_shared_result_tensor_data ? 
                     ggml_numa_shared_result_tensor_data : 
                     tensor_data(dst);
    
    // Get cache and position data
    float* cache = (float*)params->wdata + (rope_params.ne0 + CACHE_LINE_SIZE_F32) * params->ith;
    const int32_t* pos = (const int32_t*)tensor_data(dst->src[1]);
    const float* freq_factors = dst->src[2] ? (const float*)tensor_data(dst->src[2]) : NULL;
    
    // Type-specific dispatch
    switch (tensor_type) {
        case GGML_TYPE_F32:
            ROPE_MAIN_COMPUTATION_LOOP(float, src_base, dst_base, 
                                       (x), (x), sizeof(float));
            break;
            
        case GGML_TYPE_F16:
            ROPE_MAIN_COMPUTATION_LOOP(ggml_fp16_t, src_base, dst_base,
                                       GGML_FP16_TO_FP32, GGML_FP32_TO_FP16, 
                                       sizeof(ggml_fp16_t));
            break;
            
        default:
            NUMA_LOG_ERROR("Unsupported tensor type for ROPE: %d", tensor_type);
            return GGML_STATUS_FAILED;
    }
    
    return GGML_STATUS_SUCCESS;
}
```

#### 4.2 Thin Type-Specific Wrappers
```c
enum ggml_status ggml_numa_kernel_rope_f32_execute(void * work_context, struct ggml_compute_params * params) {
    return ggml_numa_kernel_rope_execute_internal(work_context, params, GGML_TYPE_F32);
}

enum ggml_status ggml_numa_kernel_rope_f16_execute(void * work_context, struct ggml_compute_params * params) {
    return ggml_numa_kernel_rope_execute_internal(work_context, params, GGML_TYPE_F16);
}
```

## 📈 Expected Benefits

### Code Reduction
- **Before**: ~1000 lines
- **After**: ~400-500 lines (50-60% reduction)
- **Eliminated Duplication**: ~600-700 lines

### Maintainability Improvements
- ✅ Single source of truth for ROPE logic
- ✅ Type-safe through compile-time templates
- ✅ Easier to add new quantization types
- ✅ Easier to add new ROPE variants
- ✅ Centralized parameter handling
- ✅ Consistent debugging across types

### Performance Characteristics
- 🟢 **No Performance Loss**: Macros expand to identical code
- 🟢 **Same NUMA Efficiency**: No changes to parallelization strategy
- 🟢 **Same Memory Access Patterns**: Preserved data layout optimization
- 🟢 **Same SIMD Opportunities**: Compiler optimizations unchanged

### Risk Mitigation
- 🛡️ **Preserves Existing Tests**: All current tests continue to pass
- 🛡️ **Maintains API Compatibility**: External interfaces unchanged
- 🛡️ **Gradual Migration**: Can implement incrementally
- 🛡️ **Rollback Capability**: Easy to revert if issues arise

## 🗓️ Implementation Timeline

### Phase 1: Infrastructure (1-2 days)
- [ ] Define `numa_rope_params_t` structure
- [ ] Implement `extract_rope_params()` helper
- [ ] Implement `calculate_rope_thread_work()` helper
- [ ] Create basic computation macros

### Phase 2: Core Templates (1-2 days)
- [ ] Implement `ROPE_MAIN_COMPUTATION_LOOP` macro
- [ ] Create variant-specific sub-macros
- [ ] Implement `setup_rope_cache_for_position()` helper

### Phase 3: Unified Implementation (1 day)
- [ ] Implement `ggml_numa_kernel_rope_execute_internal()`
- [ ] Convert F32/F16 functions to thin wrappers
- [ ] Update header file with new prototypes

### Phase 4: Testing & Validation (1 day)
- [ ] Run full ROPE test suite
- [ ] Verify mathematical correctness
- [ ] Benchmark performance
- [ ] Test all ROPE variants

### Phase 5: Documentation & Cleanup (0.5 days)
- [ ] Update documentation
- [ ] Clean up old code
- [ ] Add refactoring notes

**Total Estimated Time**: 4.5-6.5 days

## 🧪 Validation Plan

### 1. Mathematical Correctness
- [ ] All existing ROPE tests must pass
- [ ] Bit-for-bit identical results with original implementation
- [ ] Cross-validation between F32 and F16 implementations

### 2. Performance Validation
- [ ] Benchmark with original implementation
- [ ] Verify NUMA scaling characteristics
- [ ] Measure compilation time impact

### 3. Integration Testing
- [ ] Run full llama.cpp integration tests
- [ ] Test with real models
- [ ] Validate all ROPE variants (Standard, NEOX, Vision, mROPE)

## 🎯 Success Criteria

✅ **Primary Goals**:
- Reduce ROPE implementation by 50-60% lines of code
- Maintain 100% mathematical equivalence
- Preserve all existing functionality
- No performance degradation

✅ **Secondary Goals**:
- Improved code maintainability
- Easier addition of new quantization types
- Better debugging experience
- Consistent coding patterns across NUMA kernels

This refactoring will establish a template for similar optimizations across other NUMA kernels (ADD, MUL, RMS_NORM, etc.).
