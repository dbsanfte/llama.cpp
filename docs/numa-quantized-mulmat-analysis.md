# NUMA MUL_MAT Quantized Types Support Analysis

**Date**: 2025-08-26  
**Scope**: Quantized Type Support for NUMA MUL_MAT Kernel  
**Priority**: Critical - Required for production quantized model support  

## 📋 Current State Analysis

### ❌ **Current Limitations**
Our NUMA MUL_MAT kernel currently **REJECTS ALL non-F32 types**:
```c
// From numa-kernels/mul_mat.c:538
if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) {
    NUMA_LOG_DEBUG("MUL_MAT query: REJECTING - src0_type=%d, src1_type=%d (only F32 supported)", 
                   src0->type, src1->type);
    result.supported = false;
    return result;
}
```

### 🏗️ **Reference Implementation Patterns**

#### **1. Type Trait System**
The reference implementation uses `type_traits_cpu` to handle different quantized types:

```c
// Key type traits for quantized types:
[GGML_TYPE_Q4_0] = {
    .from_float     = quantize_row_q4_0,           // F32 → Q4_0 conversion
    .vec_dot        = ggml_vec_dot_q4_0_q8_0,      // Q4_0 × Q8_0 → F32
    .vec_dot_type   = GGML_TYPE_Q8_0,              // src1 conversion target  
    .nrows          = 1,                           // SIMD rows processed
},
[GGML_TYPE_Q8_0] = {
    .from_float     = quantize_row_q8_0,           // F32 → Q8_0 conversion
    .vec_dot        = ggml_vec_dot_q8_0_q8_0,      // Q8_0 × Q8_0 → F32
    .vec_dot_type   = GGML_TYPE_Q8_0,              // src1 conversion target
    .nrows          = 1,
},
```

#### **2. Type Conversion Logic**
```c
// From ggml-cpu.c:1235-1270
if (src1->type != vec_dot_type) {
    char * wdata = params->wdata;                  // Work buffer for conversion
    
    const size_t nbw0 = ggml_type_size(vec_dot_type);
    const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
    const size_t nbw2 = nbw1*ne11;
    const size_t nbw3 = nbw2*ne12;
    
    assert(params->wsize >= ne13*nbw3);            // Ensure sufficient work buffer
    GGML_ASSERT(src1->type == GGML_TYPE_F32);      // src1 must be F32 for conversion
    
    // Convert F32 src1 to vec_dot_type using from_float function
    for (int64_t i13 = 0; i13 < ne13; ++i13) {
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
            for (int64_t i11 = 0; i11 < ne11; ++i11) {
                size_t bs = ggml_blck_size(vec_dot_type);
                int64_t ne10_block_start = (ith * ne10/bs) / nth;
                int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                
                from_float((float *)((char *) tensor_data(src1) + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                          (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                          (ne10_block_end - ne10_block_start) * bs);
            }
        }
    }
}
```

#### **3. Computation Dispatch**
```c
// From ggml-cpu.c:1102-1103
ggml_vec_dot_t const vec_dot      = type_traits_cpu[type].vec_dot;
enum ggml_type const vec_dot_type = type_traits_cpu[type].vec_dot_type;

// In computation loop:
vec_dot(ne00, &dst_col[ir0], 0, src0_row + ir0*nb01, 0, src1_col, 0, 1);
```

## 🎯 **Quantized MUL_MAT Operation Patterns**

### **Common Quantized Patterns**
1. **Q4_0 × F32 → F32**: `Q4_0` (src0) × `F32` (src1) → `F32` (dst)
   - src1 gets converted to Q8_0 using `quantize_row_q4_0`
   - Computation uses `ggml_vec_dot_q4_0_q8_0`
   
2. **Q8_0 × F32 → F32**: `Q8_0` (src0) × `F32` (src1) → `F32` (dst)
   - src1 gets converted to Q8_0 using `quantize_row_q8_0`
   - Computation uses `ggml_vec_dot_q8_0_q8_0`
   
3. **Q6_K × F32 → F32**: `Q6_K` (src0) × `F32` (src1) → `F32` (dst)
   - src1 gets converted to Q8_K using `quantize_row_q6_K`
   - Computation uses `ggml_vec_dot_q6_K_q8_K`

### **Critical Assertions and Validations**
```c
// Type size validations (from ggml-cpu.c:1194-1206)
GGML_ASSERT(nb00 == ggml_type_size(src0->type));    // src0 stride matches type
GGML_ASSERT(nb10 == ggml_type_size(src1->type));    // src1 stride matches type
GGML_ASSERT(nb0 == sizeof(float));                  // dst is always F32
GGML_ASSERT(nb0 <= nb1);                            // Memory layout constraints
GGML_ASSERT(nb1 <= nb2);
GGML_ASSERT(nb2 <= nb3);

// Work buffer size validation
assert(params->wsize >= ne13*nbw3);                 // Sufficient conversion space
GGML_ASSERT(src1->type == GGML_TYPE_F32);           // src1 must be F32 for conversion
```

## 🏗️ **Implementation Strategy**

### **Phase 1: Core Infrastructure**
1. **Remove F32-only restriction** from NUMA kernel query function
2. **Add type trait extraction** using `ggml_get_type_traits_cpu(src0->type)`
3. **Implement src1 conversion logic** when `src1->type != vec_dot_type`
4. **Add comprehensive type assertions** matching reference implementation

### **Phase 2: Conversion Logic Implementation**
```c
// Planned implementation pattern:
enum ggml_status ggml_numa_kernel_mul_mat_execute(struct ggml_tensor * tensor, struct ggml_cplan * cplan) {
    const struct ggml_tensor * src0 = tensor->src[0];
    const struct ggml_tensor * src1 = tensor->src[1];
    
    // Get type traits for src0
    const struct ggml_type_traits_cpu * type_traits = ggml_get_type_traits_cpu(src0->type);
    ggml_vec_dot_t const vec_dot = type_traits->vec_dot;
    enum ggml_type const vec_dot_type = type_traits->vec_dot_type;
    ggml_from_float_t const from_float = ggml_get_type_traits_cpu(vec_dot_type)->from_float;
    
    // Add all reference implementation assertions
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));
    GGML_ASSERT(nb0 == sizeof(float));
    // ... etc
    
    // Handle src1 type conversion if needed
    const void * wdata;
    if (src1->type != vec_dot_type) {
        GGML_ASSERT(src1->type == GGML_TYPE_F32);  // Must be F32 for conversion
        GGML_ASSERT(from_float != NULL);           // Conversion function must exist
        
        // Use cplan work buffer for conversion
        wdata = cplan->wdata;
        
        // Perform NUMA-aware conversion (distribute across threads)
        // ... conversion logic ...
    } else {
        wdata = tensor_data(src1);  // No conversion needed
    }
    
    // Use type-specific vec_dot in computation loops
    vec_dot(ne00, dst_ptr, 0, src0_ptr, 0, src1_ptr, 0, 1);
}
```

### **Phase 3: Comprehensive Testing**
1. **Create quantized test matrices** for Q4_0, Q8_0, Q6_K types
2. **Test mathematical equivalence** with reference implementation
3. **Validate memory usage** with work buffers
4. **Edge case testing** for different tensor sizes and thread counts

## 🧪 **Test Strategy**

### **Test Matrix**
```
Quantized Types to Test:
├── Q4_0 × F32 → F32 (common LLM weights)
├── Q8_0 × F32 → F32 (higher precision quantized)
├── Q6_K × F32 → F32 (K-quantization)
├── Q4_K × F32 → F32 (K-quantization)
└── Q5_K × F32 → F32 (K-quantization)

Test Dimensions:
├── Tiny: 4×4, 8×8 matrices
├── Small: 32×32, 64×64 matrices  
├── Medium: 128×128, 256×256 matrices
└── Large: 512×512, 1024×1024 matrices

Thread Counts: 1, 2, 4, 6, 8 threads
NUMA Modes: Single-node, Data-parallel
```

### **Mathematical Validation**
```c
// Test pattern for each quantized type
bool test_quantized_mul_mat(enum ggml_type src0_type, int m, int n, int k, int num_threads) {
    // 1. Create quantized src0 matrix of specified type
    // 2. Create F32 src1 matrix
    // 3. Execute with NUMA kernel
    // 4. Execute with reference implementation  
    // 5. Compare results with appropriate tolerance
    // 6. Validate memory usage and conversion correctness
}
```

## ⚠️ **Critical Implementation Notes**

### **Memory Management**
- **Work buffer sizing**: Must calculate `ne13 * nbw3` accurately
- **NUMA allocation**: Work buffers should be NUMA-local
- **Block size awareness**: Handle `ggml_blck_size(vec_dot_type)` correctly

### **Type Safety**
- **Always validate** `from_float != NULL` before conversion
- **Assert src1 is F32** when conversion is needed
- **Check vec_dot function** exists for source type

### **Performance Considerations**
- **Conversion overhead**: F32→quantized conversion is expensive
- **Memory bandwidth**: Quantized types reduce memory pressure
- **SIMD utilization**: Type-specific vec_dot functions are optimized

## 📋 **Implementation Checklist**

- [ ] Remove F32-only restriction from query function
- [ ] Add type trait extraction and validation
- [ ] Implement src1 conversion logic with work buffers
- [ ] Add comprehensive type assertions matching reference
- [ ] Create quantized type test matrices
- [ ] Implement mathematical equivalence tests
- [ ] Validate memory usage and NUMA locality
- [ ] Test edge cases and error conditions
- [ ] Performance benchmark vs reference implementation
- [ ] Documentation and code comments

## 🎯 **Success Criteria**

1. **Functional**: All common quantized types (Q4_0, Q8_0, Q6_K) work correctly
2. **Mathematical**: Perfect equivalence with reference implementation
3. **Performance**: No significant regression vs reference
4. **Memory**: Efficient work buffer usage with NUMA awareness
5. **Robust**: Comprehensive error handling and edge case coverage

**Priority**: High - Required for production quantized model support in NUMA environments.
