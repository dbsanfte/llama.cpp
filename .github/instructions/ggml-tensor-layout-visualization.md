# GGML Tensor Layout Visualization Guide

This document provides comprehensive visual examples of GGML tensor layouts and data formats to help developers understand the memory organization, stride calculations, and quantization patterns when debugging NUMA kernels and mathematical operations.

## Table of Contents

1. [Basic Tensor Structure](#basic-tensor-structure)
2. [Memory Layout and Strides](#memory-layout-and-strides) 
3. [Floating Point Formats](#floating-point-formats)
4. [Quantization Formats](#quantization-formats)
5. [Multi-dimensional Tensors](#multi-dimensional-tensors)
6. [Practical Examples for NUMA Debugging](#practical-examples-for-numa-debugging)

## Basic Tensor Structure

Every GGML tensor is represented by the `struct ggml_tensor` which contains:

```c
struct ggml_tensor {
    enum ggml_type type;                    // Data type (F32, F16, Q4_0, etc.)
    int64_t ne[GGML_MAX_DIMS];             // Number of elements [ne0, ne1, ne2, ne3]
    size_t  nb[GGML_MAX_DIMS];             // Stride in bytes [nb0, nb1, nb2, nb3]
    void * data;                           // Pointer to actual data
    // ... other fields
};
```

### Key Relationships:
- `ne[0]` = width (columns)  
- `ne[1]` = height (rows)
- `ne[2]` = depth (sequences/channels) 
- `ne[3]` = batch dimension
- `nb[0]` = bytes per element
- `nb[1]` = bytes per row  
- `nb[2]` = bytes per 2D slice
- `nb[3]` = bytes per 3D volume

## Memory Layout and Strides

### 2D Matrix Example: 3×4 F32 matrix

```
Tensor dimensions: ne = [4, 3, 1, 1]  (4 columns, 3 rows)
Tensor strides:    nb = [4, 16, 48, 48] (4 bytes/element, 16 bytes/row)

Memory layout (row-major):
┌────────────────────────────────────────────┐
│ Row 0: [a₀₀] [a₀₁] [a₀₂] [a₀₃]             │ ← nb[1] = 16 bytes
│ Row 1: [a₁₀] [a₁₁] [a₁₂] [a₁₃]             │ 
│ Row 2: [a₂₀] [a₂₁] [a₂₂] [a₂₃]             │
└────────────────────────────────────────────┘
        ↑ nb[0] = 4 bytes per F32

Element access formula:
element(row, col) = *(float*)((char*)data + row * nb[1] + col * nb[0])
```

### 3D Tensor Example: 2×3×4 F32 tensor

```
Tensor dimensions: ne = [4, 3, 2, 1]  (4 cols, 3 rows, 2 depth)
Tensor strides:    nb = [4, 16, 48, 96] 

Memory layout:
┌─── Slice 0 (depth=0) ──────────────────────┐
│ Row 0: [a₀₀₀] [a₀₀₁] [a₀₀₂] [a₀₀₃]         │
│ Row 1: [a₁₀₀] [a₁₀₁] [a₁₀₂] [a₁₀₃]         │ ← nb[2] = 48 bytes
│ Row 2: [a₂₀₀] [a₂₀₁] [a₂₀₂] [a₂₀₃]         │   to next slice
└────────────────────────────────────────────┘
┌─── Slice 1 (depth=1) ──────────────────────┐
│ Row 0: [a₀₁₀] [a₀₁₁] [a₀₁₂] [a₀₁₃]         │
│ Row 1: [a₁₁₀] [a₁₁₁] [a₁₁₂] [a₁₁₃]         │
│ Row 2: [a₂₁₀] [a₂₁₁] [a₂₁₂] [a₂₁₃]         │
└────────────────────────────────────────────┘

Element access:
element(row, col, depth) = *(float*)((char*)data + depth * nb[2] + row * nb[1] + col * nb[0])
```

## Floating Point Formats

### F32 (32-bit Float)
```
Size: 4 bytes per element
Range: ±3.4 × 10³⁸ (IEEE 754 single precision)
Block size: 1 (no blocking)

Memory layout for [1.0f, 2.5f, -3.14f]:
┌──────────┬──────────┬──────────┐
│ 3F800000 │ 40200000 │ C048F5C3 │ (hex representation)
│   1.0f   │   2.5f   │  -3.14f  │ (values)
└──────────┴──────────┴──────────┘
```

### F16 (16-bit Half)
```
Size: 2 bytes per element  
Range: ±65,504 (reduced precision)
Block size: 1 (no blocking)

Memory layout for [1.0, 2.5, -3.14]:
┌─────┬─────┬─────┐
│3C00 │4100 │C648 │ (hex representation) 
│ 1.0 │ 2.5 │-3.14│ (values)
└─────┴─────┴─────┘
```

### BF16 (16-bit Brain Float)
```
Size: 2 bytes per element
Range: Same as F32 but reduced precision (truncated F32)
Block size: 1 (no blocking)

Memory layout for [1.0, 2.5, -3.14]:
┌─────┬─────┬─────┐
│3F80 │4020 │C048 │ (hex representation)
│ 1.0 │ 2.5 │-3.14│ (values) 
└─────┴─────┴─────┘
```

## Quantization Formats

Quantized formats use blocks to store multiple values with shared scaling factors.

### Q4_0 (4-bit, 32 elements per block)

```c
#define QK4_0 32
typedef struct {
    ggml_half d;        // scale (2 bytes)
    uint8_t qs[QK4_0/2]; // quantized values (16 bytes, 2 values per byte)
} block_q4_0;
// Total: 18 bytes per 32 elements = 0.5625 bytes per element
```

```
Block layout for 32 F32 values quantized to Q4_0:
┌─────────┬─────────────────────────────────────────────┐
│ Scale   │ Quantized nibbles (2 per byte)              │
│ (F16)   │ [q₀q₁] [q₂q₃] [q₄q₅] ... [q₃₀q₃₁]           │
│ 2 bytes │ 16 bytes                                    │
└─────────┴─────────────────────────────────────────────┘

Dequantization: f32_value = scale * (int4_nibble - 8)

Example memory:
Scale: 0.125 (as F16: 0x3400)
Nibbles: [0x89, 0xAB, 0xCD, ...] = [8,9,A,B,C,D,...]
Dequantized: [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, ...]
```

### Q8_0 (8-bit, 32 elements per block)

```c
#define QK8_0 32
typedef struct {
    ggml_half d;       // scale (2 bytes)
    int8_t qs[QK8_0];  // quantized values (32 bytes)
} block_q8_0;
// Total: 34 bytes per 32 elements = 1.0625 bytes per element
```

```
Block layout:
┌─────────┬────────────────────────────────┐
│ Scale   │ Signed 8-bit values            │
│ (F16)   │ [q₀] [q₁] [q₂] ... [q₃₁]       │
│ 2 bytes │ 32 bytes                       │
└─────────┴────────────────────────────────┘

Dequantization: f32_value = scale * int8_value

Example:
Scale: 0.01 (as F16)
Values: [-128, -64, 0, 64, 127, ...]
Dequantized: [-1.28, -0.64, 0.0, 0.64, 1.27, ...]
```

### Q4_K (4-bit K-quantization, 256 elements per super-block)

```c
#define QK_K 256
#define K_SCALE_SIZE 12
typedef struct {
    union {
        struct {
            ggml_half d;    // super-block scale
            ggml_half dmin; // super-block scale for mins
        };
        ggml_half2 dm;
    };
    uint8_t scales[K_SCALE_SIZE]; // scales and mins (12 bytes)
    uint8_t qs[QK_K/2];           // 4-bit quants (128 bytes)
} block_q4_K;
// Total: 144 bytes per 256 elements = 0.5625 bytes per element
```

```
Super-block layout (256 elements organized as 8 sub-blocks of 32):
┌──────┬──────┬──────────────┬─────────────────────────────────────┐
│  d   │ dmin │   scales     │ quantized nibbles                   │
│(F16) │(F16) │ (12 bytes)   │ (128 bytes)                         │
│2 byte│2 byte│              │ [q₀q₁][q₂q₃]...[q₂₅₄q₂₅₅]           │
└──────┴──────┴──────────────┴─────────────────────────────────────┘

Sub-block organization:
Block 0: elements 0-31    → scales[0], qs[0-15]
Block 1: elements 32-63   → scales[1], qs[16-31]  
...
Block 7: elements 224-255 → scales[7], qs[112-127]

Dequantization involves multiple scales and complex bit manipulation
```

## Multi-dimensional Tensors

### 4D Tensor Navigation

For a 4D tensor with dimensions `ne = [W, H, D, B]`:

```c
// Element access patterns for different iteration orders:

// 1. Element-wise (flatten everything)
for (int i = 0; i < ne[0] * ne[1] * ne[2] * ne[3]; i++) {
    // Linear access: data[i] 
}

// 2. Dimension-wise (preserve tensor structure)  
for (int b = 0; b < ne[3]; b++) {           // batch
    for (int d = 0; d < ne[2]; d++) {       // depth  
        for (int h = 0; h < ne[1]; h++) {   // height (rows)
            for (int w = 0; w < ne[0]; w++) { // width (cols)
                float* element = (float*)((char*)data + 
                    b * nb[3] + d * nb[2] + h * nb[1] + w * nb[0]);
            }
        }
    }
}

// 3. Row-wise (common for matrix operations)
for (int b = 0; b < ne[3]; b++) {
    for (int d = 0; d < ne[2]; d++) {
        for (int h = 0; h < ne[1]; h++) {
            float* row_start = (float*)((char*)data + b * nb[3] + d * nb[2] + h * nb[1]);
            // Process entire row: row_start[0] to row_start[ne[0]-1]
        }
    }
}
```

### Broadcasting Examples

Common broadcasting patterns in GGML operations:

```c
// Matrix + Vector broadcasting: [4,3] + [4,1] → [4,3]
Matrix A (4×3):         Vector B (4×1):        Result (4×3):
┌─────────────────┐     ┌───┐                 ┌─────────────────┐
│ a₀₀ a₀₁ a₀₂ a₀₃ │  +  │b₀ │ (broadcast)  =  │a₀₀+b₀ a₀₁+b₀ ...│
│ a₁₀ a₁₁ a₁₂ a₁₃ │     │b₁ │                 │a₁₀+b₁ a₁₁+b₁ ...│  
│ a₂₀ a₂₁ a₂₂ a₂₃ │     │b₂ │                 │a₂₀+b₂ a₂₁+b₂ ...│
└─────────────────┘     └───┘                 └─────────────────┘

// Element access for broadcasting:
for (int row = 0; row < 3; row++) {
    for (int col = 0; col < 4; col++) {
        float a_val = *(float*)((char*)tensor_a->data + row * tensor_a->nb[1] + col * tensor_a->nb[0]);
        float b_val = *(float*)((char*)tensor_b->data + row * tensor_b->nb[1]); // No col offset
        result = a_val + b_val;
    }
}
```

## Practical Examples for NUMA Debugging

The following examples show common patterns when debugging NUMA kernels:

### Example 1: Element-wise Operation Data Slicing

```c
// NUMA data slicing for ADD operation on 1000-element tensor
// 2 NUMA nodes, 4 threads per node = 8 total threads

size_t total_elements = 1000;
int numa_nodes = 2;
int threads_per_node = 4;
int total_threads = 8;

// Per-NUMA-node slicing (data-parallel execution)
size_t elements_per_node = total_elements / numa_nodes;  // 500 
size_t numa_start = numa_node * elements_per_node;       // Node 0: 0, Node 1: 500
size_t numa_end = (numa_node == numa_nodes - 1) ? 
                  total_elements : numa_start + elements_per_node; // Node 0: 500, Node 1: 1000

// Per-thread slicing within NUMA node  
size_t node_elements = numa_end - numa_start;            // 500
size_t elements_per_thread = node_elements / threads_per_node; // 125
size_t thread_start = numa_start + (thread_id * elements_per_thread); // Thread 0: 0, Thread 1: 125, etc.
size_t thread_end = (thread_id == threads_per_node - 1) ? 
                    numa_end : thread_start + elements_per_thread;

// Memory layout visualization:
// Total: [0 ──────── 500 ──────── 1000]
//        │←── Node 0 ──→│←── Node 1 ──→│
//        │T0│T1│T2│T3│   │T4│T5│T6│T7│
//        0  125 250 375  500 625 750 875 1000
```

### Example 2: Row-wise Operation on 2D Matrix

```c
// Matrix dimensions: 1000 rows × 512 columns (F32)
// NUMA slicing by rows for operations like RMS_NORM

int total_rows = 1000;
int numa_nodes = 2; 
int threads_per_node = 4;

// NUMA node gets subset of rows
int rows_per_node = total_rows / numa_nodes;          // 500
int numa_row_start = numa_node * rows_per_node;       // Node 0: 0, Node 1: 500  
int numa_row_end = (numa_node == numa_nodes - 1) ? 
                   total_rows : numa_row_start + rows_per_node; // Node 0: 500, Node 1: 1000

// Thread gets subset of node's rows
int node_rows = numa_row_end - numa_row_start;        // 500
int rows_per_thread = node_rows / threads_per_node;   // 125
int thread_row_start = numa_row_start + (thread_id * rows_per_thread);
int thread_row_end = (thread_id == threads_per_node - 1) ?
                     numa_row_end : thread_row_start + rows_per_thread;

// Memory access pattern:
for (int row = thread_row_start; row < thread_row_end; row++) {
    float* row_data = (float*)((char*)tensor->data + row * tensor->nb[1]);
    // Process entire row: row_data[0] to row_data[ne[0]-1]
    for (int col = 0; col < tensor->ne[0]; col++) {
        float value = row_data[col];
        // ... process value ...
    }
}
```

### Example 3: Quantized Data Handling  

```c
// Q4_0 tensor debugging - understanding block boundaries

size_t total_elements = 1000;  // Must be multiple of QK4_0 (32) for proper blocks
size_t total_blocks = total_elements / QK4_0;  // 31.25 → 31 complete blocks + remainder

// Block-aligned slicing for NUMA  
size_t blocks_per_node = total_blocks / numa_nodes;  // 15 blocks per node
size_t numa_block_start = numa_node * blocks_per_node;
size_t numa_block_end = (numa_node == numa_nodes - 1) ? 
                        total_blocks : numa_block_start + blocks_per_node;

// Convert back to elements
size_t numa_element_start = numa_block_start * QK4_0;  // Node 0: 0, Node 1: 480
size_t numa_element_end = numa_block_end * QK4_0;      // Node 0: 480, Node 1: 960
// Note: Remainder elements (960-1000) need special handling

// Block data access:
block_q4_0* blocks = (block_q4_0*)tensor->data;
for (size_t block_idx = numa_block_start; block_idx < numa_block_end; block_idx++) {
    block_q4_0* block = &blocks[block_idx];
    ggml_half scale = block->d;
    
    // Process 32 elements in this block
    for (int i = 0; i < QK4_0; i++) {
        uint8_t packed_byte = block->qs[i / 2];
        uint8_t nibble = (i % 2 == 0) ? (packed_byte & 0x0F) : (packed_byte >> 4);
        float dequantized = ggml_fp16_to_fp32(scale) * (nibble - 8);
        // ... process dequantized value ...
    }
}
```

### Example 4: SIMD-Optimized Memory Access

```c
// SIMD processing requires aligned memory access
// Example: GGML vectorized operations

size_t thread_elements = thread_end - thread_start;  // e.g., 125 elements

// For F32 data with SIMD (using ggml_vec_add_f32)
const float* src0 = (const float*)tensor->src[0]->data + thread_start;
const float* src1 = (const float*)tensor->src[1]->data + thread_start; 
float* dst = (float*)tensor->data + thread_start;

// SIMD operation on thread's slice
ggml_vec_add_f32(thread_elements, dst, src0, src1);

// This replaces manual loop:
// for (size_t i = 0; i < thread_elements; i++) {
//     dst[i] = src0[i] + src1[i];  
// }

// SIMD benefits:
// - Processes multiple elements per instruction (4-8 F32s with AVX)
// - Better cache utilization
// - Mathematical equivalence to scalar operations
```

## Common Debugging Scenarios

### Scenario 1: Stride Calculation Errors

```c
// Wrong: Assuming contiguous memory
float value = data[row * cols + col];  // ❌ Ignores actual strides

// Correct: Using tensor strides  
float value = *(float*)((char*)data + row * nb[1] + col * nb[0]);  // ✅

// Debug stride information:
printf("Tensor: ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]\n",
       tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
       tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]);
```

### Scenario 2: Quantization Block Alignment

```c
// Check if tensor size is compatible with quantization block size
if (ggml_is_quantized(tensor->type)) {
    int block_size = ggml_blck_size(tensor->type);  // e.g., 32 for Q4_0
    if (ggml_nelements(tensor) % block_size != 0) {
        printf("⚠️  Tensor size %lld not aligned to block size %d\n", 
               ggml_nelements(tensor), block_size);
    }
}
```

### Scenario 3: NUMA Memory Locality

```c
// Verify NUMA-aware memory access
#ifdef GGML_NUMA_MIRROR
if (ggml_numa_should_mirror()) {
    // Access NUMA-local copy
    void* numa_data = tensor_data(tensor);  // Gets NUMA-appropriate data pointer
} else {
    void* data = tensor->data;  // Single memory location
}
#endif
```

This visualization guide provides the foundation for understanding GGML tensor layouts when debugging complex mathematical operations in NUMA-aware code. Use these patterns to verify correct memory access, stride calculations, and quantization handling in your kernel implementations.

## Visualization Tools

The `tools/` directory contains several practical visualization tools to help debug tensor layouts interactively:

### Available Tools

#### 1. **visualize-tensor-basic** ✅ Working
```bash
# Build and run the basic visualization tool
cmake --build build --target visualize-tensor-basic
./build/bin/visualize-tensor-basic
```

**Features:**
- **2D Matrix Layout**: Shows memory organization, stride calculations, and element access patterns
- **3D Tensor Structure**: Demonstrates multi-dimensional slicing with depth/sequence handling  
- **NUMA Slicing Visualization**: Shows how tensors are distributed across NUMA nodes and threads
- **Memory Access Patterns**: Displays byte offsets and pointer arithmetic for debugging

**Sample Output:**
```
=== 2D Matrix: 4×3 F32 ===
Tensor: ne=[4,3,1,1] nb=[4,16,48,48] type=F32
Memory layout (row-major):
Row 0: [   1.00] [   2.00] [   3.00] [   4.00]  (bytes 0-15)
Row 1: [   5.00] [   6.00] [   7.00] [   8.00]  (bytes 16-31)  
Row 2: [   9.00] [  10.00] [  11.00] [  12.00] (bytes 32-47)

=== NUMA Data Slicing (2 nodes, 4 threads) ===
Node 0 (elements 0-5): Threads 0-1
  Thread 0: elements 0-2  → [1.00, 2.00, 3.00]
  Thread 1: elements 3-5  → [4.00, 5.00, 6.00]
Node 1 (elements 6-11): Threads 2-3  
  Thread 2: elements 6-8  → [7.00, 8.00, 9.00]
  Thread 3: elements 9-11 → [10.00, 11.00, 12.00]
```

#### 2. **visualize-tensor-quantization** 🚧 Foundation Complete
```bash
# Build command (needs compilation fixes)
cmake --build build --target visualize-tensor-quantization
```

**Intended Features:**
- **Q4_0 Block Structure**: 32-element blocks with 16-byte storage (scale + 16 nibbles)
- **Q8_0 Block Structure**: 32-element blocks with 34-byte storage (scale + 32 int8)
- **Dequantization Examples**: Shows conversion from quantized to F32 values
- **Block Alignment**: Demonstrates proper quantization boundaries

#### 3. **visualize-tensor-add-types** 🚧 Foundation Complete  
```bash
# Build command (needs compilation fixes)
cmake --build build --target visualize-tensor-add-types
```

**Intended Features:**
- **Type Combination Matrix**: All 24+ supported ADD operation type pairs
- **Broadcasting Patterns**: Different tensor shape combinations (scalar, vector, matrix)
- **Memory Layout Comparison**: Side-by-side F32, F16, Q4_0, Q8_0 layouts
- **ADD Test Scenarios**: Real examples from mathematical correctness tests

#### 4. **visualize-tensor-broadcasting** 🚧 Foundation Complete
```bash  
# Build command (needs compilation fixes)
cmake --build build --target visualize-tensor-broadcasting
```

**Intended Features:**
- **Broadcasting Rules**: How tensors with different shapes combine
- **Dimension Expansion**: Visual examples of 1D → 2D → 3D broadcasting
- **Memory Access Patterns**: Shows how broadcasted elements are accessed
- **NUMA Slicing Impact**: How broadcasting affects data distribution

### Quick Start Guide

**For immediate tensor debugging:**
```bash
# 1. Build the working visualization tool
cd /workspaces/llama-cpp-dbsanfte-dev
cmake --build build --target visualize-tensor-basic

# 2. Run basic tensor visualization 
./build/bin/visualize-tensor-basic

# 3. Analyze output for your debugging needs:
#    - Memory stride calculations
#    - NUMA data slicing patterns  
#    - Element access formulas
#    - Multi-dimensional indexing
```

**Integration with NUMA debugging:**
```bash
# Enable NUMA debug logging and run visualization together
export GGML_NUMA_DEBUG=1
./build/bin/visualize-tensor-basic

# Then run your NUMA mathematical correctness test
./build/bin/test-numa-mathematical-correctness-add
```

### Development Notes

**Tool Status Summary:**
- ✅ **visualize-tensor-basic**: Fully functional, comprehensive 2D/3D examples
- 🚧 **Other tools**: Foundation complete, need compilation fixes (tensor data access, C++ syntax)

**Common compilation issues to fix:**
```cpp
// ❌ Wrong: Direct tensor->data access
float* data = (float*)tensor->data;

// ✅ Correct: Use NUMA-aware accessor  
float* data = (float*)tensor_data(tensor);

// ❌ Wrong: String multiplication (C++ issue)
std::string indent = " " * depth;

// ✅ Correct: String constructor
std::string indent(depth * 2, ' ');
```

**Adding custom tensor examples:**
```cpp
// Modify visualize-tensor-basic.cpp to add your specific debugging scenarios
void debug_my_tensor_pattern() {
    // Create tensor with your specific dimensions
    struct ggml_tensor* tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, your_cols, your_rows);
    
    // Fill with your test data
    fill_tensor_with_test_data(tensor);
    
    // Use existing visualization functions
    print_tensor_layout(tensor, "My Debug Case");
    print_numa_slicing(tensor, numa_nodes, total_threads);
}
```

### Reference Documentation

This guide focuses on **practical visualization and debugging**. For comprehensive architecture details, see:
- `docs/numa-architecture.md` - NUMA system architecture and performance characteristics
- `ggml/src/ggml.h` - Complete tensor structure definitions and type enums
- `tests/test-numa-mathematical-correctness-*.cpp` - Real-world tensor usage patterns
