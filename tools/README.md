# GGML Tensor Layout Visualization Tools

This directory contains visualization tools to help understand GGML tensor layouts, memory organization, and data access patterns. These tools are especially useful when debugging NUMA kernels and understanding quantization formats.

## Available Tools

### 1. `visualize-tensor-basic`
**Status**: ✅ Working  
Demonstrates basic tensor structure, memory layout, stride calculations, and NUMA slicing patterns.

**Usage**:
```bash
cd /workspaces/llama-cpp-dbsanfte-dev
./build/bin/visualize-tensor-basic
```

**Output**: Shows 2D matrix layouts, 3D tensor structures, NUMA data slicing visualization, and stride-based element access patterns.

### 2. `visualize-tensor-quantization` 
**Status**: 🚧 Under development  
Shows quantized tensor formats (Q4_0, Q8_0, etc.), block structures, and dequantization patterns.

### 3. `visualize-tensor-add-types`
**Status**: 🚧 Under development  
Demonstrates all tensor type combinations used in ADD mathematical correctness tests.

### 4. `visualize-tensor-broadcasting`
**Status**: 🚧 Under development  
Shows tensor broadcasting patterns for different shaped tensors.

## Building Tools

All tools are built automatically when building the main project:

```bash
cmake --build build --parallel
```

Individual tools can be built with:
```bash
cmake --build build --target visualize-tensor-basic --parallel
```

## Documentation

See [`../docs/ggml-tensor-layout-visualization.md`](../docs/ggml-tensor-layout-visualization.md) for comprehensive documentation on:

- GGML tensor structure (`struct ggml_tensor`)
- Memory layout and stride calculations
- Floating point formats (F32, F16, BF16)
- Quantization formats (Q4_0, Q8_0, Q*_K types)
- Multi-dimensional tensor navigation
- Broadcasting patterns
- NUMA slicing strategies
- Common debugging scenarios

## Use Cases

These tools are particularly helpful for:

1. **NUMA Kernel Development**: Understanding how to slice tensors across NUMA nodes and threads
2. **Quantization Debugging**: Visualizing block structures and dequantization patterns
3. **Memory Layout Analysis**: Understanding stride calculations and element access patterns
4. **Broadcasting Logic**: Debugging tensor shape compatibility and element mapping
5. **Performance Optimization**: Understanding memory access patterns for cache optimization

## Example Output

```
================================================================================
2D F32 TENSOR MEMORY LAYOUT VISUALIZATION
================================================================================

=== Tensor: 2D F32 Matrix [4x3] ===
Type: f32
Dimensions (ne): [4, 3, 1, 1]
Strides (nb):    [4, 16, 48, 48] bytes
Total elements: 12
Total bytes: 48
Element size: 4 bytes

Memory Layout (Conceptual):
┌─────────────────────────────────────────────┐
│ Row 0: [  0.0] [  1.0] [  2.0] [  3.0] │
│ Row 1: [ 10.0] [ 11.0] [ 12.0] [ 13.0] │
│ Row 2: [ 20.0] [ 21.0] [ 22.0] [ 23.0] │
└─────────────────────────────────────────────┘

Stride-based Element Access:
element[0,0] at offset   0 bytes =   0.0
element[0,1] at offset   4 bytes =   1.0
element[0,2] at offset   8 bytes =   2.0
...
```

This visualization clearly shows:
- How tensor dimensions map to memory layout
- Byte offsets for stride-based element access
- Row-major memory organization
- NUMA slicing boundaries and thread assignments

## Integration with NUMA Development

These tools directly support the patterns used in our NUMA kernel implementations:

- **Element-wise slicing**: For operations like ADD, MUL, DIV
- **Row-wise slicing**: For operations like RMS_NORM, matrix operations
- **Block-aware slicing**: For quantized tensor operations
- **Stride calculations**: For multi-dimensional tensor navigation

Use these visualizations to verify your tensor access patterns match the expected memory layout before implementing complex NUMA kernels.
