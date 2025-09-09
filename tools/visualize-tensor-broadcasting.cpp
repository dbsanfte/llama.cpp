/**
 * GGML Tensor Layout Visualizer - Broadcasting Patterns
 * 
 * This tool demonstrates tensor broadcasting patterns commonly used in GGML operations,
 * helping understand how elements are accessed in NUMA kernels when tensors have
 * different shapes but compatible dimensions.
 * 
 * @author David Sanftenberg
 */

#include <cstdio>
#include <cstring>
#include <vector>
#include <iomanip>

// GGML includes
#include "ggml.h"

/**
 * Visualize broadcasting between tensors of different shapes
 */
void visualize_broadcasting_patterns(struct ggml_context* ctx) {
    printf("\n" "="*80 "\n");
    printf("TENSOR BROADCASTING PATTERNS IN GGML\n");
    printf("="*80 "\n");
    
    printf("Broadcasting allows operations between tensors of different shapes.\n");
    printf("Smaller tensors are conceptually 'stretched' to match larger ones.\n\n");
    
    // Example 1: Matrix + Vector (column-wise broadcast)
    printf("Example 1: Matrix + Vector (Column-wise Broadcasting)\n");
    printf("=" * 50 "\n");
    
    const int ne0 = 4, ne1 = 3;
    struct ggml_tensor* matrix = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);  // 4x3 matrix
    struct ggml_tensor* vector = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, 1);   // 4x1 vector
    
    // Fill with sample data
    float* matrix_data = (float*)matrix->data;
    float* vector_data = (float*)vector->data;
    
    for (int i = 0; i < ne0 * ne1; i++) {
        matrix_data[i] = (float)(i + 1);
    }
    for (int i = 0; i < ne0; i++) {
        vector_data[i] = (float)(i + 1) * 0.1f;
    }
    
    printf("Matrix A [4×3]:          Vector B [4×1]:\n");
    for (int row = 0; row < ne1; row++) {
        printf("  Row %d: [", row);
        for (int col = 0; col < ne0; col++) {
            printf("%4.0f", matrix_data[row * ne0 + col]);
            if (col < ne0 - 1) printf(",");
        }
        printf("]");
        
        if (row == 0) {
            printf("      [");
            for (int col = 0; col < ne0; col++) {
                printf("%4.1f", vector_data[col]);
                if (col < ne0 - 1) printf(",");
            }
            printf("]");
        }
        printf("\n");
    }
    
    printf("\nBroadcasting Logic:\n");
    printf("  Result[row][col] = Matrix[row][col] + Vector[col]\n");
    printf("  Vector is reused for each row of the matrix\n");
    
    printf("\nElement Access Pattern for NUMA kernel:\n");
    printf("  for (int row = row_start; row < row_end; row++) {\n");
    printf("    for (int col = 0; col < ne0; col++) {\n");
    printf("      float matrix_val = *(float*)((char*)matrix->data + row * matrix->nb[1] + col * matrix->nb[0]);\n");
    printf("      float vector_val = *(float*)((char*)vector->data + col * vector->nb[0]);\n");  
    printf("      float result = matrix_val + vector_val;\n");
    printf("    }\n");
    printf("  }\n");
    
    printf("\nStriding Analysis:\n");
    printf("  Matrix strides: nb[0]=%zu, nb[1]=%zu\n", matrix->nb[0], matrix->nb[1]);
    printf("  Vector strides: nb[0]=%zu, nb[1]=%zu\n", vector->nb[0], vector->nb[1]);
    printf("  Note: Vector nb[1] is NOT used in broadcast operation\n");
    
    // Example 2: Matrix + Scalar broadcast
    printf("\n\nExample 2: Matrix + Scalar Broadcasting\n");
    printf("=" * 40 "\n");
    
    struct ggml_tensor* scalar = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    float* scalar_data = (float*)scalar->data;
    scalar_data[0] = 10.0f;
    
    printf("Matrix A [4×3]:                Scalar B:\n");
    for (int row = 0; row < ne1; row++) {
        printf("  Row %d: [", row);
        for (int col = 0; col < ne0; col++) {
            printf("%4.0f", matrix_data[row * ne0 + col]);
            if (col < ne0 - 1) printf(",");
        }
        printf("]");
        
        if (row == 1) {
            printf("      %4.0f", scalar_data[0]);
        }
        printf("\n");
    }
    
    printf("\nBroadcasting Logic:\n");
    printf("  Result[row][col] = Matrix[row][col] + Scalar[0]\n");
    printf("  Scalar value is reused for every element\n");
    
    printf("\nElement Access Pattern:\n");
    printf("  float scalar_val = *(float*)scalar->data;  // Load once\n");
    printf("  for (int i = element_start; i < element_end; i++) {\n");
    printf("    float matrix_val = ((float*)matrix->data)[i];\n");
    printf("    float result = matrix_val + scalar_val;\n");
    printf("  }\n");
}

/**
 * Demonstrate 3D broadcasting patterns
 */
void visualize_3d_broadcasting(struct ggml_context* ctx) {
    printf("\n" "="*80 "\n");
    printf("3D TENSOR BROADCASTING PATTERNS\n");
    printf("="*80 "\n");
    
    const int width = 4, height = 3, depth = 2;
    
    // 3D tensor [4,3,2]
    struct ggml_tensor* tensor_3d = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, height, depth);
    
    // 2D tensor [4,3,1] - broadcasts along depth dimension  
    struct ggml_tensor* tensor_2d = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, height, 1);
    
    // 1D tensor [4,1,1] - broadcasts along height and depth
    struct ggml_tensor* tensor_1d = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, 1, 1);
    
    printf("Tensor shapes for broadcasting:\n");
    printf("  3D tensor: [%d, %d, %d] = %lld elements\n", width, height, depth, ggml_nelements(tensor_3d));
    printf("  2D tensor: [%d, %d, %d] = %lld elements\n", width, height, 1, ggml_nelements(tensor_2d));
    printf("  1D tensor: [%d, %d, %d] = %lld elements\n", width, 1, 1, ggml_nelements(tensor_1d));
    
    printf("\nBroadcasting Rules:\n");
    printf("  1. Dimensions are aligned from the right (trailing dimensions)\n");
    printf("  2. Dimensions of size 1 are broadcast to match larger dimensions\n");
    printf("  3. Missing leading dimensions are treated as size 1\n");
    
    printf("\n3D + 2D Broadcasting Example:\n");
    printf("  3D[w,h,d] + 2D[w,h,1] → Result[w,h,d]\n");
    printf("  The 2D tensor is replicated across all depth slices\n");
    
    printf("\nElement Access Pattern:\n");
    printf("  for (int d = depth_start; d < depth_end; d++) {\n");
    printf("    for (int h = 0; h < height; h++) {\n");
    printf("      for (int w = 0; w < width; w++) {\n");
    printf("        float val_3d = *(float*)((char*)tensor_3d->data + d*nb[2] + h*nb[1] + w*nb[0]);\n");
    printf("        float val_2d = *(float*)((char*)tensor_2d->data + h*nb[1] + w*nb[0]);\n");
    printf("        // Note: no d*nb[2] for 2D tensor - it broadcasts\n");
    printf("        float result = val_3d + val_2d;\n");
    printf("      }\n");
    printf("    }\n");
    printf("  }\n");
    
    printf("\n3D + 1D Broadcasting Example:\n");  
    printf("  3D[w,h,d] + 1D[w,1,1] → Result[w,h,d]\n");
    printf("  The 1D tensor is replicated across all height and depth positions\n");
    
    printf("\nElement Access Pattern:\n");
    printf("  for (int d = depth_start; d < depth_end; d++) {\n");
    printf("    for (int h = 0; h < height; h++) {\n");
    printf("      for (int w = 0; w < width; w++) {\n");
    printf("        float val_3d = *(float*)((char*)tensor_3d->data + d*nb[2] + h*nb[1] + w*nb[0]);\n");
    printf("        float val_1d = *(float*)((char*)tensor_1d->data + w*nb[0]);\n");
    printf("        // Note: no h*nb[1] or d*nb[2] for 1D tensor\n");
    printf("        float result = val_3d + val_1d;\n");
    printf("      }\n");
    printf("    }\n");
    printf("  }\n");
    
    printf("\nNUMA Considerations for Broadcasting:\n");
    printf("  - Larger tensor determines NUMA slicing strategy\n");
    printf("  - Broadcast tensor may be accessed repeatedly by multiple threads\n");
    printf("  - Consider caching broadcast values in thread-local memory\n");
    printf("  - Memory locality benefits apply mainly to the larger tensor\n");
}

/**
 * Show common broadcasting errors and debugging tips
 */
void show_broadcasting_debug_tips(struct ggml_context* ctx) {
    printf("\n" "="*80 "\n");
    printf("BROADCASTING DEBUG TIPS FOR NUMA KERNELS\n");
    printf("="*80 "\n");
    
    printf("Common Broadcasting Errors in NUMA Kernels:\n\n");
    
    printf("1. Incorrect stride usage:\n");
    printf("   ❌ WRONG: Accessing broadcast tensor with full strides\n");
    printf("      float val = *(float*)((char*)broadcast->data + row*broadcast->nb[1] + col*broadcast->nb[0]);\n");
    printf("   ✅ CORRECT: Only use strides for non-broadcast dimensions\n");
    printf("      float val = *(float*)((char*)broadcast->data + col*broadcast->nb[0]);  // Skip row stride\n\n");
    
    printf("2. NUMA slicing not accounting for broadcast:\n");
    printf("   ❌ WRONG: Slicing broadcast tensor same as main tensor\n");
    printf("      size_t broadcast_start = numa_start;  // Wrong for [W,1] tensor\n");
    printf("   ✅ CORRECT: Slice only the non-broadcast dimensions\n");
    printf("      size_t broadcast_start = numa_start %% broadcast_width;  // Wrap for broadcast\n\n");
    
    printf("3. Memory access out of bounds:\n");
    printf("   ❌ WRONG: Assuming same memory layout\n");
    printf("      for (int i = 0; i < main_elements; i++) {\n");
    printf("        float val = broadcast_data[i];  // May exceed broadcast tensor size\n");
    printf("      }\n");
    printf("   ✅ CORRECT: Map indices correctly\n");
    printf("      for (int i = 0; i < main_elements; i++) {\n");
    printf("        int broadcast_idx = map_to_broadcast_index(i, main_shape, broadcast_shape);\n");
    printf("        float val = broadcast_data[broadcast_idx];\n");
    printf("      }\n\n");
    
    printf("Broadcasting Compatibility Check:\n");
    printf("  Two tensors are broadcast-compatible if:\n");
    printf("  - Each dimension pair is either equal, or one is 1\n");
    printf("  - Missing leading dimensions are treated as 1\n\n");
    
    printf("Examples:\n");
    printf("  [4,3] + [4,1] ✅ Compatible (3 broadcasts to match 3)\n");
    printf("  [4,3] + [1,3] ✅ Compatible (4 broadcasts to match 4)\n");
    printf("  [4,3] + [2,3] ❌ Incompatible (4 ≠ 2, neither is 1)\n");
    printf("  [4,3,2] + [3,2] ✅ Compatible (broadcast [1,3,2])\n");
    printf("  [4,3,2] + [4,2] ❌ Incompatible (3 ≠ 2, neither is 1)\n\n");
    
    printf("Debug Checklist for NUMA Broadcasting:\n");
    printf("  □ Verify tensor shapes are broadcast-compatible\n");
    printf("  □ Check stride calculations for each tensor separately\n");
    printf("  □ Test with various NUMA slicing patterns\n");
    printf("  □ Validate memory access bounds for broadcast tensors\n");
    printf("  □ Consider memory locality and caching for repeated broadcast access\n");
    printf("  □ Test edge cases (single element broadcasts, dimension mismatches)\n");
    
    printf("\nDebugging Tools:\n");
    printf("  1. Print tensor shapes and strides:\n");
    printf("     printf(\"Tensor: ne=[%%lld,%%lld,%%lld,%%lld] nb=[%%zu,%%zu,%%zu,%%zu]\\n\",\n");
    printf("            tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],\n");
    printf("            tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]);\n\n");
    
    printf("  2. Validate element access:\n");
    printf("     // Check if computed offset is within tensor bounds\n");
    printf("     size_t offset = row * nb[1] + col * nb[0];\n");
    printf("     assert(offset < ggml_nbytes(tensor));\n\n");
    
    printf("  3. Compare with reference implementation:\n");
    printf("     // Use ggml_get_f32_1d() for safe element access during debugging\n");
    printf("     float ref_val = ggml_get_f32_1d(tensor, linear_index);\n");
}

int main() {
    printf("GGML Tensor Layout Visualizer - Broadcasting Patterns\n");
    printf("This tool demonstrates tensor broadcasting patterns for NUMA kernel debugging.\n");
    
    // Initialize GGML context
    struct ggml_init_params params = {
        .mem_size = 16 * 1024 * 1024,  // 16MB
        .mem_buffer = nullptr,
        .no_alloc = false
    };
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        printf("Failed to initialize GGML context\n");
        return 1;
    }
    
    try {
        visualize_broadcasting_patterns(ctx);
        visualize_3d_broadcasting(ctx);
        show_broadcasting_debug_tips(ctx);
    } catch (const std::exception& e) {
        printf("Error during visualization: %s\n", e.what());
    }
    
    ggml_free(ctx);
    
    printf("\n" "="*80 "\n");
    printf("Broadcasting visualization complete!\n");
    printf("Use these patterns to debug tensor shape mismatches in NUMA kernels.\n");
    printf("See docs/ggml-tensor-layout-visualization.md for comprehensive reference.\n");
    printf("="*80 "\n");
    
    return 0;
}
