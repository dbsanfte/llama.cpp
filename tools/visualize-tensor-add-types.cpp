/**
 * GGML Tensor Layout Visualizer - ADD Operation Types
 * 
 * This tool demonstrates the specific tensor layouts and data types used in 
 * the ADD mathematical correctness tests, helping debug NUMA kernel development.
 * 
 * @author David Sanftenberg
 */

#include <cstdio>
#include <cstring>
#include <vector>
#include <iomanip>

// GGML includes
#include "ggml.h"
#include "ggml-cpu.h"

/**
 * Type combination structure matching the ADD test
 */
struct TypeCombination {
    ggml_type src0_type;
    ggml_type src1_type;
    ggml_type dst_type;
    const char* description;
    bool is_quantized;
};

/**
 * Get all type combinations from ADD test
 */
std::vector<TypeCombination> get_add_test_combinations() {
    return {
        // Non-quantized types (from binary-ops.cpp)
        {GGML_TYPE_F32,  GGML_TYPE_F32,  GGML_TYPE_F32,  "F32 + F32 → F32", false},
        {GGML_TYPE_F16,  GGML_TYPE_F16,  GGML_TYPE_F16,  "F16 + F16 → F16", false},
        {GGML_TYPE_BF16, GGML_TYPE_BF16, GGML_TYPE_BF16, "BF16 + BF16 → BF16", false},
        {GGML_TYPE_BF16, GGML_TYPE_F32,  GGML_TYPE_BF16, "BF16 + F32 → BF16", false},
        {GGML_TYPE_BF16, GGML_TYPE_F32,  GGML_TYPE_F32,  "BF16 + F32 → F32", false},
        {GGML_TYPE_F16,  GGML_TYPE_F32,  GGML_TYPE_F16,  "F16 + F32 → F16", false},
        {GGML_TYPE_F16,  GGML_TYPE_F32,  GGML_TYPE_F32,  "F16 + F32 → F32", false},
        
        // Quantized types (from ops.cpp) - all follow pattern: Quantized + F32 → Quantized
        {GGML_TYPE_Q4_0,    GGML_TYPE_F32, GGML_TYPE_Q4_0,    "Q4_0 + F32 → Q4_0", true},
        {GGML_TYPE_Q4_1,    GGML_TYPE_F32, GGML_TYPE_Q4_1,    "Q4_1 + F32 → Q4_1", true},
        {GGML_TYPE_Q5_0,    GGML_TYPE_F32, GGML_TYPE_Q5_0,    "Q5_0 + F32 → Q5_0", true},
        {GGML_TYPE_Q5_1,    GGML_TYPE_F32, GGML_TYPE_Q5_1,    "Q5_1 + F32 → Q5_1", true},
        {GGML_TYPE_Q8_0,    GGML_TYPE_F32, GGML_TYPE_Q8_0,    "Q8_0 + F32 → Q8_0", true},
        {GGML_TYPE_Q2_K,    GGML_TYPE_F32, GGML_TYPE_Q2_K,    "Q2_K + F32 → Q2_K", true},
        {GGML_TYPE_Q3_K,    GGML_TYPE_F32, GGML_TYPE_Q3_K,    "Q3_K + F32 → Q3_K", true},
        {GGML_TYPE_Q4_K,    GGML_TYPE_F32, GGML_TYPE_Q4_K,    "Q4_K + F32 → Q4_K", true},
        {GGML_TYPE_Q5_K,    GGML_TYPE_F32, GGML_TYPE_Q5_K,    "Q5_K + F32 → Q5_K", true},
        {GGML_TYPE_Q6_K,    GGML_TYPE_F32, GGML_TYPE_Q6_K,    "Q6_K + F32 → Q6_K", true}
    };
}

/**
 * Demonstrate tensor creation and layout for a specific type combination
 */
void demonstrate_type_combination(struct ggml_context* ctx, const TypeCombination& combo) {
    printf("\n" "="*80 "\n");
    printf("ADD OPERATION: %s\n", combo.description);
    printf("="*80 "\n");
    
    const int ne0 = 8, ne1 = 4;  // Small tensor for clear visualization
    
    // Create tensors
    struct ggml_tensor* src0 = ggml_new_tensor_2d(ctx, combo.src0_type, ne0, ne1);
    struct ggml_tensor* src1 = ggml_new_tensor_2d(ctx, combo.src1_type, ne0, ne1);
    struct ggml_tensor* dst = ggml_new_tensor_2d(ctx, combo.dst_type, ne0, ne1);
    
    printf("Tensor Properties:\n");
    printf("  src0: %s [%dx%d] = %zu bytes\n", ggml_type_name(combo.src0_type), ne0, ne1, ggml_nbytes(src0));
    printf("  src1: %s [%dx%d] = %zu bytes\n", ggml_type_name(combo.src1_type), ne0, ne1, ggml_nbytes(src1));
    printf("  dst:  %s [%dx%d] = %zu bytes\n", ggml_type_name(combo.dst_type), ne0, ne1, ggml_nbytes(dst));
    
    printf("\nStride Information:\n");
    printf("  src0 strides: [%zu, %zu, %zu, %zu] bytes\n", src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
    printf("  src1 strides: [%zu, %zu, %zu, %zu] bytes\n", src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]);
    printf("  dst  strides: [%zu, %zu, %zu, %zu] bytes\n", dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
    
    // Fill source tensors with sample data
    if (combo.src0_type == GGML_TYPE_F32) {
        float* data = (float*)src0->data;
        for (int i = 0; i < ne0 * ne1; i++) {
            data[i] = (float)(i + 1) * 0.5f;
        }
        
        printf("\nsrc0 data (F32):\n");
        for (int row = 0; row < ne1; row++) {
            printf("  Row %d: ", row);
            for (int col = 0; col < ne0; col++) {
                printf("%5.1f ", data[row * ne0 + col]);
            }
            printf("\n");
        }
    } else if (combo.src0_type == GGML_TYPE_F16) {
        ggml_fp16_t* data = (ggml_fp16_t*)src0->data;
        for (int i = 0; i < ne0 * ne1; i++) {
            data[i] = ggml_fp32_to_fp16((float)(i + 1) * 0.5f);
        }
        
        printf("\nsrc0 data (F16):\n");
        for (int row = 0; row < ne1; row++) {
            printf("  Row %d: ", row);
            for (int col = 0; col < ne0; col++) {
                printf("%5.1f ", ggml_fp16_to_fp32(data[row * ne0 + col]));
            }
            printf("\n");
        }
    } else if (ggml_is_quantized(combo.src0_type)) {
        // For quantized types, create F32 source and quantize
        std::vector<float> source_f32(ne0 * ne1);
        for (int i = 0; i < ne0 * ne1; i++) {
            source_f32[i] = (float)(i + 1) * 0.5f;
        }
        
        ggml_quantize_chunk(combo.src0_type, source_f32.data(), src0->data, 0, 1, ne0 * ne1, nullptr);
        
        printf("\nsrc0 data (%s, quantized from F32):\n", ggml_type_name(combo.src0_type));
        
        // Dequantize for display
        std::vector<float> dequantized(ne0 * ne1);
        if (combo.src0_type == GGML_TYPE_Q4_0) {
            ggml_dequantize_row_q4_0((const void*)src0->data, dequantized.data(), ne0 * ne1);
        } else if (combo.src0_type == GGML_TYPE_Q8_0) {
            ggml_dequantize_row_q8_0((const void*)src0->data, dequantized.data(), ne0 * ne1);
        }
        
        for (int row = 0; row < ne1; row++) {
            printf("  Row %d: ", row);
            for (int col = 0; col < ne0; col++) {
                printf("%5.1f ", dequantized[row * ne0 + col]);
            }
            printf("\n");
        }
        
        if (combo.src0_type == GGML_TYPE_Q4_0 || combo.src0_type == GGML_TYPE_Q8_0) {
            printf("\n  Block structure analysis:\n");
            int block_size = ggml_blck_size(combo.src0_type);
            int num_blocks = (ne0 * ne1 + block_size - 1) / block_size;
            printf("    Block size: %d elements\n", block_size);
            printf("    Number of blocks: %d\n", num_blocks);
            printf("    Block bytes: %zu\n", ggml_type_size(combo.src0_type));
        }
    }
    
    // Show src1 data (always F32 for quantized combinations)
    if (combo.src1_type == GGML_TYPE_F32) {
        float* data = (float*)src1->data;
        for (int i = 0; i < ne0 * ne1; i++) {
            data[i] = 0.1f;  // Small additive value
        }
        
        printf("\nsrc1 data (F32, broadcast-compatible):\n");
        printf("  All elements: 0.1\n");
    }
    
    printf("\nElement Access Pattern for NUMA Kernels:\n");
    printf("  Element-wise iteration (flattened):\n");
    printf("    for (size_t i = thread_start; i < thread_end; i++) {\n");
    printf("      // Access: src0[i], src1[i] → dst[i]\n");
    printf("    }\n");
    
    printf("\n  Row-wise iteration:\n");
    printf("    for (int row = row_start; row < row_end; row++) {\n");
    printf("      float* src0_row = (float*)((char*)src0->data + row * src0->nb[1]);\n");
    printf("      float* src1_row = (float*)((char*)src1->data + row * src1->nb[1]);\n");
    printf("      float* dst_row = (float*)((char*)dst->data + row * dst->nb[1]);\n");
    printf("      for (int col = 0; col < ne0; col++) {\n");
    printf("        // Process: src0_row[col] + src1_row[col] → dst_row[col]\n");
    printf("      }\n");
    printf("    }\n");
    
    if (combo.is_quantized) {
        printf("\n  Quantized Processing Notes:\n");
        printf("    - Source tensor may need dequantization to F32 for computation\n");
        printf("    - Result may need requantization back to target format\n");
        printf("    - Block boundaries must be respected for quantized formats\n");
        printf("    - NUMA slicing should align with block boundaries when possible\n");
    }
}

/**
 * Show memory layout comparison for different data types
 */
void compare_memory_layouts(struct ggml_context* ctx) {
    printf("\n" "="*80 "\n");
    printf("MEMORY LAYOUT COMPARISON FOR ADD OPERATIONS\n");
    printf("="*80 "\n");
    
    const int ne0 = 32, ne1 = 1;  // 32-element vector for block alignment
    
    printf("Tensor dimensions: [%d, %d] = %d elements\n", ne0, ne1, ne0 * ne1);
    printf("\nMemory usage comparison:\n");
    printf("Type      | Element Size | Total Bytes | Compression | Block Size\n");
    printf("----------|--------------|-------------|-------------|------------\n");
    
    std::vector<ggml_type> types = {
        GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
        GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_K
    };
    
    size_t f32_bytes = ne0 * ne1 * sizeof(float);
    
    for (ggml_type type : types) {
        struct ggml_tensor* tensor = ggml_new_tensor_2d(ctx, type, ne0, ne1);
        
        size_t element_size = ggml_type_size(type);
        size_t total_bytes = ggml_nbytes(tensor);
        float compression = 100.0f * total_bytes / f32_bytes;
        int block_size = ggml_blck_size(type);
        
        printf("%-9s | %12zu | %11zu | %10.1f%% | %10d\n",
               ggml_type_name(type), element_size, total_bytes, compression, block_size);
    }
    
    printf("\nNUMA Slicing Considerations:\n");
    printf("- F32/F16/BF16: Can slice at any element boundary\n");
    printf("- Q8_0/Q4_0: Prefer slicing at 32-element boundaries\n");
    printf("- Q*_K types: Prefer slicing at 256-element boundaries\n");
    printf("- Unaligned slicing may require partial block processing\n");
    
    printf("\nBroadcasting Patterns in ADD:\n");
    printf("1. Element-wise: [N] + [N] → [N]\n");
    printf("2. Matrix + Vector: [W,H] + [W,1] → [W,H]\n");
    printf("3. Matrix + Scalar: [W,H] + [1,1] → [W,H]\n");
    printf("4. Quantized + F32: [Q*] + [F32] → [Q*]\n");
}

/**
 * Demonstrate NUMA slicing for quantized tensors
 */
void demonstrate_numa_quantized_slicing(struct ggml_context* ctx) {
    printf("\n" "="*80 "\n");
    printf("NUMA SLICING FOR QUANTIZED TENSORS\n");
    printf("="*80 "\n");
    
    const int total_elements = 256;  // Exactly one Q4_K super-block
    const int numa_nodes = 2;
    const int threads_per_node = 2;
    
    struct ggml_tensor* q4k_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_Q4_K, total_elements);
    
    printf("Tensor: %s with %d elements\n", ggml_type_name(GGML_TYPE_Q4_K), total_elements);
    printf("Block size: %d elements per super-block\n", ggml_blck_size(GGML_TYPE_Q4_K));
    printf("Total blocks: %d\n", total_elements / ggml_blck_size(GGML_TYPE_Q4_K));
    printf("Total bytes: %zu\n", ggml_nbytes(q4k_tensor));
    
    printf("\nNUMA Slicing Strategy:\n");
    
    // Strategy 1: Block-aligned slicing
    int block_size = ggml_blck_size(GGML_TYPE_Q4_K);
    int total_blocks = total_elements / block_size;
    int blocks_per_node = total_blocks / numa_nodes;
    
    printf("\n1. Block-aligned slicing (recommended):\n");
    for (int node = 0; node < numa_nodes; node++) {
        int node_block_start = node * blocks_per_node;
        int node_block_end = (node == numa_nodes - 1) ? total_blocks : node_block_start + blocks_per_node;
        int node_element_start = node_block_start * block_size;
        int node_element_end = node_block_end * block_size;
        
        printf("  Node %d: blocks %d-%d, elements %d-%d\n", 
               node, node_block_start, node_block_end - 1, node_element_start, node_element_end - 1);
        
        int node_blocks = node_block_end - node_block_start;
        int blocks_per_thread = node_blocks / threads_per_node;
        
        for (int thread = 0; thread < threads_per_node; thread++) {
            int thread_block_start = node_block_start + thread * blocks_per_thread;
            int thread_block_end = (thread == threads_per_node - 1) ? 
                                  node_block_end : thread_block_start + blocks_per_thread;
            int thread_element_start = thread_block_start * block_size;
            int thread_element_end = thread_block_end * block_size;
            
            printf("    Thread %d: blocks %d-%d, elements %d-%d\n",
                   node * threads_per_node + thread,
                   thread_block_start, thread_block_end - 1,
                   thread_element_start, thread_element_end - 1);
        }
    }
    
    printf("\n2. Element-based slicing (less optimal):\n");
    int elements_per_node = total_elements / numa_nodes;
    for (int node = 0; node < numa_nodes; node++) {
        int node_start = node * elements_per_node;
        int node_end = (node == numa_nodes - 1) ? total_elements : node_start + elements_per_node;
        
        printf("  Node %d: elements %d-%d", node, node_start, node_end - 1);
        
        // Check block alignment
        if (node_start % block_size != 0 || (node_end % block_size != 0 && node_end != total_elements)) {
            printf(" ⚠️  NOT BLOCK-ALIGNED");
        }
        printf("\n");
    }
    
    printf("\nRecommendations for Quantized NUMA Kernels:\n");
    printf("- Always use block-aligned slicing when possible\n");
    printf("- For partial blocks, handle remainder elements separately\n");
    printf("- Consider dequantizing entire blocks rather than partial ranges\n");
    printf("- Cache dequantized blocks when processing multiple operations\n");
}

int main() {
    printf("GGML Tensor Layout Visualizer - ADD Operation Types\n");
    printf("This tool demonstrates tensor layouts used in ADD mathematical correctness tests.\n");
    
    // Initialize GGML context
    struct ggml_init_params params = {
        .mem_size = 64 * 1024 * 1024,  // 64MB
        .mem_buffer = nullptr,
        .no_alloc = false
    };
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        printf("Failed to initialize GGML context\n");
        return 1;
    }
    
    try {
        // Get type combinations from ADD test
        auto combinations = get_add_test_combinations();
        
        // Demonstrate a few key combinations
        std::vector<int> demo_indices = {0, 1, 2, 7, 12};  // F32+F32, F16+F16, BF16+BF16, Q4_0+F32, Q4_K+F32
        
        for (int idx : demo_indices) {
            if (idx < combinations.size()) {
                demonstrate_type_combination(ctx, combinations[idx]);
            }
        }
        
        compare_memory_layouts(ctx);
        demonstrate_numa_quantized_slicing(ctx);
        
    } catch (const std::exception& e) {
        printf("Error during visualization: %s\n", e.what());
    }
    
    ggml_free(ctx);
    
    printf("\n" "="*80 "\n");
    printf("ADD operation type visualization complete!\n");
    printf("This covers all %zu type combinations tested in test-numa-mathematical-correctness-add.cpp\n", 
           get_add_test_combinations().size());
    printf("See docs/ggml-tensor-layout-visualization.md for comprehensive reference.\n");
    printf("="*80 "\n");
    
    return 0;
}
