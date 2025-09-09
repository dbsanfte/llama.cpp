/**
 * GGML Tensor Layout Visualizer - Quantization Formats
 * 
 * This tool demonstrates quantized tensor formats used in GGML, showing
 * block structures, dequantization patterns, and memory layouts.
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
 * Print hex dump of raw memory for debugging
 */
void print_hex_dump(const void* data, size_t size, const char* label) {
    const uint8_t* bytes = (const uint8_t*)data;
    printf("\n%s (hex dump, %zu bytes):\n", label, size);
    
    for (size_t i = 0; i < size; i += 16) {
        printf("%04zx: ", i);
        
        // Hex bytes
        for (size_t j = 0; j < 16 && (i + j) < size; j++) {
            printf("%02x ", bytes[i + j]);
        }
        
        // Padding for alignment
        for (size_t j = size - i; j < 16 && j > 0; j++) {
            printf("   ");
        }
        
        printf(" |");
        
        // ASCII representation
        for (size_t j = 0; j < 16 && (i + j) < size; j++) {
            uint8_t c = bytes[i + j];
            printf("%c", (c >= 32 && c <= 126) ? c : '.');
        }
        
        printf("|\n");
    }
}

/**
 * Visualize F32, F16, and BF16 formats 
 */
void visualize_float_formats(struct ggml_context* ctx) {
    printf("\n" "="*80 "\n");
    printf("FLOATING POINT FORMAT VISUALIZATION\n");
    printf("="*80 "\n");
    
    const int num_values = 4;
    float test_values[] = {1.0f, 2.5f, -3.14159f, 0.0f};
    
    // F32 tensor
    struct ggml_tensor* f32_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_values);
    float* f32_data = (float*)f32_tensor->data;
    memcpy(f32_data, test_values, sizeof(test_values));
    
    // F16 tensor  
    struct ggml_tensor* f16_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, num_values);
    ggml_fp16_t* f16_data = (ggml_fp16_t*)f16_tensor->data;
    for (int i = 0; i < num_values; i++) {
        f16_data[i] = ggml_fp32_to_fp16(test_values[i]);
    }
    
    printf("Test values: ");
    for (int i = 0; i < num_values; i++) {
        printf("%8.4f ", test_values[i]);
    }
    printf("\n");
    
    printf("\nF32 Format (4 bytes per element):\n");
    printf("Values:  ");
    for (int i = 0; i < num_values; i++) {
        printf("%8.4f ", f32_data[i]);
    }
    printf("\n");
    printf("Hex:     ");
    for (int i = 0; i < num_values; i++) {
        uint32_t* hex = (uint32_t*)&f32_data[i];
        printf("%08X ", *hex);
    }
    printf("\n");
    
    printf("\nF16 Format (2 bytes per element):\n");
    printf("Values:  ");
    for (int i = 0; i < num_values; i++) {
        printf("%8.4f ", ggml_fp16_to_fp32(f16_data[i]));
    }
    printf("\n");  
    printf("Hex:     ");
    for (int i = 0; i < num_values; i++) {
        printf("    %04X ", f16_data[i]);
    }
    printf("\n");
    
    print_hex_dump(f32_data, num_values * sizeof(float), "F32 Raw Memory");
    print_hex_dump(f16_data, num_values * sizeof(ggml_fp16_t), "F16 Raw Memory");
    
    printf("\nSize comparison:\n");
    printf("- F32: %d elements × %zu bytes = %zu bytes total\n", 
           num_values, sizeof(float), num_values * sizeof(float));
    printf("- F16: %d elements × %zu bytes = %zu bytes total (%.1f%% of F32)\n",
           num_values, sizeof(ggml_fp16_t), num_values * sizeof(ggml_fp16_t),
           100.0f * sizeof(ggml_fp16_t) / sizeof(float));
}

/**
 * Visualize Q4_0 quantization format
 */
void visualize_q4_0_format(struct ggml_context* ctx) {
    printf("\n" "="*80 "\n");
    printf("Q4_0 QUANTIZATION FORMAT VISUALIZATION\n");
    printf("="*80 "\n");
    
    const int num_elements = 32;  // QK4_0 block size
    
    // Create F32 source data
    std::vector<float> source_data(num_elements);
    for (int i = 0; i < num_elements; i++) {
        source_data[i] = (float)(i - 16) * 0.5f;  // Range: -8.0 to 7.5
    }
    
    printf("Source F32 data (%d elements):\n", num_elements);
    for (int i = 0; i < num_elements; i += 8) {
        printf("  ");
        for (int j = 0; j < 8 && (i + j) < num_elements; j++) {
            printf("%6.1f ", source_data[i + j]);
        }
        printf("\n");
    }
    
    // Create Q4_0 tensor and quantize
    struct ggml_tensor* q4_0_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_Q4_0, num_elements);
    ggml_quantize_chunk(GGML_TYPE_Q4_0, source_data.data(), q4_0_tensor->data, 0, 1, num_elements, nullptr);
    
    // Examine the Q4_0 block structure
    typedef struct {
        ggml_fp16_t d;        // scale (2 bytes)
        uint8_t qs[16];       // quantized values (16 bytes, 2 nibbles per byte)
    } block_q4_0;
    
    block_q4_0* block = (block_q4_0*)q4_0_tensor->data;
    
    printf("\nQ4_0 Block Structure:\n");
    printf("  Scale (d): %f (hex: 0x%04X)\n", ggml_fp16_to_fp32(block->d), block->d);
    printf("  Quantized values (qs): ");
    for (int i = 0; i < 16; i++) {
        printf("%02X ", block->qs[i]);
    }
    printf("\n");
    
    printf("\nNibble extraction and dequantization:\n");
    printf("Byte | Hex | Nibbles | Dequantized Values\n");
    printf("-----|-----|---------|-------------------\n");
    
    float scale = ggml_fp16_to_fp32(block->d);
    for (int i = 0; i < 16; i++) {
        uint8_t byte_val = block->qs[i];
        uint8_t nibble0 = byte_val & 0x0F;         // Lower nibble
        uint8_t nibble1 = (byte_val >> 4) & 0x0F; // Upper nibble
        
        float deq0 = scale * (nibble0 - 8);
        float deq1 = scale * (nibble1 - 8);
        
        printf(" %2d  | %02X  |  %X, %X   | %6.2f, %6.2f\n", 
               i, byte_val, nibble0, nibble1, deq0, deq1);
    }
    
    // Dequantize and compare
    std::vector<float> dequantized(num_elements);
    ggml_dequantize_row_q4_0((const block_q4_0*)q4_0_tensor->data, dequantized.data(), num_elements);
    
    printf("\nQuantization accuracy comparison:\n");
    printf("Index | Original | Quantized | Error\n");
    printf("------|----------|-----------|-------\n");
    for (int i = 0; i < num_elements; i++) {
        float error = dequantized[i] - source_data[i];
        printf(" %3d  | %8.3f | %9.3f | %6.3f\n", i, source_data[i], dequantized[i], error);
    }
    
    print_hex_dump(q4_0_tensor->data, ggml_type_size(GGML_TYPE_Q4_0), "Q4_0 Block Raw Memory");
    
    printf("\nQ4_0 Summary:\n");
    printf("- Block size: %d elements\n", 32);
    printf("- Storage: 2 bytes (scale) + 16 bytes (quantized) = 18 bytes\n");
    printf("- Compression: %.2f%% of F32 (18 vs %d bytes)\n", 
           100.0f * 18 / (32 * sizeof(float)), 32 * (int)sizeof(float));
    printf("- Bits per element: %.3f\n", 8.0f * 18 / 32);
}

/**
 * Visualize Q8_0 quantization format
 */
void visualize_q8_0_format(struct ggml_context* ctx) {
    printf("\n" "="*80 "\n");
    printf("Q8_0 QUANTIZATION FORMAT VISUALIZATION\n");
    printf("="*80 "\n");
    
    const int num_elements = 32;  // QK8_0 block size
    
    // Create F32 source data
    std::vector<float> source_data(num_elements);
    for (int i = 0; i < num_elements; i++) {
        source_data[i] = (float)(i - 16) * 0.1f;  // Range: -1.6 to 1.5
    }
    
    printf("Source F32 data (%d elements):\n", num_elements);
    for (int i = 0; i < num_elements; i += 8) {
        printf("  ");
        for (int j = 0; j < 8 && (i + j) < num_elements; j++) {
            printf("%6.2f ", source_data[i + j]);
        }
        printf("\n");
    }
    
    // Create Q8_0 tensor and quantize
    struct ggml_tensor* q8_0_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_Q8_0, num_elements);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, source_data.data(), q8_0_tensor->data, 0, 1, num_elements, nullptr);
    
    // Examine the Q8_0 block structure
    typedef struct {
        ggml_fp16_t d;        // scale (2 bytes)
        int8_t qs[32];        // quantized values (32 bytes)
    } block_q8_0;
    
    block_q8_0* block = (block_q8_0*)q8_0_tensor->data;
    
    printf("\nQ8_0 Block Structure:\n");
    printf("  Scale (d): %f (hex: 0x%04X)\n", ggml_fp16_to_fp32(block->d), block->d);
    printf("  Quantized values (qs):\n");
    for (int i = 0; i < 32; i += 8) {
        printf("    ");
        for (int j = 0; j < 8 && (i + j) < 32; j++) {
            printf("%4d ", block->qs[i + j]);
        }
        printf("\n");
    }
    
    // Dequantize and compare
    std::vector<float> dequantized(num_elements);
    ggml_dequantize_row_q8_0((const block_q8_0*)q8_0_tensor->data, dequantized.data(), num_elements);
    
    printf("\nQuantization accuracy comparison (first 16 elements):\n");
    printf("Index | Original | Int8 | Quantized | Error\n");
    printf("------|----------|------|-----------|-------\n");
    float scale = ggml_fp16_to_fp32(block->d);
    for (int i = 0; i < 16; i++) {
        float error = dequantized[i] - source_data[i];
        printf(" %3d  | %8.3f | %4d | %9.3f | %6.3f\n", 
               i, source_data[i], block->qs[i], dequantized[i], error);
    }
    
    print_hex_dump(q8_0_tensor->data, ggml_type_size(GGML_TYPE_Q8_0), "Q8_0 Block Raw Memory");
    
    printf("\nQ8_0 Summary:\n");
    printf("- Block size: %d elements\n", 32);
    printf("- Storage: 2 bytes (scale) + 32 bytes (quantized) = 34 bytes\n");
    printf("- Compression: %.2f%% of F32 (34 vs %d bytes)\n", 
           100.0f * 34 / (32 * sizeof(float)), 32 * (int)sizeof(float));
    printf("- Bits per element: %.3f\n", 8.0f * 34 / 32);
}

/**
 * Compare different quantization formats side by side
 */
void compare_quantization_formats(struct ggml_context* ctx) {
    printf("\n" "="*80 "\n");
    printf("QUANTIZATION FORMAT COMPARISON\n");
    printf("="*80 "\n");
    
    const int num_elements = 32;
    
    // Create test data with varying ranges to show quantization effects
    std::vector<float> source_data(num_elements);
    for (int i = 0; i < num_elements; i++) {
        // Mix of small and large values
        if (i < 16) {
            source_data[i] = (float)(i - 8) * 0.1f;       // Small values: -0.8 to 0.7
        } else {
            source_data[i] = (float)(i - 24) * 2.0f;      // Larger values: -16 to 14
        }
    }
    
    printf("Source data (mixed range):\n");
    for (int i = 0; i < num_elements; i += 8) {
        printf("  ");
        for (int j = 0; j < 8 && (i + j) < num_elements; j++) {
            printf("%6.1f ", source_data[i + j]);
        }
        printf("\n");
    }
    
    // Create tensors for different quantization types
    std::vector<ggml_type> quant_types = {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0};
    std::vector<const char*> type_names = {"F32", "F16", "Q8_0", "Q4_0"};
    
    printf("\nSize and accuracy comparison:\n");
    printf("Format | Size (bytes) | Ratio | Max Error | Avg Error\n");
    printf("-------|--------------|-------|-----------|----------\n");
    
    for (size_t t = 0; t < quant_types.size(); t++) {
        ggml_type type = quant_types[t];
        
        struct ggml_tensor* tensor = ggml_new_tensor_1d(ctx, type, num_elements);
        
        if (type == GGML_TYPE_F32) {
            memcpy(tensor->data, source_data.data(), num_elements * sizeof(float));
        } else {
            ggml_quantize_chunk(type, source_data.data(), tensor->data, 0, 1, num_elements, nullptr);
        }
        
        // Dequantize for comparison
        std::vector<float> dequantized(num_elements);
        if (type == GGML_TYPE_F32) {
            memcpy(dequantized.data(), tensor->data, num_elements * sizeof(float));
        } else if (type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((ggml_fp16_t*)tensor->data, dequantized.data(), num_elements);
        } else if (type == GGML_TYPE_Q8_0) {
            ggml_dequantize_row_q8_0((const void*)tensor->data, dequantized.data(), num_elements);
        } else if (type == GGML_TYPE_Q4_0) {
            ggml_dequantize_row_q4_0((const void*)tensor->data, dequantized.data(), num_elements);
        }
        
        // Calculate errors
        float max_error = 0.0f, sum_error = 0.0f;
        for (int i = 0; i < num_elements; i++) {
            float error = fabsf(dequantized[i] - source_data[i]);
            max_error = fmaxf(max_error, error);
            sum_error += error;
        }
        float avg_error = sum_error / num_elements;
        
        size_t bytes = ggml_nbytes(tensor);
        float ratio = 100.0f * bytes / (num_elements * sizeof(float));
        
        printf("%-6s | %12zu | %5.1f%% | %9.4f | %8.4f\n",
               type_names[t], bytes, ratio, max_error, avg_error);
    }
    
    printf("\nKey observations:\n");
    printf("- F32: Full precision, largest size\n");
    printf("- F16: Good precision, 50%% size reduction\n");
    printf("- Q8_0: Block-based, ~33%% size, small quantization error\n");
    printf("- Q4_0: Aggressive compression, ~18%% size, higher quantization error\n");
}

int main() {
    printf("GGML Tensor Layout Visualizer - Quantization Formats\n");
    printf("This tool demonstrates quantized tensor formats and their memory layouts.\n");
    
    // Initialize GGML context
    struct ggml_init_params params = {
        .mem_size = 32 * 1024 * 1024,  // 32MB
        .mem_buffer = nullptr,
        .no_alloc = false
    };
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        printf("Failed to initialize GGML context\n");
        return 1;
    }
    
    try {
        visualize_float_formats(ctx);
        visualize_q4_0_format(ctx);
        visualize_q8_0_format(ctx);
        compare_quantization_formats(ctx);
    } catch (const std::exception& e) {
        printf("Error during visualization: %s\n", e.what());
    }
    
    ggml_free(ctx);
    
    printf("\n" "="*80 "\n");
    printf("Quantization visualization complete!\n");
    printf("See docs/ggml-tensor-layout-visualization.md for detailed explanations.\n");
    printf("="*80 "\n");
    
    return 0;
}
