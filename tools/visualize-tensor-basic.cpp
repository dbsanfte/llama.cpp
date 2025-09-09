/**
 * GGML Tensor Layout Visualizer - Basic Tensor Structure
 * 
 * This tool creates and visualizes basic GGML tensor structures to help understand
 * memory layout, stride calculations, and element access patterns.
 * 
 * @author David Sanftenberg
 */

#include <cstdio>
#include <cstring>
#include <vector>
#include <iomanip>
#include <string>

// GGML includes
#include "ggml.h"

/**
 * Print detailed tensor information including dimensions, strides, and memory layout
 */
void print_tensor_info(const struct ggml_tensor* tensor, const char* name) {
    printf("\n=== Tensor: %s ===\n", name);
    printf("Type: %s\n", ggml_type_name(tensor->type));
    printf("Dimensions (ne): [%ld, %ld, %ld, %ld]\n", 
           (long)tensor->ne[0], (long)tensor->ne[1], (long)tensor->ne[2], (long)tensor->ne[3]);
    printf("Strides (nb):    [%zu, %zu, %zu, %zu] bytes\n",
           tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]);
    printf("Total elements: %ld\n", (long)ggml_nelements(tensor));
    printf("Total bytes: %zu\n", ggml_nbytes(tensor));
    printf("Element size: %zu bytes\n", ggml_type_size(tensor->type));
    
    if (ggml_is_quantized(tensor->type)) {
        printf("Block size: %ld elements\n", (long)ggml_blck_size(tensor->type));
        printf("Bytes per block: %zu\n", ggml_type_size(tensor->type));
    }
}

/**
 * Visualize memory layout of a 2D F32 tensor
 */
void visualize_2d_f32_layout(struct ggml_context* ctx) {
    printf("\n%s\n", std::string(80, '=').c_str());
    printf("2D F32 TENSOR MEMORY LAYOUT VISUALIZATION\n");
    printf("%s\n", std::string(80, '=').c_str());
    
    // Create a small 3x4 F32 tensor for visualization
    const int rows = 3, cols = 4;
    struct ggml_tensor* tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
    
    print_tensor_info(tensor, "2D F32 Matrix [4x3]");
    
    // Fill with sample data
    float* data = (float*)tensor_data(tensor);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            data[r * cols + c] = (float)(r * 10 + c);
        }
    }
    
    printf("\nMemory Layout (Conceptual):\n");
    printf("┌─────────────────────────────────────────────┐\n");
    for (int r = 0; r < rows; r++) {
        printf("│ Row %d: ", r);
        for (int c = 0; c < cols; c++) {
            printf("[%5.1f] ", data[r * cols + c]);
        }
        printf("│\n");
    }
    printf("└─────────────────────────────────────────────┘\n");
    
    printf("\nStride-based Element Access:\n");
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            // Using GGML stride-based access
            float* element = (float*)((char*)tensor_data(tensor) + r * tensor->nb[1] + c * tensor->nb[0]);
            size_t offset = (char*)element - (char*)tensor_data(tensor);
            printf("element[%d,%d] at offset %3zu bytes = %5.1f\n", r, c, offset, *element);
        }
    }
    
    printf("\nRow-wise Access Pattern (Common in NUMA kernels):\n");
    for (int r = 0; r < rows; r++) {
        float* row_start = (float*)((char*)tensor_data(tensor) + r * tensor->nb[1]);
        printf("Row %d starts at offset %3zu: ", r, (char*)row_start - (char*)tensor_data(tensor));
        for (int c = 0; c < cols; c++) {
            printf("%5.1f ", row_start[c]);
        }
        printf("\n");
    }
}

/**
 * Visualize NUMA-style data slicing patterns
 */
void visualize_numa_slicing(struct ggml_context* ctx) {
    printf("\n%s\n", std::string(80, '=').c_str());
    printf("NUMA DATA SLICING VISUALIZATION\n");
    printf("%s\n", std::string(80, '=').c_str());
    
    const int total_elements = 32;
    struct ggml_tensor* tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, total_elements);
    
    print_tensor_info(tensor, "1D Vector for NUMA Slicing");
    
    // Fill with sequential data
    float* data = (float*)tensor_data(tensor);
    for (int i = 0; i < total_elements; i++) {
        data[i] = (float)i;
    }
    
    // Simulate NUMA slicing
    const int numa_nodes = 2;
    const int threads_per_node = 2;
    
    printf("\nOriginal data: ");
    for (int i = 0; i < total_elements; i++) {
        printf("%2.0f ", data[i]);
    }
    printf("\n");
    
    printf("\nNUMA Slicing Pattern (%d nodes, %d threads per node):\n", numa_nodes, threads_per_node);
    
    size_t elements_per_node = total_elements / numa_nodes;
    
    for (int node = 0; node < numa_nodes; node++) {
        size_t node_start = node * elements_per_node;
        size_t node_end = (node == numa_nodes - 1) ? total_elements : node_start + elements_per_node;
        
        printf("\n  NUMA Node %d: elements %zu-%zu\n", node, node_start, node_end - 1);
        
        size_t node_elements = node_end - node_start;
        size_t elements_per_thread = node_elements / threads_per_node;
        
        for (int thread = 0; thread < threads_per_node; thread++) {
            size_t thread_start = node_start + (thread * elements_per_thread);
            size_t thread_end = (thread == threads_per_node - 1) ? 
                               node_end : thread_start + elements_per_thread;
            
            printf("    Thread %d: elements %zu-%zu = [", 
                   node * threads_per_node + thread, thread_start, thread_end - 1);
            
            for (size_t i = thread_start; i < thread_end; i++) {
                printf("%2.0f", data[i]);
                if (i < thread_end - 1) printf(",");
            }
            printf("]\n");
        }
    }
    
    printf("\nVisualization:\n");
    printf("Total: [");
    for (int i = 0; i < total_elements; i++) {
        printf("%2.0f", data[i]);
        if (i == (int)elements_per_node - 1) printf("|");
        else if (i < total_elements - 1) printf(" ");
    }
    printf("]\n");
    printf("       │←─── Node 0 ───→│←─── Node 1 ───→│\n");
}

/**
 * Visualize 3D tensor structure and access patterns
 */
void visualize_3d_tensor(struct ggml_context* ctx) {
    printf("\n%s\n", std::string(80, '=').c_str());
    printf("3D TENSOR STRUCTURE VISUALIZATION\n");
    printf("%s\n", std::string(80, '=').c_str());
    
    const int width = 4, height = 3, depth = 2;
    struct ggml_tensor* tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, height, depth);
    
    print_tensor_info(tensor, "3D Tensor [4x3x2]");
    
    // Fill with structured data: depth*100 + row*10 + col
    float* data = (float*)tensor_data(tensor);
    for (int d = 0; d < depth; d++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int linear_idx = d * (height * width) + h * width + w;
                data[linear_idx] = (float)(d * 100 + h * 10 + w);
            }
        }
    }
    
    printf("\nMemory Layout by Depth Slices:\n");
    for (int d = 0; d < depth; d++) {
        printf("\n┌─── Depth %d ──────────────────────────────┐\n", d);
        for (int h = 0; h < height; h++) {
            printf("│ Row %d: ", h);
            for (int w = 0; w < width; w++) {
                float* element = (float*)((char*)tensor_data(tensor) + 
                                        d * tensor->nb[2] + h * tensor->nb[1] + w * tensor->nb[0]);
                printf("[%5.0f] ", *element);
            }
            printf("│\n");
        }
        printf("└────────────────────────────────────────────┘\n");
    }
    
    printf("\nStride Analysis:\n");
    printf("- nb[0] = %zu bytes (element stride)\n", tensor->nb[0]);
    printf("- nb[1] = %zu bytes (row stride) = %zu elements\n", tensor->nb[1], tensor->nb[1] / tensor->nb[0]);
    printf("- nb[2] = %zu bytes (depth stride) = %zu elements\n", tensor->nb[2], tensor->nb[2] / tensor->nb[0]);
    
    printf("\nAccess Pattern Examples:\n");
    printf("Element [depth=1, row=2, col=3]:\n");
    printf("  Offset = 1*nb[2] + 2*nb[1] + 3*nb[0] = 1*%zu + 2*%zu + 3*%zu = %zu bytes\n",
           tensor->nb[2], tensor->nb[1], tensor->nb[0], 
           1*tensor->nb[2] + 2*tensor->nb[1] + 3*tensor->nb[0]);
    
    float* target = (float*)((char*)tensor_data(tensor) + 1*tensor->nb[2] + 2*tensor->nb[1] + 3*tensor->nb[0]);
    printf("  Value = %.0f\n", *target);
}

int main() {
    printf("GGML Tensor Layout Visualizer - Basic Structure\n");
    printf("This tool demonstrates GGML tensor memory layouts and access patterns.\n");
    
    // Initialize GGML context
    struct ggml_init_params params = {
        /*.mem_size =*/ 16 * 1024 * 1024,  // 16MB
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc =*/ false
    };
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        printf("Failed to initialize GGML context\n");
        return 1;
    }
    
    try {
        visualize_2d_f32_layout(ctx);
        visualize_numa_slicing(ctx);
        visualize_3d_tensor(ctx);
    } catch (const std::exception& e) {
        printf("Error during visualization: %s\n", e.what());
    }
    
    ggml_free(ctx);
    
    printf("\n%s\n", std::string(80, '=').c_str());
    printf("Visualization complete! See docs/ggml-tensor-layout-visualization.md for detailed explanations.\n");
    printf("%s\n", std::string(80, '=').c_str());
    
    return 0;
}
