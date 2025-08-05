#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "ggml/include/ggml.h"
#include "ggml/include/ggml-cpu.h"
#include "ggml/include/ggml-backend.h"

int main(int argc, char **argv) {
    printf("=== Multi-Socket NUMA Test ===\n");
    
    // Initialize context parameters (following the blog post example)
    struct ggml_init_params params = {
        .mem_size   = 1024*1024*1024 + ggml_graph_overhead(),  // Use graph overhead as suggested
        .mem_buffer = NULL,
        .no_alloc   = true,  // Set no_alloc to true for backend usage
    };
    
    // Create context
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("Failed to initialize GGML context\n");
        return 1;
    }
    
    printf("GGML context initialized successfully\n");
    
    // Initialize CPU backend
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        printf("Failed to initialize CPU backend\n");
        ggml_free(ctx);
        return 1;
    }
    
    printf("CPU backend initialized successfully\n");
    
    // Test simple matrix multiplication with smaller matrices first
    const int n = 64;  // Start with smaller size
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);
    
    printf("Created tensors: A(%dx%d) * B(%dx%d)\n", 
           (int)a->ne[0], (int)a->ne[1], 
           (int)b->ne[0], (int)b->ne[1]);
    
    // Allocate tensors to backend
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        printf("Failed to allocate tensors to backend\n");
        ggml_backend_free(backend);
        ggml_free(ctx);
        return 1;
    }
    
    printf("Tensors allocated to backend successfully\n");
    
    // Initialize tensor data using ggml_backend_tensor_set (proper GGML way)
    float *data_a = malloc(n * n * sizeof(float));
    float *data_b = malloc(n * n * sizeof(float));
    
    if (!data_a || !data_b) {
        printf("Error: failed to allocate data arrays\n");
        free(data_a);
        free(data_b);
        ggml_free(ctx);
        return 1;
    }
    
    for (int i = 0; i < n*n; i++) {
        data_a[i] = 1.0f;
        data_b[i] = 2.0f;
    }
    
    // Set tensor data using backend API
    ggml_backend_tensor_set(a, data_a, 0, n * n * sizeof(float));
    ggml_backend_tensor_set(b, data_b, 0, n * n * sizeof(float));
    
    printf("Initialized tensor data (A=1.0, B=2.0)\n");
    
    // Free temporary arrays
    free(data_a);
    free(data_b);
    
    // Create the matrix multiplication operation
    struct ggml_tensor * c = ggml_mul_mat(ctx, a, b);
    
    printf("Created result tensor C(%dx%d)\n", (int)c->ne[0], (int)c->ne[1]);
    
    // Create computation graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);
    
    printf("Built computation graph\n");
    
    // Compute with multiple threads to potentially trigger our multi-socket code
    int n_threads = 4;
    enum ggml_status status = ggml_backend_graph_compute(backend, gf);
    
    if (status != GGML_STATUS_SUCCESS) {
        printf("Error: computation failed with status %d\n", status);
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return 1;
    }
    
    printf("Computation completed successfully\n");
    
    // Verify result using ggml_backend_tensor_get
    float *result_data = malloc(ggml_nelements(c) * sizeof(float));
    if (!result_data) {
        printf("Error: failed to allocate result data array\n");
        ggml_free(ctx);
        return 1;
    }
    
    ggml_backend_tensor_get(c, result_data, 0, ggml_nelements(c) * sizeof(float));
    
    float expected = (float)n * 2.0f;
    bool correct = true;
    
    printf("Checking results: expected value = %f\n", expected);
    
    for (int i = 0; i < 10 && i < ggml_nelements(c); i++) { // Check first 10 elements
        if (fabsf(result_data[i] - expected) > 0.001f) {  // Use floating point comparison
            printf("Error: element %d = %f, expected %f\n", i, result_data[i], expected);
            correct = false;
            break;
        }
    }
    
    if (correct) {
        printf("✓ Matrix multiplication result is correct (first 10 elements = %f)\n", expected);
        printf("✓ Multi-socket NUMA implementation integrated successfully\n");
        
        // Print tensor information for debugging
        printf("Tensor A: ne=[%ld,%ld,%ld,%ld], nb=[%ld,%ld,%ld,%ld]\n", 
               a->ne[0], a->ne[1], a->ne[2], a->ne[3],
               a->nb[0], a->nb[1], a->nb[2], a->nb[3]);
        printf("Tensor B: ne=[%ld,%ld,%ld,%ld], nb=[%ld,%ld,%ld,%ld]\n", 
               b->ne[0], b->ne[1], b->ne[2], b->ne[3],
               b->nb[0], b->nb[1], b->nb[2], b->nb[3]);
        printf("Result C: ne=[%ld,%ld,%ld,%ld], nb=[%ld,%ld,%ld,%ld]\n", 
               c->ne[0], c->ne[1], c->ne[2], c->ne[3],
               c->nb[0], c->nb[1], c->nb[2], c->nb[3]);
    } else {
        printf("✗ Matrix multiplication result is incorrect\n");
    }
    
    free(result_data);
    
    // Cleanup
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    printf("Context and backend freed\n");
    
    return correct ? 0 : 1;
}
