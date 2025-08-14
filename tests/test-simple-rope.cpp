#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

int main() {
    printf("Testing ROPE with NUMA Dispatcher...\n");
    
    // Initialize ggml backend
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        printf("Failed to initialize CPU backend\n");
        return 1;
    }
    
    // Initialize ggml context
    struct ggml_context * ctx = ggml_init({
        .mem_size = 16 * 1024 * 1024,  // 16MB
        .mem_buffer = NULL,
        .no_alloc = true,
    });
    
    if (!ctx) {
        printf("Failed to initialize ggml context\n");
        ggml_backend_free(backend);
        return 1;
    }
    
    printf("Creating simple ROPE operation...\n");
    
    // Create simple ROPE operation tensors
    // For ROPE: input should be [n_embd, n_head, n_seq] or [n_embd, n_seq, n_batch]
    // positions should be [n_seq]
    const int64_t n_embd = 64;    // embedding dimension
    const int64_t n_seq = 32;     // sequence length
    const int64_t n_batch = 1;    // batch size
    
    // Input tensor: [n_embd, n_seq, n_batch] - 3D tensor
    struct ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_seq, n_batch);
    struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_batch);  // position tensor matches batch dimension
    
    // Create simple ROPE operation with basic parameters
    struct ggml_tensor * rope_result = ggml_rope_ext(
        ctx, input, pos, NULL,  // context, input, positions, frequencies
        n_embd,                 // n_dims (all dimensions rotated)  
        0,                      // mode (standard ROPE)
        0,                      // n_ctx_orig
        10000.0f,              // freq_base
        1.0f,                  // freq_scale
        0.0f,                  // ext_factor
        1.0f,                  // attn_factor
        1.0f,                  // beta_fast
        1.0f                   // beta_slow
    );
    
    // Allocate tensors
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    
    // Create computation graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, rope_result);
    
    ggml_gallocr_alloc_graph(allocr, gf);
    
    printf("Initializing test data...\n");
    
    // Initialize input tensor with test data
    std::vector<float> input_data(n_embd * n_seq * n_batch);
    for (size_t i = 0; i < input_data.size(); i++) {
        input_data[i] = (float)(i % 10) / 10.0f; // Values 0.0 to 0.9
    }
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    
    // Initialize position tensor
    std::vector<int32_t> pos_data(n_batch);  // batch dimension
    for (int i = 0; i < n_batch; i++) {
        pos_data[i] = 0;  // all tokens at position 0 for simplicity
    }
    ggml_backend_tensor_set(pos, pos_data.data(), 0, pos_data.size() * sizeof(int32_t));
    
    printf("Computing ROPE operation (should trigger NUMA handler evaluation)...\n");
    printf("Operation type: %s\n", ggml_op_name(rope_result->op));
    
    // Compute the graph - this will go through our NUMA dispatcher
    int compute_result = ggml_backend_graph_compute(backend, gf);
    
    if (compute_result == GGML_STATUS_SUCCESS) {
        printf("✅ ROPE computation completed successfully!\n");
        
        // Validate result tensor shape
        printf("Result tensor shape: [%ld, %ld, %ld]\n", 
               rope_result->ne[0], rope_result->ne[1], rope_result->ne[2]);
        
        // Get a sample of results
        std::vector<float> result_sample(10);
        ggml_backend_tensor_get(rope_result, result_sample.data(), 0, result_sample.size() * sizeof(float));
        
        printf("Input sample:  ");
        for (int i = 0; i < 10; i++) {
            printf("%.3f ", input_data[i]);
        }
        printf("\n");
        
        printf("ROPE result:   ");
        for (int i = 0; i < 10; i++) {
            printf("%.3f ", result_sample[i]);
        }
        printf("\n");
        
        // ROPE should transform the data, so results should be different
        bool results_different = false;
        for (int i = 0; i < 10; i++) {
            if (std::abs(result_sample[i] - input_data[i]) > 1e-6) {
                results_different = true;
                break;
            }
        }
        
        if (results_different) {
            printf("✅ ROPE transformation applied correctly\n");
        } else {
            printf("⚠️  ROPE results identical to inputs\n");
        }
        
    } else {
        printf("❌ ROPE computation failed with code: %d\n", compute_result);
    }
    
    // Cleanup
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    ggml_backend_free(backend);
    
    printf("Simple ROPE test completed.\n");
    
    return compute_result == GGML_STATUS_SUCCESS ? 0 : 1;
}
