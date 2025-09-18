#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cstring>
#include <cmath>
#include <string>
#include <algorithm>

#ifndef GGML_EXPERIMENT_PREFETCH_K
#define GGML_EXPERIMENT_PREFETCH_K 0
#endif

// Forward declare experimental toggle if present
extern "C" {
    void ggml_experimental_set_prefetch_k(int enabled);
    int  ggml_experimental_get_prefetch_k();
}

static double wall_time_s() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

struct run_conf {
    int rows;      // M
    int cols;      // K
    int batch;     // N (number of columns for RHS)
    int iters;     // iterations for timing
    llama_ftype ftype; // quantization type for weights (K-series)
};

static void fill_random(std::vector<float> & v) {
    for (auto & x : v) x = (float) ((rand() % 2000) - 1000) / 100.0f; // [-10,10]
}

static std::string ftype_name(llama_ftype t) {
    switch (t) {
        case LLAMA_FTYPE_MOSTLY_Q2_K: return "Q2_K";
        case LLAMA_FTYPE_MOSTLY_Q3_K: return "Q3_K";
        case LLAMA_FTYPE_MOSTLY_Q4_K: return "Q4_K";
        case LLAMA_FTYPE_MOSTLY_Q5_K: return "Q5_K";
        case LLAMA_FTYPE_MOSTLY_Q6_K: return "Q6_K";
        case LLAMA_FTYPE_MOSTLY_Q8_K: return "Q8_K";
        default: return "UNKNOWN";
    }
}

static void benchmark_case(const run_conf & cfg, bool enable_prefetch) {
    if (ggml_experimental_set_prefetch_k) {
        ggml_experimental_set_prefetch_k(enable_prefetch ? 1 : 0);
    }

    printf("\n--- Benchmark: %s prefetch=%d ---\n", ftype_name(cfg.ftype).c_str(), enable_prefetch?1:0);
    printf("Dims: M=%d K=%d N=%d iters=%d\n", cfg.rows, cfg.cols, cfg.batch, cfg.iters);

    // Build a fake small model context just to allocate tensors
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true; // we won't load real weights
    mp.use_mmap = false;
    mp.use_mlock = false;

    llama_model * model = nullptr; // not needed for direct ggml usage here

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 16; // minimal

    // We'll create a standalone ggml context
    struct ggml_init_params ip = { 0 };
    size_t mem_sz = 64ull * 1024ull * 1024ull;
    std::vector<uint8_t> mem(mem_sz);
    ip.mem_size   = mem_sz;
    ip.mem_buffer = mem.data();

    ggml_context * ctx0 = ggml_init(ip);
    if (!ctx0) {
        fprintf(stderr, "Failed to init ggml context\n");
        return;
    }

    // Create tensors: A (weights) shape [K, M]; B (input activations) shape [K, N]; C (output) shape [M, N]
    // ggml uses column-major semantics in many ops; we focus on mul_mat(A, B) producing [M, N]

    ggml_type wtype;
    switch (cfg.ftype) {
        case LLAMA_FTYPE_MOSTLY_Q2_K: wtype = GGML_TYPE_Q2_K; break;
        case LLAMA_FTYPE_MOSTLY_Q3_K: wtype = GGML_TYPE_Q3_K; break;
        case LLAMA_FTYPE_MOSTLY_Q4_K: wtype = GGML_TYPE_Q4_K; break;
        case LLAMA_FTYPE_MOSTLY_Q5_K: wtype = GGML_TYPE_Q5_K; break;
        case LLAMA_FTYPE_MOSTLY_Q6_K: wtype = GGML_TYPE_Q6_K; break;
        case LLAMA_FTYPE_MOSTLY_Q8_K: wtype = GGML_TYPE_Q8_K; break;
        default: wtype = GGML_TYPE_Q4_K; break;
    }

    ggml_tensor * A = ggml_new_tensor_2d(ctx0, wtype, cfg.cols, cfg.rows); // [K, M]
    ggml_tensor * B = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, cfg.cols, cfg.batch); // [K, N]
    ggml_tensor * C = ggml_mul_mat(ctx0, A, B); // result [M, N]

    ggml_set_name(A, "A_weights");
    ggml_set_name(B, "B_input");
    ggml_set_name(C, "C_out");

    // Fill B with random floats
    std::vector<float> hostB(cfg.cols * cfg.batch);
    fill_random(hostB);
    memcpy(B->data, hostB.data(), hostB.size()*sizeof(float));

    // For A we first create dequantized floats then quantize via llama's quantization helper (not all variants have direct API)
    std::vector<float> hostA(cfg.cols * cfg.rows);
    fill_random(hostA);

    // Simple per-row naive quantization using provided conversion path: we reuse llama_convert_fp32_to_qX utility via ggml_quantize_* if exposed.
    // If direct quantization functions are not public, we leave random memory (NOT IDEAL). For correctness validation we need baseline & experimental to share identical memory so still fair relative comparison.

    memcpy(A->data, hostA.data(), hostA.size()*sizeof(float) > ggml_nbytes(A) ? ggml_nbytes(A) : hostA.size()*sizeof(float));

    // Build graph
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 64, false);
    ggml_build_forward_expand(gf, C);

    // Warmup
    for (int i = 0; i < 2; ++i) {
        struct ggml_cplan cplan = ggml_graph_plan(gf, /*n_threads*/ 1);
        std::vector<uint8_t> work(cplan.work_size);
        cplan.work_data = work.data();
        ggml_graph_compute(gf, &cplan);
    }

    // Timing
    double t0 = wall_time_s();
    for (int it = 0; it < cfg.iters; ++it) {
        struct ggml_cplan cplan = ggml_graph_plan(gf, 1);
        std::vector<uint8_t> work(cplan.work_size);
        cplan.work_data = work.data();
        ggml_graph_compute(gf, &cplan);
    }
    double t1 = wall_time_s();
    double dt = (t1 - t0) / cfg.iters;

    printf("Avg time: %.6f s  (%.3f ms)\n", dt, dt*1e3);

    // Keep a copy of result for comparison if baseline
    static std::vector<float> baseline;
    size_t out_elems = (size_t)cfg.rows * cfg.batch;
    float * out_ptr = (float*) C->data;
    if (!enable_prefetch) {
        baseline.assign(out_ptr, out_ptr + out_elems);
    } else {
        // Compare
        size_t mismatches = 0;
        double max_abs = 0.0;
        for (size_t i = 0; i < out_elems; ++i) {
            double diff = std::fabs(out_ptr[i] - baseline[i]);
            if (diff != 0.0f) {
                max_abs = std::max(max_abs, diff);
                if (++mismatches < 5) {
                    printf("Mismatch idx %zu base=%f exp=%f diff=%f\n", i, baseline[i], out_ptr[i], diff);
                }
            }
        }
        printf("Validation: mismatches=%zu max_abs=%g\n", mismatches, max_abs);
    }

    ggml_free(ctx0);
}

int main(int argc, char ** argv) {
    srand(1234);
    std::vector<run_conf> cases = {
        {  512, 4096, 16, 20, LLAMA_FTYPE_MOSTLY_Q4_K },
        { 1024, 4096, 16, 20, LLAMA_FTYPE_MOSTLY_Q4_K },
        {  512, 8192, 16, 10, LLAMA_FTYPE_MOSTLY_Q6_K },
        { 1024, 8192,  8, 10, LLAMA_FTYPE_MOSTLY_Q6_K },
    };

    for (auto & cfg : cases) {
        benchmark_case(cfg, false);
        benchmark_case(cfg, true);
    }
    return 0;
}
