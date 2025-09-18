// -----------------------------------------------------------------------------
// test-prefetch-k.cpp
// Micro-benchmark for experimental K-series software prefetch in generic vec_dot
//
// Build:
//   cmake -S . -B build -DGGML_EXPERIMENT_PREFETCH_K=ON
//   cmake --build build --target test-prefetch-k -j
// Run:
//   ./build/bin/test-prefetch-k
// Output:
//   For each case prints baseline (prefetch disabled) vs experimental timings,
//   speedup percentage, and numerical validation stats (mismatches, max_abs, mse).
// Notes:
//   1. This uses raw float data copied into quant tensor buffers (no real quantize step) –
//      goal is relative performance delta only.
//   2. Prefetch is compile-time gated by GGML_EXPERIMENT_PREFETCH_K and runtime toggled via
//      ggml_experimental_set_prefetch_k().
//   3. If GGML_EXPERIMENT_PREFETCH_K was OFF at build, the runtime toggle is inert.
// -----------------------------------------------------------------------------
#include "llama.h"
#include "ggml-experimental.h"
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <algorithm>

static double wall_time_s() { using clock = std::chrono::steady_clock; return std::chrono::duration<double>(clock::now().time_since_epoch()).count(); }

struct bench_case { int M; int K; int N; int iters; ggml_type wtype; const char * label; };

static void fill(std::vector<float> & v) { for (auto & x : v) x = (float)((rand()%2000)-1000)/200.0f; }

static void run_case(const bench_case & bc) {
    printf("\n[case %s] M=%d K=%d N=%d iters=%d type=%d\n", bc.label, bc.M, bc.K, bc.N, bc.iters, (int)bc.wtype);
    struct ggml_init_params ip = { 0 };
    size_t mem_sz = 64ull*1024ull*1024ull; // 64 MB scratch
    static std::vector<uint8_t> arena; arena.resize(mem_sz);
    ip.mem_size = mem_sz; ip.mem_buffer = arena.data();
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { fprintf(stderr, "ctx init failed\n"); return; }

    ggml_tensor * A = ggml_new_tensor_2d(ctx, bc.wtype, bc.K, bc.M); // [K,M]
    ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, bc.K, bc.N); // [K,N]
    ggml_tensor * C = ggml_mul_mat(ctx, A, B); // [M,N]

    std::vector<float> hA((size_t)bc.K * bc.M);
    std::vector<float> hB((size_t)bc.K * bc.N);
    fill(hA); fill(hB);
    // store raw floats into quant tensor buffer (not true quantization; relative comparison only)
    memcpy(tensor_data(A), hA.data(), std::min((size_t)ggml_nbytes(A), hA.size()*sizeof(float)));
    memcpy(tensor_data(B), hB.data(), hB.size()*sizeof(float));

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(gf, C);

    // run baseline (prefetch off)
    ggml_experimental_set_prefetch_k(0);
    std::vector<float> baseline((size_t)bc.M * bc.N);
    // warmup
    {
    struct ggml_cplan cplan = ggml_graph_plan(gf, 1, NULL);
        std::vector<uint8_t> work(cplan.work_size); cplan.work_data = work.data();
        ggml_graph_compute(gf, &cplan);
    }
    double t0 = wall_time_s();
    for (int it=0; it<bc.iters; ++it) {
    struct ggml_cplan cplan = ggml_graph_plan(gf, 1, NULL);
        std::vector<uint8_t> work(cplan.work_size); cplan.work_data = work.data();
        ggml_graph_compute(gf, &cplan);
    }
    double t1 = wall_time_s();
    memcpy(baseline.data(), tensor_data(C), baseline.size()*sizeof(float));
    double base_avg = (t1 - t0)/bc.iters;

    // run experimental (prefetch on)
    ggml_experimental_set_prefetch_k(1);
    // small re-warm since flag toggled
    {
    struct ggml_cplan cplan = ggml_graph_plan(gf, 1, NULL);
        std::vector<uint8_t> work(cplan.work_size); cplan.work_data = work.data();
        ggml_graph_compute(gf, &cplan);
    }
    double e0 = wall_time_s();
    for (int it=0; it<bc.iters; ++it) {
    struct ggml_cplan cplan = ggml_graph_plan(gf, 1, NULL);
        std::vector<uint8_t> work(cplan.work_size); cplan.work_data = work.data();
        ggml_graph_compute(gf, &cplan);
    }
    double e1 = wall_time_s();
    double exp_avg = (e1 - e0)/bc.iters;

    size_t elems = (size_t)bc.M * bc.N;
    float * out_ptr = (float*)tensor_data(C);
    size_t mismatches = 0; double max_abs=0.0; double mse=0.0;
    for (size_t i=0;i<elems;++i){ double d=fabs(out_ptr[i]-baseline[i]); if(d!=0.0){ if(++mismatches<5) printf("diff idx %zu base=%f exp=%f d=%f\n", i, baseline[i], out_ptr[i], d); max_abs = d>max_abs?d:max_abs; } mse += d*d; }
    mse /= elems;
    printf("baseline=%.6f s  experimental=%.6f s  speedup=%.2f%%  mismatches=%zu max_abs=%g mse=%g\n", base_avg, exp_avg, (base_avg/exp_avg - 1.0)*100.0, mismatches, max_abs, mse);

    ggml_free(ctx);
}

int main(){
    srand(42);
    std::vector<bench_case> cases = {
        {512, 4096, 16, 10, GGML_TYPE_Q4_K, "q4k_small"},
        {1024, 4096, 16, 10, GGML_TYPE_Q4_K, "q4k_med"},
        {512, 8192, 16, 6, GGML_TYPE_Q6_K, "q6k_largeK_smallM"},
        {1024, 8192, 8, 6, GGML_TYPE_Q6_K, "q6k_large"}
    };
    for (auto & bc : cases) run_case(bc);
    return 0;
}
