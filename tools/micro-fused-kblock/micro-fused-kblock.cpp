// micro-fused-kblock.cpp (revised for current ggml backend API)
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <chrono>

static ggml_type pick_type() { return GGML_TYPE_Q6_K; }

int main() {
    int M = 128, N = 128, K = 4096; // tunable via env MFKB_SHAPE=MxNxK
    if (const char * env = std::getenv("MFKB_SHAPE")) {
        int m,n,k; if (std::sscanf(env, "%dx%dx%d", &m,&n,&k) == 3) { M=m; N=n; K=k; }
    }


    // Use no_alloc = true so that backend allocator will own tensor buffers
    struct ggml_init_params params = { 32*1024*1024, NULL, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) { fprintf(stderr, "init failed\n"); return 1; }

    ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M); // shape (K,M)
    ggml_tensor * B = ggml_new_tensor_2d(ctx, pick_type(), K, N);   // quant weights (K,N)
    // Orientation for fused path: activations (F32) first, quant weights second
    ggml_tensor * C = ggml_mul_mat(ctx, A, B);
    ggml_set_name(C, "C");

    // Allocate via backend
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) { fprintf(stderr, "backend init failed\n"); return 1; }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { fprintf(stderr, "alloc tensors failed\n"); return 1; }

    // Fill A data
    float * a_ptr = ggml_get_data_f32(A);
    for (int i=0;i<M;i++) for (int j=0;j<K;j++) a_ptr[i*K + j] = (float)((i + j) % 17) - 8.0f;
    // Zero B underlying storage (synthetic). This does not reflect real Q6_K packing but exercises dequant path.
    std::memset(ggml_get_data(B), 0, ggml_nbytes(B));

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(gf, C);

    auto t0 = std::chrono::high_resolution_clock::now();
    ggml_status st = ggml_backend_graph_compute(backend, gf);
    auto t1 = std::chrono::high_resolution_clock::now();
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed: %d\n", (int)st);
        return 1;
    }
    double ms = std::chrono::duration<double,std::milli>(t1 - t0).count();
    printf("micro-fused-kblock: M=%d N=%d K=%d time=%.2f ms (A=f32 B=%s)\n", M, N, K, ms, ggml_type_name(B->type));

    // Simple checksum
    float * cdata = ggml_get_data_f32(C);
    double sum = 0.0; int limit = (N*M < 64 ? N*M : 64);
    for (int i=0;i<limit;i++) sum += cdata[i];
    fprintf(stderr, "partial-sum=%f\n", sum);

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
