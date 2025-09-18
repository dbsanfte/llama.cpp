#include "ggml.h"
#include "ggml-experimental.h"

#ifdef GGML_EXPERIMENT_PREFETCH_K
static int g_prefetch_k = 0;
#else
static int g_prefetch_k = 0; // always 0 when feature not compiled
#endif

extern "C" void ggml_experimental_set_prefetch_k(int enabled) {
#ifdef GGML_EXPERIMENT_PREFETCH_K
    g_prefetch_k = enabled ? 1 : 0;
#else
    (void)enabled;
#endif
}

extern "C" int ggml_experimental_get_prefetch_k(void) {
    return g_prefetch_k;
}
