// Experimental runtime toggles for ggml - not part of stable API.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Enable (1) or disable (0) K-series software prefetch if built with GGML_EXPERIMENT_PREFETCH_K.
// If library not built with that option the setter is a no-op and getter returns 0.
void ggml_experimental_set_prefetch_k(int enabled);
int  ggml_experimental_get_prefetch_k(void);

#ifdef __cplusplus
}
#endif
