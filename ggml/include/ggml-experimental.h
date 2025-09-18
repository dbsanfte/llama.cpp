#pragma once
// Experimental runtime toggles for ggml - unstable API (subject to removal / change)
#ifdef __cplusplus
extern "C" {
#endif

void ggml_experimental_set_prefetch_k(int enabled);
int  ggml_experimental_get_prefetch_k(void);

#ifdef __cplusplus
}
#endif
