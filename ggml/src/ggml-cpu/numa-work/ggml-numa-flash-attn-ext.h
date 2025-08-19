#pragma once

#include "../ggml-numa-operation-dispatch.h"
#include "../ggml-numa-work-shared.h"
#include "../../include/ggml.h"

// FLASH_ATTN_EXT operation NUMA work functions
enum ggml_status ggml_numa_work_function_flash_attn_ext_chunk(void * work_context, struct ggml_compute_params * params);
