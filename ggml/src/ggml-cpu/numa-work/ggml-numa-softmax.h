#pragma once

#include "../ggml-numa-operation-dispatch.h"
#include "../ggml-numa-work-shared.h"
#include "../ggml-numa-work-shared.h"
#include "../../include/ggml.h"

// SOFT_MAX operation NUMA work functions
enum ggml_status ggml_numa_work_function_soft_max(void * work_context, struct ggml_compute_params * params);
enum ggml_status ggml_numa_work_function_soft_max_chunk(void * work_context, struct ggml_compute_params * params);
