#pragma once

#include "../ggml-numa-operation-dispatch.h"
#include "../ggml-numa-work-shared.h"
#include "../../include/ggml.h"

// ADD operation NUMA work functions
enum ggml_status ggml_numa_work_function_add_single(void * work_context, struct ggml_compute_params * params);
enum ggml_status ggml_numa_work_function_add_chunk(void * work_context, struct ggml_compute_params * params);
