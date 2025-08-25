#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-numa-simple-coordinator.h"
#include "upi-traffic-monitor.h"
#include <vector>
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cstring>

#ifdef __linux__
#include <numa.h>
#include <sched.h>
#endif

// Supported NUMA operation types for testing
enum class NumaOperationType {
    ADD                 // Element-wise addition (a + b) - IMPLEMENTED
};

// Supported NUMA strategies for testing
enum class TestNumaStrategy {
    ISOLATE_NODE_0,     // Execute only on NUMA node 0
    ISOLATE_NODE_1,     // Execute only on NUMA node 1  
    MIRROR              // Execute across all NUMA nodes
};

// Supported tensor sizes for testing
enum class TensorSize {
    SMALL,              // Small tensor (1 MB)
    LARGE,              // Large tensor (32 MB)
    HUGE,               // Huge tensor (1 GB)
    GIGANTIC_1GB,       // Gigantic tensor (1 GB)
    GIGANTIC_2GB,       // Gigantic tensor (2 GB) 
    GIGANTIC_4GB,       // Gigantic tensor (4 GB)
    GIGANTIC_8GB,       // Gigantic tensor (8 GB)
    GIGANTIC_16GB       // Gigantic tensor (16 GB)
};

// Single test configuration
struct TestConfig {
    NumaOperationType operation;
    TestNumaStrategy strategy;
    TensorSize size;
    int dim1, dim2, dim3;
    std::string name;
    std::string description;
};

// Operation setup for tensor creation
struct OperationSetup {
    ggml_tensor* tensor_a;
    ggml_tensor* tensor_b;
    ggml_tensor* result;
    std::string operation_name;
};

// Forward declarations
void verify_numa_setup(TestNumaStrategy strategy);

// Helper functions to convert enums to strings
std::string operation_type_to_string(NumaOperationType operation) {
    switch (operation) {
        case NumaOperationType::ADD: return "ADD";
        default: return "UNKNOWN";
    }
}

std::string strategy_to_string(TestNumaStrategy strategy) {
    switch (strategy) {
        case TestNumaStrategy::ISOLATE_NODE_0: return "ISOLATE_NODE_0";
        case TestNumaStrategy::ISOLATE_NODE_1: return "ISOLATE_NODE_1";
        case TestNumaStrategy::MIRROR: return "MIRROR";
        default: return "UNKNOWN";
    }
}

std::string size_to_string(TensorSize size) {
    switch (size) {
        case TensorSize::SMALL: return "SMALL";
        case TensorSize::LARGE: return "LARGE";
        case TensorSize::HUGE: return "HUGE";
        case TensorSize::GIGANTIC_1GB: return "GIGANTIC_1GB";
        case TensorSize::GIGANTIC_2GB: return "GIGANTIC_2GB";
        case TensorSize::GIGANTIC_4GB: return "GIGANTIC_4GB";
        case TensorSize::GIGANTIC_8GB: return "GIGANTIC_8GB";
        case TensorSize::GIGANTIC_16GB: return "GIGANTIC_16GB";
        default: return "UNKNOWN";
    }
}

// Parse command line arguments
TestConfig parse_arguments(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Usage: %s <OPERATION> <STRATEGY> <SIZE>\n", argv[0]);
        printf("  OPERATION: ADD\n");
        printf("  STRATEGY: ISOLATE_NODE_0, ISOLATE_NODE_1, MIRROR\n");
        printf("  SIZE: SMALL, LARGE, HUGE, GIGANTIC_1GB, GIGANTIC_2GB, GIGANTIC_4GB, GIGANTIC_8GB, GIGANTIC_16GB\n");
        exit(1);
    }
    
    TestConfig config = {};
    
    // Parse operation
    if (strcmp(argv[1], "ADD") == 0) {
        config.operation = NumaOperationType::ADD;
    } else {
        printf("❌ Unknown operation: %s\n", argv[1]);
        exit(1);
    }
    
    // Parse strategy
    if (strcmp(argv[2], "ISOLATE_NODE_0") == 0) {
        config.strategy = TestNumaStrategy::ISOLATE_NODE_0;
    } else if (strcmp(argv[2], "ISOLATE_NODE_1") == 0) {
        config.strategy = TestNumaStrategy::ISOLATE_NODE_1;
    } else if (strcmp(argv[2], "MIRROR") == 0) {
        config.strategy = TestNumaStrategy::MIRROR;
    } else {
        printf("❌ Unknown strategy: %s\n", argv[2]);
        exit(1);
    }
    
    // Parse size and set dimensions
    if (strcmp(argv[3], "SMALL") == 0) {
        config.size = TensorSize::SMALL;
        config.dim1 = 128; config.dim2 = 128; config.dim3 = 16;   // ~1 MB
        config.name = "ADD_SMALL";
        config.description = "Small tensor (1 MB)";
    } else if (strcmp(argv[3], "LARGE") == 0) {
        config.size = TensorSize::LARGE;
        config.dim1 = 256; config.dim2 = 256; config.dim3 = 64;   // ~32 MB
        config.name = "ADD_LARGE";  
        config.description = "Large tensor (32 MB)";
    } else if (strcmp(argv[3], "HUGE") == 0) {
        config.size = TensorSize::HUGE;
        config.dim1 = 512; config.dim2 = 512; config.dim3 = 256;  // ~1 GB
        config.name = "ADD_HUGE";
        config.description = "Huge tensor (1 GB)";
    } else if (strcmp(argv[3], "GIGANTIC_1GB") == 0) {
        config.size = TensorSize::GIGANTIC_1GB;
        config.dim1 = 645; config.dim2 = 645; config.dim3 = 645;  // ~1 GB (268M elements)
        config.name = "ADD_GIGANTIC_1GB";
        config.description = "Gigantic tensor (1 GB)";
    } else if (strcmp(argv[3], "GIGANTIC_2GB") == 0) {
        config.size = TensorSize::GIGANTIC_2GB;
        config.dim1 = 813; config.dim2 = 813; config.dim3 = 813;  // ~2 GB (537M elements)
        config.name = "ADD_GIGANTIC_2GB";
        config.description = "Gigantic tensor (2 GB)";
    } else if (strcmp(argv[3], "GIGANTIC_4GB") == 0) {
        config.size = TensorSize::GIGANTIC_4GB;
        config.dim1 = 1024; config.dim2 = 1024; config.dim3 = 1024;  // ~4 GB (1073M elements)
        config.name = "ADD_GIGANTIC_4GB";
        config.description = "Gigantic tensor (4 GB)";
    } else if (strcmp(argv[3], "GIGANTIC_8GB") == 0) {
        config.size = TensorSize::GIGANTIC_8GB;
        config.dim1 = 1290; config.dim2 = 1290; config.dim3 = 1290;  // ~8 GB (2146M elements)
        config.name = "ADD_GIGANTIC_8GB";
        config.description = "Gigantic tensor (8 GB)";
    } else if (strcmp(argv[3], "GIGANTIC_16GB") == 0) {
        config.size = TensorSize::GIGANTIC_16GB;
        config.dim1 = 1625; config.dim2 = 1625; config.dim3 = 1625;  // ~16 GB (4291M elements)
        config.name = "ADD_GIGANTIC_16GB";
        config.description = "Gigantic tensor (16 GB)";
    } else {
        printf("❌ Unknown size: %s\n", argv[3]);
        exit(1);
    }
    
    return config;
}

// Setup NUMA strategy and CPU affinity
void setup_numa_strategy(TestNumaStrategy strategy) {
    switch (strategy) {
        case TestNumaStrategy::ISOLATE_NODE_0:
            printf("🔧 Setting up ISOLATE strategy for NUMA node 0\n");
            
            // Set CPU affinity to node 0
#ifdef __linux__
            {
                cpu_set_t mask;
                CPU_ZERO(&mask);
                // Bind to physical cores of NUMA node 0 (cores 0-27)
                for (int i = 0; i < 28; i++) {
                    CPU_SET(i, &mask);
                }
                if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {
                    printf("❌ Failed to set CPU affinity to NUMA node 0\n");
                    exit(1);
                }
                printf("✅ CPU affinity set to NUMA node 0 (cores 0-27)\n");
            }
#endif
            
            // Initialize NUMA coordinator with ISOLATE strategy
            ggml_numa_init_with_node(GGML_NUMA_STRATEGY_ISOLATE, 0);
            setenv("GGML_NUMA_NODES", "1", 1);
            printf("✅ NUMA coordinator initialized: ISOLATE node 0\n");
            break;
            
        case TestNumaStrategy::ISOLATE_NODE_1:
            printf("🔧 Setting up ISOLATE strategy for NUMA node 1\n");
            
            // Set CPU affinity to node 1
#ifdef __linux__
            {
                cpu_set_t mask;
                CPU_ZERO(&mask);
                // Bind to physical cores of NUMA node 1 (cores 28-55)
                for (int i = 28; i < 56; i++) {
                    CPU_SET(i, &mask);
                }
                if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {
                    printf("❌ Failed to set CPU affinity to NUMA node 1\n");
                    exit(1);
                }
                printf("✅ CPU affinity set to NUMA node 1 (cores 28-55)\n");
            }
#endif
            
            // Initialize NUMA coordinator with ISOLATE strategy
            ggml_numa_init_with_node(GGML_NUMA_STRATEGY_ISOLATE, 1);
            setenv("GGML_NUMA_NODES", "1", 1);
            printf("✅ NUMA coordinator initialized: ISOLATE node 1\n");
            break;
            
        case TestNumaStrategy::MIRROR:
            printf("🔧 Setting up MIRROR strategy for all NUMA nodes\n");
            
            // Clear CPU affinity (allow all cores)
#ifdef __linux__
            {
                cpu_set_t mask;
                CPU_ZERO(&mask);
                for (int i = 0; i < 112; i++) {  // All logical cores
                    CPU_SET(i, &mask);
                }
                if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {
                    printf("❌ Failed to clear CPU affinity\n");
                    exit(1);
                }
                printf("✅ CPU affinity cleared (all cores available)\n");
            }
#endif
            
            // Initialize NUMA coordinator with MIRROR strategy
            ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);
            setenv("GGML_NUMA_NODES", "2", 1);
            printf("✅ NUMA coordinator initialized: MIRROR across 2 nodes\n");
            break;
    }
    
    // Verify the setup
    verify_numa_setup(strategy);
}

// Verify NUMA setup is correct
void verify_numa_setup(TestNumaStrategy strategy) {
    int actual_strategy = ggml_numa_get_strategy();
    int num_nodes = ggml_numa_simple_coordinator_get_num_nodes();
    
    printf("🔍 Verifying NUMA setup...\n");
    printf("   Strategy: %d\n", actual_strategy);
    printf("   Coordinator nodes: %d\n", num_nodes);
    
    switch (strategy) {
        case TestNumaStrategy::ISOLATE_NODE_0:
        case TestNumaStrategy::ISOLATE_NODE_1:
            if (actual_strategy != GGML_NUMA_STRATEGY_ISOLATE) {
                printf("❌ Expected ISOLATE strategy (%d), got %d\n", GGML_NUMA_STRATEGY_ISOLATE, actual_strategy);
                exit(1);
            }
            if (num_nodes != 1) {
                printf("❌ Expected 1 node for ISOLATE, got %d\n", num_nodes);
                exit(1);
            }
            break;
            
        case TestNumaStrategy::MIRROR:
            if (actual_strategy != GGML_NUMA_STRATEGY_MIRROR) {
                printf("❌ Expected MIRROR strategy (%d), got %d\n", GGML_NUMA_STRATEGY_MIRROR, actual_strategy);
                exit(1);
            }
            if (num_nodes != 2) {
                printf("❌ Expected 2 nodes for MIRROR, got %d\n", num_nodes);
                exit(1);
            }
            break;
    }
    
    printf("✅ NUMA setup verification passed\n");
}

// Create operation based on config type
OperationSetup create_operation(struct ggml_context* ctx, const TestConfig& config) {
    OperationSetup setup = {nullptr, nullptr, nullptr, ""};
    
    switch (config.operation) {
        case NumaOperationType::ADD: {
            setup.tensor_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, config.dim1, config.dim2, config.dim3);
            setup.tensor_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, config.dim1, config.dim2, config.dim3);
            setup.result = ggml_add(ctx, setup.tensor_a, setup.tensor_b);
            setup.operation_name = "ADD";
            break;
        }
        default:
            printf("❌ Unsupported operation type\n");
            break;
    }
    
    return setup;
}

// Initialize operation data
void initialize_operation_data(const OperationSetup& setup, const TestConfig& config) {
    switch (config.operation) {
        case NumaOperationType::ADD: {
            float* a_data = (float*)ggml_get_data(setup.tensor_a);
            float* b_data = (float*)ggml_get_data(setup.tensor_b);
            size_t total_elements = ggml_nelements(setup.tensor_a);
            
            for (size_t i = 0; i < total_elements; i++) {
                a_data[i] = 1.5f + (i % 100) * 0.01f;
                b_data[i] = 2.5f + (i % 100) * 0.01f;
            }
            break;
        }
        default:
            printf("❌ Unsupported operation type for data initialization\n");
            break;
    }
}

// Run the actual test
double run_test(const TestConfig& config) {
    size_t tensor_size = (size_t)config.dim1 * config.dim2 * config.dim3 * sizeof(float);
    size_t ctx_size = tensor_size * 3 + 1024*1024*1024;  // 3 tensors + 1GB overhead
    
    printf("\n🔹 Testing %s with %s strategy\n", config.name.c_str(), strategy_to_string(config.strategy).c_str());
    printf("   Tensor dimensions: %dx%dx%d (%.1f MB)\n", 
           config.dim1, config.dim2, config.dim3, tensor_size / (1024.0 * 1024.0));
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        printf("❌ Failed to create context\n");
        return -1.0;
    }
    
    // Create operation based on config type
    OperationSetup op = create_operation(ctx, config);
    
    if (!op.tensor_a || !op.result || (config.operation == NumaOperationType::ADD && !op.tensor_b)) {
        printf("❌ Failed to create tensors for %s operation\n", op.operation_name.c_str());
        ggml_free(ctx);
        return -1.0;
    }
    
    // Initialize data based on operation type
    initialize_operation_data(op, config);
    
    struct ggml_cgraph* cgraph = ggml_new_graph(ctx);
    ggml_build_forward_expand(cgraph, op.result);
    
    int num_cores = (config.strategy == TestNumaStrategy::MIRROR) ? 112 : 56;
    printf("   Using %d cores for %s operation\n", num_cores, op.operation_name.c_str());
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        enum ggml_status warmup_status = ggml_graph_compute_with_ctx(ctx, cgraph, num_cores);
        if (warmup_status != GGML_STATUS_SUCCESS) {
            printf("❌ Warmup %d failed\n", i+1);
        }
    }
    
    const int num_runs = 10;
    std::vector<double> times;
    
    for (int run = 0; run < num_runs; run++) {
        auto start = std::chrono::high_resolution_clock::now();
        enum ggml_status status = ggml_graph_compute_with_ctx(ctx, cgraph, num_cores);
        auto end = std::chrono::high_resolution_clock::now();
        
        if (status != GGML_STATUS_SUCCESS) {
            printf("❌ Computation failed on run %d\n", run);
            ggml_free(ctx);
            return -1.0;
        }
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        if (time_ms > 0.001) {
            times.push_back(time_ms);
        }
    }
    
    ggml_free(ctx);
    
    if (times.empty()) {
        printf("❌ No valid timing measurements\n");
        return -1.0;
    }
    
    // Remove outliers
    std::sort(times.begin(), times.end());
    if (times.size() > 4) {
        times.erase(times.begin());
        times.pop_back();
    }
    
    double avg_time = 0.0;
    for (double t : times) {
        avg_time += t;
    }
    avg_time /= times.size();
    
    printf("   ✅ Average time: %.3f ms\n", avg_time);
    return avg_time;
}

// Main function
int main(int argc, char* argv[]) {
    printf("🚀 NUMA Execution Modes Test (Single Configuration)\n");
    printf("=================================================\n");
    
    // Parse command line arguments
    TestConfig config = parse_arguments(argc, argv);
    
    printf("Test Configuration:\n");
    printf("  Operation: %s\n", operation_type_to_string(config.operation).c_str());
    printf("  Strategy: %s\n", strategy_to_string(config.strategy).c_str());
    printf("  Size: %s\n", size_to_string(config.size).c_str());
    printf("  Description: %s\n", config.description.c_str());
    
    // Setup NUMA strategy
    setup_numa_strategy(config.strategy);
    
    // Run the test
    double execution_time = run_test(config);
    
    if (execution_time < 0) {
        printf("❌ Test failed\n");
        return 1;
    }
    
    // Output machine-readable result for the orchestrator script
    printf("\n📊 RESULT: %s,%s,%s,%.3f\n", 
           operation_type_to_string(config.operation).c_str(),
           strategy_to_string(config.strategy).c_str(),
           size_to_string(config.size).c_str(),
           execution_time);
    
    printf("✅ Test completed successfully\n");
    return 0;
}
