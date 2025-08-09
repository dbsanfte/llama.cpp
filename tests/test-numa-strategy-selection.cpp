/**
 * NUMA Coordinator Strategy Selection Test
 * 
 * This test demonstrates the adaptive strategy selection functionality
 * and allows testing of different memory management strategies.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

#include "ggml.h"
#include "../ggml/src/ggml-cpu/ggml-numa-coordinator.h"

void print_strategy_name(enum ggml_numa_memory_strategy strategy) {
    switch (strategy) {
        case GGML_NUMA_STRATEGY_AUTO:
            std::cout << "AUTO (adaptive selection)";
            break;
        case GGML_NUMA_STRATEGY_MATRIX_REDUCTION:
            std::cout << "MATRIX_REDUCTION (memory efficient)";
            break;
        case GGML_NUMA_STRATEGY_CHUNKED_PROCESSING:
            std::cout << "CHUNKED_PROCESSING (high throughput)";
            break;
        case GGML_NUMA_STRATEGY_HYBRID:
            std::cout << "HYBRID (dynamic switching)";
            break;
        default:
            std::cout << "UNKNOWN";
            break;
    }
}

void test_strategy_selection_logic() {
    std::cout << "=== Strategy Selection Logic Test ===\n\n";
    
    std::vector<struct ggml_numa_workload_info> test_workloads = {
        // Small matrices - should prefer chunked processing
        {512, 32, 32, false, GGML_NUMA_STRATEGY_AUTO},
        {256, 16, 16, false, GGML_NUMA_STRATEGY_AUTO},
        
        // Large matrices - should prefer matrix reduction
        {1024, 64, 32, false, GGML_NUMA_STRATEGY_AUTO},
        {2048, 128, 32, false, GGML_NUMA_STRATEGY_AUTO},
        
        // Memory-constrained environment
        {512, 32, 8, false, GGML_NUMA_STRATEGY_AUTO}, // 8GB memory
        
        // Scaling accuracy priority
        {512, 32, 32, true, GGML_NUMA_STRATEGY_AUTO},
        {1024, 64, 32, true, GGML_NUMA_STRATEGY_AUTO},
        
        // User overrides
        {512, 32, 32, false, GGML_NUMA_STRATEGY_MATRIX_REDUCTION},
        {1024, 64, 32, false, GGML_NUMA_STRATEGY_CHUNKED_PROCESSING},
    };
    
    std::cout << std::left << std::setw(12) << "Matrix Dim" 
              << std::setw(10) << "Batch" 
              << std::setw(8) << "Memory" 
              << std::setw(10) << "Priority" 
              << std::setw(20) << "Override" 
              << std::setw(30) << "Chosen Strategy" << std::endl;
    
    std::cout << std::string(100, '-') << std::endl;
    
    for (const auto& workload : test_workloads) {
        enum ggml_numa_memory_strategy chosen = ggml_numa_choose_strategy(&workload);
        
        std::cout << std::left << std::setw(12) << workload.matrix_dim
                  << std::setw(10) << workload.batch_size
                  << std::setw(8) << (std::to_string(workload.available_memory_gb) + "GB")
                  << std::setw(10) << (workload.prioritize_scaling_accuracy ? "Scaling" : "Throughput")
                  << std::setw(20);
        
        if (workload.user_override == GGML_NUMA_STRATEGY_AUTO) {
            std::cout << "None";
        } else {
            print_strategy_name(workload.user_override);
        }
        
        std::cout << std::setw(30);
        print_strategy_name(chosen);
        std::cout << std::endl;
    }
    
    std::cout << std::endl;
}

void test_coordinator_strategy_api() {
    std::cout << "=== Coordinator Strategy API Test ===\n\n";
    
    // Create a coordinator manager for testing
    struct ggml_numa_coordinator_manager * mgr = ggml_numa_coordinator_manager_get_global(4, false);
    if (!mgr) {
        std::cout << "❌ Failed to create coordinator manager\n";
        return;
    }
    
    // Test getting default strategy
    enum ggml_numa_memory_strategy default_strategy = ggml_numa_coordinator_manager_get_strategy(mgr);
    std::cout << "Default strategy: ";
    print_strategy_name(default_strategy);
    std::cout << std::endl;
    
    // Test setting different strategies
    std::vector<enum ggml_numa_memory_strategy> test_strategies = {
        GGML_NUMA_STRATEGY_MATRIX_REDUCTION,
        GGML_NUMA_STRATEGY_CHUNKED_PROCESSING,
        GGML_NUMA_STRATEGY_HYBRID,
        GGML_NUMA_STRATEGY_AUTO
    };
    
    for (auto strategy : test_strategies) {
        int result = ggml_numa_coordinator_manager_set_strategy(mgr, strategy);
        if (result == 0) {
            enum ggml_numa_memory_strategy current = ggml_numa_coordinator_manager_get_strategy(mgr);
            std::cout << "✅ Set strategy to: ";
            print_strategy_name(current);
            std::cout << std::endl;
        } else {
            std::cout << "❌ Failed to set strategy\n";
        }
    }
    
    std::cout << std::endl;
}

void test_adaptive_work_submission() {
    std::cout << "=== Adaptive Work Submission Test ===\n\n";
    
    struct ggml_numa_coordinator_manager * mgr = ggml_numa_coordinator_manager_get_global(4, false);
    if (!mgr) {
        std::cout << "❌ Failed to create coordinator manager\n";
        return;
    }
    
    // Initialize GGML context for tensor creation
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024 * 256, // 256MB
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::cout << "❌ Failed to initialize GGML context\n";
        return;
    }
    
    // Create test tensor
    struct ggml_tensor * test_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 512);
    
    // Test different workload scenarios
    std::vector<struct ggml_numa_workload_info> scenarios = {
        {512, 32, 32, false, GGML_NUMA_STRATEGY_AUTO},  // Small matrix, auto selection
        {1024, 64, 16, false, GGML_NUMA_STRATEGY_AUTO}, // Large matrix, memory constrained
        {768, 128, 32, true, GGML_NUMA_STRATEGY_AUTO},  // Medium matrix, scaling priority
    };
    
    for (size_t i = 0; i < scenarios.size(); i++) {
        std::cout << "Scenario " << (i + 1) << ": Matrix " << scenarios[i].matrix_dim 
                  << "x" << scenarios[i].matrix_dim << ", Batch " << scenarios[i].batch_size
                  << ", Memory " << scenarios[i].available_memory_gb << "GB\n";
        
        enum ggml_numa_memory_strategy chosen = ggml_numa_choose_strategy(&scenarios[i]);
        std::cout << "  Chosen strategy: ";
        print_strategy_name(chosen);
        std::cout << std::endl;
        
        // Test adaptive work submission
        int work_id = ggml_numa_coordinator_manager_submit_adaptive_work(mgr, test_tensor, &scenarios[i]);
        if (work_id >= 0) {
            std::cout << "  ✅ Successfully submitted adaptive work (ID: " << work_id << ")\n";
        } else {
            std::cout << "  ❌ Failed to submit adaptive work\n";
        }
        std::cout << std::endl;
    }
    
    // Cleanup
    ggml_free(ctx);
    
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "🧠 NUMA Coordinator Strategy Selection Test\n";
    std::cout << "==========================================\n\n";
    
    bool run_all = (argc == 1);
    bool run_logic = run_all || (argc > 1 && std::string(argv[1]) == "--logic");
    bool run_api = run_all || (argc > 1 && std::string(argv[1]) == "--api");
    bool run_adaptive = run_all || (argc > 1 && std::string(argv[1]) == "--adaptive");
    
    if (argc > 1 && std::string(argv[1]) == "--help") {
        std::cout << "Usage: " << argv[0] << " [--logic|--api|--adaptive|--help]\n";
        std::cout << "  --logic     Test strategy selection logic only\n";
        std::cout << "  --api       Test coordinator strategy API only\n";
        std::cout << "  --adaptive  Test adaptive work submission only\n";
        std::cout << "  --help      Show this help\n";
        std::cout << "  (no args)   Run all tests\n";
        return 0;
    }
    
    if (run_logic) {
        test_strategy_selection_logic();
    }
    
    if (run_api) {
        test_coordinator_strategy_api();
    }
    
    if (run_adaptive) {
        test_adaptive_work_submission();
    }
    
    std::cout << "🎯 Strategy Selection Test Complete!\n";
    return 0;
}
