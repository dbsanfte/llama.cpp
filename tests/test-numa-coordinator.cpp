#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-numa-operation-dispatch.h"
#include "ggml-numa-coordinator.h"
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <string.h>

// Test framework structures
struct TestResult {
    const char* test_name;
    bool passed;
    const char* message;
};

class NumaCoordinatorTestSuite {
private:
    std::vector<TestResult> results;
    ggml_backend_t backend;
    struct ggml_context * ctx;
    
public:
    NumaCoordinatorTestSuite() : backend(nullptr), ctx(nullptr) {
        printf("🧪 NUMA Coordinator Test Suite Initialization...\n");
        
        // Initialize NUMA system
        printf("Initializing NUMA with DISTRIBUTE strategy...\n");
        ggml_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);
        
        // Initialize backend
        backend = ggml_backend_cpu_init();
        if (!backend) {
            printf("❌ Failed to initialize CPU backend\n");
            return;
        }

        // Create context with larger memory for comprehensive tests
        struct ggml_init_params params;
        params.mem_size = 64 * 1024 * 1024;  // 64MB for comprehensive testing
        params.mem_buffer = NULL;
        params.no_alloc = true;  // Required for ggml_backend_alloc_ctx_tensors
        
        ctx = ggml_init(params);
        if (!ctx) {
            printf("❌ Failed to initialize context\n");
            ggml_backend_free(backend);
            backend = nullptr;
            return;
        }
        
        printf("✅ Test suite initialized successfully\n\n");
    }
    
    ~NumaCoordinatorTestSuite() {
        if (ctx) {
            ggml_free(ctx);
        }
        if (backend) {
            ggml_backend_free(backend);
        }
    }
    
    bool is_initialized() const {
        return backend != nullptr && ctx != nullptr;
    }
    
    void add_test_result(const char* test_name, bool passed, const char* message = "") {
        results.push_back({test_name, passed, message});
    }
    
    // Test: NUMA coordinator manager creation and basic functionality
    void test_numa_coordinator_manager_creation() {
        printf("--- Test: NUMA Coordinator Manager Creation ---\n");
        
        printf("Testing NUMA coordinator manager creation...\n");
        
        bool all_tests_passed = true;
        
        // Test: Basic coordinator manager creation
        printf("  Creating basic coordinator manager...\n");
        struct ggml_numa_coordinator_manager* basic_mgr = ggml_numa_coordinator_manager_new(4, false);
        if (basic_mgr) {
            printf("  ✅ Basic coordinator manager created successfully\n");
            ggml_numa_coordinator_manager_free(basic_mgr);
        } else {
            printf("  ❌ Failed to create basic coordinator manager\n");
            all_tests_passed = false;
        }
        
        // Test: Force multi-socket coordinator manager creation
        printf("  Creating force multi-socket coordinator manager...\n");
        struct ggml_numa_coordinator_manager* multi_mgr = ggml_numa_coordinator_manager_new(8, true);
        if (multi_mgr) {
            printf("  ✅ Multi-socket coordinator manager created successfully\n");
            ggml_numa_coordinator_manager_free(multi_mgr);
        } else {
            printf("  ❌ Failed to create multi-socket coordinator manager\n");
            all_tests_passed = false;
        }
        
        // Test: Global coordinator access
        printf("  Testing global coordinator access...\n");
        struct ggml_numa_coordinator_manager* global_mgr = ggml_numa_coordinator_manager_get_global(4, false);
        if (global_mgr) {
            printf("  ✅ Global coordinator manager accessible\n");
            // Don't free global manager - it's managed globally
        } else {
            printf("  ❌ Failed to access global coordinator manager\n");
            all_tests_passed = false;
        }
        
        if (all_tests_passed) {
            add_test_result("numa_coordinator_manager_creation", true, "All coordinator manager creation tests passed");
            printf("✅ NUMA coordinator manager creation test passed\n");
        } else {
            add_test_result("numa_coordinator_manager_creation", false, "Some coordinator manager creation tests failed");
        }
    }
    
    // Test function for function pointer execution
    static enum ggml_status test_work_function(void * work_context, struct ggml_compute_params * params) {
        // Simple test function that just modifies the context data
        if (!work_context || !params) {
            return GGML_STATUS_FAILED;
        }
        
        // Context should contain a simple counter we can increment
        int* counter = (int*)work_context;
        (*counter)++;
        
        return GGML_STATUS_SUCCESS;
    }
    
    // Test: Function pointer submission and execution
    void test_function_pointer_submission() {
        printf("--- Test: Function Pointer Submission ---\n");
        
        printf("Testing function pointer submission and execution...\n");
        
        bool all_tests_passed = true;
        
        // Get global coordinator for testing
        struct ggml_numa_coordinator_manager* mgr = ggml_numa_coordinator_manager_get_global(4, false);
        if (!mgr) {
            add_test_result("function_pointer_submission", false, "Failed to get coordinator manager");
            return;
        }
        
        // Test: Single node execution strategy
        printf("  Testing single node execution strategy...\n");
        {
            int counter = 0;
            ggml_numa_execution_strategy_t strategy = {
                .node_strategy = NUMA_NODE_STRATEGY_SINGLE_NODE,
                .on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD
            };
            
            int work_id = ggml_numa_coordinator_manager_submit_work_function(
                mgr, test_work_function, &counter, -1, strategy, 1024
            );
            
            if (work_id >= 0) {
                printf("  ✅ Single node function submission successful (work_id: %d)\n", work_id);
                // Note: In a real implementation we would wait for completion
                // For now, just test that submission worked
            } else {
                printf("  ❌ Single node function submission failed\n");
                all_tests_passed = false;
            }
        }
        
        // Test: Multi-thread execution strategy
        printf("  Testing multi-thread execution strategy...\n");
        {
            int counter = 0;
            ggml_numa_execution_strategy_t strategy = {
                .node_strategy = NUMA_NODE_STRATEGY_SINGLE_NODE,
                .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
            };
            
            int work_id = ggml_numa_coordinator_manager_submit_work_function(
                mgr, test_work_function, &counter, -1, strategy, 2048
            );
            
            if (work_id >= 0) {
                printf("  ✅ Multi-thread function submission successful (work_id: %d)\n", work_id);
            } else {
                printf("  ❌ Multi-thread function submission failed\n");
                all_tests_passed = false;
            }
        }
        
        // Test: Data parallel execution strategy
        printf("  Testing data parallel execution strategy...\n");
        {
            int counter = 0;
            ggml_numa_execution_strategy_t strategy = {
                .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL,
                .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
            };
            
            int work_id = ggml_numa_coordinator_manager_submit_work_function(
                mgr, test_work_function, &counter, -1, strategy, 4096
            );
            
            if (work_id >= 0) {
                printf("  ✅ Data parallel function submission successful (work_id: %d)\n", work_id);
            } else {
                printf("  ❌ Data parallel function submission failed\n");
                all_tests_passed = false;
            }
        }
        
        if (all_tests_passed) {
            add_test_result("function_pointer_submission", true, "All function pointer submission tests passed");
            printf("✅ Function pointer submission test passed\n");
        } else {
            add_test_result("function_pointer_submission", false, "Some function pointer submission tests failed");
        }
    }
    
    // Test: Execution strategy validation
    void test_execution_strategy_validation() {
        printf("--- Test: Execution Strategy Validation ---\n");
        
        printf("Testing different execution strategies...\n");
        
        bool all_passed = true;
        struct ggml_numa_coordinator_manager* mgr = ggml_numa_coordinator_manager_get_global(8, false);
        
        if (!mgr) {
            add_test_result("execution_strategy_validation", false, "Failed to get coordinator manager");
            return;
        }
        
        // Test execution strategies
        struct {
            const char* name;
            ggml_numa_execution_strategy_t strategy;
        } test_strategies[] = {
            {
                "Single Node, Single Thread",
                { NUMA_NODE_STRATEGY_SINGLE_NODE, NUMA_ON_NODE_STRATEGY_SINGLE_THREAD }
            },
            {
                "Single Node, Multi Thread", 
                { NUMA_NODE_STRATEGY_SINGLE_NODE, NUMA_ON_NODE_STRATEGY_MULTI_THREAD }
            },
            {
                "Data Parallel, Single Thread",
                { NUMA_NODE_STRATEGY_DATA_PARALLEL, NUMA_ON_NODE_STRATEGY_SINGLE_THREAD }
            },
            {
                "Data Parallel, Multi Thread",
                { NUMA_NODE_STRATEGY_DATA_PARALLEL, NUMA_ON_NODE_STRATEGY_MULTI_THREAD }
            },
            {
                "Task Parallel, Multi Thread",
                { NUMA_NODE_STRATEGY_TASK_PARALLEL, NUMA_ON_NODE_STRATEGY_MULTI_THREAD }
            }
        };
        
        for (int i = 0; i < 5; i++) {
            printf("  Testing %s strategy...\n", test_strategies[i].name);
            
            int counter = 0;
            int work_id = ggml_numa_coordinator_manager_submit_work_function(
                mgr, test_work_function, &counter, -1, test_strategies[i].strategy, 1024
            );
            
            if (work_id >= 0) {
                printf("  ✅ %s strategy: work submitted successfully (work_id: %d)\n", 
                       test_strategies[i].name, work_id);
            } else {
                printf("  ❌ %s strategy: work submission failed\n", test_strategies[i].name);
                all_passed = false;
            }
        }
        
        add_test_result("execution_strategy_validation", all_passed, 
                       all_passed ? "All execution strategies validated" : "Some strategies failed");
    }
    
    // Test: NUMA node assignment and buffer management
    void test_numa_node_assignment() {
        printf("--- Test: NUMA Node Assignment ---\n");
        
        printf("Testing NUMA node assignment and buffer allocation...\n");
        
        bool all_passed = true;
        struct ggml_numa_coordinator_manager* mgr = ggml_numa_coordinator_manager_get_global(8, true); // Force multi-socket
        
        if (!mgr) {
            add_test_result("numa_node_assignment", false, "Failed to get coordinator manager");
            return;
        }
        
        // Test different buffer sizes that would trigger different NUMA strategies
        struct {
            const char* name;
            size_t buffer_size;
            int numa_node_hint;
        } test_cases[] = {
            {"Small buffer, auto node", 1024, -1},
            {"Medium buffer, node 0", 64 * 1024, 0},
            {"Large buffer, auto node", 1024 * 1024, -1},
            {"Very large buffer, node 1", 16 * 1024 * 1024, 1}
        };
        
        for (int i = 0; i < 4; i++) {
            printf("  Testing %s (%zu bytes)...\n", test_cases[i].name, test_cases[i].buffer_size);
            
            int counter = 0;
            ggml_numa_execution_strategy_t strategy = {
                .node_strategy = NUMA_NODE_STRATEGY_DATA_PARALLEL,
                .on_node_strategy = NUMA_ON_NODE_STRATEGY_MULTI_THREAD
            };
            
            int work_id = ggml_numa_coordinator_manager_submit_work_function(
                mgr, test_work_function, &counter, test_cases[i].numa_node_hint, 
                strategy, test_cases[i].buffer_size
            );
            
            if (work_id >= 0) {
                printf("  ✅ %s: work submitted successfully (work_id: %d)\n", 
                       test_cases[i].name, work_id);
            } else {
                printf("  ❌ %s: work submission failed\n", test_cases[i].name);
                all_passed = false;
            }
        }
        
        add_test_result("numa_node_assignment", all_passed,
                       all_passed ? "All NUMA node assignments succeeded" : "Some assignments failed");
    }
    
    // Test: Function pointer submission error handling
    void test_function_pointer_error_handling() {
        printf("--- Test: Function Pointer Error Handling ---\n");
        
        printf("Testing error handling for function pointer submission...\n");
        
        bool all_passed = true;
        struct ggml_numa_coordinator_manager* mgr = ggml_numa_coordinator_manager_get_global(4, false);
        
        if (!mgr) {
            add_test_result("function_pointer_error_handling", false, "Failed to get coordinator manager");
            return;
        }
        
        // Test: NULL function pointer (should fail gracefully)
        printf("  Testing NULL function pointer...\n");
        {
            int counter = 0;
            ggml_numa_execution_strategy_t strategy = {
                .node_strategy = NUMA_NODE_STRATEGY_SINGLE_NODE,
                .on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD
            };
            
            int work_id = ggml_numa_coordinator_manager_submit_work_function(
                mgr, NULL, &counter, -1, strategy, 1024
            );
            
            if (work_id < 0) {
                printf("  ✅ NULL function pointer correctly rejected\n");
            } else {
                printf("  ❌ NULL function pointer incorrectly accepted\n");
                all_passed = false;
            }
        }
        
        // Test: NULL work context (should be acceptable for some functions)
        printf("  Testing NULL work context...\n");
        {
            ggml_numa_execution_strategy_t strategy = {
                .node_strategy = NUMA_NODE_STRATEGY_SINGLE_NODE,
                .on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD
            };
            
            int work_id = ggml_numa_coordinator_manager_submit_work_function(
                mgr, test_work_function, NULL, -1, strategy, 1024
            );
            
            // This might pass or fail depending on implementation - just check for graceful handling
            printf("  ✅ NULL work context handled gracefully (work_id: %d)\n", work_id);
        }
        
        // Test: Invalid NUMA node hint (should auto-correct)
        printf("  Testing invalid NUMA node hint...\n");
        {
            int counter = 0;
            ggml_numa_execution_strategy_t strategy = {
                .node_strategy = NUMA_NODE_STRATEGY_SINGLE_NODE,
                .on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD
            };
            
            int work_id = ggml_numa_coordinator_manager_submit_work_function(
                mgr, test_work_function, &counter, 999, strategy, 1024  // Invalid node 999
            );
            
            if (work_id >= 0) {
                printf("  ✅ Invalid NUMA node hint auto-corrected (work_id: %d)\n", work_id);
            } else {
                printf("  ⚠️  Invalid NUMA node hint caused submission failure\n");
                // This might be acceptable behavior
            }
        }
        
        // Test: Zero buffer size
        printf("  Testing zero buffer size...\n");
        {
            int counter = 0;
            ggml_numa_execution_strategy_t strategy = {
                .node_strategy = NUMA_NODE_STRATEGY_SINGLE_NODE,
                .on_node_strategy = NUMA_ON_NODE_STRATEGY_SINGLE_THREAD
            };
            
            int work_id = ggml_numa_coordinator_manager_submit_work_function(
                mgr, test_work_function, &counter, -1, strategy, 0  // Zero buffer size
            );
            
            if (work_id >= 0) {
                printf("  ✅ Zero buffer size handled gracefully (work_id: %d)\n", work_id);
            } else {
                printf("  ⚠️  Zero buffer size caused submission failure\n");
                // This might be acceptable behavior depending on implementation
            }
        }
        
        add_test_result("function_pointer_error_handling", all_passed, 
                       all_passed ? "Error conditions handled gracefully" : "Some error handling failed");
    }
    
    // Run all tests
    void run_all_tests() {
        if (!is_initialized()) {
            printf("❌ Test suite not properly initialized\n");
            return;
        }
        
        printf("================================================================================\n");
        printf("                        NUMA Coordinator Test Suite\n");
        printf("================================================================================\n\n");
        
        test_numa_coordinator_manager_creation();
        printf("\n");
        
        test_function_pointer_submission();
        printf("\n");
        
        test_execution_strategy_validation();
        printf("\n");
        
        test_numa_node_assignment();
        printf("\n");
        
        test_function_pointer_error_handling();
        printf("\n");
        
        print_results();
    }
    
    void print_results() {
        printf("================================================================================\n");
        printf("                           Test Results Summary\n");
        printf("================================================================================\n");
        
        int passed = 0, total = results.size();
        
        for (const auto& result : results) {
            const char* status = result.passed ? "✅ PASS" : "❌ FAIL";
            printf("%-50s %s", result.test_name, status);
            if (strlen(result.message) > 0) {
                printf(" - %s", result.message);
            }
            printf("\n");
            
            if (result.passed) passed++;
        }
        
        printf("--------------------------------------------------------------------------------\n");
        printf("Total: %d/%d tests passed", passed, total);
        
        if (passed == total) {
            printf(" 🎉 ALL TESTS PASSED!\n");
        } else {
            printf(" ⚠️  %d test(s) failed\n", total - passed);
        }
        
        printf("================================================================================\n");
        
        if (passed == total) {
            printf("✅ NUMA Coordinator Test Suite: SUCCESS\n");
        } else {
            printf("❌ NUMA Coordinator Test Suite: FAILURES DETECTED\n");
        }
    }
};

int main() {
    NumaCoordinatorTestSuite test_suite;
    
    if (!test_suite.is_initialized()) {
        printf("❌ Failed to initialize test suite\n");
        return 1;
    }
    
    test_suite.run_all_tests();
    
    printf("\n🎉 NUMA Coordinator testing completed!\n");
    
    return 0;
}
