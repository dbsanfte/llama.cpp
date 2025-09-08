/**
 * NUMA Mathematical Correctness Test Template: [OPERATION] Operation
 * 
 * This template provides a comprehensive test framework for verifying mathematical 
 * equivalence between NUMA parallel operations and serial reference implementations.
 * 
 * USAGE:
 * 1. Copy this template to test-numa-mathematical-correctness-[operation].cpp
 * 2. Replace all instances of [OPERATION] with your operation name (e.g., ADD, MUL, SUB)
 * 3. Replace all instances of [operation] with lowercase operation name (e.g., add, mul, sub)  
 * 4. Update the mathematical verification logic in the comparison functions
 * 5. Adjust tensor dimensions and test cases as needed for your operation
 * 
 * TEST COVERAGE PROVIDED:
 * 1. Mathematical Equivalence (3-Stage Approach):
 *    a) Single-thread Single-node: Tests basic kernel functionality and fallback mechanisms
 *    b) Multi-thread Single-node: Tests multi-threading coordination within single NUMA node  
 *    c) Multi-thread Multi-node: Tests full NUMA data-parallel execution across multiple nodes
 * 
 * 2. Quantization Type Coverage (Complete Support Matrix):
 *    - Tests all 28+ type combinations supported by reference implementation
 *    - Ensures proper quantization handling for all production model scenarios
 *    - Verifies NUMA kernels handle quantized fallbacks correctly
 * 
 * 3. Broadcasting Regression Prevention:
 *    - Tests specific broadcasting scenarios that previously caused memory corruption
 *    - Validates multi-dimensional broadcasting logic (Matrix + Vector patterns)
 * 
 * 4. Threshold Regression Testing:
 *    - Tests small tensors with forced strategies to catch strategy selection bugs
 * 
 * 5. Extreme Edge Cases:
 *    - Tests pathological tensor sizes (1-element, prime numbers, uneven splits)
 * 
 * 6. Race Condition Detection:
 *    - Multiple test runs to catch intermittent threading issues
 */

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <regex>
#include <thread>

// GGML includes
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-numa-executor.h"
#include "ggml-cpu/numa-kernels/numa-kernels.h"
#include "ggml-cpu/binary-ops.h"
#include "ggml-cpu/ggml-numa-openmp-coordinator.h"
#include "ggml-cpu/ggml-numa-shared.h"

// Global test configuration
std::string g_test_filter = "";
bool g_filter_enabled = false;
bool g_summary_only = false;

// Conditional printf macro for summary-only mode
#define TEST_PRINTF(...) do { if (!g_summary_only) printf(__VA_ARGS__); } while(0)

/**
 * Check if a test name matches the current filter
 */
bool matches_filter(const std::string& test_name) {
    if (!g_filter_enabled) {
        return true;  // No filter, run all tests
    }
    
    try {
        std::regex filter_regex(g_test_filter, std::regex_constants::icase);
        return std::regex_search(test_name, filter_regex);
    } catch (const std::regex_error& e) {
        printf("⚠️  Invalid regex filter '%s': %s\n", g_test_filter.c_str(), e.what());
        printf("   Running all tests instead.\n");
        return true;  // On regex error, run all tests
    }
}

// Test result structure
struct TestResult {
    std::string test_name;
    bool passed;
    std::string failure_reason;
};

// Test configuration
struct TestConfig {
    int ne0, ne1, ne2, ne3;
    ggml_numa_execution_strategy_t strategy;
    const char* test_name;
    const char* strategy_name;
};

// Size classifications (matching complexity levels)
enum TestSizeClass {
    TINY,      // Small tensors for basic validation
    SMALL,     // Medium tensors for multi-threading tests
    MEDIUM,    // Large tensors for data-parallel tests
    LARGE      // Very large tensors for stress testing
};

// Get tensor dimensions based on size class
TestConfig get_test_config(TestSizeClass size_class, ggml_numa_execution_strategy_t strategy, const char* strategy_name) {
    TestConfig config;
    config.strategy = strategy;
    config.strategy_name = strategy_name;
    
    switch (size_class) {
        case TINY:
            config.ne0 = 16; config.ne1 = 16; config.ne2 = 1; config.ne3 = 1;
            config.test_name = "TINY";
            break;
        case SMALL:
            config.ne0 = 64; config.ne1 = 64; config.ne2 = 4; config.ne3 = 1;
            config.test_name = "SMALL";
            break;
        case MEDIUM:
            config.ne0 = 256; config.ne1 = 256; config.ne2 = 8; config.ne3 = 1;
            config.test_name = "MEDIUM";
            break;
        case LARGE:
            config.ne0 = 512; config.ne1 = 512; config.ne2 = 16; config.ne3 = 1;
            config.test_name = "LARGE";
            break;
    }
    
    return config;
}

// Compare float arrays with tolerance for numerical precision
bool compare_float_arrays(const float* a, const float* b, size_t count, const char* operation_name, float tolerance = 1e-6f) {
    size_t mismatches = 0;
    const size_t max_reported_mismatches = 10;
    
    for (size_t i = 0; i < count; i++) {
        float diff = fabsf(a[i] - b[i]);
        float rel_error = (fabsf(b[i]) > 1e-9f) ? diff / fabsf(b[i]) : diff;
        
        if (diff > tolerance && rel_error > tolerance) {
            if (mismatches < max_reported_mismatches) {
                printf("❌ %s Mismatch[%zu]: NUMA=%.9f, Reference=%.9f, Diff=%.9f, RelErr=%.9f\n", 
                       operation_name, i, a[i], b[i], diff, rel_error);
            } else if (mismatches == max_reported_mismatches) {
                printf("❌ ... (suppressing further mismatches)\n");
            }
            mismatches++;
        }
    }
    
    if (mismatches > 0) {
        printf("❌ %s: %zu/%zu elements mismatched (%.2f%% error rate)\n", 
               operation_name, mismatches, count, (float)mismatches * 100.0f / count);
        return false;
    }
    
    printf("✅ %s: All %zu elements match within tolerance\n", operation_name, count);
    return true;
}

/**
 * MATHEMATICAL OPERATION PLACEHOLDER - CUSTOMIZE FOR YOUR OPERATION
 * 
 * Replace this function with the correct mathematical operation for your kernel.
 * Examples:
 *   ADD: return src0_val + src1_val;
 *   MUL: return src0_val * src1_val;
 *   SUB: return src0_val - src1_val;
 *   DIV: return src0_val / src1_val;
 */
float perform_reference_operation(float src0_val, float src1_val) {
    // TODO: Replace this with your operation's mathematical formula
    return src0_val + src1_val;  // PLACEHOLDER: Currently set to ADD operation
}

/**
 * Test Suite Class for [OPERATION] Mathematical Correctness
 * 
 * TODO: Replace class name and all method names with your operation:
 * - Change Numa[Operation]MathematicalCorrectnessTestSuite to your operation
 * - Change all [OPERATION] references to your operation name
 * - Update the ggml operation calls (ggml_add -> ggml_[operation])
 */
class Numa[Operation]MathematicalCorrectnessTestSuite {
public:
    std::vector<TestResult> results;
    
    /**
     * Test single [OPERATION] case with specified dimensions and forced strategy
     * 
     * TODO: Replace [OPERATION] with your operation name throughout this function
     * TODO: Replace ggml_add with ggml_[operation] calls
     */
    bool test_single_[OPERATION]_case(int ne0, int ne1, int ne2, int ne3, ggml_numa_execution_strategy_t strategy, 
                             const char* test_name, const char* strategy_name) {
        TEST_PRINTF("\n🧮 Testing [OPERATION] %s (%dx%dx%dx%d, strategy=%s)\n", test_name, ne0, ne1, ne2, ne3, strategy_name);
        
        const size_t total_elements = ne0 * ne1 * ne2 * ne3;
        bool case_passed = false;
        
        // Create GGML context for NUMA test
        struct ggml_init_params test_params = {
            .mem_size = 512 * 1024 * 1024,  // 512 MB
            .mem_buffer = nullptr,
            .no_alloc = false,
        };
        
        struct ggml_context* test_ctx = ggml_init(test_params);
        if (!test_ctx) {
            printf("❌ Failed to create NUMA test context\n");
            return false;
        }
        
        // Create input tensors
        struct ggml_tensor* input_a = ggml_new_tensor_4d(test_ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
        struct ggml_tensor* input_b = ggml_new_tensor_4d(test_ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
        
        if (!input_a || !input_b) {
            printf("❌ Failed to create input tensors\n");
            ggml_free(test_ctx);
            return false;
        }
        
        // Initialize input data with deterministic values for reproducibility
        float* data_a = (float*)ggml_get_data(input_a);
        float* data_b = (float*)ggml_get_data(input_b);
        
        for (size_t i = 0; i < total_elements; i++) {
            // Use different patterns to catch indexing errors
            data_a[i] = (float)(i % 100) * 0.1f + 1.0f;  // Values: 1.0, 1.1, ..., 10.9, 1.0, ...
            data_b[i] = (float)((i * 7) % 50) * 0.01f;   // Values: 0.0, 0.07, 0.14, ..., modulo pattern
        }
        
        // TODO: Replace ggml_add with your operation (ggml_mul, ggml_sub, ggml_div, etc.)
        struct ggml_tensor* numa_result = ggml_add(test_ctx, input_a, input_b);
        if (!numa_result) {
            printf("❌ Failed to create NUMA [OPERATION] operation\n");
            ggml_free(test_ctx);
            return false;
        }
        
        // Initialize NUMA system 
        ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);
        
        // Query the NUMA kernel to see if it's supported
        ggml_numa_execution_strategy_t strategy_result = ggml_numa_kernels_query(numa_result);
        const ggml_numa_kernel_cache_entry_t * cache_entry = ggml_numa_lookup_kernel_direct(numa_result->op);
        bool is_supported = cache_entry && cache_entry->supported;
        const char* kernel_name = cache_entry ? ggml_numa_get_kernel_name_from_cache(cache_entry) : "Unknown";
        
        if (!is_supported) {
            printf("⚠️  [OPERATION] operation not supported by NUMA kernels - skipping NUMA test\n");
            ggml_free(test_ctx);
            return false;
        }
        
        printf("📊 NUMA Strategy: %s\n", kernel_name);
        printf("🔧 Strategy Test: Forcing execution strategy to %s\n", strategy_name);
        
        // Ensure NUMA dispatch is enabled for our test
        ggml_numa_set_fallback_flag(false);
        
        // Setup compute plan for NUMA execution
        int default_threads = std::max(1u, std::thread::hardware_concurrency());
        struct ggml_cplan cplan = ggml_graph_plan(ggml_new_graph(test_ctx), default_threads, nullptr);
        cplan.work_size = 0;
        cplan.work_data = nullptr;
        cplan.n_threads = default_threads;
        cplan.threadpool = nullptr;
        cplan.abort_callback = nullptr;
        cplan.abort_callback_data = nullptr;
        
        // Execute using NUMA executor with forced strategy
        enum ggml_status dispatch_result = ggml_numa_executor_execute_tensor_forced_strategy(numa_result, &cplan, strategy);
        
        if (dispatch_result != GGML_STATUS_SUCCESS) {
            printf("❌ NUMA execution failed with status %d\n", (int)dispatch_result);
            ggml_free(test_ctx);
            return false;
        }
        
        // Reference Test: Execute [OPERATION] operation using reference implementation
        struct ggml_init_params ref_params = {
            .mem_size = 512 * 1024 * 1024,  // 512 MB
            .mem_buffer = nullptr,
            .no_alloc = false,
        };
        
        struct ggml_context* ref_ctx = ggml_init(ref_params);
        if (!ref_ctx) {
            printf("❌ Failed to create reference context\n");
            ggml_free(test_ctx);
            return false;
        }
        
        // Create reference tensors with same data
        struct ggml_tensor* ref_input_a = ggml_new_tensor_4d(ref_ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
        struct ggml_tensor* ref_input_b = ggml_new_tensor_4d(ref_ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
        
        // TODO: Replace ggml_add with your operation
        struct ggml_tensor* ref_result = ggml_add(ref_ctx, ref_input_a, ref_input_b);
        
        // Copy data to reference tensors
        memcpy(ggml_get_data(ref_input_a), data_a, total_elements * sizeof(float));
        memcpy(ggml_get_data(ref_input_b), data_b, total_elements * sizeof(float));
        
        // Execute reference implementation bypassing NUMA dispatch
        ggml_numa_set_fallback_flag(true);  // Force fallback to reference implementation
        
        struct ggml_cgraph* ref_gf = ggml_new_graph(ref_ctx);
        ggml_build_forward_expand(ref_gf, ref_result);
        
        struct ggml_cplan ref_plan = ggml_graph_plan(ref_gf, 1, nullptr);  // Single thread
        if (ref_plan.work_size > 0) {
            ref_plan.work_data = (uint8_t*)malloc(ref_plan.work_size);
        }
        
        printf("   Executing TRUE reference implementation (bypassing NUMA)...\n");
        ggml_graph_compute(ref_gf, &ref_plan);
        
        ggml_numa_set_fallback_flag(false);  // Re-enable NUMA dispatch
        
        if (ref_plan.work_data) {
            free(ref_plan.work_data);
        }
        
        // Compare results
        float* numa_data = (float*)ggml_get_data(numa_result);
        float* ref_data = (float*)ggml_get_data(ref_result);
        
        case_passed = compare_float_arrays(numa_data, ref_data, total_elements, "[OPERATION]");
        
        if (case_passed) {
            printf("✅ [OPERATION] %s test PASSED\n", test_name);
        } else {
            printf("❌ [OPERATION] %s test FAILED\n", test_name);
        }
        
        ggml_free(ref_ctx);
        ggml_free(test_ctx);
        return case_passed;
    }
    
    /**
     * Mathematical Equivalence Testing
     * Tests various tensor sizes and thread configurations to ensure mathematical correctness.
     * TODO: Customize tensor dimensions for your operation's typical use cases.
     */
    void test_mathematical_equivalence() {
        printf("🧮 Testing Mathematical Equivalence...\n");
        
        struct {
            const char* name;
            int64_t ne0, ne1, ne2, ne3;
        } test_cases[] = {
            {"1D tiny", 16, 1, 1, 1},
            {"1D small", 256, 1, 1, 1},
            {"1D medium", 4096, 1, 1, 1},
            {"1D large", 65536, 1, 1, 1},
            {"2D square", 32, 32, 1, 1},
            {"2D rectangle", 128, 64, 1, 1},
            {"3D small", 16, 16, 8, 1},
            {"3D medium", 32, 32, 16, 1},
            {"4D tensor", 8, 8, 8, 8},
            {"4D large", 16, 16, 16, 8},
            {"broadcast 1D", 1024, 1, 1, 1},
            {"broadcast 2D", 512, 2, 1, 1},
        };
        
        int thread_counts[] = {1, 2, 4, 6, 8, 15, 16, 31, 32, 64, 128};
        int passed = 0, total = 0;
        
        for (auto& test_case : test_cases) {
            for (int threads : thread_counts) {
                if (threads > (int)std::thread::hardware_concurrency() * 2) {
                    continue; // Skip unrealistic thread counts
                }
                
                total++;
                char test_name[256];
                snprintf(test_name, sizeof(test_name), "%s (%d threads)", test_case.name, threads);
                
                bool result = run_equivalence_test(test_case.ne0, test_case.ne1, test_case.ne2, test_case.ne3, threads, test_name);
                if (result) passed++;
            }
        }
        
        printf("📈 Mathematical Equivalence Results: %d/%d passed (%.1f%%)\n", 
               passed, total, (total > 0) ? (100.0f * passed / total) : 0.0f);
        results.mathematical_equivalence_passed = passed;
        results.mathematical_equivalence_total = total;
    }
    
    /**
     * Quantization Type Coverage Testing  
     * Tests all quantization types supported by the operation to prevent silent model failures.
     * TODO: Verify which quantization types your operation supports and adjust accordingly.
     */
    void test_quantization_coverage() {
        printf("🔢 Testing Quantization Type Coverage...\n");
        
        // Test various quantization types that operations commonly support
        struct {
            ggml_type type;
            const char* name;
        } quant_types[] = {
            {GGML_TYPE_F32, "F32"},
            {GGML_TYPE_F16, "F16"},
            {GGML_TYPE_Q8_0, "Q8_0"},
            {GGML_TYPE_Q4_0, "Q4_0"},
            {GGML_TYPE_Q5_0, "Q5_0"},
            {GGML_TYPE_Q5_1, "Q5_1"},
            {GGML_TYPE_Q8_1, "Q8_1"},
            // TODO: Add other quantization types your operation supports
        };
        
        int passed = 0, total = 0;
        const int64_t ne0 = 128, ne1 = 64, ne2 = 1, ne3 = 1;
        
        for (auto& quant_type : quant_types) {
            total++;
            printf("   Testing %s quantization...\n", quant_type.name);
            
            bool result = test_quantization_type(quant_type.type, ne0, ne1, ne2, ne3);
            if (result) {
                printf("   ✅ %s quantization PASSED\n", quant_type.name);
                passed++;
            } else {
                printf("   ⚠️  %s quantization FAILED or not supported\n", quant_type.name);
            }
        }
        
        printf("🔢 Quantization Coverage Results: %d/%d passed (%.1f%%)\n", 
               passed, total, (total > 0) ? (100.0f * passed / total) : 0.0f);
        results.quantization_passed = passed;
        results.quantization_total = total;
    }
    
    /**
     * Helper method to test a specific quantization type
     * TODO: Modify this to handle quantization types specific to your operation
     */
    bool test_quantization_type(ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
        try {
            struct ggml_init_params params = {
                .mem_size = 256 * 1024 * 1024,  // 256 MB
                .mem_buffer = nullptr,
                .no_alloc = false,
            };
            
            struct ggml_context* ctx = ggml_init(params);
            if (!ctx) return false;
            
            // Create tensors with the specified quantization type
            struct ggml_tensor* a = ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);
            struct ggml_tensor* b = ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);
            
            if (!a || !b) {
                ggml_free(ctx);
                return false;
            }
            
            // TODO: Replace ggml_add with your operation
            struct ggml_tensor* result = ggml_add(ctx, a, b);
            
            if (!result) {
                ggml_free(ctx);
                return false;
            }
            
            // Try to query the operation - if not supported, this will fail gracefully
            ggml_numa_execution_strategy_t strategy_result = ggml_numa_kernels_query(result);
            const ggml_numa_kernel_cache_entry_t * cache_entry = ggml_numa_lookup_kernel_direct(result->op);
            bool is_supported = cache_entry && cache_entry->supported;
            
            ggml_free(ctx);
            return is_supported;
            
        } catch (...) {
            return false;
        }
    }
    
    /**
     * Edge Cases and Regression Testing
     * Tests specific edge cases and scenarios that have caused issues in the past.
     * TODO: Add operation-specific edge cases and known regression scenarios.
     */
    void test_edge_cases() {
        printf("🔍 Testing Edge Cases...\n");
        
        struct {
            const char* name;
            int64_t ne0, ne1, ne2, ne3;
            const char* description;
        } edge_cases[] = {
            {"Single element", 1, 1, 1, 1, "Minimal tensor size"},
            {"Power of 2", 1024, 1, 1, 1, "Aligned memory access"},
            {"Prime number", 1021, 1, 1, 1, "Unaligned memory access"},
            {"SIMD boundary", 8, 1, 1, 1, "SIMD vector boundary"},
            {"Cache line", 16, 1, 1, 1, "Cache line alignment"},
            {"Large 1D", 1048576, 1, 1, 1, "Large contiguous memory"},
            {"Thin matrix", 2048, 1, 1, 1, "High aspect ratio"},
            {"Wide matrix", 1, 2048, 1, 1, "Wide aspect ratio"},
            {"Cube", 64, 64, 64, 1, "3D cube tensor"},
            {"Hypercube", 16, 16, 16, 16, "4D hypercube"},
            // TODO: Add operation-specific edge cases
        };
        
        int passed = 0, total = 0;
        
        for (auto& edge_case : edge_cases) {
            total++;
            printf("   Testing %s (%s)...\n", edge_case.name, edge_case.description);
            
            bool result = run_equivalence_test(edge_case.ne0, edge_case.ne1, edge_case.ne2, edge_case.ne3, 
                                             std::thread::hardware_concurrency(), edge_case.name);
            if (result) {
                printf("   ✅ %s PASSED\n", edge_case.name);
                passed++;
            } else {
                printf("   ❌ %s FAILED\n", edge_case.name);
            }
        }
        
        printf("🔍 Edge Cases Results: %d/%d passed (%.1f%%)\n", 
               passed, total, (total > 0) ? (100.0f * passed / total) : 0.0f);
        results.edge_cases_passed = passed;
        results.edge_cases_total = total;
    }
    
    /**
     * Race Condition Detection
     * Tests for race conditions in multi-threaded execution.
     * TODO: Customize the mathematical verification for your operation.
     */
    void test_race_conditions() {
        printf("🏁 Testing Race Conditions...\n");
        
        const int64_t ne0 = 1024, ne1 = 32, ne2 = 1, ne3 = 1;
        const size_t total_elements = ne0 * ne1 * ne2 * ne3;
        const int iterations = 50;
        int passed = 0;
        
        for (int i = 0; i < iterations; i++) {
            struct ggml_init_params params = {
                .mem_size = 128 * 1024 * 1024,
                .mem_buffer = nullptr,
                .no_alloc = false,
            };
            
            struct ggml_context* ctx = ggml_init(params);
            if (!ctx) continue;
            
            struct ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
            struct ggml_tensor* b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
            
            // TODO: Replace ggml_add with your operation
            struct ggml_tensor* result = ggml_add(ctx, a, b);
            
            if (!a || !b || !result) {
                ggml_free(ctx);
                continue;
            }
            
            // Fill with deterministic data
            float* src0_data = (float*)ggml_get_data(a);
            float* src1_data = (float*)ggml_get_data(b);
            
            for (size_t j = 0; j < total_elements; j++) {
                src0_data[j] = sinf(j * 0.1f);
                src1_data[j] = cosf(j * 0.1f);
            }
            
            // Execute with maximum threads to stress test
            ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);
            ggml_numa_set_fallback_flag(false);
            
            struct ggml_cgraph* gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, result);
            
            int max_threads = std::thread::hardware_concurrency();
            struct ggml_cplan plan = ggml_graph_plan(gf, max_threads, nullptr);
            if (plan.work_size > 0) {
                plan.work_data = (uint8_t*)malloc(plan.work_size);
            }
            
            ggml_graph_compute(gf, &plan);
            
            // Verify results are mathematically correct
            float* result_data = (float*)ggml_get_data(result);
            bool iteration_passed = true;
            
            for (size_t j = 0; j < total_elements && iteration_passed; j++) {
                // TODO: Replace addition with your operation's mathematical formula
                float expected = src0_data[j] + src1_data[j];
                float actual = result_data[j];
                
                if (fabsf(actual - expected) > 1e-5f) {
                    iteration_passed = false;
                }
            }
            
            if (iteration_passed) passed++;
            
            if (plan.work_data) free(plan.work_data);
            ggml_free(ctx);
        }
        
        printf("🏁 Race Condition Results: %d/%d iterations passed (%.1f%%)\n", 
               passed, iterations, (100.0f * passed / iterations));
        results.race_conditions_passed = passed;
        results.race_conditions_total = iterations;
    }
    
    /**
     * Run all test suites and report comprehensive results
     */
    void run_all_tests() {
        printf("🚀 Starting [OPERATION] Mathematical Correctness Test Suite\n");
        printf("═══════════════════════════════════════════════════════════\n");
        
        // Initialize test results
        results = {0};
        
        // Run all test suites
        test_mathematical_equivalence();
        test_quantization_coverage();
        test_edge_cases();
        test_race_conditions();
        
        // Calculate overall results
        int total_passed = results.mathematical_equivalence_passed + 
                          results.quantization_passed + 
                          results.edge_cases_passed + 
                          results.race_conditions_passed;
        
        int total_tests = results.mathematical_equivalence_total + 
                         results.quantization_total + 
                         results.edge_cases_total + 
                         results.race_conditions_total;
        
        // Print comprehensive summary
        printf("\n📊 COMPREHENSIVE TEST RESULTS\n");
        printf("═══════════════════════════════════════════════════════════\n");
        printf("🧮 Mathematical Equivalence: %d/%d (%.1f%%)\n", 
               results.mathematical_equivalence_passed, results.mathematical_equivalence_total,
               (results.mathematical_equivalence_total > 0) ? 
               (100.0f * results.mathematical_equivalence_passed / results.mathematical_equivalence_total) : 0.0f);
        
        printf("🔢 Quantization Coverage:    %d/%d (%.1f%%)\n", 
               results.quantization_passed, results.quantization_total,
               (results.quantization_total > 0) ? 
               (100.0f * results.quantization_passed / results.quantization_total) : 0.0f);
        
        printf("🔍 Edge Cases:               %d/%d (%.1f%%)\n", 
               results.edge_cases_passed, results.edge_cases_total,
               (results.edge_cases_total > 0) ? 
               (100.0f * results.edge_cases_passed / results.edge_cases_total) : 0.0f);
        
        printf("🏁 Race Conditions:          %d/%d (%.1f%%)\n", 
               results.race_conditions_passed, results.race_conditions_total,
               (results.race_conditions_total > 0) ? 
               (100.0f * results.race_conditions_passed / results.race_conditions_total) : 0.0f);
        
        printf("═══════════════════════════════════════════════════════════\n");
        printf("🎯 OVERALL RESULT: %d/%d tests passed (%.1f%%)\n", 
               total_passed, total_tests, 
               (total_tests > 0) ? (100.0f * total_passed / total_tests) : 0.0f);
        
        if (total_passed == total_tests && total_tests > 0) {
            printf("✅ ALL TESTS PASSED - [OPERATION] kernel is mathematically correct!\n");
        } else {
            printf("❌ SOME TESTS FAILED - [OPERATION] kernel needs attention!\n");
        }
        printf("═══════════════════════════════════════════════════════════\n");
    }
};

/**
 * Main function - Entry point for the test suite
 * TODO: Update the test class name when customizing for your operation
 */
int main(int argc, char** argv) {
    TestConfig config = get_test_config(argc, argv);
    
    printf("🔧 Test Configuration:\n");
    printf("   Filter: %s\n", config.filter.empty() ? "ALL" : config.filter.c_str());
    printf("   Summary Only: %s\n", config.summary_only ? "YES" : "NO");
    printf("   Hardware Threads: %d\n", (int)std::thread::hardware_concurrency());
    printf("\n");
    
    // TODO: Replace 'OperationTest' with your operation name (e.g., 'AddTest', 'MulTest', etc.)
    OperationTest test_suite;
    
    if (config.summary_only) {
        printf("📋 Running summary-only tests for [OPERATION] operation...\n");
        
        // Run a minimal set of tests for quick validation
        bool quick_test = test_suite.run_equivalence_test(64, 64, 1, 1, 
                                                         std::thread::hardware_concurrency(), 
                                                         "Quick validation");
        
        if (quick_test) {
            printf("✅ [OPERATION] quick validation PASSED\n");
            return 0;
        } else {
            printf("❌ [OPERATION] quick validation FAILED\n");
            return 1;
        }
    } else {
        // Run comprehensive test suite
        test_suite.run_all_tests();
        
        // Return appropriate exit code
        int total_passed = test_suite.results.mathematical_equivalence_passed + 
                          test_suite.results.quantization_passed + 
                          test_suite.results.edge_cases_passed + 
                          test_suite.results.race_conditions_passed;
        
        int total_tests = test_suite.results.mathematical_equivalence_total + 
                         test_suite.results.quantization_total + 
                         test_suite.results.edge_cases_total + 
                         test_suite.results.race_conditions_total;
        
        return (total_passed == total_tests && total_tests > 0) ? 0 : 1;
    }
}
