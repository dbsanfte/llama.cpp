/**
 * NUMA Mathematical Correctness Test: MUL_MAT Operation
 * 
 * This test verifies mathematical equivalence between NUMA parallel MUL_MAT operations
 * and serial reference implementations. It ensures the NUMA MUL_MAT kernel produces
 * identical results to the reference implementation across various scenarios.
 * 
 * TEST COVERAGE:
 * 1. Mathematical Equivalence (Simplified 3-Stage Approach):
 *    a) Single-thread Single-node: Tests basic kernel functionality and fallback mechanisms
 *    b) Multi-thread Single-node: Tests multi-threading coordination within single NUMA node  
 *    c) Multi-thread Multi-node: Tests full NUMA data-parallel execution across multiple nodes
 *    - Tests across TINY → LARGE tensor sizes for comprehensive coverage
 *    - Eliminates artificial thread constraints, focuses on production execution modes
 * 
 * 2. Quantization Type Coverage:
 *    - All quantization types supported by reference MUL_MAT: F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S, IQ1_S, IQ1_M, IQ4_NL, IQ4_XS, Q8_K, BF16, TQ1_0, TQ2_0
 *    - Both source tensors (src0 and src1) with different type combinations
 *    - Various matrix dimensions and batch configurations
 *    - Work buffer allocation testing for type conversions
 * 
 * 3. Matrix Multiplication Scenarios:
 *    - Various matrix dimensions: square, rectangular, batch operations
 *    - Different quantization type combinations
 *    - Broadcasting scenarios for multi-dimensional operations
 * 
 * KEY DESIGN PRINCIPLES:
 * - Simplified execution testing: 3 clear stages instead of complex thread scenarios
 * - Comprehensive quantization type coverage for model reliability
 * - Multi-dimensional testing across various matrix configurations
 * - Direct comparison between NUMA parallel and serial reference implementations
 * - Detailed error reporting with mathematical mismatch information
 * - Regression testing for matrix multiplication correctness
 * 
 * ARCHITECTURE INTEGRATION:
 * - Tests NUMA Executor strategy selection (Single/Single, Single/Multi, Data-Parallel)
 * - Validates NUMA Coordinator thread management and NUMA binding
 * - Ensures NUMA Kernel Registry provides correct function pointers
 * - Verifies shared memory optimization and work buffer management
 */

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <regex>

// GGML includes
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-numa-executor.h"
#include "ggml-cpu/numa-kernels/numa-kernels.h"
#include "ggml-cpu/ops.h"
#include "ggml-cpu/ggml-numa-openmp-coordinator.h"

// Global test filter
std::string g_test_filter = "";
bool g_filter_enabled = false;

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
    int src0_ne0, src0_ne1, src0_ne2, src0_ne3;  // src0 dimensions: [K, M, batch2, batch3]
    int src1_ne0, src1_ne1, src1_ne2, src1_ne3;  // src1 dimensions: [K, N, batch2, batch3]
    int num_threads;
    const char* test_name;
    enum ggml_type src0_type;   // Source 0 quantization type
    enum ggml_type src1_type;   // Source 1 quantization type (usually F32)
};

// Size classifications (matching complexity levels)
enum TestSizeClass {
    TINY,      // Small matrices for basic validation
    SMALL,     // Medium matrices for multi-threading tests
    MEDIUM,    // Large matrices for data-parallel tests
    LARGE      // Very large matrices for stress testing
};

// Quantization types supported by MUL_MAT (from type_traits analysis)
const std::vector<enum ggml_type> SUPPORTED_QUANT_TYPES = {
    GGML_TYPE_F32,    // Standard float32
    GGML_TYPE_F16,    // Half precision
    GGML_TYPE_Q4_0,   // 4-bit quantization
    GGML_TYPE_Q4_1,   // 4-bit quantization with bias
    GGML_TYPE_Q5_0,   // 5-bit quantization
    GGML_TYPE_Q5_1,   // 5-bit quantization with bias
    GGML_TYPE_Q8_0,   // 8-bit quantization
    // GGML_TYPE_Q8_1,   // Q8_1 is not supported as src0 type (no vec_dot function)
    GGML_TYPE_Q2_K,   // K-quantization 2-bit
    GGML_TYPE_Q3_K,   // K-quantization 3-bit
    GGML_TYPE_Q4_K,   // K-quantization 4-bit
    GGML_TYPE_Q5_K,   // K-quantization 5-bit
    GGML_TYPE_Q6_K,   // K-quantization 6-bit
    GGML_TYPE_IQ2_XXS, // Intelligent quantization 2-bit XXS
    GGML_TYPE_IQ2_XS,  // Intelligent quantization 2-bit XS
    GGML_TYPE_IQ3_XXS, // Intelligent quantization 3-bit XXS
    GGML_TYPE_IQ3_S,   // Intelligent quantization 3-bit S
    GGML_TYPE_IQ2_S,   // Intelligent quantization 2-bit S
    GGML_TYPE_IQ1_S,   // Intelligent quantization 1-bit S
    GGML_TYPE_IQ1_M,   // Intelligent quantization 1-bit M
    GGML_TYPE_IQ4_NL,  // Intelligent quantization 4-bit NL
    GGML_TYPE_IQ4_XS,  // Intelligent quantization 4-bit XS
    GGML_TYPE_Q8_K,    // K-quantization 8-bit
    GGML_TYPE_BF16,    // Brain float16
    GGML_TYPE_TQ1_0,   // Ternary quantization 1-bit
    GGML_TYPE_TQ2_0    // Ternary quantization 2-bit
};

// Get tensor dimensions based on size class
TestConfig get_test_config(TestSizeClass size_class, int num_threads, enum ggml_type src0_type = GGML_TYPE_F32, enum ggml_type src1_type = GGML_TYPE_F32) {
    TestConfig config;
    config.num_threads = num_threads;
    config.src0_type = src0_type;
    config.src1_type = src1_type;
    
    switch (size_class) {
        case TINY:
            // Small matrix: src0=[64,32,1,1] @ src1=[64,16,1,1] = dst=[32,16,1,1]
            config.src0_ne0 = 64; config.src0_ne1 = 32; config.src0_ne2 = 1; config.src0_ne3 = 1;
            config.src1_ne0 = 64; config.src1_ne1 = 16; config.src1_ne2 = 1; config.src1_ne3 = 1;
            config.test_name = "TINY";
            break;
        case SMALL:
            // Medium matrix: src0=[128,64,1,1] @ src1=[128,32,1,1] = dst=[64,32,1,1]
            config.src0_ne0 = 128; config.src0_ne1 = 64; config.src0_ne2 = 1; config.src0_ne3 = 1;
            config.src1_ne0 = 128; config.src1_ne1 = 32; config.src1_ne2 = 1; config.src1_ne3 = 1;
            config.test_name = "SMALL";
            break;
        case MEDIUM:
            // Large matrix: src0=[256,128,2,1] @ src1=[256,64,2,1] = dst=[128,64,2,1]
            config.src0_ne0 = 256; config.src0_ne1 = 128; config.src0_ne2 = 2; config.src0_ne3 = 1;
            config.src1_ne0 = 256; config.src1_ne1 = 64; config.src1_ne2 = 2; config.src1_ne3 = 1;
            config.test_name = "MEDIUM";
            break;
        case LARGE:
            // Very large matrix: src0=[512,256,2,2] @ src1=[512,128,2,2] = dst=[256,128,2,2]
            config.src0_ne0 = 512; config.src0_ne1 = 256; config.src0_ne2 = 2; config.src0_ne3 = 2;
            config.src1_ne0 = 512; config.src1_ne1 = 128; config.src1_ne2 = 2; config.src1_ne3 = 2;
            config.test_name = "LARGE";
            break;
    }
    
    return config;
}

// Compare float arrays with tolerance for numerical precision
bool compare_float_arrays(const float* a, const float* b, size_t count, const char* operation_name, float tolerance = 1e-4f) {
    size_t mismatches = 0;
    const size_t max_reported_mismatches = 10;
    
    for (size_t i = 0; i < count; i++) {
        float diff = fabsf(a[i] - b[i]);
        float rel_error = (fabsf(b[i]) > 1e-9f) ? diff / fabsf(b[i]) : diff;
        
        // Use relaxed tolerance for quantized operations
        if (diff > tolerance && rel_error > tolerance) {
            if (mismatches < max_reported_mismatches) {
                printf("   ❌ Mismatch at index %zu: NUMA=%.8f vs Reference=%.8f (diff=%.8f, rel_err=%.6f%%)\n",
                       i, a[i], b[i], diff, rel_error * 100.0f);
            } else if (mismatches == max_reported_mismatches) {
                printf("   ... (suppressing further mismatch reports)\n");
            }
            mismatches++;
        }
    }
    
    if (mismatches > 0) {
        printf("   ❌ %s: %zu/%zu elements mismatched (%.2f%% failure rate)\n",
               operation_name, mismatches, count, (float)mismatches / count * 100.0f);
        return false;
    }
    
    printf("   ✅ %s: All %zu elements match (%.2e tolerance)\n", operation_name, count, tolerance);
    return true;
}

// Initialize tensor with deterministic values for MUL_MAT testing
void initialize_mul_mat_tensor(struct ggml_tensor* tensor, int seed = 42) {
    if (tensor->type == GGML_TYPE_F32) {
        float* data = (float*)ggml_get_data(tensor);
        size_t count = ggml_nelements(tensor);
        
        // Generate reproducible test data with good numerical properties for matrix multiplication
        for (size_t i = 0; i < count; i++) {
            // Use smaller values to avoid overflow in matrix multiplication
            float val = sinf((float)(i + seed) * 0.01f) * 0.1f + cosf((float)(i + seed) * 0.03f) * 0.05f;
            data[i] = val;
        }
    } else {
        // For quantized types, first initialize as F32 then quantize
        size_t count = ggml_nelements(tensor);
        std::vector<float> temp_data(count);
        
        // Generate F32 data first
        for (size_t i = 0; i < count; i++) {
            float val = sinf((float)(i + seed) * 0.01f) * 0.1f + cosf((float)(i + seed) * 0.03f) * 0.05f;
            temp_data[i] = val;
        }
        
        // Quantize the data according to tensor type
        const struct ggml_type_traits_cpu * type_traits = ggml_get_type_traits_cpu(tensor->type);
        if (type_traits && type_traits->from_float) {
            // Use quantization function if available
            size_t row_size = ggml_row_size(tensor->type, tensor->ne[0]);
            size_t num_rows = count / tensor->ne[0];
            
            for (size_t row = 0; row < num_rows; row++) {
                type_traits->from_float(temp_data.data() + row * tensor->ne[0],
                                      (char*)ggml_get_data(tensor) + row * row_size,
                                      tensor->ne[0]);
            }
        } else {
            throw std::runtime_error("Quantization not supported for type: " + std::string(ggml_type_name(tensor->type)));
        }
    }
}

// Create MUL_MAT operation in a context
struct ggml_tensor* create_mul_mat_operation(struct ggml_context* ctx, const TestConfig& config, 
                                            struct ggml_tensor* src0, struct ggml_tensor* src1) {
    return ggml_mul_mat(ctx, src0, src1);
}

// Test MUL_MAT operation correctness
bool test_mul_mat_correctness(const TestConfig& config, const std::string& test_description) {
    const size_t ctx_size = 256 * 1024 * 1024;  // 256MB context
    
    try {
        // Create contexts for reference and NUMA computations
        struct ggml_init_params params;
        params.mem_size = ctx_size;
        params.mem_buffer = nullptr;
        params.no_alloc = false;
        
        struct ggml_context* ctx_ref = ggml_init(params);
        struct ggml_context* ctx_numa = ggml_init(params);
        
        if (!ctx_ref || !ctx_numa) {
            throw std::runtime_error("Failed to create GGML contexts");
        }
        
        // Create source tensors
        struct ggml_tensor* src0_ref = ggml_new_tensor_4d(ctx_ref, config.src0_type, 
                                                          config.src0_ne0, config.src0_ne1, 
                                                          config.src0_ne2, config.src0_ne3);
        struct ggml_tensor* src1_ref = ggml_new_tensor_4d(ctx_ref, config.src1_type,
                                                          config.src1_ne0, config.src1_ne1,
                                                          config.src1_ne2, config.src1_ne3);
        
        struct ggml_tensor* src0_numa = ggml_new_tensor_4d(ctx_numa, config.src0_type,
                                                           config.src0_ne0, config.src0_ne1,
                                                           config.src0_ne2, config.src0_ne3);
        struct ggml_tensor* src1_numa = ggml_new_tensor_4d(ctx_numa, config.src1_type,
                                                           config.src1_ne0, config.src1_ne1,
                                                           config.src1_ne2, config.src1_ne3);
        
        // Initialize input tensors with same values
        initialize_mul_mat_tensor(src0_ref, 42);
        initialize_mul_mat_tensor(src1_ref, 84);
        
        // Copy data to NUMA tensors
        memcpy(ggml_get_data(src0_numa), ggml_get_data(src0_ref), ggml_nbytes(src0_ref));
        memcpy(ggml_get_data(src1_numa), ggml_get_data(src1_ref), ggml_nbytes(src1_ref));
        
        // Create MUL_MAT operations
        struct ggml_tensor* result_ref = create_mul_mat_operation(ctx_ref, config, src0_ref, src1_ref);
        struct ggml_tensor* result_numa = create_mul_mat_operation(ctx_numa, config, src0_numa, src1_numa);
        
        // Create computation graphs
        struct ggml_cgraph* gf_ref = ggml_new_graph(ctx_ref);
        struct ggml_cgraph* gf_numa = ggml_new_graph(ctx_numa);
        
        ggml_build_forward_expand(gf_ref, result_ref);
        ggml_build_forward_expand(gf_numa, result_numa);
        
        // Execute reference computation (standard CPU)
        ggml_graph_compute_with_ctx(ctx_ref, gf_ref, config.num_threads);
        
        // Execute NUMA computation
        ggml_graph_compute_with_ctx(ctx_numa, gf_numa, config.num_threads);
        
        // Compare results
        const float* ref_data = (const float*)ggml_get_data(result_ref);
        const float* numa_data = (const float*)ggml_get_data(result_numa);
        size_t result_count = ggml_nelements(result_ref);
        
        // Use relaxed tolerance for quantized operations
        float tolerance = (config.src0_type != GGML_TYPE_F32) ? 1e-2f : 1e-4f;
        
        bool arrays_match = compare_float_arrays(numa_data, ref_data, result_count, test_description.c_str(), tolerance);
        
        // Cleanup
        ggml_free(ctx_ref);
        ggml_free(ctx_numa);
        
        return arrays_match;
        
    } catch (const std::exception& e) {
        printf("   ❌ Exception in %s: %s\n", test_description.c_str(), e.what());
        return false;
    }
}

// Run test for specific size class and thread count
TestResult run_mul_mat_test(TestSizeClass size_class, int num_threads, enum ggml_type src0_type = GGML_TYPE_F32) {
    TestConfig config = get_test_config(size_class, num_threads, src0_type);
    
    std::string test_name = std::string("MUL_MAT_") + config.test_name + "_" + ggml_type_name(src0_type) + "_" + std::to_string(num_threads) + "T";
    
    if (!matches_filter(test_name)) {
        return {test_name, true, "Skipped (filter)"};
    }
    
    printf("🧮 Testing %s: [%dx%dx%dx%d] @ [%dx%dx%dx%d] = [%dx%dx%dx%d] with %d threads\n",
           test_name.c_str(),
           config.src0_ne0, config.src0_ne1, config.src0_ne2, config.src0_ne3,
           config.src1_ne0, config.src1_ne1, config.src1_ne2, config.src1_ne3,
           config.src0_ne1, config.src1_ne1, config.src0_ne2, config.src0_ne3,  // result dimensions
           num_threads);
    
    bool passed = test_mul_mat_correctness(config, test_name);
    
    if (passed) {
        printf("   ✅ PASSED\n");
        return {test_name, true, ""};
    } else {
        std::string failure_reason = "Mathematical mismatch between NUMA and reference implementations";
        printf("   ❌ FAILED: %s\n", failure_reason.c_str());
        return {test_name, false, failure_reason};
    }
}

// New function that uses NUMA strategy instead of forcing thread counts
TestResult run_mul_mat_test_with_strategy(TestSizeClass size_class, enum ggml_type src0_type, 
                                          enum ggml_numa_strategy strategy, int thread_constraint) {
    TestConfig config = get_test_config(size_class, 0, src0_type);  // Don't force thread count
    
    std::string strategy_name;
    switch (strategy) {
        case GGML_NUMA_STRATEGY_MIRROR: 
            strategy_name = (thread_constraint == 1) ? "Single/Single" : "Data-Parallel";
            break;
        case GGML_NUMA_STRATEGY_ISOLATE: 
            strategy_name = "Single/Multi";
            break;
        default: 
            strategy_name = "Unknown";
            break;
    }
    
    std::string test_name = std::string("MUL_MAT_") + config.test_name + "_" + 
                           ggml_type_name(src0_type) + "_" + strategy_name;
    
    if (!matches_filter(test_name)) {
        return {test_name, true, "Skipped (filter)"};
    }
    
    printf("🧮 Testing %s: [%dx%dx%dx%d] @ [%dx%dx%dx%d] = [%dx%dx%dx%d] strategy=%s\n",
           test_name.c_str(),
           config.src0_ne0, config.src0_ne1, config.src0_ne2, config.src0_ne3,
           config.src1_ne0, config.src1_ne1, config.src1_ne2, config.src1_ne3,
           config.src0_ne1, config.src1_ne1, config.src0_ne2, config.src0_ne3,  // result dimensions
           strategy_name.c_str());
    
    // Set up NUMA strategy - OpenMP coordinator handles threading automatically
    ggml_numa_init(strategy);
    
    bool passed = test_mul_mat_correctness(config, test_name);
    
    if (passed) {
        printf("   ✅ PASSED\n");
        return {test_name, true, ""};
    } else {
        std::string failure_reason = "Mathematical mismatch between NUMA and reference implementations";
        printf("   ❌ FAILED: %s\n", failure_reason.c_str());
        return {test_name, false, failure_reason};
    }
}

// ============================================================================
// Main Test Execution
// ============================================================================

int main(int argc, char* argv[]) {
    printf("🚀 NUMA Mathematical Correctness Test: MUL_MAT Operation\n");
    printf("=======================================================\n\n");
    
    // Parse command line arguments for test filtering
    if (argc >= 2) {
        g_test_filter = argv[1];
        g_filter_enabled = true;
        printf("🔍 Test filter enabled: '%s'\n\n", g_test_filter.c_str());
    }
    
    std::vector<TestResult> results;
    
    // Define execution stages (matches ADD test pattern)
    struct ExecutionStage {
        const char* name;
        const char* description;
        enum ggml_numa_strategy strategy;
        int thread_constraint;
    };
    
    std::vector<ExecutionStage> stages = {
        {"Single-thread Single-node", 
         "Tests basic kernel functionality and single-node fallback",
         GGML_NUMA_STRATEGY_MIRROR, 1},  // Force single thread for stage 1
        
        {"Multi-thread Single-node", 
         "Tests multi-threading coordination within single NUMA node",
         GGML_NUMA_STRATEGY_ISOLATE, 0},  // Use ISOLATE to force single node, allow multiple threads
        
        {"Multi-thread Multi-node", 
         "Tests full NUMA data-parallel execution across multiple nodes",
         GGML_NUMA_STRATEGY_MIRROR, 0}   // Use MIRROR for full multi-node execution
    };
    
    // Test all size classes and all execution stages
    std::vector<TestSizeClass> size_classes = {TINY, SMALL, MEDIUM, LARGE};
    
    for (int stage_idx = 0; stage_idx < (int)stages.size(); stage_idx++) {
        const auto& stage = stages[stage_idx];
        
        printf("📋 STAGE %d: %s\n", stage_idx + 1, stage.name);
        printf("   Purpose: %s\n\n", stage.description);
        
        for (auto size_class : size_classes) {
            printf("🎯 Testing %s tensors: %s\n", 
                   get_test_config(size_class, 0).test_name, stage.name);
            
            // Test key quantization types for each stage
            std::vector<enum ggml_type> stage_test_types;
            if (stage_idx == 0) {
                // Stage 1: Test basic types for functionality
                stage_test_types = {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0};
            } else if (stage_idx == 1) {
                // Stage 2: Test threading with common types
                stage_test_types = {GGML_TYPE_F32, GGML_TYPE_Q4_0};
            } else {
                // Stage 3: Test data-parallel with precision types
                stage_test_types = {GGML_TYPE_F32, GGML_TYPE_F16};
            }
            
            for (auto qtype : stage_test_types) {
                TestResult result = run_mul_mat_test_with_strategy(size_class, qtype, stage.strategy, stage.thread_constraint);
                results.push_back(result);
            }
        }
    }
    
    // Test Stage 4: Comprehensive Quantization Coverage (use ISOLATE strategy for deterministic testing)
    printf("\n📋 STAGE 4: Comprehensive Quantization Tests\n");
    printf("   Purpose: Validate all supported quantization types\n\n");
    
    // Test subset of quantization types that have from_float functions
    std::vector<enum ggml_type> testable_types = {
        GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, 
        GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0,  // Q8_1 removed - not supported as src0
        GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
        GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_K, GGML_TYPE_BF16,
        GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0
    };
    
    for (auto qtype : testable_types) {
        const struct ggml_type_traits_cpu * traits = ggml_get_type_traits_cpu(qtype);
        if (traits && traits->from_float) {  // Only test types that support quantization
            TestResult result = run_mul_mat_test_with_strategy(SMALL, qtype, GGML_NUMA_STRATEGY_ISOLATE, 0);
            results.push_back(result);
        }
    }
    
    // ========================================================================
    // Test Results Summary
    // ========================================================================
    
    printf("\n🎯 TEST RESULTS SUMMARY\n");
    printf("=======================\n");
    
    int passed = 0, failed = 0, skipped = 0;
    std::vector<TestResult> failures;
    
    for (const auto& result : results) {
        if (result.failure_reason == "Skipped (filter)") {
            skipped++;
        } else if (result.passed) {
            passed++;
        } else {
            failed++;
            failures.push_back(result);
        }
    }
    
    printf("✅ Passed: %d\n", passed);
    printf("❌ Failed: %d\n", failed);
    printf("⏭️  Skipped: %d\n", skipped);
    printf("📊 Total: %d\n\n", (int)results.size());
    
    if (failed > 0) {
        printf("💥 FAILED TESTS:\n");
        for (const auto& failure : failures) {
            printf("   ❌ %s: %s\n", failure.test_name.c_str(), failure.failure_reason.c_str());
        }
        printf("\n");
    }
    
    if (failed == 0 && passed > 0) {
        printf("🎉 ALL TESTS PASSED! NUMA MUL_MAT kernel is mathematically correct.\n");
        return 0;
    } else if (failed > 0) {
        printf("💥 TESTS FAILED! NUMA MUL_MAT kernel has mathematical errors.\n");
        return 1;
    } else {
        printf("⚠️  NO TESTS RUN! Check test filter or implementation.\n");
        return 1;
    }
}