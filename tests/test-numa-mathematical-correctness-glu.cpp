/**
 * NUMA Mathematical Correctness Test: GLU Operation
 * 
 * This test verifies mathematical equivalence between NUMA parallel GLU operations
 * and serial reference implementations across all GLU variants.
 * 
 * @author David Sanftenberg
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
#include <random>
#include <memory>

// GGML includes
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cpu/ggml-numa-executor.h"
#include "ggml-cpu/numa-kernels/numa-kernels.h"
#include "ggml-cpu/ggml-numa-openmp-coordinator.h"
#include "ggml-cpu/ggml-numa-shared.h"

// Test configuration constants
constexpr float EPSILON_F32 = 1e-5f;
constexpr float EPSILON_F16 = 1e-3f;
constexpr int RANDOM_SEED = 12345;

// Global test filter
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

// Tensor size categories for comprehensive testing
enum tensor_size_category {
    TENSOR_SIZE_TINY = 0,        // 8 elements - minimal overhead testing
    TENSOR_SIZE_SMALL,           // 256 elements - cache-friendly
    TENSOR_SIZE_MEDIUM,          // 8192 elements - multi-threading threshold
    TENSOR_SIZE_LARGE,           // 262144 elements - data-parallel threshold
    TENSOR_SIZE_COUNT
};

// GLU operation variants to test
const std::vector<ggml_glu_op> glu_variants = {
    GGML_GLU_OP_REGLU,
    GGML_GLU_OP_SWIGLU,
    GGML_GLU_OP_GEGLU,
    GGML_GLU_OP_GEGLU_ERF,
    GGML_GLU_OP_GEGLU_QUICK
};

// Data types to test
const std::vector<ggml_type> test_data_types = {
    GGML_TYPE_F32,
    GGML_TYPE_F16
};

// Execution strategies for testing (strategy-based approach)
struct ExecutionStrategy {
    ggml_numa_execution_strategy_t strategy;
    const char* name;
    const char* description;
};

const std::vector<ExecutionStrategy> execution_strategies = {
    // Strategy 1: Single-thread, single-node
    {NUMA_STRATEGY_SINGLE_THREAD, 
     "Single-Single", "Single-thread execution on single NUMA node"},
    
    // Strategy 2: Multi-thread, single-node
    {NUMA_STRATEGY_SINGLE_NODE, 
     "Single-Multi", "Multi-thread execution within single NUMA node"},
    
    // Strategy 3: Multi-thread, multi-node (data-parallel)
    {NUMA_STRATEGY_DATA_PARALLEL, 
     "Data-Parallel", "Data-parallel execution across multiple NUMA nodes"}
};

/**
 * @brief Get tensor dimensions for specified size category
 */
std::vector<int64_t> get_tensor_dimensions(tensor_size_category size_cat) {
    switch (size_cat) {
        case TENSOR_SIZE_TINY:
            return {8, 1, 1, 1};                    // 8 elements total
        case TENSOR_SIZE_SMALL:
            return {16, 16, 1, 1};                  // 256 elements total
        case TENSOR_SIZE_MEDIUM:
            return {32, 32, 8, 1};                  // 8192 elements total
        case TENSOR_SIZE_LARGE:
            return {64, 64, 64, 1};                 // 262144 elements total
        default:
            return {8, 1, 1, 1};
    }
}

/**
 * @brief Get human-readable name for tensor size category
 */
const char* get_size_category_name(tensor_size_category size_cat) {
    switch (size_cat) {
        case TENSOR_SIZE_TINY: return "TINY";
        case TENSOR_SIZE_SMALL: return "SMALL";
        case TENSOR_SIZE_MEDIUM: return "MEDIUM";
        case TENSOR_SIZE_LARGE: return "LARGE";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Get human-readable name for GLU operation
 */
const char* get_glu_op_name(ggml_glu_op op) {
    switch (op) {
        case GGML_GLU_OP_REGLU: return "REGLU";
        case GGML_GLU_OP_SWIGLU: return "SWIGLU";
        case GGML_GLU_OP_GEGLU: return "GEGLU";
        case GGML_GLU_OP_GEGLU_ERF: return "GEGLU_ERF";
        case GGML_GLU_OP_GEGLU_QUICK: return "GEGLU_QUICK";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Fill tensor with random data for testing
 */
void fill_random_data(struct ggml_tensor* tensor, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    if (tensor->type == GGML_TYPE_F32) {
        float* data = (float*)tensor_data(tensor);
        size_t n_elements = ggml_nelements(tensor);
        for (size_t i = 0; i < n_elements; ++i) {
            data[i] = dist(rng);
        }
    } else if (tensor->type == GGML_TYPE_F16) {
        ggml_fp16_t* data = (ggml_fp16_t*)tensor_data(tensor);
        size_t n_elements = ggml_nelements(tensor);
        for (size_t i = 0; i < n_elements; ++i) {
            data[i] = ggml_fp32_to_fp16(dist(rng));
        }
    }
}

/**
 * @brief Test GLU mathematical correctness for specific configuration
 */
bool test_glu_correctness(ggml_glu_op glu_op, ggml_type data_type, tensor_size_category size_cat, 
                         ggml_numa_execution_strategy_t strategy) {
    
    // Get tensor dimensions
    std::vector<int64_t> dims = get_tensor_dimensions(size_cat);
    
    // Initialize contexts with smaller memory allocation
    struct ggml_init_params params;
    params.mem_size = 64 * 1024 * 1024;  // 64MB context (was 1GB)
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx_ref(ggml_init(params), ggml_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx_numa(ggml_init(params), ggml_free);
    
    if (!ctx_ref.get() || !ctx_numa.get()) {
        return false;
    }
    
    // Create input tensors (GLU needs input with even first dimension to split)
    if (dims[0] % 2 != 0) {
        dims[0] = dims[0] + 1;  // Make even
    }
    
    struct ggml_tensor* src_ref = ggml_new_tensor_4d(ctx_ref.get(), data_type, dims[0], dims[1], dims[2], dims[3]);
    struct ggml_tensor* src_numa = ggml_new_tensor_4d(ctx_numa.get(), data_type, dims[0], dims[1], dims[2], dims[3]);
    
    // Fill with identical random data
    std::mt19937 rng(RANDOM_SEED);
    fill_random_data(src_ref, rng);
    
    // Copy data to NUMA tensor
    memcpy(tensor_data(src_numa), tensor_data(src_ref), ggml_nbytes(src_ref));
    
    // Create GLU operations
    struct ggml_tensor* result_ref = ggml_glu(ctx_ref.get(), src_ref, glu_op, false);
    struct ggml_tensor* result_numa = ggml_glu(ctx_numa.get(), src_numa, glu_op, false);
    
    // Build graphs
    struct ggml_cgraph* graph_ref = ggml_new_graph(ctx_ref.get());
    struct ggml_cgraph* graph_numa = ggml_new_graph(ctx_numa.get());
    
    ggml_build_forward_expand(graph_ref, result_ref);
    ggml_build_forward_expand(graph_numa, result_numa);
    
    // Compute reference (single-threaded)
    if (ggml_graph_compute_with_ctx(ctx_ref.get(), graph_ref, 1) != GGML_STATUS_SUCCESS) {
        return false;
    }
    
    // Compute NUMA using strategy-based execution
    int n_threads = 1;  // Default for single-thread strategy
    if (strategy == NUMA_STRATEGY_SINGLE_NODE) {
        n_threads = std::thread::hardware_concurrency();  // Use available threads for single-node
    } else if (strategy == NUMA_STRATEGY_DATA_PARALLEL) {
        n_threads = std::thread::hardware_concurrency();  // Use available threads for data-parallel
    }
    
    if (ggml_graph_compute_with_ctx(ctx_numa.get(), graph_numa, n_threads) != GGML_STATUS_SUCCESS) {
        return false;
    }
    
    // Compare results
    if (ggml_nelements(result_ref) != ggml_nelements(result_numa)) {
        return false;
    }
    
    float epsilon = (data_type == GGML_TYPE_F32) ? EPSILON_F32 : EPSILON_F16;
    size_t n_elements = ggml_nelements(result_ref);
    
    if (data_type == GGML_TYPE_F32) {
        const float* ref_data = (const float*)tensor_data(result_ref);
        const float* numa_data = (const float*)tensor_data(result_numa);
        
        for (size_t i = 0; i < n_elements; ++i) {
            if (std::abs(ref_data[i] - numa_data[i]) > epsilon) {
                return false;
            }
        }
    } else if (data_type == GGML_TYPE_F16) {
        const ggml_fp16_t* ref_data = (const ggml_fp16_t*)tensor_data(result_ref);
        const ggml_fp16_t* numa_data = (const ggml_fp16_t*)tensor_data(result_numa);
        
        for (size_t i = 0; i < n_elements; ++i) {
            float ref_val = ggml_fp16_to_fp32(ref_data[i]);
            float numa_val = ggml_fp16_to_fp32(numa_data[i]);
            if (std::abs(ref_val - numa_val) > epsilon) {
                return false;
            }
        }
    }
    
    return true;
}

/**
 * Run comprehensive GLU tests
 */
bool run_comprehensive_glu_tests() {
    printf("🧪 GLU NUMA Mathematical Correctness Test Suite\n");
    printf("════════════════════════════════════════════════\n\n");
    
    std::vector<TestResult> results;
    int total_tests = 0;
    int passed_tests = 0;
    
    printf("📐 Mathematical Equivalence Testing\n");
    printf("───────────────────────────────────\n");
    
    // Strategy-based testing approach (like ROPE and MUL_MAT)
    for (int size_idx = 0; size_idx < TENSOR_SIZE_COUNT; size_idx++) {
        auto size_cat = static_cast<tensor_size_category>(size_idx);
        
        for (const auto& strategy : execution_strategies) {
            printf("\n🎯 Testing %s tensors: %s strategy\n", 
                   get_size_category_name(size_cat), strategy.name);
            printf("   %s\n", strategy.description);
            
            for (auto glu_op : glu_variants) {
                for (auto data_type : test_data_types) {
                    // Skip F16 for large tensors and complex strategies to keep test time reasonable
                    if (data_type == GGML_TYPE_F16 && 
                        (size_cat >= TENSOR_SIZE_LARGE || strategy.strategy == NUMA_STRATEGY_DATA_PARALLEL)) {
                        continue;
                    }
                    
                    std::string test_name = std::string("glu_") + get_glu_op_name(glu_op) + 
                                          "_" + ggml_type_name(data_type) + "_" + 
                                          get_size_category_name(size_cat) + "_" + strategy.name;
                    
                    if (!matches_filter(test_name)) {
                        continue;
                    }
                    
                    TEST_PRINTF("  Testing %s...", test_name.c_str());
                    
                    bool test_passed = test_glu_correctness(glu_op, data_type, size_cat, strategy.strategy);
                    
                    total_tests++;
                    if (test_passed) {
                        passed_tests++;
                        results.push_back({test_name, true, ""});
                        TEST_PRINTF(" ✅\n");
                    } else {
                        results.push_back({test_name, false, "Mathematical equivalence failed"});
                        TEST_PRINTF(" ❌\n");
                    }
                }
            }
        }
    }
    
    // Print final summary
    printf("\n📊 Final Test Summary\n");
    printf("════════════════════\n");
    printf("Total Tests:  %d\n", total_tests);
    printf("Passed Tests: %d\n", passed_tests);
    printf("Failed Tests: %d\n", total_tests - passed_tests);
    
    if (passed_tests == total_tests && total_tests > 0) {
        printf("\n🎉 ALL TESTS PASSED! GLU NUMA implementation is working correctly.\n");
        return true;
    } else {
        printf("\n⚠️  Some tests failed. Please review the failures above.\n");
        return false;
    }
}

/**
 * Parse command line arguments
 */
struct TestConfigCLI {
    std::string filter = "";
    bool summary_only = false;
    bool show_help = false;
};

void print_help() {
    printf("\nNUMA GLU Mathematical Correctness Test Suite\n");
    printf("============================================\n\n");
    printf("Usage: test-numa-mathematical-correctness-glu [OPTIONS]\n\n");
    printf("Options:\n");
    printf("  --filter <pattern>     Run only tests matching pattern (e.g., 'REGLU', 'F32', 'TINY')\n");
    printf("  --summary-only         Skip detailed tests, show summary only\n"); 
    printf("  --help                 Show this help message\n\n");
    printf("Examples:\n");
    printf("  test-numa-mathematical-correctness-glu --filter REGLU\n");
    printf("  test-numa-mathematical-correctness-glu --summary-only\n");
    printf("  test-numa-mathematical-correctness-glu --filter F32\n\n");
    printf("GLU Variants: REGLU, SWIGLU, GEGLU, GEGLU_ERF, GEGLU_QUICK\n");
    printf("Data Types: F32, F16\n");
    printf("Tensor Sizes: TINY(8), SMALL(256), MEDIUM(8K), LARGE(256K)\n");
}

TestConfigCLI parse_arguments(int argc, char** argv) {
    TestConfigCLI config;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            config.show_help = true;
        } else if (arg == "--summary-only") {
            config.summary_only = true;
        } else if (arg == "--filter" && i + 1 < argc) {
            config.filter = argv[++i];
        } else {
            printf("Unknown argument: %s\n", arg.c_str());
            config.show_help = true;
        }
    }
    
    return config;
}

/**
 * @brief Main test function
 */
int main(int argc, char** argv) {
    TestConfigCLI config = parse_arguments(argc, argv);
    
    if (config.show_help) {
        print_help();
        return 0;
    }
    
    // Set global configuration
    g_summary_only = config.summary_only;
    g_test_filter = config.filter;
    g_filter_enabled = !config.filter.empty();
    
    // Initialize NUMA system properly
    printf("Initializing NUMA system...\n");
    ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);
    
    printf("Initializing NUMA kernels...\n");
    ggml_numa_kernels_init();
    
    if (!config.filter.empty()) {
        printf("🔍 Running tests matching filter: '%s'\n", config.filter.c_str());
    }
    
    if (config.summary_only) {
        printf("📊 Summary mode: Running key validation tests only\n");
    }
    
    printf("\n");
    
    // Run comprehensive test suite
    bool all_passed = run_comprehensive_glu_tests();
    
    return all_passed ? 0 : 1;
}
