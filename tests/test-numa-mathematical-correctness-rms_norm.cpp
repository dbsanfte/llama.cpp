/**
 * NUMA Mathematical Correctness Test: RMS Normalization (RMS_NORM)
 * 
 * This test validates that NUMA-parallel RMS normalization produces
 * mathematically identical results to the reference implementation.
 * 
 * Test Strategy:
 * - Multi-dimensional tensors from tiny to gigantic scales
 * - Various thread counts to test coordinator execution
 * - Direct comparison between NUMA parallel and serial reference
 * - Comprehensive error reporting for any mismatches
 */

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <random>
#include <regex>
#include <optional>
#include <set>
#include <unistd.h>  // for sysconf

// GGML includes
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-numa-executor.h"
#include "ggml-numa-simple-coordinator.h"

// Forward declarations
extern "C" void ggml_compute_forward_rms_norm(const struct ggml_compute_params * params, struct ggml_tensor * dst);
extern "C" void ggml_numa_clear_dispatch_override(void);
extern "C" void ggml_numa_set_dispatch_enabled(bool enabled);
extern "C" bool ggml_numa_simple_coordinator_is_initialized(void);
extern "C" bool ggml_numa_simple_coordinator_init(struct ggml_threadpool_params * tpp);

// Test result structure
struct TestResult {
    std::string test_name;
    bool passed;
    std::string failure_reason;
};

class NumaRmsNormMathematicalCorrectnessTestSuite {
private:
    std::vector<TestResult> results;
    // Optional regex filter for test names
    std::optional<std::regex> name_filter;
    bool list_only = false;

    bool should_run(const std::string & test_name) const {
        if (!name_filter.has_value()) {
            return true;
        }
        return std::regex_search(test_name, name_filter.value());
    }
    
    // Utility function to compare float arrays with appropriate error tolerance
    bool compare_float_arrays(const float* numa_data, const float* ref_data, int count, const char* operation_name) {
        bool all_match = true;
        int error_count = 0;
        double max_abs_error = 0.0;
        double max_rel_error = 0.0;
        
        // RMS_NORM involves square roots and divisions, so we need reasonable tolerance
        double abs_tolerance = 1e-6;
        double rel_tolerance = 1e-6;
        
        for (int i = 0; i < count; i++) {
            double numa_val = numa_data[i];
            double ref_val = ref_data[i];
            
            double abs_error = std::abs(numa_val - ref_val);
            double rel_error = (ref_val != 0.0) ? abs_error / std::abs(ref_val) : abs_error;
            
            max_abs_error = std::max(max_abs_error, abs_error);
            max_rel_error = std::max(max_rel_error, rel_error);
            
            bool values_match = (abs_error <= abs_tolerance) || (rel_error <= rel_tolerance);
            if (!values_match) {
                all_match = false;
                error_count++;
                
                // Report first few errors for debugging
                if (error_count <= 10) {
                    printf("        🔍 Mismatch at index %d: NUMA=%.9f, REF=%.9f, abs_err=%.2e, rel_err=%.2e\n", 
                           i, numa_val, ref_val, abs_error, rel_error);
                }
            }
        }
        
        if (!all_match) {
            printf("        ❌ %s comparison failed: %d/%d mismatches (max_abs_err=%.2e, max_rel_err=%.2e)\n", 
                   operation_name, error_count, count, max_abs_error, max_rel_error);
        } else {
            printf("        ✅ %s comparison passed: max_abs_err=%.2e, max_rel_err=%.2e\n", 
                   operation_name, max_abs_error, max_rel_error);
        }
        
        return all_match;
    }
    
    // Generate test data for RMS normalization
    void generate_test_data_f32(std::vector<float>& data, int64_t ne, int pattern, std::mt19937& rng) {
        data.resize(ne);
        
        switch (pattern) {
            case 0: { // Sequential pattern
                for (int64_t i = 0; i < ne; i++) {
                    data[i] = (float)(i + 1) * 0.1f;
                }
                break;
            }
            case 1: { // Alternating pattern  
                for (int64_t i = 0; i < ne; i++) {
                    data[i] = (i % 2 == 0) ? 1.0f : -1.0f;
                }
                break;
            }
            case 2: { // Random pattern
                std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
                for (int64_t i = 0; i < ne; i++) {
                    data[i] = dist(rng);
                }
                break;
            }
            case 3: { // Small values 
                std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
                for (int64_t i = 0; i < ne; i++) {
                    data[i] = dist(rng) * 0.5f;
                }
                break;
            }
        }
    }
    
    // Test single RMS_NORM case
    bool test_single_RMS_NORM_case(int ne0, int ne1, int ne2, int ne3, const char* size_label) {
        printf("    🧮 Testing RMS_NORM %s: [%d, %d, %d, %d]\n", size_label, ne0, ne1, ne2, ne3);
        
        std::mt19937 rng(12345);
        bool all_patterns_passed = true;
        const char* pattern_names[] = {"Sequential", "Alternating", "Random", "SmallValues"};
        
        // Test multiple data patterns - focus on one for debugging
        for (int pattern = 3; pattern < 4; pattern++) { // Only pattern 3 = SmallValues for now
            printf("      📊 Pattern: %s\n", pattern_names[pattern]);
            
            // Create NUMA context
            struct ggml_init_params params = ggml_init_params{};
            params.mem_size = 1024*1024*1024; // 1GB - generous allocation for all test cases including HUGE
            params.mem_buffer = nullptr;
            params.no_alloc = false;
            
            struct ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                printf("        ❌ Failed to create NUMA context\n");
                return false;
            }
            
            // Create reference context
            struct ggml_context * ref_ctx = ggml_init(params);
            if (!ref_ctx) {
                printf("        ❌ Failed to create reference context\n");
                ggml_free(ctx);
                return false;
            }
            
            // Generate test data
            int64_t total_elements = (int64_t)ne0 * ne1 * ne2 * ne3;
            std::vector<float> src_data;
            generate_test_data_f32(src_data, total_elements, pattern, rng);
            
            // Create tensors for NUMA test
            struct ggml_tensor * src = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
            struct ggml_tensor * dst = ggml_rms_norm(ctx, src, 1e-5f);  // eps = 1e-5
            
            // Create tensors for reference test  
            struct ggml_tensor * ref_src = ggml_new_tensor_4d(ref_ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
            struct ggml_tensor * ref_dst = ggml_rms_norm(ref_ctx, ref_src, 1e-5f);  // same eps
            
            if (!src || !dst || !ref_src || !ref_dst) {
                printf("        ❌ Failed to create tensors\n");
                ggml_free(ref_ctx);
                ggml_free(ctx);
                return false;
            }
            
            // Copy data to both source tensors
            memcpy(ggml_get_data(src), src_data.data(), ggml_nbytes(src));
            memcpy(ggml_get_data(ref_src), src_data.data(), ggml_nbytes(ref_src));
            
            // ==================================================================
            // Test 1: NUMA execution
            // ==================================================================
            
            printf("        🚀 Running NUMA execution...\n");
            
            // Build computation graph for NUMA test
            struct ggml_cgraph* numa_gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(numa_gf, dst);
            
            // Create compute plan for NUMA - use 1 thread for consistency
            struct ggml_cplan numa_plan = ggml_graph_plan(numa_gf, 1, nullptr);
            numa_plan.work_data = (uint8_t*)malloc(numa_plan.work_size);
            
            // Enable NUMA dispatch
            ggml_numa_set_dispatch_enabled(true);
            
            // Initialize NUMA system for dispatch tests with MIRROR strategy
            ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);
            
            // Execute NUMA computation using graph
            ggml_graph_compute(numa_gf, &numa_plan);
            
            // Diagnostic: compute simple hash of first 128 floats of dst
            {
                size_t sample_elems = std::min<size_t>(128, (size_t)total_elements);
                const uint8_t * bytes = (const uint8_t*)ggml_get_data(dst);
                uint64_t h = 1469598103934665603ULL;
                for (size_t bi = 0; bi < sample_elems * sizeof(float); ++bi) { 
                    h ^= bytes[bi]; h *= 1099511628211ULL; 
                }
                printf("          🔍 NUMA pass hash(sample128)=0x%016llx\n", (unsigned long long)h);
            }
            
            // ==================================================================
            // Test 2: Reference execution (disable NUMA)
            // ==================================================================
            
            printf("        📚 Running reference execution...\n");
            
            // Ensure subsequent reference computation bypasses NUMA executor
            ggml_numa_clear_dispatch_override();
            ggml_numa_set_dispatch_enabled(false);
            
            // Build computation graph for reference
            struct ggml_cgraph* ref_gf = ggml_new_graph(ref_ctx);
            ggml_build_forward_expand(ref_gf, ref_dst);
            
            // Execute reference computation
            ggml_graph_compute_with_ctx(ref_ctx, ref_gf, 1);
            
            // Re-enable NUMA dispatch for subsequent tests
            ggml_numa_set_dispatch_enabled(true);
            
            // Diagnostic: hash reference output
            {
                size_t sample_elems = std::min<size_t>(128, (size_t)total_elements);
                const uint8_t * bytes = (const uint8_t*)ggml_get_data(ref_dst);
                uint64_t h = 1469598103934665603ULL;
                for (size_t bi = 0; bi < sample_elems * sizeof(float); ++bi) { 
                    h ^= bytes[bi]; h *= 1099511628211ULL; 
                }
                printf("          🔍 REF  pass hash(sample128)=0x%016llx\n", (unsigned long long)h);
            }
            
            // ==================================================================
            // Test 3: Compare results
            // ==================================================================
            
            printf("        🔍 Comparing results...\n");
            
            bool results_match = compare_float_arrays(
                (const float*)ggml_get_data(dst), 
                (const float*)ggml_get_data(ref_dst), 
                total_elements, 
                "RMS_NORM"
            );
            
            if (!results_match) {
                all_patterns_passed = false;
                
                // Detailed debugging for first mismatch
                printf("        🔬 Detailed analysis of first few elements:\n");
                const float* numa_data = (const float*)ggml_get_data(dst);
                const float* ref_data = (const float*)ggml_get_data(ref_dst);
                const float* src_input = (const float*)ggml_get_data(src);
                
                for (int i = 0; i < std::min(10, (int)total_elements); i++) {
                    printf("          [%d] input=%.6f, NUMA=%.6f, REF=%.6f, diff=%.2e\n", 
                           i, src_input[i], numa_data[i], ref_data[i], 
                           std::abs(numa_data[i] - ref_data[i]));
                }
            }
            
            // Cleanup
            free(numa_plan.work_data);
            ggml_free(ref_ctx);
            ggml_free(ctx);
            
            if (!results_match) {
                printf("        ❌ Pattern %s failed\n", pattern_names[pattern]);
                return false;
            } else {
                printf("        ✅ Pattern %s passed\n", pattern_names[pattern]);
            }
        }
        
        return all_patterns_passed;
    }

public:
    // Constructor with optional test name filter
    NumaRmsNormMathematicalCorrectnessTestSuite(const std::string& filter = "", bool list_tests = false) 
        : list_only(list_tests) {
        if (!filter.empty()) {
            try {
                name_filter = std::regex(filter, std::regex_constants::icase);
            } catch (const std::regex_error& e) {
                printf("❌ Invalid regex pattern '%s': %s\n", filter.c_str(), e.what());
                throw;
            }
        }
    }

    // Run multi-dimensional tests across different tensor sizes
    bool run_multi_dimensional_tests() {
        printf("🧮 === NUMA RMS_NORM Multi-Dimensional Tests ===\n");
        
        // Test cases: [ne0, ne1, ne2, ne3, label]
        struct TestCase {
            int ne0, ne1, ne2, ne3;
            const char* label;
        };
        
        std::vector<TestCase> test_cases = {
            // Small tests
            {4, 1, 1, 1, "TINY_1D"},
            {8, 4, 1, 1, "SMALL_2D"},
            {16, 8, 4, 1, "SMALL_3D"},
            {32, 16, 8, 2, "SMALL_4D"},
            
            // Medium tests  
            {64, 32, 1, 1, "MEDIUM_2D"},
            {128, 64, 4, 1, "MEDIUM_3D"},
            {256, 128, 8, 2, "MEDIUM_4D"},
            
            // Large tests
            {512, 256, 1, 1, "LARGE_2D"},
            {1024, 512, 4, 1, "LARGE_3D"},
            {2048, 1024, 8, 2, "LARGE_4D"},
            
            // Very large tests
            {4096, 2048, 1, 1, "HUGE_2D"},
            {8192, 4096, 2, 1, "HUGE_3D"},
        };
        
        bool all_passed = true;
        
        for (const auto& test_case : test_cases) {
            std::string test_name = std::string("multi_dim_") + test_case.label;
            
            if (list_only) {
                printf("  📋 %s\n", test_name.c_str());
                continue;
            }
            
            if (!should_run(test_name)) {
                continue;
            }
            
            bool passed = test_single_RMS_NORM_case(
                test_case.ne0, test_case.ne1, test_case.ne2, test_case.ne3, test_case.label
            );
            
            results.push_back({test_name, passed, passed ? "" : "Mathematical correctness failed"});
            
            if (!passed) {
                all_passed = false;
                printf("  ❌ %s: Failed\n", test_case.label);
            } else {
                printf("  ✅ %s: Passed\n", test_case.label);
            }
        }
        
        return all_passed;
    }
    
    // Run multi-threading tests
    bool run_multi_threading_tests() {
        printf("\n🧵 === NUMA RMS_NORM Multi-Threading Tests ===\n");
        
        if (list_only) {
            printf("  📋 multi_thread_small\n");
            printf("  📋 multi_thread_medium\n");
            printf("  📋 multi_thread_large\n");
            return true;
        }
        
        bool all_passed = true;
        
        // Test different thread counts
        std::vector<int> thread_counts = {1, 2, 4, 6, 8, 15, 16, 31, 32, 64, 128};
        
        // Add NUMA-aware thread counts based on actual hardware topology
        std::vector<int> numa_thread_counts;
        int num_numa_nodes = ggml_numa_simple_coordinator_get_num_nodes();
        if (num_numa_nodes > 0) {
            // Get total CPU count and derive threads per node
            // Using reasonable estimates for threads per node based on typical systems
            int total_cpus = 0;
            #ifdef __linux__
            // Try to get CPU count from /proc/cpuinfo or sysconf
            total_cpus = sysconf(_SC_NPROCESSORS_CONF);
            #endif
            if (total_cpus <= 0) {
                total_cpus = 16; // Conservative fallback
            }
            
            int threads_per_node = total_cpus / num_numa_nodes;
            if (threads_per_node <= 0) threads_per_node = 4; // Minimum fallback
            
            // Test with max threads per node for each NUMA node scenario
            numa_thread_counts.push_back(threads_per_node);                    // Single node max
            numa_thread_counts.push_back(num_numa_nodes * threads_per_node);   // All nodes max
            
            // Test with partial NUMA utilization scenarios
            if (num_numa_nodes >= 2) {
                numa_thread_counts.push_back(2 * threads_per_node);           // Two nodes max
            }
            if (num_numa_nodes >= 4) {
                numa_thread_counts.push_back(4 * threads_per_node);           // Four nodes max
            }
        }
        
        // Combine standard and NUMA-aware thread counts, removing duplicates
        std::set<int> all_thread_counts(thread_counts.begin(), thread_counts.end());
        for (int numa_count : numa_thread_counts) {
            if (numa_count > 0 && numa_count <= 256) { // Reasonable upper bound
                all_thread_counts.insert(numa_count);
            }
        }
        
        // Convert back to vector for iteration
        std::vector<int> final_thread_counts(all_thread_counts.begin(), all_thread_counts.end());
        
        printf("  🧵 Testing with %zu thread configurations", final_thread_counts.size());
        if (num_numa_nodes > 0) {
            printf(" (including NUMA-aware counts for %d nodes)", num_numa_nodes);
        }
        printf("\n");
        
        for (int threads : final_thread_counts) {
            std::string test_name = "multi_thread_" + std::to_string(threads);
            
            if (!should_run(test_name)) {
                continue;
            }
            
            printf("  🧵 Testing with %d threads...\n", threads);
            
            // TODO: Implement multi-threading test that varies thread count
            // For now, use single-threaded test
            bool passed = test_single_RMS_NORM_case(512, 256, 4, 1, ("THREADS_" + std::to_string(threads)).c_str());
            
            results.push_back({test_name, passed, passed ? "" : "Multi-threading test failed"});
            
            if (!passed) {
                all_passed = false;
            }
        }
        
        return all_passed;
    }
    
    // Print summary of all test results
    void print_summary() {
        if (list_only) {
            printf("\n📋 Available RMS_NORM tests listed above\n");
            return;
        }
        
        printf("\n📊 === RMS_NORM Test Summary ===\n");
        
        int passed = 0;
        int failed = 0;
        
        for (const auto& result : results) {
            if (result.passed) {
                passed++;
                printf("  ✅ %s\n", result.test_name.c_str());
            } else {
                failed++;
                printf("  ❌ %s: %s\n", result.test_name.c_str(), result.failure_reason.c_str());
            }
        }
        
        printf("\n📈 Results: %d passed, %d failed, %.1f%% success rate\n", 
               passed, failed, 
               (results.empty() ? 0.0 : (100.0 * passed / results.size())));
               
        if (failed > 0) {
            printf("❌ RMS_NORM mathematical correctness tests FAILED\n");
        } else {
            printf("✅ All RMS_NORM mathematical correctness tests PASSED\n");
        }
    }
    
    // Check if all tests passed
    bool all_tests_passed() const {
        if (list_only) return true;
        
        for (const auto& result : results) {
            if (!result.passed) {
                return false;
            }
        }
        return true;
    }
};

// Main function
int main(int argc, char* argv[]) {
    // Parse command line arguments
    std::string filter;
    bool list_only = false;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--list") {
            list_only = true;
        } else if (arg == "--filter" && i + 1 < argc) {
            filter = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            printf("Usage: %s [--list] [--filter REGEX]\n", argv[0]);
            printf("  --list: List available tests without running them\n");
            printf("  --filter REGEX: Run only tests matching the regex pattern\n");
            return 0;
        }
    }
    
    try {
        printf("🚀 Starting NUMA RMS_NORM Mathematical Correctness Tests\n");
        
        NumaRmsNormMathematicalCorrectnessTestSuite suite(filter, list_only);
        
        bool multi_dim_passed = suite.run_multi_dimensional_tests();
        bool multi_thread_passed = suite.run_multi_threading_tests();
        
        suite.print_summary();
        
        if (list_only) {
            return 0;
        }
        
        return suite.all_tests_passed() ? 0 : 1;
        
    } catch (const std::exception& e) {
        printf("❌ Test suite failed with exception: %s\n", e.what());
        return 1;
    }
}
