/**
 * NUMA Mathematical Correctness Test: CPY Operation
 * 
 * This test validates that NUMA-parallel CPY operation produces
 * mathematically identical results to the reference implementation.
 * 
 * CPY operation copies data from source to destination tensor, potentially
 * with type conversion. The operation must handle:
 * - Same-type copying (optimized memcpy path)
 * - Type conversion (F32<->F16, quantized types)
 * - Contiguous and non-contiguous memory layouts
 * - Multi-dimensional tensors with different strides
 * 
 * Test Strategy:
 * - Multi-dimensional tensors from tiny to large scales
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

// GGML includes
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-numa-executor.h"
#include "ggml-numa-simple-coordinator.h"
#include "ggml-cpu/binary-ops.h"

// Forward declarations
extern "C" void ggml_compute_forward_dup(const struct ggml_compute_params * params, struct ggml_tensor * dst);
extern "C" void ggml_numa_clear_dispatch_override(void);
extern "C" void ggml_numa_set_dispatch_enabled(bool enabled);

// Test result structure
struct TestResult {
    std::string test_name;
    bool passed;
    std::string failure_reason;
};

class NumaCpyMathematicalCorrectnessTestSuite {
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
    
    // Utility function to compare float arrays with detailed error reporting
    bool compare_float_arrays(const float* numa_data, const float* ref_data, int count, const char* operation_name, ggml_type type = GGML_TYPE_F32) {
        bool all_match = true;
        int error_count = 0;
        double max_abs_error = 0.0;
        double max_rel_error = 0.0;
        
        // Set tolerance based on data type
        double abs_tolerance, rel_tolerance;
        switch (type) {
            case GGML_TYPE_F16:
                abs_tolerance = 1e-3;  // F16 has limited precision
                rel_tolerance = 1e-3;
                break;
            case GGML_TYPE_F32:
            default:
                abs_tolerance = 1e-6;  // F32 high precision
                rel_tolerance = 1e-6;
                break;
        }
        
        for (int i = 0; i < count; i++) {
            double numa_val = numa_data[i];
            double ref_val = ref_data[i];
            double abs_error = fabs(numa_val - ref_val);
            double rel_error = (ref_val != 0.0) ? fabs(abs_error / ref_val) : abs_error;
            
            max_abs_error = std::max(max_abs_error, abs_error);
            max_rel_error = std::max(max_rel_error, rel_error);
            
            if (abs_error > abs_tolerance && rel_error > rel_tolerance) {
                if (error_count < 5) { // Limit error reporting to first 5 mismatches
                    printf("        ❌ %s: Mismatch at index %d: NUMA=%.6f, Reference=%.6f, AbsErr=%.6e, RelErr=%.6e\n",
                           operation_name, i, numa_val, ref_val, abs_error, rel_error);
                }
                error_count++;
                all_match = false;
            }
        }
        
        if (!all_match) {
            printf("        ❌ %s: Total mismatches: %d/%d (%.2f%%), MaxAbsErr=%.6e, MaxRelErr=%.6e\n", 
                   operation_name, error_count, count, 100.0 * error_count / count, max_abs_error, max_rel_error);
        } else {
            printf("        ✅ %s: All values match (MaxAbsErr=%.6e, MaxRelErr=%.6e)\n", 
                   operation_name, max_abs_error, max_rel_error);
        }
        
        return all_match;
    }

    // Test function for a single CPY operation case
    bool test_single_CPY_case(int dim1, int dim2, int dim3, int num_threads, ggml_type src_type, ggml_type dst_type, const char* size_label) {
        printf("    🧪 Testing CPY %s [%d×%d×%d] with %d threads (%s->%s)\n", 
               size_label, dim1, dim2, dim3, num_threads, ggml_type_name(src_type), ggml_type_name(dst_type));
        
        try {
            // Create contexts
            size_t mem_size = 512 * 1024 * 1024;  // 512MB
            struct ggml_init_params params = {
                /*.mem_size   =*/ mem_size,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ false,
            };
            
            struct ggml_context* ctx = ggml_init(params);
            struct ggml_context* ref_ctx = ggml_init(params);
            if (!ctx || !ref_ctx) {
                printf("      ❌ Failed to create GGML contexts\n");
                return false;
            }
            
            // Create input tensor and fill with deterministic test data
            struct ggml_tensor* input = ggml_new_tensor_3d(ctx, src_type, dim1, dim2, dim3);
            struct ggml_tensor* ref_input = ggml_new_tensor_3d(ref_ctx, src_type, dim1, dim2, dim3);
            if (!input || !ref_input) {
                printf("      ❌ Failed to create input tensors\n");
                ggml_free(ctx);
                ggml_free(ref_ctx);
                return false;
            }
            
            // Fill input with test data pattern - both contexts get identical data
            const int total_elements = dim1 * dim2 * dim3;
            if (src_type == GGML_TYPE_F32) {
                float* input_data = (float*)ggml_get_data(input);
                float* ref_input_data = (float*)ggml_get_data(ref_input);
                for (int i = 0; i < total_elements; i++) {
                    float val = sinf(i * 0.1f) + cosf(i * 0.05f); // Deterministic test pattern
                    input_data[i] = val;
                    ref_input_data[i] = val;
                }
            } else if (src_type == GGML_TYPE_F16) {
                ggml_fp16_t* input_data = (ggml_fp16_t*)ggml_get_data(input);
                ggml_fp16_t* ref_input_data = (ggml_fp16_t*)ggml_get_data(ref_input);
                for (int i = 0; i < total_elements; i++) {
                    float val = sinf(i * 0.1f) + cosf(i * 0.05f);
                    ggml_fp16_t fp16_val = ggml_fp32_to_fp16(val);
                    input_data[i] = fp16_val;
                    ref_input_data[i] = fp16_val;
                }
            }
            
            // Create destination tensors for NUMA and reference tests
            struct ggml_tensor* numa_dest = ggml_new_tensor_3d(ctx, dst_type, dim1, dim2, dim3);
            struct ggml_tensor* ref_dest = ggml_new_tensor_3d(ref_ctx, dst_type, dim1, dim2, dim3);
            
            if (!numa_dest || !ref_dest) {
                printf("      ❌ Failed to create destination tensors\n");
                ggml_free(ctx);
                ggml_free(ref_ctx);
                return false;
            }
            
            // Create CPY operations
            struct ggml_tensor* numa_result = ggml_cpy(ctx, input, numa_dest);
            struct ggml_tensor* ref_result = ggml_cpy(ref_ctx, ref_input, ref_dest);
            
            if (!numa_result || !ref_result) {
                printf("      ❌ Failed to create CPY operations\n");
                ggml_free(ctx);
                ggml_free(ref_ctx);
                return false;
            }
            
            // Execute NUMA version with computation graph
            struct ggml_cgraph* numa_gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(numa_gf, numa_result);
            
            struct ggml_cplan numa_plan = ggml_graph_plan(numa_gf, num_threads, nullptr);
            if (numa_plan.work_size > 0) {
                numa_plan.work_data = (uint8_t*)malloc(numa_plan.work_size);
                if (!numa_plan.work_data) {
                    printf("      ❌ Failed to allocate NUMA work buffer\n");
                    ggml_free(ctx);
                    ggml_free(ref_ctx);
                    return false;
                }
            }
            
            // Enable NUMA dispatch and execute
            ggml_numa_set_dispatch_enabled(true);
            ggml_graph_compute(numa_gf, &numa_plan);
            
            // Execute reference version with dispatch disabled
            ggml_numa_clear_dispatch_override();
            ggml_numa_set_dispatch_enabled(false);
            
            struct ggml_cgraph* ref_gf = ggml_new_graph(ref_ctx);
            ggml_build_forward_expand(ref_gf, ref_result);
            
            // Execute reference computation (single-threaded)
            ggml_graph_compute_with_ctx(ref_ctx, ref_gf, 1);
            
            // Re-enable dispatch for subsequent tests
            ggml_numa_set_dispatch_enabled(true);
            
            // Compare results - convert to F32 for comparison if needed
            std::vector<float> numa_f32_data(total_elements);
            std::vector<float> ref_f32_data(total_elements);
            
            // Convert NUMA result to F32
            if (dst_type == GGML_TYPE_F32) {
                memcpy(numa_f32_data.data(), ggml_get_data(numa_dest), total_elements * sizeof(float));
            } else if (dst_type == GGML_TYPE_F16) {
                const ggml_fp16_t* src = (const ggml_fp16_t*)ggml_get_data(numa_dest);
                for (int i = 0; i < total_elements; i++) {
                    numa_f32_data[i] = ggml_fp16_to_fp32(src[i]);
                }
            }
            
            // Convert reference result to F32
            if (dst_type == GGML_TYPE_F32) {
                memcpy(ref_f32_data.data(), ggml_get_data(ref_dest), total_elements * sizeof(float));
            } else if (dst_type == GGML_TYPE_F16) {
                const ggml_fp16_t* src = (const ggml_fp16_t*)ggml_get_data(ref_dest);
                for (int i = 0; i < total_elements; i++) {
                    ref_f32_data[i] = ggml_fp16_to_fp32(src[i]);
                }
            }
            
            bool case_passed = compare_float_arrays(numa_f32_data.data(), ref_f32_data.data(), total_elements, "CPY", dst_type);
            
            if (case_passed) {
                printf("      ✅ %s: CPY results match\n", size_label);
            } else {
                printf("      ❌ %s: CPY results do not match\n", size_label);
            }
            
            // Cleanup
            if (numa_plan.work_data) free(numa_plan.work_data);
            ggml_free(ctx);
            ggml_free(ref_ctx);
            return case_passed;
            
        } catch (const std::exception& e) {
            printf("      ❌ CPY %s exception: %s\n", size_label, e.what());
            return false;
        }
    }

    // Test mathematical equivalence across different sizes and thread counts
    void test_CPY_mathematical_equivalence() {
        if (!should_run("CPY_mathematical_equivalence")) return;
        
        printf("--- Test: CPY Mathematical Equivalence (Multi-Dimensional) ---\n");
        printf("Testing NUMA parallel CPY vs serial reference implementation...\n");
        
        struct TestCase {
            int dim1, dim2, dim3;
            const char* size_label;
        };
        
        const std::vector<TestCase> test_cases = {
            // Small tests for quick validation
            {16, 16, 1, "Small 2D"},
            {32, 32, 1, "Medium 2D"},
            {8, 8, 8, "Small 3D"},
            
            // Medium tests for realistic workloads
            {64, 64, 1, "Large 2D"},
            {16, 16, 16, "Medium 3D"},
            {128, 32, 1, "Wide 2D"},
            
            // Large tests for performance validation
            {128, 128, 1, "Very Large 2D"},
            {32, 32, 32, "Large 3D"},
        };
        
        const std::vector<int> thread_counts = {1, 2, 4, 6, 8};
        
        // Test different type combinations
        const std::vector<std::pair<ggml_type, ggml_type>> type_combinations = {
            {GGML_TYPE_F32, GGML_TYPE_F32}, // Same type (most common)
            {GGML_TYPE_F32, GGML_TYPE_F16}, // F32 to F16 conversion
            {GGML_TYPE_F16, GGML_TYPE_F32}, // F16 to F32 conversion
            {GGML_TYPE_F16, GGML_TYPE_F16}, // F16 same type
        };
        
        bool overall_test_passed = true;
        const char* failure_reason = nullptr;
        
        for (const auto& type_combo : type_combinations) {
            printf("  Testing type conversion: %s -> %s\n", 
                   ggml_type_name(type_combo.first), ggml_type_name(type_combo.second));
            
            for (const auto& test_case : test_cases) {
                for (int num_threads : thread_counts) {
                    if (list_only) {
                        printf("    Would test: CPY %s [%d×%d×%d] with %d threads (%s->%s)\n", 
                               test_case.size_label, test_case.dim1, test_case.dim2, test_case.dim3, 
                               num_threads, ggml_type_name(type_combo.first), ggml_type_name(type_combo.second));
                        continue;
                    }
                    
                    bool case_passed = test_single_CPY_case(
                        test_case.dim1, test_case.dim2, test_case.dim3, 
                        num_threads, type_combo.first, type_combo.second, test_case.size_label
                    );
                    
                    if (!case_passed) {
                        overall_test_passed = false;
                        failure_reason = "Mathematical equivalence test failed";
                    }
                }
            }
        }
        
        results.push_back({"CPY_mathematical_equivalence", overall_test_passed, failure_reason ? failure_reason : ""});
    }

public:
    void run_all_tests() {
        printf("\n🧪 NUMA CPY Kernel Mathematical Correctness Tests\n");
        printf("=================================================\n");
        
        if (list_only) {
            printf("List mode - showing all tests that would be run:\n");
        }
        
        // Initialize NUMA system
        ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);
        
        // Enable NUMA dispatch
        ggml_numa_set_dispatch_enabled(true);
        
        // Run test suites
        test_CPY_mathematical_equivalence();
        
        if (list_only) {
            printf("List complete - %zu tests would be run.\n", results.size());
            return;
        }
        
        // Print summary
        printf("\n=== TEST SUMMARY ===\n");
        int passed = 0, total = 0;
        
        for (const auto& result : results) {
            total++;
            if (result.passed) {
                passed++;
                printf("✅ %s: PASSED\n", result.test_name.c_str());
            } else {
                printf("❌ %s: FAILED - %s\n", result.test_name.c_str(), result.failure_reason.c_str());
            }
        }
        
        printf("\nResults: %d/%d tests passed (%.1f%%)\n", 
               passed, total, total > 0 ? 100.0f * passed / total : 0.0f);
        
        if (passed == total) {
            printf("\n🎉 ALL CPY TESTS PASSED!\n");
            printf("The NUMA CPY kernel is mathematically correct and ready for production.\n");
        } else {
            printf("\n❌ SOME CPY TESTS FAILED!\n");
            printf("The NUMA CPY kernel has mathematical errors that must be fixed.\n");
        }
    }
    
    void set_filter(const std::string& pattern) {
        name_filter = std::regex(pattern, std::regex::icase);
    }
    
    void set_list_only(bool list) {
        list_only = list;
    }
};

int main(int argc, char* argv[]) {
    NumaCpyMathematicalCorrectnessTestSuite test_suite;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--list") {
            test_suite.set_list_only(true);
        } else if (arg == "--filter" && i + 1 < argc) {
            test_suite.set_filter(argv[++i]);
        } else if (arg == "--help") {
            printf("Usage: %s [--list] [--filter PATTERN]\n", argv[0]);
            printf("  --list: List all tests without running them\n");
            printf("  --filter PATTERN: Run only tests matching the regex pattern\n");
            return 0;
        }
    }
    
    test_suite.run_all_tests();
    
    return 0;
}
