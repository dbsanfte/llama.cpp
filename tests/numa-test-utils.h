#pragma once

#include <cmath>
#include <cstdio>
#include <cstddef>
#include <cstdint>

// Shared utilities for NUMA mathematical correctness testing
namespace NumaTestUtils {

// Comprehensive corruption detection result
struct CorruptionAnalysis {
    bool has_nan;
    bool has_inf;
    bool has_corruption;
    size_t nan_count;
    size_t inf_count;
    size_t valid_count;
    double mean;
    double variance;
    double min_val;
    double max_val;
    size_t first_nan_index;
    size_t first_inf_index;
};

// Enhanced corruption detection with detailed analysis
inline CorruptionAnalysis analyze_tensor_corruption(const float* data, size_t num_elements, 
                                                   const char* tensor_name = "tensor",
                                                   bool verbose = true) {
    CorruptionAnalysis result = {};
    result.first_nan_index = SIZE_MAX;
    result.first_inf_index = SIZE_MAX;
    result.min_val = INFINITY;
    result.max_val = -INFINITY;
    
    if (!data || num_elements == 0) {
        if (verbose) {
            printf("        ⚠️  %s: NULL data or zero elements\n", tensor_name);
        }
        result.has_corruption = true;
        return result;
    }
    
    double sum = 0.0;
    double sum_squares = 0.0;
    
    for (size_t i = 0; i < num_elements; i++) {
        float val = data[i];
        
        if (std::isnan(val)) {
            result.nan_count++;
            if (result.first_nan_index == SIZE_MAX) {
                result.first_nan_index = i;
                result.has_nan = true;
                if (verbose) {
                    printf("        🚨 %s: First NaN detected at index %zu\n", tensor_name, i);
                }
            }
            continue;
        }
        
        if (std::isinf(val)) {
            result.inf_count++;
            if (result.first_inf_index == SIZE_MAX) {
                result.first_inf_index = i;
                result.has_inf = true;
                if (verbose) {
                    printf("        🚨 %s: First infinity detected at index %zu\n", tensor_name, i);
                }
            }
            continue;
        }
        
        // Track valid values for statistics
        result.valid_count++;
        sum += val;
        sum_squares += val * val;
        
        if (val < result.min_val) result.min_val = val;
        if (val > result.max_val) result.max_val = val;
    }
    
    // Calculate statistics for valid values
    if (result.valid_count > 0) {
        result.mean = sum / result.valid_count;
        result.variance = (sum_squares / result.valid_count) - (result.mean * result.mean);
    }
    
    // Determine if corruption exists
    result.has_corruption = result.has_nan || result.has_inf;
    
    // Additional corruption checks
    if (result.valid_count == 0 && num_elements > 0) {
        if (verbose) {
            printf("        🚨 %s: All %zu values are NaN or inf\n", tensor_name, num_elements);
        }
        result.has_corruption = true;
    } else if (result.valid_count < num_elements / 2 && num_elements > 10) {
        if (verbose) {
            printf("        ⚠️  %s: High corruption rate: %zu/%zu values invalid\n", 
                   tensor_name, num_elements - result.valid_count, num_elements);
        }
        result.has_corruption = true;
    }
    
    // Check for suspicious statistical patterns
    if (result.valid_count > 10) {
        if (result.variance > 1e10) {
            if (verbose) {
                printf("        ⚠️  %s: Suspicious high variance: %e\n", tensor_name, result.variance);
            }
        }
        
        if (std::fabs(result.max_val - result.min_val) > 1e6) {
            if (verbose) {
                printf("        ⚠️  %s: Suspicious value range: [%e, %e]\n", 
                       tensor_name, result.min_val, result.max_val);
            }
        }
        
        // CRITICAL: Detect "all zeros" corruption for matrix operations
        // Matrix multiplication with non-zero inputs should not produce all zeros
        if (result.min_val == 0.0 && result.max_val == 0.0 && result.valid_count > 50) {
            if (verbose) {
                printf("        🚨 MATHEMATICAL CORRUPTION: All %zu elements are exactly zero!\n", result.valid_count);
                printf("           This indicates incorrect matrix computation - should have non-zero results.\n");
            }
            result.has_corruption = true;
        }
    }
    
    return result;
}

// Print detailed corruption analysis report
inline void print_corruption_report(const CorruptionAnalysis& analysis, const char* tensor_name = "tensor") {
    printf("    📊 Corruption Analysis for %s:\n", tensor_name);
    printf("       Valid values: %zu\n", analysis.valid_count);
    
    if (analysis.nan_count > 0) {
        printf("       NaN values: %zu (first at index %zu)\n", analysis.nan_count, analysis.first_nan_index);
    }
    
    if (analysis.inf_count > 0) {
        printf("       Inf values: %zu (first at index %zu)\n", analysis.inf_count, analysis.first_inf_index);
    }
    
    if (analysis.valid_count > 0) {
        printf("       Statistics: mean=%.6f, variance=%.6e\n", analysis.mean, analysis.variance);
        printf("       Range: [%.6f, %.6f]\n", analysis.min_val, analysis.max_val);
    }
    
    printf("       Overall: %s\n", analysis.has_corruption ? "🚨 CORRUPTED" : "✅ CLEAN");
}

// Quick corruption check (returns true if corrupted)
inline bool has_corruption(const float* data, size_t num_elements, const char* tensor_name = nullptr) {
    CorruptionAnalysis analysis = analyze_tensor_corruption(data, num_elements, tensor_name, false);
    return analysis.has_corruption;
}

// Initialize deterministic test data for F32 tensors
inline void init_f32_test_data(float* data, size_t num_elements, float base_value = 0.1f, float increment = 0.01f) {
    for (size_t i = 0; i < num_elements; i++) {
        data[i] = base_value + increment * (i % 100);
    }
}

// Validate that two F32 tensors are mathematically equivalent within tolerance
inline bool tensors_equal(const float* a, const float* b, size_t num_elements, 
                         double abs_tolerance = 1e-5, double rel_tolerance = 1e-4,
                         bool verbose = true) {
    if (!a || !b) {
        if (verbose) printf("        ❌ NULL tensor data in comparison\n");
        return false;
    }
    
    size_t error_count = 0;
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    size_t first_error_index = SIZE_MAX;
    
    for (size_t i = 0; i < num_elements; i++) {
        float val_a = a[i];
        float val_b = b[i];
        
        // Check for NaN/inf in either tensor
        if (std::isnan(val_a) || std::isnan(val_b) || std::isinf(val_a) || std::isinf(val_b)) {
            if (!(std::isnan(val_a) && std::isnan(val_b)) && !(std::isinf(val_a) && std::isinf(val_b))) {
                error_count++;
                if (first_error_index == SIZE_MAX) {
                    first_error_index = i;
                    if (verbose) {
                        printf("        🚨 NaN/inf mismatch at index %zu: a=%f, b=%f\n", i, val_a, val_b);
                    }
                }
            }
            continue;
        }
        
        double abs_error = std::fabs(val_a - val_b);
        double rel_error = (std::fabs(val_b) > 1e-10) ? abs_error / std::fabs(val_b) : abs_error;
        
        if (abs_error > abs_tolerance && rel_error > rel_tolerance) {
            error_count++;
            if (first_error_index == SIZE_MAX) {
                first_error_index = i;
                if (verbose) {
                    printf("        🚨 Value mismatch at index %zu: a=%f, b=%f (abs_err=%e, rel_err=%e)\n", 
                           i, val_a, val_b, abs_error, rel_error);
                }
            }
            
            if (abs_error > max_abs_error) max_abs_error = abs_error;
            if (rel_error > max_rel_error) max_rel_error = rel_error;
        }
    }
    
    if (error_count > 0 && verbose) {
        printf("        📊 Comparison summary: %zu/%zu mismatches, max_abs_err=%e, max_rel_err=%e\n",
               error_count, num_elements, max_abs_error, max_rel_error);
    }
    
    return error_count == 0;
}

// Test configuration structure
struct TestConfig {
    bool test_quantized_types;
    bool verbose_output;
    bool force_multi_socket;
    int num_threads;
    double corruption_tolerance;
    double comparison_tolerance;
};

// Default test configuration
inline TestConfig default_test_config() {
    TestConfig config = {};
    config.test_quantized_types = true;
    config.verbose_output = true;
    config.force_multi_socket = true;
    config.num_threads = 4;
    config.corruption_tolerance = 1e-10;
    config.comparison_tolerance = 1e-5;
    return config;
}

} // namespace NumaTestUtils
