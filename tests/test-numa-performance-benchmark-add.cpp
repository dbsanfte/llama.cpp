/**
 * NUMA Performance Benchmark Test for ADD Operation
 * 
 * This test benchmarks the performance of NUMA ADD kernels against the fallback CPU implementation.
 * It provides comprehensive performance analysis across different tensor sizes and thread configurations.
 * 
 * Based on: test-numa-performance-benchmark-template.cpp
 * Operation: Element-wise addition (GGML_OP_ADD)
 * 
 * BENCHMARK CHARACTERISTICS:
 * - Memory-bound operation (limited by memory bandwidth)
 * - Linear memory access pattern (cache-friendly)
 * - Highly parallelizable (embarrassingly parallel)
 * - Expected NUMA benefits: High (data-parallel scaling)
 * 
 * PERFORMANCE EXPECTATIONS:
 * - NUMA kernels should show significant speedup on multi-socket systems
 * - Memory bandwidth should be primary limiting factor
 * - Scaling should be near-linear with number of NUMA nodes
 * - Small tensors may show overhead, large tensors should show strong benefits
 */

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <chrono>
#include <numeric>
#include <iomanip>

// GGML includes
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-numa-executor.h"
#include "ggml-numa-coordinator.h"

// Force execution control functions
extern void ggml_numa_set_dispatch_enabled(bool enabled);
extern bool ggml_numa_get_dispatch_enabled(void);

// Benchmark result structure with comprehensive statistics
struct BenchmarkResult {
    std::string test_name;
    std::string execution_path;  // "NUMA" or "Fallback"
    
    // Timing statistics (microseconds)
    double min_time_us;
    double avg_time_us; 
    double max_time_us;
    double stddev_time_us;
    
    // Performance metrics
    double throughput_gbps;      // GB/s for memory-bound operations
    double throughput_gflops;    // GFLOP/s for compute-bound operations
    double memory_bandwidth_gbps; // Actual memory bandwidth utilization
    
    // Test configuration
    int num_runs;
    int num_threads;
    size_t tensor_size_bytes;
    size_t total_elements;
    
    // Success indicators
    bool completed_successfully;
    std::string failure_reason;
    
    BenchmarkResult() : min_time_us(0), avg_time_us(0), max_time_us(0), stddev_time_us(0),
                       throughput_gbps(0), throughput_gflops(0), memory_bandwidth_gbps(0),
                       num_runs(0), num_threads(0), tensor_size_bytes(0), total_elements(0),
                       completed_successfully(false) {}
};

class NumaAddPerformanceBenchmarkSuite {
private:
    std::vector<BenchmarkResult> results;
    bool verbose_output;
    
    // Performance test configuration
    static constexpr int WARMUP_RUNS = 3;
    static constexpr int BENCHMARK_RUNS = 10;
    static constexpr int MIN_BENCHMARK_TIME_MS = 100;  // Minimum benchmark duration
    
    // High-resolution timing utilities
    using high_res_clock = std::chrono::high_resolution_clock;
    using time_point = std::chrono::time_point<high_res_clock>;
    
    time_point get_time() const {
        return high_res_clock::now();
    }
    
    double get_duration_us(const time_point& start, const time_point& end) const {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return static_cast<double>(duration.count());
    }
    
    // Statistical analysis utilities
    double calculate_mean(const std::vector<double>& values) const {
        if (values.empty()) return 0.0;
        return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    }
    
    double calculate_stddev(const std::vector<double>& values, double mean) const {
        if (values.size() <= 1) return 0.0;
        double variance = 0.0;
        for (double value : values) {
            variance += (value - mean) * (value - mean);
        }
        return std::sqrt(variance / (values.size() - 1));
    }
    
    // Memory and compute throughput calculations
    double calculate_memory_throughput_gbps(size_t bytes_accessed, double time_us) const {
        if (time_us <= 0) return 0.0;
        double bytes_per_second = bytes_accessed / (time_us / 1e6);
        return bytes_per_second / (1024.0 * 1024.0 * 1024.0);  // Convert to GB/s
    }
    
    double calculate_compute_throughput_gflops(size_t operations, double time_us) const {
        if (time_us <= 0) return 0.0;
        double ops_per_second = operations / (time_us / 1e6);
        return ops_per_second / 1e9;  // Convert to GFLOP/s
    }
    
    // Force execution path control
    void force_numa_execution() {
        ggml_numa_set_dispatch_enabled(true);
        if (verbose_output) {
            printf("    🎯 Forcing NUMA kernel execution path\n");
        }
    }
    
    void force_fallback_execution() {
        ggml_numa_set_dispatch_enabled(false);
        if (verbose_output) {
            printf("    🔧 Forcing CPU fallback execution path\n");
        }
    }
    
    void restore_default_execution() {
        ggml_numa_set_dispatch_enabled(true);  // Default: NUMA enabled
        if (verbose_output) {
            printf("    ↩️  Restored default execution path\n");
        }
    }
    
    // Create tensors for ADD operation
    struct ggml_tensor* create_add_tensors(struct ggml_context* ctx, int dim1, int dim2, int dim3, 
                                          struct ggml_tensor** input_a, struct ggml_tensor** input_b) {
        // Create two input tensors for element-wise addition
        *input_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim1, dim2, dim3);
        *input_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim1, dim2, dim3);
        
        if (!*input_a || !*input_b) {
            return nullptr;
        }
        
        // Create ADD operation
        struct ggml_tensor* result = ggml_add(ctx, *input_a, *input_b);
        
        if (verbose_output) {
            printf("      Created ADD operation: [%d×%d×%d] + [%d×%d×%d] → [%d×%d×%d]\n",
                   dim1, dim2, dim3, dim1, dim2, dim3, dim1, dim2, dim3);
        }
        
        return result;
    }
    
    // Initialize test tensors with deterministic data for consistent benchmarking
    void initialize_tensors(struct ggml_tensor* input_a, struct ggml_tensor* input_b) {
        if (input_a) {
            float* a_data = (float*)ggml_get_data(input_a);
            size_t a_elements = ggml_nelements(input_a);
            for (size_t i = 0; i < a_elements; i++) {
                a_data[i] = 0.1f + (i % 97) * 0.01f;  // Deterministic pattern
            }
        }
        
        if (input_b) {
            float* b_data = (float*)ggml_get_data(input_b);
            size_t b_elements = ggml_nelements(input_b);
            for (size_t i = 0; i < b_elements; i++) {
                b_data[i] = 0.2f + (i % 73) * 0.01f;  // Different deterministic pattern
            }
        }
    }
    
    // Calculate total memory access for ADD operation
    size_t calculate_memory_access_bytes(struct ggml_tensor* output, struct ggml_tensor* input_a, struct ggml_tensor* input_b) {
        size_t total_bytes = 0;
        
        // ADD operation memory access pattern:
        // - Read input_a: full tensor
        // - Read input_b: full tensor  
        // - Write output: full tensor
        // Total: 3 × tensor_size
        
        if (output) {
            total_bytes += ggml_nbytes(output);  // Write output
        }
        if (input_a) {
            total_bytes += ggml_nbytes(input_a);  // Read input A
        }
        if (input_b) {
            total_bytes += ggml_nbytes(input_b);  // Read input B
        }
        
        return total_bytes;
    }
    
    // Calculate FLOP count for ADD operation
    size_t calculate_operation_flops(struct ggml_tensor* output, struct ggml_tensor* input_a, struct ggml_tensor* input_b) {
        // ADD operation: 1 floating-point addition per output element
        return ggml_nelements(output);
    }
    
    // Benchmark a single ADD operation case with specific execution path
    BenchmarkResult benchmark_single_case(const std::string& test_name, const std::string& execution_path,
                                         int dim1, int dim2, int dim3, int num_threads, const char* size_label) {
        BenchmarkResult result;
        result.test_name = test_name;
        result.execution_path = execution_path;
        result.num_threads = num_threads;
        
        if (verbose_output) {
            printf("    📊 Benchmarking %s: %s execution [%dx%dx%d] (threads=%d)\n", 
                   size_label, execution_path.c_str(), dim1, dim2, dim3, num_threads);
        }
        
        // Create test context with sufficient memory
        struct ggml_init_params params;
        params.mem_size = std::max((size_t)(512 * 1024 * 1024), (size_t)(dim1 * dim2 * dim3) * sizeof(float) * 8);
        params.mem_buffer = nullptr;
        params.no_alloc = false;
        
        struct ggml_context* test_ctx = ggml_init(params);
        if (!test_ctx) {
            result.failure_reason = "Failed to create test context";
            return result;
        }
        
        try {
            // Create tensors for ADD operation
            struct ggml_tensor* input_a = nullptr;
            struct ggml_tensor* input_b = nullptr;
            struct ggml_tensor* output = create_add_tensors(test_ctx, dim1, dim2, dim3, &input_a, &input_b);
            
            if (!output) {
                result.failure_reason = "Failed to create ADD operation tensors";
                ggml_free(test_ctx);
                return result;
            }
            
            // Initialize tensors with test data
            initialize_tensors(input_a, input_b);
            
            // Calculate performance metrics
            result.tensor_size_bytes = ggml_nbytes(output);
            result.total_elements = ggml_nelements(output);
            size_t memory_access_bytes = calculate_memory_access_bytes(output, input_a, input_b);
            size_t operation_flops = calculate_operation_flops(output, input_a, input_b);
            
            if (verbose_output) {
                printf("      Tensor size: %zu bytes (%zu elements)\n", result.tensor_size_bytes, result.total_elements);
                printf("      Memory access: %zu bytes (%.2f GB)\n", memory_access_bytes, memory_access_bytes / (1024.0*1024.0*1024.0));
                printf("      Operations: %zu FLOPs\n", operation_flops);
            }
            
            // Set execution path
            if (execution_path == "NUMA") {
                force_numa_execution();
            } else {
                force_fallback_execution();
            }
            
            // Warmup runs
            for (int i = 0; i < WARMUP_RUNS; i++) {
                struct ggml_compute_params compute_params = {0, num_threads, 0, nullptr};
                enum ggml_status status = ggml_numa_executor_execute_tensor(output, (struct ggml_cplan*)&compute_params);
                if (status != GGML_STATUS_SUCCESS) {
                    result.failure_reason = "Warmup execution failed";
                    restore_default_execution();
                    ggml_free(test_ctx);
                    return result;
                }
            }
            
            // Benchmark runs
            std::vector<double> run_times_us;
            run_times_us.reserve(BENCHMARK_RUNS);
            
            for (int run = 0; run < BENCHMARK_RUNS; run++) {
                time_point start_time = get_time();
                
                struct ggml_compute_params compute_params = {0, num_threads, 0, nullptr};
                enum ggml_status status = ggml_numa_executor_execute_tensor(output, (struct ggml_cplan*)&compute_params);
                
                time_point end_time = get_time();
                
                if (status != GGML_STATUS_SUCCESS) {
                    result.failure_reason = "Benchmark execution failed on run " + std::to_string(run);
                    restore_default_execution();
                    ggml_free(test_ctx);
                    return result;
                }
                
                double run_time_us = get_duration_us(start_time, end_time);
                run_times_us.push_back(run_time_us);
                
                if (verbose_output && (run == 0 || run == BENCHMARK_RUNS - 1)) {
                    printf("      Run %d: %.2f μs\n", run + 1, run_time_us);
                }
            }
            
            // Calculate statistics
            result.num_runs = BENCHMARK_RUNS;
            result.min_time_us = *std::min_element(run_times_us.begin(), run_times_us.end());
            result.max_time_us = *std::max_element(run_times_us.begin(), run_times_us.end());
            result.avg_time_us = calculate_mean(run_times_us);
            result.stddev_time_us = calculate_stddev(run_times_us, result.avg_time_us);
            
            // Calculate performance metrics using minimum time (best case performance)
            result.throughput_gbps = calculate_memory_throughput_gbps(memory_access_bytes, result.min_time_us);
            result.throughput_gflops = calculate_compute_throughput_gflops(operation_flops, result.min_time_us);
            result.memory_bandwidth_gbps = result.throughput_gbps;  // Same for memory-bound operations
            
            result.completed_successfully = true;
            
            if (verbose_output) {
                printf("      Results: %.2f μs avg (±%.2f), %.2f GB/s, %.2f GFLOP/s\n",
                       result.avg_time_us, result.stddev_time_us, result.throughput_gbps, result.throughput_gflops);
            }
            
        } catch (const std::exception& e) {
            result.failure_reason = std::string("Exception: ") + e.what();
        }
        
        restore_default_execution();
        ggml_free(test_ctx);
        return result;
    }
    
public:
    NumaAddPerformanceBenchmarkSuite(bool verbose = false) : verbose_output(verbose) {}
    
    void benchmark_add_performance() {
        printf("--- Performance Benchmark: ADD (NUMA vs Fallback) ---\n");
        printf("Comparing NUMA ADD kernel performance against CPU fallback implementation...\n");
        printf("ADD operation characteristics: Memory-bound, linear access, highly parallelizable\n");
        printf("Testing across multiple tensor sizes and thread configurations\n\n");
        
        // Define test dimensions optimized for ADD operation
        struct {
            int dim1, dim2, dim3;
            const char* label;
        } test_cases[] = {
            {32, 32, 16, "TINY"},        // ~16K elements (~64KB) - L1 cache friendly
            {64, 64, 32, "SMALL"},       // ~128K elements (~512KB) - L2 cache friendly  
            {128, 128, 64, "MEDIUM"},    // ~1M elements (~4MB) - L3 cache size
            {256, 256, 128, "LARGE"},    // ~8M elements (~32MB) - Memory-bound
            {512, 512, 256, "HUGE"}      // ~64M elements (~256MB) - Large memory-bound
        };
        
        // Thread strategies for testing coordinator scaling
        int thread_strategies[] = {1, 2, 4, 8};
        int num_strategies = sizeof(thread_strategies) / sizeof(thread_strategies[0]);
        int num_test_cases = sizeof(test_cases) / sizeof(test_cases[0]);
        
        printf("  🎯 Testing %d tensor sizes with %d thread configurations (%d total benchmarks per execution path)\n\n", 
               num_test_cases, num_strategies, num_test_cases * num_strategies);
        
        // Run benchmarks for each test case and thread configuration
        for (int case_idx = 0; case_idx < num_test_cases; case_idx++) {
            printf("  📏 Benchmarking %s tensors (%dx%dx%d, %.1f MB):\n", 
                   test_cases[case_idx].label, 
                   test_cases[case_idx].dim1, 
                   test_cases[case_idx].dim2, 
                   test_cases[case_idx].dim3,
                   (test_cases[case_idx].dim1 * test_cases[case_idx].dim2 * test_cases[case_idx].dim3 * sizeof(float)) / (1024.0*1024.0));
            
            for (int strategy_idx = 0; strategy_idx < num_strategies; strategy_idx++) {
                int num_threads = thread_strategies[strategy_idx];
                
                // Benchmark NUMA execution
                BenchmarkResult numa_result = benchmark_single_case(
                    "ADD", "NUMA",
                    test_cases[case_idx].dim1, test_cases[case_idx].dim2, test_cases[case_idx].dim3,
                    num_threads, test_cases[case_idx].label);
                
                // Benchmark fallback execution  
                BenchmarkResult fallback_result = benchmark_single_case(
                    "ADD", "Fallback",
                    test_cases[case_idx].dim1, test_cases[case_idx].dim2, test_cases[case_idx].dim3,
                    num_threads, test_cases[case_idx].label);
                
                // Store results
                results.push_back(numa_result);
                results.push_back(fallback_result);
                
                // Calculate and display speedup
                if (numa_result.completed_successfully && fallback_result.completed_successfully) {
                    double speedup = fallback_result.min_time_us / numa_result.min_time_us;
                    double bandwidth_improvement = numa_result.throughput_gbps / fallback_result.throughput_gbps;
                    const char* speedup_color = speedup >= 1.5 ? "🚀" : speedup >= 1.2 ? "✅" : speedup >= 1.0 ? "🔷" : "⚠️";
                    
                    printf("    %s Threads=%d: NUMA=%.2fμs (%.2fGB/s), Fallback=%.2fμs (%.2fGB/s), Speedup=%.2fx\n",
                           speedup_color, num_threads, 
                           numa_result.min_time_us, numa_result.throughput_gbps,
                           fallback_result.min_time_us, fallback_result.throughput_gbps, 
                           speedup);
                } else {
                    printf("    ❌ Threads=%d: Benchmark failed - NUMA:%s, Fallback:%s\n",
                           num_threads, 
                           numa_result.completed_successfully ? "OK" : numa_result.failure_reason.c_str(),
                           fallback_result.completed_successfully ? "OK" : fallback_result.failure_reason.c_str());
                }
            }
            printf("\n");
        }
    }
    
    void print_performance_summary() {
        printf("=== ADD OPERATION PERFORMANCE BENCHMARK SUMMARY ===\n\n");
        
        // Group results by test configuration for analysis
        std::vector<std::pair<BenchmarkResult, BenchmarkResult>> comparisons;
        
        for (size_t i = 0; i < results.size(); i += 2) {
            if (i + 1 < results.size() && 
                results[i].execution_path == "NUMA" && 
                results[i + 1].execution_path == "Fallback") {
                comparisons.push_back({results[i], results[i + 1]});
            }
        }
        
        if (comparisons.empty()) {
            printf("❌ No valid benchmark comparisons found\n");
            return;
        }
        
        printf("📊 Performance Summary for %zu test configurations:\n\n", comparisons.size());
        printf("%-12s %-10s %-12s %-10s %-12s %-12s %-10s\n", 
               "Threads", "NUMA(μs)", "Fallback(μs)", "Speedup", "NUMA(GB/s)", "Fall(GB/s)", "BW Ratio");
        printf("%-12s %-10s %-12s %-10s %-12s %-12s %-10s\n", 
               "--------", "--------", "-----------", "-------", "---------", "---------", "--------");
        
        double total_speedup = 0.0;
        double total_bandwidth_ratio = 0.0;
        int successful_comparisons = 0;
        double best_speedup = 0.0;
        double worst_speedup = std::numeric_limits<double>::max();
        double best_bandwidth = 0.0;
        
        for (const auto& comparison : comparisons) {
            const auto& numa = comparison.first;
            const auto& fallback = comparison.second;
            
            if (numa.completed_successfully && fallback.completed_successfully) {
                double speedup = fallback.min_time_us / numa.min_time_us;
                double bandwidth_ratio = numa.throughput_gbps / fallback.throughput_gbps;
                total_speedup += speedup;
                total_bandwidth_ratio += bandwidth_ratio;
                successful_comparisons++;
                best_speedup = std::max(best_speedup, speedup);
                worst_speedup = std::min(worst_speedup, speedup);
                best_bandwidth = std::max(best_bandwidth, numa.throughput_gbps);
                
                const char* status = speedup >= 1.5 ? "🚀" : speedup >= 1.2 ? "✅" : speedup >= 1.0 ? "🔷" : "⚠️";
                
                printf("%-12s%d %-10.2f %-12.2f %-10.2fx %c %-12.2f %-12.2f %-10.2fx\n",
                       "", numa.num_threads, numa.min_time_us, fallback.min_time_us, speedup, 
                       status[0], numa.throughput_gbps, fallback.throughput_gbps, bandwidth_ratio);
            }
        }
        
        printf("\n");
        
        if (successful_comparisons > 0) {
            double avg_speedup = total_speedup / successful_comparisons;
            double avg_bandwidth_ratio = total_bandwidth_ratio / successful_comparisons;
            printf("🎯 ADD OPERATION PERFORMANCE ANALYSIS:\n");
            printf("  Average speedup: %.2fx\n", avg_speedup);
            printf("  Best speedup: %.2fx\n", best_speedup);
            printf("  Worst speedup: %.2fx\n", worst_speedup);
            printf("  Average bandwidth improvement: %.2fx\n", avg_bandwidth_ratio);
            printf("  Peak bandwidth achieved: %.2f GB/s\n", best_bandwidth);
            printf("  Successful tests: %d/%zu\n", successful_comparisons, comparisons.size());
            
            if (avg_speedup >= 1.5) {
                printf("  ✅ NUMA ADD kernel shows excellent performance improvement!\n");
                printf("     This confirms ADD operation benefits significantly from NUMA data-parallel execution.\n");
            } else if (avg_speedup >= 1.2) {
                printf("  ✅ NUMA ADD kernel shows good performance improvement\n");
                printf("     Memory-bound nature of ADD operation benefits from NUMA memory access optimization.\n");
            } else if (avg_speedup >= 1.0) {
                printf("  🔷 NUMA ADD kernel performs as well as fallback\n");
                printf("     No performance degradation, but room for further optimization.\n");
            } else {
                printf("  ⚠️  NUMA ADD kernel underperforms fallback - optimization needed\n");
                printf("     Consider investigating NUMA work distribution overhead vs. kernel efficiency.\n");
            }
        } else {
            printf("❌ No successful benchmark comparisons completed\n");
        }
        
        printf("\n");
    }
    
    int run_all_benchmarks() {
        printf("🚀 NUMA ADD Performance Benchmark Suite\n");
        printf("========================================\n");
        printf("Operation: Element-wise addition (GGML_OP_ADD)\n");
        printf("Characteristics: Memory-bound, linear access, highly parallelizable\n");
        printf("Benchmark runs per test: %d (after %d warmup runs)\n", BENCHMARK_RUNS, WARMUP_RUNS);
        printf("Verbose output: %s\n", verbose_output ? "enabled" : "disabled");
        printf("\n");
        
        try {
            benchmark_add_performance();
            print_performance_summary();
            
            // Determine overall success
            int successful_tests = 0;
            int total_tests = 0;
            
            for (const auto& result : results) {
                total_tests++;
                if (result.completed_successfully) {
                    successful_tests++;
                }
            }
            
            printf("🏁 Benchmark completion: %d/%d tests successful\n", successful_tests, total_tests);
            
            if (successful_tests == total_tests) {
                printf("✅ All ADD performance benchmarks completed successfully!\n");
                return 0;
            } else {
                printf("⚠️  Some ADD performance benchmarks failed\n");
                return 1;
            }
            
        } catch (const std::exception& e) {
            printf("❌ ADD benchmark suite failed with exception: %s\n", e.what());
            return 1;
        }
    }
};

// Command line argument processing
bool should_show_verbose_output(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
            return true;
        }
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [--verbose] [--help]\n", argv[0]);
            printf("\n");
            printf("NUMA Performance Benchmark for ADD Operation\n");
            printf("Compares NUMA ADD kernel performance against CPU fallback implementation.\n");
            printf("\n");
            printf("ADD Operation Characteristics:\n");
            printf("  - Memory-bound operation (limited by memory bandwidth)\n");
            printf("  - Linear memory access pattern (cache-friendly)\n");
            printf("  - Highly parallelizable (embarrassingly parallel)\n");
            printf("  - Expected NUMA benefits: High (data-parallel scaling)\n");
            printf("\n");
            printf("Options:\n");
            printf("  --verbose, -v    Show detailed benchmark output\n");
            printf("  --help, -h       Show this help message\n");
            printf("\n");
            exit(0);
        }
    }
    return false;
}

int main(int argc, char** argv) {
    bool verbose = should_show_verbose_output(argc, argv);
    
    NumaAddPerformanceBenchmarkSuite benchmark_suite(verbose);
    return benchmark_suite.run_all_benchmarks();
}
