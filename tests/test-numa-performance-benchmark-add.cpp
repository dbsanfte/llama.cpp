/**
 * Comprehensive NUMA Performance Benchmark for ADD Operation
 * 
 * This test measures detailed performance comparisons betwee        // For fallback, ensure we use standard GGML computation, not empty operations
        if (!use_numa) {
            // Force NUMA dispatch off and ensure actual computation happens
            ggml_numa_set_dispatch_enabled(false);
            printf("    🔧 DEBUG: Fallback mode - NUMA dispatch disabled, threads=%d\n", thread_count);
        } else {
            ggml_numa_set_dispatch_enabled(true);
            printf("    🔧 DEBUG: NUMA mode - NUMA dispatch enabled, threads=%d\n", thread_count);
        }
        
        // Create computation graph
        struct ggml_cgraph* cgraph = ggml_new_graph(ctx);
        ggml_build_forward_expand(cgraph, result);
        
        // Add detailed threadpool debugging
        printf("    📊 DEBUG: Graph has %d nodes, cplan will use %d threads\n", cgraph->n_nodes, thread_count);llback execution
 * for the ADD operation across multiple tensor sizes and execution strategies:
 * 1. Single-threaded NUMA node 0 vs Fallback single-threaded
 * 2. Multi-threaded NUMA node 0 vs Fallback multi-threaded  
 * 3. Multi-node NUMA vs Fallback multi-threaded
 * 
 * Measures GFLOPS, Memory bandwidth, and provides comprehensive analysis.
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
#include <map>
#include <thread>
#include <unistd.h>      // For dup, dup2, close, STDOUT_FILENO, STDERR_FILENO
#include <fcntl.h>       // For open, O_WRONLY

// GGML includes
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-numa-executor.h"
#include "ggml-numa-simple-coordinator.h"
#include "ggml-numa-perf.h"  // Performance instrumentation

// Test case structure for comprehensive dimensionality testing
struct TestCase {
    int dim1, dim2, dim3;
    const char* name;
    const char* description;
};

// Performance result structure with comprehensive metrics
struct PerformanceResult {
    std::string test_name;
    std::string execution_strategy;
    bool numa_enabled;
    int thread_count;
    double avg_time_ms;
    double min_time_ms;
    double max_time_ms;
    size_t tensor_elements;
    size_t memory_size_mb;
    double throughput_gb_per_sec;
    double gflops;
    double memory_bandwidth_gb_per_sec;
};

class ComprehensiveNumaPerformanceBenchmark {
private:
    std::vector<PerformanceResult> results;
    std::vector<TestCase> test_cases;
    
    // High-resolution timing utilities
    using high_res_clock = std::chrono::high_resolution_clock;
    using time_point = std::chrono::time_point<high_res_clock>;
    
    time_point get_time() const {
        return high_res_clock::now();
    }
    
    double get_duration_ms(const time_point& start, const time_point& end) const {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return static_cast<double>(duration.count()) / 1000.0; // Convert to milliseconds
    }
    
    // Initialize comprehensive test cases with proper dimensionality
    void initialize_test_cases() {
        test_cases = {
            {32, 32, 16, "TINY", "~16K elements (~64KB) - L1 cache friendly"},
            {64, 64, 32, "SMALL", "~128K elements (~512KB) - L2 cache friendly"},  
            {128, 128, 64, "MEDIUM", "~1M elements (~4MB) - L3 cache size"},
            {256, 256, 128, "LARGE", "~8M elements (~32MB) - Memory-bound"},
            {512, 512, 256, "HUGE", "~64M elements (~256MB) - Large memory-bound"}
        };
    }
    
    // Test a single ADD case with comprehensive performance measurement
    bool test_single_ADD_performance(const TestCase& test_case, const std::string& strategy, 
                                   bool use_numa, int thread_count) {
        printf("    ⏱️  Testing %s: %s strategy [%d,%d,%d] (%s, %d threads)\n", 
               test_case.name, strategy.c_str(), test_case.dim1, test_case.dim2, test_case.dim3, 
               use_numa ? "NUMA" : "Fallback", thread_count);
        
        // Create test context with sufficient memory
        struct ggml_init_params params;
        size_t tensor_size = (size_t)test_case.dim1 * test_case.dim2 * test_case.dim3 * sizeof(float);
        params.mem_size = std::max((size_t)(1024 * 1024 * 1024), tensor_size * 8); // At least 1GB
        params.mem_buffer = nullptr;
        params.no_alloc = false;
        
        struct ggml_context* ctx = ggml_init(params);
        if (!ctx) {
            printf("❌ Failed to initialize GGML context\n");
            return false;
        }
        
        // Create tensors with the specified dimensions
        struct ggml_tensor* a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, test_case.dim1, test_case.dim2, test_case.dim3);
        struct ggml_tensor* b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, test_case.dim1, test_case.dim2, test_case.dim3);
        struct ggml_tensor* result = ggml_add(ctx, a, b);
        
        if (!a || !b || !result) {
            printf("❌ Failed to create tensors\n");
            ggml_free(ctx);
            return false;
        }
        
        // Initialize input data with meaningful patterns to prevent optimizations
        float* a_data = (float*)ggml_get_data(a);
        float* b_data = (float*)ggml_get_data(b);
        float* result_data = (float*)ggml_get_data(result);
        size_t total_elements = ggml_nelements(a);
        
        // Use different patterns to prevent optimization
        for (size_t i = 0; i < total_elements; i++) {
            a_data[i] = sinf((float)i * 0.1f);  // Prevent constant folding
            b_data[i] = cosf((float)i * 0.1f);  // Prevent constant folding
        }
        
        // Clear result to ensure computation happens
        memset(result_data, 0, total_elements * sizeof(float));
        
        // Set execution mode
        ggml_numa_set_dispatch_enabled(use_numa);
        
        // For fallback, ensure we use standard GGML computation, not empty operations
        if (!use_numa) {
            // Force NUMA dispatch off and ensure actual computation happens
            ggml_numa_set_dispatch_enabled(false);
            printf("    🔧 DEBUG: Fallback mode - NUMA dispatch disabled, threads=%d\n", thread_count);
        } else {
            ggml_numa_set_dispatch_enabled(true);
            printf("    🔧 DEBUG: NUMA mode - NUMA dispatch enabled, threads=%d\n", thread_count);
        }
        
        // Create computation graph
        struct ggml_cgraph* cgraph = ggml_new_graph(ctx);
        ggml_build_forward_expand(cgraph, result);
        
        // Debug threadpool information
        printf("    📋 DEBUG: Threadpool analysis:\n");
        printf("        - Target threads: %d\n", thread_count);
        printf("        - Hardware concurrency: %d\n", std::thread::hardware_concurrency());
        printf("        - Mode: %s\n", use_numa ? "NUMA" : "Fallback");
        
        // Extended warmup runs to stabilize timing (5 iterations)
        for (int i = 0; i < 5; i++) {
            ggml_graph_compute_with_ctx(ctx, cgraph, thread_count);
            // Add memory barrier to prevent optimization
            __asm__ __volatile__ ("" ::: "memory");
            
            // Validate computation actually occurred during warmup
            if (i == 0) {
                bool computation_verified = false;
                for (size_t j = 0; j < std::min(total_elements, (size_t)100); j++) {
                    if (result_data[j] != 0.0f) {
                        computation_verified = true;
                        break;
                    }
                }
                if (!computation_verified) {
                    printf("⚠️  Warning: No computation detected in %s mode\n", use_numa ? "NUMA" : "Fallback");
                }
            }
        }
        
        // Performance measurement runs with higher precision
        const int num_runs = 20;  // More runs for better statistics
        std::vector<double> times;
        times.reserve(num_runs);
        
        for (int run = 0; run < num_runs; run++) {
            // Clear cache between runs for more realistic timing
            if (run % 5 == 0) {
                // Touch memory to ensure consistent state
                volatile float sum = 0.0f;
                for (size_t i = 0; i < total_elements; i += 1024) {
                    sum += result_data[i];
                }
                __asm__ __volatile__ ("" :: "r" (sum) : "memory");
            }
            
            auto start = get_time();
            ggml_graph_compute_with_ctx(ctx, cgraph, thread_count);
            // Memory barrier to ensure completion
            __asm__ __volatile__ ("" ::: "memory");
            auto end = get_time();
            
            double time_ms = get_duration_ms(start, end);
            
            // More strict filtering - realistic minimum time based on data size
            double min_realistic_time_ms = std::max(0.005, (double)total_elements / 100000000.0);  // ~100M ops/ms max
            if (time_ms >= min_realistic_time_ms) {
                times.push_back(time_ms);
            } else {
                printf("⚠️  Filtered unrealistic timing: %.6f ms (expected >= %.3f ms for %zu elements)\n", 
                       time_ms, min_realistic_time_ms, total_elements);
            }
        }
        
        if (times.empty()) {
            printf("❌ No valid timing measurements obtained\n");
            ggml_free(ctx);
            return false;
        }
        
        // Calculate comprehensive statistics (remove outliers)
        std::sort(times.begin(), times.end());
        
        // Remove extreme outliers (top/bottom 10%)
        size_t outlier_count = times.size() / 10;
        if (outlier_count > 0 && times.size() > 4) {
            times.erase(times.begin(), times.begin() + outlier_count);
            times.erase(times.end() - outlier_count, times.end());
        }
        
        double min_time_ms = times.front();
        double max_time_ms = times.back();
        double avg_time_ms = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        
        // Calculate performance metrics
        size_t bytes_processed = 3 * total_elements * sizeof(float); // read A, read B, write result
        size_t memory_size_mb = (total_elements * sizeof(float)) / (1024 * 1024);
        
        // Throughput calculation (GB/s)
        double throughput_gb_per_sec = (bytes_processed / (1024.0 * 1024.0 * 1024.0)) / (avg_time_ms / 1000.0);
        
        // GFLOPS calculation (ADD operation: 1 FLOP per element)
        double gflops = (total_elements / (avg_time_ms / 1000.0)) / 1e9;
        
        // Memory bandwidth (effective bandwidth achieved)
        double memory_bandwidth_gb_per_sec = throughput_gb_per_sec;
        
        // Validate result to ensure computation actually happened
        volatile float validation_sum = 0.0f;
        bool computation_verified = false;
        for (size_t i = 0; i < std::min(total_elements, (size_t)1000); i++) {
            validation_sum += result_data[i];
            if (result_data[i] != 0.0f) {
                computation_verified = true;
            }
        }
        
        if (!computation_verified) {
            printf("❌ ERROR: No computation detected - all results are zero!\n");
            ggml_free(ctx);
            return false;
        }
        
        printf("      ⏱️  Time: %.3f ms (min=%.3f, max=%.3f), GFLOPS: %.2f, Bandwidth: %.2f GB/s\n", 
               avg_time_ms, min_time_ms, max_time_ms, gflops, memory_bandwidth_gb_per_sec);
        
        // Store comprehensive result
        PerformanceResult perf_result;
        perf_result.test_name = std::string(test_case.name);
        perf_result.execution_strategy = strategy;
        perf_result.numa_enabled = use_numa;
        perf_result.thread_count = thread_count;
        perf_result.avg_time_ms = avg_time_ms;
        perf_result.min_time_ms = min_time_ms;
        perf_result.max_time_ms = max_time_ms;
        perf_result.tensor_elements = total_elements;
        perf_result.memory_size_mb = memory_size_mb;
        perf_result.throughput_gb_per_sec = throughput_gb_per_sec;
        perf_result.gflops = gflops;
        perf_result.memory_bandwidth_gb_per_sec = memory_bandwidth_gb_per_sec;
        results.push_back(perf_result);
        
        ggml_free(ctx);
        return true;
    }
    
public:
    ComprehensiveNumaPerformanceBenchmark() {
        initialize_test_cases();
    }
    
    void test_ADD_performance() {
        printf("=== Comprehensive NUMA Performance Benchmark for ADD Operation ===\n");
        
        // Initialize NUMA with MIRROR strategy for optimal performance
        printf("🪞 Initializing NUMA with MIRROR strategy...\n");
        ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);
        
        // Debug: Check if coordinator is actually initialized
        extern bool ggml_numa_simple_coordinator_is_initialized(void);
        bool coord_initialized = ggml_numa_simple_coordinator_is_initialized();
        printf("🔍 DEBUG: Coordinator initialized = %s\n", coord_initialized ? "✅ YES" : "❌ NO");
        
        if (!coord_initialized) {
            printf("⚠️  WARNING: NUMA coordinator failed to initialize - NUMA execution will be disabled!\n");
        }
        
        printf("📊 Testing ADD performance with comprehensive strategies...\n\n");
        
        for (const auto& test_case : test_cases) {
            printf("  🔍 %s complexity [%d,%d,%d] - %s:\n", 
                   test_case.name, test_case.dim1, test_case.dim2, test_case.dim3, test_case.description);
            
            // 1. Single-threaded NUMA node 0 vs Fallback single-threaded
            test_single_ADD_performance(test_case, "SingleThread-Node0", false, 1);  // Fallback ST
            test_single_ADD_performance(test_case, "SingleThread-Node0", true, 1);   // NUMA ST
            
            // 2. Multi-threaded NUMA node 0 vs Fallback multi-threaded
            int max_threads = std::min(8, static_cast<int>(std::thread::hardware_concurrency()));
            test_single_ADD_performance(test_case, "MultiThread-Node0", false, max_threads);  // Fallback MT
            test_single_ADD_performance(test_case, "MultiThread-Node0", true, max_threads);   // NUMA MT Single Node
            
            // 3. Multi-node NUMA vs Fallback multi-threaded (for all tensor sizes)
            test_single_ADD_performance(test_case, "MultiNode-DataParallel", false, max_threads);  // Fallback MT
            test_single_ADD_performance(test_case, "MultiNode-DataParallel", true, max_threads);   // NUMA Multi-node
            
            printf("\n");
        }
    }
    
    void print_comprehensive_analysis() {
        printf("=== Comprehensive Performance Analysis ===\n");
        
        // Group results by test case and compare strategies
        std::map<std::string, std::vector<PerformanceResult>> grouped_results;
        for (const auto& result : results) {
            grouped_results[result.test_name].push_back(result);
        }
        
        printf("%-10s %-20s %-10s %-12s %-12s %-10s %-10s %-12s\n", 
               "Size", "Strategy", "Type", "Time(ms)", "GFLOPS", "BW(GB/s)", "Speedup", "Efficiency");
        printf("%-10s %-20s %-10s %-12s %-12s %-10s %-10s %-12s\n", 
               "----", "--------", "----", "-------", "------", "-------", "-------", "----------");
        
        double total_speedup = 0.0;
        int speedup_count = 0;
        double best_speedup = 0.0;
        int successful_tests = 0;
        int total_benchmark_tests = 0;
        
        for (const auto& group : grouped_results) {
            const std::string& size_name = group.first;
            const auto& size_results = group.second;
            
            // Group by strategy
            std::map<std::string, std::pair<PerformanceResult, PerformanceResult>> strategy_pairs;
            
            for (const auto& result : size_results) {
                auto key = result.execution_strategy;
                if (!result.numa_enabled) {
                    strategy_pairs[key].first = result;  // Fallback
                } else {
                    strategy_pairs[key].second = result; // NUMA
                }
            }
            
            // Analyze each strategy pair
            for (const auto& pair : strategy_pairs) {
                const std::string& strategy = pair.first;
                const auto& fallback = pair.second.first;
                const auto& numa = pair.second.second;
                
                total_benchmark_tests += 2; // Fallback + NUMA
                
                if (fallback.avg_time_ms > 0 && numa.avg_time_ms > 0) {
                    double speedup = fallback.avg_time_ms / numa.avg_time_ms;
                    double efficiency = (speedup - 1.0) * 100.0; // Percentage improvement
                    
                    printf("%-10s %-20s %-10s %-12.3f %-12.2f %-10.2f %-10s %-12s\n", 
                           size_name.c_str(), strategy.c_str(), "Fallback", 
                           fallback.avg_time_ms, fallback.gflops, fallback.memory_bandwidth_gb_per_sec, 
                           "-", "-");
                    printf("%-10s %-20s %-10s %-12.3f %-12.2f %-10.2f %-9.2fx %-11.1f%%\n", 
                           "", "", "NUMA", 
                           numa.avg_time_ms, numa.gflops, numa.memory_bandwidth_gb_per_sec, 
                           speedup, efficiency);
                    
                    total_speedup += speedup;
                    speedup_count++;
                    best_speedup = std::max(best_speedup, speedup);
                    successful_tests += 2;
                    
                    printf("\n");
                }
            }
        }
        
        // Print summary statistics compatible with test runner parsing
        if (speedup_count > 0) {
            double avg_speedup = total_speedup / speedup_count;
            printf("=== Performance Summary ===\n");
            printf("Average speedup: %.2fx\n", avg_speedup);
            printf("Best speedup: %.2fx\n", best_speedup);
            printf("Successful tests: %d/%d\n", successful_tests, total_benchmark_tests);
            
            if (avg_speedup > 1.2) {
                printf("🎉 NUMA shows significant performance improvement!\n");
            } else if (avg_speedup > 1.05) {
                printf("✅ NUMA shows modest performance improvement\n");
            } else {
                printf("⚠️  NUMA performance similar to fallback\n");
            }
        }
        
        // Detailed memory hierarchy analysis
        printf("\n=== Memory Hierarchy Performance Analysis ===\n");
        for (const auto& group : grouped_results) {
            if (!group.second.empty()) {
                auto sample = group.second[0];
                printf("%s (%zu MB): ", group.first.c_str(), sample.memory_size_mb);
                
                if (sample.memory_size_mb < 1) {
                    printf("L1 cache friendly - expect high NUMA efficiency\n");
                } else if (sample.memory_size_mb < 8) {
                    printf("L2/L3 cache range - moderate NUMA benefit expected\n");
                } else {
                    printf("Memory-bound - significant NUMA opportunity\n");
                }
            }
        }
    }
    
    void run_comprehensive_benchmark() {
        printf("🚀 Starting Comprehensive NUMA Performance Benchmark for ADD Operation...\n\n");
        
        test_ADD_performance();
        print_comprehensive_analysis();
        
        printf("\n🏁 Comprehensive performance benchmarking complete!\n");
    }
};

int main() {
    printf("🚀 Initializing NUMA Performance Instrumentation...\n");
    
    // Initialize performance measurement system
    if (!ggml_numa_perf_init()) {
        printf("❌ Failed to initialize performance measurement system\n");
        return 1;
    }
    
    // Enable performance measurement but disable detailed logging to reduce noise
    ggml_numa_perf_set_enabled(true);
    ggml_numa_perf_set_detailed_logging(false);
    
    printf("✅ Performance instrumentation enabled\n\n");
    
    try {
        // First, run one quick test with debug output to see what's happening
        printf("🔍 Running debug test to analyze NUMA execution patterns...\n");
        
        ComprehensiveNumaPerformanceBenchmark debug_benchmark;
        
        // Test just one case with debug output enabled
        printf("DEBUG TEST: Running LARGE tensor test with NUMA...\n");
        
        // Run a single large test case to see debug output
        // Create a medium tensor that fits in context
        size_t ctx_size = 64 * 1024 * 1024; // 64MB
        struct ggml_init_params params = {
            .mem_size = ctx_size,
            .mem_buffer = NULL,
            .no_alloc = false,
        };
        
        struct ggml_context* ctx = ggml_init(params);
        if (!ctx) {
            printf("❌ Failed to create debug context\n");
            return 1;
        }
        
        // Create MEDIUM tensors (4MB total)
        int dim1 = 512, dim2 = 512, dim3 = 4;
        struct ggml_tensor* a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim1, dim2, dim3);
        struct ggml_tensor* b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim1, dim2, dim3);
        struct ggml_tensor* result = ggml_add(ctx, a, b);
        
        // Initialize data
        float* a_data = (float*)ggml_get_data(a);
        float* b_data = (float*)ggml_get_data(b);
        size_t total_elements = dim1 * dim2 * dim3;
        for (size_t i = 0; i < total_elements; i++) {
            a_data[i] = 1.0f;
            b_data[i] = 2.0f;
        }
        
        // Create computation graph
        struct ggml_cgraph* cgraph = ggml_new_graph(ctx);
        ggml_build_forward_expand(cgraph, result);
        
        printf("🎯 Executing MEDIUM tensor [%dx%dx%d] with NUMA (%.1f MB)...\n", 
               dim1, dim2, dim3, (total_elements * sizeof(float)) / (1024.0 * 1024.0));
        
        // Execute with NUMA and show debug output
        enum ggml_status status = ggml_graph_compute_with_ctx(ctx, cgraph, 8);
        
        printf("✅ Debug test completed with status %d\n\n", status);
        ggml_free(ctx);
        
        printf("🔇 Now running full benchmark with output suppressed...\n\n");
        
        // Redirect stdout and stderr to /dev/null during benchmark execution
        int stdout_backup = dup(STDOUT_FILENO);
        int stderr_backup = dup(STDERR_FILENO);
        
        int devnull = open("/dev/null", O_WRONLY);
        dup2(devnull, STDOUT_FILENO);
        dup2(devnull, STDERR_FILENO);
        close(devnull);
        
        ComprehensiveNumaPerformanceBenchmark benchmark;
        benchmark.run_comprehensive_benchmark();
        
        // Restore stdout and stderr
        dup2(stdout_backup, STDOUT_FILENO);
        dup2(stderr_backup, STDERR_FILENO);
        close(stdout_backup);
        close(stderr_backup);
        
        // Print the comprehensive performance analysis
        benchmark.print_comprehensive_analysis();
        
        // Shutdown performance measurement (automatically prints summary)
        ggml_numa_perf_shutdown();
        
        return 0;
    } catch (const std::exception& e) {
        printf("❌ Benchmark failed with exception: %s\n", e.what());
        ggml_numa_perf_shutdown();
        return 1;
    } catch (...) {
        printf("❌ Benchmark failed with unknown exception\n");
        ggml_numa_perf_shutdown();
        return 1;
    }
}
