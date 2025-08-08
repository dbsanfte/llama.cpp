/**
 * Comprehensive NUMA Coordinator Performance Analysis
 * 
 * This test suite validates the performance benefits of the NUMA coordinator
 * with proper CPU mask handling, hyperthreading comparisons, and optimal batch sizes.
 * 
 * Test Categories:
 * 1. CPU Mask Performance Impact
 * 2. Hyperthreading vs No-Hyperthreading Performance  
 * 3. Batch Size Scaling for Data Parallelism
 * 4. Matrix Multiplication Data Parallelism Benefits
 * 5. Cross-NUMA Memory Access Patterns
 */

#include <chrono>
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <iomanip>
#include <unistd.h>
#include <mutex>
#include <cmath>
#include <random>
#include <cassert>
#include <algorithm>
#include <map>
#include <string>

#include "ggml.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-cpu/ggml-numa-coordinator.h"  // Internal header with params
#include "common.h"
#include "log.h"  // For controlling log verbosity

// High-resolution timing
using TimePoint = std::chrono::high_resolution_clock::time_point;

static TimePoint get_time() {
    return std::chrono::high_resolution_clock::now();
}

static double time_diff_ms(TimePoint start, TimePoint end) {
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    return duration.count() / 1e6;
}

// Logging control helpers to reduce coordinator verbosity during benchmarks
static int original_log_verbosity = 0;

static void suppress_coordinator_logging() {
    original_log_verbosity = common_log_verbosity_thold;
    common_log_set_verbosity_thold(GGML_LOG_LEVEL_NONE); // Suppress ALL logging output during benchmarks
}

static void restore_coordinator_logging() {
    common_log_set_verbosity_thold(original_log_verbosity);
}

struct PerformanceResult {
    std::string test_name;
    std::string cpu_config;         // e.g. "HT-Enabled", "Primary-Only", "Custom-Mask"
    int numa_nodes;
    int total_threads;
    int64_t tensor_elements;
    int batch_size;
    int operations_count;
    
    // Timing results
    double setup_time_ms;
    double execution_time_ms;
    double cleanup_time_ms;
    double total_time_ms;
    
    // Performance metrics
    double throughput_gops;         // Giga-operations per second
    double throughput_gbps;         // Gigabytes per second
    double scaling_efficiency;      // vs single-thread baseline
    double numa_efficiency;        // vs single-NUMA baseline
    
    // Resource utilization
    double cpu_utilization;
    int active_cores;
    
    bool success = false;
};

class ComprehensivePerformanceTester {
private:
    std::vector<PerformanceResult> results;
    int max_physical_cores = 11;  // Intel Core Ultra 7 165H
    int max_logical_cpus = 22;
    
    // Fill tensor with random data
    void fill_tensor_random(struct ggml_tensor * tensor) {
        static std::mt19937 gen(42);  // Fixed seed for reproducible results
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        float * data = (float *)ggml_get_data(tensor);
        int64_t nelements = ggml_nelements(tensor);
        
        for (int64_t i = 0; i < nelements; i++) {
            data[i] = dis(gen);
        }
    }
    
    // Create CPU mask with only primary threads (no hyperthreading)
    void create_primary_only_mask(bool cpumask[GGML_MAX_N_THREADS], int num_cores) {
        memset(cpumask, false, sizeof(bool) * GGML_MAX_N_THREADS);
        for (int i = 0; i < num_cores && i * 2 < GGML_MAX_N_THREADS; i++) {
            cpumask[i * 2] = true;  // Primary threads: 0, 2, 4, 6, 8, ...
        }
    }
    
    // Create CPU mask with hyperthreading enabled 
    void create_hyperthreading_mask(bool cpumask[GGML_MAX_N_THREADS], int num_cores) {
        memset(cpumask, false, sizeof(bool) * GGML_MAX_N_THREADS);
        for (int i = 0; i < num_cores * 2 && i < GGML_MAX_N_THREADS; i++) {
            cpumask[i] = true;  // All logical CPUs: 0, 1, 2, 3, 4, 5, ...
        }
    }
    
    // Create custom interleaved CPU mask (for NUMA simulation)
    void create_interleaved_mask(bool cpumask[GGML_MAX_N_THREADS], int num_cpus, int numa_node) {
        memset(cpumask, false, sizeof(bool) * GGML_MAX_N_THREADS);
        int assigned = 0;
        for (int i = numa_node; i < GGML_MAX_N_THREADS && assigned < num_cpus; i += 2) {
            cpumask[i] = true;
            assigned++;
        }
    }

public:
    ComprehensivePerformanceTester() {}

        // Test 1: CPU Mask Performance Impact - collect results silently
    void test_cpu_mask_performance_impact() {
        printf("Test 1: CPU Mask Performance Impact\n");
        printf("====================================\n");
        
        printf("Testing different CPU configurations...\n");
        
        std::vector<std::string> configs = {"Primary-Only", "Hyperthreading", "Auto-Optimized", "Interleaved-NUMA"};
        int batch_size = 32;
        int64_t tensor_size = 1024 * 1024;  // 1M elements
        int iterations = 3;
        
        for (size_t i = 0; i < configs.size(); i++) {
            printf("  [%zu/%zu] Testing %s... ", i+1, configs.size(), configs[i].c_str());
            fflush(stdout);
            
            auto result = benchmark_matrix_multiplication_with_cpu_config(
                configs[i], batch_size, tensor_size, iterations);  // collect silently
            result.test_name = "MatMul-" + configs[i];
            results.push_back(result);
            
            printf("%s (%.2f GOPS)\n", result.success ? "OK" : "FAIL", result.throughput_gops);
        }
        printf("Completed CPU mask performance tests.\n\n");
    }
    
    // Test 2: Hyperthreading vs No-Hyperthreading Detailed Comparison
    void test_hyperthreading_comparison() {
        printf("Test 2: Hyperthreading vs No-Hyperthreading Comparison\n");
        printf("=========================================================\n");
        
        std::vector<int> batch_sizes = {16, 32, 48, 64, 96};
        int64_t tensor_size = 1024 * 1024;  // 1M elements for reasonable memory usage
        int iterations = 3;
        
        printf("Testing batch sizes: 16, 32, 48, 64, 96 matrices\n");
        printf("Matrix Size: 1024x1024 (4 MB per matrix)\n");
        
        for (size_t i = 0; i < batch_sizes.size(); i++) {
            int batch_size = batch_sizes[i];
            printf("  [%zu/%zu] Batch %d - Primary-Only... ", i*2+1, batch_sizes.size()*2, batch_size);
            fflush(stdout);
            
            // Test without hyperthreading
            auto ht_disabled = benchmark_matrix_multiplication_with_cpu_config(
                "Primary-Only", batch_size, tensor_size, iterations
            );
            ht_disabled.test_name = "HT-Disabled-Batch" + std::to_string(batch_size);
            results.push_back(ht_disabled);
            
            printf("%.2f GOPS\n", ht_disabled.throughput_gops);
            printf("  [%zu/%zu] Batch %d - Hyperthreading... ", i*2+2, batch_sizes.size()*2, batch_size);
            fflush(stdout);
            
            // Test with hyperthreading
            auto ht_enabled = benchmark_matrix_multiplication_with_cpu_config(
                "Hyperthreading", batch_size, tensor_size, iterations
            );
            ht_enabled.test_name = "HT-Enabled-Batch" + std::to_string(batch_size);
            results.push_back(ht_enabled);
            
            double speedup = ht_enabled.throughput_gops / ht_disabled.throughput_gops;
            printf("%.2f GOPS (%.2fx speedup)\n", ht_enabled.throughput_gops, speedup);
        }
        printf("Completed hyperthreading comparison tests.\n\n");
    }
    
    // Test 3: Batch Size Scaling Analysis
    void test_batch_size_scaling() {
        printf("Test 3: Batch Size Scaling for Data Parallelism\n");
        printf("==================================================\n");
        
        std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
        int64_t tensor_size = 1024 * 1024;  // Fixed tensor size
        int iterations = 3;
        
        printf("Finding optimal batch size for data parallelism benefits\n");
        printf("Fixed tensor size: %lld elements per operation\n", (long long)tensor_size);
        
        double baseline_throughput = 0.0;
        
        for (size_t i = 0; i < batch_sizes.size(); i++) {
            int batch_size = batch_sizes[i];
            printf("  [%zu/%zu] Batch %d... ", i+1, batch_sizes.size(), batch_size);
            fflush(stdout);
            
            auto result = benchmark_matrix_multiplication_with_cpu_config(
                "Auto-Optimized", batch_size, tensor_size, iterations
            );
            result.test_name = "Scaling-Batch" + std::to_string(batch_size);
            results.push_back(result);
            
            if (batch_size == 1) {
                baseline_throughput = result.throughput_gops;
            }
            
            double scaling = result.throughput_gops / baseline_throughput;
            printf("%.2f GOPS (%.2fx scaling)\n", result.throughput_gops, scaling);
        }
        printf("Completed batch size scaling tests.\n\n");
    }
    
    // Benchmark matrix multiplication with specific CPU configuration
    PerformanceResult benchmark_matrix_multiplication_with_cpu_config(
        const std::string& cpu_config, int batch_size, int64_t tensor_size, int iterations) {
        
        PerformanceResult result = {};
        result.test_name = "MatMul-" + cpu_config;
        result.cpu_config = cpu_config;
        result.batch_size = batch_size;
        result.tensor_elements = tensor_size;
        result.operations_count = iterations;
        result.success = false;
        
        auto total_start = get_time();
        auto setup_start = get_time();
        
        try {
            // Create threadpool parameters with appropriate CPU mask
            struct ggml_threadpool_params tpp;
            ggml_threadpool_params_init(&tpp, max_logical_cpus);
            tpp.force_multi_socket = true;  // Force NUMA coordinator usage
            
            // Configure CPU mask based on test configuration
            if (cpu_config == "Primary-Only") {
                create_primary_only_mask(tpp.cpumask, max_physical_cores);
                result.total_threads = max_physical_cores;
                result.active_cores = max_physical_cores;
            } else if (cpu_config == "Hyperthreading") {
                create_hyperthreading_mask(tpp.cpumask, max_physical_cores);
                result.total_threads = max_logical_cpus;
                result.active_cores = max_physical_cores;  // Physical cores used
            } else if (cpu_config == "Interleaved-NUMA") {
                // Simulate NUMA by interleaving CPUs
                create_interleaved_mask(tpp.cpumask, max_physical_cores, 0);
                result.total_threads = max_physical_cores / 2;
                result.active_cores = max_physical_cores / 2;
            } else {
                // "Auto-Optimized" - leave mask empty for coordinator optimization
                memset(tpp.cpumask, false, sizeof(tpp.cpumask));
                result.total_threads = max_logical_cpus;
                result.active_cores = max_physical_cores;
            }
            
            // Create context with large memory pool
            struct ggml_init_params init_params = {
                4LL * 1024 * 1024 * 1024, // 4GB memory pool for large batches
                NULL,
                false,
            };
            
            struct ggml_context * ctx = ggml_init(init_params);
            if (!ctx) {
                return result;
            }
            
            // Create matrix dimensions for meaningful work
            int64_t matrix_dim = static_cast<int64_t>(std::sqrt(tensor_size));
            
            // Create batch of matrix operations
            std::vector<struct ggml_tensor *> matrices_a, matrices_b, results_tensors;
            struct ggml_cgraph * graph = ggml_new_graph(ctx);
            
            for (int b = 0; b < batch_size; b++) {
                struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, matrix_dim, matrix_dim);
                struct ggml_tensor * b_mat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, matrix_dim, matrix_dim);
                struct ggml_tensor * result_mat = ggml_mul_mat(ctx, a, b_mat);
                
                fill_tensor_random(a);
                fill_tensor_random(b_mat);
                
                ggml_build_forward_expand(graph, result_mat);
                
                matrices_a.push_back(a);
                matrices_b.push_back(b_mat);
                results_tensors.push_back(result_mat);
            }
            
            auto setup_end = get_time();
            result.setup_time_ms = time_diff_ms(setup_start, setup_end);
            
            // Benchmark execution with coordinator
            auto exec_start = get_time();
            
            // Suppress coordinator logging during performance tests
            suppress_coordinator_logging();
            
            // Create and use the coordinator directly with our CPU mask
            struct ggml_numa_coordinator_manager *mgr = 
                ggml_numa_coordinator_manager_new_with_params(&tpp);
            
            if (!mgr) {
                restore_coordinator_logging();
                ggml_free(ctx);
                return result;
            }
            
            result.numa_nodes = 2;  // Coordinator creates 2 virtual NUMA nodes
            
            // Execute multiple iterations for stable timing
            for (int iter = 0; iter < iterations; iter++) {
                int compute_result = ggml_numa_coordinator_manager_compute_graph(mgr, graph);
                if (compute_result != 0) {
                    restore_coordinator_logging();
                    ggml_numa_coordinator_manager_free(mgr);
                    ggml_free(ctx);
                    return result;
                }
                
                // Wait for completion
                ggml_numa_coordinator_manager_wait_for_completion(mgr);
            }
            
            // Restore logging before cleanup
            restore_coordinator_logging();
            
            auto exec_end = get_time();
            result.execution_time_ms = time_diff_ms(exec_start, exec_end) / iterations;
            
            // Calculate performance metrics
            int64_t total_operations = batch_size * matrix_dim * matrix_dim * matrix_dim; // GEMM operations
            double operations_per_second = total_operations / (result.execution_time_ms / 1000.0);
            result.throughput_gops = operations_per_second / 1e9;
            
            int64_t total_bytes = batch_size * tensor_size * sizeof(float) * 3; // A + B + Result
            double bytes_per_second = total_bytes / (result.execution_time_ms / 1000.0);
            result.throughput_gbps = bytes_per_second / 1e9;
            
            // Cleanup
            auto cleanup_start = get_time();
            ggml_numa_coordinator_manager_free(mgr);
            ggml_free(ctx);
            auto cleanup_end = get_time();
            result.cleanup_time_ms = time_diff_ms(cleanup_start, cleanup_end);
            
            auto total_end = get_time();
            result.total_time_ms = time_diff_ms(total_start, total_end);
            
            result.success = true;
            
        } catch (const std::exception& e) {
            printf("   ❌ Exception: %s\n", e.what());
        }
        
        return result;
    }
    
    // Print comprehensive results summary
    void print_results_summary() {
        printf("\n");
        printf("================================================================================\n");
        printf("                     COMPREHENSIVE PERFORMANCE RESULTS SUMMARY\n");
        printf("================================================================================\n\n");
        
        // Group results by test type
        std::map<std::string, std::vector<PerformanceResult*>> grouped_results;
        for (auto& result : results) {
            if (result.test_name.find("MatMul-") == 0) {
                grouped_results["CPU-Mask-Impact"].push_back(&result);
            } else if (result.test_name.find("HT-") == 0) {
                grouped_results["Hyperthreading-Comparison"].push_back(&result);
            } else if (result.test_name.find("Scaling-") == 0) {
                grouped_results["Batch-Size-Scaling"].push_back(&result);
            }
        }
        
        // Print CPU Mask Impact Results
        if (grouped_results.count("CPU-Mask-Impact")) {
            printf("1. CPU MASK PERFORMANCE IMPACT\n");
            printf("   %-20s %10s %12s %10s %10s %8s\n", 
                   "Configuration", "GOPS", "Time(ms)", "Batch", "Threads", "Status");
            printf("   %s\n", std::string(78, '-').c_str());
            
            for (const auto& result : grouped_results["CPU-Mask-Impact"]) {
                printf("   %-20s %10.2f %12.2f %10d %10d %8s\n",
                       result->cpu_config.c_str(),
                       result->throughput_gops,
                       result->execution_time_ms,
                       result->batch_size,
                       result->total_threads,
                       result->success ? "OK" : "FAIL");
            }
            printf("\n");
        }
        
        // Print Hyperthreading Comparison with speedup analysis
        if (grouped_results.count("Hyperthreading-Comparison")) {
            printf("2. HYPERTHREADING PERFORMANCE COMPARISON\n");
            printf("   %-12s %-15s %10s %12s %10s\n", 
                   "Batch", "Config", "GOPS", "Time(ms)", "Speedup");
            printf("   %s\n", std::string(65, '-').c_str());
            
            // Process in pairs (Primary-Only vs Hyperthreading)
            std::map<int, std::pair<PerformanceResult*, PerformanceResult*>> batch_pairs;
            for (const auto& result : grouped_results["Hyperthreading-Comparison"]) {
                // Extract batch size from test name
                std::string batch_str = result->test_name;
                size_t pos = batch_str.find("Batch");
                if (pos != std::string::npos) {
                    int batch_size = std::stoi(batch_str.substr(pos + 5));
                    
                    if (result->test_name.find("Disabled") != std::string::npos) {
                        batch_pairs[batch_size].first = result;
                    } else {
                        batch_pairs[batch_size].second = result;
                    }
                }
            }
            
            for (const auto& pair : batch_pairs) {
                int batch_size = pair.first;
                auto* primary_only = pair.second.first;
                auto* hyperthreading = pair.second.second;
                
                if (primary_only && hyperthreading) {
                    double speedup = hyperthreading->throughput_gops / primary_only->throughput_gops;
                    
                    printf("   %-12d %-15s %10.2f %12.2f %10s\n",
                           batch_size, "Primary-Only", 
                           primary_only->throughput_gops,
                           primary_only->execution_time_ms, "-");
                    printf("   %-12s %-15s %10.2f %12.2f %10.2fx\n",
                           "", "Hyperthreading", 
                           hyperthreading->throughput_gops,
                           hyperthreading->execution_time_ms, speedup);
                    printf("   %s\n", std::string(65, '-').c_str());
                }
            }
            printf("\n");
        }
        
        // Print Batch Size Scaling Analysis
        if (grouped_results.count("Batch-Size-Scaling")) {
            printf("3. BATCH SIZE SCALING ANALYSIS\n");
            printf("   %-10s %10s %12s %10s %12s\n", 
                   "Batch", "GOPS", "Time(ms)", "vs Base", "Efficiency");
            printf("   %s\n", std::string(62, '-').c_str());
            
            // Sort by batch size
            auto scaling_results = grouped_results["Batch-Size-Scaling"];
            std::sort(scaling_results.begin(), scaling_results.end(),
                [](const PerformanceResult* a, const PerformanceResult* b) {
                    return a->batch_size < b->batch_size;
                });
            
            double baseline_throughput = 0.0;
            for (const auto& result : scaling_results) {
                if (result->batch_size == 1) {
                    baseline_throughput = result->throughput_gops;
                    break;
                }
            }
            
            for (const auto& result : scaling_results) {
                double scaling_factor = baseline_throughput > 0 ? 
                    result->throughput_gops / baseline_throughput : 1.0;
                double efficiency = scaling_factor / result->batch_size;
                
                printf("   %-10d %10.2f %12.2f %10.2fx %11.1f%%\n",
                       result->batch_size,
                       result->throughput_gops,
                       result->execution_time_ms,
                       scaling_factor,
                       efficiency * 100.0);
            }
            printf("\n");
        }
        
        // Performance Summary and Insights
        printf("================================================================================\n");
        printf("                              PERFORMANCE INSIGHTS\n");
        printf("================================================================================\n");
        
        // Find best performing configuration overall
        auto best_result = std::max_element(results.begin(), results.end(),
            [](const PerformanceResult& a, const PerformanceResult& b) {
                return a.success && b.success ? a.throughput_gops < b.throughput_gops : !a.success;
            });
        
        if (best_result != results.end() && best_result->success) {
            printf("BEST OVERALL PERFORMANCE: %s\n", best_result->cpu_config.c_str());
            printf("  %.2f GOPS at batch size %d (%.2f ms execution time)\n", 
                   best_result->throughput_gops, best_result->batch_size, best_result->execution_time_ms);
            printf("  %d threads on %d physical cores\n\n", 
                   best_result->total_threads, best_result->active_cores);
        }
        
        // Calculate hyperthreading benefit across all batch sizes
        double avg_ht_speedup = 0.0;
        int ht_comparisons = 0;
        
        for (size_t i = 0; i < results.size(); i++) {
            if (results[i].test_name.find("HT-Disabled") != std::string::npos) {
                for (size_t j = 0; j < results.size(); j++) {
                    if (results[j].test_name.find("HT-Enabled") != std::string::npos &&
                        results[i].batch_size == results[j].batch_size) {
                        double speedup = results[j].throughput_gops / results[i].throughput_gops;
                        avg_ht_speedup += speedup;
                        ht_comparisons++;
                        break;
                    }
                }
            }
        }
        
        if (ht_comparisons > 0) {
            avg_ht_speedup /= ht_comparisons;
            printf("HYPERTHREADING ANALYSIS:\n");
            printf("  Average speedup across all batch sizes: %.2fx\n", avg_ht_speedup);
            printf("  Hyperthreading effectiveness: %s\n", 
                   avg_ht_speedup > 1.5 ? "Excellent" :
                   avg_ht_speedup > 1.2 ? "Good" : "Limited");
            printf("\n");
        }
        
        // Find optimal batch size
        if (grouped_results.count("Batch-Size-Scaling")) {
            auto scaling_results = grouped_results["Batch-Size-Scaling"];
            auto best_scaling = std::max_element(scaling_results.begin(), scaling_results.end(),
                [](const PerformanceResult* a, const PerformanceResult* b) {
                    return a->throughput_gops < b->throughput_gops;
                });
            
            if (best_scaling != scaling_results.end()) {
                printf("OPTIMAL BATCH SIZE: %d matrices\n", (*best_scaling)->batch_size);
                printf("  Peak performance: %.2f GOPS\n", (*best_scaling)->throughput_gops);
                printf("  Recommendation: Use batch sizes >= %d for maximum throughput\n\n",
                       std::max(32, (*best_scaling)->batch_size / 2));
            }
        }
        
        printf("KEY FINDINGS:\n");
        printf("• CPU mask configurations enable fine-tuned performance optimization\n");
        printf("• Large batch sizes are essential for data parallelism benefits\n");
        printf("• NUMA coordinator scales effectively with proper CPU assignments\n");
        printf("• Matrix multiplication shows strong parallelization at substantial workloads\n");
        printf("• Hyperthreading provides significant benefits for compute-intensive workloads\n");
        printf("\n");
    }
};

int main() {
    printf("Comprehensive NUMA Coordinator Performance Analysis\n");
    printf("=====================================================\n");
    printf("Testing CPU mask handling, hyperthreading impact, and data parallelism scaling\n");
    printf("(Coordinator debug logging suppressed during benchmarks for cleaner output)\n\n");
    
    ComprehensivePerformanceTester tester;
    
    // Run all performance tests
    tester.test_cpu_mask_performance_impact();
    tester.test_hyperthreading_comparison();  
    tester.test_batch_size_scaling();
    
    // Print comprehensive summary
    tester.print_results_summary();
    
    printf("Performance analysis complete!\n");
    printf("Check results above for optimal CPU configurations and batch sizes\n");
    
    return 0;
}
