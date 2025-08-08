/**
 * NUMA Coordinator Hotspot and Scaling Analysis
 * 
 * Comprehensive performance analysis focusing on:
 * 1. Proper warmup to amortize initialization costs
 * 2. Batch size scaling to find coordinator sweet spots
 * 3. Detailed timing breakdown for hotspot identification
 * 4. Contention analysis for synchronization bottlenecks
 * 5. Workload size scaling to determine minimum viable batch sizes
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
#include "ggml-numa-coordinator.h"
#include "common.h"

struct DetailedTiming {
    double initialization_ms = 0.0;
    double warmup_ms = 0.0;
    double execution_ms = 0.0;
    double synchronization_ms = 0.0;
    double cleanup_ms = 0.0;
    double total_ms = 0.0;
    
    // NUMA-specific timings
    double coordinator_startup_ms = 0.0;
    double work_submission_ms = 0.0;
    double work_completion_ms = 0.0;
    
    // Concurrency metrics
    int active_numa_nodes = 0;
    int total_work_items = 0;
    double parallelism_efficiency = 0.0;
};

struct ScalingResult {
    std::string test_name;
    std::string operation;
    int64_t tensor_elements;
    int batch_size;
    int iterations;
    
    DetailedTiming baseline_timing;
    DetailedTiming coordinator_timing;
    
    double baseline_throughput_gflops = 0.0;
    double coordinator_throughput_gflops = 0.0;
    double speedup = 0.0;
    double efficiency_percent = 0.0;
    
    bool coordinator_beneficial = false;
    std::string analysis_notes;
};

class NumaHotspotAnalyzer {
private:
    std::vector<ScalingResult> results;
    int max_threads;
    
public:
    NumaHotspotAnalyzer(int max_threads) : max_threads(max_threads) {}

    // High-resolution timing helper
    std::chrono::high_resolution_clock::time_point get_time() {
        return std::chrono::high_resolution_clock::now();
    }
    
    double time_diff_ms(std::chrono::high_resolution_clock::time_point start, 
                       std::chrono::high_resolution_clock::time_point end) {
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    // Comprehensive baseline benchmark with detailed timing
    DetailedTiming benchmark_baseline_detailed(const std::string& operation, 
                                             int64_t tensor_size, 
                                             int batch_size, 
                                             int iterations) {
        DetailedTiming timing = {};
        
        std::cout << "🔧 Baseline " << operation << " (tensor: " << tensor_size 
                 << ", batch: " << batch_size << ", iterations: " << iterations << ")" << std::endl;

        auto total_start = get_time();
        auto init_start = get_time();

        // Initialization phase
        struct ggml_init_params init_params = {
            1024 * 1024 * 1024, // 1GB memory pool for large batch testing
            NULL,
            false,
        };
        
        struct ggml_context * ctx = ggml_init(init_params);
        if (!ctx) {
            std::cout << "  ❌ Failed to create context" << std::endl;
            return timing;
        }

        // Create tensor batch
        std::vector<struct ggml_tensor *> tensors_a, tensors_b, results;
        for (int b = 0; b < batch_size; b++) {
            struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, tensor_size);
            struct ggml_tensor * b_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, tensor_size);
            struct ggml_tensor * result = nullptr;
            
            // Fill with random data
            fill_tensor_random(a);
            fill_tensor_random(b_tensor);
            
            if (operation == "ADD") {
                result = ggml_add(ctx, a, b_tensor);
            } else if (operation == "MUL") {
                result = ggml_mul(ctx, a, b_tensor);
            } else if (operation == "MUL_MAT") {
                int64_t dim = static_cast<int64_t>(std::sqrt(tensor_size));
                a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, dim);
                b_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, dim);
                fill_tensor_random(a);
                fill_tensor_random(b_tensor);
                result = ggml_mul_mat(ctx, a, b_tensor);
            }
            
            tensors_a.push_back(a);
            tensors_b.push_back(b_tensor);
            results.push_back(result);
        }

        auto init_end = get_time();
        timing.initialization_ms = time_diff_ms(init_start, init_end);

        // Warmup phase
        auto warmup_start = get_time();
        
        struct ggml_cgraph * warmup_graph = ggml_new_graph(ctx);
        for (auto& result : results) {
            ggml_build_forward_expand(warmup_graph, result);
        }
        
        struct ggml_cplan cplan = ggml_graph_plan(warmup_graph, max_threads, nullptr);
        
        // Warmup iterations
        for (int w = 0; w < 3; w++) {
            ggml_graph_compute(warmup_graph, &cplan);
        }
        
        auto warmup_end = get_time();
        timing.warmup_ms = time_diff_ms(warmup_start, warmup_end);

        // Main execution phase
        auto exec_start = get_time();
        
        for (int i = 0; i < iterations; i++) {
            ggml_graph_compute(warmup_graph, &cplan);
        }
        
        auto exec_end = get_time();
        timing.execution_ms = time_diff_ms(exec_start, exec_end);
        
        auto total_end = get_time();
        timing.total_ms = time_diff_ms(total_start, total_end);
        
        timing.synchronization_ms = 0.0; // No explicit sync in baseline
        timing.cleanup_ms = timing.total_ms - (timing.initialization_ms + timing.warmup_ms + timing.execution_ms);

        std::cout << "  ✅ Init: " << std::fixed << std::setprecision(2) << timing.initialization_ms << "ms, "
                 << "Warmup: " << timing.warmup_ms << "ms, "
                 << "Exec: " << timing.execution_ms << "ms" << std::endl;

        ggml_free(ctx);
        return timing;
    }

    // Comprehensive coordinator benchmark with detailed timing
    DetailedTiming benchmark_coordinator_detailed(const std::string& operation, 
                                                int64_t tensor_size, 
                                                int batch_size, 
                                                int iterations) {
        DetailedTiming timing = {};
        
        std::cout << "🔧 NUMA Coordinator " << operation << " (tensor: " << tensor_size 
                 << ", batch: " << batch_size << ", iterations: " << iterations << ")" << std::endl;

        auto total_start = get_time();
        auto init_start = get_time();

        // Initialization phase
        struct ggml_init_params init_params = {
            1024 * 1024 * 1024, // 1GB memory pool for large batch testing
            NULL,
            false,
        };
        
        struct ggml_context * ctx = ggml_init(init_params);
        if (!ctx) {
            std::cout << "  ❌ Failed to create context" << std::endl;
            return timing;
        }

        auto coordinator_start = get_time();
        
        // Get coordinator manager (this includes initialization cost)
        struct ggml_numa_coordinator_manager * mgr = 
            ggml_numa_coordinator_manager_get_global(max_threads, true);
        if (!mgr) {
            std::cout << "  ❌ Failed to create coordinator manager" << std::endl;
            ggml_free(ctx);
            return timing;
        }
        
        auto coordinator_end = get_time();
        timing.coordinator_startup_ms = time_diff_ms(coordinator_start, coordinator_end);

        // Create tensor batch
        std::vector<struct ggml_tensor *> tensors_a, tensors_b, results;
        for (int b = 0; b < batch_size; b++) {
            struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, tensor_size);
            struct ggml_tensor * b_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, tensor_size);
            struct ggml_tensor * result = nullptr;
            
            // Fill with random data
            fill_tensor_random(a);
            fill_tensor_random(b_tensor);
            
            if (operation == "ADD") {
                result = ggml_add(ctx, a, b_tensor);
            } else if (operation == "MUL") {
                result = ggml_mul(ctx, a, b_tensor);
            } else if (operation == "MUL_MAT") {
                int64_t dim = static_cast<int64_t>(std::sqrt(tensor_size));
                a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, dim);
                b_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, dim);
                fill_tensor_random(a);
                fill_tensor_random(b_tensor);
                result = ggml_mul_mat(ctx, a, b_tensor);
            }
            
            tensors_a.push_back(a);
            tensors_b.push_back(b_tensor);
            results.push_back(result);
        }

        auto init_end = get_time();
        timing.initialization_ms = time_diff_ms(init_start, init_end);

        // Warmup phase
        auto warmup_start = get_time();
        
        // Warmup iterations with coordinator
        for (int w = 0; w < 3; w++) {
            for (auto& result : results) {
                struct ggml_cgraph * warmup_graph = ggml_new_graph(ctx);
                ggml_build_forward_expand(warmup_graph, result);
                ggml_numa_coordinator_manager_compute_graph(mgr, warmup_graph);
            }
        }
        
        auto warmup_end = get_time();
        timing.warmup_ms = time_diff_ms(warmup_start, warmup_end);

        // Main execution phase with detailed timing
        auto submission_start = get_time();
        auto exec_start = get_time();
        
        for (int i = 0; i < iterations; i++) {
            auto iter_submission_start = get_time();
            
            // Submit all work in batch
            for (auto& result : results) {
                struct ggml_cgraph * cgraph = ggml_new_graph(ctx);
                ggml_build_forward_expand(cgraph, result);
                
                int compute_result = ggml_numa_coordinator_manager_compute_graph(mgr, cgraph);
                if (compute_result != 0) {
                    std::cout << "  ⚠️ Coordinator computation failed for iteration " << i << std::endl;
                    // Continue with other operations
                }
            }
            
            auto iter_submission_end = get_time();
            timing.work_submission_ms += time_diff_ms(iter_submission_start, iter_submission_end);
        }
        
        auto exec_end = get_time();
        timing.execution_ms = time_diff_ms(exec_start, exec_end);
        timing.work_completion_ms = timing.execution_ms - timing.work_submission_ms;
        
        auto total_end = get_time();
        timing.total_ms = time_diff_ms(total_start, total_end);
        
        timing.synchronization_ms = timing.work_completion_ms; // Most coordinator time is sync
        timing.cleanup_ms = timing.total_ms - (timing.initialization_ms + timing.warmup_ms + timing.execution_ms);

        // Record concurrency metrics
        timing.active_numa_nodes = 2; // We know we have 2 simulated nodes
        timing.total_work_items = batch_size * iterations;
        timing.parallelism_efficiency = timing.active_numa_nodes > 0 ? 
            (timing.execution_ms / timing.total_ms) : 0.0;

        std::cout << "  ✅ Init: " << std::fixed << std::setprecision(2) << timing.initialization_ms << "ms, "
                 << "Coordinator: " << timing.coordinator_startup_ms << "ms, "
                 << "Warmup: " << timing.warmup_ms << "ms, "
                 << "Exec: " << timing.execution_ms << "ms" << std::endl;
        std::cout << "      Submission: " << timing.work_submission_ms << "ms, "
                 << "Completion: " << timing.work_completion_ms << "ms" << std::endl;

        ggml_free(ctx);
        return timing;
    }

    // Fill tensor with random data
    void fill_tensor_random(struct ggml_tensor * tensor) {
        if (tensor->type != GGML_TYPE_F32) return;
        
        float * data = (float*)ggml_get_data(tensor);
        int64_t total_elements = ggml_nelements(tensor);
        
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<float> dis(0.1f, 2.0f);
        
        for (int64_t i = 0; i < total_elements; i++) {
            data[i] = dis(gen);
        }
    }

    // Calculate FLOPS for different operations
    double calculate_flops(const std::string& operation, int64_t tensor_size, int batch_size, int iterations) {
        double ops_per_element = 1.0;
        
        if (operation == "MUL_MAT") {
            int64_t dim = static_cast<int64_t>(std::sqrt(tensor_size));
            ops_per_element = 2.0 * dim; // 2 ops per element for matrix multiplication
        }
        
        return tensor_size * ops_per_element * batch_size * iterations;
    }

    // Comprehensive scaling analysis
    void run_scaling_analysis() {
        std::cout << "🏁 NUMA Coordinator Hotspot and Scaling Analysis" << std::endl;
        std::cout << "==================================================" << std::endl;
        std::cout << "Max threads: " << max_threads << std::endl << std::endl;

        // Test different combinations of tensor sizes and batch sizes
        std::vector<std::string> operations = {"ADD"};  // Focus on ADD for now
        std::vector<int64_t> tensor_sizes = {100000, 1000000, 5000000}; // 100K to 5M - more memory efficient
        std::vector<int> batch_sizes = {1, 5, 10, 15, 20}; // Conservative batch sizes
        int iterations = 10;

        for (const auto& operation : operations) {
            std::cout << "🧮 Analyzing " << operation << " Operation Scaling" << std::endl;
            std::cout << "===============================================" << std::endl;

            for (auto tensor_size : tensor_sizes) {
                std::cout << "  📊 Tensor size: " << tensor_size << " elements" << std::endl;

                for (auto batch_size : batch_sizes) {
                    ScalingResult result = {};
                    result.test_name = "Hotspot Analysis";
                    result.operation = operation;
                    result.tensor_elements = tensor_size;
                    result.batch_size = batch_size;
                    result.iterations = iterations;

                    // Run baseline benchmark
                    result.baseline_timing = benchmark_baseline_detailed(operation, tensor_size, batch_size, iterations);

                    // Calculate baseline FLOPS
                    double total_flops = calculate_flops(operation, tensor_size, batch_size, iterations);
                    result.baseline_throughput_gflops = (total_flops / 1e9) / (result.baseline_timing.execution_ms / 1000.0);

                    // Run coordinator benchmark
                    result.coordinator_timing = benchmark_coordinator_detailed(operation, tensor_size, batch_size, iterations);

                    // Calculate coordinator FLOPS
                    result.coordinator_throughput_gflops = (total_flops / 1e9) / (result.coordinator_timing.execution_ms / 1000.0);

                    // Analysis
                    result.speedup = result.coordinator_throughput_gflops / result.baseline_throughput_gflops;
                    result.efficiency_percent = (result.speedup / max_threads) * 100.0;
                    result.coordinator_beneficial = result.speedup > 1.0;

                    // Analysis notes
                    if (result.coordinator_timing.coordinator_startup_ms > result.coordinator_timing.execution_ms) {
                        result.analysis_notes = "Startup overhead dominates";
                    } else if (result.coordinator_timing.work_submission_ms > result.coordinator_timing.work_completion_ms) {
                        result.analysis_notes = "Submission overhead high";
                    } else if (result.speedup > 1.2) {
                        result.analysis_notes = "Good parallelization benefit";
                    } else if (result.speedup > 0.8) {
                        result.analysis_notes = "Marginal benefit";
                    } else {
                        result.analysis_notes = "Coordinator overhead too high";
                    }

                    results.push_back(result);

                    std::cout << "    Batch " << batch_size << ": " 
                             << std::fixed << std::setprecision(2) << result.speedup << "x speedup, "
                             << result.efficiency_percent << "% efficiency ("
                             << result.analysis_notes << ")" << std::endl;
                }
                std::cout << std::endl;
            }
        }

        print_comprehensive_analysis();
    }

    // Print comprehensive analysis with hotspot identification
    void print_comprehensive_analysis() {
        std::cout << "📊 Comprehensive Hotspot and Scaling Analysis" << std::endl;
        std::cout << "==============================================" << std::endl;

        // Find optimal configurations
        ScalingResult best_speedup = *std::max_element(results.begin(), results.end(),
            [](const ScalingResult& a, const ScalingResult& b) { return a.speedup < b.speedup; });

        ScalingResult best_efficiency = *std::max_element(results.begin(), results.end(),
            [](const ScalingResult& a, const ScalingResult& b) { return a.efficiency_percent < b.efficiency_percent; });

        std::cout << std::endl << "🏆 Best Performance Configurations:" << std::endl;
        std::cout << "Best Speedup: " << std::fixed << std::setprecision(2) << best_speedup.speedup 
                 << "x (" << best_speedup.operation << ", " << best_speedup.tensor_elements 
                 << " elements, batch " << best_speedup.batch_size << ")" << std::endl;
        std::cout << "Best Efficiency: " << best_efficiency.efficiency_percent << "% (" 
                 << best_efficiency.operation << ", " << best_efficiency.tensor_elements 
                 << " elements, batch " << best_efficiency.batch_size << ")" << std::endl;

        // Hotspot analysis
        std::cout << std::endl << "🔥 Hotspot Analysis:" << std::endl;
        
        double avg_startup_overhead = 0.0;
        double avg_submission_overhead = 0.0;
        double avg_sync_overhead = 0.0;
        int beneficial_configs = 0;

        for (const auto& result : results) {
            double total_overhead = result.coordinator_timing.total_ms - result.baseline_timing.total_ms;
            if (total_overhead > 0) {
                avg_startup_overhead += result.coordinator_timing.coordinator_startup_ms;
                avg_submission_overhead += result.coordinator_timing.work_submission_ms;
                avg_sync_overhead += result.coordinator_timing.synchronization_ms;
            }
            if (result.coordinator_beneficial) beneficial_configs++;
        }

        int total_configs = results.size();
        avg_startup_overhead /= total_configs;
        avg_submission_overhead /= total_configs;
        avg_sync_overhead /= total_configs;

        std::cout << "Average Coordinator Startup: " << std::fixed << std::setprecision(2) 
                 << avg_startup_overhead << "ms" << std::endl;
        std::cout << "Average Work Submission: " << avg_submission_overhead << "ms" << std::endl;
        std::cout << "Average Synchronization: " << avg_sync_overhead << "ms" << std::endl;
        std::cout << "Beneficial Configurations: " << beneficial_configs << "/" << total_configs 
                 << " (" << (100.0 * beneficial_configs / total_configs) << "%)" << std::endl;

        // Scaling recommendations
        std::cout << std::endl << "💡 Optimization Recommendations:" << std::endl;
        
        if (avg_startup_overhead > 10.0) {
            std::cout << "- ⚠️  HIGH STARTUP OVERHEAD: Consider persistent coordinator pools" << std::endl;
        }
        if (avg_submission_overhead > avg_sync_overhead) {
            std::cout << "- ⚠️  SUBMISSION BOTTLENECK: Optimize work queue operations" << std::endl;
        }
        if (beneficial_configs < total_configs * 0.5) {
            std::cout << "- ⚠️  LOW BENEFIT RATIO: Coordinator needs significant optimization" << std::endl;
        }
        
        // Find minimum viable batch sizes
        std::map<int64_t, int> min_batch_size;
        for (const auto& result : results) {
            if (result.coordinator_beneficial && min_batch_size.find(result.tensor_elements) == min_batch_size.end()) {
                min_batch_size[result.tensor_elements] = result.batch_size;
            }
        }
        
        std::cout << "- 📏 Minimum Viable Batch Sizes:" << std::endl;
        for (const auto& pair : min_batch_size) {
            std::cout << "    " << pair.first << " elements: batch size " << pair.second << "+" << std::endl;
        }

        std::cout << std::endl;
    }
};

int main() {
    // Get system configuration
    int cpu_count = std::thread::hardware_concurrency();
    int max_test_threads = std::min(cpu_count, 16);
    
    std::cout << "🔧 System Configuration" << std::endl;
    std::cout << "Available CPUs: " << cpu_count << std::endl;
    std::cout << "Max test threads: " << max_test_threads << std::endl << std::endl;
    
    NumaHotspotAnalyzer analyzer(max_test_threads);
    analyzer.run_scaling_analysis();
    
    return 0;
}
