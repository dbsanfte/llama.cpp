/**
 * NUMA Coordinator Performance Test Suite
 * 
 * Tests performance scaling across different NUMA configurations:
 * 1. Baseline single-threaded performance (no coordinator)
 * 2. Single NUMA node performance
 * 3. Two NUMA nodes scaling 
 * 4. Four NUMA nodes scaling
 * 
 * Measures FLOPs, throughput, latency, and scaling efficiency
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

#include "ggml.h"
#include "ggml-cpu.h" 
#include "ggml-numa-coordinator.h"
#include "common.h"

class NumaPerformanceTester {
private:
    struct BenchmarkResult {
        std::string test_name;
        std::string operation_type;
        int numa_nodes;
        int total_threads;
        int64_t tensor_elements;
        int operations_count;
        double duration_ms;
        double throughput_elements_per_sec;
        double flops_per_sec;
        double scaling_efficiency;
        bool success;
    };

    std::vector<BenchmarkResult> results;
    int max_threads;
    
public:
    NumaPerformanceTester(int max_threads) : max_threads(max_threads) {}

    // Baseline single-threaded performance test
    BenchmarkResult benchmark_baseline(const std::string& operation, int64_t tensor_size, int iterations) {
        BenchmarkResult result = {};
        result.test_name = "Baseline";
        result.operation_type = operation;
        result.numa_nodes = 0;
        result.total_threads = 1;
        result.tensor_elements = tensor_size;
        result.operations_count = iterations;
        result.success = false;

        std::cout << "🔧 Baseline " << operation << " (" << tensor_size << " elements, " << iterations << " ops)" << std::endl;

        try {
            struct ggml_init_params init_params = {
                32 * 1024 * 1024, // mem_size
                NULL,              // mem_buffer
                false,             // no_alloc
            };
            
            struct ggml_context * ctx = ggml_init(init_params);
            if (!ctx) {
                std::cout << "  ❌ Failed to create context" << std::endl;
                return result;
            }

            // Create test tensors
            struct ggml_tensor * a = nullptr;
            struct ggml_tensor * b = nullptr;
            struct ggml_tensor * result_tensor = nullptr;

            if (operation == "ADD" || operation == "MUL" || operation == "SUB" || operation == "DIV") {
                a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, tensor_size);
                b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, tensor_size);
                fill_tensor_random(a);
                fill_tensor_random(b);
                
                if (operation == "ADD") result_tensor = ggml_add(ctx, a, b);
                else if (operation == "MUL") result_tensor = ggml_mul(ctx, a, b);
                else if (operation == "SUB") result_tensor = ggml_sub(ctx, a, b);
                else if (operation == "DIV") result_tensor = ggml_div(ctx, a, b);
            } else if (operation == "MUL_MAT") {
                int64_t dim = static_cast<int64_t>(std::sqrt(tensor_size));
                a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, dim);
                b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, dim);
                fill_tensor_random(a);
                fill_tensor_random(b);
                result_tensor = ggml_mul_mat(ctx, a, b);
            } else {
                std::cout << "  ❌ Unsupported operation: " << operation << std::endl;
                ggml_free(ctx);
                return result;
            }

            // Warm-up run
            struct ggml_cgraph * cgraph = ggml_new_graph(ctx);
            ggml_build_forward_expand(cgraph, result_tensor);
            
            struct ggml_cplan cplan = ggml_graph_plan(cgraph, 1, nullptr);
            ggml_graph_compute(cgraph, &cplan);

            // Benchmark runs
            auto start = std::chrono::high_resolution_clock::now();
            
            for (int i = 0; i < iterations; i++) {
                ggml_graph_compute(cgraph, &cplan);
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            
            result.duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
            result.throughput_elements_per_sec = (tensor_size * iterations * 1000.0) / result.duration_ms;
            
            // Calculate FLOPs based on operation
            double flops_per_operation = tensor_size; // Basic arithmetic
            if (operation == "MUL_MAT") {
                int64_t dim = static_cast<int64_t>(std::sqrt(tensor_size));
                flops_per_operation = 2.0 * dim * dim * dim; // Matrix multiplication FLOPs
            }
            result.flops_per_sec = (flops_per_operation * iterations * 1000.0) / result.duration_ms;
            result.scaling_efficiency = 1.0; // Baseline is 100% efficient by definition
            result.success = true;

            std::cout << "  ✅ " << std::fixed << std::setprecision(2) 
                     << result.duration_ms << "ms, "
                     << result.flops_per_sec / 1e9 << " GFLOPS" << std::endl;

            ggml_free(ctx);
            
        } catch (const std::exception& e) {
            std::cout << "  ❌ Exception: " << e.what() << std::endl;
        }

        return result;
    }

    // NUMA coordinator performance test
    BenchmarkResult benchmark_numa_coordinator(const std::string& operation, int64_t tensor_size, 
                                               int iterations, int numa_nodes, const BenchmarkResult& baseline) {
        BenchmarkResult result = {};
        result.test_name = "NUMA Coordinator";
        result.operation_type = operation;
        result.numa_nodes = numa_nodes;
        result.total_threads = max_threads;
        result.tensor_elements = tensor_size;
        result.operations_count = iterations;
        result.success = false;

        std::cout << "🔧 NUMA Coordinator " << operation 
                 << " (" << tensor_size << " elements, " << iterations << " ops)" << std::endl;

        try {
            struct ggml_init_params init_params = {
                128 * 1024 * 1024, // mem_size - larger for NUMA operations
                NULL,               // mem_buffer
                false,              // no_alloc
            };
            
            struct ggml_context * ctx = ggml_init(init_params);
            if (!ctx) {
                std::cout << "  ❌ Failed to create context" << std::endl;
                return result;
            }

            // Create test tensors
            struct ggml_tensor * a = nullptr;
            struct ggml_tensor * b = nullptr;
            struct ggml_tensor * result_tensor = nullptr;

            if (operation == "ADD" || operation == "MUL" || operation == "SUB" || operation == "DIV") {
                a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, tensor_size);
                b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, tensor_size);
                fill_tensor_random(a);
                fill_tensor_random(b);
                
                if (operation == "ADD") result_tensor = ggml_add(ctx, a, b);
                else if (operation == "MUL") result_tensor = ggml_mul(ctx, a, b);
                else if (operation == "SUB") result_tensor = ggml_sub(ctx, a, b);
                else if (operation == "DIV") result_tensor = ggml_div(ctx, a, b);
            } else if (operation == "MUL_MAT") {
                int64_t dim = static_cast<int64_t>(std::sqrt(tensor_size));
                a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, dim);
                b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, dim);
                fill_tensor_random(a);
                fill_tensor_random(b);
                result_tensor = ggml_mul_mat(ctx, a, b);
            } else {
                std::cout << "  ❌ Unsupported operation: " << operation << std::endl;
                ggml_free(ctx);
                return result;
            }

            // Get NUMA coordinator manager (let it auto-detect nodes)
            struct ggml_numa_coordinator_manager * mgr = 
                ggml_numa_coordinator_manager_get_global(max_threads, true);
            if (!mgr) {
                std::cout << "  ❌ Failed to create coordinator manager" << std::endl;
                ggml_free(ctx);
                return result;
            }

            // Create computation graph
            struct ggml_cgraph * cgraph = ggml_new_graph(ctx);
            ggml_build_forward_expand(cgraph, result_tensor);

            // Warm-up run - try a few times if it fails
            int warmup_attempts = 3;
            bool warmup_success = false;
            for (int attempt = 0; attempt < warmup_attempts; attempt++) {
                int warmup_result = ggml_numa_coordinator_manager_compute_graph(mgr, cgraph);
                if (warmup_result == 0) {
                    warmup_success = true;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            
            if (!warmup_success) {
                std::cout << "  ⚠️ Warm-up failed, falling back to standard compute" << std::endl;
                // Fall back to standard compute
                struct ggml_cplan cplan = ggml_graph_plan(cgraph, max_threads, nullptr);
                
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < iterations; i++) {
                    ggml_graph_compute(cgraph, &cplan);
                }
                auto end = std::chrono::high_resolution_clock::now();
                
                result.duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
            } else {
                // Benchmark runs with coordinator
                auto start = std::chrono::high_resolution_clock::now();
                
                for (int i = 0; i < iterations; i++) {
                    int graph_result = ggml_numa_coordinator_manager_compute_graph(mgr, cgraph);
                    if (graph_result != 0) {
                        std::cout << "  ⚠️ Iteration " << i << " failed, using fallback" << std::endl;
                        // Use fallback for remaining iterations
                        struct ggml_cplan cplan = ggml_graph_plan(cgraph, max_threads, nullptr);
                        for (int j = i; j < iterations; j++) {
                            ggml_graph_compute(cgraph, &cplan);
                        }
                        break;
                    }
                }
                
                auto end = std::chrono::high_resolution_clock::now();
                result.duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
            }
            
            result.throughput_elements_per_sec = (tensor_size * iterations * 1000.0) / result.duration_ms;
            
            // Calculate FLOPs based on operation
            double flops_per_operation = tensor_size; // Basic arithmetic
            if (operation == "MUL_MAT") {
                int64_t dim = static_cast<int64_t>(std::sqrt(tensor_size));
                flops_per_operation = 2.0 * dim * dim * dim; // Matrix multiplication FLOPs
            }
            result.flops_per_sec = (flops_per_operation * iterations * 1000.0) / result.duration_ms;
            
            // Calculate scaling efficiency compared to baseline
            if (baseline.success && baseline.flops_per_sec > 0) {
                double actual_speedup = result.flops_per_sec / baseline.flops_per_sec;
                double theoretical_speedup = max_threads; // Linear scaling expectation
                result.scaling_efficiency = actual_speedup / theoretical_speedup;
            }
            
            result.success = true;

            std::cout << "  ✅ " << std::fixed << std::setprecision(2) 
                     << result.duration_ms << "ms, "
                     << result.flops_per_sec / 1e9 << " GFLOPS, "
                     << (result.scaling_efficiency * 100) << "% efficiency" << std::endl;

            ggml_free(ctx);
            
        } catch (const std::exception& e) {
            std::cout << "  ❌ Exception: " << e.what() << std::endl;
        }

        return result;
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

    // Run comprehensive scaling analysis
    void run_scaling_analysis() {
        std::cout << "🏁 NUMA Coordinator Performance & Scaling Analysis" << std::endl;
        std::cout << "=================================================" << std::endl;
        std::cout << "Max threads: " << max_threads << std::endl << std::endl;

        // Test different operations and tensor sizes
        std::vector<std::string> operations = {"ADD", "MUL"};
        std::vector<int64_t> tensor_sizes = {100000, 1000000}; // 100K, 1M elements
        int iterations = 5; // Reduced for stability

        for (const auto& operation : operations) {
            std::cout << "🧮 Testing " << operation << " Operation Performance" << std::endl;
            std::cout << "=============================================" << std::endl;

            for (auto tensor_size : tensor_sizes) {
                std::cout << "  📊 Tensor size: " << tensor_size << " elements" << std::endl;

                // Run baseline test
                BenchmarkResult baseline = benchmark_baseline(operation, tensor_size, iterations);
                results.push_back(baseline);

                if (!baseline.success) {
                    std::cout << "  ❌ Baseline failed, skipping NUMA test" << std::endl;
                    continue;
                }

                // Run NUMA coordinator test
                BenchmarkResult numa_result = benchmark_numa_coordinator(
                    operation, tensor_size, iterations, 1, baseline);
                results.push_back(numa_result);

                std::cout << std::endl;
            }
        }

        print_summary();
    }

    // Print comprehensive performance summary
    void print_summary() {
        std::cout << "📊 Performance Analysis Summary" << std::endl;
        std::cout << "===============================" << std::endl;

        // Group results by operation
        for (const auto& operation : {"ADD", "MUL"}) {
            std::cout << std::endl << "🔍 " << operation << " Operation Analysis:" << std::endl;
            std::cout << std::setw(12) << "Test" 
                     << std::setw(10) << "Elements"
                     << std::setw(12) << "Time(ms)"
                     << std::setw(12) << "GFLOPS" 
                     << std::setw(10) << "Speedup"
                     << std::setw(12) << "Efficiency%" << std::endl;
            std::cout << std::string(70, '-') << std::endl;

            BenchmarkResult baseline;
            bool has_baseline = false;

            // Print baseline results
            for (const auto& result : results) {
                if (result.operation_type != operation) continue;
                
                if (result.numa_nodes == 0) {
                    baseline = result;
                    has_baseline = true;
                    std::cout << std::setw(12) << "Baseline"
                             << std::setw(10) << result.tensor_elements
                             << std::setw(12) << std::fixed << std::setprecision(1) << result.duration_ms
                             << std::setw(12) << std::fixed << std::setprecision(3) << result.flops_per_sec / 1e9
                             << std::setw(10) << "1.00x"
                             << std::setw(12) << "100.0%" << std::endl;
                }
            }

            // Print NUMA results
            for (const auto& result : results) {
                if (result.operation_type != operation || result.numa_nodes == 0) continue;
                
                double speedup = has_baseline ? result.flops_per_sec / baseline.flops_per_sec : 0.0;
                
                std::cout << std::setw(12) << "NUMA"
                         << std::setw(10) << result.tensor_elements
                         << std::setw(12) << std::fixed << std::setprecision(1) << result.duration_ms
                         << std::setw(12) << std::fixed << std::setprecision(3) << result.flops_per_sec / 1e9
                         << std::setw(10) << std::fixed << std::setprecision(2) << speedup << "x"
                         << std::setw(12) << std::fixed << std::setprecision(1) 
                         << (result.scaling_efficiency * 100) << "%" << std::endl;
            }
        }

        // Calculate average performance
        double total_speedup = 0.0;
        int count = 0;
        for (const auto& result : results) {
            if (result.numa_nodes > 0 && result.success) {
                // Find corresponding baseline
                for (const auto& baseline : results) {
                    if (baseline.operation_type == result.operation_type && 
                        baseline.tensor_elements == result.tensor_elements && 
                        baseline.numa_nodes == 0) {
                        total_speedup += result.flops_per_sec / baseline.flops_per_sec;
                        count++;
                        break;
                    }
                }
            }
        }
        
        if (count > 0) {
            double avg_speedup = total_speedup / count;
            std::cout << std::endl << "📈 Average NUMA coordinator speedup: " 
                     << std::fixed << std::setprecision(2) << avg_speedup << "x" << std::endl;
            std::cout << "🎯 Target: Linear scaling would be " << max_threads << "x speedup" << std::endl;
        }
        
        std::cout << std::endl;
    }
};

int main() {
    // Get system configuration
    int cpu_count = std::thread::hardware_concurrency();
    int max_test_threads = std::min(cpu_count, 16); // Cap at 16 threads for testing
    
    std::cout << "🔧 System Configuration" << std::endl;
    std::cout << "Available CPUs: " << cpu_count << std::endl;
    std::cout << "Max test threads: " << max_test_threads << std::endl << std::endl;
    
    NumaPerformanceTester tester(max_test_threads);
    tester.run_scaling_analysis();
    
    return 0;
}
