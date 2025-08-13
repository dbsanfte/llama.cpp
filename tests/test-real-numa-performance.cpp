/**
 * Real NUMA Hardware Performance Test
 * 
 * This test specifically validates NUMA mirror performance benefits on real NUMA hardware.
 * It skips virtual NUMA testing and focuses purely on hardware-based NUMA scaling.
 * 
 * Test Strategy:
 * 1. Detect real NUMA hardware availability and topology
 * 2. Run matrix multiplication benchmarks with different NUMA node configurations:
 *    - Single NUMA node (max_numa_nodes=1) - baseline with full CPU utilization
 *    - Dual NUMA nodes (max_numa_nodes=2) - if hardware supports it
 *    - Quad NUMA nodes (max_numa_nodes=4) - if hardware supports it
 * 3. Compare performance to identify NUMA mirroring benefits
 * 4. Debug performance issues if no speedup is observed
 * 
 * Matrix Operations Tested:
 * - Large matrix multiplication (data parallel workloads)
 * - Multiple batch sizes to test memory bandwidth scaling
 * - Various matrix dimensions to test compute vs memory bottlenecks
 * 
 * Usage:
 *   ./test-real-numa-performance [--quick] [--debug]
 * 
 * Requirements:
 *   - Real NUMA hardware (numa_available() >= 0)
 *   - GGML_NUMA_MIRROR enabled
 */

#include <chrono>
#include <iostream>
#include <vector>
#include <iomanip>
#include <thread>
#include <cmath>
#include <random>
#include <algorithm>
#include <map>
#include <string>
#include <cstring>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-cpu/ggml-numa-coordinator.h"
#include "common.h"

#ifdef GGML_NUMA_MIRROR
#include <numa.h>
#include <numaif.h>
#endif

// Test configuration
struct RealNUMATestConfig {
    bool quick_mode = false;
    bool debug_mode = false;
    
    // Matrix dimensions for different test scenarios
    struct MatrixTest {
        int m, n, k;           // Matrix dimensions: C[m,n] = A[m,k] * B[k,n]
        const char* description;
        bool test_batching;    // Whether to test multiple batch sizes
    };
    
    std::vector<MatrixTest> matrix_tests;
    std::vector<int> batch_sizes = {1, 4, 8, 16, 32};
    int iterations = 10;  // Increased for better accuracy
    
    RealNUMATestConfig() {
        // Default test configurations focusing on different compute/memory profiles
        matrix_tests = {
            // Compute-bound scenarios
            {1024, 1024, 1024, "Large Square Matrix (1K x 1K)", false},
            {2048, 2048, 512, "Wide Matrix Multiplication", false},
            {512, 2048, 2048, "Tall Matrix Multiplication", false},
            
            // Memory bandwidth scenarios  
            {4096, 256, 256, "Memory Bandwidth Test", false},
            
            // Data parallel batching scenarios
            {512, 512, 512, "Batch Processing Test", true}
        };
    }
    
    void set_quick_mode() {
        quick_mode = true;
        iterations = 3;  // Reduced but still sufficient for quick testing
        matrix_tests = {
            {1024, 1024, 1024, "Quick Square Matrix", false},
            {512, 512, 512, "Quick Batch Test", true}
        };
        batch_sizes = {1, 4, 8};
    }

    void set_medium_mode() {
        quick_mode = false;
        iterations = 5;  // Balanced for accuracy vs speed
        matrix_tests = {
            {1024, 1024, 1024, "Large Square (1024x1024)", false},
            {2048, 2048, 2048, "XL Square (2048x2048)", false},
            {1024, 4096, 2048, "Wide Matrix (1024x4096)", false}
        };
        batch_sizes = {1, 4, 8};
    }
};

// Performance measurement utilities
class PerformanceTimer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration = std::chrono::duration<double>;
    
private:
    TimePoint start_time;
    
public:
    void start() { start_time = Clock::now(); }
    
    double elapsed_ms() {
        auto end_time = Clock::now();
        Duration elapsed = end_time - start_time;
        return elapsed.count() * 1000.0; // Convert to milliseconds
    }
    
    double elapsed_seconds() {
        auto end_time = Clock::now();
        Duration elapsed = end_time - start_time;
        return elapsed.count();
    }
};

// NUMA hardware detection and topology analysis
class RealNUMADetector {
public:
    struct NUMATopology {
        int total_numa_nodes = 0;
        int total_cpus = 0;
        std::vector<int> cpus_per_node;
        bool is_real_numa = false;
    };
    
    static NUMATopology detect() {
        NUMATopology topology;
        
#ifdef GGML_NUMA_MIRROR
        if (numa_available() < 0) {
            std::cout << "❌ NUMA not available on this system" << std::endl;
            return topology;
        }
        
        topology.total_numa_nodes = numa_num_configured_nodes();
        topology.total_cpus = numa_num_configured_cpus();
        topology.is_real_numa = true;
        
        std::cout << "🖥️  NUMA Hardware Detected:" << std::endl;
        std::cout << "   Total NUMA nodes: " << topology.total_numa_nodes << std::endl;
        std::cout << "   Total CPUs: " << topology.total_cpus << std::endl;
        
        // Get CPU count per NUMA node
        for (int node = 0; node < topology.total_numa_nodes; node++) {
            struct bitmask* cpus = numa_allocate_cpumask();
            if (numa_node_to_cpus(node, cpus) == 0) {
                int cpu_count = 0;
                for (int cpu = 0; cpu < topology.total_cpus; cpu++) {
                    if (numa_bitmask_isbitset(cpus, cpu)) {
                        cpu_count++;
                    }
                }
                topology.cpus_per_node.push_back(cpu_count);
                std::cout << "   Node " << node << ": " << cpu_count << " CPUs" << std::endl;
            }
            numa_free_cpumask(cpus);
        }
        
        // Validate that we have real multi-node NUMA
        if (topology.total_numa_nodes < 2) {
            std::cout << "⚠️  Single NUMA node detected - limited scaling tests available" << std::endl;
        }
        
#else
        std::cout << "❌ GGML_NUMA_MIRROR not enabled - cannot run NUMA tests" << std::endl;
#endif
        
        return topology;
    }
};

// Global results collection for summary table
struct TestResult {
    std::string test_name;
    std::string matrix_config;  // e.g., "1024x1024x1024" or "batch=8"
    int numa_nodes;
    double time_ms;
    double gflops;
    double memory_bw_gb_s;
    double speedup;
    
    TestResult(const std::string& name, const std::string& config, int nodes, 
               double time, double gf, double bw, double sp = 1.0)
        : test_name(name), matrix_config(config), numa_nodes(nodes), 
          time_ms(time), gflops(gf), memory_bw_gb_s(bw), speedup(sp) {}
};

class SummaryCollector {
public:
    static std::vector<TestResult> all_results;
    
    static void add_result(const TestResult& result) {
        all_results.push_back(result);
    }
    
    static void display_final_summary();
};

std::vector<TestResult> SummaryCollector::all_results;

void SummaryCollector::display_final_summary() {
    if (all_results.empty()) {
        std::cout << "\n❌ No results collected for summary" << std::endl;
        return;
    }
    
    std::cout << "\n" << std::string(115, '=') << std::endl;
    std::cout << "🏆 COMPREHENSIVE NUMA PERFORMANCE SUMMARY" << std::endl;
    std::cout << std::string(115, '=') << std::endl;
    
    // Group results by test name
    std::map<std::string, std::vector<TestResult>> grouped_results;
    for (const auto& result : all_results) {
        grouped_results[result.test_name].push_back(result);
    }
    
    // Display summary table with improved formatting
    std::cout << std::left;
    std::cout << std::setw(28) << "Test Name" 
              << std::setw(22) << "Matrix Config"
              << std::setw(6) << "NUMA"
              << std::setw(11) << "Time (ms)"
              << std::setw(9) << "GFLOPS"
              << std::setw(12) << "Memory BW"
              << std::setw(10) << "Speedup" << std::endl;
    std::cout << std::string(115, '-') << std::endl;
    
    for (const auto& [test_name, results] : grouped_results) {
        bool first_in_group = true;
        for (const auto& result : results) {
            // Improved memory bandwidth display (handle small values better)
            std::string mem_bw_str;
            if (result.memory_bw_gb_s >= 1.0) {
                mem_bw_str = std::to_string(static_cast<int>(result.memory_bw_gb_s * 10) / 10.0) + " GB/s";
            } else if (result.memory_bw_gb_s >= 0.1) {
                mem_bw_str = std::to_string(static_cast<int>(result.memory_bw_gb_s * 100) / 100.0) + " GB/s";
            } else {
                mem_bw_str = std::to_string(static_cast<int>(result.memory_bw_gb_s * 1000) / 10.0) + " MB/s";
            }
            
            std::cout << std::setw(28) << (first_in_group ? test_name : "")
                      << std::setw(22) << (first_in_group ? result.matrix_config : "")
                      << std::setw(6) << result.numa_nodes
                      << std::setw(11) << std::fixed << std::setprecision(1) << result.time_ms
                      << std::setw(9) << std::fixed << std::setprecision(1) << result.gflops
                      << std::setw(12) << mem_bw_str
                      << std::setw(10) << std::fixed << std::setprecision(2) << result.speedup << "x" << std::endl;
            first_in_group = false;
        }
        std::cout << std::string(115, '-') << std::endl;
    }
    
    // Performance analysis
    std::cout << "\n📈 OVERALL PERFORMANCE ANALYSIS:" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    // Find best speedups for each test
    for (const auto& [test_name, results] : grouped_results) {
        if (results.size() <= 1) continue;
        
        auto best_result = std::max_element(results.begin(), results.end(),
            [](const TestResult& a, const TestResult& b) {
                return a.speedup < b.speedup;
            });
        
        std::cout << "• " << test_name << ":" << std::endl;
        std::cout << "  Best speedup: " << std::fixed << std::setprecision(2) 
                  << best_result->speedup << "x with " << best_result->numa_nodes << " NUMA nodes" << std::endl;
        
        if (best_result->speedup > 1.5) {
            std::cout << "  ✅ Excellent NUMA scaling!" << std::endl;
        } else if (best_result->speedup > 1.2) {
            std::cout << "  ✅ Good NUMA scaling" << std::endl;
        } else if (best_result->speedup > 1.1) {
            std::cout << "  ⚠️  Modest NUMA benefit" << std::endl;
        } else {
            std::cout << "  ❌ Limited NUMA benefit - investigate bottlenecks" << std::endl;
        }
        std::cout << std::endl;
    }
    
    // Overall statistics
    double total_tests = all_results.size();
    double avg_speedup = 0.0;
    double max_speedup = 0.0;
    int best_numa_nodes = 1;
    
    for (const auto& result : all_results) {
        avg_speedup += result.speedup;
        if (result.speedup > max_speedup) {
            max_speedup = result.speedup;
            best_numa_nodes = result.numa_nodes;
        }
    }
    avg_speedup /= total_tests;
    
    std::cout << "🔍 SUMMARY STATISTICS:" << std::endl;
    std::cout << "   Total test configurations: " << static_cast<int>(total_tests) << std::endl;
    std::cout << "   Average speedup across all tests: " << std::fixed << std::setprecision(2) << avg_speedup << "x" << std::endl;
    std::cout << "   Maximum speedup achieved: " << std::fixed << std::setprecision(2) << max_speedup << "x" << std::endl;
    std::cout << "   Best performing configuration: " << best_numa_nodes << " NUMA nodes" << std::endl;
    
    std::cout << std::string(115, '=') << std::endl;
}

// Matrix multiplication benchmark runner
class MatrixBenchmark {
private:
    RealNUMATestConfig config;
    RealNUMADetector::NUMATopology topology;
    bool is_virtual_numa;
    
public:
    MatrixBenchmark(const RealNUMATestConfig& cfg, const RealNUMADetector::NUMATopology& topo, bool virtual_numa = false) 
        : config(cfg), topology(topo), is_virtual_numa(virtual_numa) {}
    
    struct BenchmarkResult {
        double avg_time_ms = 0.0;
        double std_dev_ms = 0.0;
        double gflops = 0.0;
        double memory_bandwidth_gb_s = 0.0;
        int numa_nodes_used = 0;
        std::string config_description;
    };
    
    // Run matrix multiplication with specific NUMA configuration
    BenchmarkResult run_matrix_multiply(int m, int n, int k, int numa_nodes, int batch_size = 1) {
        BenchmarkResult result;
        result.numa_nodes_used = numa_nodes;
        result.config_description = std::to_string(numa_nodes) + " NUMA nodes";
        
        // CRITICAL FIX: Use no_alloc = true to enable NUMA-aware tensor allocation
        // The old approach allocated all tensor data from a single context buffer on one NUMA node!
        size_t ctx_size = ggml_tensor_overhead() * (3 * batch_size) + ggml_graph_overhead();
        
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,  // FIXED: Enable NUMA-aware allocation
        };
        
        std::vector<double> times;
        times.reserve(config.iterations);
        
        for (int iter = 0; iter < config.iterations; iter++) {
            struct ggml_context* ctx = ggml_init(params);
            if (!ctx) {
                std::cerr << "❌ Failed to initialize GGML context" << std::endl;
                return result;
            }
            
            // Create matrices - tensors will have no data allocated yet
            std::vector<ggml_tensor*> tensors_a, tensors_b, tensors_c;
            
            for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
                // Create tensors following GGML convention: C[m,n] = A[m,k] * B[k,n]
                // A is (k, m) and B is (k, n) where k is the shared inner dimension (width)
                struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
                struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);  
                struct ggml_tensor* c = ggml_mul_mat(ctx, a, b);
                
                tensors_a.push_back(a);
                tensors_b.push_back(b);
                tensors_c.push_back(c);
            }
            
            // CRITICAL FIX: Use NUMA-aware backend buffer allocation
            // This ensures tensor data is distributed across NUMA nodes properly
            ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, ggml_backend_cpu_init());
            if (!buffer) {
                std::cerr << "❌ Failed to allocate NUMA-aware tensor buffer" << std::endl;
                ggml_free(ctx);
                return result;
            }
            
            if (config.debug_mode) {
                std::cout << "✅ NUMA-aware tensor buffer allocated successfully" << std::endl;
            }
            
            
            // Initialize matrices with random data
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<float> dist(0.0f, 1.0f);
            
            for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
                // Allocate and initialize tensor A data
                size_t size_a = ggml_nbytes(tensors_a[batch_idx]);
                std::vector<float> data_a(ggml_nelements(tensors_a[batch_idx]));
                for (size_t i = 0; i < data_a.size(); i++) {
                    data_a[i] = dist(gen);
                }
                ggml_backend_tensor_set(tensors_a[batch_idx], data_a.data(), 0, size_a);
                
                // Allocate and initialize tensor B data
                size_t size_b = ggml_nbytes(tensors_b[batch_idx]);
                std::vector<float> data_b(ggml_nelements(tensors_b[batch_idx]));
                for (size_t i = 0; i < data_b.size(); i++) {
                    data_b[i] = dist(gen);
                }
                ggml_backend_tensor_set(tensors_b[batch_idx], data_b.data(), 0, size_b);
                
                if (config.debug_mode) {
                    std::cout << "   Batch " << batch_idx << ": A=" << ggml_nelements(tensors_a[batch_idx])
                              << " elements, B=" << ggml_nelements(tensors_b[batch_idx]) << " elements" << std::endl;
                }
            }
            
            // Set up NUMA configuration
            struct ggml_threadpool_params tpp;
            ggml_threadpool_params_init(&tpp, -1); // Auto-detect threads
            
            // Configure based on whether we're using real or virtual NUMA
            if (!is_virtual_numa && topology.total_numa_nodes >= 2) {
                // Real NUMA hardware
                tpp.force_multi_socket = false;
                tpp.max_numa_nodes = numa_nodes;
            } else {
                // Virtual NUMA simulation
                tpp.force_multi_socket = true;
                tpp.max_numa_nodes = numa_nodes;
            }
            
            memset(tpp.cpumask, false, sizeof(tpp.cpumask)); // Auto-optimization
            
            if (config.debug_mode) {
                std::cout << "🔧 Running with " << numa_nodes << " NUMA nodes, batch size " << batch_size << std::endl;
            }
            
            // Build computation graph
            struct ggml_cgraph* gf = ggml_new_graph(ctx);
            for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
                ggml_build_forward_expand(gf, tensors_c[batch_idx]);
            }
            
            // Create NUMA coordinator manager
            struct ggml_numa_coordinator_manager* mgr = 
                ggml_numa_coordinator_manager_new_with_params(&tpp);
            if (!mgr) {
                std::cerr << "❌ Failed to create NUMA coordinator manager" << std::endl;
                ggml_backend_buffer_free(buffer);
                ggml_free(ctx);
                return result;
            }
            
            // Time the computation
            PerformanceTimer timer;
            timer.start();
            
            // Execute the computation using NUMA coordinator
            int compute_result = ggml_numa_coordinator_manager_compute_graph(mgr, gf);
            if (compute_result != 0) {
                std::cerr << "❌ NUMA computation failed with code " << compute_result << std::endl;
                ggml_numa_coordinator_manager_free(mgr);
                ggml_backend_buffer_free(buffer);
                ggml_free(ctx);
                return result;
            }
            
            // Wait for completion
            ggml_numa_coordinator_manager_wait_for_completion(mgr);
            
            double elapsed_ms = timer.elapsed_ms();
            times.push_back(elapsed_ms);
            
            // Cleanup
            ggml_numa_coordinator_manager_free(mgr);
            ggml_backend_buffer_free(buffer);  // Free NUMA-aware buffer
            
            if (config.debug_mode) {
                std::cout << "   Iteration " << iter + 1 << ": " << std::fixed << std::setprecision(2) 
                          << elapsed_ms << " ms" << std::endl;
            }
            
            ggml_free(ctx);
        }
        
        // Calculate statistics
        double sum = 0.0;
        for (double time : times) {
            sum += time;
        }
        result.avg_time_ms = sum / times.size();
        
        // Calculate standard deviation
        double variance = 0.0;
        for (double time : times) {
            double diff = time - result.avg_time_ms;
            variance += diff * diff;
        }
        result.std_dev_ms = std::sqrt(variance / times.size());
        
        // Calculate performance metrics
        int64_t total_flops = (int64_t)m * n * k * 2 * batch_size; // 2 flops per multiply-add
        result.gflops = (total_flops / 1e9) / (result.avg_time_ms / 1000.0);
        
        // Estimate memory bandwidth (rough approximation)
        int64_t memory_ops = (int64_t)(m * k + k * n + m * n) * sizeof(float) * batch_size;
        result.memory_bandwidth_gb_s = (memory_ops / 1e9) / (result.avg_time_ms / 1000.0);
        
        return result;
    }
    
    // Run comprehensive benchmark comparing different NUMA configurations
    void run_comprehensive_benchmark() {
        std::cout << "\n🧪 Real NUMA Hardware Performance Benchmark" << std::endl;
        std::cout << "Collecting results for final summary table..." << std::endl;
        
        // Test configurations
        std::vector<std::pair<int, int>> numa_configs;
        if (topology.total_numa_nodes >= 1) numa_configs.push_back({1, 1});
        if (topology.total_numa_nodes >= 2) numa_configs.push_back({2, 2});  
        if (topology.total_numa_nodes >= 4) numa_configs.push_back({4, 4});
        if (topology.total_numa_nodes >= 8) numa_configs.push_back({8, 8});
        
        // Matrix multiplication tests
        struct MatrixTest {
            int m, n, k;
            std::string description;
        };
        
        std::vector<MatrixTest> matrix_tests;
        
        if (config.quick_mode) {
            matrix_tests = {
                {1024, 1024, 1024, "Quick Square (1024x1024)"}
            };
        } else {
            matrix_tests = {
                {1024, 1024, 1024, "Large Square (1024x1024)"},
                {2048, 2048, 2048, "XL Square (2048x2048)"},
                {4096, 1024, 2048, "Deep Matrix (4096x1024)"},
                {1024, 4096, 2048, "Wide Matrix (1024x4096)"},
                {2048, 512, 1024, "Wide Rect (2048x512)"},
                {512, 2048, 1024, "Tall Rect (512x2048)"}
            };
        }
        
        // Run matrix tests with single operations first
        for (const auto& matrix_test : matrix_tests) {
            std::cout << "\n📊 Testing " << matrix_test.description << "..." << std::endl;
            
            if (numa_configs.size() > 1) {
                std::map<int, BenchmarkResult> results;
                for (const auto& [max_nodes, numa_nodes] : numa_configs) {
                    std::cout << "   Running with " << numa_nodes << " NUMA nodes..." << std::endl;
                    results[numa_nodes] = run_matrix_multiply(matrix_test.m, matrix_test.n, matrix_test.k, numa_nodes);
                }
                
                // Store results in global collector
                store_results_for_summary(results, matrix_test.description, 
                    std::to_string(matrix_test.m) + "x" + std::to_string(matrix_test.n) + "x" + std::to_string(matrix_test.k));
            }
        }
        
        // Run compute-intensive batch tests only for selected matrices
        if (!config.quick_mode && numa_configs.size() > 1) {
            std::cout << "\n🔥 Running compute-intensive batch tests..." << std::endl;
            
            // Only test batch processing on matrices that showed good single scaling
            std::vector<MatrixTest> batch_test_matrices = {
                {1024, 1024, 1024, "Large Square (1024x1024)"},
                {2048, 2048, 2048, "XL Square (2048x2048)"},
                {1024, 4096, 2048, "Wide Matrix (1024x4096)"}
            };
            
            for (const auto& batch_matrix : batch_test_matrices) {
                for (int batch_size : {4, 8, 16}) {
                    std::cout << "\n   Testing " << batch_matrix.description << " with batch size " << batch_size << "..." << std::endl;
                    std::map<int, BenchmarkResult> batch_results;
                    for (const auto& [max_nodes, numa_nodes] : numa_configs) {
                        batch_results[numa_nodes] = run_matrix_multiply(batch_matrix.m, batch_matrix.n, batch_matrix.k, numa_nodes, batch_size);
                    }
                    
                    // Store batch results in global collector
                    store_results_for_summary(batch_results, batch_matrix.description + " (batch=" + std::to_string(batch_size) + ")",
                        std::to_string(batch_matrix.m) + "x" + std::to_string(batch_matrix.n) + "x" + std::to_string(batch_matrix.k) + " batch=" + std::to_string(batch_size));
                }
            }
        }
    }

private:
    // Store results in global collector for final summary
    void store_results_for_summary(const std::map<int, BenchmarkResult>& results, 
                                   const std::string& test_name, const std::string& matrix_config) {
        double baseline_time = 0.0;
        bool first = true;
        
        for (const auto& [numa_nodes, result] : results) {
            if (first) {
                baseline_time = result.avg_time_ms;
                first = false;
            }
            
            double speedup = baseline_time / result.avg_time_ms;
            
            SummaryCollector::add_result(TestResult(
                test_name,
                matrix_config,
                numa_nodes,
                result.avg_time_ms,
                result.gflops,
                result.memory_bandwidth_gb_s,
                speedup
            ));
        }
    }

    void display_results(const std::map<int, BenchmarkResult>& results, const std::string& test_name) {
        std::cout << "\n   Results for " << test_name << ":" << std::endl;
        std::cout << "   " << std::string(60, '-') << std::endl;
        std::cout << "   NUMA Nodes | Time (ms)  | GFLOPS | Memory BW | Speedup" << std::endl;
        std::cout << "   " << std::string(60, '-') << std::endl;
        
        double baseline_time = 0.0;
        bool first = true;
        
        for (const auto& [numa_nodes, result] : results) {
            if (first) {
                baseline_time = result.avg_time_ms;
                first = false;
            }
            
            double speedup = baseline_time / result.avg_time_ms;
            
            std::cout << "   " << std::setw(10) << numa_nodes 
                      << " | " << std::setw(9) << std::fixed << std::setprecision(1) << result.avg_time_ms
                      << " | " << std::setw(6) << std::fixed << std::setprecision(1) << result.gflops
                      << " | " << std::setw(8) << std::fixed << std::setprecision(1) << result.memory_bandwidth_gb_s << " GB/s"
                      << " | " << std::setw(6) << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
        }
        
        // Analysis
        if (results.size() > 1) {
            auto baseline = results.begin();
            auto best = std::max_element(results.begin(), results.end(),
                [](const auto& a, const auto& b) {
                    return a.second.gflops < b.second.gflops;
                });
            
            double max_speedup = baseline->second.avg_time_ms / best->second.avg_time_ms;
            
            std::cout << "\n   📈 Analysis:" << std::endl;
            if (max_speedup > 1.1) {  // >10% improvement
                std::cout << "   ✅ NUMA scaling working! Best speedup: " << std::fixed << std::setprecision(2) 
                          << max_speedup << "x with " << best->first << " NUMA nodes" << std::endl;
            } else if (max_speedup > 1.05) {  // 5-10% improvement
                std::cout << "   ⚠️  Modest NUMA benefit: " << std::fixed << std::setprecision(2) 
                          << max_speedup << "x with " << best->first << " NUMA nodes" << std::endl;
            } else {
                std::cout << "   ❌ No significant NUMA speedup detected (max " << std::fixed << std::setprecision(2) 
                          << max_speedup << "x)" << std::endl;
                std::cout << "   🔍 Possible issues to investigate:" << std::endl;
                std::cout << "      - Memory bandwidth already saturated" << std::endl;
                std::cout << "      - Thread synchronization overhead" << std::endl;
                std::cout << "      - Insufficient workload size for scaling" << std::endl;
                std::cout << "      - NUMA affinity not properly configured" << std::endl;
            }
        }
    }
};

// Main test function
int run_real_numa_performance_test(int argc, char** argv) {
    RealNUMATestConfig config;
    
    // Parse command line arguments
    bool force_virtual_numa = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--quick") == 0) {
            config.set_quick_mode();
            std::cout << "🏃 Quick mode enabled" << std::endl;
        } else if (strcmp(argv[i], "--debug") == 0) {
            config.debug_mode = true;
            std::cout << "🔍 Debug mode enabled" << std::endl;
        } else if (strcmp(argv[i], "--medium") == 0) {
            config.set_medium_mode();
            std::cout << "⚡ Medium mode enabled (faster than full but more comprehensive than quick)" << std::endl;
        } else if (strcmp(argv[i], "--force-virtual") == 0) {
            force_virtual_numa = true;
            std::cout << "⚠️  Force virtual NUMA mode enabled" << std::endl;
        } else if (strcmp(argv[i], "--help") == 0) {
            std::cout << "Real NUMA Hardware Performance Test\n"
                      << "Usage: " << argv[0] << " [OPTIONS]\n"
                      << "Options:\n"
                      << "  --quick           Run quick test with smaller matrices (fastest)\n"
                      << "  --medium          Run medium test with selected large matrices (balanced)\n"
                      << "  --debug           Enable debug output\n"
                      << "  --force-virtual   Force test with virtual NUMA (for demo purposes)\n"
                      << "  --help            Show this help message\n"
                      << "\n"
                      << "Default: Full comprehensive test with all matrix sizes and batch processing\n";
            return 0;
        }
    }
    
    std::cout << "🚀 Real NUMA Hardware Performance Test" << std::endl;
    std::cout << "======================================" << std::endl;
    
    // Detect NUMA hardware
    auto topology = RealNUMADetector::detect();
    if (!topology.is_real_numa && !force_virtual_numa) {
        std::cout << "\n❌ This test requires real NUMA hardware." << std::endl;
        std::cout << "\n💡 Alternative testing suggestions:" << std::endl;
        std::cout << "• Run this test on a multi-socket server with real NUMA hardware" << std::endl;
        std::cout << "• Use numactl to simulate NUMA: numactl --cpunodebind=0 --membind=0 <command>" << std::endl;
        std::cout << "• Test with force_multi_socket mode using test-comprehensive-numa-performance" << std::endl;
        std::cout << "• Use --force-virtual flag to run virtual NUMA demo" << std::endl;
        std::cout << "\n🔍 Current system NUMA status:" << std::endl;
#ifdef GGML_NUMA_MIRROR
        std::cout << "• NUMA library available: " << (numa_available() >= 0 ? "YES" : "NO") << std::endl;
        std::cout << "• Hardware NUMA nodes: " << numa_num_configured_nodes() << std::endl;
        std::cout << "• Total CPUs: " << numa_num_configured_cpus() << std::endl;
#else
        std::cout << "• NUMA library available: NO (GGML_NUMA_MIRROR not enabled)" << std::endl;
        std::cout << "• Hardware NUMA nodes: N/A" << std::endl;
        std::cout << "• Total CPUs: " << std::thread::hardware_concurrency() << std::endl;
#endif
        
        // Show a sample of what the test would do
        std::cout << "\n📝 This test would benchmark:" << std::endl;
        std::cout << "• Matrix multiplication performance with 1, 2, and 4 NUMA nodes" << std::endl;
        std::cout << "• Data parallel workloads across NUMA boundaries" << std::endl;
        std::cout << "• Memory bandwidth scaling with NUMA-aware allocation" << std::endl;
        std::cout << "• Thread affinity and CPU mask optimization benefits" << std::endl;
        
        return 1;
    }
    
    if (force_virtual_numa) {
        std::cout << "\n⚠️  Running in VIRTUAL NUMA mode for demonstration purposes" << std::endl;
        std::cout << "This will use force_multi_socket to simulate NUMA scaling." << std::endl;
        std::cout << "Results may not reflect real NUMA hardware performance!" << std::endl;
        
        // Override topology for virtual NUMA demo
        topology.is_real_numa = true; // Allow test to proceed
        topology.total_numa_nodes = 2; // Simulate 2 NUMA nodes
        topology.total_cpus = std::thread::hardware_concurrency();
        topology.cpus_per_node = {topology.total_cpus / 2, topology.total_cpus / 2};
    }
    
    if (topology.total_numa_nodes < 2) {
        std::cout << "\n⚠️  Only one NUMA node detected. Limited testing available." << std::endl;
        std::cout << "This test will only benchmark single-node performance." << std::endl;
    }
    
    // Run benchmarks
    MatrixBenchmark benchmark(config, topology, force_virtual_numa);
    benchmark.run_comprehensive_benchmark();
    
    // Display final comprehensive summary
    SummaryCollector::display_final_summary();
    
    std::cout << "\n🎯 Test Complete!" << std::endl;
    std::cout << "\nIf you see limited NUMA scaling benefits, consider:" << std::endl;
    std::cout << "• Increasing matrix sizes to become more compute-bound" << std::endl;
    std::cout << "• Testing with different batch sizes for data parallelism" << std::endl;
    std::cout << "• Checking NUMA memory allocation with 'numactl --hardware'" << std::endl;
    std::cout << "• Verifying thread affinity with 'htop' during execution" << std::endl;
    
    return 0;
}

int main(int argc, char** argv) {
    return run_real_numa_performance_test(argc, argv);
}
