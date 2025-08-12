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
    int iterations = 5;
    
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
        iterations = 2;
        matrix_tests = {
            {512, 512, 512, "Quick Square Matrix", false},
            {256, 256, 256, "Quick Batch Test", true}
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
        
        // Initialize GGML context
        size_t ctx_size = ggml_tensor_overhead() * (3 * batch_size) + ggml_graph_overhead();
        ctx_size += (size_t)m * k * sizeof(float) * batch_size;  // Matrix A
        ctx_size += (size_t)k * n * sizeof(float) * batch_size;  // Matrix B  
        ctx_size += (size_t)m * n * sizeof(float) * batch_size;  // Matrix C
        ctx_size = ((ctx_size + 63) / 64) * 64; // Align to 64 bytes
        
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ false,
        };
        
        std::vector<double> times;
        times.reserve(config.iterations);
        
        for (int iter = 0; iter < config.iterations; iter++) {
            struct ggml_context* ctx = ggml_init(params);
            if (!ctx) {
                std::cerr << "❌ Failed to initialize GGML context" << std::endl;
                return result;
            }
            
            // Create matrices
            std::vector<ggml_tensor*> tensors_a, tensors_b, tensors_c;
            
            for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
                struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
                struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, k);  
                struct ggml_tensor* c = ggml_mul_mat(ctx, a, b);
                
                tensors_a.push_back(a);
                tensors_b.push_back(b);
                tensors_c.push_back(c);
            }
            
            // Initialize matrices with random data
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<float> dist(0.0f, 1.0f);
            
            for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
                float* data_a = (float*)ggml_get_data(tensors_a[batch_idx]);
                float* data_b = (float*)ggml_get_data(tensors_b[batch_idx]);
                
                for (int i = 0; i < ggml_nelements(tensors_a[batch_idx]); i++) {
                    data_a[i] = dist(gen);
                }
                for (int i = 0; i < ggml_nelements(tensors_b[batch_idx]); i++) {
                    data_b[i] = dist(gen);
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
                ggml_free(ctx);
                return result;
            }
            
            // Wait for completion
            ggml_numa_coordinator_manager_wait_for_completion(mgr);
            
            double elapsed_ms = timer.elapsed_ms();
            times.push_back(elapsed_ms);
            
            // Cleanup
            ggml_numa_coordinator_manager_free(mgr);
            
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
        std::cout << "=============================================" << std::endl;
        
        // Determine which NUMA configurations to test
        std::vector<int> numa_configs_to_test;
        numa_configs_to_test.push_back(1); // Always test single NUMA node
        
        if (topology.total_numa_nodes >= 2) {
            numa_configs_to_test.push_back(2);
        }
        if (topology.total_numa_nodes >= 4) {
            numa_configs_to_test.push_back(4);
        }
        
        std::cout << "Testing NUMA configurations: ";
        for (size_t i = 0; i < numa_configs_to_test.size(); i++) {
            std::cout << numa_configs_to_test[i];
            if (i < numa_configs_to_test.size() - 1) std::cout << ", ";
        }
        std::cout << " nodes" << std::endl << std::endl;
        
        // Run tests for each matrix configuration
        for (const auto& matrix_test : config.matrix_tests) {
            std::cout << "📊 Testing: " << matrix_test.description << std::endl;
            std::cout << "   Matrix dimensions: " << matrix_test.m << "x" << matrix_test.n 
                      << " = " << matrix_test.m << "x" << matrix_test.k 
                      << " * " << matrix_test.k << "x" << matrix_test.n << std::endl;
            
            if (!matrix_test.test_batching) {
                // Single matrix test
                std::map<int, BenchmarkResult> results;
                
                for (int numa_nodes : numa_configs_to_test) {
                    results[numa_nodes] = run_matrix_multiply(matrix_test.m, matrix_test.n, matrix_test.k, numa_nodes);
                }
                
                // Display results
                display_results(results, matrix_test.description);
            } else {
                // Batch size scaling test
                std::cout << "   Testing batch size scaling..." << std::endl;
                
                for (int batch_size : config.batch_sizes) {
                    std::cout << "\n   📦 Batch size: " << batch_size << std::endl;
                    std::map<int, BenchmarkResult> batch_results;
                    
                    for (int numa_nodes : numa_configs_to_test) {
                        batch_results[numa_nodes] = run_matrix_multiply(matrix_test.m, matrix_test.n, matrix_test.k, numa_nodes, batch_size);
                    }
                    
                    display_results(batch_results, "Batch size " + std::to_string(batch_size));
                }
            }
            
            std::cout << std::endl;
        }
    }
    
private:
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
        } else if (strcmp(argv[i], "--force-virtual") == 0) {
            force_virtual_numa = true;
            std::cout << "⚠️  Force virtual NUMA mode enabled" << std::endl;
        } else if (strcmp(argv[i], "--help") == 0) {
            std::cout << "Real NUMA Hardware Performance Test\n"
                      << "Usage: " << argv[0] << " [OPTIONS]\n"
                      << "Options:\n"
                      << "  --quick           Run quick test with smaller matrices\n"
                      << "  --debug           Enable debug output\n"
                      << "  --force-virtual   Force test with virtual NUMA (for demo purposes)\n"
                      << "  --help            Show this help message\n";
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
        std::cout << "• NUMA library available: " << (numa_available() >= 0 ? "YES" : "NO") << std::endl;
        std::cout << "• Hardware NUMA nodes: " << numa_num_configured_nodes() << std::endl;
        std::cout << "• Total CPUs: " << numa_num_configured_cpus() << std::endl;
        
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
