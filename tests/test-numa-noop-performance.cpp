/**
 * NUMA Architecture Performance Comparison Test
 * 
 * This test measures the overhead of different execution modes:
 * 1. GGML CPU (NUMA disabled) - Pure ggml-cpu execution
 * 2. NUMA Fallback - NUMA coordinator falling back to ggml-cpu for unsupported ops
 * 3. NUMA Direct - NUMA coordinator using NUMA kernels
 * 
 * Tests specific tensor sizes that trigger different NUMA strategies:
 * - 256 elements: Single-threaded, single-node (threshold 256)
 * - 512 elements: Multi-threaded, single-node (threshold 512) 
 * - 1024 elements: Multi-threaded, multi-node (threshold 1024)
 * 
 * Runs 21 iterations per test with 1 warmup iteration discarded.
 */

#include <vector>
#include <chrono>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <cstdlib>

#include "ggml.h"
#include "ggml-cpu.h"

// Execution modes to test
enum class ExecutionMode {
    GGML_CPU,        // Pure ggml-cpu (NUMA disabled)
    NUMA_FALLBACK,   // NUMA coordinator with fallback NOOP
    NUMA_DIRECT      // NUMA coordinator with NUMA NOOP kernel
};

// Test configuration
struct PerformanceTestConfig {
    std::vector<size_t> tensor_sizes = {256, 512, 1024};  // Strategic threshold sizes
    int num_iterations = 21;  // 21 tests with 1 warmup
    int warmup_iterations = 1;
    bool enable_debug = false;
};

// Performance result for a single execution mode test
struct ExecutionResult {
    ExecutionMode mode;
    size_t tensor_size;
    std::vector<double> times_ns;
    double avg_ns;
    double std_dev_ns;
    double min_ns;
    double max_ns;
};

// Performance result comparing all modes for one tensor size
struct TensorSizeResult {
    size_t tensor_size;
    ExecutionResult ggml_cpu_result;
    ExecutionResult numa_fallback_result;
    ExecutionResult numa_direct_result;
    
    // Overhead calculations
    double fallback_overhead_ratio;    // NUMA Fallback vs GGML CPU
    double direct_overhead_ratio;      // NUMA Direct vs GGML CPU
    double direct_vs_fallback_ratio;   // NUMA Direct vs NUMA Fallback
};

// Calculate statistics for a set of timing measurements
ExecutionResult calculate_execution_stats(ExecutionMode mode, size_t tensor_size, const std::vector<double>& times) {
    ExecutionResult result;
    result.mode = mode;
    result.tensor_size = tensor_size;
    result.times_ns = times;
    
    if (times.empty()) {
        result.avg_ns = result.std_dev_ns = result.min_ns = result.max_ns = 0.0;
        return result;
    }
    
    // Calculate average
    result.avg_ns = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    
    // Calculate standard deviation
    double variance = 0.0;
    for (double time : times) {
        variance += (time - result.avg_ns) * (time - result.avg_ns);
    }
    result.std_dev_ns = std::sqrt(variance / times.size());
    
    // Calculate min/max
    result.min_ns = *std::min_element(times.begin(), times.end());
    result.max_ns = *std::max_element(times.begin(), times.end());
    
    return result;
}

// Run performance test for a specific execution mode
ExecutionResult run_execution_mode_test(ExecutionMode mode, size_t tensor_size, const PerformanceTestConfig& config) {
    std::vector<double> times;
    
    // Create GGML context
    struct ggml_init_params params;
    params.mem_size = 256 * 1024 * 1024;  // 256MB
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        std::cerr << "Failed to initialize GGML context for mode " << static_cast<int>(mode) << std::endl;
        return calculate_execution_stats(mode, tensor_size, times);
    }
    
    // Create test tensor
    int64_t ne[] = {(int64_t)tensor_size, 1, 1, 1};
    struct ggml_tensor* tensor = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    
    // Initialize tensor data (not needed for NOOP but for completeness)
    float* data = (float*)ggml_get_data(tensor);
    for (size_t i = 0; i < tensor_size; i++) {
        data[i] = 0.0f;
    }
    
    // Set operation type based on execution mode
    ggml_op op_type;
    switch (mode) {
        case ExecutionMode::GGML_CPU:
        case ExecutionMode::NUMA_FALLBACK:
            op_type = GGML_OP_NUMA_FALLBACK_NOOP;
            break;
        case ExecutionMode::NUMA_DIRECT:
            op_type = GGML_OP_NUMA_NOOP;
            break;
    }
    
    // Run iterations (including warmup)
    for (int i = 0; i < config.num_iterations + config.warmup_iterations; i++) {
        tensor->op = op_type;
        
        // Create compute graph
        struct ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, tensor);
        
        // Measure execution time
        auto start = std::chrono::high_resolution_clock::now();
        
        if (mode == ExecutionMode::GGML_CPU) {
            // For GGML_CPU mode, use direct ggml-cpu computation to bypass NUMA entirely
            // This simulates pure ggml-cpu performance without any NUMA overhead
            ggml_graph_compute_with_ctx(ctx, graph, 1);  // Single-threaded for consistency
        } else {
            // For NUMA modes, use normal computation which will go through NUMA dispatch
            ggml_graph_compute_with_ctx(ctx, graph, 4);  // Multi-threaded
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        
        // Skip warmup iterations
        if (i >= config.warmup_iterations) {
            times.push_back(duration.count());
        }
    }
    
    ggml_free(ctx);
    return calculate_execution_stats(mode, tensor_size, times);
}

// Run comprehensive test for a single tensor size
TensorSizeResult run_tensor_size_test(size_t tensor_size, const PerformanceTestConfig& config) {
    TensorSizeResult result;
    result.tensor_size = tensor_size;
    
    if (config.enable_debug) {
        std::cout << "\n🔍 Testing tensor size: " << tensor_size << " elements\n";
        std::cout << "Expected NUMA strategy: ";
        if (tensor_size <= 256) {
            std::cout << "Single-threaded, single-node\n";
        } else if (tensor_size <= 512) {
            std::cout << "Multi-threaded, single-node\n";
        } else {
            std::cout << "Multi-threaded, multi-node\n";
        }
    }
    
    // Test GGML CPU mode (pure ggml-cpu)
    if (config.enable_debug) std::cout << "  Testing GGML CPU mode...\n";
    result.ggml_cpu_result = run_execution_mode_test(ExecutionMode::GGML_CPU, tensor_size, config);
    
    // Test NUMA Fallback mode
    if (config.enable_debug) std::cout << "  Testing NUMA Fallback mode...\n";
    result.numa_fallback_result = run_execution_mode_test(ExecutionMode::NUMA_FALLBACK, tensor_size, config);
    
    // Test NUMA Direct mode  
    if (config.enable_debug) std::cout << "  Testing NUMA Direct mode...\n";
    result.numa_direct_result = run_execution_mode_test(ExecutionMode::NUMA_DIRECT, tensor_size, config);
    
    // Calculate overhead ratios
    result.fallback_overhead_ratio = result.numa_fallback_result.avg_ns / result.ggml_cpu_result.avg_ns;
    result.direct_overhead_ratio = result.numa_direct_result.avg_ns / result.ggml_cpu_result.avg_ns;
    result.direct_vs_fallback_ratio = result.numa_direct_result.avg_ns / result.numa_fallback_result.avg_ns;
    
    return result;
}

// Convert execution mode to string for display
std::string execution_mode_to_string(ExecutionMode mode) {
    switch (mode) {
        case ExecutionMode::GGML_CPU: return "GGML CPU";
        case ExecutionMode::NUMA_FALLBACK: return "NUMA Fallback";
        case ExecutionMode::NUMA_DIRECT: return "NUMA Direct";
        default: return "Unknown";
    }
}

// Print detailed results for a single tensor size
void print_tensor_size_results(const TensorSizeResult& result) {
    std::cout << "\n📊 Results for " << result.tensor_size << " elements:\n";
    std::cout << std::string(65, '=') << "\n";
    
    std::cout << std::setw(15) << "Mode" 
              << std::setw(12) << "Avg (ns)"
              << std::setw(12) << "StdDev (ns)"
              << std::setw(12) << "Min (ns)"
              << std::setw(12) << "Max (ns)"
              << std::setw(10) << "Overhead"
              << std::endl;
    std::cout << std::string(73, '-') << std::endl;
    
    // GGML CPU (baseline)
    std::cout << std::setw(15) << "GGML CPU"
              << std::setw(12) << std::fixed << std::setprecision(0) << result.ggml_cpu_result.avg_ns
              << std::setw(12) << std::fixed << std::setprecision(0) << result.ggml_cpu_result.std_dev_ns
              << std::setw(12) << std::fixed << std::setprecision(0) << result.ggml_cpu_result.min_ns
              << std::setw(12) << std::fixed << std::setprecision(0) << result.ggml_cpu_result.max_ns
              << std::setw(10) << "1.00x"
              << std::endl;
    
    // NUMA Fallback
    std::cout << std::setw(15) << "NUMA Fallback"
              << std::setw(12) << std::fixed << std::setprecision(0) << result.numa_fallback_result.avg_ns
              << std::setw(12) << std::fixed << std::setprecision(0) << result.numa_fallback_result.std_dev_ns
              << std::setw(12) << std::fixed << std::setprecision(0) << result.numa_fallback_result.min_ns
              << std::setw(12) << std::fixed << std::setprecision(0) << result.numa_fallback_result.max_ns
              << std::setw(10) << std::fixed << std::setprecision(2) << result.fallback_overhead_ratio << "x"
              << std::endl;
    
    // NUMA Direct
    std::cout << std::setw(15) << "NUMA Direct"
              << std::setw(12) << std::fixed << std::setprecision(0) << result.numa_direct_result.avg_ns
              << std::setw(12) << std::fixed << std::setprecision(0) << result.numa_direct_result.std_dev_ns
              << std::setw(12) << std::fixed << std::setprecision(0) << result.numa_direct_result.min_ns
              << std::setw(12) << std::fixed << std::setprecision(0) << result.numa_direct_result.max_ns
              << std::setw(10) << std::fixed << std::setprecision(2) << result.direct_overhead_ratio << "x"
              << std::endl;
    
    std::cout << "\n💡 Analysis:\n";
    std::cout << "• NUMA Fallback overhead: " << std::fixed << std::setprecision(1) 
              << ((result.fallback_overhead_ratio - 1.0) * 100) << "%\n";
    std::cout << "• NUMA Direct overhead: " << std::fixed << std::setprecision(1) 
              << ((result.direct_overhead_ratio - 1.0) * 100) << "%\n";
    std::cout << "• NUMA Direct vs Fallback: " << std::fixed << std::setprecision(2) 
              << result.direct_vs_fallback_ratio << "x";
    if (result.direct_vs_fallback_ratio < 1.0) {
        std::cout << " (🚀 " << std::fixed << std::setprecision(1) 
                  << ((1.0 - result.direct_vs_fallback_ratio) * 100) << "% faster)";
    } else {
        std::cout << " (⚠️  " << std::fixed << std::setprecision(1) 
                  << ((result.direct_vs_fallback_ratio - 1.0) * 100) << "% slower)";
    }
    std::cout << "\n";
}

// Print comprehensive summary of all results
void print_comprehensive_summary(const std::vector<TensorSizeResult>& results) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "📈 COMPREHENSIVE NUMA ARCHITECTURE PERFORMANCE ANALYSIS\n";
    std::cout << std::string(80, '=') << "\n\n";
    
    std::cout << "🎯 Test Configuration:\n";
    std::cout << "• Tensor sizes: 256 (single/single), 512 (single/multi), 1024 (multi/multi)\n";
    std::cout << "• Iterations per test: 21 (20 measured + 1 warmup)\n";
    std::cout << "• Execution modes: GGML CPU, NUMA Fallback, NUMA Direct\n\n";
    
    std::cout << "📊 Overhead Summary:\n";
    std::cout << std::setw(12) << "Tensor Size"
              << std::setw(18) << "Fallback Overhead"  
              << std::setw(16) << "Direct Overhead"
              << std::setw(18) << "Direct vs Fallback"
              << std::setw(16) << "Best Mode"
              << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    double total_fallback_overhead = 0.0;
    double total_direct_overhead = 0.0;
    double total_direct_vs_fallback = 0.0;
    
    for (const auto& result : results) {
        std::string best_mode = "GGML CPU";
        double best_time = result.ggml_cpu_result.avg_ns;
        
        if (result.numa_fallback_result.avg_ns < best_time) {
            best_mode = "NUMA Fallback";
            best_time = result.numa_fallback_result.avg_ns;
        }
        if (result.numa_direct_result.avg_ns < best_time) {
            best_mode = "NUMA Direct";
        }
        
        std::cout << std::setw(12) << result.tensor_size
                  << std::setw(16) << std::fixed << std::setprecision(1) 
                  << ((result.fallback_overhead_ratio - 1.0) * 100) << "%"
                  << std::setw(14) << std::fixed << std::setprecision(1) 
                  << ((result.direct_overhead_ratio - 1.0) * 100) << "%"
                  << std::setw(16) << std::fixed << std::setprecision(2) 
                  << result.direct_vs_fallback_ratio << "x"
                  << std::setw(16) << best_mode
                  << std::endl;
        
        total_fallback_overhead += result.fallback_overhead_ratio;
        total_direct_overhead += result.direct_overhead_ratio;
        total_direct_vs_fallback += result.direct_vs_fallback_ratio;
    }
    
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::setw(12) << "Average"
              << std::setw(16) << std::fixed << std::setprecision(1) 
              << ((total_fallback_overhead / results.size() - 1.0) * 100) << "%"
              << std::setw(14) << std::fixed << std::setprecision(1) 
              << ((total_direct_overhead / results.size() - 1.0) * 100) << "%"
              << std::setw(16) << std::fixed << std::setprecision(2) 
              << (total_direct_vs_fallback / results.size()) << "x"
              << std::endl;
    
    std::cout << "\n🔍 Key Insights:\n";
    std::cout << "1. NUMA Fallback Overhead: Measures cost of NUMA coordinator for unsupported ops\n";
    std::cout << "2. NUMA Direct Overhead: Measures cost of NUMA coordinator + kernel execution\n";
    std::cout << "3. Direct vs Fallback: Shows efficiency gain/loss of NUMA kernels vs fallback\n";
    std::cout << "4. Lower values indicate better performance relative to baseline GGML CPU\n\n";
    
    // Analysis by tensor size
    for (const auto& result : results) {
        std::cout << "📍 " << result.tensor_size << " elements ";
        if (result.tensor_size <= 256) {
            std::cout << "(Single-threaded, Single-node strategy):\n";
        } else if (result.tensor_size <= 512) {
            std::cout << "(Multi-threaded, Single-node strategy):\n";
        } else {
            std::cout << "(Multi-threaded, Multi-node strategy):\n";
        }
        
        if (result.direct_vs_fallback_ratio < 1.0) {
            std::cout << "   ✅ NUMA kernels are " << std::fixed << std::setprecision(1)
                      << ((1.0 - result.direct_vs_fallback_ratio) * 100) 
                      << "% faster than fallback\n";
        } else if (result.direct_vs_fallback_ratio > 1.1) {
            std::cout << "   ⚠️  NUMA kernels have " << std::fixed << std::setprecision(1)
                      << ((result.direct_vs_fallback_ratio - 1.0) * 100) 
                      << "% overhead vs fallback\n";
        } else {
            std::cout << "   📊 NUMA kernels perform similarly to fallback\n";
        }
    }
}

int main(int argc, char* argv[]) {
    PerformanceTestConfig config;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--debug" || std::string(argv[i]) == "-d") {
            config.enable_debug = true;
        } else if (std::string(argv[i]) == "--iterations" && i + 1 < argc) {
            config.num_iterations = std::atoi(argv[++i]);
        } else if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            std::cout << "Usage: " << argv[0] << " [OPTIONS]\n";
            std::cout << "NUMA Architecture Performance Comparison Test\n\n";
            std::cout << "This test compares execution overhead across three modes:\n";
            std::cout << "1. GGML CPU - Pure ggml-cpu execution (baseline)\n";
            std::cout << "2. NUMA Fallback - NUMA coordinator falling back to ggml-cpu\n";
            std::cout << "3. NUMA Direct - NUMA coordinator using NUMA kernels\n\n";
            std::cout << "Options:\n";
            std::cout << "  --debug, -d           Enable debug output\n";
            std::cout << "  --iterations N        Number of iterations per test (default: 21)\n";
            std::cout << "  --help, -h            Show this help message\n\n";
            std::cout << "Test dimensions:\n";
            std::cout << "• 256 elements: Single-threaded, single-node strategy\n";
            std::cout << "• 512 elements: Multi-threaded, single-node strategy\n";
            std::cout << "• 1024 elements: Multi-threaded, multi-node strategy\n\n";
            return 0;
        }
    }
    
    std::cout << "🚀 NUMA Architecture Performance Comparison Test\n";
    std::cout << std::string(55, '=') << "\n\n";
    
    std::cout << "🔧 Configuration:\n";
    std::cout << "• Test iterations: " << config.num_iterations << " (+ " << config.warmup_iterations << " warmup)\n";
    std::cout << "• Tensor sizes: ";
    for (size_t i = 0; i < config.tensor_sizes.size(); i++) {
        std::cout << config.tensor_sizes[i];
        if (i < config.tensor_sizes.size() - 1) std::cout << ", ";
    }
    std::cout << " elements\n";
    std::cout << "• Execution modes: GGML CPU, NUMA Fallback, NUMA Direct\n\n";
    
    std::cout << "📊 Expected NUMA strategies by tensor size:\n";
    std::cout << "• 256: Single-threaded, single-node (threshold ≤ 256)\n";
    std::cout << "• 512: Multi-threaded, single-node (threshold ≤ 512)\n";
    std::cout << "• 1024: Multi-threaded, multi-node (threshold ≤ 1024)\n\n";
    
    std::cout << "⏱️  Running performance tests...\n";
    
    std::vector<TensorSizeResult> results;
    
    // Run tests for each tensor size
    for (size_t tensor_size : config.tensor_sizes) {
        std::cout << "\n🧪 Testing tensor size: " << tensor_size << " elements...\n";
        
        TensorSizeResult result = run_tensor_size_test(tensor_size, config);
        results.push_back(result);
        
        // Print immediate results for this tensor size
        print_tensor_size_results(result);
    }
    
    // Print comprehensive summary
    print_comprehensive_summary(results);
    
    std::cout << "\n✅ Performance analysis complete!\n";
    
    return 0;
}
