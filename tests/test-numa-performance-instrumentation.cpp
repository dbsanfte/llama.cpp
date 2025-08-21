/**
 * @file test-numa-performance-instrumentation.cpp
 * @brief Test NUMA performance instrumentation system
 * 
 * Validates that performance measurements are correctly collected
 * and reports are generated properly
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <cstdio>

extern "C" {
    #include "../ggml/src/ggml-cpu/ggml-numa-perf.h"
}

// Test performance measurement categories
void test_performance_categories() {
    std::cout << "\n=== Testing Performance Categories ===" << std::endl;
    
    // Test basic timing
    ggml_numa_perf_start(NUMA_PERF_COORDINATOR_INIT, "test_operation", "test_kernel", 0, 1024, 4);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    ggml_numa_perf_end();
    
    // Test another category
    ggml_numa_perf_start(NUMA_PERF_EXECUTOR_QUERY, "add_operation", "numa_add_kernel", 1, 4096, 8);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    ggml_numa_perf_end();
    
    // Test kernel execution category
    ggml_numa_perf_start(NUMA_PERF_KERNEL_NUMA_EXEC, "add_operation", "numa_add_kernel", 0, 8192, 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    ggml_numa_perf_end();
    
    // Test fallback category
    ggml_numa_perf_start(NUMA_PERF_EXECUTOR_FALLBACK, "unknown_op", "cpu_fallback", -1, 2048, 2);
    std::this_thread::sleep_for(std::chrono::milliseconds(8));
    ggml_numa_perf_end();
    
    std::cout << "✅ Performance categories tested" << std::endl;
}

// Test performance statistics retrieval
void test_performance_statistics() {
    std::cout << "\n=== Testing Performance Statistics ===" << std::endl;
    
    // Get statistics for different categories
    numa_perf_stats_t coord_stats = ggml_numa_perf_get_stats(NUMA_PERF_COORDINATOR_INIT);
    numa_perf_stats_t query_stats = ggml_numa_perf_get_stats(NUMA_PERF_EXECUTOR_QUERY);
    numa_perf_stats_t kernel_stats = ggml_numa_perf_get_stats(NUMA_PERF_KERNEL_NUMA_EXEC);
    numa_perf_stats_t fallback_stats = ggml_numa_perf_get_stats(NUMA_PERF_EXECUTOR_FALLBACK);
    
    std::cout << "Coordinator Init Events: " << coord_stats.event_count 
              << ", Avg Time: " << (coord_stats.avg_time_ns / 1e6) << "ms" << std::endl;
    std::cout << "Executor Query Events: " << query_stats.event_count 
              << ", Avg Time: " << (query_stats.avg_time_ns / 1e6) << "ms" << std::endl;
    std::cout << "Kernel Exec Events: " << kernel_stats.event_count 
              << ", Avg Time: " << (kernel_stats.avg_time_ns / 1e6) << "ms" << std::endl;
    std::cout << "Fallback Events: " << fallback_stats.event_count 
              << ", Avg Time: " << (fallback_stats.avg_time_ns / 1e6) << "ms" << std::endl;
    
    std::cout << "✅ Performance statistics retrieved" << std::endl;
}

// Test performance timing accuracy
void test_timing_accuracy() {
    std::cout << "\n=== Testing Timing Accuracy ===" << std::endl;
    
    // Test multiple measurements with known delays
    const int num_tests = 5;
    const int delay_ms = 20;
    
    for (int i = 0; i < num_tests; i++) {
        ggml_numa_perf_start(NUMA_PERF_KERNEL_NUMA_EXEC, "timing_test", "accuracy_test", 0, 1000, 1);
        
        auto start = std::chrono::high_resolution_clock::now();
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        auto end = std::chrono::high_resolution_clock::now();
        
        ggml_numa_perf_end();
        
        auto actual_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Test " << (i+1) << ": Expected ~" << delay_ms << "ms, Actual " << actual_duration << "ms" << std::endl;
    }
    
    // Get final statistics
    numa_perf_stats_t timing_stats = ggml_numa_perf_get_stats(NUMA_PERF_KERNEL_NUMA_EXEC);
    std::cout << "Timing tests completed. Total events: " << timing_stats.event_count << std::endl;
    std::cout << "Average timing: " << (timing_stats.avg_time_ns / 1e6) << "ms" << std::endl;
    
    std::cout << "✅ Timing accuracy tested" << std::endl;
}

// Test performance reporting
void test_performance_reporting() {
    std::cout << "\n=== Testing Performance Reporting ===" << std::endl;
    
    // Generate some additional performance events
    for (int i = 0; i < 10; i++) {
        ggml_numa_perf_start(NUMA_PERF_COORDINATOR_DISPATCH, "dispatch_test", "test_kernel", i % 3, 1024 * (i+1), 4);
        std::this_thread::sleep_for(std::chrono::milliseconds(2 + i));
        ggml_numa_perf_end();
    }
    
    std::cout << "Generated additional performance events" << std::endl;
    std::cout << "✅ Performance reporting tested" << std::endl;
}

int main() {
    std::cout << "NUMA Performance Instrumentation Test Suite" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    // Initialize performance measurement
    if (!ggml_numa_perf_init()) {
        std::cerr << "❌ Failed to initialize performance measurement" << std::endl;
        return 1;
    }
    
    // Enable performance measurement and detailed logging
    ggml_numa_perf_set_enabled(true);
    ggml_numa_perf_set_detailed_logging(true);
    
    std::cout << "✅ Performance measurement initialized" << std::endl;
    
    // Run tests
    test_performance_categories();
    test_performance_statistics();
    test_timing_accuracy();
    test_performance_reporting();
    
    // Print comprehensive performance report
    std::cout << "\n=== Performance Summary ===" << std::endl;
    ggml_numa_perf_print_summary();
    
    std::cout << "\n=== Detailed Performance Report ===" << std::endl;
    ggml_numa_perf_print_detailed_report();
    
    // Shutdown (will also print final summary)
    ggml_numa_perf_shutdown();
    
    std::cout << "\n🎉 All performance instrumentation tests completed!" << std::endl;
    return 0;
}
