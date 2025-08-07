#include <chrono>
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <iomanip>
#include <unistd.h>
#include <mutex>

#include "ggml.h"
#include "ggml-cpu.h" 
#include "ggml-numa-coordinator.h"
#include "common.h"

class NumaCoordinatorTester {
private:
    int max_numa_nodes;
    int max_cpus;
    
    // Progress callback tracking
    std::atomic<int> callback_count{0};
    std::atomic<int> callback_errors{0};
    std::vector<int> work_ids_completed;
    std::mutex callback_mutex;

    // Progress callback function
    static void progress_callback_func(int work_id, int numa_node, struct ggml_tensor * tensor, void * user_data) {
        NumaCoordinatorTester * tester = (NumaCoordinatorTester*)user_data;
        tester->handleProgressCallback(work_id, numa_node, tensor);
    }
    
    void handleProgressCallback(int work_id, int numa_node, struct ggml_tensor * tensor) {
        callback_count++;
        
        // Validate callback parameters
        if (work_id < 0) {
            std::cout << "⚠️  Invalid work_id in callback: " << work_id << std::endl;
            callback_errors++;
        }
        
        if (numa_node < 0 || numa_node >= max_numa_nodes) {
            std::cout << "⚠️  Invalid numa_node in callback: " << numa_node << std::endl;
            callback_errors++;
        }
        
        if (!tensor) {
            std::cout << "⚠️  NULL tensor in callback" << std::endl;
            callback_errors++;
        }
        
        // Thread-safe logging
        {
            std::lock_guard<std::mutex> lock(callback_mutex);
            work_ids_completed.push_back(work_id);
            std::cout << "[PROGRESS] Work " << work_id << " completed on NUMA node " << numa_node 
                      << " (tensor: " << tensor << ")" << std::endl;
        }
    }

public:
    NumaCoordinatorTester() {
        max_numa_nodes = ggml_numa_node_count();
        max_cpus = sysconf(_SC_NPROCESSORS_ONLN);
        
        std::cout << "=== NUMA Coordinator Validation ===" << std::endl;
        std::cout << "Available CPUs: " << max_cpus << std::endl;
        std::cout << "NUMA nodes: " << max_numa_nodes << std::endl;
        std::cout << "====================================" << std::endl;
    }

    bool testCoordinatorBasics() {
        std::cout << "\n--- Testing Coordinator Creation ---" << std::endl;
        
        // Reset callback tracking
        callback_count = 0;
        callback_errors = 0;
        work_ids_completed.clear();
        
        // Test coordinator creation
        struct ggml_numa_coordinator_manager * coordinator = ggml_numa_coordinator_manager_get_global(max_cpus, false);
        if (!coordinator) {
            std::cout << "❌ Failed to create NUMA coordinator" << std::endl;
            return false;
        }
        std::cout << "✅ NUMA coordinator created successfully" << std::endl;
        
        // Set up progress callback
        if (ggml_numa_coordinator_manager_set_progress_callback(coordinator, progress_callback_func, this) != 0) {
            std::cout << "❌ Failed to set progress callback" << std::endl;
            return false;
        }
        std::cout << "✅ Progress callback enabled" << std::endl;
        
        // Create a simple GGML context and graph
        struct ggml_init_params params = {
            /*.mem_size   =*/ 64 * 1024 * 1024, // 64MB
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ false,
        };
        
        struct ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            std::cout << "❌ Failed to create GGML context" << std::endl;
            return false;
        }
        
        // Create simple tensor operations
        struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
        struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
        struct ggml_tensor * result = ggml_add(ctx, a, b);
        
        // Initialize tensors
        float * data_a = (float*)ggml_get_data(a);
        float * data_b = (float*)ggml_get_data(b);
        
        for (int i = 0; i < 1024; i++) {
            data_a[i] = 1.0f + i * 0.001f;
            data_b[i] = 2.0f + i * 0.002f;
        }
        
        // Create computation graph
        struct ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, result);
        
        std::cout << "✅ GGML context and graph created" << std::endl;
        
        // Test coordinator graph setup
        if (ggml_numa_coordinator_manager_set_cgraph(coordinator, graph) != 0) {
            std::cout << "❌ Failed to set cgraph in coordinator" << std::endl;
            ggml_free(ctx);
            return false;
        }
        std::cout << "✅ Coordinator cgraph set successfully" << std::endl;
        
        // Test coordinator startup
        if (ggml_numa_coordinator_manager_start(coordinator) != 0) {
            std::cout << "❌ Failed to start coordinator" << std::endl;
            ggml_free(ctx);
            return false;
        }
        std::cout << "✅ Coordinator started successfully" << std::endl;
        
        // Test work submission
        int work_id = ggml_numa_coordinator_manager_submit_work(coordinator, result, -1);
        if (work_id < 0) {
            std::cout << "❌ Failed to submit work to coordinator" << std::endl;
            ggml_free(ctx);
            return false;
        }
        std::cout << "✅ Work submitted successfully (work_id=" << work_id << ")" << std::endl;
        
        // Wait for completion
        if (ggml_numa_coordinator_manager_wait_for_completion(coordinator) != 0) {
            std::cout << "❌ Failed to wait for work completion" << std::endl;
            ggml_free(ctx);
            return false;
        }
        std::cout << "✅ Work completed successfully" << std::endl;
        
        // Get performance statistics
        struct ggml_numa_perf_stats stats = ggml_numa_coordinator_manager_get_stats(coordinator, -1);
        std::cout << "✅ Coordinator stats:" << std::endl;
        std::cout << "   - Total work items: " << stats.total_work_items << std::endl;
        std::cout << "   - Throughput: " << std::fixed << std::setprecision(2) 
                  << stats.throughput_items_per_sec << " items/sec" << std::endl;
        std::cout << "   - Avg processing time: " << stats.average_processing_time_us << " µs" << std::endl;
        
        // Validate result
        float * result_data = (float*)ggml_get_data(result);
        bool computation_correct = true;
        for (int i = 0; i < 10; i++) { // Check first 10 elements
            float expected = (1.0f + i * 0.001f) + (2.0f + i * 0.002f);
            if (std::abs(result_data[i] - expected) > 1e-6) {
                computation_correct = false;
                break;
            }
        }
        
        if (computation_correct) {
            std::cout << "✅ Computation results are correct" << std::endl;
        } else {
            std::cout << "❌ Computation results are incorrect" << std::endl;
            ggml_free(ctx);
            return false;
        }
        
        // Validate progress callbacks
        std::cout << "\n--- Progress Callback Validation ---" << std::endl;
        std::cout << "Callbacks received: " << callback_count.load() << std::endl;
        std::cout << "Callback errors: " << callback_errors.load() << std::endl;
        std::cout << "Expected work items: " << stats.total_work_items << std::endl;
        
        if (callback_count.load() != stats.total_work_items) {
            std::cout << "❌ Callback count mismatch: expected " << stats.total_work_items 
                      << ", got " << callback_count.load() << std::endl;
            ggml_free(ctx);
            return false;
        }
        
        if (callback_errors.load() > 0) {
            std::cout << "❌ Callback validation errors detected: " << callback_errors.load() << std::endl;
            ggml_free(ctx);
            return false;
        }
        
        // Verify work_id was correct
        bool found_work_id = false;
        for (int completed_id : work_ids_completed) {
            if (completed_id == work_id) {
                found_work_id = true;
                break;
            }
        }
        
        if (!found_work_id) {
            std::cout << "❌ Expected work_id " << work_id << " not found in callbacks" << std::endl;
            ggml_free(ctx);
            return false;
        }
        
        std::cout << "✅ Progress callbacks validated successfully" << std::endl;
        
        ggml_free(ctx);
        return true;
    }

    bool testCoordinatorPerformance() {
        std::cout << "\n--- Testing Coordinator Performance ---" << std::endl;
        
        struct ggml_numa_coordinator_manager * coordinator = ggml_numa_coordinator_manager_get_global(max_cpus, false);
        if (!coordinator) {
            std::cout << "❌ Failed to get NUMA coordinator" << std::endl;
            return false;
        }
        
        // Set up progress callback for performance testing
        if (ggml_numa_coordinator_manager_set_progress_callback(coordinator, progress_callback_func, this) != 0) {
            std::cout << "❌ Failed to set progress callback" << std::endl;
            return false;
        }
        
        // Test with different workload sizes
        std::vector<int> test_sizes = {100, 500, 1000, 2000};
        
        std::cout << "Operations | Time (ms) | Throughput (ops/s) | Coord Items | Coord Throughput | Callbacks" << std::endl;
        std::cout << "-----------|-----------|--------------------|--------------|-----------------|-----------" << std::endl;
        
        for (int num_ops : test_sizes) {
            // Reset callback tracking for this test
            callback_count = 0;
            callback_errors = 0;
            work_ids_completed.clear();
            
            auto start_time = std::chrono::high_resolution_clock::now();
            
            for (int i = 0; i < num_ops; i++) {
                // Create context for this operation
                struct ggml_init_params params = {
                    /*.mem_size   =*/ 16 * 1024 * 1024, // 16MB
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ false,
                };
                
                struct ggml_context * ctx = ggml_init(params);
                if (!ctx) continue;
                
                // Simple tensor operation
                struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512);
                struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512);
                struct ggml_tensor * result = ggml_add(ctx, a, b);
                
                // Initialize data
                float * data_a = (float*)ggml_get_data(a);
                float * data_b = (float*)ggml_get_data(b);
                for (int j = 0; j < 512; j++) {
                    data_a[j] = 1.0f + j * 0.001f;
                    data_b[j] = 2.0f + j * 0.001f;
                }
                
                // Create and process graph
                struct ggml_cgraph * graph = ggml_new_graph(ctx);
                ggml_build_forward_expand(graph, result);
                
                if (ggml_numa_coordinator_manager_set_cgraph(coordinator, graph) == 0) {
                    if (ggml_numa_coordinator_manager_start(coordinator) == 0) {
                        int work_id = ggml_numa_coordinator_manager_submit_work(coordinator, result, -1);
                        if (work_id >= 0) {
                            ggml_numa_coordinator_manager_wait_for_completion(coordinator);
                        }
                    }
                }
                
                ggml_free(ctx);
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<double, std::milli>(end_time - start_time);
            
            double throughput = num_ops / (duration.count() / 1000.0);
            
            // Get coordinator stats
            struct ggml_numa_perf_stats stats = ggml_numa_coordinator_manager_get_stats(coordinator, -1);
            
            std::cout << std::setw(10) << num_ops
                      << " | " << std::setw(9) << std::fixed << std::setprecision(2) << duration.count()
                      << " | " << std::setw(18) << std::fixed << std::setprecision(1) << throughput
                      << " | " << std::setw(12) << stats.total_work_items
                      << " | " << std::setw(15) << std::fixed << std::setprecision(1) 
                      << stats.throughput_items_per_sec
                      << " | " << std::setw(9) << callback_count.load() << std::endl;
                      
            // Validate callbacks for this test size
            if (callback_count.load() != stats.total_work_items) {
                std::cout << "⚠️  Callback count mismatch for " << num_ops << " ops: expected " 
                          << stats.total_work_items << ", got " << callback_count.load() << std::endl;
            }
            
            if (callback_errors.load() > 0) {
                std::cout << "⚠️  " << callback_errors.load() << " callback errors for " 
                          << num_ops << " ops" << std::endl;
            }
        }
        
        return true;
    }
};

int main() {
    std::cout << "NUMA Coordinator Validation Suite" << std::endl;
    std::cout << "==================================" << std::endl;
    
    try {
        NumaCoordinatorTester tester;
        
        // Test basic coordinator functionality
        if (!tester.testCoordinatorBasics()) {
            std::cerr << "❌ Basic coordinator tests failed" << std::endl;
            return 1;
        }
        
        // Test coordinator performance
        if (!tester.testCoordinatorPerformance()) {
            std::cerr << "❌ Coordinator performance tests failed" << std::endl;
            return 1;
        }
        
        std::cout << "\n=== All NUMA Coordinator Tests Passed ===" << std::endl;
        std::cout << "✅ 3-tier NUMA coordinator is working correctly" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error during coordinator testing: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
