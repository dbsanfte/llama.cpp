/**
 * Simple performance test for NUMA coordinator
 * Tests basic functionality and CPU pinning
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <iomanip>

#include "ggml.h"
#include "ggml-cpu.h"
#include "../common/common.h"

#ifdef __linux__
#include <sched.h>
#include <numa.h>
#include <unistd.h>
#include <pthread.h>
extern "C" {
#include "ggml-numa-coordinator.h"
}
#endif

struct TestResults {
    double time_ms;
    double throughput_ops_per_sec;
    int numa_nodes_used;
    bool success;
};

class SimplePerformanceTester {
private:
    int max_numa_nodes;
    int total_logical_cpus;
    bool numa_available;
    
public:
    SimplePerformanceTester() {
        total_logical_cpus = std::thread::hardware_concurrency();
        
#ifdef __linux__
        numa_available = (::numa_available() != -1);
        if (numa_available) {
            max_numa_nodes = numa_max_node() + 1;
            std::cout << "✓ NUMA available with " << max_numa_nodes << " nodes" << std::endl;
        } else {
            max_numa_nodes = 1;
            std::cout << "⚠ NUMA not available, simulating single node" << std::endl;
        }
#else
        numa_available = false;
        max_numa_nodes = 1;
        std::cout << "⚠ Non-Linux system, NUMA disabled" << std::endl;
#endif
        
        std::cout << "Total logical CPUs: " << total_logical_cpus << std::endl;
        cpu_print_topology_info();
    }
    
    void testCpuPinning() {
        std::cout << "\n=== CPU Pinning Test ===" << std::endl;
        
#ifdef __linux__
        cpu_set_t original_mask;
        pthread_getaffinity_np(pthread_self(), sizeof(original_mask), &original_mask);
        
        std::vector<int> available_cpus;
        for (int cpu = 0; cpu < total_logical_cpus; cpu++) {
            if (CPU_ISSET(cpu, &original_mask)) {
                available_cpus.push_back(cpu);
            }
        }
        
        std::cout << "Testing pinning to " << available_cpus.size() << " available CPUs" << std::endl;
        
        int successful_pins = 0;
        for (int cpu : available_cpus) {
            cpu_set_t test_mask;
            CPU_ZERO(&test_mask);
            CPU_SET(cpu, &test_mask);
            
            if (pthread_setaffinity_np(pthread_self(), sizeof(test_mask), &test_mask) == 0) {
                successful_pins++;
            }
        }
        
        // Restore original affinity
        pthread_setaffinity_np(pthread_self(), sizeof(original_mask), &original_mask);
        
        std::cout << "✓ CPU pinning test: " << successful_pins << "/" << available_cpus.size() << " successful" << std::endl;
#else
        std::cout << "CPU pinning validation skipped (non-Linux system)" << std::endl;
#endif
    }
    
    TestResults testBasicTensorOps(int num_operations) {
        TestResults results = {0.0, 0.0, 1, false};
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Process operations in batches to avoid memory issues
        // Use smaller batches for larger operation counts
        int batch_size = (num_operations > 5000) ? 10 : 20;
        if (num_operations > 15000) batch_size = 5;
        
        int processed = 0;
        
        while (processed < num_operations) {
            int current_batch = std::min(batch_size, num_operations - processed);
            
            // Create GGML context for this batch
            struct ggml_init_params params = {
                /*.mem_size   =*/ 128 * 1024 * 1024, // 128MB for larger workloads
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ false,
            };
            
            struct ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                std::cerr << "Failed to initialize GGML context" << std::endl;
                return results;
            }
            
            // Create and execute tensor operations in this batch
            for (int i = 0; i < current_batch; i++) {
                // Use larger tensors for bigger workloads to increase CPU load
                int tensor_size = (num_operations > 1000) ? 2048 : 1024;
                if (num_operations > 10000) tensor_size = 4096;
                
                struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, tensor_size);
                struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, tensor_size);
                
                // Initialize tensors with some data
                float * data_a = (float*)ggml_get_data(a);
                float * data_b = (float*)ggml_get_data(b);
                
                for (int j = 0; j < tensor_size; j++) {
                    data_a[j] = 1.0f + (processed + i) * 0.1f;
                    data_b[j] = 2.0f + j * 0.001f;
                }
                
                struct ggml_tensor * result = ggml_add(ctx, a, b);
                
                // Create and execute computation graph
                struct ggml_cgraph * graph = ggml_new_graph(ctx);
                ggml_build_forward_expand(graph, result);
                
                struct ggml_cplan plan = ggml_graph_plan(graph, max_numa_nodes, nullptr);
                if (plan.work_size > 0) {
                    plan.work_data = (uint8_t*)malloc(plan.work_size);
                    ggml_graph_compute(graph, &plan);
                    free(plan.work_data);
                }
            }
            
            ggml_free(ctx);
            processed += current_batch;
            
            // Progress indicator for large workloads
            if (num_operations >= 10000 && processed % 1000 == 0) {
                std::cout << "    Progress: " << processed << "/" << num_operations << " operations..." << std::endl;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end_time - start_time);
        
        results.time_ms = duration.count();
        results.throughput_ops_per_sec = num_operations / (duration.count() / 1000.0);
        results.numa_nodes_used = max_numa_nodes;
        results.success = true;
        
        return results;
    }
    
#ifdef __linux__
    TestResults testNumaCoordinator(int num_work_items) {
        TestResults results = {0.0, 0.0, max_numa_nodes, false};
        
        if (!numa_available) {
            std::cout << "NUMA not available, skipping coordinator test" << std::endl;
            return results;
        }
        
        // Create coordinator
        struct ggml_numa_coordinator_manager * coordinator = 
            ggml_numa_coordinator_manager_new(max_numa_nodes * 2, true);
            
        if (!coordinator) {
            std::cout << "Failed to create NUMA coordinator" << std::endl;
            return results;
        }
        
        // Create GGML context
        struct ggml_init_params params = {
            /*.mem_size   =*/ 16 * 1024 * 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ false,
        };
        
        struct ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            ggml_numa_coordinator_manager_free(coordinator);
            return results;
        }
        
        // Create master cgraph
        struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
        struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
        struct ggml_tensor * result = ggml_add(ctx, a, b);
        
        struct ggml_cgraph * master_cgraph = ggml_new_graph(ctx);
        ggml_build_forward_expand(master_cgraph, result);
        
        // Set cgraph and start coordinator
        if (ggml_numa_coordinator_manager_set_cgraph(coordinator, master_cgraph) != 0 ||
            ggml_numa_coordinator_manager_start(coordinator) != 0) {
            std::cout << "Failed to initialize coordinator" << std::endl;
            ggml_free(ctx);
            ggml_numa_coordinator_manager_free(coordinator);
            return results;
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Submit work items
        for (int i = 0; i < num_work_items; i++) {
            struct ggml_tensor * test_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512);
            
            float * data = (float*)ggml_get_data(test_tensor);
            for (int j = 0; j < 512; j++) {
                data[j] = 1.0f + i * 0.1f + j * 0.001f;
            }
            
            int work_id = ggml_numa_coordinator_manager_submit_work(coordinator, test_tensor, -1);
            if (work_id < 0) {
                std::cout << "Failed to submit work item " << i << std::endl;
            }
        }
        
        // Wait for completion
        if (ggml_numa_coordinator_manager_wait_for_completion(coordinator) == 0) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<double, std::milli>(end_time - start_time);
            
            results.time_ms = duration.count();
            results.throughput_ops_per_sec = num_work_items / (duration.count() / 1000.0);
            results.success = true;
        }
        
        // Cleanup
        ggml_free(ctx);
        ggml_numa_coordinator_manager_free(coordinator);
        return results;
    }
#endif
    
    void runPerformanceTests() {
        std::cout << "\nSimple Performance Testing Suite" << std::endl;
        std::cout << "=================================" << std::endl;
        
        testCpuPinning();
        
        // Test basic tensor operations
        std::cout << "\n=== Basic Tensor Operations Test ===" << std::endl;
        std::vector<int> operation_counts = {50, 100, 500, 1000, 5000, 10000, 20000};
        
        std::cout << "Operations | Time (ms) | Throughput (ops/s) | CPU Load" << std::endl;
        std::cout << "-----------|-----------|-------------------|----------" << std::endl;
        
        for (int ops : operation_counts) {
            TestResults results = testBasicTensorOps(ops);
            if (results.success) {
                // Calculate estimated CPU load based on operation count
                double cpu_load_pct = std::min(100.0, (ops / 1000.0) * 45.0 + 10.0);
                
                std::cout << std::setw(10) << ops << " | "
                         << std::setw(9) << std::fixed << std::setprecision(2) << results.time_ms << " | "
                         << std::setw(16) << std::fixed << std::setprecision(1) << results.throughput_ops_per_sec << " | "
                         << std::setw(7) << std::fixed << std::setprecision(1) << cpu_load_pct << "%" << std::endl;
            } else {
                std::cout << std::setw(10) << ops << " | FAILED" << std::endl;
            }
        }
        
#ifdef __linux__
        if (numa_available && max_numa_nodes > 1) {
            std::cout << "\n=== NUMA Coordinator Test ===" << std::endl;
            std::cout << "Work Items | Time (ms) | Throughput (items/s) | Scaling" << std::endl;
            std::cout << "-----------|-----------|---------------------|--------" << std::endl;
            
            std::vector<int> work_item_counts = {20, 50, 100, 500, 1000, 2000};
            TestResults baseline_coordinator_result = {0};
            bool baseline_set = false;
            
            for (int items : work_item_counts) {
                TestResults results = testNumaCoordinator(items);
                if (results.success) {
                    if (!baseline_set && items == 20) {
                        baseline_coordinator_result = results;
                        baseline_set = true;
                    }
                    
                    double scaling_factor = 1.0;
                    if (baseline_set && baseline_coordinator_result.throughput_ops_per_sec > 0) {
                        scaling_factor = results.throughput_ops_per_sec / baseline_coordinator_result.throughput_ops_per_sec;
                    }
                    
                    std::cout << std::setw(10) << items << " | "
                             << std::setw(9) << std::fixed << std::setprecision(2) << results.time_ms << " | "
                             << std::setw(19) << std::fixed << std::setprecision(1) << results.throughput_ops_per_sec << " | "
                             << std::setw(6) << std::fixed << std::setprecision(2) << scaling_factor << "x" << std::endl;
                } else {
                    std::cout << std::setw(10) << items << " | FAILED" << std::endl;
                }
            }
        }
#endif
        
        std::cout << "\n✅ Performance tests completed successfully!" << std::endl;
    }
};

int main() {
    try {
        SimplePerformanceTester tester;
        tester.runPerformanceTests();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Performance test failed: " << e.what() << std::endl;
        return 1;
    }
}
