#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "common.h"

#include <chrono>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <thread>
#include <sched.h>

#ifdef __linux__
#include <numa.h>
#include <numaif.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

using namespace std::chrono;

struct BaselineResult {
    std::string test_name;
    std::string core_type;          // "Physical-Core" or "Hyperthread-Sibling"
    int cpu_id;
    double avg_time_ms;
    double min_time_ms;
    double max_time_ms;
    double ops_per_second;
    double gflops;
};

// Pin current thread to a specific CPU core
bool pin_to_cpu(int cpu_id) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0) {
        printf("❌ Failed to pin thread to CPU %d\n", cpu_id);
        return false;
    }
    
    // Verify the pinning worked
    cpu_set_t verify_set;
    CPU_ZERO(&verify_set);
    if (sched_getaffinity(0, sizeof(verify_set), &verify_set) == 0) {
        if (CPU_ISSET(cpu_id, &verify_set)) {
            printf("✅ Thread successfully pinned to CPU %d\n", cpu_id);
            return true;
        }
    }
    printf("⚠️  CPU pinning verification failed for CPU %d\n", cpu_id);
    return false;
#else
    printf("⚠️  CPU pinning not supported on this platform\n");
    return false;
#endif
}

// Get CPU topology to identify physical cores vs hyperthreads
std::vector<std::pair<int, bool>> get_cpu_topology() {
    std::vector<std::pair<int, bool>> cpu_info; // cpu_id, is_hyperthread
    
#ifdef __linux__
    // Read CPU topology from /sys/devices/system/cpu/
    int max_cpu = std::thread::hardware_concurrency();
    
    for (int cpu = 0; cpu < max_cpu; cpu++) {
        std::string thread_siblings_path = "/sys/devices/system/cpu/cpu" + 
                                         std::to_string(cpu) + "/topology/thread_siblings_list";
        
        FILE* fp = fopen(thread_siblings_path.c_str(), "r");
        if (fp) {
            char buffer[256];
            if (fgets(buffer, sizeof(buffer), fp)) {
                // Parse sibling list to determine if this is a hyperthread
                std::string siblings(buffer);
                size_t comma_pos = siblings.find(',');
                bool is_hyperthread = false;
                
                if (comma_pos != std::string::npos) {
                    // Has siblings, check if this CPU is the higher numbered one
                    int first_sibling = std::stoi(siblings.substr(0, comma_pos));
                    is_hyperthread = (cpu != first_sibling);
                }
                
                cpu_info.push_back({cpu, is_hyperthread});
            }
            fclose(fp);
        } else {
            // Fallback: assume even CPUs are physical, odd are hyperthreads
            cpu_info.push_back({cpu, cpu % 2 == 1});
        }
    }
#else
    // Fallback for non-Linux systems
    int max_cpu = std::thread::hardware_concurrency();
    for (int cpu = 0; cpu < max_cpu; cpu++) {
        cpu_info.push_back({cpu, cpu % 2 == 1}); // Simple even/odd assumption
    }
#endif
    
    return cpu_info;
}

BaselineResult run_single_core_benchmark(int cpu_id, bool is_hyperthread, int M, int N, int K, int num_iterations = 10) {
    BaselineResult result = {};
    result.cpu_id = cpu_id;
    result.core_type = is_hyperthread ? "Hyperthread-Sibling" : "Physical-Core";
    result.test_name = "Matrix Multiplication " + std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);
    
    printf("🧪 Testing %s on CPU %d (%s)\n", 
           result.test_name.c_str(), cpu_id, result.core_type.c_str());
    
    // Pin to the specific CPU
    if (!pin_to_cpu(cpu_id)) {
        result.avg_time_ms = -1.0; // Error indicator
        return result;
    }
    
    // Create test matrices using a simpler direct approach
    struct ggml_init_params params = {
        /*.mem_size   =*/ 256*1024*1024, // 256MB should be enough
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        printf("❌ Failed to initialize ggml context\n");
        result.avg_time_ms = -1.0;
        return result;
    }
    
    // Create tensors
    struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    struct ggml_tensor* result_tensor = ggml_mul_mat(ctx, a, b);
    
    // Initialize test data
    float* a_data = (float*)ggml_get_data(a);
    float* b_data = (float*)ggml_get_data(b);
    
    size_t a_elements = ggml_nelements(a);
    size_t b_elements = ggml_nelements(b);
    
    // Initialize test data with proven simple pattern
    for (size_t i = 0; i < a_elements; i++) {
        a_data[i] = 1.0f; // All 1s for simple math
    }
    for (size_t i = 0; i < b_elements; i++) {
        b_data[i] = 2.0f; // All 2s for simple math  
    }
    
    // Create computation graph (proven pattern)
    struct ggml_cgraph* cgraph = ggml_new_graph(ctx);
    ggml_build_forward_expand(cgraph, result_tensor);
    
    // Use direct GGML computation (proven working approach)
    ggml_cplan cplan = ggml_graph_plan(cgraph, 1, NULL); // 1 thread, no threadpool
    
    std::vector<double> times;
    times.reserve(num_iterations);
    
    // Warm-up run (proven pattern)
    ggml_status status = ggml_graph_compute(cgraph, &cplan);
    if (status != GGML_STATUS_SUCCESS) {
        printf("❌ Warm-up computation failed with status: %d\n", status);
        ggml_free(ctx);
        result.avg_time_ms = -1.0;
        return result;
    }
    
    // Benchmark runs (proven pattern)
    for (int i = 0; i < num_iterations; i++) {
        auto start = high_resolution_clock::now();
        
        status = ggml_graph_compute(cgraph, &cplan);
        
        auto end = high_resolution_clock::now();
        
        if (status != GGML_STATUS_SUCCESS) {
            printf("❌ Graph computation failed on iteration %d with status: %d\n", i, status);
            break;
        }
        
        double time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        times.push_back(time_ms);
        
        if (i == 0) {
            printf("   First iteration: %.2f ms\n", time_ms);
        }
    }
    
    ggml_free(ctx);
    
    if (times.empty()) {
        result.avg_time_ms = -1.0;
        return result;
    }
    
    // Calculate statistics
    result.avg_time_ms = 0.0;
    result.min_time_ms = times[0];
    result.max_time_ms = times[0];
    
    for (double time : times) {
        result.avg_time_ms += time;
        result.min_time_ms = std::min(result.min_time_ms, time);
        result.max_time_ms = std::max(result.max_time_ms, time);
    }
    result.avg_time_ms /= times.size();
    
    // Calculate performance metrics
    double total_ops = (double)M * N * K * 2.0; // Multiply-add operations
    result.ops_per_second = total_ops / (result.avg_time_ms / 1000.0);
    result.gflops = result.ops_per_second / 1e9;
    
    printf("   ✅ Average: %.2f ms, Min: %.2f ms, Max: %.2f ms\n", 
           result.avg_time_ms, result.min_time_ms, result.max_time_ms);
    printf("   📊 %.2f GFLOPS\n", result.gflops);
    
    return result;
}

void print_baseline_summary(const std::vector<BaselineResult>& results) {
    printf("\n================================================================================\n");
    printf("SINGLE-CORE BASELINE PERFORMANCE SUMMARY\n");
    printf("================================================================================\n");
    
    printf("%-40s %-20s %-8s %-12s %-12s %-12s\n", 
           "Test", "Core Type", "CPU", "Avg (ms)", "GFLOPS", "Efficiency");
    printf("----------------------------------------------------------------------------------------------------\n");
    
    // Find best physical core performance for efficiency calculation
    double best_physical_gflops = 0.0;
    for (const auto& result : results) {
        if (result.core_type == "Physical-Core" && result.gflops > best_physical_gflops) {
            best_physical_gflops = result.gflops;
        }
    }
    
    for (const auto& result : results) {
        if (result.avg_time_ms < 0) continue; // Skip failed tests
        
        double efficiency = (best_physical_gflops > 0) ? 
                          (result.gflops / best_physical_gflops * 100.0) : 100.0;
        
        printf("%-40s %-20s %-8d %-12.2f %-12.2f %-11.1f%%\n",
               result.test_name.c_str(),
               result.core_type.c_str(),
               result.cpu_id,
               result.avg_time_ms,
               result.gflops,
               efficiency);
    }
    
    printf("\n");
    
    // Summary statistics
    double total_physical_gflops = 0.0, total_ht_gflops = 0.0;
    int physical_count = 0, ht_count = 0;
    
    for (const auto& result : results) {
        if (result.avg_time_ms < 0) continue;
        
        if (result.core_type == "Physical-Core") {
            total_physical_gflops += result.gflops;
            physical_count++;
        } else {
            total_ht_gflops += result.gflops;
            ht_count++;
        }
    }
    
    if (physical_count > 0 && ht_count > 0) {
        double avg_physical = total_physical_gflops / physical_count;
        double avg_ht = total_ht_gflops / ht_count;
        double ht_efficiency = (avg_ht / avg_physical) * 100.0;
        
        printf("📈 BASELINE INSIGHTS:\n");
        printf("   Physical Cores Average: %.2f GFLOPS\n", avg_physical);
        printf("   Hyperthread Average:    %.2f GFLOPS\n", avg_ht);
        printf("   HT Efficiency:          %.1f%% of physical core performance\n", ht_efficiency);
        
        if (ht_efficiency < 70.0) {
            printf("   💡 Hyperthreading shows significant performance drop - prefer physical cores\n");
        } else if (ht_efficiency > 90.0) {
            printf("   🚀 Hyperthreading performs nearly as well as physical cores\n");
        } else {
            printf("   ⚖️  Hyperthreading provides moderate performance reduction\n");
        }
    }
}

int main() {
    printf("Single-Core Baseline Performance Test\n");
    printf("====================================\n");
    printf("Establishing performance baseline for single-core execution\n\n");
    
    // Get CPU topology
    auto cpu_topology = get_cpu_topology();
    if (cpu_topology.empty()) {
        printf("❌ Failed to get CPU topology information\n");
        return 1;
    }
    
    printf("🖥️  CPU Topology detected:\n");
    for (const auto& [cpu_id, is_ht] : cpu_topology) {
        printf("   CPU %d: %s\n", cpu_id, is_ht ? "Hyperthread" : "Physical Core");
    }
    printf("\n");
    
    std::vector<BaselineResult> all_results;
    
    // Test different matrix sizes for comprehensive baseline
    std::vector<std::tuple<int, int, int>> test_sizes = {
        {512, 512, 512},    // Medium workload
        {1024, 1024, 1024}, // Large workload  
        {2048, 512, 1024},  // Rectangular workload
    };
    
    for (const auto& [M, N, K] : test_sizes) {
        printf("🧮 Matrix Size: %dx%dx%d\n", M, N, K);
        printf("==================================================\n");
        
        // Test one physical core and one hyperthread for comparison
        int physical_cpu = -1, hyperthread_cpu = -1;
        
        for (const auto& [cpu_id, is_ht] : cpu_topology) {
            if (!is_ht && physical_cpu == -1) {
                physical_cpu = cpu_id;
            } else if (is_ht && hyperthread_cpu == -1) {
                hyperthread_cpu = cpu_id;
            }
            
            if (physical_cpu != -1 && hyperthread_cpu != -1) break;
        }
        
        // Test physical core
        if (physical_cpu != -1) {
            BaselineResult result = run_single_core_benchmark(physical_cpu, false, M, N, K);
            all_results.push_back(result);
        }
        
        // Test hyperthread
        if (hyperthread_cpu != -1) {
            BaselineResult result = run_single_core_benchmark(hyperthread_cpu, true, M, N, K);
            all_results.push_back(result);
        }
        
        printf("\n");
    }
    
    // Print comprehensive summary
    print_baseline_summary(all_results);
    
    printf("✅ Baseline testing complete!\n");
    printf("💡 Use these results to compare against multi-core NUMA performance\n");
    
    return 0;
}
