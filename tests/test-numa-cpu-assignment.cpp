/*
 * NUMA CPU Assignment Analysis Test
 * 
 * This test analyzes CPU core assignment patterns to identify
 * hyperthreading conflicts that may limit NUMA coordinator scaling.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <string.h>
#include <sys/syscall.h>
#include <thread>
#include <atomic>
#include <vector>
#include <chrono>
#include <cmath>
#include <string>
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "common.h"

// Thread analysis data
struct cpu_analysis {
    int thread_id;
    pid_t tid;
    int current_cpu;
    int physical_core;
    bool is_hyperthread_sibling;
};

// Global tracking
std::vector<cpu_analysis> cpu_usage;
std::atomic<bool> analysis_active{false};

// Function to get current CPU assignment
cpu_analysis get_current_cpu_info(int thread_id) {
    cpu_analysis info = {};
    info.thread_id = thread_id;
    info.tid = syscall(SYS_gettid);
    
    // Get current CPU using sched_getcpu()
    info.current_cpu = sched_getcpu();
    
    if (info.current_cpu >= 0) {
        // Calculate physical core (Intel hyperthreading pattern)
        // CPUs 0,1 -> core 0, CPUs 2,3 -> core 1, etc.
        info.physical_core = info.current_cpu / 2;
        info.is_hyperthread_sibling = (info.current_cpu % 2 == 1);
    } else {
        info.physical_core = -1;
        info.is_hyperthread_sibling = false;
    }
    
    return info;
}

// Computation worker function for CPU analysis
void cpu_analysis_worker(int worker_id, std::atomic<int>& active_workers) {
    // Record CPU assignment
    cpu_analysis info = get_current_cpu_info(worker_id);
    cpu_usage.push_back(info);
    
    printf("   Worker %2d: TID=%6d, CPU=%2d, Core=%2d, HT=%s\n",
           worker_id, info.tid, info.current_cpu, info.physical_core,
           info.is_hyperthread_sibling ? "Yes" : "No ");
    
    // Do some actual computation to keep CPU busy
    volatile double result = 0.0;
    for (int i = 0; i < 10000000 && analysis_active; i++) {
        result += sin(i * 0.001) * cos(i * 0.001);
    }
    
    active_workers--;
}

void analyze_cpu_assignment_patterns() {
    printf("🔍 CPU Assignment Pattern Analysis\n");
    printf("==================================\n\n");
    
    printf("📊 System Information:\n");
    printf("   Physical cores: 11\n");
    printf("   Logical CPUs: 22 (with hyperthreading)\n");
    printf("   CPU pairs: 0,1->Core0  2,3->Core1  4,5->Core2  etc.\n\n");
    
    // Clear previous data
    cpu_usage.clear();
    
    // Test different thread counts to see assignment patterns
    std::vector<int> test_thread_counts = {2, 4, 8, 16, 22};
    
    for (int thread_count : test_thread_counts) {
        printf("🧵 Testing %d threads:\n", thread_count);
        
        analysis_active = true;
        std::atomic<int> active_workers{thread_count};
        std::vector<std::thread> workers;
        
        // Launch worker threads
        for (int i = 0; i < thread_count; i++) {
            workers.emplace_back(cpu_analysis_worker, i, std::ref(active_workers));
        }
        
        // Let them run briefly
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        analysis_active = false;
        
        // Wait for completion
        for (auto& worker : workers) {
            worker.join();
        }
        
        // Analyze the results
        std::vector<int> core_usage(11, 0);  // Track usage per physical core
        std::vector<int> ht_conflicts(11, 0); // Track hyperthreading conflicts
        
        for (const auto& info : cpu_usage) {
            if (info.physical_core >= 0 && info.physical_core < 11) {
                core_usage[info.physical_core]++;
                if (core_usage[info.physical_core] > 1) {
                    ht_conflicts[info.physical_core]++;
                }
            }
        }
        
        // Report conflicts
        int total_conflicts = 0;
        for (int i = 0; i < 11; i++) {
            if (ht_conflicts[i] > 0) {
                total_conflicts++;
                printf("     ⚠️  Core %d: %d threads (hyperthreading conflict!)\n", 
                       i, core_usage[i]);
            }
        }
        
        if (total_conflicts == 0) {
            printf("     ✅ No hyperthreading conflicts detected\n");
        } else {
            printf("     🚨 %d physical cores have hyperthreading conflicts\n", total_conflicts);
        }
        
        printf("\n");
        cpu_usage.clear();
    }
}

void test_numa_coordinator_cpu_usage() {
    printf("🎯 NUMA Coordinator CPU Usage Test\n");
    printf("===================================\n\n");
    
    // Create a simple computational workload
    struct ggml_init_params params = {};
    params.mem_size = 64 * 1024 * 1024;  // 64MB
    params.mem_buffer = NULL;
    params.no_alloc = false;
    
    struct ggml_context *ctx = ggml_init(params);
    if (!ctx) {
        printf("❌ Failed to create GGML context\n");
        return;
    }
    
    printf("📊 Creating computational workload for CPU analysis...\n");
    
    // Create computation that should use multiple threads
    const int64_t size = 500000;  // 500K elements
    struct ggml_tensor *a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, size);
    struct ggml_tensor *b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, size);
    struct ggml_tensor *result = ggml_add(ctx, a, b);
    
    // Initialize data
    float *a_data = (float *)ggml_get_data(a);
    float *b_data = (float *)ggml_get_data(b);
    
    for (int64_t i = 0; i < size; i++) {
        a_data[i] = 1.0f + i * 0.001f;
        b_data[i] = 2.0f + i * 0.001f;
    }
    
    // Build computation graph
    struct ggml_cgraph *cgraph = ggml_new_graph(ctx);
    ggml_build_forward_expand(cgraph, result);
    
    printf("✅ Created ADD operation with %ld elements\n", size);
    printf("🚀 Executing computation while monitoring CPU usage...\n\n");
    
    // Execute multiple times to see CPU assignment patterns
    for (int run = 0; run < 3; run++) {
        printf("Run %d:\n", run + 1);
        
        // Execute computation
        int n_threads = 22;  // Use all logical CPUs
        struct ggml_cplan cplan = ggml_graph_plan(cgraph, n_threads, nullptr);
        cplan.work_data = (uint8_t *)malloc(cplan.work_size);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // This should trigger multi-threaded execution
        ggml_graph_compute(cgraph, &cplan);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        printf("   Execution time: %ld μs\n", duration.count());
        
        // Verify result
        float *result_data = (float *)ggml_get_data(result);
        float expected = 3.0f;  // 1.0 + 2.0
        printf("   Result verification: %.3f (expected %.3f) %s\n", 
               result_data[0], expected, 
               (fabs(result_data[0] - expected) < 0.01f) ? "✅" : "❌");
        
        free(cplan.work_data);
        printf("\n");
    }
    
    printf("💡 Key Insights:\n");
    printf("================\n");
    printf("• If hyperthreading conflicts exist, we'll see reduced scaling\n");
    printf("• Virtual NUMA nodes should use distinct physical cores\n");
    printf("• Optimal assignment: separate even/odd CPUs or complete core groups\n");
    
    ggml_free(ctx);
}

int main() {
    printf("🧪 NUMA CPU Assignment Analysis\n");
    printf("===============================\n");
    printf("Analyzing CPU core assignment patterns for NUMA coordinator optimization\n\n");
    
    // Analyze basic CPU assignment patterns
    analyze_cpu_assignment_patterns();
    
    printf("\n");
    printf("============================================================\n");
    printf("\n");
    
    // Test NUMA coordinator CPU usage
    test_numa_coordinator_cpu_usage();
    
    printf("\n📋 Analysis completed. Check for hyperthreading conflicts that limit scaling.\n");
    return 0;
}
