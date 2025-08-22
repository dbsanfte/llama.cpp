/**
 * NUMA CPU Thread Affinity and Memory Access Pattern Tester
 * 
 * Since memory allocation shows perfect locality, let's check if the issue
 * is with CPU thread affinity or actual memory access patterns during execution.
 */

#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cstring>
#include <iomanip>

// NUMA headers
#ifdef __linux__
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#include <unistd.h>
#endif

class NumaCpuAffinityTester {
private:
    int num_numa_nodes;
    int num_cpus;
    
public:
    NumaCpuAffinityTester() {
#ifdef __linux__
        if (numa_available() == -1) {
            std::cerr << "❌ NUMA not available" << std::endl;
            exit(1);
        }
        
        num_numa_nodes = numa_num_configured_nodes();
        num_cpus = numa_num_configured_cpus();
        
        std::cout << "🧪 NUMA CPU Affinity & Memory Access Tester" << std::endl;
        std::cout << "===========================================" << std::endl;
        std::cout << "📋 System: " << num_numa_nodes << " NUMA nodes, " << num_cpus << " CPUs" << std::endl;
#endif
    }
    
    /**
     * Get which CPUs belong to which NUMA node
     */
    void print_cpu_numa_mapping() {
        std::cout << "\n🗺️  CPU to NUMA Node Mapping:" << std::endl;
        
#ifdef __linux__
        for (int node = 0; node < num_numa_nodes; node++) {
            struct bitmask* cpumask = numa_allocate_cpumask();
            numa_node_to_cpus(node, cpumask);
            
            std::cout << "   NUMA Node " << node << ": CPUs ";
            bool first = true;
            for (int cpu = 0; cpu < num_cpus; cpu++) {
                if (numa_bitmask_isbitset(cpumask, cpu)) {
                    if (!first) std::cout << ", ";
                    std::cout << cpu;
                    first = false;
                }
            }
            std::cout << std::endl;
            
            numa_free_cpumask(cpumask);
        }
#endif
    }
    
    /**
     * Test memory bandwidth from different CPUs to different NUMA nodes
     */
    void test_memory_bandwidth_pattern() {
        std::cout << "\n🚀 Memory Bandwidth Test (Local vs Remote Access)" << std::endl;
        std::cout << "=================================================" << std::endl;
        
        size_t test_size = 64 * 1024 * 1024; // 64 MB like our ADD test
        int iterations = 5;
        
#ifdef __linux__
        for (int cpu_node = 0; cpu_node < num_numa_nodes; cpu_node++) {
            // Get a CPU from this node
            struct bitmask* cpumask = numa_allocate_cpumask();
            numa_node_to_cpus(cpu_node, cpumask);
            
            int test_cpu = -1;
            for (int cpu = 0; cpu < num_cpus; cpu++) {
                if (numa_bitmask_isbitset(cpumask, cpu)) {
                    test_cpu = cpu;
                    break;
                }
            }
            numa_free_cpumask(cpumask);
            
            if (test_cpu == -1) continue;
            
            std::cout << "\n🔸 Testing from CPU " << test_cpu << " (NUMA Node " << cpu_node << "):" << std::endl;
            
            for (int mem_node = 0; mem_node < num_numa_nodes; mem_node++) {
                // Allocate memory on specific node
                void* src1 = numa_alloc_onnode(test_size, mem_node);
                void* src2 = numa_alloc_onnode(test_size, mem_node);
                void* dst = numa_alloc_onnode(test_size, mem_node);
                
                if (!src1 || !src2 || !dst) {
                    std::cerr << "   ❌ Allocation failed for node " << mem_node << std::endl;
                    continue;
                }
                
                // Initialize memory
                memset(src1, 0x42, test_size);
                memset(src2, 0x43, test_size);
                
                // Set thread affinity to specific CPU
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(test_cpu, &cpuset);
                pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
                
                // Verify we're on the right CPU
                int actual_cpu = sched_getcpu();
                if (actual_cpu != test_cpu) {
                    std::cout << "   ⚠️  Warning: requested CPU " << test_cpu << " but got " << actual_cpu << std::endl;
                }
                
                // Run memory bandwidth test (ADD-like operation)
                double total_time = 0.0;
                for (int i = 0; i < iterations; i++) {
                    auto start = std::chrono::high_resolution_clock::now();
                    
                    // Simulate ADD operation - memory intensive
                    float* s1 = (float*)src1;
                    float* s2 = (float*)src2;
                    float* d = (float*)dst;
                    size_t elements = test_size / sizeof(float);
                    
                    for (size_t j = 0; j < elements; j++) {
                        d[j] = s1[j] + s2[j];
                    }
                    
                    auto end = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                    total_time += duration.count() / 1000.0; // Convert to milliseconds
                }
                
                double avg_time = total_time / iterations;
                double bandwidth = (3.0 * test_size / 1024 / 1024 / 1024) / (avg_time / 1000.0); // GB/s
                
                std::string access_type = (cpu_node == mem_node) ? "LOCAL " : "REMOTE";
                std::cout << "   " << access_type << " access to Node " << mem_node 
                          << ": " << std::fixed << std::setprecision(1) << avg_time << " ms"
                          << " (" << std::setprecision(2) << bandwidth << " GB/s)" << std::endl;
                
                numa_free(src1, test_size);
                numa_free(src2, test_size);
                numa_free(dst, test_size);
            }
        }
#endif
    }
    
    /**
     * Test what happens when we bind threads to wrong NUMA nodes
     */
    void test_wrong_cpu_binding() {
        std::cout << "\n⚠️  Testing Wrong CPU Binding Effects" << std::endl;
        std::cout << "======================================" << std::endl;
        
        size_t test_size = 64 * 1024 * 1024; // 64 MB
        
#ifdef __linux__
        for (int mem_node = 0; mem_node < num_numa_nodes; mem_node++) {
            std::cout << "\n🔸 Memory allocated on NUMA Node " << mem_node << ":" << std::endl;
            
            // Allocate memory on specific node
            void* src1 = numa_alloc_onnode(test_size, mem_node);
            void* src2 = numa_alloc_onnode(test_size, mem_node);
            void* dst = numa_alloc_onnode(test_size, mem_node);
            
            if (!src1 || !src2 || !dst) {
                std::cerr << "   ❌ Allocation failed" << std::endl;
                continue;
            }
            
            memset(src1, 0x42, test_size);
            memset(src2, 0x43, test_size);
            
            // Test execution from different NUMA nodes
            for (int cpu_node = 0; cpu_node < num_numa_nodes; cpu_node++) {
                // Get a CPU from this node
                struct bitmask* cpumask = numa_allocate_cpumask();
                numa_node_to_cpus(cpu_node, cpumask);
                
                int test_cpu = -1;
                for (int cpu = 0; cpu < num_cpus; cpu++) {
                    if (numa_bitmask_isbitset(cpumask, cpu)) {
                        test_cpu = cpu;
                        break;
                    }
                }
                numa_free_cpumask(cpumask);
                
                if (test_cpu == -1) continue;
                
                // Bind to specific CPU
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(test_cpu, &cpuset);
                pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
                
                // Run test
                auto start = std::chrono::high_resolution_clock::now();
                
                float* s1 = (float*)src1;
                float* s2 = (float*)src2;
                float* d = (float*)dst;
                size_t elements = test_size / sizeof(float);
                
                for (size_t j = 0; j < elements; j++) {
                    d[j] = s1[j] + s2[j];
                }
                
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                double time_ms = duration.count() / 1000.0;
                double bandwidth = (3.0 * test_size / 1024 / 1024 / 1024) / (time_ms / 1000.0);
                
                int actual_cpu = sched_getcpu();
                int actual_node = numa_node_of_cpu(actual_cpu);
                
                std::string binding_type = (cpu_node == mem_node) ? "CORRECT" : "WRONG  ";
                std::cout << "   " << binding_type << " CPU binding (CPU " << actual_cpu 
                          << ", Node " << actual_node << "): " 
                          << std::fixed << std::setprecision(1) << time_ms << " ms"
                          << " (" << std::setprecision(2) << bandwidth << " GB/s)" << std::endl;
            }
            
            numa_free(src1, test_size);
            numa_free(src2, test_size);
            numa_free(dst, test_size);
        }
#endif
    }
    
    /**
     * Check current process thread scheduling
     */
    void check_thread_scheduling() {
        std::cout << "\n📋 Current Process Thread Scheduling" << std::endl;
        std::cout << "====================================" << std::endl;
        
#ifdef __linux__
        int policy = sched_getscheduler(0);
        std::cout << "📊 Scheduling policy: ";
        switch (policy) {
            case SCHED_OTHER: std::cout << "SCHED_OTHER (normal)"; break;
            case SCHED_RR: std::cout << "SCHED_RR (round-robin)"; break;
            case SCHED_FIFO: std::cout << "SCHED_FIFO (first-in-first-out)"; break;
            default: std::cout << "Unknown (" << policy << ")"; break;
        }
        std::cout << std::endl;
        
        struct sched_param param;
        if (sched_getparam(0, &param) == 0) {
            std::cout << "📊 Priority: " << param.sched_priority << std::endl;
        }
        
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        if (sched_getaffinity(0, sizeof(cpuset), &cpuset) == 0) {
            std::cout << "📊 Current CPU affinity allows " << CPU_COUNT(&cpuset) << " CPUs" << std::endl;
        }
        
        int current_cpu = sched_getcpu();
        int current_node = numa_node_of_cpu(current_cpu);
        std::cout << "📊 Currently running on CPU " << current_cpu << " (NUMA Node " << current_node << ")" << std::endl;
#endif
    }
};

int main() {
    try {
        NumaCpuAffinityTester tester;
        
        // Show CPU-NUMA mapping
        tester.print_cpu_numa_mapping();
        
        // Check current scheduling
        tester.check_thread_scheduling();
        
        // Test memory access patterns
        tester.test_memory_bandwidth_pattern();
        
        // Test wrong CPU binding effects
        tester.test_wrong_cpu_binding();
        
        std::cout << "\n🏁 CPU affinity and memory access testing complete!" << std::endl;
        std::cout << "Look for performance differences between local vs remote access." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
