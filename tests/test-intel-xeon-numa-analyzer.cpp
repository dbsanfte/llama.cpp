/**
 * Intel Xeon NUMA Performance Analysis Tool
 * 
 * This tool specifically investigates why NUMA Node 1 performs better than Node 0
 * in our ADD kernel, taking into account the Intel Xeon topology with hyperthreading.
 */

#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <map>
#include <set>

// NUMA headers
#ifdef __linux__
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#include <unistd.h>
#endif

class IntelXeonNumaAnalyzer {
private:
    struct CpuInfo {
        int logical_cpu;
        int physical_core;
        int socket;
        int numa_node;
        double frequency_mhz;
    };
    
    std::vector<CpuInfo> cpu_topology;
    
public:
    IntelXeonNumaAnalyzer() {
        std::cout << "🔬 Intel Xeon NUMA Performance Analyzer" << std::endl;
        std::cout << "=======================================" << std::endl;
        
        build_cpu_topology();
        print_topology_summary();
    }
    
    /**
     * Build CPU topology map
     */
    void build_cpu_topology() {
        std::cout << "📋 Building CPU topology map..." << std::endl;
        
        // Read CPU topology from /proc/cpuinfo
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::string line;
        CpuInfo current_cpu = {};
        
        while (std::getline(cpuinfo, line)) {
            std::istringstream iss(line);
            std::string key, value;
            
            if (std::getline(iss, key, ':') && std::getline(iss, value)) {
                // Trim whitespace
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);
                
                if (key == "processor") {
                    current_cpu.logical_cpu = std::stoi(value);
                } else if (key == "physical id") {
                    current_cpu.socket = std::stoi(value);
                } else if (key == "core id") {
                    current_cpu.physical_core = std::stoi(value);
                } else if (key == "cpu MHz") {
                    current_cpu.frequency_mhz = std::stod(value);
                    
                    // Determine NUMA node from CPU number
                    int cpu = current_cpu.logical_cpu;
                    if ((cpu >= 0 && cpu <= 27) || (cpu >= 56 && cpu <= 83)) {
                        current_cpu.numa_node = 0;
                    } else {
                        current_cpu.numa_node = 1;
                    }
                    
                    cpu_topology.push_back(current_cpu);
                    current_cpu = {};
                }
            }
        }
    }
    
    /**
     * Print topology summary
     */
    void print_topology_summary() {
        std::cout << "\n📊 CPU Topology Analysis:" << std::endl;
        std::cout << "=========================" << std::endl;
        
        // Analyze frequency distribution
        double node0_freq_sum = 0, node1_freq_sum = 0;
        int node0_count = 0, node1_count = 0;
        double node0_min = 999999, node0_max = 0;
        double node1_min = 999999, node1_max = 0;
        
        for (const auto& cpu : cpu_topology) {
            if (cpu.numa_node == 0) {
                node0_freq_sum += cpu.frequency_mhz;
                node0_count++;
                node0_min = std::min(node0_min, cpu.frequency_mhz);
                node0_max = std::max(node0_max, cpu.frequency_mhz);
            } else {
                node1_freq_sum += cpu.frequency_mhz;
                node1_count++;
                node1_min = std::min(node1_min, cpu.frequency_mhz);
                node1_max = std::max(node1_max, cpu.frequency_mhz);
            }
        }
        
        std::cout << "🔸 NUMA Node 0 (" << node0_count << " CPUs):" << std::endl;
        std::cout << "   Frequency: " << std::fixed << std::setprecision(1) 
                  << (node0_freq_sum / node0_count) << " MHz avg"
                  << " (min: " << node0_min << ", max: " << node0_max << ")" << std::endl;
        
        std::cout << "🔸 NUMA Node 1 (" << node1_count << " CPUs):" << std::endl;
        std::cout << "   Frequency: " << std::fixed << std::setprecision(1) 
                  << (node1_freq_sum / node1_count) << " MHz avg"
                  << " (min: " << node1_min << ", max: " << node1_max << ")" << std::endl;
        
        // Check hyperthreading pattern
        std::cout << "\n🧵 Hyperthreading Analysis:" << std::endl;
        std::cout << "===========================" << std::endl;
        
        // Show physical core to logical CPU mapping for first few cores
        std::cout << "Sample core-to-CPU mapping:" << std::endl;
        std::map<std::pair<int,int>, std::vector<int>> core_map; // socket,core -> logical CPUs
        
        for (const auto& cpu : cpu_topology) {
            core_map[{cpu.socket, cpu.physical_core}].push_back(cpu.logical_cpu);
        }
        
        int shown = 0;
        for (const auto& entry : core_map) {
            if (shown >= 8) break; // Show first 8 cores
            
            int socket = entry.first.first;
            int core = entry.first.second;
            const auto& cpus = entry.second;
            
            std::cout << "   Socket " << socket << " Core " << std::setw(2) << core << ": CPUs ";
            for (size_t i = 0; i < cpus.size(); i++) {
                std::cout << std::setw(2) << cpus[i];
                if (i < cpus.size() - 1) std::cout << ",";
            }
            std::cout << std::endl;
            shown++;
        }
    }
    
    /**
     * Test performance across different CPU patterns with controlled process affinity
     */
    void test_cpu_performance_patterns() {
        std::cout << "\n🚀 CPU Performance Pattern Analysis" << std::endl;
        std::cout << "===================================" << std::endl;
        
        // First, test from different process starting points
        test_process_affinity_effects();
        
        size_t test_size = 64 * 1024 * 1024; // 64 MB
        
        // Test pattern 1: Physical cores only (no hyperthreading)
        test_physical_cores_only(test_size);
        
        // Test pattern 2: Hyperthreading siblings
        test_hyperthreading_effect(test_size);
        
        // Test pattern 3: Socket-specific performance
        test_socket_performance(test_size);
        
        // Test pattern 4: Memory controller effects
        test_memory_controller_effects(test_size);
        
        // Test pattern 5: Cross-node effects with controlled affinity
        test_cross_node_controlled_affinity(test_size);
    }
    
private:
    /**
     * Test how the process's initial CPU affinity affects measurements
     */
    void test_process_affinity_effects() {
        std::cout << "\n🔸 Process Affinity Effects Test:" << std::endl;
        std::cout << "   Testing same operation from different starting CPUs..." << std::endl;
        
        size_t test_size = 32 * 1024 * 1024; // 32 MB for quick test
        
        // Test representative CPUs from each node
        std::vector<int> test_cpus = {0, 28, 56, 84}; // One from each major CPU range
        
        for (int starting_cpu : test_cpus) {
            std::cout << "\n   🔄 Starting from CPU " << starting_cpu 
                      << " (Node " << numa_node_of_cpu(starting_cpu) << "):" << std::endl;
            
            // Bind process to starting CPU
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(starting_cpu, &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
            
            // Test memory access to both NUMA nodes from this starting point
            for (int mem_node = 0; mem_node < 2; mem_node++) {
                void* memory = numa_alloc_onnode(test_size, mem_node);
                if (!memory) continue;
                
                memset(memory, 0x42, test_size);
                
                auto start = std::chrono::high_resolution_clock::now();
                
                // ADD-like operation
                float* data = (float*)memory;
                size_t elements = test_size / sizeof(float);
                for (size_t i = 0; i < elements; i++) {
                    data[i] = data[i] + 1.0f;
                }
                
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                double time_ms = duration.count() / 1000.0;
                double bandwidth = (2.0 * test_size / 1024.0 / 1024.0 / 1024.0) / (time_ms / 1000.0);
                
                int actual_cpu = sched_getcpu();
                std::string access_type = (numa_node_of_cpu(actual_cpu) == mem_node) ? "LOCAL " : "REMOTE";
                
                std::cout << "     " << access_type << " access to Node " << mem_node 
                          << " (actual CPU: " << actual_cpu << "): "
                          << std::fixed << std::setprecision(1) << time_ms << " ms"
                          << " (" << std::setprecision(2) << bandwidth << " GB/s)" << std::endl;
                
                numa_free(memory, test_size);
            }
        }
        
        // Reset to allow all CPUs
        cpu_set_t full_cpuset;
        CPU_ZERO(&full_cpuset);
        for (int cpu = 0; cpu < 112; cpu++) {
            CPU_SET(cpu, &full_cpuset);
        }
        pthread_setaffinity_np(pthread_self(), sizeof(full_cpuset), &full_cpuset);
    }
    
    /**
     * Test cross-node effects with tightly controlled CPU affinity
     */
    void test_cross_node_controlled_affinity(size_t test_size) {
        std::cout << "\n🔸 Cross-Node Controlled Affinity Test:" << std::endl;
        std::cout << "   Systematically testing all CPU→Memory combinations..." << std::endl;
        
        // Test matrix: each NUMA node's CPU accessing each NUMA node's memory
        struct TestResult {
            int cpu_node, mem_node, test_cpu, actual_cpu;
            double time_ms, bandwidth;
        };
        
        std::vector<TestResult> results;
        
        // Get representative CPUs from each node (avoid hyperthreading siblings)
        std::vector<int> node0_cpus = {0, 2, 4, 6};   // Physical cores from Node 0
        std::vector<int> node1_cpus = {28, 30, 32, 34}; // Physical cores from Node 1
        
        std::vector<std::vector<int>> test_cpus_by_node = {node0_cpus, node1_cpus};
        
        for (int cpu_node = 0; cpu_node < 2; cpu_node++) {
            for (int mem_node = 0; mem_node < 2; mem_node++) {
                std::cout << "\n     CPU Node " << cpu_node << " → Memory Node " << mem_node << ":" << std::endl;
                
                double total_time = 0, total_bandwidth = 0;
                int valid_tests = 0;
                
                for (int test_cpu : test_cpus_by_node[cpu_node]) {
                    // Strictly bind to specific CPU
                    cpu_set_t cpuset;
                    CPU_ZERO(&cpuset);
                    CPU_SET(test_cpu, &cpuset);
                    
                    if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) != 0) {
                        std::cout << "       ⚠️  Failed to bind to CPU " << test_cpu << std::endl;
                        continue;
                    }
                    
                    // Allocate memory on specific node
                    void* memory = numa_alloc_onnode(test_size / 4, mem_node); // Smaller for more tests
                    if (!memory) {
                        std::cout << "       ⚠️  Failed to allocate on Node " << mem_node << std::endl;
                        continue;
                    }
                    
                    memset(memory, 0x42, test_size / 4);
                    
                    // Verify we're on the right CPU
                    int actual_cpu = sched_getcpu();
                    if (actual_cpu != test_cpu) {
                        std::cout << "       ⚠️  CPU binding failed: requested " << test_cpu 
                                  << " got " << actual_cpu << std::endl;
                        numa_free(memory, test_size / 4);
                        continue;
                    }
                    
                    // Run performance test
                    auto start = std::chrono::high_resolution_clock::now();
                    
                    float* data = (float*)memory;
                    size_t elements = (test_size / 4) / sizeof(float);
                    for (size_t i = 0; i < elements; i++) {
                        data[i] = data[i] + 1.0f;
                    }
                    
                    auto end = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                    double time_ms = duration.count() / 1000.0;
                    double bandwidth = (2.0 * test_size / 4 / 1024.0 / 1024.0 / 1024.0) / (time_ms / 1000.0);
                    
                    results.push_back({cpu_node, mem_node, test_cpu, actual_cpu, time_ms, bandwidth});
                    
                    total_time += time_ms;
                    total_bandwidth += bandwidth;
                    valid_tests++;
                    
                    std::cout << "       CPU " << std::setw(2) << test_cpu << ": "
                              << std::fixed << std::setprecision(1) << time_ms << " ms"
                              << " (" << std::setprecision(2) << bandwidth << " GB/s)" << std::endl;
                    
                    numa_free(memory, test_size / 4);
                }
                
                if (valid_tests > 0) {
                    double avg_time = total_time / valid_tests;
                    double avg_bandwidth = total_bandwidth / valid_tests;
                    std::cout << "       📊 Average: " << std::fixed << std::setprecision(1) << avg_time << " ms"
                              << " (" << std::setprecision(2) << avg_bandwidth << " GB/s)" << std::endl;
                }
            }
        }
        
        // Print summary matrix
        std::cout << "\n   📋 Performance Matrix Summary:" << std::endl;
        std::cout << "      CPU Node → Memory Node | Avg Bandwidth" << std::endl;
        std::cout << "      ----------------------|---------------" << std::endl;
        
        for (int cpu_node = 0; cpu_node < 2; cpu_node++) {
            for (int mem_node = 0; mem_node < 2; mem_node++) {
                double total_bandwidth = 0;
                int count = 0;
                
                for (const auto& result : results) {
                    if (result.cpu_node == cpu_node && result.mem_node == mem_node) {
                        total_bandwidth += result.bandwidth;
                        count++;
                    }
                }
                
                if (count > 0) {
                    double avg_bandwidth = total_bandwidth / count;
                    std::string pattern = (cpu_node == mem_node) ? "LOCAL " : "REMOTE";
                    std::cout << "      Node " << cpu_node << " → Node " << mem_node 
                              << " (" << pattern << ")  | " 
                              << std::fixed << std::setprecision(2) << avg_bandwidth << " GB/s" << std::endl;
                }
            }
        }
        
        // Reset affinity
        cpu_set_t full_cpuset;
        CPU_ZERO(&full_cpuset);
        for (int cpu = 0; cpu < 112; cpu++) {
            CPU_SET(cpu, &full_cpuset);
        }
        pthread_setaffinity_np(pthread_self(), sizeof(full_cpuset), &full_cpuset);
    }

    void test_physical_cores_only(size_t test_size) {
        std::cout << "\n🔸 Physical Cores Only Test (No Hyperthreading):" << std::endl;
        
        // Get one logical CPU per physical core for each NUMA node
        std::vector<int> node0_physical_cpus, node1_physical_cpus;
        std::set<std::pair<int,int>> used_cores; // socket,core
        
        for (const auto& cpu : cpu_topology) {
            std::pair<int,int> core_id = {cpu.socket, cpu.physical_core};
            if (used_cores.find(core_id) == used_cores.end()) {
                if (cpu.numa_node == 0) {
                    node0_physical_cpus.push_back(cpu.logical_cpu);
                } else {
                    node1_physical_cpus.push_back(cpu.logical_cpu);
                }
                used_cores.insert(core_id);
            }
        }
        
        std::cout << "   Node 0 physical cores: " << node0_physical_cpus.size() << std::endl;
        std::cout << "   Node 1 physical cores: " << node1_physical_cpus.size() << std::endl;
        
        // Test a few representative CPUs from each node
        for (int i = 0; i < std::min(4, (int)node0_physical_cpus.size()); i++) {
            test_single_cpu_performance(node0_physical_cpus[i], test_size, "Node 0");
        }
        
        for (int i = 0; i < std::min(4, (int)node1_physical_cpus.size()); i++) {
            test_single_cpu_performance(node1_physical_cpus[i], test_size, "Node 1");
        }
    }
    
    void test_hyperthreading_effect(size_t test_size) {
        std::cout << "\n🔸 Hyperthreading Effect Test:" << std::endl;
        
        // Find hyperthreading pairs
        std::map<std::pair<int,int>, std::vector<int>> core_map;
        for (const auto& cpu : cpu_topology) {
            core_map[{cpu.socket, cpu.physical_core}].push_back(cpu.logical_cpu);
        }
        
        // Test a few HT pairs from each node
        int pairs_tested = 0;
        for (const auto& entry : core_map) {
            if (pairs_tested >= 4) break;
            
            const auto& cpus = entry.second;
            if (cpus.size() == 2) { // Hyperthreading pair
                int cpu1 = cpus[0], cpu2 = cpus[1];
                int node1 = numa_node_of_cpu(cpu1);
                int node2 = numa_node_of_cpu(cpu2);
                
                if (node1 == node2) { // Same NUMA node
                    std::cout << "   HT pair CPUs " << cpu1 << "," << cpu2 
                              << " (Node " << node1 << "):" << std::endl;
                    
                    test_single_cpu_performance(cpu1, test_size / 2, "  Primary  ");
                    test_single_cpu_performance(cpu2, test_size / 2, "  Secondary");
                    pairs_tested++;
                }
            }
        }
    }
    
    void test_socket_performance(size_t test_size) {
        std::cout << "\n🔸 Socket Performance Comparison:" << std::endl;
        
        // Test representative CPUs from each socket
        std::vector<int> socket0_cpus, socket1_cpus;
        
        for (const auto& cpu : cpu_topology) {
            if (cpu.socket == 0 && socket0_cpus.size() < 4) {
                socket0_cpus.push_back(cpu.logical_cpu);
            } else if (cpu.socket == 1 && socket1_cpus.size() < 4) {
                socket1_cpus.push_back(cpu.logical_cpu);
            }
        }
        
        std::cout << "   Socket 0 CPUs:" << std::endl;
        for (int cpu : socket0_cpus) {
            test_single_cpu_performance(cpu, test_size, "  Socket 0");
        }
        
        std::cout << "   Socket 1 CPUs:" << std::endl;
        for (int cpu : socket1_cpus) {
            test_single_cpu_performance(cpu, test_size, "  Socket 1");
        }
    }
    
    void test_memory_controller_effects(size_t test_size) {
        std::cout << "\n🔸 Memory Controller Effects:" << std::endl;
        
        // Test same CPU accessing different NUMA node memories
        int test_cpu = 28; // Node 1 CPU
        
        for (int mem_node = 0; mem_node < 2; mem_node++) {
            void* memory = numa_alloc_onnode(test_size, mem_node);
            if (!memory) continue;
            
            memset(memory, 0x42, test_size);
            
            // Bind to specific CPU
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(test_cpu, &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
            
            auto start = std::chrono::high_resolution_clock::now();
            
            // Memory bandwidth test
            volatile float* data = (volatile float*)memory;
            size_t elements = test_size / sizeof(float);
            float sum = 0;
            
            for (size_t i = 0; i < elements; i++) {
                sum += data[i];
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            double time_ms = duration.count() / 1000.0;
            double bandwidth = (test_size / 1024.0 / 1024.0 / 1024.0) / (time_ms / 1000.0);
            
            std::cout << "   CPU " << test_cpu << " → Node " << mem_node 
                      << " memory: " << std::fixed << std::setprecision(1) << time_ms << " ms"
                      << " (" << std::setprecision(2) << bandwidth << " GB/s)" << std::endl;
            
            numa_free(memory, test_size);
        }
    }
    
    void test_single_cpu_performance(int cpu, size_t test_size, const std::string& label) {
#ifdef __linux__
        void* memory = numa_alloc_local(test_size);
        if (!memory) return;
        
        memset(memory, 0x42, test_size);
        
        // Bind to specific CPU
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Simple ADD-like operation
        float* data = (float*)memory;
        size_t elements = test_size / sizeof(float);
        
        for (size_t i = 0; i < elements; i++) {
            data[i] = data[i] + 1.0f;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double time_ms = duration.count() / 1000.0;
        double bandwidth = (2.0 * test_size / 1024.0 / 1024.0 / 1024.0) / (time_ms / 1000.0);
        
        int actual_cpu = sched_getcpu();
        int numa_node = numa_node_of_cpu(actual_cpu);
        
        std::cout << "   " << label << " CPU " << std::setw(3) << cpu 
                  << " (actual: " << std::setw(3) << actual_cpu << ", node " << numa_node << "): "
                  << std::fixed << std::setprecision(1) << time_ms << " ms"
                  << " (" << std::setprecision(2) << bandwidth << " GB/s)" << std::endl;
        
        numa_free(memory, test_size);
#endif
    }
};

int main() {
    try {
        IntelXeonNumaAnalyzer analyzer;
        analyzer.test_cpu_performance_patterns();
        
        std::cout << "\n🏁 Intel Xeon NUMA analysis complete!" << std::endl;
        std::cout << "Look for systematic performance differences between nodes/sockets." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
