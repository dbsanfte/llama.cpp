/**
 * Physical Core Only NUMA Performance Test
 * 
 * This test strictly uses physical cores only (no hyperthreading) with
 * forced CPU binding to eliminate scheduler interference and HT effects.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <map>
#include <set>
#include <fstream>
#include <cmath>
#include <sstream>

// NUMA headers
#ifdef __linux__
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#include <unistd.h>
#endif

class PhysicalCoreNumaTest {
private:
    struct PhysicalCore {
        int logical_cpu;      // Primary logical CPU for this physical core
        int socket;          // Which socket (0 or 1)
        int numa_node;       // Which NUMA node (0 or 1)
        int physical_core_id;// Physical core ID within socket
    };
    
    struct TestResult {
        int cpu_node, mem_node;
        double avg_time_ms, avg_bandwidth_gbps;
        std::vector<double> individual_bandwidths;
    };
    
    std::vector<PhysicalCore> physical_cores;
    
public:
    PhysicalCoreNumaTest() {
        std::cout << "🔧 Physical Core Only NUMA Performance Test" << std::endl;
        std::cout << "===========================================" << std::endl;
        
        discover_physical_cores();
        print_physical_core_layout();
    }
    
    /**
     * Discover physical cores (avoiding hyperthreading siblings)
     */
    void discover_physical_cores() {
        std::cout << "📋 Discovering physical cores..." << std::endl;
        
        std::map<std::pair<int,int>, int> socket_core_to_logical; // (socket,core) -> logical_cpu
        
        // Parse /proc/cpuinfo to find physical core mapping
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::string line;
        
        int current_logical = -1, current_socket = -1, current_core = -1;
        
        while (std::getline(cpuinfo, line)) {
            if (line.find("processor") == 0) {
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    current_logical = std::stoi(line.substr(colon + 1));
                }
            } else if (line.find("physical id") == 0) {
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    current_socket = std::stoi(line.substr(colon + 1));
                }
            } else if (line.find("core id") == 0) {
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    current_core = std::stoi(line.substr(colon + 1));
                    
                    // We have all info for this logical CPU
                    std::pair<int,int> socket_core = {current_socket, current_core};
                    
                    // Only store the first logical CPU for each physical core
                    if (socket_core_to_logical.find(socket_core) == socket_core_to_logical.end()) {
                        socket_core_to_logical[socket_core] = current_logical;
                        
                        PhysicalCore pc;
                        pc.logical_cpu = current_logical;
                        pc.socket = current_socket;
                        pc.physical_core_id = current_core;
                        
                        // Determine NUMA node from logical CPU
                        int cpu = current_logical;
                        if ((cpu >= 0 && cpu <= 27) || (cpu >= 56 && cpu <= 83)) {
                            pc.numa_node = 0;
                        } else {
                            pc.numa_node = 1;
                        }
                        
                        physical_cores.push_back(pc);
                    }
                }
            }
        }
        
        std::cout << "   Found " << physical_cores.size() << " physical cores" << std::endl;
    }
    
    /**
     * Print physical core layout
     */
    void print_physical_core_layout() {
        std::cout << "\n📊 Physical Core Layout:" << std::endl;
        std::cout << "========================" << std::endl;
        
        std::map<int, std::vector<PhysicalCore>> cores_by_node;
        for (const auto& core : physical_cores) {
            cores_by_node[core.numa_node].push_back(core);
        }
        
        for (const auto& entry : cores_by_node) {
            int node = entry.first;
            const auto& cores = entry.second;
            
            std::cout << "🔸 NUMA Node " << node << " (" << cores.size() << " physical cores):" << std::endl;
            std::cout << "   CPUs: ";
            
            for (size_t i = 0; i < cores.size() && i < 10; i++) {
                std::cout << cores[i].logical_cpu;
                if (i < cores.size() - 1 && i < 9) std::cout << ", ";
            }
            if (cores.size() > 10) std::cout << "...";
            std::cout << std::endl;
        }
    }
    
    /**
     * Run comprehensive physical core performance test
     */
    void run_physical_core_performance_test() {
        std::cout << "\n🚀 Physical Core Performance Matrix" << std::endl;
        std::cout << "===================================" << std::endl;
        
        size_t test_size = 64 * 1024 * 1024; // 64 MB
        int iterations = 3;
        
        // Get representative physical cores from each NUMA node
        std::vector<PhysicalCore> node0_cores, node1_cores;
        
        for (const auto& core : physical_cores) {
            if (core.numa_node == 0 && node0_cores.size() < 4) {
                node0_cores.push_back(core);
            } else if (core.numa_node == 1 && node1_cores.size() < 4) {
                node1_cores.push_back(core);
            }
        }
        
        std::cout << "Testing with " << node0_cores.size() << " cores from Node 0 and " 
                  << node1_cores.size() << " cores from Node 1" << std::endl;
        
        // Test matrix: CPU node × Memory node
        
        std::vector<TestResult> results;
        
        for (int cpu_node = 0; cpu_node < 2; cpu_node++) {
            const auto& test_cores = (cpu_node == 0) ? node0_cores : node1_cores;
            
            for (int mem_node = 0; mem_node < 2; mem_node++) {
                std::cout << "\n🔸 CPU Node " << cpu_node << " → Memory Node " << mem_node << ":" << std::endl;
                
                TestResult result;
                result.cpu_node = cpu_node;
                result.mem_node = mem_node;
                
                double total_time = 0;
                double total_bandwidth = 0;
                int valid_tests = 0;
                
                for (const auto& core : test_cores) {
                    std::cout << "   Testing physical core " << core.logical_cpu 
                              << " (socket " << core.socket << ")..." << std::endl;
                    
                    double core_bandwidth = test_single_physical_core(
                        core.logical_cpu, mem_node, test_size, iterations);
                    
                    if (core_bandwidth > 0) {
                        result.individual_bandwidths.push_back(core_bandwidth);
                        total_bandwidth += core_bandwidth;
                        valid_tests++;
                        
                        std::cout << "     → " << std::fixed << std::setprecision(2) 
                                  << core_bandwidth << " GB/s" << std::endl;
                    }
                }
                
                if (valid_tests > 0) {
                    result.avg_bandwidth_gbps = total_bandwidth / valid_tests;
                    result.avg_time_ms = 0; // Calculate if needed
                    
                    std::cout << "   📊 Average: " << std::fixed << std::setprecision(2) 
                              << result.avg_bandwidth_gbps << " GB/s" << std::endl;
                    
                    // Calculate standard deviation
                    double variance = 0;
                    for (double bw : result.individual_bandwidths) {
                        variance += (bw - result.avg_bandwidth_gbps) * (bw - result.avg_bandwidth_gbps);
                    }
                    variance /= valid_tests;
                    double stddev = sqrt(variance);
                    
                    std::cout << "   📈 Std Dev: ±" << std::fixed << std::setprecision(2) 
                              << stddev << " GB/s" << std::endl;
                    
                    results.push_back(result);
                }
            }
        }
        
        // Print summary matrix
        print_performance_matrix(results);
    }
    
private:
    /**
     * Test a single physical core with strict CPU binding
     */
    double test_single_physical_core(int logical_cpu, int mem_node, size_t test_size, int iterations) {
#ifdef __linux__
        // Strict CPU binding
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(logical_cpu, &cpuset);
        
        if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) != 0) {
            std::cerr << "     ❌ Failed to bind to CPU " << logical_cpu << std::endl;
            return 0;
        }
        
        // Verify binding
        int actual_cpu = sched_getcpu();
        if (actual_cpu != logical_cpu) {
            std::cerr << "     ❌ CPU binding failed: requested " << logical_cpu 
                      << " got " << actual_cpu << std::endl;
            return 0;
        }
        
        // Allocate memory on specific NUMA node
        std::cout << "     🔧 Allocating " << (test_size / (1024*1024)) << " MB on node " << mem_node << "...";
        void* memory = numa_alloc_onnode(test_size, mem_node);
        if (!memory) {
            std::cerr << " ❌ Failed!" << std::endl;
            std::cerr << "     ❌ Failed to allocate memory on node " << mem_node << std::endl;
            return 0;
        }
        std::cout << " ✓" << std::endl;
        
        // Initialize memory to force page allocation
        std::cout << "     🔧 Initializing memory...";
        memset(memory, 0x42, test_size);
        std::cout << " ✓" << std::endl;
        
        // Initialize memory to force page allocation
        std::cout << "     🔧 Initializing memory...";
        memset(memory, 0x42, test_size);
        std::cout << " ✓" << std::endl;
        
        // Verify memory placement
        std::vector<void*> pages;
        std::vector<int> status;
        size_t page_size = getpagesize();
        size_t num_pages = (test_size + page_size - 1) / page_size;
        
        for (size_t i = 0; i < num_pages; i += 1000) { // Sample every 1000th page
            pages.push_back((char*)memory + i * page_size);
        }
        status.resize(pages.size());
        
        if (move_pages(0, pages.size(), pages.data(), nullptr, status.data(), 0) == 0) {
            int correct_placement = 0;
            for (int stat : status) {
                if (stat >= 0 && stat == mem_node) correct_placement++;
            }
            double locality_ratio = (double)correct_placement / pages.size();
            if (locality_ratio < 0.9) {
                std::cerr << "     ⚠️  Poor memory locality: " << std::fixed 
                          << std::setprecision(1) << (locality_ratio * 100) << "%" 
                          << " (wanted node " << mem_node << ", got ";
                for (size_t i = 0; i < std::min(pages.size(), (size_t)5); ++i) {
                    std::cerr << status[i] << " ";
                }
                std::cerr << "...)" << std::endl;
            }
        }
        
        // Initialize memory
        memset(memory, 0x42, test_size);
        
        // Run performance test multiple times
        std::vector<double> bandwidths;
        
        for (int iter = 0; iter < iterations; iter++) {
            // Verify we're still on the right CPU
            actual_cpu = sched_getcpu();
            if (actual_cpu != logical_cpu) {
                std::cerr << "     ⚠️  CPU migrated during test: " << actual_cpu << std::endl;
                // Re-bind
                pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
            }
            
            auto start = std::chrono::high_resolution_clock::now();
            
            // Memory-intensive ADD-like operation
            float* data = (float*)memory;
            size_t elements = test_size / sizeof(float);
            
            // Unrolled loop for better performance measurement
            for (size_t i = 0; i < elements; i += 4) {
                data[i] = data[i] + 1.0f;
                data[i+1] = data[i+1] + 1.0f;
                data[i+2] = data[i+2] + 1.0f;
                data[i+3] = data[i+3] + 1.0f;
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            double time_ms = duration.count() / 1000.0;
            
            // Calculate bandwidth (read + write)
            double bandwidth = (2.0 * test_size / 1024.0 / 1024.0 / 1024.0) / (time_ms / 1000.0);
            bandwidths.push_back(bandwidth);
        }
        
        numa_free(memory, test_size);
        
        // Return average bandwidth
        double total_bandwidth = 0;
        for (double bw : bandwidths) {
            total_bandwidth += bw;
        }
        return total_bandwidth / bandwidths.size();
#else
        return 0;
#endif
    }
    
    /**
     * Print performance matrix summary
     */
    void print_performance_matrix(const std::vector<TestResult>& results) {
        std::cout << "\n📋 Physical Core Performance Matrix Summary" << std::endl;
        std::cout << "===========================================" << std::endl;
        std::cout << "CPU Node → Memory Node | Avg Bandwidth | Pattern" << std::endl;
        std::cout << "----------------------|----------------|--------" << std::endl;
        
        for (const auto& result : results) {
            std::string pattern = (result.cpu_node == result.mem_node) ? "LOCAL " : "REMOTE";
            std::cout << "   Node " << result.cpu_node << " → Node " << result.mem_node 
                      << "        | " << std::setw(10) << std::fixed << std::setprecision(2) 
                      << result.avg_bandwidth_gbps << " GB/s | " << pattern << std::endl;
        }
        
        std::cout << "\n🔍 Analysis:" << std::endl;
        
        // Find best and worst performers
        double best_bandwidth = 0, worst_bandwidth = 1000;
        std::string best_pattern, worst_pattern;
        
        for (const auto& result : results) {
            if (result.avg_bandwidth_gbps > best_bandwidth) {
                best_bandwidth = result.avg_bandwidth_gbps;
                best_pattern = "Node " + std::to_string(result.cpu_node) + 
                              " → Node " + std::to_string(result.mem_node);
            }
            if (result.avg_bandwidth_gbps < worst_bandwidth) {
                worst_bandwidth = result.avg_bandwidth_gbps;
                worst_pattern = "Node " + std::to_string(result.cpu_node) + 
                               " → Node " + std::to_string(result.mem_node);
            }
        }
        
        std::cout << "   🏆 Best performance:  " << best_pattern << " (" 
                  << std::fixed << std::setprecision(2) << best_bandwidth << " GB/s)" << std::endl;
        std::cout << "   🐌 Worst performance: " << worst_pattern << " (" 
                  << std::fixed << std::setprecision(2) << worst_bandwidth << " GB/s)" << std::endl;
        
        double performance_ratio = best_bandwidth / worst_bandwidth;
        std::cout << "   📊 Performance ratio: " << std::fixed << std::setprecision(2) 
                  << performance_ratio << "x" << std::endl;
        
        if (performance_ratio > 2.0) {
            std::cout << "   ⚠️  SIGNIFICANT performance asymmetry detected!" << std::endl;
            std::cout << "       This could explain ADD kernel performance issues." << std::endl;
        }
    }
};

int main() {
    try {
        PhysicalCoreNumaTest tester;
        tester.run_physical_core_performance_test();
        
        std::cout << "\n🏁 Physical core NUMA test complete!" << std::endl;
        std::cout << "This test eliminates hyperthreading and scheduler interference." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
