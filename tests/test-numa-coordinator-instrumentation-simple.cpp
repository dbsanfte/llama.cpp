/**
 * NUMA Coordinator Instrumentation Tests
 * 
 * Simplified version focusing on key hotspot detection:
 * - Basic mutex contention monitoring
 * - Thread activity tracking
 * - Memory usage analysis
 */

#include <chrono>
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <iomanip>
#include <unistd.h>
#include <mutex>
#include <cmath>
#include <random>
#include <cassert>

#include "ggml.h"
#include "ggml-cpu.h" 
#include "ggml-numa-coordinator.h"
#include "common.h"

class NumaCoordinatorInstrumentation {
public:
    struct MutexStats {
        std::atomic<uint64_t> lock_count{0};
        std::atomic<uint64_t> wait_time_ns{0};
        std::atomic<uint64_t> contention_count{0};
        
        MutexStats() = default;
        MutexStats(const MutexStats& other) :
            lock_count{other.lock_count.load()},
            wait_time_ns{other.wait_time_ns.load()},
            contention_count{other.contention_count.load()} {}
    };

    struct ThreadStats {
        std::atomic<uint64_t> operations_completed{0};
        std::atomic<uint64_t> active_time_ns{0};
        std::atomic<uint64_t> idle_time_ns{0};
        
        ThreadStats() = default;
        ThreadStats(const ThreadStats& other) :
            operations_completed{other.operations_completed.load()},
            active_time_ns{other.active_time_ns.load()},
            idle_time_ns{other.idle_time_ns.load()} {}
    };

    struct MemoryStats {
        std::atomic<size_t> peak_usage{0};
        std::atomic<size_t> current_usage{0};
        std::atomic<size_t> allocation_count{0};
    };

private:
    static std::vector<MutexStats> mutex_stats;
    static std::vector<ThreadStats> thread_stats;
    static MemoryStats memory_stats;
    static std::atomic<bool> instrumentation_enabled;
    static std::mutex stats_mutex;

public:
    // Initialize instrumentation
    static void initialize_instrumentation(int num_threads, int num_mutexes) {
        std::lock_guard<std::mutex> lock(stats_mutex);
        
        // Initialize stats vectors
        mutex_stats.clear();
        thread_stats.clear();
        
        // Pre-allocate with proper initialization
        for (int i = 0; i < num_mutexes; i++) {
            mutex_stats.emplace_back();
        }
        
        for (int i = 0; i < num_threads; i++) {
            thread_stats.emplace_back();
        }
        
        // Reset memory stats
        memory_stats.peak_usage = 0;
        memory_stats.current_usage = 0;
        memory_stats.allocation_count = 0;
        
        instrumentation_enabled = true;
        
        std::cout << "📊 Instrumentation initialized:" << std::endl;
        std::cout << "   Threads: " << num_threads << std::endl;
        std::cout << "   Mutexes: " << num_mutexes << std::endl;
    }

    // Record mutex lock event
    static void record_mutex_lock(int mutex_id, uint64_t wait_time_ns, bool contended) {
        if (!instrumentation_enabled || mutex_id >= (int)mutex_stats.size()) return;
        
        mutex_stats[mutex_id].lock_count.fetch_add(1);
        mutex_stats[mutex_id].wait_time_ns.fetch_add(wait_time_ns);
        if (contended) {
            mutex_stats[mutex_id].contention_count.fetch_add(1);
        }
    }

    // Record thread activity
    static void record_thread_activity(int thread_id, uint64_t active_time_ns, bool completed_operation) {
        if (!instrumentation_enabled || thread_id >= (int)thread_stats.size()) return;
        
        thread_stats[thread_id].active_time_ns.fetch_add(active_time_ns);
        if (completed_operation) {
            thread_stats[thread_id].operations_completed.fetch_add(1);
        }
    }

    // Record memory allocation
    static void record_memory_allocation(size_t size) {
        if (!instrumentation_enabled) return;
        
        memory_stats.allocation_count.fetch_add(1);
        memory_stats.current_usage.fetch_add(size);
        
        // Update peak usage
        size_t current = memory_stats.current_usage.load();
        size_t peak = memory_stats.peak_usage.load();
        while (current > peak && !memory_stats.peak_usage.compare_exchange_weak(peak, current)) {
            peak = memory_stats.peak_usage.load();
        }
    }

    // Simulate instrumented operation
    void run_instrumented_test(int numa_nodes, int threads_per_node, int duration_ms) {
        std::cout << "🔍 Running Instrumented Test" << std::endl;
        std::cout << "   NUMA nodes: " << numa_nodes << std::endl;
        std::cout << "   Threads per node: " << threads_per_node << std::endl;
        std::cout << "   Duration: " << duration_ms << " ms" << std::endl;

        int total_threads = numa_nodes * threads_per_node;
        int num_mutexes = numa_nodes; // One mutex per NUMA node
        
        initialize_instrumentation(total_threads, num_mutexes);
        
        // Simulate coordinated execution
        std::vector<std::thread> threads;
        std::atomic<bool> stop_flag{false};
        
        // Launch worker threads
        for (int i = 0; i < total_threads; i++) {
            threads.emplace_back([this, i, numa_nodes, &stop_flag]() {
                int numa_node = i % numa_nodes;
                simulate_worker_thread(i, numa_node, stop_flag);
            });
        }
        
        // Run for specified duration
        std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
        stop_flag = true;
        
        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }
        
        // Analyze and report results
        analyze_instrumentation_results();
    }

private:
    void simulate_worker_thread(int thread_id, int numa_node, std::atomic<bool>& stop_flag) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> work_dist(1, 10); // Simulate 1-10ms work
        std::uniform_int_distribution<> mutex_dist(0, 9); // 10% chance of mutex contention
        
        while (!stop_flag) {
            auto start = std::chrono::high_resolution_clock::now();
            
            // Simulate mutex lock with potential contention
            bool contended = mutex_dist(gen) == 0;
            uint64_t wait_time = contended ? 1000000 : 100000; // 1ms vs 0.1ms wait
            record_mutex_lock(numa_node, wait_time, contended);
            
            // Simulate work
            int work_time = work_dist(gen);
            std::this_thread::sleep_for(std::chrono::milliseconds(work_time));
            
            auto end = std::chrono::high_resolution_clock::now();
            uint64_t active_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            
            // Record thread activity
            record_thread_activity(thread_id, active_time, true);
            
            // Simulate memory allocation
            record_memory_allocation(1024); // 1KB allocation
        }
    }

    void analyze_instrumentation_results() {
        std::cout << "\n📊 Instrumentation Analysis Results" << std::endl;
        std::cout << "====================================" << std::endl;
        
        // Mutex contention analysis
        std::cout << "\n🔒 Mutex Contention Analysis:" << std::endl;
        for (size_t i = 0; i < mutex_stats.size(); i++) {
            uint64_t locks = mutex_stats[i].lock_count.load();
            uint64_t contentions = mutex_stats[i].contention_count.load();
            uint64_t wait_time = mutex_stats[i].wait_time_ns.load();
            
            if (locks > 0) {
                double contention_rate = (double)contentions / locks * 100.0;
                double avg_wait_time = (double)wait_time / locks / 1000000.0; // Convert to ms
                
                std::cout << "   Mutex " << i << ":" << std::endl;
                std::cout << "     Locks: " << locks << std::endl;
                std::cout << "     Contentions: " << contentions << " (" << std::fixed << std::setprecision(1) 
                          << contention_rate << "%)" << std::endl;
                std::cout << "     Avg wait time: " << std::fixed << std::setprecision(3) 
                          << avg_wait_time << " ms" << std::endl;
                
                if (contention_rate > 20.0) {
                    std::cout << "     ⚠️  HIGH CONTENTION DETECTED!" << std::endl;
                }
            }
        }
        
        // Thread activity analysis
        std::cout << "\n🧵 Thread Activity Analysis:" << std::endl;
        uint64_t total_operations = 0;
        uint64_t total_active_time = 0;
        
        for (size_t i = 0; i < thread_stats.size(); i++) {
            uint64_t ops = thread_stats[i].operations_completed.load();
            uint64_t active = thread_stats[i].active_time_ns.load();
            
            total_operations += ops;
            total_active_time += active;
            
            if (i < 3 || ops == 0) { // Show first 3 threads and any idle threads
                std::cout << "   Thread " << i << ": " << ops << " operations, " 
                          << std::fixed << std::setprecision(2) << active / 1000000.0 << " ms active" << std::endl;
            }
        }
        
        std::cout << "   Total operations: " << total_operations << std::endl;
        std::cout << "   Avg operations per thread: " << std::fixed << std::setprecision(1) 
                  << (double)total_operations / thread_stats.size() << std::endl;
        
        // Memory usage analysis
        std::cout << "\n💾 Memory Usage Analysis:" << std::endl;
        std::cout << "   Peak usage: " << memory_stats.peak_usage.load() / 1024 << " KB" << std::endl;
        std::cout << "   Current usage: " << memory_stats.current_usage.load() / 1024 << " KB" << std::endl;
        std::cout << "   Allocations: " << memory_stats.allocation_count.load() << std::endl;
        
        // Hotspot detection
        detect_hotspots();
    }

    void detect_hotspots() {
        std::cout << "\n🔥 Hotspot Detection:" << std::endl;
        
        // Check for mutex hotspots
        for (size_t i = 0; i < mutex_stats.size(); i++) {
            uint64_t locks = mutex_stats[i].lock_count.load();
            uint64_t contentions = mutex_stats[i].contention_count.load();
            
            if (locks > 0) {
                double contention_rate = (double)contentions / locks * 100.0;
                if (contention_rate > 15.0) {
                    std::cout << "   🔒 Mutex " << i << " hotspot: " << std::fixed << std::setprecision(1) 
                              << contention_rate << "% contention rate" << std::endl;
                }
            }
        }
        
        // Check for thread imbalance
        if (!thread_stats.empty()) {
            uint64_t min_ops = UINT64_MAX, max_ops = 0;
            for (const auto& stats : thread_stats) {
                uint64_t ops = stats.operations_completed.load();
                min_ops = std::min(min_ops, ops);
                max_ops = std::max(max_ops, ops);
            }
            
            if (max_ops > 0 && min_ops < max_ops * 0.7) {
                double imbalance = (1.0 - (double)min_ops / max_ops) * 100.0;
                std::cout << "   ⚖️  Thread imbalance detected: " << std::fixed << std::setprecision(1) 
                          << imbalance << "% variation" << std::endl;
            }
        }
        
        // Check memory usage
        size_t peak_mb = memory_stats.peak_usage.load() / (1024 * 1024);
        if (peak_mb > 100) {
            std::cout << "   💾 High memory usage: " << peak_mb << " MB peak" << std::endl;
        }
    }

public:
    void run_comprehensive_instrumentation() {
        std::cout << "🔬 NUMA Coordinator Comprehensive Instrumentation" << std::endl;
        std::cout << "==================================================" << std::endl;
        
        int cpu_count = cpu_get_num_math();
        std::cout << "System CPU count: " << cpu_count << std::endl;
        
        // Test different NUMA configurations
        std::vector<std::pair<int, int>> configs = {
            {1, 4},  // 1 NUMA node, 4 threads
            {2, 2},  // 2 NUMA nodes, 2 threads each  
            {4, 1}   // 4 NUMA nodes, 1 thread each
        };
        
        for (auto config : configs) {
            std::cout << "\n" << std::string(50, '=') << std::endl;
            run_instrumented_test(config.first, config.second, 1000); // 1 second test
        }
        
        std::cout << "\n✅ Comprehensive instrumentation completed!" << std::endl;
    }
};

// Static member definitions
std::vector<NumaCoordinatorInstrumentation::MutexStats> NumaCoordinatorInstrumentation::mutex_stats;
std::vector<NumaCoordinatorInstrumentation::ThreadStats> NumaCoordinatorInstrumentation::thread_stats;
NumaCoordinatorInstrumentation::MemoryStats NumaCoordinatorInstrumentation::memory_stats;
std::atomic<bool> NumaCoordinatorInstrumentation::instrumentation_enabled{false};
std::mutex NumaCoordinatorInstrumentation::stats_mutex;

int main() {
    try {
        NumaCoordinatorInstrumentation tester;
        tester.run_comprehensive_instrumentation();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}
