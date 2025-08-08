/**
 * NUMA Coordinator Performance Instrumentation Test
 * 
 * Instruments the coordinator to detect performance hotspots:
 * - Mutex contention analysis
 * - Thread wait time vs busy time
 * - Queue depth monitoring
 * - Memory allocation patterns
 * - CPU utilization per NUMA node
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
#include <map>
#include <algorithm>

#include "ggml.h"
#include "ggml-cpu.h" 
#include "ggml-numa-coordinator.h"
#include "common.h"

class NumaCoordinatorInstrumentation {
private:
    // Instrumentation data structures
    struct MutexStats {
        std::atomic<uint64_t> lock_attempts{0};
        std::atomic<uint64_t> lock_acquisitions{0};
        std::atomic<uint64_t> total_wait_time_us{0};
        std::atomic<uint64_t> total_hold_time_us{0};
        std::atomic<uint64_t> max_wait_time_us{0};
        std::atomic<uint64_t> max_hold_time_us{0};
        std::string name;
    };

    struct ThreadStats {
        int thread_id;
        int numa_node;
        std::atomic<uint64_t> busy_time_us{0};
        std::atomic<uint64_t> wait_time_us{0};
        std::atomic<uint64_t> total_operations{0};
        std::atomic<uint64_t> queue_waits{0};
        std::atomic<uint64_t> condition_waits{0};
        std::chrono::high_resolution_clock::time_point start_time;
    };

    struct QueueStats {
        int numa_node;
        std::atomic<uint64_t> max_depth{0};
        std::atomic<uint64_t> total_enqueues{0};
        std::atomic<uint64_t> total_dequeues{0};
        std::atomic<uint64_t> total_wait_time_us{0};
        std::atomic<uint64_t> current_depth{0};
    };

    struct MemoryStats {
        std::atomic<uint64_t> total_allocations{0};
        std::atomic<uint64_t> total_bytes_allocated{0};
        std::atomic<uint64_t> peak_memory_usage{0};
        std::atomic<uint64_t> current_memory_usage{0};
    };

    // Global instrumentation state
    static std::vector<MutexStats> mutex_stats;
    static std::vector<ThreadStats> thread_stats;
    static std::vector<QueueStats> queue_stats;
    static MemoryStats memory_stats;
    static std::atomic<bool> instrumentation_enabled;
    static std::mutex stats_mutex;

    // Timing helpers
    static uint64_t get_time_us() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }

public:
    // Initialize instrumentation
    static void initialize_instrumentation(int numa_nodes, int threads_per_node) {
        std::lock_guard<std::mutex> lock(stats_mutex);
        
        // Initialize mutex stats for common mutex types
        mutex_stats.clear();
        mutex_stats.push_back({0, 0, 0, 0, 0, 0, "main_sync_mutex"});
        mutex_stats.push_back({0, 0, 0, 0, 0, 0, "queue_mutex"});
        mutex_stats.push_back({0, 0, 0, 0, 0, 0, "completion_mutex"});
        mutex_stats.push_back({0, 0, 0, 0, 0, 0, "groups_mutex"});
        
        // Initialize thread stats
        thread_stats.clear();
        for (int node = 0; node < numa_nodes; node++) {
            for (int thread = 0; thread < threads_per_node; thread++) {
                ThreadStats stats;
                stats.thread_id = node * threads_per_node + thread;
                stats.numa_node = node;
                stats.start_time = std::chrono::high_resolution_clock::now();
                thread_stats.push_back(std::move(stats));
            }
        }
        
        // Initialize queue stats
        queue_stats.clear();
        for (int node = 0; node < numa_nodes; node++) {
            QueueStats stats;
            stats.numa_node = node;
            queue_stats.push_back(std::move(stats));
        }
        
        // Reset memory stats
        memory_stats = MemoryStats{};
        
        instrumentation_enabled = true;
        
        std::cout << "🔍 Instrumentation initialized for " << numa_nodes 
                 << " NUMA nodes, " << threads_per_node << " threads each" << std::endl;
    }

    // Instrumented mutex class
    class InstrumentedMutex {
    private:
        std::mutex actual_mutex;
        int stats_index;
        
    public:
        InstrumentedMutex(int index) : stats_index(index) {}
        
        void lock() {
            if (!instrumentation_enabled || stats_index >= (int)mutex_stats.size()) {
                actual_mutex.lock();
                return;
            }
            
            auto& stats = mutex_stats[stats_index];
            stats.lock_attempts++;
            
            uint64_t wait_start = get_time_us();
            actual_mutex.lock();
            uint64_t wait_end = get_time_us();
            
            uint64_t wait_time = wait_end - wait_start;
            stats.lock_acquisitions++;
            stats.total_wait_time_us += wait_time;
            
            // Update max wait time
            uint64_t current_max = stats.max_wait_time_us.load();
            while (wait_time > current_max && 
                   !stats.max_wait_time_us.compare_exchange_weak(current_max, wait_time)) {
                // Retry until successful or wait_time is no longer max
            }
        }
        
        void unlock() {
            actual_mutex.unlock();
        }
        
        bool try_lock() {
            if (!instrumentation_enabled || stats_index >= (int)mutex_stats.size()) {
                return actual_mutex.try_lock();
            }
            
            auto& stats = mutex_stats[stats_index];
            stats.lock_attempts++;
            
            if (actual_mutex.try_lock()) {
                stats.lock_acquisitions++;
                return true;
            }
            return false;
        }
    };

    // Progress callback that records thread activity
    static void instrumented_progress_callback(int work_id, int numa_node, struct ggml_tensor * tensor, void * user_data) {
        if (!instrumentation_enabled || numa_node >= (int)thread_stats.size()) return;
        
        // Find the thread stats for this NUMA node (simplified)
        for (auto& stats : thread_stats) {
            if (stats.numa_node == numa_node) {
                stats.total_operations++;
                break;
            }
        }
        
        // Update queue stats
        if (numa_node >= 0 && numa_node < (int)queue_stats.size()) {
            queue_stats[numa_node].total_dequeues++;
            queue_stats[numa_node].current_depth--;
        }
    }

    // Create instrumented test workload
    struct ggml_cgraph * create_instrumented_test_graph(struct ggml_context * ctx, 
                                                        int complexity_level) {
        std::cout << "    Creating instrumented test graph (complexity " << complexity_level << ")" << std::endl;
        
        // Record memory allocation
        memory_stats.total_allocations++;
        
        int64_t base_size = 10000 * complexity_level;
        int num_operations = 5 * complexity_level;
        
        // Create tensors with size based on complexity
        std::vector<struct ggml_tensor*> tensors;
        for (int i = 0; i < 4; i++) {
            struct ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, base_size);
            
            // Record memory usage
            uint64_t tensor_bytes = base_size * sizeof(float);
            memory_stats.total_bytes_allocated += tensor_bytes;
            memory_stats.current_memory_usage += tensor_bytes;
            
            // Update peak memory if needed
            uint64_t current_peak = memory_stats.peak_memory_usage.load();
            uint64_t current_usage = memory_stats.current_memory_usage.load();
            while (current_usage > current_peak && 
                   !memory_stats.peak_memory_usage.compare_exchange_weak(current_peak, current_usage)) {
                // Retry
            }
            
            // Fill with test data
            float * data = (float*)t->data;
            for (int64_t j = 0; j < base_size; j++) {
                data[j] = 1.0f + (float)(j % 100) * 0.001f;
            }
            
            tensors.push_back(t);
        }
        
        // Create operation chain
        std::vector<struct ggml_tensor*> results;
        results.push_back(tensors[0]);
        
        for (int op = 0; op < num_operations; op++) {
            struct ggml_tensor * prev = results.back();
            struct ggml_tensor * operand = tensors[op % 4];
            struct ggml_tensor * result;
            
            // Vary operations to test different code paths
            switch (op % 8) {
                case 0: result = ggml_add(ctx, prev, operand); break;
                case 1: result = ggml_mul(ctx, prev, operand); break;
                case 2: result = ggml_sub(ctx, prev, operand); break;
                case 3: result = ggml_sum(ctx, prev); break;
                case 4: result = ggml_rms_norm(ctx, prev, 1e-6f); break;
                case 5: result = ggml_soft_max(ctx, prev); break;
                case 6: result = ggml_mean(ctx, prev); result = ggml_add(ctx, result, operand); break;
                case 7: result = ggml_scale(ctx, prev, 1.1f); break;
            }
            
            results.push_back(result);
        }
        
        // Build computation graph
        struct ggml_cgraph * cgraph = ggml_new_graph(ctx);
        for (auto* tensor : results) {
            ggml_build_forward_expand(cgraph, tensor);
        }
        
        std::cout << "      Created graph with " << cgraph->n_nodes 
                 << " nodes, " << (base_size * results.size()) << " total elements" << std::endl;
        
        return cgraph;
    }

    // Run instrumented performance test
    void run_instrumented_test(int numa_nodes, int threads_per_node, int complexity) {
        std::cout << "\n🔬 Running Instrumented Test" << std::endl;
        std::cout << "NUMA nodes: " << numa_nodes << ", Threads per node: " << threads_per_node 
                 << ", Complexity: " << complexity << std::endl;
        
        // Initialize instrumentation
        initialize_instrumentation(numa_nodes, threads_per_node);
        
        // Create GGML context
        struct ggml_init_params init_params = {
            .mem_size   = 512 * 1024 * 1024,  // 512MB
            .mem_buffer = NULL,
            .no_alloc   = false,
        };
        
        struct ggml_context * ctx = ggml_init(init_params);
        if (!ctx) {
            std::cout << "❌ Failed to create GGML context" << std::endl;
            return;
        }
        
        // Create test graph
        struct ggml_cgraph * cgraph = create_instrumented_test_graph(ctx, complexity);
        
        // Create coordinator
        int total_threads = numa_nodes * threads_per_node;
        struct ggml_numa_coordinator_manager * mgr = 
            ggml_numa_coordinator_manager_get_global(total_threads, true);
        
        if (!mgr) {
            std::cout << "❌ Failed to create coordinator manager" << std::endl;
            ggml_free(ctx);
            return;
        }
        
        // Set instrumented callback
        ggml_numa_coordinator_manager_set_progress_callback(mgr, instrumented_progress_callback, this);
        
        // Run test multiple times to get good statistics
        const int num_runs = 3;
        std::vector<double> run_times;
        
        for (int run = 0; run < num_runs; run++) {
            std::cout << "  Run " << (run + 1) << "/" << num_runs << "..." << std::endl;
            
            // Reset some counters for this run
            for (auto& stats : queue_stats) {
                stats.total_enqueues = 0;
                stats.total_dequeues = 0;
                stats.current_depth = 0;
            }
            
            auto start = std::chrono::high_resolution_clock::now();
            
            int result = ggml_numa_coordinator_manager_compute_graph(mgr, cgraph);
            if (result != 0) {
                std::cout << "    ❌ Computation failed" << std::endl;
                continue;
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
            run_times.push_back(duration_ms);
            
            std::cout << "    Duration: " << std::fixed << std::setprecision(1) << duration_ms << "ms" << std::endl;
            
            // Brief pause between runs
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        // Analyze results
        analyze_instrumentation_results(run_times);
        
        ggml_free(ctx);
    }

    // Analyze and report instrumentation results
    void analyze_instrumentation_results(const std::vector<double>& run_times) {
        std::cout << "\n📊 Instrumentation Analysis" << std::endl;
        std::cout << "============================" << std::endl;
        
        if (!run_times.empty()) {
            double avg_time = 0.0;
            for (double time : run_times) avg_time += time;
            avg_time /= run_times.size();
            
            double min_time = *std::min_element(run_times.begin(), run_times.end());
            double max_time = *std::max_element(run_times.begin(), run_times.end());
            
            std::cout << "⏱️  Timing Results:" << std::endl;
            std::cout << "   Average: " << std::fixed << std::setprecision(1) << avg_time << "ms" << std::endl;
            std::cout << "   Min: " << min_time << "ms" << std::endl;
            std::cout << "   Max: " << max_time << "ms" << std::endl;
            std::cout << "   Variation: ±" << std::setprecision(1) << ((max_time - min_time) / avg_time * 100.0) << "%" << std::endl;
        }
        
        // Mutex contention analysis
        std::cout << "\n🔒 Mutex Contention Analysis:" << std::endl;
        for (size_t i = 0; i < mutex_stats.size(); i++) {
            const auto& stats = mutex_stats[i];
            uint64_t attempts = stats.lock_attempts.load();
            uint64_t acquisitions = stats.lock_acquisitions.load();
            
            if (attempts > 0) {
                double success_rate = (double)acquisitions / attempts * 100.0;
                double avg_wait_us = attempts > 0 ? (double)stats.total_wait_time_us.load() / attempts : 0.0;
                
                std::cout << "   " << stats.name << ":" << std::endl;
                std::cout << "     Attempts: " << attempts << ", Success rate: " << std::fixed << std::setprecision(1) << success_rate << "%" << std::endl;
                std::cout << "     Avg wait: " << std::setprecision(2) << avg_wait_us << "μs, Max wait: " << stats.max_wait_time_us.load() << "μs" << std::endl;
                
                if (avg_wait_us > 100.0) {
                    std::cout << "     ⚠️  HIGH CONTENTION detected!" << std::endl;
                } else if (avg_wait_us > 10.0) {
                    std::cout << "     ⚠️  Moderate contention" << std::endl;
                } else {
                    std::cout << "     ✅ Low contention" << std::endl;
                }
            }
        }
        
        // Thread activity analysis
        std::cout << "\n🧵 Thread Activity Analysis:" << std::endl;
        std::map<int, std::vector<ThreadStats*>> numa_groups;
        for (auto& stats : thread_stats) {
            numa_groups[stats.numa_node].push_back(&stats);
        }
        
        for (const auto& group : numa_groups) {
            int numa_node = group.first;
            const auto& threads = group.second;
            
            uint64_t total_ops = 0;
            uint64_t total_waits = 0;
            
            for (const auto* thread : threads) {
                total_ops += thread->total_operations.load();
                total_waits += thread->queue_waits.load() + thread->condition_waits.load();
            }
            
            std::cout << "   NUMA " << numa_node << ": " << total_ops << " operations, " << total_waits << " waits" << std::endl;
            
            // Check for load imbalance
            if (threads.size() > 1) {
                uint64_t min_ops = threads[0]->total_operations.load();
                uint64_t max_ops = threads[0]->total_operations.load();
                
                for (const auto* thread : threads) {
                    uint64_t ops = thread->total_operations.load();
                    min_ops = std::min(min_ops, ops);
                    max_ops = std::max(max_ops, ops);
                }
                
                if (max_ops > 0) {
                    double imbalance = (double)(max_ops - min_ops) / max_ops * 100.0;
                    std::cout << "     Load imbalance: " << std::fixed << std::setprecision(1) << imbalance << "%";
                    
                    if (imbalance > 20.0) {
                        std::cout << " ⚠️  HIGH IMBALANCE!";
                    } else if (imbalance > 10.0) {
                        std::cout << " ⚠️  Moderate imbalance";
                    } else {
                        std::cout << " ✅ Good balance";
                    }
                    std::cout << std::endl;
                }
            }
        }
        
        // Memory usage analysis
        std::cout << "\n💾 Memory Usage Analysis:" << std::endl;
        uint64_t total_allocs = memory_stats.total_allocations.load();
        uint64_t total_bytes = memory_stats.total_bytes_allocated.load();
        uint64_t peak_bytes = memory_stats.peak_memory_usage.load();
        uint64_t current_bytes = memory_stats.current_memory_usage.load();
        
        std::cout << "   Total allocations: " << total_allocs << std::endl;
        std::cout << "   Total bytes allocated: " << (total_bytes / 1024 / 1024) << " MB" << std::endl;
        std::cout << "   Peak memory usage: " << (peak_bytes / 1024 / 1024) << " MB" << std::endl;
        std::cout << "   Current usage: " << (current_bytes / 1024 / 1024) << " MB" << std::endl;
        
        if (current_bytes > peak_bytes * 0.9) {
            std::cout << "   ⚠️  Memory usage remains high - possible leak?" << std::endl;
        } else {
            std::cout << "   ✅ Memory usage looks normal" << std::endl;
        }
        
        // Queue depth analysis
        std::cout << "\n📋 Queue Analysis:" << std::endl;
        for (const auto& stats : queue_stats) {
            uint64_t enqueues = stats.total_enqueues.load();
            uint64_t dequeues = stats.total_dequeues.load();
            uint64_t max_depth = stats.max_depth.load();
            
            std::cout << "   NUMA " << stats.numa_node << " queue: " 
                     << enqueues << " enqueues, " << dequeues << " dequeues, max depth " << max_depth << std::endl;
            
            if (max_depth > 100) {
                std::cout << "     ⚠️  High queue depth - possible bottleneck!" << std::endl;
            } else if (max_depth > 20) {
                std::cout << "     ⚠️  Moderate queuing" << std::endl;
            } else {
                std::cout << "     ✅ Low queuing" << std::endl;
            }
        }
    }

    // Run comprehensive instrumentation test suite
    void run_comprehensive_instrumentation() {
        std::cout << "🔬 NUMA Coordinator Instrumentation Test Suite" << std::endl;
        std::cout << "===============================================" << std::endl;
        
        int cpu_count = cpu_count_math_cpus(0, true);
        std::cout << "Available CPUs: " << cpu_count << std::endl << std::endl;
        
        // Test different configurations
        std::vector<std::tuple<int, int, int>> test_configs = {
            {1, std::min(cpu_count, 4), 2},     // Single node, low complexity
            {2, std::min(cpu_count/2, 4), 3},   // Dual node, medium complexity
            {4, std::min(cpu_count/4, 2), 4},   // Quad node, high complexity
        };
        
        for (const auto& config : test_configs) {
            int numa_nodes = std::get<0>(config);
            int threads_per_node = std::get<1>(config);
            int complexity = std::get<2>(config);
            
            if (threads_per_node < 1) threads_per_node = 1;
            
            run_instrumented_test(numa_nodes, threads_per_node, complexity);
        }
        
        std::cout << "\n🏁 Instrumentation test suite completed!" << std::endl;
    }

    // Static member definitions
    static std::vector<MutexStats> mutex_stats;
    static std::vector<ThreadStats> thread_stats;
    static std::vector<QueueStats> queue_stats;
    static MemoryStats memory_stats;
    static std::atomic<bool> instrumentation_enabled;
    static std::mutex stats_mutex;
};

// Static member definitions
std::vector<NumaCoordinatorInstrumentation::MutexStats> NumaCoordinatorInstrumentation::mutex_stats;
std::vector<NumaCoordinatorInstrumentation::ThreadStats> NumaCoordinatorInstrumentation::thread_stats;
std::vector<NumaCoordinatorInstrumentation::QueueStats> NumaCoordinatorInstrumentation::queue_stats;
NumaCoordinatorInstrumentation::MemoryStats NumaCoordinatorInstrumentation::memory_stats;
std::atomic<bool> NumaCoordinatorInstrumentation::instrumentation_enabled{false};
std::mutex NumaCoordinatorInstrumentation::stats_mutex;

int main() {
    NumaCoordinatorInstrumentation instrumenter;
    instrumenter.run_comprehensive_instrumentation();
    return 0;
}
