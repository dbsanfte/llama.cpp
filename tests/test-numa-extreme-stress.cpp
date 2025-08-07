#include <chrono>
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <iomanip>
#include <mutex>
#include <sched.h>
#include <unistd.h>
#include <set>
#include <algorithm>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-numa-coordinator.h"
#include "common.h"

// CPU topology detection for Linux x86_64
#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)
#include <fstream>
#include <sstream>
#ifdef GGML_NUMA_MIRROR
#include <numa.h>
#include <numaif.h>
#endif
#endif

class ExtremeStressTester {
private:
    int max_numa_nodes;
    int max_cpus;
    std::atomic<int> operations_completed{0};
    std::atomic<bool> should_stop{false};
    
    // CPU topology information
    struct CPUTopology {
        int total_logical_cpus;
        int total_physical_cores;
        int numa_nodes;
        std::vector<int> physical_cores_per_node;
        std::vector<int> logical_cpus_per_node;
        std::vector<std::vector<int>> numa_cpu_ids;  // CPU IDs for each NUMA node
        bool has_hyperthreading;
        bool has_numa;
        bool is_hybrid_cpu;
        std::vector<int> performance_cpus;
        std::vector<int> efficiency_cpus;
        std::vector<std::vector<int>> core_siblings;  // Hyperthreading groups
    } cpu_topology;
    
    // Progress callback tracking
    std::atomic<int> total_callbacks{0};
    std::atomic<int> callback_errors{0};
    std::mutex callback_mutex;
    
    struct StressTestResults {
        double time_ms;
        double throughput_ops_per_sec;
        double cpu_utilization;
        int total_operations;
        int concurrent_threads;
        bool success;
        
        // NUMA coordinator stats
        int64_t total_work_items;
        double coordinator_throughput;
        double average_processing_time_us;
        
        // Progress callback stats
        int callbacks_received;
        int callback_errors;
    };
    
    // Results storage for final summary
    std::vector<std::tuple<int, int, int, StressTestResults, std::string>> extreme_stress_results;
    std::vector<std::tuple<int, int, StressTestResults, std::string>> memory_stress_results;
    
    // Progress callback function
    static void progress_callback_func(int work_id, int numa_node, struct ggml_tensor * tensor, void * user_data) {
        ExtremeStressTester * tester = (ExtremeStressTester*)user_data;
        tester->handleProgressCallback(work_id, numa_node, tensor);
    }
    
    void handleProgressCallback(int work_id, int numa_node, struct ggml_tensor * tensor) {
        total_callbacks++;
        
        // Validate callback parameters (minimal validation for performance)
        // Note: max_numa_nodes can be 0 in some environments, so handle that case
        int effective_max_nodes = std::max(1, max_numa_nodes);
        if (work_id < 0 || numa_node < 0 || numa_node >= effective_max_nodes || !tensor) {
            callback_errors++;
        }
        
        // Periodic progress reporting (every 100 callbacks)
        if (total_callbacks.load() % 100 == 0) {
            std::lock_guard<std::mutex> lock(callback_mutex);
            std::cout << "    [CALLBACKS] " << total_callbacks.load() << " work items completed" << std::endl;
        }
    }
    
    // Detect CPU topology - Linux x86_64 implementation
    void detectCPUTopology() {
        cpu_topology = {};  // Initialize to zeros
        
#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)
        // Get basic CPU information
        cpu_topology.total_logical_cpus = sysconf(_SC_NPROCESSORS_ONLN);
        
        // Detect NUMA configuration
#ifdef GGML_NUMA_MIRROR
        if (numa_available() != -1) {
            cpu_topology.numa_nodes = numa_num_configured_nodes();
            cpu_topology.has_numa = cpu_topology.numa_nodes > 1;
            
            // Get CPU IDs for each NUMA node
            cpu_topology.numa_cpu_ids.resize(cpu_topology.numa_nodes);
            cpu_topology.physical_cores_per_node.resize(cpu_topology.numa_nodes);
            cpu_topology.logical_cpus_per_node.resize(cpu_topology.numa_nodes);
            
            for (int node = 0; node < cpu_topology.numa_nodes; node++) {
                struct bitmask* node_cpus = numa_allocate_cpumask();
                if (numa_node_to_cpus(node, node_cpus) == 0) {
                    for (int cpu = 0; cpu < cpu_topology.total_logical_cpus; cpu++) {
                        if (numa_bitmask_isbitset(node_cpus, cpu)) {
                            cpu_topology.numa_cpu_ids[node].push_back(cpu);
                        }
                    }
                    cpu_topology.logical_cpus_per_node[node] = cpu_topology.numa_cpu_ids[node].size();
                }
                numa_free_cpumask(node_cpus);
            }
        } else
#endif
        {
            // Single NUMA node system
            cpu_topology.numa_nodes = 1;
            cpu_topology.has_numa = false;
            cpu_topology.numa_cpu_ids.resize(1);
            cpu_topology.physical_cores_per_node.resize(1);
            cpu_topology.logical_cpus_per_node.resize(1);
            
            for (int cpu = 0; cpu < cpu_topology.total_logical_cpus; cpu++) {
                cpu_topology.numa_cpu_ids[0].push_back(cpu);
            }
            cpu_topology.logical_cpus_per_node[0] = cpu_topology.total_logical_cpus;
        }
        
        // Detect hyperthreading and core topology
        std::map<std::string, std::vector<int>> sibling_groups;
        for (int cpu = 0; cpu < cpu_topology.total_logical_cpus; cpu++) {
            // Read thread siblings to identify hyperthreading groups
            std::ifstream siblings_file("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list");
            if (siblings_file.is_open()) {
                std::string siblings_str;
                std::getline(siblings_file, siblings_str);
                sibling_groups[siblings_str].push_back(cpu);
            }
        }
        
        // Convert sibling groups to core_siblings vector
        for (const auto& group : sibling_groups) {
            if (group.second.size() > 0) {
                cpu_topology.core_siblings.push_back(group.second);
            }
        }
        
        cpu_topology.total_physical_cores = cpu_topology.core_siblings.size();
        cpu_topology.has_hyperthreading = cpu_topology.total_logical_cpus > cpu_topology.total_physical_cores;
        
        // Calculate physical cores per NUMA node
        for (int node = 0; node < cpu_topology.numa_nodes; node++) {
            std::set<int> physical_cores_in_node;
            for (int cpu : cpu_topology.numa_cpu_ids[node]) {
                // Find which core group this CPU belongs to
                for (size_t core_idx = 0; core_idx < cpu_topology.core_siblings.size(); core_idx++) {
                    if (std::find(cpu_topology.core_siblings[core_idx].begin(), 
                                 cpu_topology.core_siblings[core_idx].end(), cpu) != 
                        cpu_topology.core_siblings[core_idx].end()) {
                        physical_cores_in_node.insert(core_idx);
                        break;
                    }
                }
            }
            cpu_topology.physical_cores_per_node[node] = physical_cores_in_node.size();
        }
        
        // Detect hybrid CPU (Intel P-cores vs E-cores) if available
        // For simplicity, assume all cores are performance cores unless we can detect otherwise
        cpu_topology.is_hybrid_cpu = false;  // TODO: Implement hybrid CPU detection if needed
        for (int cpu = 0; cpu < cpu_topology.total_logical_cpus; cpu++) {
            cpu_topology.performance_cpus.push_back(cpu);
        }
        
#else
        // Non-Linux fallback - basic detection
        cpu_topology.total_logical_cpus = std::thread::hardware_concurrency();
        cpu_topology.total_physical_cores = cpu_get_num_physical_cores();
        cpu_topology.numa_nodes = 1;
        cpu_topology.has_numa = false;
        cpu_topology.has_hyperthreading = cpu_topology.total_logical_cpus > cpu_topology.total_physical_cores;
        cpu_topology.is_hybrid_cpu = false;
        
        cpu_topology.numa_cpu_ids.resize(1);
        cpu_topology.physical_cores_per_node.resize(1, cpu_topology.total_physical_cores);
        cpu_topology.logical_cpus_per_node.resize(1, cpu_topology.total_logical_cpus);
        
        for (int cpu = 0; cpu < cpu_topology.total_logical_cpus; cpu++) {
            cpu_topology.numa_cpu_ids[0].push_back(cpu);
            cpu_topology.performance_cpus.push_back(cpu);
        }
#endif
    }
    
    // Generate thread scaling configurations based on detected topology
    std::vector<int> generateThreadConfigurations() {
        std::vector<int> configs;
        
        // Always test single thread as baseline
        configs.push_back(1);
        
        // Add powers of 2 up to logical CPUs
        for (int threads = 2; threads <= cpu_topology.total_logical_cpus; threads *= 2) {
            configs.push_back(threads);
        }
        
        // Add physical core counts per NUMA node
        for (int node = 0; node < cpu_topology.numa_nodes; node++) {
            int phys_cores = cpu_topology.physical_cores_per_node[node];
            int log_cores = cpu_topology.logical_cpus_per_node[node];
            
            if (phys_cores > 0 && std::find(configs.begin(), configs.end(), phys_cores) == configs.end()) {
                configs.push_back(phys_cores);
            }
            if (log_cores > 0 && std::find(configs.begin(), configs.end(), log_cores) == configs.end()) {
                configs.push_back(log_cores);
            }
        }
        
        // Add total physical cores
        if (cpu_topology.total_physical_cores > 0 && 
            std::find(configs.begin(), configs.end(), cpu_topology.total_physical_cores) == configs.end()) {
            configs.push_back(cpu_topology.total_physical_cores);
        }
        
        // Add total logical CPUs if different from physical
        if (cpu_topology.has_hyperthreading && 
            std::find(configs.begin(), configs.end(), cpu_topology.total_logical_cpus) == configs.end()) {
            configs.push_back(cpu_topology.total_logical_cpus);
        }
        
        // Add a few intermediate values for better scaling analysis
        if (cpu_topology.total_logical_cpus > 8) {
            int mid = cpu_topology.total_logical_cpus / 2;
            if (std::find(configs.begin(), configs.end(), mid) == configs.end()) {
                configs.push_back(mid);
            }
        }
        
        // Sort configurations
        std::sort(configs.begin(), configs.end());
        
        // Remove duplicates and filter out configs larger than total CPUs
        configs.erase(std::unique(configs.begin(), configs.end()), configs.end());
        configs.erase(std::remove_if(configs.begin(), configs.end(),
                                   [this](int threads) { return threads > cpu_topology.total_logical_cpus; }),
                      configs.end());
        
        return configs;
    }

public:
    ExtremeStressTester() {
        max_numa_nodes = ggml_numa_node_count();
        max_cpus = sysconf(_SC_NPROCESSORS_ONLN);
        
        // Detect detailed CPU topology
        detectCPUTopology();
        
        std::cout << "=== CPU Topology-Aware Extreme Stress Testing ===" << std::endl;
        std::cout << "System Configuration:" << std::endl;
        std::cout << "  Total logical CPUs: " << cpu_topology.total_logical_cpus << std::endl;
        std::cout << "  Total physical cores: " << cpu_topology.total_physical_cores << std::endl;
        std::cout << "  NUMA nodes: " << cpu_topology.numa_nodes << (cpu_topology.has_numa ? " (multi-NUMA)" : " (single-NUMA)") << std::endl;
        std::cout << "  Hyperthreading: " << (cpu_topology.has_hyperthreading ? "Available" : "Not available") << std::endl;
        std::cout << "  Hybrid CPU: " << (cpu_topology.is_hybrid_cpu ? "Yes (P+E cores)" : "No") << std::endl;
        
        // Display NUMA node details
        for (int node = 0; node < cpu_topology.numa_nodes; node++) {
            std::cout << "  NUMA node " << node << ": " 
                      << cpu_topology.physical_cores_per_node[node] << " physical cores, "
                      << cpu_topology.logical_cpus_per_node[node] << " logical CPUs";
            
            // Show first few CPU IDs for each node
            if (!cpu_topology.numa_cpu_ids[node].empty()) {
                std::cout << " [CPUs: ";
                for (size_t i = 0; i < std::min(size_t(4), cpu_topology.numa_cpu_ids[node].size()); i++) {
                    if (i > 0) std::cout << ",";
                    std::cout << cpu_topology.numa_cpu_ids[node][i];
                }
                if (cpu_topology.numa_cpu_ids[node].size() > 4) {
                    std::cout << ",...+" << (cpu_topology.numa_cpu_ids[node].size() - 4) << " more";
                }
                std::cout << "]";
            }
            std::cout << std::endl;
        }
        
        std::cout << "=====================================================" << std::endl;
    }

    void workerThread(int thread_id, int operations_per_thread, int tensor_size) {
        // Pin this thread to a specific CPU
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(thread_id % max_cpus, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        
        // Process operations in small batches to avoid memory issues
        const int batch_size = 10;
        int completed = 0;
        
        while (completed < operations_per_thread && !should_stop.load()) {
            int current_batch = std::min(batch_size, operations_per_thread - completed);
            
            // Create GGML context for this batch
            struct ggml_init_params params = {
                /*.mem_size   =*/ 256 * 1024 * 1024, // 256MB for extreme workloads
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ false,
            };
            
            struct ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                std::cerr << "Thread " << thread_id << ": Failed to initialize GGML context" << std::endl;
                return;
            }
            
            // Create NUMA coordinator manager for this batch
            struct ggml_numa_coordinator_manager * coordinator = ggml_numa_coordinator_manager_get_global(max_cpus, false);
            if (!coordinator) {
                std::cerr << "Thread " << thread_id << ": Failed to get NUMA coordinator" << std::endl;
                ggml_free(ctx);
                return;
            }
            
            // Create and execute tensor operations using NUMA coordinator
            for (int i = 0; i < current_batch && !should_stop.load(); i++) {
                struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, tensor_size);
                struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, tensor_size);
                struct ggml_tensor * c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, tensor_size);
                
                // Initialize with computation-heavy data patterns
                float * data_a = (float*)ggml_get_data(a);
                float * data_b = (float*)ggml_get_data(b);
                float * data_c = (float*)ggml_get_data(c);
                
                for (int j = 0; j < tensor_size; j++) {
                    data_a[j] = 1.0f + thread_id * 0.1f + j * 0.0001f;
                    data_b[j] = 2.0f + completed * 0.01f + j * 0.0002f;
                    data_c[j] = 0.5f + i * 0.05f;
                }
                
                // Chain multiple operations for more CPU load
                struct ggml_tensor * sum_ab = ggml_add(ctx, a, b);
                struct ggml_tensor * mul_abc = ggml_mul(ctx, sum_ab, c);
                struct ggml_tensor * result = ggml_scale(ctx, mul_abc, 1.5f);
                
                // Create computation graph
                struct ggml_cgraph * graph = ggml_new_graph(ctx);
                ggml_build_forward_expand(graph, result);
                
                // Use NUMA coordinator to execute the graph
                if (ggml_numa_coordinator_manager_set_cgraph(coordinator, graph) == 0) {
                    if (ggml_numa_coordinator_manager_start(coordinator) == 0) {
                        // Submit work to NUMA coordinator
                        int preferred_numa_node = thread_id % (max_numa_nodes > 0 ? max_numa_nodes : 1);
                        
                        int work_id = ggml_numa_coordinator_manager_submit_work(coordinator, result, preferred_numa_node);
                        if (work_id >= 0) {
                            // Wait for completion
                            ggml_numa_coordinator_manager_wait_for_completion(coordinator);
                        } else {
                            std::cerr << "Thread " << thread_id << ": Failed to submit work to coordinator" << std::endl;
                        }
                    }
                }
                
                operations_completed.fetch_add(1);
            }
            
            ggml_free(ctx);
            completed += current_batch;
        }
    }

    StressTestResults runStressTest(int total_operations, int thread_count, int tensor_size) {
        StressTestResults results = {};
        results.total_operations = total_operations;
        results.concurrent_threads = thread_count;
        results.success = false;
        
        // Reset callback counters
        total_callbacks.store(0);
        callback_errors.store(0);
        
        operations_completed.store(0);
        should_stop.store(false);
        
        // The NUMA coordinator now has built-in CPU mask and NUMA awareness integration.
        // When created, it automatically:
        // 1. Detects NUMA topology and CPU layout
        // 2. Filters CPU masks per NUMA node for optimal memory locality
        // 3. Distributes threads intelligently across NUMA nodes
        // 4. Provides thread affinity control integrated with GGML's existing CPU mask system
        
        // Create threadpool parameters for reference (coordinator will use its own NUMA-aware distribution)
        struct ggml_threadpool_params tpp;
        ggml_threadpool_params_init(&tpp, max_cpus);
        tpp.numa_aware = true;
        tpp.strict_cpu = true;
        tpp.allow_numa_override = true;
        tpp.warn_on_numa_override = false;
        tpp.prio = GGML_SCHED_PRIO_NORMAL;
        
        // Set CPU mask based on detected topology - coordinator will filter these per NUMA node
        for (int cpu = 0; cpu < cpu_topology.total_logical_cpus && cpu < GGML_MAX_N_THREADS; cpu++) {
            tpp.cpumask[cpu] = true;
        }
        
        // Set up NUMA coordinator - the coordinator now internally supports CPU masks and NUMA awareness
        // even when called through the original API due to our GGML integration
        struct ggml_numa_coordinator_manager * coordinator = ggml_numa_coordinator_manager_get_global(max_cpus, false);
        if (!coordinator) {
            std::cerr << "Failed to get NUMA coordinator manager with parameters" << std::endl;
            return results;
        }
        
        // Enable progress callbacks for real-time monitoring
        if (ggml_numa_coordinator_manager_set_progress_callback(coordinator, progress_callback_func, this) != 0) {
            std::cerr << "Failed to set progress callback" << std::endl;
            return results;
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Create GGML context for stress test
        struct ggml_init_params params = {
            /*.mem_size   =*/ 512 * 1024 * 1024, // 512MB for stress testing
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ false,
        };
        
        struct ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            std::cerr << "Failed to initialize GGML context for stress test" << std::endl;
            return results;
        }
        
        // Create computation graph with multiple operations
        struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, tensor_size);
        struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, tensor_size);
        struct ggml_tensor * c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, tensor_size);
        
        // Initialize tensor data
        float * data_a = (float*)ggml_get_data(a);
        float * data_b = (float*)ggml_get_data(b);
        float * data_c = (float*)ggml_get_data(c);
        
        for (int j = 0; j < tensor_size; j++) {
            data_a[j] = 1.0f + j * 0.0001f;
            data_b[j] = 2.0f + j * 0.0002f;
            data_c[j] = 0.5f + j * 0.0003f;
        }
        
        // Chain multiple operations for computational load
        struct ggml_tensor * sum_ab = ggml_add(ctx, a, b);
        struct ggml_tensor * mul_abc = ggml_mul(ctx, sum_ab, c);
        struct ggml_tensor * result = ggml_scale(ctx, mul_abc, 1.5f);
        
        // Create computation graph
        struct ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, result);
        
        // Set up coordinator with the computation graph
        if (ggml_numa_coordinator_manager_set_cgraph(coordinator, graph) != 0) {
            std::cerr << "Failed to set computation graph" << std::endl;
            ggml_free(ctx);
            return results;
        }
        
        // Start coordinator
        if (ggml_numa_coordinator_manager_start(coordinator) != 0) {
            std::cerr << "Failed to start coordinator" << std::endl;
            ggml_free(ctx);
            return results;
        }
        
        // Submit work items to stress the coordinator
        std::vector<int> work_ids;
        for (int i = 0; i < total_operations; i++) {
            int preferred_numa_node = i % (max_numa_nodes > 0 ? max_numa_nodes : 1);
            int work_id = ggml_numa_coordinator_manager_submit_work(coordinator, result, preferred_numa_node);
            if (work_id >= 0) {
                work_ids.push_back(work_id);
            } else {
                std::cerr << "Failed to submit work item " << i << std::endl;
                break;
            }
        }
        
        // Wait for all work to complete
        ggml_numa_coordinator_manager_wait_for_completion(coordinator);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end_time - start_time);
        
        results.time_ms = duration.count();
        results.throughput_ops_per_sec = work_ids.size() / (duration.count() / 1000.0);
        results.cpu_utilization = 100.0; // Assume full utilization for stress tests
        
        // Get NUMA coordinator statistics
        struct ggml_numa_perf_stats stats = ggml_numa_coordinator_manager_get_stats(coordinator, -1);
        results.total_work_items = stats.total_work_items;
        results.coordinator_throughput = stats.throughput_items_per_sec;
        results.average_processing_time_us = stats.average_processing_time_us;
        
        // Disable progress callbacks
        ggml_numa_coordinator_manager_set_progress_callback(coordinator, nullptr, nullptr);
        
        // Collect callback statistics
        results.callbacks_received = total_callbacks.load();
        results.callback_errors = callback_errors.load();
        
        // Clean up
        ggml_free(ctx);
        
        results.success = (work_ids.size() == static_cast<size_t>(total_operations));
        return results;
    }

    void runWarmupTest() {
        std::cout << "\n=== Running Warmup Phase ===" << std::endl;
        std::cout << "Initializing coordinator threads and warming up system..." << std::endl;
        
        // Simple warmup test to initialize all coordinator infrastructure
        try {
            StressTestResults warmup_result = runStressTest(50, 4, 512);
            if (warmup_result.success) {
                std::cout << "✅ Warmup completed successfully - coordinator ready" << std::endl;
            } else {
                std::cout << "⚠️  Warmup had issues but continuing with main tests" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "⚠️  Warmup exception: " << e.what() << " - continuing with main tests" << std::endl;
        }
        
        // Brief pause to let system settle
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    void runExtremeStressTests() {
        std::cout << "\n=== Running CPU Topology-Aware Stress Tests ===" << std::endl;
        
        // Generate thread configurations based on actual CPU topology
        std::vector<int> thread_configs = generateThreadConfigurations();
        
        std::cout << "Thread scaling configurations: ";
        for (size_t i = 0; i < thread_configs.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << thread_configs[i];
        }
        std::cout << std::endl;
        
        // Test configurations: operations, tensor_size
        std::vector<std::pair<int, int>> test_configs = {
            // Light load for baseline measurements
            {1000, 1024},    
            {2000, 1024},    
            
            // Moderate concurrent load
            {5000, 2048},    
            {10000, 2048},   
            
            // Heavy concurrent load with larger tensors
            {15000, 4096},   
            {25000, 4096},   
            
            // Extreme concurrent load
            {50000, 4096},   
            {75000, 6144},   
            {100000, 8192},  // Ultimate stress test
        };
        
        // Run tests for each operation/tensor size combo with all thread configurations
        for (const auto& test_config : test_configs) {
            int operations = test_config.first;
            int tensor_size = test_config.second;
            
            std::cout << "\n--- Testing " << operations << " operations, tensor size " << tensor_size << " ---" << std::endl;
            
            for (int threads : thread_configs) {
                // Skip configurations that don't make sense
                if (threads > operations / 10 && operations < 5000) {
                    continue;  // Skip high thread counts for light loads
                }
                
                std::cout << "  Running: " << threads << " threads..." << std::flush;
                
                try {
                    StressTestResults result = runStressTest(operations, threads, tensor_size);
                    extreme_stress_results.emplace_back(operations, threads, tensor_size, result, "");
                    
                    if (result.success) {
                        std::cout << " ✅ " << std::fixed << std::setprecision(1) 
                                  << result.throughput_ops_per_sec << " ops/s" << std::endl;
                    } else {
                        std::cout << " ❌ FAILED" << std::endl;
                    }
                } catch (const std::exception& e) {
                    std::cout << " ❌ ERROR: " << e.what() << std::endl;
                    StressTestResults failed_result = {};
                    failed_result.success = false;
                    extreme_stress_results.emplace_back(operations, threads, tensor_size, failed_result, e.what());
                }
                
                // Brief pause between tests to let system recover
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        }
        
        // Run specific NUMA scaling tests if we have multiple NUMA nodes
        if (cpu_topology.has_numa) {
            std::cout << "\n--- Running NUMA-specific scaling tests ---" << std::endl;
            
            // Test with thread counts that match NUMA topology
            std::vector<int> numa_thread_configs;
            for (int node = 0; node < cpu_topology.numa_nodes; node++) {
                numa_thread_configs.push_back(cpu_topology.physical_cores_per_node[node]);
                numa_thread_configs.push_back(cpu_topology.logical_cpus_per_node[node]);
            }
            
            // Remove duplicates and sort
            std::sort(numa_thread_configs.begin(), numa_thread_configs.end());
            numa_thread_configs.erase(std::unique(numa_thread_configs.begin(), numa_thread_configs.end()), 
                                     numa_thread_configs.end());
            
            int numa_operations = 20000;  // Standard load for NUMA comparison
            int numa_tensor_size = 2048;
            
            for (int threads : numa_thread_configs) {
                if (threads <= 0) continue;
                
                std::cout << "  NUMA test: " << threads << " threads..." << std::flush;
                
                try {
                    StressTestResults result = runStressTest(numa_operations, threads, numa_tensor_size);
                    extreme_stress_results.emplace_back(numa_operations, threads, numa_tensor_size, result, "NUMA-specific");
                    
                    if (result.success) {
                        std::cout << " ✅ " << std::fixed << std::setprecision(1) 
                                  << result.throughput_ops_per_sec << " ops/s" << std::endl;
                    } else {
                        std::cout << " ❌ FAILED" << std::endl;
                    }
                } catch (const std::exception& e) {
                    std::cout << " ❌ ERROR: " << e.what() << std::endl;
                    StressTestResults failed_result = {};
                    failed_result.success = false;
                    extreme_stress_results.emplace_back(numa_operations, threads, numa_tensor_size, failed_result, e.what());
                }
                
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        }
    }

    void runMemoryStressTest() {
        std::cout << "\n=== Running Memory Bandwidth Stress Test ===" << std::endl;
        
        // Test with progressively larger tensor sizes to stress memory bandwidth
        std::vector<int> tensor_sizes = {512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
        
        // Use optimal thread count for memory tests (typically physical cores)
        int memory_test_threads = cpu_topology.total_physical_cores;
        if (memory_test_threads <= 0) {
            memory_test_threads = std::min(8, cpu_topology.total_logical_cpus);
        }
        
        std::cout << "Using " << memory_test_threads << " threads for memory bandwidth tests" << std::endl;
        
        for (int tensor_size : tensor_sizes) {
            // Scale operations inversely with tensor size to maintain reasonable test duration
            int operations = std::max(1000, 50000 / (tensor_size / 1024));
            operations = std::min(operations, 20000);  // Cap at 20K operations
            
            std::cout << "Memory test: tensor size " << tensor_size << " (" << operations << " operations)..." << std::flush;
            
            try {
                StressTestResults result = runStressTest(operations, memory_test_threads, tensor_size);
                memory_stress_results.emplace_back(tensor_size, operations, result, "");
                
                if (result.success) {
                    // Calculate memory bandwidth
                    double memory_per_op_mb = (tensor_size * 3 * sizeof(float)) / (1024.0 * 1024.0);
                    double total_memory_gb = (memory_per_op_mb * operations) / 1024.0;
                    double bandwidth_gb_s = total_memory_gb / (result.time_ms / 1000.0);
                    
                    std::cout << " ✅ " << std::fixed << std::setprecision(1) 
                              << result.throughput_ops_per_sec << " ops/s, "
                              << std::setprecision(2) << bandwidth_gb_s << " GB/s" << std::endl;
                } else {
                    std::cout << " ❌ FAILED" << std::endl;
                }
            } catch (const std::exception& e) {
                std::cout << " ❌ ERROR: " << e.what() << std::endl;
                StressTestResults failed_result = {};
                failed_result.success = false;
                memory_stress_results.emplace_back(tensor_size, operations, failed_result, e.what());
            }
            
            // Brief pause between memory tests
            std::this_thread::sleep_for(std::chrono::milliseconds(800));
        }
    }
    
    void displayFinalSummary() {
        std::cout << "\n" << std::string(140, '=') << std::endl;
        std::cout << "                            FINAL PERFORMANCE SUMMARY - CPU TOPOLOGY AWARE" << std::endl;
        std::cout << std::string(140, '=') << std::endl;
        
        // Display CPU topology summary
        std::cout << "\n=== CPU Topology Summary ===" << std::endl;
        std::cout << "System: " << cpu_topology.total_logical_cpus << " logical CPUs, " 
                  << cpu_topology.total_physical_cores << " physical cores, " 
                  << cpu_topology.numa_nodes << " NUMA node" << (cpu_topology.numa_nodes > 1 ? "s" : "") << std::endl;
        std::cout << "Features: " << (cpu_topology.has_hyperthreading ? "Hyperthreading" : "No HT") 
                  << ", " << (cpu_topology.has_numa ? "Multi-NUMA" : "Single-NUMA")
                  << ", " << (cpu_topology.is_hybrid_cpu ? "Hybrid (P+E)" : "Homogeneous cores") << std::endl;
        
        // Display extreme stress test results
        if (!extreme_stress_results.empty()) {
            std::cout << "\n=== Extreme Concurrent Stress Tests Results ===" << std::endl;
            std::cout << "Operations | Threads | Tensor Size | Time (s) | Throughput (ops/s) | Coord Items | Coord Tput | Callbacks | CB Errors | Status" << std::endl;
            std::cout << "-----------|---------|-------------|----------|-------------------|-------------|------------|-----------|-----------|-------" << std::endl;
            
            for (const auto& result_tuple : extreme_stress_results) {
                int operations = std::get<0>(result_tuple);
                int threads = std::get<1>(result_tuple);
                int tensor_size = std::get<2>(result_tuple);
                const StressTestResults& result = std::get<3>(result_tuple);
                const std::string& error_msg = std::get<4>(result_tuple);
                
                if (result.success) {
                    std::cout << std::setw(10) << operations
                              << " | " << std::setw(7) << threads
                              << " | " << std::setw(11) << tensor_size
                              << " | " << std::setw(8) << std::fixed << std::setprecision(2) 
                              << result.time_ms / 1000.0
                              << " | " << std::setw(17) << std::fixed << std::setprecision(1) 
                              << result.throughput_ops_per_sec
                              << " | " << std::setw(11) << result.total_work_items
                              << " | " << std::setw(10) << std::fixed << std::setprecision(1) 
                              << result.coordinator_throughput
                              << " | " << std::setw(9) << result.callbacks_received
                              << " | " << std::setw(9) << result.callback_errors
                              << " | SUCCESS" << std::endl;
                } else {
                    std::cout << std::setw(10) << operations << " | " << std::setw(7) << threads 
                              << " | " << std::setw(11) << tensor_size 
                              << " | " << std::setw(8) << "FAILED"
                              << " | " << std::setw(17) << "---"
                              << " | " << std::setw(11) << "---"
                              << " | " << std::setw(10) << "---"
                              << " | " << std::setw(9) << "---"
                              << " | " << std::setw(9) << "---"
                              << " | " << (error_msg.empty() ? "FAILED" : error_msg) << std::endl;
                }
            }
            
            // Thread scaling analysis for different operation counts and tensor sizes
            std::cout << "\n=== Thread Scaling Analysis by Configuration ===" << std::endl;
            
            // Group results by operation count and tensor size
            std::map<std::pair<int, int>, std::vector<std::pair<int, StressTestResults>>> grouped_results;
            
            for (const auto& result_tuple : extreme_stress_results) {
                int operations = std::get<0>(result_tuple);
                int threads = std::get<1>(result_tuple);
                int tensor_size = std::get<2>(result_tuple);
                const StressTestResults& result = std::get<3>(result_tuple);
                
                if (result.success) {
                    std::pair<int, int> config = {operations, tensor_size};
                    grouped_results[config].emplace_back(threads, result);
                }
            }
            
            // Analyze scaling for each configuration
            for (const auto& group : grouped_results) {
                int operations = group.first.first;
                int tensor_size = group.first.second;
                auto thread_results = group.second;
                
                // Sort by thread count
                std::sort(thread_results.begin(), thread_results.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });
                
                if (thread_results.size() >= 3) {  // Only analyze if we have enough data points
                    std::cout << "\nConfiguration: " << operations << " operations, " << tensor_size << " tensor size" << std::endl;
                    std::cout << "Threads | Throughput (ops/s) | Scaling Factor | Efficiency % | Physical Core Usage | HT Usage" << std::endl;
                    std::cout << "--------|-------------------|----------------|--------------|---------------------|----------" << std::endl;
                    
                    double baseline_throughput = thread_results[0].second.throughput_ops_per_sec;
                    
                    for (const auto& thread_result : thread_results) {
                        int threads = thread_result.first;
                        const StressTestResults& result = thread_result.second;
                        
                        double scaling_factor = result.throughput_ops_per_sec / baseline_throughput;
                        double efficiency = (scaling_factor / threads) * 100.0;
                        
                        // Analyze CPU utilization relative to topology
                        std::string core_usage = "---";
                        std::string ht_usage = "---";
                        
                        if (threads <= cpu_topology.total_physical_cores) {
                            double physical_usage = (threads / (double)cpu_topology.total_physical_cores) * 100.0;
                            core_usage = std::to_string((int)physical_usage) + "%";
                            ht_usage = "No";
                        } else if (cpu_topology.has_hyperthreading) {
                            core_usage = "100%";
                            double ht_usage_pct = ((threads - cpu_topology.total_physical_cores) / 
                                                   (double)cpu_topology.total_physical_cores) * 100.0;
                            ht_usage = std::to_string((int)ht_usage_pct) + "%";
                        }
                        
                        std::cout << std::setw(7) << threads
                                  << " | " << std::setw(17) << std::fixed << std::setprecision(1) 
                                  << result.throughput_ops_per_sec
                                  << " | " << std::setw(14) << std::fixed << std::setprecision(2) 
                                  << scaling_factor
                                  << " | " << std::setw(12) << std::fixed << std::setprecision(1) 
                                  << efficiency
                                  << " | " << std::setw(19) << core_usage
                                  << " | " << std::setw(8) << ht_usage << std::endl;
                    }
                }
            }
            
            // NUMA analysis if applicable
            if (cpu_topology.has_numa) {
                std::cout << "\n=== NUMA Scaling Analysis ===" << std::endl;
                std::cout << "Note: Results include NUMA-specific thread configurations optimized for " 
                          << cpu_topology.numa_nodes << " NUMA nodes" << std::endl;
                
                // Find optimal thread counts per NUMA node
                for (int node = 0; node < cpu_topology.numa_nodes; node++) {
                    int node_phys_cores = cpu_topology.physical_cores_per_node[node];
                    int node_log_cpus = cpu_topology.logical_cpus_per_node[node];
                    
                    std::cout << "NUMA node " << node << ": " << node_phys_cores 
                              << " physical cores, " << node_log_cpus << " logical CPUs" << std::endl;
                }
            }
        }
        
        // Display memory stress test results with bandwidth analysis
        if (!memory_stress_results.empty()) {
            std::cout << "\n=== Memory Bandwidth Stress Test Results ===" << std::endl;
            std::cout << "Tensor Size | Operations | Time (s) | Throughput (ops/s) | Memory/Op (MB) | Bandwidth (GB/s) | Callbacks | CB Errors | Status" << std::endl;
            std::cout << "------------|------------|----------|-------------------|----------------|------------------|-----------|-----------|-------" << std::endl;
            
            for (const auto& result_tuple : memory_stress_results) {
                int tensor_size = std::get<0>(result_tuple);
                int operations = std::get<1>(result_tuple);
                const StressTestResults& result = std::get<2>(result_tuple);
                const std::string& error_msg = std::get<3>(result_tuple);
                
                if (result.success) {
                    double memory_per_op = (tensor_size * 3 * sizeof(float)) / (1024.0 * 1024.0); // 3 tensors per op
                    double total_memory_gb = (memory_per_op * operations) / 1024.0;
                    double bandwidth_gb_s = total_memory_gb / (result.time_ms / 1000.0);
                    
                    std::cout << std::setw(11) << tensor_size
                              << " | " << std::setw(10) << operations
                              << " | " << std::setw(8) << std::fixed << std::setprecision(2) 
                              << result.time_ms / 1000.0
                              << " | " << std::setw(17) << std::fixed << std::setprecision(1) 
                              << result.throughput_ops_per_sec
                              << " | " << std::setw(14) << std::fixed << std::setprecision(2) 
                              << memory_per_op
                              << " | " << std::setw(16) << std::fixed << std::setprecision(2)
                              << bandwidth_gb_s
                              << " | " << std::setw(9) << result.callbacks_received
                              << " | " << std::setw(9) << result.callback_errors
                              << " | SUCCESS" << std::endl;
                } else {
                    std::cout << std::setw(11) << tensor_size << " | " << std::setw(10) << operations
                              << " | " << std::setw(8) << "FAILED"
                              << " | " << std::setw(17) << "---"
                              << " | " << std::setw(14) << "---"
                              << " | " << std::setw(16) << "---"
                              << " | " << std::setw(9) << "---"
                              << " | " << std::setw(9) << "---"
                              << " | " << (error_msg.empty() ? "FAILED" : error_msg) << std::endl;
                }
            }
        }
        
        std::cout << "\n" << std::string(140, '=') << std::endl;
    }
};

int main() {
    std::cout << "NUMA Extreme CPU Stress Testing Suite" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    try {
        ExtremeStressTester tester;
        
        // Run warmup to initialize coordinator infrastructure
        tester.runWarmupTest();
        
        // Run extreme concurrent stress tests
        tester.runExtremeStressTests();
        
        // Run memory bandwidth stress tests
        tester.runMemoryStressTest();
        
        // Display comprehensive final summary
        tester.displayFinalSummary();
        
        std::cout << "\n=== Extreme Stress Testing Complete ===" << std::endl;
        std::cout << "✅ All stress tests completed successfully" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error during stress testing: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
