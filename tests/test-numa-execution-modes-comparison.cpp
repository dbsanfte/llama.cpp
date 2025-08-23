#include <chrono>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>

#ifdef __linux__
#include <numa.h>
#include <sched.h>
#include <unistd.h>
#endif

#include "ggml.h"
#include "ggml-cpu.h"

// Test case structure
struct TestCase {
    const char* name;
    const char* description;
    int dim1, dim2, dim3;
    size_t expected_elements;
};

class NumaExecutionModesComparison {
private:
    std::vector<TestCase> test_cases = {
        {"SMALL",  "Small tensor (cache-friendly)",     128, 128,   4,     65536},
        {"MEDIUM", "Medium tensor (L3 cache)",          256, 256,   8,    524288}, 
        {"LARGE",  "Large tensor (memory-bound)",       512, 512,  16,   4194304},
        {"HUGE",   "Huge tensor (bandwidth-limited)",   512, 512, 256,  67108864}
    };

    int total_physical_cores;

public:
    NumaExecutionModesComparison() {
        total_physical_cores = get_physical_core_count();
        printf("🖥️  Detected %d physical cores\n", total_physical_cores);
    }

private:
    int get_physical_core_count() {
        return static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
    }

#ifdef __linux__
    void bind_to_numa_node_0() {
        // Bind current process to NUMA node 0 only
        struct bitmask* mask = numa_allocate_cpumask();
        numa_node_to_cpus(0, mask);
        
        if (numa_sched_setaffinity(0, mask) == 0) {
            printf("      ✅ Process bound to NUMA node 0 CPUs\n");
        } else {
            printf("      ❌ Failed to bind to NUMA node 0\n");
        }
        
        numa_free_cpumask(mask);
    }

    void clear_cpu_binding() {
        // Allow process to run on any CPU
        cpu_set_t mask;
        CPU_ZERO(&mask);
        for (int i = 0; i < total_physical_cores; i++) {
            CPU_SET(i, &mask);
        }
        
        if (sched_setaffinity(0, sizeof(mask), &mask) == 0) {
            printf("      ✅ Process unbound (can run on any CPU)\n");
        } else {
            printf("      ❌ Failed to clear CPU binding\n");
        }
    }

    void validate_current_numa_setup(const char* test_name) {
        int current_cpu = sched_getcpu();
        int current_node = numa_node_of_cpu(current_cpu);
        printf("      🔍 [%s] Currently running on CPU %d (NUMA node %d)\n", 
               test_name, current_cpu, current_node);
    }
#endif

    // Mode 1: Pure fallback execution on NUMA node 0
    double test_fallback_node0_only(const TestCase& test_case) {
        printf("\n🔵 Mode 1: Fallback (NUMA Node 0 Only, All Threads)\n");
        printf("    Description: Traditional CPU execution bound to node 0\n");
        
#ifdef __linux__
        bind_to_numa_node_0();
        validate_current_numa_setup("Fallback-Node0");
#endif

        // Create context without NUMA
        size_t tensor_size = (size_t)test_case.dim1 * test_case.dim2 * test_case.dim3 * sizeof(float);
        size_t ctx_size = tensor_size * 8 + 64*1024*1024;

        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };

        struct ggml_context* ctx = ggml_init(params);
        if (!ctx) {
            printf("❌ Failed to create fallback context\n");
            return -1.0;
        }

        struct ggml_tensor* tensor_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, test_case.dim1, test_case.dim2, test_case.dim3);
        struct ggml_tensor* tensor_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, test_case.dim1, test_case.dim2, test_case.dim3);
        struct ggml_tensor* result = ggml_add(ctx, tensor_a, tensor_b);

        if (!tensor_a || !tensor_b || !result) {
            printf("❌ Failed to create tensors\n");
            ggml_free(ctx);
            return -1.0;
        }

        // Initialize data
        float* a_data = (float*)ggml_get_data(tensor_a);
        float* b_data = (float*)ggml_get_data(tensor_b);
        size_t total_elements = ggml_nelements(tensor_a);

        for (size_t i = 0; i < total_elements; i++) {
            a_data[i] = 1.5f + (i % 100) * 0.01f;
            b_data[i] = 2.5f + (i % 100) * 0.01f;
        }

        struct ggml_cgraph* cgraph = ggml_new_graph(ctx);
        ggml_build_forward_expand(cgraph, result);

        printf("      Using %d threads (fallback, node 0 bound)\n", total_physical_cores);
        struct ggml_cplan cplan = ggml_graph_plan(cgraph, total_physical_cores, NULL);

        // Warmup
        for (int i = 0; i < 3; i++) {
            ggml_graph_compute(cgraph, &cplan);
        }

        // Measure performance
        const int num_runs = 10;
        std::vector<double> times;

        for (int run = 0; run < num_runs; run++) {
            auto start = std::chrono::high_resolution_clock::now();
            ggml_graph_compute(cgraph, &cplan);
            auto end = std::chrono::high_resolution_clock::now();

            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            if (time_ms > 0.001) {
                times.push_back(time_ms);
            }
        }

        ggml_free(ctx);

        if (times.empty()) {
            return -1.0;
        }

        std::sort(times.begin(), times.end());
        if (times.size() > 4) {
            times.erase(times.begin());
            times.pop_back();
        }

        double avg_time = 0.0;
        for (double time : times) {
            avg_time += time;
        }
        avg_time /= times.size();

        printf("      ✅ Average time: %.3f ms\n", avg_time);
        return avg_time;
    }

    // Mode 2: NUMA coordinator on node 0 only
    double test_numa_coordinator_node0_only(const TestCase& test_case) {
        printf("\n🟡 Mode 2: NUMA Coordinator (NUMA Node 0 Only, All Threads)\n");
        printf("    Description: NUMA coordinator with work restricted to node 0\n");

#ifdef __linux__
        bind_to_numa_node_0();
        validate_current_numa_setup("NUMA-Node0");
#endif

        // Initialize NUMA
        ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);

        size_t tensor_size = (size_t)test_case.dim1 * test_case.dim2 * test_case.dim3 * sizeof(float);
        size_t ctx_size = tensor_size * 8 + 64*1024*1024;

        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };

        struct ggml_context* ctx = ggml_init(params);
        if (!ctx) {
            printf("❌ Failed to create NUMA context\n");
            return -1.0;
        }

        struct ggml_tensor* tensor_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, test_case.dim1, test_case.dim2, test_case.dim3);
        struct ggml_tensor* tensor_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, test_case.dim1, test_case.dim2, test_case.dim3);
        struct ggml_tensor* result = ggml_add(ctx, tensor_a, tensor_b);

        if (!tensor_a || !tensor_b || !result) {
            printf("❌ Failed to create tensors\n");
            ggml_free(ctx);
            return -1.0;
        }

        // Initialize data
        float* a_data = (float*)ggml_get_data(tensor_a);
        float* b_data = (float*)ggml_get_data(tensor_b);
        size_t total_elements = ggml_nelements(tensor_a);

        for (size_t i = 0; i < total_elements; i++) {
            a_data[i] = 1.5f + (i % 100) * 0.01f;
            b_data[i] = 2.5f + (i % 100) * 0.01f;
        }

        struct ggml_cgraph* cgraph = ggml_new_graph(ctx);
        ggml_build_forward_expand(cgraph, result);

        printf("      Using %d threads (NUMA coordinator, node 0 only)\n", total_physical_cores);

        // Warmup
        for (int i = 0; i < 3; i++) {
            ggml_graph_compute_with_ctx(ctx, cgraph, total_physical_cores);
        }

        // Measure performance
        const int num_runs = 10;
        std::vector<double> times;

        for (int run = 0; run < num_runs; run++) {
            auto start = std::chrono::high_resolution_clock::now();
            enum ggml_status status = ggml_graph_compute_with_ctx(ctx, cgraph, total_physical_cores);
            auto end = std::chrono::high_resolution_clock::now();

            if (status != GGML_STATUS_SUCCESS) {
                printf("❌ NUMA computation failed\n");
                ggml_free(ctx);
                return -1.0;
            }

            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            if (time_ms > 0.001) {
                times.push_back(time_ms);
            }
        }

        ggml_free(ctx);

        if (times.empty()) {
            return -1.0;
        }

        std::sort(times.begin(), times.end());
        if (times.size() > 4) {
            times.erase(times.begin());
            times.pop_back();
        }

        double avg_time = 0.0;
        for (double time : times) {
            avg_time += time;
        }
        avg_time /= times.size();

        printf("      ✅ Average time: %.3f ms\n", avg_time);
        return avg_time;
    }

    // Mode 3: NUMA coordinator across all nodes
    double test_numa_coordinator_all_nodes(const TestCase& test_case) {
        printf("\n🟢 Mode 3: NUMA Coordinator (All Nodes, All Threads)\n");
        printf("    Description: NUMA coordinator with data-parallel across all nodes\n");

#ifdef __linux__
        clear_cpu_binding();
        validate_current_numa_setup("NUMA-AllNodes");
#endif

        // Initialize NUMA
        ggml_numa_init(GGML_NUMA_STRATEGY_MIRROR);

        size_t tensor_size = (size_t)test_case.dim1 * test_case.dim2 * test_case.dim3 * sizeof(float);
        size_t ctx_size = tensor_size * 8 + 64*1024*1024;

        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };

        struct ggml_context* ctx = ggml_init(params);
        if (!ctx) {
            printf("❌ Failed to create NUMA context\n");
            return -1.0;
        }

        struct ggml_tensor* tensor_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, test_case.dim1, test_case.dim2, test_case.dim3);
        struct ggml_tensor* tensor_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, test_case.dim1, test_case.dim2, test_case.dim3);
        struct ggml_tensor* result = ggml_add(ctx, tensor_a, tensor_b);

        if (!tensor_a || !tensor_b || !result) {
            printf("❌ Failed to create tensors\n");
            ggml_free(ctx);
            return -1.0;
        }

        // Initialize data
        float* a_data = (float*)ggml_get_data(tensor_a);
        float* b_data = (float*)ggml_get_data(tensor_b);
        size_t total_elements = ggml_nelements(tensor_a);

        for (size_t i = 0; i < total_elements; i++) {
            a_data[i] = 1.5f + (i % 100) * 0.01f;
            b_data[i] = 2.5f + (i % 100) * 0.01f;
        }

        struct ggml_cgraph* cgraph = ggml_new_graph(ctx);
        ggml_build_forward_expand(cgraph, result);

        printf("      Using %d threads (NUMA coordinator, all nodes)\n", total_physical_cores);

        // Warmup
        for (int i = 0; i < 3; i++) {
            ggml_graph_compute_with_ctx(ctx, cgraph, total_physical_cores);
        }

        // Measure performance
        const int num_runs = 10;
        std::vector<double> times;

        for (int run = 0; run < num_runs; run++) {
            auto start = std::chrono::high_resolution_clock::now();
            enum ggml_status status = ggml_graph_compute_with_ctx(ctx, cgraph, total_physical_cores);
            auto end = std::chrono::high_resolution_clock::now();

            if (status != GGML_STATUS_SUCCESS) {
                printf("❌ NUMA computation failed\n");
                ggml_free(ctx);
                return -1.0;
            }

            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            if (time_ms > 0.001) {
                times.push_back(time_ms);
            }
        }

        ggml_free(ctx);

        if (times.empty()) {
            return -1.0;
        }

        std::sort(times.begin(), times.end());
        if (times.size() > 4) {
            times.erase(times.begin());
            times.pop_back();
        }

        double avg_time = 0.0;
        for (double time : times) {
            avg_time += time;
        }
        avg_time /= times.size();

        printf("      ✅ Average time: %.3f ms\n", avg_time);
        return avg_time;
    }

public:
    void run_comparison() {
        printf("🎯 NUMA EXECUTION MODES COMPARISON\n");
        printf("===================================\n");
        printf("Comparing three execution modes:\n");
        printf("  🔵 Mode 1: Fallback (Node 0, All Threads)\n");
        printf("  🟡 Mode 2: NUMA Coordinator (Node 0 Only, All Threads)\n");
        printf("  🟢 Mode 3: NUMA Coordinator (All Nodes, All Threads)\n\n");

#ifdef __linux__
        printf("🖥️  NUMA Hardware Information:\n");
        printf("    Physical cores: %d\n", total_physical_cores);
        printf("    NUMA nodes: %d\n", ggml_numa_node_count());
        printf("    NUMA available: %s\n\n", ggml_is_numa() ? "true" : "false");
#endif

        struct ComparisonResult {
            std::string test_name;
            double fallback_time;
            double numa_node0_time;
            double numa_all_nodes_time;
            double node0_speedup;
            double all_nodes_speedup;
            size_t tensor_size_mb;
        };

        std::vector<ComparisonResult> results;

        for (const auto& test_case : test_cases) {
            printf("\n📊 Testing %s: %s\n", test_case.name, test_case.description);
            printf("=====================================\n");

            double fallback_time = test_fallback_node0_only(test_case);
            double numa_node0_time = test_numa_coordinator_node0_only(test_case);
            double numa_all_nodes_time = test_numa_coordinator_all_nodes(test_case);

            if (fallback_time > 0 && numa_node0_time > 0 && numa_all_nodes_time > 0) {
                ComparisonResult result;
                result.test_name = test_case.name;
                result.fallback_time = fallback_time;
                result.numa_node0_time = numa_node0_time;
                result.numa_all_nodes_time = numa_all_nodes_time;
                result.node0_speedup = fallback_time / numa_node0_time;
                result.all_nodes_speedup = fallback_time / numa_all_nodes_time;
                result.tensor_size_mb = (test_case.dim1 * test_case.dim2 * test_case.dim3 * sizeof(float)) / (1024 * 1024);

                results.push_back(result);

                printf("\n📈 PERFORMANCE SUMMARY FOR %s:\n", test_case.name);
                printf("  🔵 Fallback (Node 0):     %.3f ms\n", fallback_time);
                printf("  🟡 NUMA Coord (Node 0):   %.3f ms (%.2fx vs fallback)\n", numa_node0_time, result.node0_speedup);
                printf("  🟢 NUMA Coord (All):      %.3f ms (%.2fx vs fallback)\n", numa_all_nodes_time, result.all_nodes_speedup);
                printf("  📏 Tensor Size:            %zu MB\n", result.tensor_size_mb);
            } else {
                printf("❌ Test failed - one or more measurements invalid\n");
            }
        }

        // Print final comparison table
        printf("\n\n");
        printf("╔════════════════════════════════════════════════════════════════════════════════╗\n");
        printf("║                         NUMA EXECUTION MODES COMPARISON                       ║\n");
        printf("╠════════════════════════════════════════════════════════════════════════════════╣\n");
        printf("║ Test     │  Fallback │ NUMA Node0│ NUMA All  │ Node0 vs  │ All vs    │ Size  ║\n");
        printf("║ Case     │  (Node 0) │ (Node 0)  │ (All)     │ Fallback  │ Fallback  │ (MB)  ║\n");
        printf("╠══════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════╣\n");

        for (const auto& result : results) {
            printf("║ %-8s │ %8.3f  │ %8.3f  │ %8.3f  │   %5.2fx   │   %5.2fx   │ %5zu ║\n",
                   result.test_name.c_str(),
                   result.fallback_time,
                   result.numa_node0_time,
                   result.numa_all_nodes_time,
                   result.node0_speedup,
                   result.all_nodes_speedup,
                   result.tensor_size_mb);
        }

        if (!results.empty()) {
            // Calculate averages
            double avg_node0_speedup = 0.0;
            double avg_all_nodes_speedup = 0.0;
            for (const auto& result : results) {
                avg_node0_speedup += result.node0_speedup;
                avg_all_nodes_speedup += result.all_nodes_speedup;
            }
            avg_node0_speedup /= results.size();
            avg_all_nodes_speedup /= results.size();

            printf("╠══════════╧═══════════╧═══════════╧═══════════╧═══════════╧═══════════╧═══════╣\n");
            printf("║ Average Speedup:                            │   %5.2fx   │   %5.2fx   │       ║\n",
                   avg_node0_speedup, avg_all_nodes_speedup);
        }

        printf("╚════════════════════════════════════════════════════════════════════════════════╝\n");

        printf("\n🔍 KEY INSIGHTS:\n");
        printf("  • Fallback: Traditional CPU execution bound to NUMA node 0\n");
        printf("  • NUMA Node0: NUMA coordinator optimizations on single node\n");
        printf("  • NUMA All: NUMA coordinator with data-parallel across all nodes\n");
        printf("  • Higher speedup = better performance vs fallback\n");
        printf("  • All modes use same number of threads (%d)\n", total_physical_cores);
        
        printf("\nComparison completed successfully!\n");
    }
};

int main() {
    NumaExecutionModesComparison comparison;
    comparison.run_comparison();
    return 0;
}
