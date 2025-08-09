/**
 * Chunking vs Non-Chunking A/B Performance Test
 * 
 * This test directly compares two buffer allocation strategies:
 * A) Matrix size reduction (current approach)
 * B) Chunked processing with full matrices (previous approach)
 * 
 * The goal is to understand the performance implications of each approach
 * and determine which gives more accurate scaling measurements.
 */

#include <chrono>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <sstream>

#include "ggml.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-cpu/ggml-numa-coordinator.h"
#include "../common/common.h"
#include "../common/log.h"

using namespace std::chrono;

// Logging control helpers to reduce coordinator verbosity during benchmarks
static int original_log_verbosity = 0;
static ggml_log_callback original_ggml_callback = nullptr;
static void* original_ggml_user_data = nullptr;

// Custom GGML log callback that suppresses DEBUG messages
static void suppress_debug_callback(ggml_log_level level, const char* text, void* user_data) {
    (void)user_data; // Suppress unused parameter warning
    // Only allow ERROR and WARN messages through during suppression
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
        fflush(stderr);
    }
    // Suppress DEBUG, INFO, and other messages
}

static void suppress_coordinator_logging() {
    // Save original common log verbosity
    original_log_verbosity = common_log_verbosity_thold;
    common_log_set_verbosity_thold(GGML_LOG_LEVEL_NONE);
    
    // Save original GGML callback and set suppressing callback
    // Note: ggml_log_set doesn't return the old callback, so we use nullptr for simplicity
    original_ggml_callback = nullptr; // Can't access ggml_log_callback_default without ggml-impl.h
    original_ggml_user_data = nullptr;
    ggml_log_set(suppress_debug_callback, nullptr);
}

static void restore_coordinator_logging() {
    // Restore common log verbosity
    common_log_set_verbosity_thold(original_log_verbosity);
    
    // For now, just leave logging suppressed - user can restart program for full logging
    // TODO: Implement proper callback restoration when ggml-impl.h access is available
}

struct TestResult {
    std::string approach_name;
    int batch_size;
    int64_t matrix_dim;
    int64_t tensor_size;
    double execution_time_ms;
    double throughput_gops;
    int64_t memory_used_mb;
    bool used_chunking;
    int num_chunks;
    std::string notes;
    bool success = false;
};

class ChunkingABTester {
private:
    static auto get_time() { return high_resolution_clock::now(); }
    static double time_diff_ms(high_resolution_clock::time_point start, high_resolution_clock::time_point end) {
        return duration<double, std::milli>(end - start).count();
    }
    
    static void fill_tensor_random(struct ggml_tensor* tensor) {
        if (tensor->type == GGML_TYPE_F32) {
            float* data = (float*)ggml_get_data(tensor);
            size_t n = ggml_nelements(tensor);
            for (size_t i = 0; i < n; i++) {
                data[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
            }
        }
    }

public:
    // Approach A: Matrix Size Reduction (Current)
    TestResult test_matrix_reduction_approach(int batch_size, int64_t base_matrix_dim, int iterations) {
        TestResult result;
        result.approach_name = "Matrix Size Reduction";
        result.batch_size = batch_size;
        result.matrix_dim = base_matrix_dim;
        result.used_chunking = false;
        result.num_chunks = 1;
        
        try {
            // Calculate memory requirements
            int64_t bytes_per_matrix = base_matrix_dim * base_matrix_dim * sizeof(float);
            int64_t total_tensor_memory = batch_size * bytes_per_matrix * 3; // A + B + Result
            int64_t context_overhead = 256 * 1024 * 1024; // 256MB
            int64_t required_memory = total_tensor_memory + context_overhead;
            
            // Apply matrix size reduction if needed
            int64_t matrix_dim = base_matrix_dim;
            const int64_t max_memory = 8LL * 1024 * 1024 * 1024; // 8GB limit
            
            if (required_memory > max_memory) {
                while (matrix_dim > 64 && required_memory > max_memory) {
                    matrix_dim = matrix_dim * 3 / 4; // Reduce by 25%
                    bytes_per_matrix = matrix_dim * matrix_dim * sizeof(float);
                    total_tensor_memory = batch_size * bytes_per_matrix * 3;
                    required_memory = total_tensor_memory + context_overhead;
                }
                result.notes = "Matrix reduced from " + std::to_string(base_matrix_dim) + 
                              " to " + std::to_string(matrix_dim);
            }
            
            result.matrix_dim = matrix_dim;
            result.memory_used_mb = required_memory / (1024 * 1024);
            
            // Create context with calculated size
            struct ggml_init_params init_params = {
                static_cast<size_t>(required_memory),
                NULL,
                false,
            };
            
            struct ggml_context * ctx = ggml_init(init_params);
            if (!ctx) {
                result.notes += " [Context creation failed]";
                return result;
            }
            
            // Create threadpool for coordinator
            struct ggml_threadpool_params tpp;
            ggml_threadpool_params_init(&tpp, 22);
            tpp.force_multi_socket = true;
            
            struct ggml_numa_coordinator_manager *mgr = 
                ggml_numa_coordinator_manager_new_with_params(&tpp);
            if (!mgr) {
                ggml_free(ctx);
                result.notes += " [Coordinator creation failed]";
                return result;
            }
            
            // Create full batch of matrix operations
            std::vector<struct ggml_tensor *> matrices_a, matrices_b, results_tensors;
            struct ggml_cgraph * graph = ggml_new_graph(ctx);
            
            for (int b = 0; b < batch_size; b++) {
                struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, matrix_dim, matrix_dim);
                struct ggml_tensor * b_mat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, matrix_dim, matrix_dim);
                struct ggml_tensor * result_mat = ggml_mul_mat(ctx, a, b_mat);
                
                fill_tensor_random(a);
                fill_tensor_random(b_mat);
                
                ggml_build_forward_expand(graph, result_mat);
                
                matrices_a.push_back(a);
                matrices_b.push_back(b_mat);
                results_tensors.push_back(result_mat);
            }
            
            // Warmup
            ggml_numa_coordinator_manager_compute_graph(mgr, graph);
            ggml_numa_coordinator_manager_wait_for_completion(mgr);
            
            // Timed execution
            auto start = get_time();
            for (int iter = 0; iter < iterations; iter++) {
                ggml_numa_coordinator_manager_compute_graph(mgr, graph);
                ggml_numa_coordinator_manager_wait_for_completion(mgr);
            }
            auto end = get_time();
            
            result.execution_time_ms = time_diff_ms(start, end) / iterations;
            
            // Calculate performance metrics
            int64_t total_operations = batch_size * matrix_dim * matrix_dim * matrix_dim;
            double operations_per_second = total_operations / (result.execution_time_ms / 1000.0);
            result.throughput_gops = operations_per_second / 1e9;
            
            result.success = true;
            
            ggml_numa_coordinator_manager_free(mgr);
            ggml_free(ctx);
            
        } catch (const std::exception& e) {
            result.notes += " [Exception: " + std::string(e.what()) + "]";
        }
        
        return result;
    }
    
    // Approach B: Chunked Processing (Previous)
    TestResult test_chunked_processing_approach(int batch_size, int64_t base_matrix_dim, int iterations) {
        TestResult result;
        result.approach_name = "Chunked Processing";
        result.batch_size = batch_size;
        result.matrix_dim = base_matrix_dim; // Preserve full matrix size
        result.used_chunking = true;
        
        try {
            // Calculate chunk size based on memory limits
            int64_t bytes_per_matrix = base_matrix_dim * base_matrix_dim * sizeof(float);
            int64_t context_overhead = 256 * 1024 * 1024; // 256MB
            const int64_t max_memory = 8LL * 1024 * 1024 * 1024; // 8GB limit
            
            int64_t max_tensor_memory = max_memory - context_overhead;
            int effective_batch_size = static_cast<int>(max_tensor_memory / (bytes_per_matrix * 3));
            effective_batch_size = std::max(1, effective_batch_size);
            
            result.num_chunks = (batch_size + effective_batch_size - 1) / effective_batch_size;
            result.memory_used_mb = max_memory / (1024 * 1024);
            result.notes = "Chunked into " + std::to_string(result.num_chunks) + 
                          " chunks of " + std::to_string(effective_batch_size);
            
            // Create threadpool for coordinator
            struct ggml_threadpool_params tpp;
            ggml_threadpool_params_init(&tpp, 22);
            tpp.force_multi_socket = true;
            
            struct ggml_numa_coordinator_manager *mgr = 
                ggml_numa_coordinator_manager_new_with_params(&tpp);
            if (!mgr) {
                result.notes += " [Coordinator creation failed]";
                return result;
            }
            
            // Process in chunks
            double total_execution_time = 0.0;
            int remaining_batches = batch_size;
            
            struct ggml_init_params init_params = {
                static_cast<size_t>(max_memory),
                NULL,
                false,
            };
            
            for (int chunk = 0; chunk < result.num_chunks; chunk++) {
                int current_chunk_size = std::min(remaining_batches, effective_batch_size);
                
                struct ggml_context * ctx = ggml_init(init_params);
                if (!ctx) {
                    result.notes += " [Context creation failed at chunk " + std::to_string(chunk) + "]";
                    ggml_numa_coordinator_manager_free(mgr);
                    return result;
                }
                
                // Create batch of matrix operations for this chunk
                std::vector<struct ggml_tensor *> matrices_a, matrices_b, results_tensors;
                struct ggml_cgraph * graph = ggml_new_graph(ctx);
                
                for (int b = 0; b < current_chunk_size; b++) {
                    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, base_matrix_dim, base_matrix_dim);
                    struct ggml_tensor * b_mat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, base_matrix_dim, base_matrix_dim);
                    struct ggml_tensor * result_mat = ggml_mul_mat(ctx, a, b_mat);
                    
                    fill_tensor_random(a);
                    fill_tensor_random(b_mat);
                    
                    ggml_build_forward_expand(graph, result_mat);
                    
                    matrices_a.push_back(a);
                    matrices_b.push_back(b_mat);
                    results_tensors.push_back(result_mat);
                }
                
                // Warmup for first chunk only
                if (chunk == 0) {
                    ggml_numa_coordinator_manager_compute_graph(mgr, graph);
                    ggml_numa_coordinator_manager_wait_for_completion(mgr);
                }
                
                // Timed execution for this chunk
                auto chunk_start = get_time();
                for (int iter = 0; iter < iterations; iter++) {
                    ggml_numa_coordinator_manager_compute_graph(mgr, graph);
                    ggml_numa_coordinator_manager_wait_for_completion(mgr);
                }
                auto chunk_end = get_time();
                
                total_execution_time += time_diff_ms(chunk_start, chunk_end) / iterations;
                remaining_batches -= current_chunk_size;
                
                ggml_free(ctx);
            }
            
            result.execution_time_ms = total_execution_time;
            
            // Calculate performance metrics based on FULL batch and FULL matrix size
            int64_t total_operations = batch_size * base_matrix_dim * base_matrix_dim * base_matrix_dim;
            double operations_per_second = total_operations / (result.execution_time_ms / 1000.0);
            result.throughput_gops = operations_per_second / 1e9;
            
            result.success = true;
            
            ggml_numa_coordinator_manager_free(mgr);
            
        } catch (const std::exception& e) {
            result.notes += " [Exception: " + std::string(e.what()) + "]";
        }
        
        return result;
    }
    
    void run_ab_comparison() {
        // Use default parameters
        std::vector<int> batch_sizes = {32, 64, 128, 256, 512};
        int64_t base_matrix_dim = 1024;
        int iterations = 3;
        run_ab_comparison_with_config(batch_sizes, base_matrix_dim, iterations);
    }
    
    void run_ab_comparison_with_config(const std::vector<int>& batch_sizes, int64_t base_matrix_dim, int iterations) {
        printf("┌─────────────────────────────────────────────────────────────────────────┐\n");
        printf("│                    CHUNKING vs NON-CHUNKING A/B TEST                   │\n");
        printf("└─────────────────────────────────────────────────────────────────────────┘\n\n");
        
        printf("Test Configuration:\n");
        printf("  Batch Sizes: ");
        for (size_t i = 0; i < batch_sizes.size(); i++) {
            printf("%d", batch_sizes[i]);
            if (i < batch_sizes.size() - 1) printf(", ");
        }
        printf("\n");
        printf("  Base Matrix Size: %ldx%ld\n", base_matrix_dim, base_matrix_dim);
        printf("  Iterations per test: %d\n", iterations);
        printf("  Memory limit: 8GB\n\n");
        
        printf("Approach Comparison:\n");
        printf("  A) Matrix Size Reduction: Reduces matrix dimensions to fit full batch in memory\n");
        printf("  B) Chunked Processing: Keeps full matrix size, processes in chunks\n\n");
        
        std::vector<TestResult> results_a, results_b;
        
        for (int batch_size : batch_sizes) {
            printf("🧪 Testing Batch Size %d:\n", batch_size);
            
            // Test Approach A (Matrix Reduction)
            printf("  [A] Matrix Size Reduction... ");
            fflush(stdout);
            auto result_a = test_matrix_reduction_approach(batch_size, base_matrix_dim, iterations);
            results_a.push_back(result_a);
            
            if (result_a.success) {
                printf("✅ %.3f GOPS (matrix: %ldx%ld, mem: %ldMB)\n", 
                       result_a.throughput_gops, result_a.matrix_dim, result_a.matrix_dim, result_a.memory_used_mb);
            } else {
                printf("❌ Failed: %s\n", result_a.notes.c_str());
            }
            
            // Test Approach B (Chunked Processing)
            printf("  [B] Chunked Processing... ");
            fflush(stdout);
            auto result_b = test_chunked_processing_approach(batch_size, base_matrix_dim, iterations);
            results_b.push_back(result_b);
            
            if (result_b.success) {
                printf("✅ %.3f GOPS (%d chunks, mem: %ldMB)\n", 
                       result_b.throughput_gops, result_b.num_chunks, result_b.memory_used_mb);
            } else {
                printf("❌ Failed: %s\n", result_b.notes.c_str());
            }
            
            printf("\n");
        }
        
        // Print detailed comparison
        printf("┌─────────────────────────────────────────────────────────────────────────┐\n");
        printf("│                           DETAILED COMPARISON                          │\n");
        printf("└─────────────────────────────────────────────────────────────────────────┘\n\n");
        
        printf("%-8s │ %-20s │ %-20s │ %s\n", "Batch", "A: Matrix Reduction", "B: Chunked Processing", "Winner & Analysis");
        printf("─────────┼──────────────────────┼──────────────────────┼─────────────────────────\n");
        
        for (size_t i = 0; i < results_a.size(); i++) {
            auto& a = results_a[i];
            auto& b = results_b[i];
            
            if (a.success && b.success) {
                std::string winner = (a.throughput_gops > b.throughput_gops) ? "A" : "B";
                double ratio = std::max(a.throughput_gops, b.throughput_gops) / 
                              std::min(a.throughput_gops, b.throughput_gops);
                
                printf("%-8d │ %.3f GOPS (%ldx%ld) │ %.3f GOPS (%d chk) │ %s wins %.2fx\n",
                       a.batch_size, a.throughput_gops, a.matrix_dim, a.matrix_dim,
                       b.throughput_gops, b.num_chunks, winner.c_str(), ratio);
            } else {
                printf("%-8d │ %-20s │ %-20s │ %s\n",
                       a.batch_size,
                       a.success ? (std::to_string(a.throughput_gops) + " GOPS").c_str() : "FAILED",
                       b.success ? (std::to_string(b.throughput_gops) + " GOPS").c_str() : "FAILED",
                       "Data incomplete");
            }
        }
        
        // Analysis
        printf("\n📊 ANALYSIS:\n");
        
        // Calculate trends
        int a_wins = 0, b_wins = 0;
        double total_a_perf = 0, total_b_perf = 0;
        int valid_comparisons = 0;
        
        for (size_t i = 0; i < results_a.size(); i++) {
            if (results_a[i].success && results_b[i].success) {
                if (results_a[i].throughput_gops > results_b[i].throughput_gops) a_wins++;
                else b_wins++;
                
                total_a_perf += results_a[i].throughput_gops;
                total_b_perf += results_b[i].throughput_gops;
                valid_comparisons++;
            }
        }
        
        if (valid_comparisons > 0) {
            printf("  🏆 Winner Summary: A wins %d times, B wins %d times\n", a_wins, b_wins);
            printf("  📈 Average Performance: A = %.3f GOPS, B = %.3f GOPS\n", 
                   total_a_perf / valid_comparisons, total_b_perf / valid_comparisons);
            
            if (a_wins > b_wins) {
                printf("  🎯 Matrix Size Reduction approach performs better overall\n");
                printf("  💡 Implications: Current approach provides more accurate scaling measurements\n");
            } else {
                printf("  🎯 Chunked Processing approach performs better overall\n");
                printf("  💡 Implications: Previous approach may show inflated but more dramatic scaling\n");
            }
        }
        
        printf("\n  🔍 Key Insights:\n");
        printf("  • Matrix reduction preserves batch size but reduces computational intensity\n");
        printf("  • Chunked processing preserves computational intensity but breaks batch semantics\n");
        printf("  • The 'better' approach depends on whether you want to measure batch scaling or raw throughput\n");
    }
};

void print_usage(const char* program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  --quick              Quick test mode (batch sizes 16,32 with smaller matrices)\n");
    printf("  --batch-sizes LIST   Comma-separated list of batch sizes (e.g., 16,32,64)\n");
    printf("  --base-matrix SIZE   Base matrix dimension size (default: 1024)\n");
    printf("  --iterations NUM     Number of iterations per test (default: 3)\n");
    printf("  --help              Show this help message\n");
}

std::vector<int> parse_batch_sizes(const std::string& batch_str) {
    std::vector<int> batch_sizes;
    std::stringstream ss(batch_str);
    std::string item;
    
    while (std::getline(ss, item, ',')) {
        batch_sizes.push_back(std::stoi(item));
    }
    return batch_sizes;
}

int main(int argc, char* argv[]) {
    // Default parameters
    std::vector<int> batch_sizes = {32, 64, 128, 256, 512};
    int64_t base_matrix_dim = 1024;
    int iterations = 3;
    bool quick_mode = false;
    
    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--quick") == 0) {
            quick_mode = true;
        } else if (strcmp(argv[i], "--batch-sizes") == 0 && i + 1 < argc) {
            batch_sizes = parse_batch_sizes(argv[i + 1]);
            i++; // Skip next argument
        } else if (strcmp(argv[i], "--base-matrix") == 0 && i + 1 < argc) {
            base_matrix_dim = std::stoi(argv[i + 1]);
            i++; // Skip next argument
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = std::stoi(argv[i + 1]);
            i++; // Skip next argument
        } else {
            printf("Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Apply quick mode settings
    if (quick_mode) {
        batch_sizes = {16, 32};
        base_matrix_dim = 512; // Smaller base matrix for quick tests
        iterations = 2; // Fewer iterations
    }
    
    printf("Starting Chunking vs Non-Chunking A/B Test...\n\n");
    
    // Suppress coordinator verbose logging for cleaner test output
    suppress_coordinator_logging();
    
    ChunkingABTester tester;
    tester.run_ab_comparison_with_config(batch_sizes, base_matrix_dim, iterations);
    
    // Restore original logging
    restore_coordinator_logging();
    
    printf("\nA/B Test Complete!\n");
    return 0;
}
