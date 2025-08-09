#include <iostream>
#include <cassert>
#include <cstdint>

// Include the NUMA coordinator header
#include "../ggml/src/ggml-cpu/ggml-numa-coordinator.h"

void test_cache_detection() {
    std::cout << "=== Cache Detection Test ===" << std::endl;
    
    struct ggml_numa_cache_info cache_info = {0};
    int result = ggml_numa_detect_cache_info(&cache_info);
    
    if (result == 0) {
        std::cout << "✅ Cache detection successful" << std::endl;
        std::cout << "  L1 Cache: " << cache_info.l1_cache_size << " bytes" << std::endl;
        std::cout << "  L2 Cache: " << cache_info.l2_cache_size << " bytes" << std::endl;
        std::cout << "  L3 Cache: " << cache_info.l3_cache_size << " bytes" << std::endl;
        std::cout << "  Cache Line Size: " << cache_info.cache_line_size << " bytes" << std::endl;
        std::cout << "  L3 Sharing Cores: " << cache_info.l3_sharing_cores << std::endl;
        std::cout << "  Detection Status: " << (cache_info.cache_detection_successful ? "SUCCESS" : "FAILED") << std::endl;
    } else {
        std::cout << "❌ Cache detection failed" << std::endl;
    }
}

void test_optimal_tile_size() {
    std::cout << "\n=== Optimal Tile Size Test ===" << std::endl;
    
    struct ggml_numa_cache_info cache_info = {0};
    ggml_numa_detect_cache_info(&cache_info);
    
    // Test different element sizes and cache levels
    struct {
        const char* name;
        size_t element_size;
        int cache_level;
    } test_cases[] = {
        {"Float32 - L1 Cache", sizeof(float), 1},
        {"Float32 - L2 Cache", sizeof(float), 2},
        {"Float32 - L3 Cache", sizeof(float), 3},
        {"Float16 - L1 Cache", sizeof(uint16_t), 1},
        {"Float16 - L2 Cache", sizeof(uint16_t), 2},
        {"Float16 - L3 Cache", sizeof(uint16_t), 3},
    };
    
    for (size_t i = 0; i < sizeof(test_cases) / sizeof(test_cases[0]); i++) {
        auto& test = test_cases[i];
        int64_t tile_size = ggml_numa_optimal_tile_size(&cache_info, test.element_size, test.cache_level);
        
        std::cout << "  " << test.name << ":" << std::endl;
        std::cout << "    Element size: " << test.element_size << " bytes" << std::endl;
        std::cout << "    Cache level: L" << test.cache_level << std::endl;
        std::cout << "    Optimal tile size: " << tile_size << "x" << tile_size << std::endl;
        std::cout << "    Tile memory usage: " << (tile_size * tile_size * test.element_size) << " bytes" << std::endl;
        
        // Verify tile size is reasonable
        assert(tile_size > 0);
        assert(tile_size <= 8192); // Reasonable upper bound
    }
}

void test_cache_aware_chunk_size() {
    std::cout << "\n=== Cache-Aware Chunk Size Test ===" << std::endl;
    
    struct ggml_numa_cache_info cache_info = {0};
    ggml_numa_detect_cache_info(&cache_info);
    
    // Test different workload scenarios
    struct {
        const char* name;
        int64_t matrix_dim;
        int batch_size;
        size_t element_size;
    } test_cases[] = {
        {"Small Batch Processing", 512, 32, sizeof(float)},
        {"Large Batch Processing", 1024, 128, sizeof(float)},
        {"Mixed Precision Small", 768, 64, sizeof(uint16_t)},
        {"High Throughput", 2048, 256, sizeof(float)},
    };
    
    for (size_t i = 0; i < sizeof(test_cases) / sizeof(test_cases[0]); i++) {
        auto& test = test_cases[i];
        int64_t chunk_size = ggml_numa_cache_aware_chunk_size(&cache_info, test.matrix_dim, test.batch_size, test.element_size);
        
        std::cout << "  " << test.name << ":" << std::endl;
        std::cout << "    Matrix: " << test.matrix_dim << "x" << test.matrix_dim << ", Batch: " << test.batch_size << std::endl;
        std::cout << "    Optimal chunk size: " << chunk_size << std::endl;
        
        // Calculate chunk memory usage
        int64_t chunk_memory = chunk_size * test.matrix_dim * test.matrix_dim * test.element_size;
        std::cout << "    Chunk memory usage: " << chunk_memory << " bytes (" << (chunk_memory / 1024) << " KB)" << std::endl;
        
        // Verify chunk size is reasonable
        assert(chunk_size > 0);
        assert(chunk_size <= test.batch_size);
        
        // Check if chunk fits in L3 cache (with some headroom)
        if (cache_info.l3_cache_size > 0) {
            bool fits_in_l3 = chunk_memory <= (cache_info.l3_cache_size * 0.8); // 80% utilization
            std::cout << "    Fits in L3 cache: " << (fits_in_l3 ? "YES" : "NO") << std::endl;
        }
    }
}

void test_cache_optimization_integration() {
    std::cout << "\n=== Cache Optimization Integration Test ===" << std::endl;
    
    struct ggml_numa_cache_info cache_info = {0};
    ggml_numa_detect_cache_info(&cache_info);
    
    // Test how cache information affects strategy selection
    struct ggml_numa_workload_info workload;
    workload.matrix_dim = 1024;
    workload.batch_size = 64;
    workload.available_memory_gb = 32;
    workload.prioritize_scaling_accuracy = false;
    workload.user_override = GGML_NUMA_STRATEGY_AUTO;
    workload.cache_info = cache_info;
    
    enum ggml_numa_memory_strategy strategy = ggml_numa_choose_strategy(&workload);
    
    std::cout << "  Workload: 1024x1024 matrix, batch=64, memory=32GB" << std::endl;
    std::cout << "  Cache-aware strategy choice: ";
    
    switch (strategy) {
        case GGML_NUMA_STRATEGY_AUTO:
            std::cout << "AUTO (adaptive selection)";
            break;
        case GGML_NUMA_STRATEGY_MATRIX_REDUCTION:
            std::cout << "MATRIX_REDUCTION (memory efficient)";
            break;
        case GGML_NUMA_STRATEGY_CHUNKED_PROCESSING:
            std::cout << "CHUNKED_PROCESSING (high throughput)";
            break;
        case GGML_NUMA_STRATEGY_HYBRID:
            std::cout << "HYBRID (dynamic switching)";
            break;
        default:
            std::cout << "UNKNOWN (" << static_cast<int>(strategy) << ")";
            break;
    }
    std::cout << std::endl;
    
    // Calculate matrix memory footprint
    size_t matrix_memory = workload.matrix_dim * workload.matrix_dim * sizeof(float);
    std::cout << "  Matrix memory: " << matrix_memory << " bytes (" << (matrix_memory / 1024) << " KB)" << std::endl;
    
    // Check cache utilization
    if (cache_info.l3_cache_size > 0) {
        bool fits_in_l3 = matrix_memory <= (size_t)cache_info.l3_cache_size;
        std::cout << "  Fits in L3 cache: " << (fits_in_l3 ? "YES" : "NO") << std::endl;
        std::cout << "  L3 utilization: " << (100.0 * matrix_memory / cache_info.l3_cache_size) << "%" << std::endl;
    }
    
    if (cache_info.l2_cache_size > 0) {
        bool fits_in_l2 = matrix_memory <= (size_t)cache_info.l2_cache_size;
        std::cout << "  Fits in L2 cache: " << (fits_in_l2 ? "YES" : "NO") << std::endl;
    }
}

int main() {
    std::cout << "🧠 NUMA Cache Optimization Test" << std::endl;
    std::cout << "================================" << std::endl;
    
    try {
        test_cache_detection();
        test_optimal_tile_size();
        test_cache_aware_chunk_size();
        test_cache_optimization_integration();
        
        std::cout << "\n🎯 Cache Optimization Test Complete!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
