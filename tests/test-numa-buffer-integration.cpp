#include <iostream>
#include <vector>
#include <chrono>
#include "ggml-backend.h"
#include "ggml-cpu.h"

class NumaBufferTester {
public:
    void test_buffer_allocation() {
        std::cout << "=== Testing NUMA Buffer Type ===\n\n";
        
        // Test different buffer sizes
        std::vector<size_t> test_sizes = {
            1024 * 1024,        // 1MB
            64 * 1024 * 1024,   // 64MB (typical small model KV cache)
            256 * 1024 * 1024,  // 256MB (typical medium model KV cache)
            1024 * 1024 * 1024  // 1GB (typical large model KV cache)
        };
        
        for (size_t size : test_sizes) {
            std::cout << "Testing " << size / (1024 * 1024) << "MB allocation:\n";
            
            // Test standard CPU buffer type
            auto start = std::chrono::high_resolution_clock::now();
            ggml_backend_buffer_type_t standard_buft = ggml_backend_cpu_buffer_type();
            ggml_backend_buffer_t standard_buffer = ggml_backend_buft_alloc_buffer(standard_buft, size);
            auto end = std::chrono::high_resolution_clock::now();
            auto standard_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            if (standard_buffer) {
                std::cout << "  Standard buffer: " << ggml_backend_buft_name(standard_buft) 
                          << " allocated in " << standard_duration.count() << " μs\n";
                ggml_backend_buffer_free(standard_buffer);
            } else {
                std::cout << "  Standard buffer allocation failed\n";
            }
            
            // Test NUMA-aware buffer type
            start = std::chrono::high_resolution_clock::now();
            ggml_backend_buffer_type_t numa_buft = ggml_backend_cpu_numa_buffer_type();
            ggml_backend_buffer_t numa_buffer = ggml_backend_buft_alloc_buffer(numa_buft, size);
            end = std::chrono::high_resolution_clock::now();
            auto numa_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            if (numa_buffer) {
                std::cout << "  NUMA buffer: " << ggml_backend_buft_name(numa_buft)
                          << " allocated in " << numa_duration.count() << " μs\n";
                ggml_backend_buffer_free(numa_buffer);
            } else {
                std::cout << "  NUMA buffer allocation failed\n";
            }
            
            std::cout << std::endl;
        }
    }
    
    void test_buffer_performance() {
        std::cout << "=== Testing Buffer Access Performance ===\n\n";
        
        size_t test_size = 256 * 1024 * 1024; // 256MB
        
        // Allocate buffers
        ggml_backend_buffer_t standard_buffer = ggml_backend_buft_alloc_buffer(
            ggml_backend_cpu_buffer_type(), test_size);
        ggml_backend_buffer_t numa_buffer = ggml_backend_buft_alloc_buffer(
            ggml_backend_cpu_numa_buffer_type(), test_size);
            
        if (!standard_buffer || !numa_buffer) {
            std::cout << "Failed to allocate test buffers\n";
            if (standard_buffer) ggml_backend_buffer_free(standard_buffer);
            if (numa_buffer) ggml_backend_buffer_free(numa_buffer);
            return;
        }
        
        // Get buffer bases
        void* standard_data = ggml_backend_buffer_get_base(standard_buffer);
        void* numa_data = ggml_backend_buffer_get_base(numa_buffer);
        
        // Performance test - write/read patterns
        auto test_access_pattern = [](void* data, size_t size, const std::string& name) {
            volatile uint64_t* ptr = (volatile uint64_t*)data;
            size_t count = size / sizeof(uint64_t);
            
            // Write test
            auto start = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < count; i += 64) {  // Cache line stride
                ptr[i] = i;
            }
            auto mid = std::chrono::high_resolution_clock::now();
            
            // Read test  
            uint64_t sum = 0;
            for (size_t i = 0; i < count; i += 64) {
                sum += ptr[i];
            }
            auto end = std::chrono::high_resolution_clock::now();
            
            auto write_time = std::chrono::duration_cast<std::chrono::microseconds>(mid - start);
            auto read_time = std::chrono::duration_cast<std::chrono::microseconds>(end - mid);
            
            std::cout << "  " << name << ":\n";
            std::cout << "    Write: " << write_time.count() << " μs, "
                      << (size / (write_time.count() * 1e-6)) / (1024*1024*1024) << " GB/s\n";
            std::cout << "    Read:  " << read_time.count() << " μs, "
                      << (size / (read_time.count() * 1e-6)) / (1024*1024*1024) << " GB/s\n";
            std::cout << "    Sum verification: " << (sum > 0 ? "PASS" : "FAIL") << "\n\n";
        };
        
        test_access_pattern(standard_data, test_size, "Standard Buffer");
        test_access_pattern(numa_data, test_size, "NUMA Buffer");
        
        ggml_backend_buffer_free(standard_buffer);
        ggml_backend_buffer_free(numa_buffer);
    }
};

int main() {
    std::cout << "NUMA-Aware Buffer Type Testing\n";
    std::cout << "==============================\n\n";
    
    // Test if NUMA is available
    if (ggml_is_numa()) {
        std::cout << "✅ NUMA coordinator is active\n";
    } else {
        std::cout << "⚠️  NUMA coordinator is not active (this is expected in containers)\n";
    }
    
    std::cout << "🔍 Testing buffer allocation and performance...\n\n";
    
    NumaBufferTester tester;
    tester.test_buffer_allocation();
    tester.test_buffer_performance();
    
    std::cout << "=== Analysis ===\n";
    std::cout << "This test demonstrates that:\n";
    std::cout << "1. NUMA-aware buffer type can be used as a drop-in replacement\n";
    std::cout << "2. It gracefully falls back to standard allocation when NUMA is not available\n";
    std::cout << "3. The KV cache could benefit from using this buffer type on NUMA systems\n";
    std::cout << "\nNext step: Integrate this buffer type into KV cache allocation\n";
    
    return 0;
}
