#include <chrono>
#include <iostream>
#include <vector>
#include <cstring>
#include <immintrin.h> // For AVX/AVX2

// Simple SIMD ADD function using AVX2
void simd_add_f32(size_t n, float* dst, const float* src0, const float* src1) {
    const size_t simd_width = 8; // AVX2 processes 8 floats at once
    const size_t simd_end = (n / simd_width) * simd_width;
    
    // Process SIMD chunks
    for (size_t i = 0; i < simd_end; i += simd_width) {
        __m256 a = _mm256_load_ps(&src0[i]);
        __m256 b = _mm256_load_ps(&src1[i]);
        __m256 result = _mm256_add_ps(a, b);
        _mm256_store_ps(&dst[i], result);
    }
    
    // Handle remainder
    for (size_t i = simd_end; i < n; i++) {
        dst[i] = src0[i] + src1[i];
    }
}

int main() {
    const size_t total_elements = 67108864; // 256MB of floats
    const size_t bytes_per_element = sizeof(float);
    const size_t total_bytes = total_elements * bytes_per_element;
    
    std::cout << "🧪 Raw ADD Performance Test" << std::endl;
    std::cout << "Elements: " << total_elements << " (" << total_bytes / (1024*1024) << " MB)" << std::endl;
    
    // Allocate aligned memory for SIMD
    std::vector<float> src0(total_elements + 32, 1.0f);
    std::vector<float> src1(total_elements + 32, 2.0f);  
    std::vector<float> dst(total_elements + 32, 0.0f);
    
    // Align pointers to 32-byte boundary for AVX2
    float* aligned_src0 = (float*)(((uintptr_t)src0.data() + 31) & ~31);
    float* aligned_src1 = (float*)(((uintptr_t)src1.data() + 31) & ~31);
    float* aligned_dst = (float*)(((uintptr_t)dst.data() + 31) & ~31);
    
    // Initialize aligned arrays
    for (size_t i = 0; i < total_elements; i++) {
        aligned_src0[i] = 1.0f;
        aligned_src1[i] = 2.0f;
        aligned_dst[i] = 0.0f;
    }
    
    // Test 1: SIMD ADD
    auto start = std::chrono::high_resolution_clock::now();
    simd_add_f32(total_elements, aligned_dst, aligned_src0, aligned_src1);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double simd_time_ms = duration.count() / 1000.0;
    
    // Test 2: Simple scalar loop for comparison
    memset(aligned_dst, 0, total_elements * sizeof(float));
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < total_elements; i++) {
        aligned_dst[i] = aligned_src0[i] + aligned_src1[i];
    }
    end = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double scalar_time_ms = duration.count() / 1000.0;
    
    // Test 3: Split across 2 "nodes" (like NUMA)
    memset(aligned_dst, 0, total_elements * sizeof(float));
    start = std::chrono::high_resolution_clock::now();
    
    // Node 0: first half
    size_t half_elements = total_elements / 2;
    simd_add_f32(half_elements, aligned_dst, aligned_src0, aligned_src1);
    
    // Node 1: second half  
    simd_add_f32(half_elements, aligned_dst + half_elements, 
                 aligned_src0 + half_elements, aligned_src1 + half_elements);
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double split_time_ms = duration.count() / 1000.0;
    
    // Results
    std::cout << std::endl;
    std::cout << "📊 Performance Results:" << std::endl;
    std::cout << "SIMD single-thread:   " << simd_time_ms << " ms" << std::endl;
    std::cout << "Scalar single-thread: " << scalar_time_ms << " ms" << std::endl;
    std::cout << "SIMD split (2 parts): " << split_time_ms << " ms" << std::endl;
    std::cout << std::endl;
    std::cout << "SIMD speedup vs scalar: " << scalar_time_ms / simd_time_ms << "x" << std::endl;
    std::cout << std::endl;
    std::cout << "Expected NUMA performance should be ~" << simd_time_ms << "-" << split_time_ms << " ms" << std::endl;
    std::cout << "Actual NUMA performance: 265ms" << std::endl;
    std::cout << "Coordination overhead: ~" << 265.0 - simd_time_ms << " ms (" << (265.0 - simd_time_ms) / 265.0 * 100 << "%)" << std::endl;
    
    return 0;
}
