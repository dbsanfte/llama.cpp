#include <omp.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>

void test_matrix_multiply(const std::string& name, int n_threads) {
    // Set OpenMP thread count
    omp_set_num_threads(n_threads);
    
    const int size = 1000;
    std::vector<float> a(size * size, 1.0f);
    std::vector<float> b(size * size, 2.0f);
    std::vector<float> c(size * size, 0.0f);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float sum = 0.0f;
            for (int k = 0; k < size; k++) {
                sum += a[i * size + k] * b[k * size + j];
            }
            c[i * size + j] = sum;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << name << " (" << n_threads << " threads): " << duration.count() << "ms" << std::endl;
    std::cout << "  Result sample: " << c[0] << std::endl;
    
    // Show which CPUs are being used
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int cpu = sched_getcpu();
        #pragma omp critical
        {
            std::cout << "  Thread " << tid << " on CPU " << cpu << std::endl;
        }
    }
}

int main() {
    std::cout << "Simple OpenMP Matrix Multiplication Test" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Test different configurations
    test_matrix_multiply("Single thread", 1);
    test_matrix_multiply("4 threads (default)", 4);
    test_matrix_multiply("8 threads", 8);
    
    // Test with specific CPU binding
    std::cout << "\nTesting with CPU binding..." << std::endl;
    setenv("OMP_PLACES", "{0,1,2,3}", 1);
    setenv("OMP_PROC_BIND", "close", 1);
    test_matrix_multiply("4 threads (CPUs 0-3)", 4);
    
    setenv("OMP_PLACES", "{10,11,12,13}", 1);
    setenv("OMP_PROC_BIND", "close", 1);
    test_matrix_multiply("4 threads (CPUs 10-13)", 4);
    
    return 0;
}
