#define _GNU_SOURCE
#include <omp.h>
#include <unistd.h>
#include <sched.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>

void set_thread_affinity(int cpu) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    
    if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
        perror("sched_setaffinity failed");
    }
}

int get_current_cpu() {
    return sched_getcpu();
}

void test_explicit_affinity() {
    const int n = 1000;
    std::vector<double> a(n * n, 1.0);
    std::vector<double> b(n * n, 2.0);
    std::vector<double> c(n * n, 0.0);
    
    // Define which CPUs to use (physical cores 0-3)
    std::vector<int> target_cpus = {0, 1, 2, 3};
    
    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel num_threads(4)
    {
        int thread_id = omp_get_thread_num();
        
        // Set affinity for this thread
        set_thread_affinity(target_cpus[thread_id]);
        
        // Verify where we're running
        int current_cpu = get_current_cpu();
        
        #pragma omp critical
        {
            std::cout << "Thread " << thread_id 
                      << " set to CPU " << target_cpus[thread_id]
                      << " actually running on CPU " << current_cpu << std::endl;
        }
        
        // Wait for all threads to set their affinity
        #pragma omp barrier
        
        // Do computation
        #pragma omp for
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    c[i * n + j] += a[i * n + k] * b[k * n + j];
                }
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Explicit affinity test took: " << duration.count() << "ms" << std::endl;
    std::cout << "Result sample: " << c[500 * n + 500] << std::endl;
}

int main() {
    std::cout << "=== Testing explicit thread affinity within OpenMP ===" << std::endl;
    test_explicit_affinity();
    return 0;
}
