#include "ggml.h"
#include "ggml-cpu.h"
#include <iostream>
#include <chrono>
#include <thread>

int main() {
    std::cout << "Simple Coordinator Test" << std::endl;
    
    // Test 1: Create and destroy coordinator threadpool
    std::cout << "=== Test 1: Create coordinator threadpool ===" << std::endl;
    struct ggml_threadpool_params tpp;
    ggml_threadpool_params_init(&tpp, 4);
    tpp.numa_aware = true;
    tpp.force_multi_socket = true; // Force coordinator creation
    
    ggml_threadpool_t pool1 = ggml_threadpool_new(&tpp);
    if (pool1) {
        std::cout << "✓ Pool 1 created" << std::endl;
        ggml_threadpool_free(pool1);
        std::cout << "✓ Pool 1 destroyed" << std::endl;
    }
    
    std::cout << "Waiting 100ms..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Test 2: Create another coordinator threadpool
    std::cout << "=== Test 2: Create second coordinator threadpool ===" << std::endl;
    ggml_threadpool_t pool2 = ggml_threadpool_new(&tpp);
    if (pool2) {
        std::cout << "✓ Pool 2 created" << std::endl;
        ggml_threadpool_free(pool2);
        std::cout << "✓ Pool 2 destroyed" << std::endl;
    }
    
    std::cout << "Test completed successfully" << std::endl;
    return 0;
}
